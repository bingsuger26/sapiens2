# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
from argparse import ArgumentParser

# Block mmpretrain: mmdet's reid modules try `import mmpretrain` inside
# try/except ImportError, but mmpretrain's BLIP language_model.py raises
# TypeError (transformers API drift) — escapes the except and kills the process.
# We don't use reid or mmpretrain, so force a clean ImportError.
sys.modules["mmpretrain"] = None

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
from sapiens.pose.evaluators import nms
from sapiens.pose.models import init_model
from tqdm import tqdm

from pose_render_utils import visualize_keypoints
from plot_metrics import plot_action_metrics

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


# Arm-related keypoint indices (shoulder, elbow, wrist, fingers, acromion, etc.)
ARM_INDICES = list([5, 6, 7, 8]) + list(range(21, 69))  # 48 points
ARM_INDICES_SET = set(ARM_INDICES)
# Mapping from original 308-index to new 0-47 index
ARM_IDX_MAP = {orig: new for new, orig in enumerate(ARM_INDICES)}


def filter_skeleton_for_arm(skeleton_links, skeleton_link_colors):
    """Keep only skeleton links where both endpoints are arm keypoints,
    and remap indices to the filtered keypoint array."""
    filtered_links = []
    filtered_colors = []
    for link, color in zip(skeleton_links, skeleton_link_colors):
        if link[0] in ARM_INDICES_SET and link[1] in ARM_INDICES_SET:
            filtered_links.append([ARM_IDX_MAP[link[0]], ARM_IDX_MAP[link[1]]])
            filtered_colors.append(color)
    return filtered_links, filtered_colors


# ---------------------------------------------------------------------------
# Pointing action detection via wrist velocity relative to shoulder anchor
# ---------------------------------------------------------------------------

# Indices in the *filtered* 48-point arm keypoint array
_LEFT_SHOULDER_IDX = ARM_INDICES.index(5)    # 0
_RIGHT_SHOULDER_IDX = ARM_INDICES.index(6)   # 1
_RIGHT_WRIST_IDX = ARM_INDICES.index(41)     # 24
_LEFT_WRIST_IDX = ARM_INDICES.index(62)      # 45


class PointingActionDetector:
    """Rule-based state machine that classifies each frame into one of:

        "idle"        – no pointing action detected
        "raising"     – hand is being raised (wrist moving fast upward/outward)
        "pointing"    – hand is held steady in a pointing pose
        "lowering"    – hand is being lowered back
        "bbox_change" – camera / person is moving too much, skip detection

    The detector tracks the *primary wrist* (whichever moves first) and uses
    velocity in an anchor-relative coordinate system.  It also reports which
    side (left / right) is driving the action.

    Parameters
    ----------
    ratio_thr : float
        Minimum ratio of faster-wrist-speed to slower-wrist-speed.
    vel_min : float
        Minimum wrist speed (anchor-normalised units / frame).
    confirm_frames : int
        Consecutive frames needed to confirm a state transition.
    history_len : int
        Number of past frames used for velocity smoothing.
    pos_diff_thr : float
        Minimum wrist displacement (in shoulder-widths) between transitions.
    pos_lookback : int
        When recording the transition position, look back this many frames
        instead of using the current frame.  This avoids the "late snapshot"
        problem where the position is already at the new pose.
    bbox_vel_thr : float
        Maximum tolerated bbox centre speed (pixels / frame, averaged over
        recent frames).  Above this the frame is labelled "bbox_change".
    bbox_history_len : int
        Number of recent bbox centres kept for bbox velocity computation.
    """

    # Action labels
    IDLE = "idle"
    RAISING = "raising"
    POINTING = "pointing"
    LOWERING = "lowering"
    BBOX_CHANGE = "bbox_change"
    NO_BBOX = "no_bbox"

    def __init__(
        self,
        ratio_thr: float = 2.8,
        vel_min: float = 0.025,
        confirm_frames: int = 3,
        history_len: int = 3,
        pos_diff_thr: float = 0.05,
        pos_lookback: int = 3,
        bbox_vel_thr: float = 8.0,
        bbox_history_len: int = 3,
    ):
        self.ratio_thr = ratio_thr
        self.vel_min = vel_min
        self.confirm_frames = confirm_frames
        self.history_len = history_len
        self.pos_diff_thr = pos_diff_thr
        self.pos_lookback = pos_lookback
        self.bbox_vel_thr = bbox_vel_thr
        self.bbox_history_len = bbox_history_len

        # State
        self.state = self.IDLE
        self._confirm_count = 0
        self._prev_rel_positions = []    # list of (left_wrist_rel, right_wrist_rel)
        self._last_transition_pos = None  # (lw_rel, rw_rel) snapshot at last transition
        self._active_side = None          # "left" or "right" or None

        # Bbox stability tracking – stores recent bbox centres (np.ndarray (2,))
        self._bbox_centres = []  # list of np.ndarray (2,) or None
        self._bbox_confirm_count = 0  # consecutive frames with bbox_vel >= thr
        self._bbox_stable_count = 0   # consecutive frames with bbox_vel < thr (while in BBOX_CHANGE)

    # ------------------------------------------------------------------
    def _compute_anchor_and_rel(self, kpts, scores, kpt_thr=0.3):
        """Compute anchor (mid-shoulder) and relative wrist positions."""
        ls = kpts[_LEFT_SHOULDER_IDX]
        rs = kpts[_RIGHT_SHOULDER_IDX]
        ls_ok = scores[_LEFT_SHOULDER_IDX] >= kpt_thr
        rs_ok = scores[_RIGHT_SHOULDER_IDX] >= kpt_thr

        if not (ls_ok and rs_ok):
            return None, None, None, 0.0

        anchor = (ls + rs) / 2.0
        anchor_scale = max(np.linalg.norm(ls - rs), 1e-6)

        lw = kpts[_LEFT_WRIST_IDX]
        rw = kpts[_RIGHT_WRIST_IDX]
        lw_ok = scores[_LEFT_WRIST_IDX] >= kpt_thr
        rw_ok = scores[_RIGHT_WRIST_IDX] >= kpt_thr

        lw_rel = (lw - anchor) / anchor_scale if lw_ok else None
        rw_rel = (rw - anchor) / anchor_scale if rw_ok else None

        return anchor, lw_rel, rw_rel, anchor_scale

    # ------------------------------------------------------------------
    def _velocity(self, current, history_key):
        """Compute smoothed velocity from recent history."""
        if current is None or len(self._prev_rel_positions) == 0:
            return 0.0

        vels = []
        for i in range(1, min(self.history_len + 1, len(self._prev_rel_positions) + 1)):
            prev = self._prev_rel_positions[-(i)]
            prev_pos = prev[0] if history_key == "left" else prev[1]
            if prev_pos is not None:
                vels.append(np.linalg.norm(current - prev_pos) / i)

        return float(np.mean(vels)) if vels else 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _vel_ratio(vel_left, vel_right):
        """Compute the ratio of the faster wrist to the slower wrist.
        Returns (ratio, faster_side).  ratio >= 1.0 always."""
        min_vel = 1e-9
        if vel_left >= vel_right:
            return vel_left / max(vel_right, min_vel), "left"
        else:
            return vel_right / max(vel_left, min_vel), "right"

    # ------------------------------------------------------------------
    def _bbox_velocity(self, bbox):
        """Compute average bbox centre speed over recent frames.

        Parameters
        ----------
        bbox : np.ndarray (4,)  – [x1, y1, x2, y2] or None

        Returns
        -------
        bbox_vel : float  – average pixel displacement per frame.
        """
        if bbox is not None:
            centre = np.array([(bbox[0] + bbox[2]) / 2.0,
                               (bbox[1] + bbox[3]) / 2.0])
        else:
            centre = None

        self._bbox_centres.append(centre)
        if len(self._bbox_centres) > self.bbox_history_len + 2:
            self._bbox_centres.pop(0)

        if centre is None or len(self._bbox_centres) < 2:
            return 0.0

        vels = []
        for i in range(1, min(self.bbox_history_len + 1, len(self._bbox_centres))):
            prev = self._bbox_centres[-(i + 1)]
            if prev is not None:
                vels.append(np.linalg.norm(centre - prev) / i)

        return float(np.mean(vels)) if vels else 0.0

    # ------------------------------------------------------------------
    def _get_lookback_pos(self):
        """Return wrist relative positions from ``pos_lookback`` frames ago.

        If not enough history, return the earliest available frame.
        Returns (lw_rel, rw_rel) – either may be None.
        """
        lb = min(self.pos_lookback, len(self._prev_rel_positions))
        if lb == 0:
            return None, None
        return self._prev_rel_positions[-lb]

    # ------------------------------------------------------------------
    def update(self, kpts, scores, bbox=None, kpt_thr=0.3):
        """Process one frame and return (action_label, debug_dict).

        Parameters
        ----------
        kpts : np.ndarray (48, 2)
        scores : np.ndarray (48,)
        bbox : np.ndarray (4,) or None  – person bbox [x1, y1, x2, y2].
        kpt_thr : float

        Returns
        -------
        state : str
        debug : dict
        """
        # --- No bbox → skip detection entirely ---
        if bbox is None:
            self._prev_rel_positions.append((None, None))
            self._bbox_centres.append(None)
            if len(self._bbox_centres) > self.bbox_history_len + 2:
                self._bbox_centres.pop(0)
            # Reset state machine so next valid frame starts from IDLE
            self.state = self.IDLE
            self._confirm_count = 0
            self._active_side = None
            self._last_transition_pos = None
            return self.NO_BBOX, {
                "reason": "no_bbox",
                "bbox_vel": 0.0,
                "active_side": self._active_side,
            }

        # --- Bbox stability check (with consecutive-frame confirmation) ---
        bbox_vel = self._bbox_velocity(bbox)
        bbox_over_thr = bbox_vel >= self.bbox_vel_thr

        anchor, lw_rel, rw_rel, anchor_scale = self._compute_anchor_and_rel(
            kpts, scores, kpt_thr
        )

        if anchor is None:
            self._prev_rel_positions.append((None, None))
            return self.state, {
                "reason": "shoulders_not_visible",
                "bbox_vel": round(bbox_vel, 3),
                "active_side": self._active_side,
            }

        vel_left = self._velocity(lw_rel, "left")
        vel_right = self._velocity(rw_rel, "right")
        ratio, faster_side = self._vel_ratio(vel_left, vel_right)

        debug = {
            "vel_left": round(vel_left, 5),
            "vel_right": round(vel_right, 5),
            "vel_ratio": round(ratio, 3),
            "faster_side": faster_side,
            "anchor_scale": round(anchor_scale, 2),
            "state_before": self.state,
            "bbox_vel": round(bbox_vel, 3),
            "active_side": self._active_side,
        }

        # --- Bbox change state machine (needs confirm_frames consecutive frames) ---
        if self.state == self.BBOX_CHANGE:
            # Already in BBOX_CHANGE: wait for confirm_frames consecutive stable frames
            if not bbox_over_thr:
                self._bbox_stable_count += 1
                self._bbox_confirm_count = 0
                if self._bbox_stable_count >= self.confirm_frames:
                    # Transition back to IDLE
                    self.state = self.IDLE
                    self._bbox_stable_count = 0
                    self._confirm_count = 0
                    self._active_side = None
                    self._last_transition_pos = None
            else:
                self._bbox_stable_count = 0
                self._bbox_confirm_count = 0

            # While in BBOX_CHANGE, skip normal action detection
            debug["state_after"] = self.state if self.state != self.IDLE else self.IDLE
            debug["bbox_unstable"] = True

            self._prev_rel_positions.append((lw_rel, rw_rel))
            if len(self._prev_rel_positions) > self.history_len + 2:
                self._prev_rel_positions.pop(0)

            # If we just transitioned back to IDLE, report IDLE; otherwise BBOX_CHANGE
            if self.state == self.IDLE:
                debug["state_after"] = self.IDLE
                return self.IDLE, debug
            else:
                debug["state_after"] = self.BBOX_CHANGE
                return self.BBOX_CHANGE, debug
        else:
            # Not in BBOX_CHANGE yet: count consecutive over-threshold frames
            if bbox_over_thr:
                self._bbox_confirm_count += 1
                self._bbox_stable_count = 0
                if self._bbox_confirm_count >= self.confirm_frames:
                    # Enter BBOX_CHANGE state
                    self.state = self.IDLE  # internal state reset
                    self._confirm_count = 0
                    self._active_side = None
                    self._last_transition_pos = None
                    self._bbox_confirm_count = 0

                    debug["state_after"] = self.BBOX_CHANGE
                    debug["bbox_unstable"] = True

                    self._prev_rel_positions.append((lw_rel, rw_rel))
                    if len(self._prev_rel_positions) > self.history_len + 2:
                        self._prev_rel_positions.pop(0)

                    # Mark state as BBOX_CHANGE for next frame's check
                    self.state = self.BBOX_CHANGE
                    return self.BBOX_CHANGE, debug
            else:
                self._bbox_confirm_count = 0
                self._bbox_stable_count = 0

        # Determine asymmetric / symmetric conditions
        is_asymmetric = (
            ratio >= self.ratio_thr
            and max(vel_left, vel_right) >= self.vel_min
        )
        is_symmetric = (
            ratio < self.ratio_thr
            and max(vel_left, vel_right) < self.vel_min
        )

        # Current wrist positions snapshot (for position-diff check)
        cur_pos = (lw_rel, rw_rel)

        def _pos_changed_enough():
            """Check if wrists moved enough since the last recorded transition."""
            if self._last_transition_pos is None:
                return True, float("inf")
            dists = []
            for cur, prev in zip(cur_pos, self._last_transition_pos):
                if cur is not None and prev is not None:
                    dists.append(float(np.linalg.norm(cur - prev)))
            if not dists:
                return True, 0.0
            max_dist = max(dists)
            return max_dist >= self.pos_diff_thr, max_dist

        def _try_transition(new_state):
            """Attempt a state transition.  Use lookback position for snapshot."""
            ok, dist = _pos_changed_enough()
            debug["pos_diff"] = round(dist, 4) if dist != float("inf") else None
            if ok:
                self.state = new_state
                # Use position from ``pos_lookback`` frames ago instead of current
                lb_lw, lb_rw = self._get_lookback_pos()
                self._last_transition_pos = (
                    lb_lw.copy() if lb_lw is not None else None,
                    lb_rw.copy() if lb_rw is not None else None,
                )
                debug["pos_check"] = "pass"
            else:
                self.state = self.IDLE
                self._last_transition_pos = None
                self._active_side = None
                debug["pos_check"] = f"fail(dist={dist:.4f}<{self.pos_diff_thr})"
            self._confirm_count = 0

        # --- State transitions ---
        if self.state == self.IDLE:
            if is_asymmetric:
                self._confirm_count += 1
                if self._confirm_count >= self.confirm_frames:
                    self._active_side = faster_side
                    _try_transition(self.RAISING)
            else:
                self._confirm_count = 0

        elif self.state == self.RAISING:
            if is_symmetric:
                self._confirm_count += 1
                if self._confirm_count >= self.confirm_frames:
                    _try_transition(self.POINTING)
            else:
                self._confirm_count = 0

        elif self.state == self.POINTING:
            if is_asymmetric:
                self._confirm_count += 1
                if self._confirm_count >= self.confirm_frames:
                    _try_transition(self.LOWERING)
            else:
                self._confirm_count = 0

        elif self.state == self.LOWERING:
            if is_symmetric:
                self._confirm_count += 1
                if self._confirm_count >= self.confirm_frames:
                    self._active_side = None
                    _try_transition(self.IDLE)
            else:
                self._confirm_count = 0

        debug["state_after"] = self.state
        debug["active_side"] = self._active_side

        # Update history (keep bounded)
        self._prev_rel_positions.append((lw_rel, rw_rel))
        if len(self._prev_rel_positions) > self.history_len + 2:
            self._prev_rel_positions.pop(0)

        return self.state, debug


def mmdet_pipeline(cfg):
    from mmdet.datasets import transforms

    if "test_dataloader" not in cfg:
        return cfg
    pipeline = cfg.test_dataloader.dataset.pipeline
    for trans in pipeline:
        if trans["type"] in dir(transforms):
            trans["type"] = "mmdet." + trans["type"]
    return cfg


def _run_pose_on_bboxes(bboxes, image, model):
    """Run pose estimation on pre-computed bboxes (shared by both pipelines).

    Args:
        bboxes: np.ndarray, shape (N, 4), format [x1, y1, x2, y2].
        image: np.ndarray, BGR image (H, W, 3).
        model: pose estimation model.

    Returns:
        keypoints, keypoint_scores, bboxes
    """
    image_w, image_h = image.shape[1], image.shape[0]

    if bboxes is None or len(bboxes) == 0:
        bboxes = np.array([[0, 0, image_w - 1, image_h - 1]], dtype=np.float32)

    inputs_list = []
    data_samples_list = []
    for bbox in bboxes:
        data_info = dict(img=image)
        data_info["bbox"] = bbox[None]  # shape (1, 4)
        data_info["bbox_score"] = np.ones(1, dtype=np.float32)  # shape (1,)
        data = model.pipeline(data_info)
        data = model.data_preprocessor(data)
        inputs_list.append(data["inputs"])
        data_samples_list.append(data["data_samples"])

    inputs = torch.cat(inputs_list, dim=0)  # B x 3 x H x W
    with torch.no_grad():
        pred = model(inputs)  # B x 3 x H x W
        if model.cfg.val_cfg is not None and model.cfg.val_cfg.get("flip_test", False):
            pred_flipped = model(inputs.flip(-1))  # B x 3 x H x W
            pred_flipped = pred_flipped.flip(-1)  ## B x K x heatmap_H x heatmap_W
            flip_indices = model.pose_metainfo["flip_indices"]
            assert len(flip_indices) == pred_flipped.shape[1]  ## K
            pred_flipped = pred_flipped[:, flip_indices]
            pred = (pred + pred_flipped) / 2.0

    # ------------------------------------------
    pred = pred.cpu().numpy()  ## B x K x heatmap_H x heatmap_W

    arm_idx = np.array(ARM_INDICES)

    keypoints = []
    keypoint_scores = []
    for i, data_samples in enumerate(data_samples_list):
        ## kps in crop image
        ## keypoints_i is 1 x K x 2
        # keypoint_scores_i is 1 x K
        keypoints_i, keypoint_scores_i = model.codec.decode(pred[i])
        input_size = data_samples["meta"]["input_size"]  ## 1 x 2, 768 x 1024
        bbox_center = data_samples["meta"]["bbox_center"]  ## 1 x 2
        bbox_scale = data_samples["meta"]["bbox_scale"]  ## 1 x 2

        keypoints_i = (
            keypoints_i / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
        )

        # Filter to arm keypoints only
        keypoints_i = keypoints_i[:, arm_idx, :]      # 1 x 48 x 2
        keypoint_scores_i = keypoint_scores_i[:, arm_idx]  # 1 x 48

        keypoints.append(keypoints_i[0])  ## remove fake batch dim
        keypoint_scores.append(keypoint_scores_i[0])  ## remove fake batch dim

    return keypoints, keypoint_scores, bboxes


def process_one_image(args, image, detector, model):
    """Detect persons with RTMDet, then run pose estimation."""
    det_result = inference_detector(detector, image)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
    )
    bboxes = bboxes[
        np.logical_and(
            pred_instance.labels == 0,  ## 0 is the person class
            pred_instance.scores > args.bbox_thr,
        )
    ]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]  ## B x 4; x1, y1, x2, y2

    return _run_pose_on_bboxes(bboxes, image, model)


def process_one_image_with_bbox(image, bboxes, model):
    """Run pose estimation with externally provided bboxes (no detector needed).

    Args:
        image: np.ndarray, BGR image (H, W, 3).
        bboxes: np.ndarray, shape (N, 4), format [x1, y1, x2, y2].
        model: pose estimation model.

    Returns:
        keypoints, keypoint_scores, bboxes
    """
    return _run_pose_on_bboxes(bboxes, image, model)


def load_track_json(track_json_path):
    """Load a track.json file and return a dict mapping frame_index -> list of
    person bboxes (each bbox is [x1, y1, x2, y2] as np.float32).

    Args:
        track_json_path: path to the track.json file.

    Returns:
        dict[int, np.ndarray]: frame_index -> bboxes array of shape (N, 4).
    """
    with open(track_json_path, "r") as f:
        data = json.load(f)

    frame_bboxes = {}
    for frame in data["track_data"]:
        fidx = frame["frame_index"]
        persons = frame.get("person_detections", [])
        if persons:
            bboxes = np.array(
                [p["bbox_xyxy"] for p in persons], dtype=np.float32
            )
        else:
            bboxes = np.empty((0, 4), dtype=np.float32)
        frame_bboxes[fidx] = bboxes

    return frame_bboxes


# -------------------------------------------------------------------------------
def _frame_index_from_name(image_name):
    """Extract integer frame index from an image filename like '00001.png'."""
    stem = os.path.splitext(image_name)[0]
    return int(stem)


# -------------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("det_config", nargs="?", default=None,
                        help="Config file for detection (not needed with --track-json)")
    parser.add_argument("det_checkpoint", nargs="?", default=None,
                        help="Checkpoint file for detection (not needed with --track-json)")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--input", help="Input image dir")
    parser.add_argument("--output", default=None, help="Path to output dir")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--track-json", default=None,
        help="Path to track.json with pre-computed person bboxes. "
             "When provided, the RTMDet detector is NOT loaded.",
    )
    parser.add_argument(
        "--radius", type=int, default=3, help="Keypoint radius for visualization"
    )
    parser.add_argument(
        "--thickness", type=int, default=1, help="Link thickness for visualization"
    )
    parser.add_argument(
        "--kpt-thr", type=float, default=0.3, help="Visualizing keypoint thresholds"
    )
    parser.add_argument(
        "--bbox-thr", type=float, default=0.3, help="Bounding box score threshold"
    )
    parser.add_argument(
        "--nms-thr", type=float, default=0.3, help="IoU threshold for bounding box NMS"
    )
    parser.add_argument(
        "--no-save-json",
        action="store_true",
        help="Disable saving per-video predictions JSON (saved by default).",
    )
    parser.add_argument(
        "--predictions-name",
        default=None,
        help="Override predictions JSON filename (used by helper for per-chunk writes).",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Skip visualization image generation; only produce the JSON output.",
    )

    args = parser.parse_args()

    use_track = args.track_json is not None
    save_json = (not args.no_save_json) or args.no_vis  # always save JSON in no-vis mode

    model = init_model(args.config, args.checkpoint, device=args.device)
    os.makedirs(args.output, exist_ok=True)

    ## add pose metainfo to model
    num_keypoints = model.cfg.num_keypoints
    if num_keypoints == 308:
        model.pose_metainfo = parse_pose_metainfo(
            dict(from_file="configs/_base_/keypoints308.py")
        )

    ## add codec to model
    codec_type = model.cfg.codec.pop("type")
    assert codec_type == "UDPHeatmap", "Only support UDPHeatmap"
    model.codec = UDPHeatmap(**model.cfg.codec)

    # build detector (only when track.json is NOT provided)
    detector = None
    if use_track:
        print(f"[vis_pose] Using pre-computed bboxes from {args.track_json}")
        frame_bboxes = load_track_json(args.track_json)
    else:
        assert args.det_config and args.det_checkpoint, (
            "det_config and det_checkpoint are required when --track-json is not provided"
        )
        detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
        detector.cfg = mmdet_pipeline(detector.cfg)
        frame_bboxes = None

    # Get image list
    if os.path.isdir(args.input):
        input_dir = args.input
        image_names = [
            name
            for name in sorted(os.listdir(input_dir))
            if name.endswith((".jpg", ".png", ".jpeg"))
        ]
    else:
        with open(args.input, "r") as f:
            image_paths = [line.strip() for line in f if line.strip()]
        image_names = [os.path.basename(path) for path in image_paths]
        input_dir = os.path.dirname(image_paths[0])

    frames_records = []
    image_size = None
    num_keypoints_seen = None

    # Pointing action detector (one per tracked person; here we use person 0)
    action_detector = PointingActionDetector()
    action_metrics = []  # per-frame metrics for summary plot

    # Action label → display color (BGR for cv2)
    ACTION_COLORS = {
        PointingActionDetector.IDLE:        (200, 200, 200),  # grey
        PointingActionDetector.RAISING:     (0, 200, 255),    # orange
        PointingActionDetector.POINTING:    (0, 255, 0),      # green
        PointingActionDetector.LOWERING:    (0, 0, 255),      # red
        PointingActionDetector.BBOX_CHANGE: (255, 0, 255),    # magenta
        PointingActionDetector.NO_BBOX:     (128, 128, 128),  # dark grey
    }
    ACTION_LABELS_CN = {
        PointingActionDetector.IDLE:        "idle",
        PointingActionDetector.RAISING:     "raising",
        PointingActionDetector.POINTING:    "pointing",
        PointingActionDetector.LOWERING:    "lowering",
        PointingActionDetector.BBOX_CHANGE: "bbox_change",
        PointingActionDetector.NO_BBOX:     "no_bbox",
    }

    for image_name in tqdm(image_names, total=len(image_names)):
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)

        if use_track:
            fidx = _frame_index_from_name(image_name)
            bboxes = frame_bboxes.get(fidx, np.empty((0, 4), dtype=np.float32))
            if len(bboxes) == 0:
                # Frame has no annotation in track.json → skip pose estimation
                keypoints, keypoint_scores = [], []
            else:
                keypoints, keypoint_scores, bboxes = process_one_image_with_bbox(
                    image, bboxes, model
                )
        else:
            keypoints, keypoint_scores, bboxes = process_one_image(
                args, image, detector, model
            )

        if image_size is None:
            image_size = [int(image.shape[0]), int(image.shape[1])]
        if num_keypoints_seen is None and len(keypoints) > 0:
            num_keypoints_seen = int(np.asarray(keypoints[0]).shape[0])

        # --- Pointing action detection (use first detected person) ---
        action_label = PointingActionDetector.IDLE
        action_debug = {}
        person_bbox = bboxes[0] if len(bboxes) > 0 else None
        if len(keypoints) > 0:
            kpts_arr = np.asarray(keypoints[0])
            scores_arr = np.asarray(keypoint_scores[0])
            action_label, action_debug = action_detector.update(
                kpts_arr, scores_arr, bbox=person_bbox, kpt_thr=args.kpt_thr
            )
        else:
            # No person detected → pass dummy kpts with bbox=None to trigger NO_BBOX
            dummy_kpts = np.zeros((len(ARM_INDICES), 2), dtype=np.float32)
            dummy_scores = np.zeros(len(ARM_INDICES), dtype=np.float32)
            action_label, action_debug = action_detector.update(
                dummy_kpts, dummy_scores, bbox=None, kpt_thr=args.kpt_thr
            )

        # Collect per-frame metrics
        action_metrics.append({
            "frame": _frame_index_from_name(image_name) if use_track else len(action_metrics) + 1,
            "vel_ratio": action_debug.get("vel_ratio", 0.0),
            "vel_left": action_debug.get("vel_left", 0.0),
            "vel_right": action_debug.get("vel_right", 0.0),
            "state": action_label,
            "pos_diff": action_debug.get("pos_diff", None),
            "bbox_vel": action_debug.get("bbox_vel", 0.0),
            "active_side": action_debug.get("active_side", None),
        })

        # Filter skeleton and colors for arm keypoints
        if not args.no_vis:
            arm_skeleton, arm_link_color = filter_skeleton_for_arm(
                model.pose_metainfo["skeleton_links"],
                model.pose_metainfo["skeleton_link_colors"],
            )
            arm_kpt_color = [model.pose_metainfo["keypoint_colors"][i] for i in ARM_INDICES]

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            vis_image_rgb = visualize_keypoints(
                image=image_rgb,
                keypoints=keypoints,
                keypoints_visible=np.ones_like(keypoint_scores) > 0,
                keypoint_scores=keypoint_scores,
                radius=args.radius,
                thickness=args.thickness,
                kpt_thr=args.kpt_thr,
                skeleton=arm_skeleton,
                kpt_color=arm_kpt_color,
                link_color=arm_link_color,
                kpt_labels=[str(i) for i in ARM_INDICES],
            )
            vis_image = cv2.cvtColor(vis_image_rgb, cv2.COLOR_RGB2BGR)

            # --- Draw action label on the image ---
            side_str = action_debug.get("active_side") or ""
            side_suffix = f" [{side_str}]" if side_str else ""
            label_text = f"Action: {ACTION_LABELS_CN.get(action_label, action_label)}{side_suffix}"
            label_color = ACTION_COLORS.get(action_label, (200, 200, 200))
            vel_text = (
                f"vL={action_debug.get('vel_left', 0):.4f}  "
                f"vR={action_debug.get('vel_right', 0):.4f}  "
                f"ratio={action_debug.get('vel_ratio', 0):.2f}"
            )
            bbox_vel_val = action_debug.get("bbox_vel", 0.0)
            bbox_text = f"bbox_vel={bbox_vel_val:.3f}"
            pos_diff_val = action_debug.get('pos_diff', None)
            pos_text = (
                f"pos_diff={pos_diff_val:.4f}  {action_debug.get('pos_check', '')}"
                if pos_diff_val is not None else ""
            )
            # Background rectangle for readability
            hud_h = 110 if pos_text else 95
            cv2.rectangle(vis_image, (10, 10), (560, hud_h), (0, 0, 0), -1)
            cv2.putText(
                vis_image, label_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2, cv2.LINE_AA,
            )
            cv2.putText(
                vis_image, vel_text, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
            )
            cv2.putText(
                vis_image, bbox_text, (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1, cv2.LINE_AA,
            )
            if pos_text:
                cv2.putText(
                    vis_image, pos_text, (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA,
                )

            save_path = os.path.join(args.output, image_name)
            cv2.imwrite(save_path, vis_image)

        if save_json:
            try:
                instances = []
                for kpts, scores, bbox in zip(keypoints, keypoint_scores, bboxes):
                    instances.append({
                        "bbox": [float(v) for v in np.asarray(bbox).reshape(-1)[:4]],
                        "keypoints": np.asarray(kpts, dtype=float).tolist(),
                        "keypoint_scores": np.asarray(scores, dtype=float).reshape(-1).tolist(),
                    })
                frames_records.append({
                    "image_name": image_name,
                    "action": action_label,
                    "active_side": action_debug.get("active_side", None),
                    "action_debug": {
                        k: v for k, v in action_debug.items()
                        if isinstance(v, (int, float, str, type(None)))
                    },
                    "instances": instances,
                })
            except Exception as e:
                print(f"[vis_pose] json record failed on {image_name}: {e}")

    if save_json:
        nn = os.path.basename(os.path.normpath(args.output))
        # strip a trailing "_output" suffix so the JSON sidecar name matches the
        # video basename (e.g. ".../v3/01/<ckpt>_output/01_predictions.json").
        # `loop.sh` wraps each video output as `<video>/<ckpt>_output/`, with the
        # video number sitting one directory up.
        parent_nn = os.path.basename(os.path.dirname(os.path.normpath(args.output)))
        video_label = parent_nn if nn.endswith("_output") else nn
        json_filename = args.predictions_name or f"{video_label}_predictions.json"
        json_path = os.path.join(args.output, json_filename)
        payload = {
            "video": video_label,
            "image_size": image_size,
            "num_keypoints": num_keypoints_seen,
            "kpt_thr_used": float(args.kpt_thr),
            "frames": frames_records,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f)
        print(f"[vis_pose] wrote predictions: {json_path} ({len(frames_records)} frames)")

    # --- Generate action metrics summary plot ---
    if action_metrics:
        plot_path = os.path.join(args.output, f"{video_label}_metrics.png")
        plot_action_metrics(
            action_metrics, plot_path,
            ratio_thr=action_detector.ratio_thr,
            bbox_vel_thr=action_detector.bbox_vel_thr,
            title=video_label,
        )
        print(f"[vis_pose] wrote metrics plot: {plot_path}")


if __name__ == "__main__":
    main()
