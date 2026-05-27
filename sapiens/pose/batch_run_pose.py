#!/usr/bin/env python
"""
Batch-process ALL pointing video clips through Sapiens2 pose + action pipeline.

Key optimisation: models are loaded **once**, then reused for every clip.

Usage:
    # Process ALL datasets
    python batch_run_pose.py

    # Process a single dataset
    python batch_run_pose.py --dataset dataset_20260325142850_ad831a18

    # Process with a specific GPU
    python batch_run_pose.py --device cuda:1

    # Resume (skip already-processed clips)
    python batch_run_pose.py          # skipping is on by default

Outputs mirror the source tree under OUTPUT_ROOT (/home/sanmeng/output/).
"""

import os
import sys
import json
import time
import glob
import argparse
import traceback

import cv2
import numpy as np
from tqdm import tqdm

# ---- Ensure the sapiens package is importable ----------------------------
POSE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(POSE_DIR, "..", ".."))
VIS_DIR = os.path.join(POSE_DIR, "tools", "vis")
for p in (REPO_ROOT, POSE_DIR, VIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Block mmpretrain from failing noisily (optional dep)
sys.modules.setdefault("mmpretrain", None)

from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
from sapiens.pose.src.models.init_model import init_model

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# Reuse helpers from vis_pose
from tools.vis.vis_pose import (
    ARM_INDICES,
    filter_skeleton_for_arm,
    PointingActionDetector,
    mmdet_pipeline,
    process_one_image,
    process_one_image_with_bbox,
    load_track_json,
)
from tools.vis.pose_render_utils import visualize_keypoints
from tools.vis.plot_metrics import plot_action_metrics, plot_summary_distribution


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = "/home/sanmeng/models/sapiens2/sapiens/pose/outputs/resized_img"
OUTPUT_ROOT = "/home/sanmeng/output"

DET_CONFIG = os.path.join(POSE_DIR, "tools/vis/rtmdet_m_640-8xb32_coco-person.py")
DET_CKPT = "/home/sanmeng/models/sapiens2/sapiens2_host/detector/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
POSE_CONFIG = os.path.join(
    POSE_DIR,
    "configs/keypoints308/shutterstock_goliath_3po/"
    "sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py",
)
POSE_CONFIG_FAST = os.path.join(
    POSE_DIR,
    "configs/keypoints308/shutterstock_goliath_3po/"
    "sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-512x384.py",
)
POSE_CKPT = "/home/sanmeng/models/sapiens2/sapiens2_host/sapiens2_0.4b_pose.safetensors"


# ---------------------------------------------------------------------------
def discover_clips(data_root, filter_dataset=None):
    """Find all .../video/color directories and return
    (input_dir, output_dir, track_json_or_None) tuples.

    track.json lives at:
      <data_root>/dataset_xxx/video_annotations/<uuid>/<segment>/track.json
    while images live at:
      <data_root>/dataset_xxx/<uuid>/<segment>/video/color/
    """
    tasks = []
    for color_dir in sorted(glob.glob(os.path.join(data_root, "dataset_*/*/[0-9]*/video/color"))):
        rel = os.path.relpath(color_dir, data_root)     # dataset_xxx/uuid/0000001/video/color
        parts = rel.split(os.sep)
        dataset_name = parts[0]
        uuid_name = parts[1]
        segment_name = parts[2]

        if filter_dataset and dataset_name != filter_dataset:
            continue

        segment_rel = os.path.join(dataset_name, uuid_name, segment_name)
        out_dir = os.path.join(OUTPUT_ROOT, segment_rel)

        # Try to locate track.json
        track_path = os.path.join(
            data_root, dataset_name, "video_annotations",
            uuid_name, segment_name, "track.json"
        )
        track_json = track_path if os.path.isfile(track_path) else None

        tasks.append((color_dir, out_dir, track_json))
    return tasks


def is_already_done(out_dir, no_vis=False):
    """Check whether an output directory already has results."""
    if not os.path.isdir(out_dir):
        return False
    jsons = glob.glob(os.path.join(out_dir, "*_predictions.json"))
    if no_vis:
        return len(jsons) > 0
    pngs = glob.glob(os.path.join(out_dir, "*.png"))
    return len(pngs) > 0 and len(jsons) > 0


def _frame_index_from_name(image_name):
    """Extract integer frame index from an image filename like '00001.png'."""
    stem = os.path.splitext(image_name)[0]
    return int(stem)


# ---------------------------------------------------------------------------
def process_clip(args, input_dir, output_dir, detector, model,
                 arm_skeleton, arm_link_color, arm_kpt_color,
                 track_json=None, no_vis=False):
    """Process a single video clip (all frames in input_dir).

    If track_json is provided, person bboxes are read from it instead of
    running RTMDet detection.
    If no_vis is True, skip visualization image generation.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_names = sorted(
        n for n in os.listdir(input_dir)
        if n.endswith((".jpg", ".png", ".jpeg"))
        and not n.startswith("video_annotation")
    )
    if not image_names:
        return 0, []

    # Load pre-computed bboxes if available
    frame_bboxes = None
    if track_json is not None:
        frame_bboxes = load_track_json(track_json)

    action_detector = PointingActionDetector()
    action_metrics = []  # per-frame metrics for summary plot

    ACTION_COLORS = {
        PointingActionDetector.IDLE:        (200, 200, 200),
        PointingActionDetector.RAISING:     (0, 200, 255),
        PointingActionDetector.POINTING:    (0, 255, 0),
        PointingActionDetector.LOWERING:    (0, 0, 255),
        PointingActionDetector.BBOX_CHANGE: (255, 0, 255),
        PointingActionDetector.NO_BBOX:     (128, 128, 128),
    }
    ACTION_LABELS = {
        PointingActionDetector.IDLE:        "idle",
        PointingActionDetector.RAISING:     "raising",
        PointingActionDetector.POINTING:    "pointing",
        PointingActionDetector.LOWERING:    "lowering",
        PointingActionDetector.BBOX_CHANGE: "shoulder_change",
        PointingActionDetector.NO_BBOX:     "no_bbox",
    }

    frames_records = []
    image_size = None
    num_keypoints_seen = None

    for image_name in image_names:
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        try:
            if frame_bboxes is not None:
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
        except Exception as e:
            print(f"  [WARN] inference failed on {image_name}: {e}")
            continue

        if image_size is None:
            image_size = [int(image.shape[0]), int(image.shape[1])]
        if num_keypoints_seen is None and len(keypoints) > 0:
            num_keypoints_seen = int(np.asarray(keypoints[0]).shape[0])

        # Action detection
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
            "frame": _frame_index_from_name(image_name) if frame_bboxes is not None else len(action_metrics) + 1,
            "vel_ratio": action_debug.get("vel_ratio", 0.0),
            "vel_left": action_debug.get("vel_left", 0.0),
            "vel_right": action_debug.get("vel_right", 0.0),
            "state": action_label,
            "pos_diff": action_debug.get("pos_diff", None),
            "bbox_vel": action_debug.get("bbox_vel", 0.0),
            "active_side": action_debug.get("active_side", None),
        })

        # Visualise
        if not no_vis:
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

            # --- Draw shoulder midpoint (cyan diamond) ---
            shoulder_mid = action_debug.get("shoulder_mid")
            if shoulder_mid is not None:
                sm = (int(round(shoulder_mid[0])), int(round(shoulder_mid[1])))
                cv2.drawMarker(vis_image, sm, (255, 255, 0), cv2.MARKER_DIAMOND, 16, 2)

            # --- Draw hand mean points (left=green circle, right=blue circle) ---
            lh_mean = action_debug.get("left_hand_mean")
            rh_mean = action_debug.get("right_hand_mean")
            if lh_mean is not None:
                lhp = (int(round(lh_mean[0])), int(round(lh_mean[1])))
                cv2.circle(vis_image, lhp, 12, (0, 255, 0), 3)
                cv2.putText(vis_image, "LH", (lhp[0] + 14, lhp[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            if rh_mean is not None:
                rhp = (int(round(rh_mean[0])), int(round(rh_mean[1])))
                cv2.circle(vis_image, rhp, 12, (255, 100, 0), 3)
                cv2.putText(vis_image, "RH", (rhp[0] + 14, rhp[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1, cv2.LINE_AA)

            # HUD
            side_str = action_debug.get("active_side") or ""
            side_suffix = f" [{side_str}]" if side_str else ""
            label_text = f"Action: {ACTION_LABELS.get(action_label, action_label)}{side_suffix}"
            label_color = ACTION_COLORS.get(action_label, (200, 200, 200))
            vel_text = (
                f"vL={action_debug.get('vel_left', 0):.4f}  "
                f"vR={action_debug.get('vel_right', 0):.4f}  "
                f"ratio={action_debug.get('vel_ratio', 0):.2f}"
            )
            shoulder_vel_val = action_debug.get("shoulder_vel", 0.0)
            shoulder_text = f"shoulder_vel={shoulder_vel_val:.3f}"
            pos_diff_val = action_debug.get("pos_diff", None)
            pos_text = (
                f"pos_diff={pos_diff_val:.4f}  {action_debug.get('pos_check', '')}"
                if pos_diff_val is not None else ""
            )
            hud_h = 110 if pos_text else 95
            cv2.rectangle(vis_image, (10, 10), (560, hud_h), (0, 0, 0), -1)
            cv2.putText(vis_image, label_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2, cv2.LINE_AA)
            cv2.putText(vis_image, vel_text, (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(vis_image, shoulder_text, (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1, cv2.LINE_AA)
            if pos_text:
                cv2.putText(vis_image, pos_text, (20, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA)

            cv2.imwrite(os.path.join(output_dir, image_name), vis_image)

        # JSON record
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
                    if isinstance(v, (int, float, str, list, type(None)))
                },
                "instances": instances,
            })
        except Exception:
            pass

    # Save predictions JSON
    if frames_records:
        segment_name = os.path.basename(os.path.normpath(output_dir))
        json_path = os.path.join(output_dir, f"{segment_name}_predictions.json")
        payload = {
            "video": segment_name,
            "image_size": image_size,
            "num_keypoints": num_keypoints_seen,
            "kpt_thr_used": float(args.kpt_thr),
            "frames": frames_records,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f)

    # --- Generate action metrics summary plot ---
    if action_metrics:
        segment_name = os.path.basename(os.path.normpath(output_dir))
        plot_path = os.path.join(output_dir, f"{segment_name}_metrics.png")
        plot_action_metrics(
            action_metrics, plot_path,
            ratio_thr=action_detector.ratio_thr,
            bbox_vel_thr=action_detector.bbox_vel_thr,
            title=segment_name,
        )

    return len(image_names), action_metrics


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch Sapiens2 pose + action")
    parser.add_argument("--dataset", default=None,
                        help="Process only this dataset (e.g. dataset_20260325142850_ad831a18)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("--thickness", type=int, default=8)
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument("--bbox-thr", type=float, default=0.3)
    parser.add_argument("--nms-thr", type=float, default=0.3)
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-process clips even if output already exists")
    parser.add_argument("--use-track", action="store_true",
                        help="Use pre-computed bboxes from track.json when available, "
                             "skipping RTMDet detection for those clips. "
                             "Clips without track.json still use the detector.")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip visualization image generation; only produce JSON output.")
    parser.add_argument("--fast", action="store_true",
                        help="Use the 512x384 config for ~4x faster inference (lower accuracy).")
    args = parser.parse_args()

    pose_config = POSE_CONFIG_FAST if args.fast else POSE_CONFIG

    # ------------------------------------------------------------------
    # Discover clips first so we know whether detector is needed
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f" Scanning {DATA_ROOT} ...")
    print("=" * 60)
    tasks = discover_clips(DATA_ROOT, filter_dataset=args.dataset)
    print(f" Found {len(tasks)} video clips")

    if not tasks:
        print("Nothing to do.")
        return

    # Check if any clip lacks track.json (needs detector)
    need_detector = (not args.use_track) or any(t[2] is None for t in tasks)
    if args.use_track:
        n_with_track = sum(1 for t in tasks if t[2] is not None)
        n_without = len(tasks) - n_with_track
        print(f"  {n_with_track} clips have track.json, {n_without} need detector")
    print()

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" Loading models (one-time) ...")
    print(f"   Pose config: {os.path.basename(pose_config)}")
    print("=" * 60)
    t0 = time.time()

    model = init_model(pose_config, POSE_CKPT, device=args.device)
    num_keypoints = model.cfg.num_keypoints
    if num_keypoints == 308:
        model.pose_metainfo = parse_pose_metainfo(
            dict(from_file=os.path.join(POSE_DIR, "configs/_base_/keypoints308.py"))
        )
    codec_type = model.cfg.codec.pop("type")
    assert codec_type == "UDPHeatmap"
    model.codec = UDPHeatmap(**model.cfg.codec)

    detector = None
    if need_detector:
        assert has_mmdet, "mmdet is required for detection"
        detector = init_detector(DET_CONFIG, DET_CKPT, device=args.device)
        detector.cfg = mmdet_pipeline(detector.cfg)
    else:
        print("  [INFO] All clips have track.json – RTMDet detector NOT loaded")

    # Pre-compute arm skeleton/colors (constant across clips)
    arm_skeleton, arm_link_color = filter_skeleton_for_arm(
        model.pose_metainfo["skeleton_links"],
        model.pose_metainfo["skeleton_link_colors"],
    )
    arm_kpt_color = [model.pose_metainfo["keypoint_colors"][i] for i in ARM_INDICES]

    print(f" Models loaded in {time.time() - t0:.1f}s")
    print()

    # ------------------------------------------------------------------
    done = 0
    skipped = 0
    failed = 0
    total_frames = 0
    all_metrics = []  # aggregated metrics across all clips

    progress = tqdm(tasks, desc="Clips", unit="clip")
    for input_dir, output_dir, track_json in progress:
        # Skip if already done
        if not args.no_skip and is_already_done(output_dir, no_vis=args.no_vis):
            skipped += 1
            progress.set_postfix(done=done, skip=skipped, fail=failed)
            continue

        # Decide whether to use track.json for this clip
        clip_track = track_json if args.use_track else None

        try:
            n_frames, clip_metrics = process_clip(
                args, input_dir, output_dir, detector, model,
                arm_skeleton, arm_link_color, arm_kpt_color,
                track_json=clip_track,
                no_vis=args.no_vis,
            )
            done += 1
            total_frames += n_frames
            all_metrics.extend(clip_metrics)
        except Exception as e:
            failed += 1
            tqdm.write(f"  [FAIL] {input_dir}: {e}")
            traceback.print_exc()

        progress.set_postfix(done=done, skip=skipped, fail=failed, frames=total_frames)

    # ------------------------------------------------------------------
    # Generate summary distribution plot across all processed clips
    # ------------------------------------------------------------------
    if all_metrics:
        summary_path = os.path.join(OUTPUT_ROOT, "summary_velocity_ratio_distribution.png")
        plot_summary_distribution(
            all_metrics, summary_path,
            ratio_thr=PointingActionDetector().ratio_thr,
            vel_min=PointingActionDetector().vel_min,
            title=f"Batch Summary ({done} clips, {total_frames} frames)",
        )
        print(f" Summary plot: {summary_path}")

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(" BATCH COMPLETE")
    print("=" * 60)
    print(f" Total clips  : {len(tasks)}")
    print(f" Processed    : {done}")
    print(f" Skipped      : {skipped}")
    print(f" Failed       : {failed}")
    print(f" Total frames : {total_frames}")
    print("=" * 60)


if __name__ == "__main__":
    main()
