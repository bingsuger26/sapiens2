#!/usr/bin/env python
"""
Batch-resize all video frame images under the pointing dataset to a lower
resolution, and rescale the bbox / point annotations in the corresponding
track.json files.  A parallel directory tree is created so originals are
untouched.

Default target size: 768 x 512  (width x height).

Output layout (mirrors source):
  <OUTPUT_ROOT>/dataset_xxx/uuid/segment/video/color/00001.png     (resized image)
  <OUTPUT_ROOT>/dataset_xxx/video_annotations/uuid/segment/track.json  (rescaled annotations)

Usage:
    python resize_images.py
    python resize_images.py --dataset dataset_20260323101357_f58270ae
    python resize_images.py --dataset dataset_a dataset_b dataset_c
    python resize_images.py --width 768 --height 512
    python resize_images.py --workers 8
"""

import argparse
import copy
import json
import os

import cv2
from tqdm import tqdm

# ---------------------------------------------------------------------------
DATA_ROOT = "/home/data/hmi/slices/pointing"
OUTPUT_ROOT = "/home/sanmeng/data/pointing_resized"


# ---------------------------------------------------------------------------
def discover_segments(data_root, filter_dataset=None):
    """Discover all segments that have track.json annotations.

    `filter_dataset` may be:
        - None       : process every dataset under data_root
        - str        : process a single dataset
        - list/tuple : process the given datasets (order preserved, dedup)

    Each returned dict:
        dataset   : dataset_xxx
        uuid      : uuid string
        segment   : 0000001
        color_dir : absolute path to .../video/color
        track_json: absolute path to track.json
    """
    # Step 1: collect dataset dirs (filter early)
    if filter_dataset:
        if isinstance(filter_dataset, (list, tuple)):
            # Preserve order, drop duplicates
            seen = set()
            candidates = []
            for d in filter_dataset:
                if d not in seen:
                    seen.add(d)
                    candidates.append(d)
        else:
            candidates = [filter_dataset]
    else:
        candidates = sorted(
            d for d in os.listdir(data_root)
            if d.startswith("dataset_") and os.path.isdir(os.path.join(data_root, d))
        )

    # Step 2: for each dataset, walk video_annotations/uuid/segment to find track.json,
    #         then check the corresponding color dir exists.
    #
    # Directory layout:
    #   dataset_xxx/
    #     <uuid>/                          ← image data
    #       <segment>/video/color/*.png
    #     video_annotations/               ← annotations
    #       <uuid>/<segment>/track.json
    segments = []
    for dataset_name in candidates:
        dataset_dir = os.path.join(data_root, dataset_name)
        annot_root = os.path.join(dataset_dir, "video_annotations")

        if not os.path.isdir(annot_root):
            continue

        for uuid_name in sorted(os.listdir(annot_root)):
            uuid_annot_dir = os.path.join(annot_root, uuid_name)
            if not os.path.isdir(uuid_annot_dir):
                continue

            for segment_name in sorted(os.listdir(uuid_annot_dir)):
                track_path = os.path.join(uuid_annot_dir, segment_name, "track.json")
                if not os.path.isfile(track_path):
                    continue

                color_dir = os.path.join(
                    dataset_dir, uuid_name, segment_name, "video", "color"
                )
                if not os.path.isdir(color_dir):
                    continue

                segments.append({
                    "dataset": dataset_name,
                    "uuid": uuid_name,
                    "segment": segment_name,
                    "color_dir": color_dir,
                    "track_json": track_path,
                })

    return segments


# ---------------------------------------------------------------------------
def resize_images(src_dir, dst_dir, width, height, skip_existing=True):
    """Resize all images in src_dir → dst_dir.  Returns (processed, orig_wh).

    orig_wh is the (width, height) of the first image read (used to compute
    the scale factors).
    """
    os.makedirs(dst_dir, exist_ok=True)

    names = sorted(
        n for n in os.listdir(src_dir)
        if n.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not names:
        return 0, None

    orig_wh = None
    count = 0
    for name in names:
        dst_path = os.path.join(dst_dir, name)
        if skip_existing and os.path.isfile(dst_path):
            # Still need original size for annotation scaling
            if orig_wh is None:
                img = cv2.imread(os.path.join(src_dir, name))
                if img is not None:
                    orig_wh = (img.shape[1], img.shape[0])
            continue

        src_path = os.path.join(src_dir, name)
        img = cv2.imread(src_path)
        if img is None:
            continue

        if orig_wh is None:
            orig_wh = (img.shape[1], img.shape[0])

        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst_path, resized)
        count += 1

    return count, orig_wh


def rescale_track_json(src_path, dst_path, sx, sy, skip_existing=True):
    """Read a track.json, scale all coordinate fields, and write to dst_path.

    sx = new_width  / orig_width
    sy = new_height / orig_height

    Scaled fields per detection (person_detections & object_detections):
      - bbox_xyxy : [x1, y1, x2, y2]  → multiply x by sx, y by sy
      - 2dpoint   : [x, y]            → multiply x by sx, y by sy
    """
    if skip_existing and os.path.isfile(dst_path):
        return

    with open(src_path, "r") as f:
        data = json.load(f)

    scaled = copy.deepcopy(data)

    for frame in scaled["track_data"]:
        for det_key in ("person_detections", "object_detections"):
            for det in frame.get(det_key, []):
                # bbox_xyxy: [x1, y1, x2, y2]
                if "bbox_xyxy" in det and len(det["bbox_xyxy"]) == 4:
                    bx = det["bbox_xyxy"]
                    det["bbox_xyxy"] = [
                        round(bx[0] * sx, 2),
                        round(bx[1] * sy, 2),
                        round(bx[2] * sx, 2),
                        round(bx[3] * sy, 2),
                    ]
                # 2dpoint: [x, y]
                if "2dpoint" in det and len(det["2dpoint"]) == 2:
                    pt = det["2dpoint"]
                    det["2dpoint"] = [
                        round(pt[0] * sx, 2),
                        round(pt[1] * sy, 2),
                    ]

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(scaled, f, indent=2)


# ---------------------------------------------------------------------------
def process_segment(seg, output_root, width, height, skip_existing):
    """Process one segment: resize images + rescale annotations.

    Returns number of images processed.
    """
    # --- images ---
    rel_color = os.path.join(
        seg["dataset"], seg["uuid"], seg["segment"], "video", "color"
    )
    dst_color = os.path.join(output_root, rel_color)
    n_images, orig_wh = resize_images(
        seg["color_dir"], dst_color, width, height, skip_existing
    )

    # --- annotations ---
    if seg["track_json"] is not None and orig_wh is not None:
        orig_w, orig_h = orig_wh
        sx = width / orig_w
        sy = height / orig_h

        rel_track = os.path.join(
            seg["dataset"], "video_annotations",
            seg["uuid"], seg["segment"], "track.json"
        )
        dst_track = os.path.join(output_root, rel_track)
        rescale_track_json(seg["track_json"], dst_track, sx, sy, skip_existing)

    return n_images


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Batch-resize pointing dataset images and rescale annotations"
    )
    parser.add_argument(
        "--dataset", default=None, nargs="+",
        help=(
            "Process only the given dataset(s); accepts one or more dataset "
            "names (e.g. --dataset dataset_20260323101357_f58270ae "
            "dataset_20260325140925_4d83e35d). If omitted, process all."
        ),
    )
    parser.add_argument("--width", type=int, default=768, help="Target width")
    parser.add_argument("--height", type=int, default=512, help="Target height")
    parser.add_argument(
        "--output-root", default=OUTPUT_ROOT,
        help=f"Root directory for output (default: {OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Re-process even if output already exists",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1, single-process)",
    )
    args = parser.parse_args()

    segments = discover_segments(DATA_ROOT, filter_dataset=args.dataset)
    n_with_track = sum(1 for s in segments if s["track_json"] is not None)
    print(f"Found {len(segments)} segments ({n_with_track} with track.json)")
    if not segments:
        print("Nothing to do.")
        return

    print(f"Target resolution : {args.width} x {args.height}")
    print(f"Output root       : {args.output_root}")
    print()

    total_images = 0
    total_tracks = 0
    skipped = 0
    skip_existing = not args.no_skip

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for seg in segments:
                fut = pool.submit(
                    process_segment, seg, args.output_root,
                    args.width, args.height, skip_existing,
                )
                futures[fut] = seg

            progress = tqdm(
                as_completed(futures), total=len(futures),
                desc="Resizing", unit="seg",
            )
            for fut in progress:
                n = fut.result()
                total_images += n
                seg = futures[fut]
                if seg["track_json"] is not None:
                    total_tracks += 1
                progress.set_postfix(images=total_images, tracks=total_tracks)
    else:
        progress = tqdm(segments, desc="Resizing", unit="seg")
        for seg in progress:
            n = process_segment(
                seg, args.output_root, args.width, args.height, skip_existing,
            )
            total_images += n
            if seg["track_json"] is not None:
                total_tracks += 1
            progress.set_postfix(images=total_images, tracks=total_tracks)

    print()
    print("=" * 55)
    print(" RESIZE COMPLETE")
    print("=" * 55)
    print(f" Segments       : {len(segments)}")
    print(f" Images resized : {total_images}")
    print(f" Tracks scaled  : {total_tracks}")
    print(f" Target size    : {args.width} x {args.height}")
    print(f" Output         : {args.output_root}")
    print("=" * 55)


if __name__ == "__main__":
    main()
