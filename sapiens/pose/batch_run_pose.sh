#!/usr/bin/env bash
# ===========================================================================
# batch_run_pose.sh
# Batch-process all pointing video clips through the Sapiens2 pose pipeline.
#
# Usage:
#   bash batch_run_pose.sh              # process ALL datasets
#   bash batch_run_pose.sh dataset_xxx  # process one specific dataset
#
# Outputs mirror the source tree under /home/sanmeng/output/
# ===========================================================================

set -euo pipefail

# ---- Paths ----------------------------------------------------------------
DATA_ROOT="/home/sanmeng/models/sapiens2/sapiens/pose/outputs/resized_img"
OUTPUT_ROOT="/home/sanmeng/outputs/results"
POSE_DIR="/home/sanmeng/models/sapiens2/sapiens/pose"
PYTHON="/home/sanmeng/envs/sapiens2/bin/python"

# Model configs / checkpoints (same as launch.json)
DET_CONFIG="${POSE_DIR}/tools/vis/rtmdet_m_640-8xb32_coco-person.py"
DET_CKPT="/home/sanmeng/models/sapiens2/sapiens2_host/detector/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
POSE_CONFIG="${POSE_DIR}/configs/keypoints308/shutterstock_goliath_3po/sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-512x384.py"
POSE_CKPT="/home/sanmeng/models/sapiens2/sapiens2_host/sapiens2_0.4b_pose.safetensors"

# Inference params
RADIUS=8
KPT_THR=0.3
THICKNESS=8
DEVICE="cuda:0"

# ---- Optional: single dataset filter --------------------------------------
FILTER_DATASET="${1:-}"

# ---- Counters --------------------------------------------------------------
TOTAL=0
DONE=0
SKIP=0
FAIL=0

# ---- Discover all video/color dirs ----------------------------------------
echo "=========================================="
echo " Scanning ${DATA_ROOT} ..."
echo "=========================================="

# Build task list: each line is   <input_dir> <output_dir>
TASK_LIST=$(mktemp)

find "${DATA_ROOT}" -path "*/video/color" -type d | sort | while read -r color_dir; do
    # Example color_dir:
    #   /home/data/hmi/slices/pointing/dataset_xxx/person_uuid/0000001/video/color

    # Extract relative path after DATA_ROOT
    rel="${color_dir#${DATA_ROOT}/}"              # dataset_xxx/uuid/0000001/video/color

    # Extract dataset name (first component)
    dataset_name="${rel%%/*}"

    # Apply optional filter
    if [[ -n "${FILTER_DATASET}" && "${dataset_name}" != "${FILTER_DATASET}" ]]; then
        continue
    fi

    # Skip non-dataset dirs (annotations, etc.)
    if [[ ! "${dataset_name}" =~ ^dataset_ ]]; then
        continue
    fi

    # Build output path: keep structure but strip "video/color" suffix
    # e.g. dataset_xxx/uuid/0000001/video/color  ->  dataset_xxx/uuid/0000001
    segment_rel="${rel%/video/color}"
    out_dir="${OUTPUT_ROOT}/${segment_rel}"

    echo "${color_dir} ${out_dir}" >> "${TASK_LIST}"
done

TOTAL=$(wc -l < "${TASK_LIST}")
echo "Found ${TOTAL} video clips to process."
echo ""

if [[ "${TOTAL}" -eq 0 ]]; then
    echo "Nothing to do."
    rm -f "${TASK_LIST}"
    exit 0
fi

# ---- Process ---------------------------------------------------------------
IDX=0
while read -r color_dir out_dir; do
    IDX=$((IDX + 1))

    # Check if already processed (output dir exists and has images)
    if [[ -d "${out_dir}" ]] && ls "${out_dir}"/*.png &>/dev/null; then
        SKIP=$((SKIP + 1))
        echo "[${IDX}/${TOTAL}] SKIP (already done): ${out_dir}"
        continue
    fi

    echo "[${IDX}/${TOTAL}] Processing: ${color_dir}"
    echo "            Output:     ${out_dir}"

    mkdir -p "${out_dir}"

    if cd "${POSE_DIR}" && \
       CUDA_VISIBLE_DEVICES=0 "${PYTHON}" tools/vis/vis_pose.py \
           "${DET_CONFIG}" \
           "${DET_CKPT}" \
           "${POSE_CONFIG}" \
           "${POSE_CKPT}" \
           --input "${color_dir}" \
           --output "${out_dir}" \
           --radius "${RADIUS}" \
           --kpt-thr "${KPT_THR}" \
           --thickness "${THICKNESS}" \
           --device "${DEVICE}" \
       2>&1 | tail -3; then
        DONE=$((DONE + 1))
        echo "            -> OK"
    else
        FAIL=$((FAIL + 1))
        echo "            -> FAILED"
    fi

    echo ""
done < "${TASK_LIST}"

rm -f "${TASK_LIST}"

# ---- Summary ---------------------------------------------------------------
echo "=========================================="
echo " BATCH COMPLETE"
echo "=========================================="
echo " Total clips : ${TOTAL}"
echo " Processed   : ${DONE}"
echo " Skipped     : ${SKIP}"
echo " Failed      : ${FAIL}"
echo "=========================================="
