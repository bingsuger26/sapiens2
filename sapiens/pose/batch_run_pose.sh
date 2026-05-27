#!/usr/bin/env bash
# ===========================================================================
# batch_run_pose.sh
# Batch-process all pointing video clips through the Sapiens2 pose pipeline.
#
# Multi-GPU + per-GPU multi-batch parallel.
#   - Total parallel workers = NUM_GPUS * PER_GPU_JOBS
#   - Each clip is dispatched to one worker; the worker pins itself to one
#     specific GPU via CUDA_VISIBLE_DEVICES.
#
# Usage:
#   bash batch_run_pose.sh                            # all datasets, auto-detect GPUs
#   bash batch_run_pose.sh dataset_xxx                # one specific dataset
#   NUM_GPUS=4 PER_GPU_JOBS=2 bash batch_run_pose.sh  # override (4 GPUs, 2 jobs/GPU = 8 workers)
#   PER_GPU_JOBS=2 bash batch_run_pose.sh             # 1 GPU, 2 concurrent clips on it
#
# Outputs mirror the source tree under ${OUTPUT_ROOT}.
# ===========================================================================

set -euo pipefail

# ---- Paths ----------------------------------------------------------------
DATA_ROOT="/home/sanmeng/data/pointing_resized"
OUTPUT_ROOT="/home/sanmeng/data/pointing_resized/output"
POSE_DIR="/home/sanmeng/models/sapiens2/sapiens/pose"
PYTHON="/home/sanmeng/envs/sapiens2/bin/python"

# Model configs / checkpoints (same as launch.json)
DET_CONFIG="${POSE_DIR}/tools/vis/rtmdet_m_640-8xb32_coco-person.py"
DET_CKPT="/home/perception_public/epm/weights/HF/sapiens2_host/detector/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
POSE_CONFIG="${POSE_DIR}/configs/keypoints308/shutterstock_goliath_3po/sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-512x384.py"
POSE_CKPT="/home/perception_public/epm/weights/HF/sapiens2_host/sapiens2_0.4b_pose.safetensors"

# Inference params
RADIUS=8
KPT_THR=0.3
THICKNESS=8

# ---- Parallelism config ---------------------------------------------------
# Auto-detect GPU count if NUM_GPUS not provided.
if [[ -z "${NUM_GPUS:-}" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    else
        NUM_GPUS=1
    fi
fi
NUM_GPUS=${NUM_GPUS:-1}
[[ "${NUM_GPUS}" -lt 1 ]] && NUM_GPUS=1

PER_GPU_JOBS="${PER_GPU_JOBS:-1}"          # concurrent clips per GPU
TOTAL_WORKERS=$((NUM_GPUS * PER_GPU_JOBS))

# ---- Optional: single dataset filter --------------------------------------
FILTER_DATASET="${1:-}"

# ---- Discover all video/color dirs ----------------------------------------
echo "=========================================="
echo " Scanning ${DATA_ROOT} ..."
echo " GPUs           : ${NUM_GPUS}"
echo " Jobs per GPU   : ${PER_GPU_JOBS}"
echo " Total workers  : ${TOTAL_WORKERS}"
[[ -n "${FILTER_DATASET}" ]] && echo " Filter dataset : ${FILTER_DATASET}"
echo "=========================================="

# Build task list: each line is   <input_dir>\t<output_dir>
TASK_LIST=$(mktemp)
trap 'rm -f "${TASK_LIST}" "${STATS_DIR:-/dev/null}"/* 2>/dev/null; rmdir "${STATS_DIR:-/dev/null}" 2>/dev/null || true' EXIT

# NOTE: piping `find | while` runs the loop in a subshell; `>>` to a real file
# is fine. We use a TAB separator so paths with spaces stay intact.
find "${DATA_ROOT}" -path "*/video/color" -type d | sort | while read -r color_dir; do
    rel="${color_dir#${DATA_ROOT}/}"          # dataset_xxx/uuid/0000001/video/color
    dataset_name="${rel%%/*}"

    # Apply optional filter
    if [[ -n "${FILTER_DATASET}" && "${dataset_name}" != "${FILTER_DATASET}" ]]; then
        continue
    fi

    # Skip non-dataset dirs (annotations, output, etc.)
    if [[ ! "${dataset_name}" =~ ^dataset_ ]]; then
        continue
    fi

    # Skip the output tree if it lives under DATA_ROOT
    case "${rel}" in output/*) continue ;; esac

    segment_rel="${rel%/video/color}"
    out_dir="${OUTPUT_ROOT}/${segment_rel}"

    printf '%s\t%s\n' "${color_dir}" "${out_dir}" >> "${TASK_LIST}"
done

TOTAL=$(wc -l < "${TASK_LIST}")
echo "Found ${TOTAL} video clips to process."
echo ""

if [[ "${TOTAL}" -eq 0 ]]; then
    echo "Nothing to do."
    exit 0
fi

# ---- Stats dir (one file per outcome, atomic & lock-free) -----------------
STATS_DIR=$(mktemp -d)

# ---- Worker function (invoked by xargs in subshells) ----------------------
run_one() {
    # Args: <slot_idx> <task_idx> <total> <color_dir> <out_dir>
    local slot_idx="$1"
    local task_idx="$2"
    local total="$3"
    local color_dir="$4"
    local out_dir="$5"

    local gpu_id=$(( slot_idx % NUM_GPUS ))

    # Skip if already done (output dir exists and has png files)
    if [[ -d "${out_dir}" ]] && compgen -G "${out_dir}/*.png" >/dev/null; then
        echo "[${task_idx}/${total}] [GPU${gpu_id}] SKIP (already done): ${out_dir}"
        : > "${STATS_DIR}/skip.${task_idx}"
        return 0
    fi

    echo "[${task_idx}/${total}] [GPU${gpu_id}] RUN : ${color_dir}"
    mkdir -p "${out_dir}"

    # Run inference. Pin to a single GPU; from the process's perspective it
    # only sees one card, so we pass --device cuda:0 inside.
    if cd "${POSE_DIR}" && \
       CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON}" tools/vis/vis_pose.py \
           "${DET_CONFIG}" \
           "${DET_CKPT}" \
           "${POSE_CONFIG}" \
           "${POSE_CKPT}" \
           --input "${color_dir}" \
           --output "${out_dir}" \
           --radius "${RADIUS}" \
           --kpt-thr "${KPT_THR}" \
           --thickness "${THICKNESS}" \
           --device "cuda:0" \
       >"${out_dir}/.pose.log" 2>&1; then
        echo "[${task_idx}/${total}] [GPU${gpu_id}] OK  : ${out_dir}"
        : > "${STATS_DIR}/done.${task_idx}"
    else
        echo "[${task_idx}/${total}] [GPU${gpu_id}] FAIL: ${out_dir} (see ${out_dir}/.pose.log)"
        : > "${STATS_DIR}/fail.${task_idx}"
    fi
}
export -f run_one
export POSE_DIR PYTHON DET_CONFIG DET_CKPT POSE_CONFIG POSE_CKPT \
       RADIUS KPT_THR THICKNESS NUM_GPUS STATS_DIR

# ---- Dispatch via xargs ---------------------------------------------------
# Each input line:  <slot_idx>\t<task_idx>\t<total>\t<color_dir>\t<out_dir>
# `xargs -P TOTAL_WORKERS -n 1 -I {}` keeps strict per-line dispatch and
# round-robins via slot_idx = (task_idx-1) % TOTAL_WORKERS.
awk -v W="${TOTAL_WORKERS}" -v T="${TOTAL}" 'BEGIN{FS="\t"; OFS="\t"} {
    slot = (NR - 1) % W
    print slot, NR, T, $1, $2
}' "${TASK_LIST}" | \
    xargs -d '\n' -P "${TOTAL_WORKERS}" -I {} \
        bash -c 'IFS=$'"'"'\t'"'"' read -r slot idx total cdir odir <<<"$1"; run_one "$slot" "$idx" "$total" "$cdir" "$odir"' _ {}

# ---- Summary --------------------------------------------------------------
DONE=$(ls "${STATS_DIR}"/done.* 2>/dev/null | wc -l)
SKIP=$(ls "${STATS_DIR}"/skip.* 2>/dev/null | wc -l)
FAIL=$(ls "${STATS_DIR}"/fail.* 2>/dev/null | wc -l)

echo ""
echo "=========================================="
echo " BATCH COMPLETE"
echo "=========================================="
echo " Total clips    : ${TOTAL}"
echo " Processed (OK) : ${DONE}"
echo " Skipped        : ${SKIP}"
echo " Failed         : ${FAIL}"
echo " GPUs used      : ${NUM_GPUS} x ${PER_GPU_JOBS} = ${TOTAL_WORKERS} workers"
echo "=========================================="

# Non-zero exit if any failed
[[ "${FAIL}" -gt 0 ]] && exit 1 || exit 0
