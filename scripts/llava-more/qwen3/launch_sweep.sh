#!/bin/bash
# Submit the full CLIP + SAE + Qwen3 training sweep.
#
# Each SAE gets its own pretrain → finetune pair, because each SAE produces
# different feature distributions — even enc+dec variants with the same output
# dimensionality encode different information.
#
# Job graph (per model size):
#
#   pretrain-baseline              → finetune-baseline
#   pretrain-imagenet-enconly      → finetune-imagenet-enconly
#   pretrain-imagenet-encdec       → finetune-imagenet-encdec
#   pretrain-cc3m_laion-enconly    → finetune-cc3m_laion-enconly
#   pretrain-cc3m_laion-encdec     → finetune-cc3m_laion-encdec
#   pretrain-llava_ov-enconly      → finetune-llava_ov-enconly    [if SAE exists]
#   pretrain-llava_ov-encdec       → finetune-llava_ov-encdec     [if SAE exists]
#
# Stage-1 strategy:
#   ≤4B:  pretrain_batch.sh — 4 configs in parallel, 1 GPU each (2 nodes)
#   8B+:  individual pretrain.sh jobs — 4 GPUs each (~4x faster, fits in 12h)
#
# Usage:
#   bash launch_sweep.sh [MODEL_SIZE]
#   MODEL_SIZE: 0.6B | 1.7B | 4B | 8B | 14B  (default: 4B)

set -e

MODEL_SIZE=${1:-"4B"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Canonical SAE symlink paths ----
SAE_BASE="$SCRATCH/grob1/sae"
SAE_IMAGENET="${SAE_BASE}/imagenet_clip_l22/ae.pt"
SAE_CC3M="${SAE_BASE}/cc3m_laion_clip_l22/ae.pt"
SAE_LLAVA_OV="${SAE_BASE}/llava_ov_clip_l22/ae.pt"

echo "============================================================"
echo " LLaVA-MORE Sweep — Qwen3-${MODEL_SIZE}"
echo "============================================================"

# Check which SAE checkpoints are available
check_sae() {
    local path=$1 tag=$2
    if [[ ! -f "${path}" ]]; then
        echo "[WARN] Not found: ${path}  → ${tag} jobs skipped"
        return 1
    fi
    return 0
}

IMAGENET_OK=1; check_sae "${SAE_IMAGENET}" "imagenet"   || IMAGENET_OK=0
CC3M_OK=1;     check_sae "${SAE_CC3M}"     "cc3m_laion" || CC3M_OK=0
LLAVOV_OK=1;   check_sae "${SAE_LLAVA_OV}" "llava_ov"   || LLAVOV_OK=0

echo ""

# ---- Submit helpers ----
submit() {
    local desc=$1; shift
    local job_id
    job_id=$(sbatch "$@" | awk '{print $NF}')
    printf "  %-50s → job %s\n" "${desc}" "${job_id}" >&2
    echo "${job_id}"
}

submit_dep() {
    local desc=$1 dep=$2; shift 2
    local job_id
    job_id=$(sbatch --dependency=afterok:"${dep}" "$@" | awk '{print $NF}')
    printf "  %-50s → job %s  (after %s)\n" "${desc}" "${job_id}" "${dep}" >&2
    echo "${job_id}"
}

# ============================================================
# Stage 1: submit pretrain jobs
# ============================================================

# Decide strategy based on model size
case "${MODEL_SIZE}" in
    8B|14B) LARGE_MODEL=1 ;;
    *)      LARGE_MODEL=0 ;;
esac

PT_JOB_IDS=()

if [[ "${LARGE_MODEL}" == "1" ]]; then
    # Individual jobs, 4 GPUs each — fits in 12h for large models
    echo "--- Stage 1: submitting individual pretrain jobs (4 GPUs each) ---"
    PT_JOB_IDS+=("$(submit "pretrain baseline" \
        "${SCRIPT_DIR}/pretrain.sh" "${MODEL_SIZE}")")
    if [[ "${IMAGENET_OK}" == "1" ]]; then
        PT_JOB_IDS+=("$(submit "pretrain enconly imagenet" \
            "${SCRIPT_DIR}/pretrain.sh" "${MODEL_SIZE}" "--sae-enconly" "imagenet")")
        PT_JOB_IDS+=("$(submit "pretrain encdec  imagenet" \
            "${SCRIPT_DIR}/pretrain.sh" "${MODEL_SIZE}" "--sae-encdec"  "imagenet")")
    fi
    if [[ "${CC3M_OK}" == "1" ]]; then
        PT_JOB_IDS+=("$(submit "pretrain enconly cc3m_laion" \
            "${SCRIPT_DIR}/pretrain.sh" "${MODEL_SIZE}" "--sae-enconly" "cc3m_laion")")
        PT_JOB_IDS+=("$(submit "pretrain encdec  cc3m_laion" \
            "${SCRIPT_DIR}/pretrain.sh" "${MODEL_SIZE}" "--sae-encdec"  "cc3m_laion")")
    fi
    if [[ "${LLAVOV_OK}" == "1" ]]; then
        PT_JOB_IDS+=("$(submit "pretrain enconly llava_ov" \
            "${SCRIPT_DIR}/pretrain.sh" "${MODEL_SIZE}" "--sae-enconly" "llava_ov")")
        PT_JOB_IDS+=("$(submit "pretrain encdec  llava_ov" \
            "${SCRIPT_DIR}/pretrain.sh" "${MODEL_SIZE}" "--sae-encdec"  "llava_ov")")
    fi
else
    # Batch jobs: 4 configs in parallel on 1 GPU each (2 nodes)
    echo "--- Stage 1: submitting 2 pretrain-batch nodes ---"
    PT_JOB_IDS+=("$(submit "pretrain-batch slot0" \
        "${SCRIPT_DIR}/pretrain_batch.sh" "${MODEL_SIZE}" "0")")
    PT_JOB_IDS+=("$(submit "pretrain-batch slot1" \
        "${SCRIPT_DIR}/pretrain_batch.sh" "${MODEL_SIZE}" "1")")
fi

# Build colon-separated dependency string for all pretrain jobs
PT_DEP=$(IFS=':'; echo "${PT_JOB_IDS[*]}")

# ============================================================
# Stage 2: finetunes — wait for all pretrain jobs
# ============================================================
echo ""
echo "--- Stage 2: finetunes (after ${PT_DEP}) ---"
submit_dep "finetune baseline"           "${PT_DEP}" "${SCRIPT_DIR}/finetune.sh" "${MODEL_SIZE}"

if [[ "${IMAGENET_OK}" == "1" ]]; then
    submit_dep "finetune enconly imagenet"   "${PT_DEP}" "${SCRIPT_DIR}/finetune.sh" "${MODEL_SIZE}" "--sae-enconly" "imagenet"
    submit_dep "finetune encdec  imagenet"   "${PT_DEP}" "${SCRIPT_DIR}/finetune.sh" "${MODEL_SIZE}" "--sae-encdec"  "imagenet"
fi

if [[ "${CC3M_OK}" == "1" ]]; then
    submit_dep "finetune enconly cc3m_laion" "${PT_DEP}" "${SCRIPT_DIR}/finetune.sh" "${MODEL_SIZE}" "--sae-enconly" "cc3m_laion"
    submit_dep "finetune encdec  cc3m_laion" "${PT_DEP}" "${SCRIPT_DIR}/finetune.sh" "${MODEL_SIZE}" "--sae-encdec"  "cc3m_laion"
fi

if [[ "${LLAVOV_OK}" == "1" ]]; then
    submit_dep "finetune enconly llava_ov"   "${PT_DEP}" "${SCRIPT_DIR}/finetune.sh" "${MODEL_SIZE}" "--sae-enconly" "llava_ov"
    submit_dep "finetune encdec  llava_ov"   "${PT_DEP}" "${SCRIPT_DIR}/finetune.sh" "${MODEL_SIZE}" "--sae-encdec"  "llava_ov"
fi

echo ""
echo "============================================================"
echo " All jobs submitted. Monitor with: squeue -u \$USER"
echo " Checkpoints: \$SCRATCH/grob1/llava-more/checkpoints/"
echo "============================================================"
