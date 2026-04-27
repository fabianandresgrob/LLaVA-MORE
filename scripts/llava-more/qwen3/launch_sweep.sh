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
#   pretrain-llava_ov-enconly      → finetune-llava_ov-enconly    [if LLAVA_OV_READY=1]
#   pretrain-llava_ov-encdec       → finetune-llava_ov-encdec     [if LLAVA_OV_READY=1]
#
# Usage:
#   bash launch_sweep.sh [MODEL_SIZE] [LLAVA_OV_READY]
#   MODEL_SIZE:     1.7B | 4B | 8B  (default: 4B)
#   LLAVA_OV_READY: 0 | 1           (default: 0)
#
# Examples:
#   bash launch_sweep.sh 4B 0   # 5 pretrain + 5 finetune = 10 jobs
#   bash launch_sweep.sh 4B 1   # 7 pretrain + 7 finetune = 14 jobs

set -e

MODEL_SIZE=${1:-"4B"}
LLAVA_OV_READY=${2:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Canonical SAE symlink paths ----
# Create with: ln -s <full/path/to/ae.pt> $SCRATCH/grob1/sae/<name>_clip_l22/ae.pt
SAE_BASE="$SCRATCH/grob1/sae"
SAE_IMAGENET="${SAE_BASE}/imagenet_clip_l22/ae.pt"
SAE_CC3M="${SAE_BASE}/cc3m_laion_clip_l22/ae.pt"
SAE_LLAVA_OV="${SAE_BASE}/llava_ov_clip_l22/ae.pt"

echo "============================================================"
echo " LLaVA-MORE Sweep — Qwen3-${MODEL_SIZE}"
echo " LLaVA-OV included: ${LLAVA_OV_READY}"
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

IMAGENET_OK=1; check_sae "${SAE_IMAGENET}" "imagenet" || IMAGENET_OK=0
CC3M_OK=1;     check_sae "${SAE_CC3M}" "cc3m_laion"  || CC3M_OK=0
LLAVOV_OK=0
if [[ "${LLAVA_OV_READY}" == "1" ]]; then
    check_sae "${SAE_LLAVA_OV}" "llava_ov" && LLAVOV_OK=1
fi

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

finetune_after() {
    local mode=$1 dataset=$2 dep=$3
    submit_dep \
        "finetune ${mode} ${dataset}" "${dep}" \
        "${SCRIPT_DIR}/finetune.sh" "${MODEL_SIZE}" "--sae-${mode}" "${dataset}"
}

# ============================================================
# Stage 1: 2 batch nodes cover all 7 pretrain configs.
#   Slot 0: baseline + imagenet ×2 + cc3m_laion-enconly  (4 runs)
#   Slot 1: cc3m_laion-encdec [+ llava_ov ×2]            (1–3 runs)
#
# Stage 2: each finetune waits for BOTH batch nodes to finish
# (its projector is guaranteed to exist once the relevant slot is done,
# but using afterok:ID0:ID1 is simpler than per-projector tracking).
# ============================================================
echo "--- Stage 1: submitting 2 pretrain-batch nodes ---"
PT_SLOT0=$(submit \
    "pretrain-batch slot0" \
    "${SCRIPT_DIR}/pretrain_batch.sh" "${MODEL_SIZE}" "0" "${LLAVA_OV_READY}")
PT_SLOT1=$(submit \
    "pretrain-batch slot1" \
    "${SCRIPT_DIR}/pretrain_batch.sh" "${MODEL_SIZE}" "1" "${LLAVA_OV_READY}")

# Both slots must finish before any finetune starts
PT_DEP="${PT_SLOT0}:${PT_SLOT1}"

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
