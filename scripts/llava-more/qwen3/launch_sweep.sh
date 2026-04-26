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

pair() {
    local mode=$1 dataset=$2
    local pt_id ft_id
    pt_id=$(submit \
        "pretrain ${mode} ${dataset}" \
        "${SCRIPT_DIR}/pretrain.sh" "${MODEL_SIZE}" "--sae-${mode}" "${dataset}")
    ft_id=$(submit_dep \
        "finetune ${mode} ${dataset}" "${pt_id}" \
        "${SCRIPT_DIR}/finetune.sh" "${MODEL_SIZE}" "--sae-${mode}" "${dataset}")
}

# ============================================================
# Submit all pairs (pretrains run in parallel immediately;
# finetunes queue behind their own pretrain)
# ============================================================
echo "--- Baseline ---"
PT_BASE=$(submit \
    "pretrain baseline" \
    "${SCRIPT_DIR}/pretrain.sh" "${MODEL_SIZE}")
submit_dep \
    "finetune baseline" "${PT_BASE}" \
    "${SCRIPT_DIR}/finetune.sh" "${MODEL_SIZE}"

echo ""
echo "--- ImageNet SAE ---"
if [[ "${IMAGENET_OK}" == "1" ]]; then
    pair enconly imagenet
    pair encdec  imagenet
else
    echo "  [SKIP] imagenet SAE not found"
fi

echo ""
echo "--- CC3M+LAION SAE ---"
if [[ "${CC3M_OK}" == "1" ]]; then
    pair enconly cc3m_laion
    pair encdec  cc3m_laion
else
    echo "  [SKIP] cc3m_laion SAE not found"
fi

echo ""
echo "--- LLaVA-OV SAE ---"
if [[ "${LLAVOV_OK}" == "1" ]]; then
    pair enconly llava_ov
    pair encdec  llava_ov
else
    echo "  [SKIP] llava_ov SAE not ready (rerun with LLAVA_OV_READY=1)"
fi

echo ""
echo "============================================================"
echo " All jobs submitted. Monitor with: squeue -u \$USER"
echo " Checkpoints: \$SCRATCH/grob1/llava-more/checkpoints/"
echo "============================================================"
