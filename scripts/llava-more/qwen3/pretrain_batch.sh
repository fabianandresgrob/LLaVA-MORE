#!/bin/bash
#SBATCH --job-name=pretrain-batch
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --partition=booster
#SBATCH --account=taco-vlm

# Run multiple Stage-1 projector pretrains sequentially on one node.
# Each pretrain uses all 4 GPUs.  Two instances of this script (SLOT=0 and
# SLOT=1) together cover all 7 configurations for one model size, using only
# 2 nodes instead of 7.
#
# Usage: sbatch pretrain_batch.sh <MODEL_SIZE> <SLOT> [LLAVA_OV_READY]
#   MODEL_SIZE:    1.7B | 4B | 8B | 14B | 0.6B
#   SLOT:          0  → baseline + imagenet-enconly + imagenet-encdec + cc3m_laion-enconly
#                  1  → cc3m_laion-encdec [+ llava_ov-enconly + llava_ov-encdec]
#   LLAVA_OV_READY: 0 | 1 (default 0)
#
# Example (4B, with llava_ov):
#   ID0=$(sbatch pretrain_batch.sh 4B 0 1 | awk '{print $NF}')
#   ID1=$(sbatch pretrain_batch.sh 4B 1 1 | awk '{print $NF}')
#   # then submit finetunes with --dependency=afterok:$ID0:$ID1

set -e

MODEL_SIZE=${1:-"4B"}
SLOT=${2:-0}
LLAVA_OV_READY=${3:-0}

VENV_PATH="$PROJECT/grob1/LLaVA/sc_venv_template"
REPO_PATH="$PROJECT/grob1/LLaVA-MORE"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${VENV_PATH}/activate.sh"
cd "${REPO_PATH}"

export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH=.
export HF_HOME=/e/scratch/taco-vlm/grob1/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

mkdir -p "${REPO_PATH}/logs"
exec > "${REPO_PATH}/logs/pretrain-batch-${MODEL_SIZE}-slot${SLOT}_${SLURM_JOB_ID}.out" \
     2>"${REPO_PATH}/logs/pretrain-batch-${MODEL_SIZE}-slot${SLOT}_${SLURM_JOB_ID}.err"

echo "=== pretrain_batch MODEL_SIZE=${MODEL_SIZE} SLOT=${SLOT} LLAVA_OV_READY=${LLAVA_OV_READY} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_pretrain() {
    local args=("$@")
    echo ""
    echo "--- Starting: pretrain.sh ${MODEL_SIZE} ${args[*]} ---"
    bash "${SCRIPT_DIR}/pretrain.sh" "${MODEL_SIZE}" "${args[@]}"
    echo "--- Done: pretrain.sh ${MODEL_SIZE} ${args[*]} ---"
}

SAE_BASE="$SCRATCH/grob1/sae"

if [[ "${SLOT}" == "0" ]]; then
    # Slot 0: baseline + imagenet (both modes) + cc3m_laion enconly
    run_pretrain
    [[ -f "${SAE_BASE}/imagenet_clip_l22/ae.pt" ]] && run_pretrain --sae-enconly imagenet
    [[ -f "${SAE_BASE}/imagenet_clip_l22/ae.pt" ]] && run_pretrain --sae-encdec  imagenet
    [[ -f "${SAE_BASE}/cc3m_laion_clip_l22/ae.pt" ]] && run_pretrain --sae-enconly cc3m_laion
else
    # Slot 1: cc3m_laion encdec + llava_ov (if ready)
    [[ -f "${SAE_BASE}/cc3m_laion_clip_l22/ae.pt" ]] && run_pretrain --sae-encdec cc3m_laion
    if [[ "${LLAVA_OV_READY}" == "1" && -f "${SAE_BASE}/llava_ov_clip_l22/ae.pt" ]]; then
        run_pretrain --sae-enconly llava_ov
        run_pretrain --sae-encdec  llava_ov
    fi
fi

echo ""
echo "=== pretrain_batch SLOT=${SLOT} complete ==="
