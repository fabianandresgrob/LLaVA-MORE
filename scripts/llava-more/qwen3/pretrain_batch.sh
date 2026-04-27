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

# Run up to 4 Stage-1 projector pretrains in parallel, one per GPU.
# Each pretrain is single-GPU (nproc=1) — the projector is tiny and the
# frozen LLM fits easily on one H200.
#
# Two instances (SLOT=0 and SLOT=1) cover all 7 configurations:
#   Slot 0: baseline + imagenet-enconly + imagenet-encdec + cc3m_laion-enconly
#   Slot 1: cc3m_laion-encdec [+ llava_ov-enconly + llava_ov-encdec]
#
# Usage: sbatch pretrain_batch.sh <MODEL_SIZE> <SLOT> [LLAVA_OV_READY]

set -e

MODEL_SIZE=${1:-"4B"}
SLOT=${2:-0}
LLAVA_OV_READY=${3:-0}

VENV_PATH="$PROJECT/grob1/LLaVA/sc_venv_template"
REPO_PATH="$PROJECT/grob1/LLaVA-MORE"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAE_BASE="$SCRATCH/grob1/sae"

source "${VENV_PATH}/activate.sh"

export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export PATH="${CUDA_HOME}/bin:${PATH}"
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

# Build list of (args...) for this slot
declare -a CONFIGS
if [[ "${SLOT}" == "0" ]]; then
    CONFIGS+=("")                                                           # baseline
    [[ -f "${SAE_BASE}/imagenet_clip_l22/ae.pt"   ]] && CONFIGS+=("--sae-enconly imagenet")
    [[ -f "${SAE_BASE}/imagenet_clip_l22/ae.pt"   ]] && CONFIGS+=("--sae-encdec  imagenet")
    [[ -f "${SAE_BASE}/cc3m_laion_clip_l22/ae.pt" ]] && CONFIGS+=("--sae-enconly cc3m_laion")
else
    [[ -f "${SAE_BASE}/cc3m_laion_clip_l22/ae.pt" ]] && CONFIGS+=("--sae-encdec  cc3m_laion")
    if [[ "${LLAVA_OV_READY}" == "1" && -f "${SAE_BASE}/llava_ov_clip_l22/ae.pt" ]]; then
        CONFIGS+=("--sae-enconly llava_ov")
        CONFIGS+=("--sae-encdec  llava_ov")
    fi
fi

echo "Configs to run (${#CONFIGS[@]}):"
for c in "${CONFIGS[@]}"; do echo "  '${c}'"; done
echo ""

# Launch one pretrain per GPU in parallel
# Each gets: CUDA_VISIBLE_DEVICES=N, PRETRAIN_NPROC=1, PRETRAIN_PORT=51NN
pids=()
gpu=0
for args in "${CONFIGS[@]}"; do
    port=$((5100 + gpu * 100))
    echo "--- GPU ${gpu} port ${port}: pretrain.sh ${MODEL_SIZE} ${args} ---"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${gpu} PRETRAIN_NPROC=1 PRETRAIN_PORT=${port} \
        bash "${SCRIPT_DIR}/pretrain.sh" "${MODEL_SIZE}" ${args} &
    pids+=($!)
    gpu=$((gpu + 1))
done

echo ""
echo "Waiting for ${#pids[@]} parallel pretrains (PIDs: ${pids[*]})..."

# Collect exit codes — fail the batch job if any pretrain failed
failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "  [OK]   config ${i}: '${CONFIGS[$i]}'"
    else
        echo "  [FAIL] config ${i}: '${CONFIGS[$i]}'"
        failed=1
    fi
done

[[ $failed -eq 1 ]] && { echo "One or more pretrains failed."; exit 1; }
echo ""
echo "=== pretrain_batch SLOT=${SLOT} complete ==="
