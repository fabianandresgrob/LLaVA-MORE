#!/bin/bash
#SBATCH --job-name=test-qwen3
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --partition=booster
#SBATCH --account=taco-vlm

# Minimal end-to-end smoke test — 20 steps, no checkpoint saved.
# Tests that the full torchrun/deepspeed/model/data stack runs without errors.
# Run this before launching the full sweep.
#
# Usage: sbatch test.sh [MODEL_SIZE] [--sae-enconly|--sae-encdec [DATASET]]
# Examples:
#   sbatch test.sh 1.7B                              # baseline (fastest)
#   sbatch test.sh 1.7B --sae-enconly imagenet       # enc-only with imagenet SAE

set -e

MODEL_SIZE=${1:-"1.7B"}
SAE_MODE=""
SAE_DATASET=""

case "${2}" in
    --sae-enconly) SAE_MODE="enconly"; SAE_DATASET="${3:-imagenet}" ;;
    --sae-encdec)  SAE_MODE="encdec";  SAE_DATASET="${3:-imagenet}" ;;
esac

VENV_PATH="$PROJECT/grob1/LLaVA/sc_venv_template"
REPO_PATH="$PROJECT/grob1/LLaVA-MORE"

source "${VENV_PATH}/activate.sh"
cd "${REPO_PATH}"

# DeepSpeed probes CUDA_HOME at import time. Ask PyTorch where its CUDA lives —
# it already resolved this correctly when the venv was built.
export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export PATH="${CUDA_HOME}/bin:${PATH}"
echo "CUDA_HOME=${CUDA_HOME}"
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled

mkdir -p "${REPO_PATH}/logs"
exec > "${REPO_PATH}/logs/test-qwen3-${MODEL_SIZE}_${SLURM_JOB_ID}.out" \
     2>"${REPO_PATH}/logs/test-qwen3-${MODEL_SIZE}_${SLURM_JOB_ID}.err"

# ---- Paths ----
MODEL_BASE="$PROJECT/grob1/models/Qwen3-${MODEL_SIZE}"
VISION_TOWER="$PROJECT/grob1/models/clip-vit-large-patch14-336"
DATA_PATH="$SCRATCH/grob1/llava-data/LLaVA-CC3M-Pretrain-595K/chat.json"
IMAGE_FOLDER="$SCRATCH/grob1/llava-data/LLaVA-CC3M-Pretrain-595K/images"
SAE_BASE="$SCRATCH/grob1/sae"

case "${SAE_MODE}" in
    enconly)
        RUN_TAG="test-sae-${SAE_DATASET}-enconly"
        SAE_ARGS="--use_sae_bottleneck True --sae_encode_only True \
            --sae_checkpoint_path ${SAE_BASE}/${SAE_DATASET}_clip_l22/ae.pt"
        ;;
    encdec)
        RUN_TAG="test-sae-${SAE_DATASET}-encdec"
        SAE_ARGS="--use_sae_bottleneck True --sae_encode_only False \
            --sae_checkpoint_path ${SAE_BASE}/${SAE_DATASET}_clip_l22/ae.pt"
        ;;
    *)
        RUN_TAG="test-baseline"
        SAE_ARGS=""
        ;;
esac

OUTPUT_DIR="/tmp/llava-more-test-${SLURM_JOB_ID}"

export TOKENIZER_PATH="${MODEL_BASE}"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
export OMP_NUM_THREADS=1

echo "=== Smoke test — Qwen3-${MODEL_SIZE} ${RUN_TAG} ==="
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

torchrun \
    --nnodes=1 --nproc-per-node=4 \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv-id="test-qwen3" \
    --rdzv-backend=c10d \
    src/llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${MODEL_BASE}" \
    --model_architecture qwen3 \
    --version plain \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy no \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none \
    ${SAE_ARGS}

echo "=== Smoke test PASSED ==="
rm -rf "${OUTPUT_DIR}"
