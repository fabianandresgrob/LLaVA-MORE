#!/bin/bash
#SBATCH --job-name=pretrain-qwen3
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --partition=booster
#SBATCH --account=taco-vlm

# Stage 1: train the MLP projector only (vision encoder + LLM frozen).
# Each SAE variant needs its own projector — different SAEs produce different
# feature distributions even at the same dimensionality.
#
# Usage: sbatch pretrain.sh <model_size> [MODE DATASET]
# model_size:  1.7B | 4B | 8B
# MODE:        --sae-enconly | --sae-encdec
# DATASET:     imagenet | cc3m_laion | llava_ov
#
# Examples:
#   sbatch pretrain.sh 4B                           # baseline (1024d projector)
#   sbatch pretrain.sh 4B --sae-enconly imagenet    # enc-only imagenet SAE (8192d)
#   sbatch pretrain.sh 4B --sae-encdec  cc3m_laion  # enc+dec cc3m SAE (1024d)

set -e

MODEL_SIZE=${1:-"4B"}
SAE_MODE=""
SAE_DATASET=""

case "${2}" in
    --sae-enconly) SAE_MODE="enconly"; SAE_DATASET="${3}" ;;
    --sae-encdec)  SAE_MODE="encdec";  SAE_DATASET="${3}" ;;
esac

VENV_PATH="$PROJECT/grob1/LLaVA/sc_venv_template"
REPO_PATH="$PROJECT/grob1/LLaVA-MORE"

# ---- Resolve run name and SAE args before redirecting logs ----
SAE_BASE="$SCRATCH/grob1/sae"

case "${SAE_MODE}" in
    enconly)
        RUN_NAME="qwen3-${MODEL_SIZE}-pretrain-${SAE_DATASET}-enconly"
        SAE_ARGS="--use_sae_bottleneck True --sae_encode_only True \
            --sae_checkpoint_path ${SAE_BASE}/${SAE_DATASET}_clip_l22/ae.pt"
        ;;
    encdec)
        RUN_NAME="qwen3-${MODEL_SIZE}-pretrain-${SAE_DATASET}-encdec"
        SAE_ARGS="--use_sae_bottleneck True --sae_encode_only False \
            --sae_checkpoint_path ${SAE_BASE}/${SAE_DATASET}_clip_l22/ae.pt"
        ;;
    *)
        RUN_NAME="qwen3-${MODEL_SIZE}-pretrain"
        SAE_ARGS=""
        ;;
esac

OUTPUT_DIR="$SCRATCH/grob1/llava-more/checkpoints/${RUN_NAME}"

source "${VENV_PATH}/activate.sh"
cd "${REPO_PATH}"

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

mkdir -p "${REPO_PATH}/logs"
exec > "${REPO_PATH}/logs/${RUN_NAME}_${SLURM_JOB_ID}.out" \
     2>"${REPO_PATH}/logs/${RUN_NAME}_${SLURM_JOB_ID}.err"

# ---- Paths ----
MODEL_BASE="$PROJECT/grob1/models/Qwen3-${MODEL_SIZE}"
VISION_TOWER="$PROJECT/grob1/models/clip-vit-large-patch14-336"
DATA_PATH="$SCRATCH/grob1/llava-data/LLaVA-CC3M-Pretrain-595K/chat.json"
IMAGE_FOLDER="$SCRATCH/grob1/llava-data/LLaVA-CC3M-Pretrain-595K/images"
# ---------------

export TOKENIZER_PATH="${MODEL_BASE}"

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
export OMP_NUM_THREADS=1

echo "=== Stage 1: Projector pretrain — ${RUN_NAME} ==="
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "SAE mode:    ${SAE_MODE:-none}  dataset: ${SAE_DATASET:-n/a}"
echo "Data:        ${DATA_PATH}"
echo "Output:      ${OUTPUT_DIR}"

torchrun \
    --nnodes=1 --nproc-per-node=4 \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv-id="${SLURM_JOB_NAME}" \
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
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    ${SAE_ARGS}

echo "=== Done. Projector at ${OUTPUT_DIR}/mm_projector.bin ==="
