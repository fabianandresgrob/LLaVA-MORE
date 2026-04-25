#!/bin/bash
#SBATCH --job-name=finetune-qwen3
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --account=taco-vlm

# Stage 2: full instruction tuning (projector + LLM unfrozen, vision encoder frozen).
# Usage: sbatch finetune.sh <model_size> [--sae]
# model_size: 1.7B | 4B | 8B
# --sae flag: enable SAE encode-only bottleneck
# Examples:
#   sbatch finetune.sh 4B           # baseline
#   sbatch finetune.sh 4B --sae     # SAE encode-only variant

set -e

MODEL_SIZE=${1:-"4B"}
USE_SAE=false
if [[ "${2}" == "--sae" ]]; then
    USE_SAE=true
fi

VENV_PATH="$PROJECT/grob1/LLaVA/sc_venv_template"
REPO_PATH="$PROJECT/grob1/LLaVA-MORE"

source "${VENV_PATH}/activate.sh"
cd "${REPO_PATH}"

export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

mkdir -p "${REPO_PATH}/logs"

# ---- Paths ----
MODEL_BASE="$PROJECT/grob1/models/Qwen3-${MODEL_SIZE}"
VISION_TOWER="$PROJECT/grob1/models/clip-vit-large-patch14-336"
PROJECTOR_PATH="$SCRATCH/grob1/llava-more/checkpoints/qwen3-${MODEL_SIZE}-pretrain/mm_projector.bin"

DATA_PATH="$SCRATCH/grob1/llava-data/llava_v1_5_mix665k.json"
IMAGE_FOLDER="$SCRATCH/grob1/llava-data/images"

# SAE checkpoint: update this to the actual ae.pt path once training completes
# Expected location after sae training: $SCRATCH/grob1/sae/<dataset>_clip_l22/checkpoints/.../trainer_0/ae.pt
SAE_CHECKPOINT="$SCRATCH/grob1/sae/imagenet_clip_l22/checkpoints/imagenet_train_batch_top_k_20_x8/trainer_0/ae.pt"

if [[ "${USE_SAE}" == "true" ]]; then
    RUN_NAME="qwen3-${MODEL_SIZE}-sae-enc-only"
    SAE_ARGS="--use_sae_bottleneck True --sae_encode_only True --sae_checkpoint_path ${SAE_CHECKPOINT}"
else
    RUN_NAME="qwen3-${MODEL_SIZE}-baseline"
    SAE_ARGS=""
fi

OUTPUT_DIR="$SCRATCH/grob1/llava-more/checkpoints/${RUN_NAME}"
# ---------------

# Redirect SLURM logs now that we know the repo path
exec > "${REPO_PATH}/logs/${RUN_NAME}_${SLURM_JOB_ID}.out" \
     2>"${REPO_PATH}/logs/${RUN_NAME}_${SLURM_JOB_ID}.err"

export TOKENIZER_PATH="${MODEL_BASE}"

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
export OMP_NUM_THREADS=1

echo "=== Stage 2: Instruction tuning — ${RUN_NAME} ==="
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "SAE: ${USE_SAE}"
echo "Data:      ${DATA_PATH}"
echo "Projector: ${PROJECTOR_PATH}"
echo "Output:    ${OUTPUT_DIR}"

torchrun \
    --nnodes=1 --nproc-per-node=4 \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv-id="${SLURM_JOB_NAME}" \
    --rdzv-backend=c10d \
    src/llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path "${MODEL_BASE}" \
    --model_architecture qwen3 \
    --version qwen3 \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower "${VISION_TOWER}" \
    --pretrain_mm_mlp_adapter "${PROJECTOR_PATH}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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

echo "=== Done. Checkpoint at ${OUTPUT_DIR} ==="
