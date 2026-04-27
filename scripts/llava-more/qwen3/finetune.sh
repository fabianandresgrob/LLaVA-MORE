#!/bin/bash
#SBATCH --job-name=finetune-qwen3
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --partition=booster
#SBATCH --account=taco-vlm

# Stage 2: full instruction tuning (projector + LLM unfrozen, vision encoder frozen).
# Usage: sbatch finetune.sh <model_size> [MODE [DATASET]]
# model_size:  1.7B | 4B | 8B
# MODE:
#   (none)                  baseline, no SAE
#   --sae-enconly DATASET   SAE encode-only  (8192d projector from pretrain-enconly)
#   --sae-encdec  DATASET   SAE encode+dec   (1024d projector from pretrain-baseline)
#   --sae                   legacy: enc-only with imagenet SAE (backward compat)
# DATASET: imagenet | cc3m_laion | llava_ov
# Examples:
#   sbatch finetune.sh 4B
#   sbatch finetune.sh 4B --sae-enconly imagenet
#   sbatch finetune.sh 4B --sae-encdec  cc3m_laion

set -e

MODEL_SIZE=${1:-"4B"}
SAE_MODE=""    # enconly | encdec | ""
SAE_DATASET="" # imagenet | cc3m_laion | llava_ov

case "${2}" in
    --sae-enconly) SAE_MODE="enconly"; SAE_DATASET="${3}" ;;
    --sae-encdec)  SAE_MODE="encdec";  SAE_DATASET="${3}" ;;
    --sae)         SAE_MODE="enconly"; SAE_DATASET="imagenet" ;; # legacy
esac

VENV_PATH="$PROJECT/grob1/LLaVA/sc_venv_template"
REPO_PATH="$PROJECT/grob1/LLaVA-MORE"

source "${VENV_PATH}/activate.sh"
cd "${REPO_PATH}"

export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export PATH="${CUDA_HOME}/bin:${PATH}"
echo "CUDA_HOME=${CUDA_HOME}"
export PYTHONPATH=.
export HF_HOME=/e/scratch/taco-vlm/grob1/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

mkdir -p "${REPO_PATH}/logs"

# ---- Paths ----
MODEL_BASE="Qwen/Qwen3-${MODEL_SIZE}"
VISION_TOWER="openai/clip-vit-large-patch14-336"

DATA_PATH="$SCRATCH/grob1/llava_data/llava_v1_5_mix665k.json"
IMAGE_FOLDER="$SCRATCH/grob1/llava_data"

SAE_BASE="$SCRATCH/grob1/sae"

# ---- Resolve run name, projector path, and SAE args ----
case "${SAE_MODE}" in
    enconly)
        RUN_NAME="qwen3-${MODEL_SIZE}-sae-${SAE_DATASET}-enconly"
        PROJECTOR_PATH="$SCRATCH/grob1/llava-more/checkpoints/qwen3-${MODEL_SIZE}-pretrain-${SAE_DATASET}-enconly/mm_projector.bin"
        SAE_ARGS="--use_sae_bottleneck True --sae_encode_only True \
            --sae_checkpoint_path ${SAE_BASE}/${SAE_DATASET}_clip_l22/ae.pt"
        ;;
    encdec)
        RUN_NAME="qwen3-${MODEL_SIZE}-sae-${SAE_DATASET}-encdec"
        PROJECTOR_PATH="$SCRATCH/grob1/llava-more/checkpoints/qwen3-${MODEL_SIZE}-pretrain-${SAE_DATASET}-encdec/mm_projector.bin"
        SAE_ARGS="--use_sae_bottleneck True --sae_encode_only False \
            --sae_checkpoint_path ${SAE_BASE}/${SAE_DATASET}_clip_l22/ae.pt"
        ;;
    *)
        RUN_NAME="qwen3-${MODEL_SIZE}-baseline"
        PROJECTOR_PATH="$SCRATCH/grob1/llava-more/checkpoints/qwen3-${MODEL_SIZE}-pretrain/mm_projector.bin"
        SAE_ARGS=""
        ;;
esac

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
echo "SAE mode: ${SAE_MODE:-none}  dataset: ${SAE_DATASET:-n/a}"
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
    --eval_strategy no \
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
