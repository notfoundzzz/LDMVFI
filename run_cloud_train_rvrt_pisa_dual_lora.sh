#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

unset LD_LIBRARY_PATH || true
unset CUDA_HOME || true
unset CUDA_PATH || true

DEFAULT_PYTHON_BIN="/data/Shenzhen/zhahongli/envs/ldmvfi/bin/python"
if [ -x "$DEFAULT_PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
PYTHON_DIR="$(dirname "$PYTHON_BIN")"
if [ -d "$PYTHON_DIR" ]; then
  export PATH="$PYTHON_DIR:$PATH"
fi

GPU_IDS="${GPU_IDS:-0,}"
CONFIG_PATH="${CONFIG_PATH:-configs/ldm/rvrt-pisa-dual-lora-stsr-x4.yaml}"
LOGDIR="${LOGDIR:-$ROOT_DIR/logs}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUM="${ACCUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
MAX_STEPS="${MAX_STEPS:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
VQ_CKPT="${VQ_CKPT:-/data/Shenzhen/zhahongli/models/ldmvfi/vqflow-extracted.ckpt}"
BASE_LDM_CKPT="${BASE_LDM_CKPT:-/data/Shenzhen/zhahongli/models/ldmvfi/ldmvfi-vqflow-f32-c256-concat_max.ckpt}"
RVRT_ROOT="${RVRT_ROOT:-/data/Shenzhen/zhahongli/RVRT}"
RVRT_CKPT="${RVRT_CKPT:-$RVRT_ROOT/model_zoo/rvrt/002_RVRT_videosr_bi_Vimeo_14frames.pth}"
SAVE_TRAIN_IMAGES="${SAVE_TRAIN_IMAGES:-0}"
MODEL_BASE_LR="${MODEL_BASE_LR:-}"
PIXEL_LORA_RANK="${PIXEL_LORA_RANK:-}"
PIXEL_LORA_ALPHA="${PIXEL_LORA_ALPHA:-}"
SEMANTIC_LORA_RANK="${SEMANTIC_LORA_RANK:-}"
SEMANTIC_LORA_ALPHA="${SEMANTIC_LORA_ALPHA:-}"
PIXEL_LORA_GROUPS="${PIXEL_LORA_GROUPS:-}"
SEMANTIC_LORA_GROUPS="${SEMANTIC_LORA_GROUPS:-}"
PIXEL_TARGET_SUFFIXES="${PIXEL_TARGET_SUFFIXES:-}"
SEMANTIC_TARGET_SUFFIXES="${SEMANTIC_TARGET_SUFFIXES:-}"
SEMANTIC_LR_SCALE="${SEMANTIC_LR_SCALE:-}"
USE_FLOW_GUIDANCE="${USE_FLOW_GUIDANCE:-}"
FLOW_GUIDANCE_STRENGTH="${FLOW_GUIDANCE_STRENGTH:-}"
FLOW_BACKEND="${FLOW_BACKEND:-}"
FLOW_RAFT_VARIANT="${FLOW_RAFT_VARIANT:-}"
FLOW_RAFT_CKPT="${FLOW_RAFT_CKPT:-}"
DUAL_LORA_TRAIN_MODE="${DUAL_LORA_TRAIN_MODE:-}"
SEMANTIC_START_STEP="${SEMANTIC_START_STEP:-}"
PIXEL_SCALE="${PIXEL_SCALE:-}"
SEMANTIC_SCALE="${SEMANTIC_SCALE:-}"

if [[ -z "$VQ_CKPT" || ! -f "$VQ_CKPT" ]]; then
  echo "VQ_CKPT is required"
  exit 1
fi
if [[ -n "$RESUME_CKPT" && ! -f "$RESUME_CKPT" ]]; then
  echo "RESUME_CKPT does not exist"
  exit 1
fi
if [[ -z "$RESUME_CKPT" && ( -z "$BASE_LDM_CKPT" || ! -f "$BASE_LDM_CKPT" ) ]]; then
  echo "BASE_LDM_CKPT is required"
  exit 1
fi

mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOGDIR/train_rvrt_pisa_dual_lora_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOGDIR/latest_train_rvrt_pisa_dual_lora.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
echo "path=$PATH"
echo "gpus=$GPU_IDS"
echo "config=$CONFIG_PATH"
echo "batch_size=$BATCH_SIZE"
echo "accum=$ACCUM"
echo "num_workers=$NUM_WORKERS"
echo "resume_ckpt=${RESUME_CKPT:-none}"
echo "vq_ckpt=$VQ_CKPT"
echo "base_ldm_ckpt=$BASE_LDM_CKPT"
echo "rvrt_root=$RVRT_ROOT"
echo "rvrt_ckpt=$RVRT_CKPT"
echo "pixel_lora_groups=${PIXEL_LORA_GROUPS:-default}"
echo "semantic_lora_groups=${SEMANTIC_LORA_GROUPS:-default}"
echo "pixel_target_suffixes=${PIXEL_TARGET_SUFFIXES:-default}"
echo "semantic_target_suffixes=${SEMANTIC_TARGET_SUFFIXES:-default}"
echo "semantic_lr_scale=${SEMANTIC_LR_SCALE:-default}"
echo "use_flow_guidance=${USE_FLOW_GUIDANCE:-default}"
echo "flow_guidance_strength=${FLOW_GUIDANCE_STRENGTH:-default}"
echo "flow_backend=${FLOW_BACKEND:-default}"
echo "flow_raft_variant=${FLOW_RAFT_VARIANT:-default}"
echo "flow_raft_ckpt=${FLOW_RAFT_CKPT:-default}"
echo "dual_lora_train_mode=${DUAL_LORA_TRAIN_MODE:-default}"

TRAINER_DOTLIST=()
if [[ -n "$MAX_EPOCHS" ]]; then
  TRAINER_DOTLIST+=("lightning.trainer.max_epochs=$MAX_EPOCHS")
fi
if [[ -n "$MAX_STEPS" ]]; then
  TRAINER_DOTLIST+=("lightning.trainer.max_steps=$MAX_STEPS")
fi
if [[ -n "$MODEL_BASE_LR" ]]; then
  TRAINER_DOTLIST+=("model.base_learning_rate=$MODEL_BASE_LR")
fi
if [[ -n "$PIXEL_LORA_RANK" ]]; then
  TRAINER_DOTLIST+=("model.params.pixel_lora_rank=$PIXEL_LORA_RANK")
fi
if [[ -n "$PIXEL_LORA_ALPHA" ]]; then
  TRAINER_DOTLIST+=("model.params.pixel_lora_alpha=$PIXEL_LORA_ALPHA")
fi
if [[ -n "$SEMANTIC_LORA_RANK" ]]; then
  TRAINER_DOTLIST+=("model.params.semantic_lora_rank=$SEMANTIC_LORA_RANK")
fi
if [[ -n "$SEMANTIC_LORA_ALPHA" ]]; then
  TRAINER_DOTLIST+=("model.params.semantic_lora_alpha=$SEMANTIC_LORA_ALPHA")
fi
if [[ -n "$PIXEL_LORA_GROUPS" ]]; then
  TRAINER_DOTLIST+=("model.params.pixel_lora_groups=$PIXEL_LORA_GROUPS")
fi
if [[ -n "$SEMANTIC_LORA_GROUPS" ]]; then
  TRAINER_DOTLIST+=("model.params.semantic_lora_groups=$SEMANTIC_LORA_GROUPS")
fi
if [[ -n "$PIXEL_TARGET_SUFFIXES" ]]; then
  TRAINER_DOTLIST+=("model.params.pixel_target_suffixes=$PIXEL_TARGET_SUFFIXES")
fi
if [[ -n "$SEMANTIC_TARGET_SUFFIXES" ]]; then
  TRAINER_DOTLIST+=("model.params.semantic_target_suffixes=$SEMANTIC_TARGET_SUFFIXES")
fi
if [[ -n "$SEMANTIC_LR_SCALE" ]]; then
  TRAINER_DOTLIST+=("model.params.semantic_lr_scale=$SEMANTIC_LR_SCALE")
fi
if [[ -n "$USE_FLOW_GUIDANCE" ]]; then
  TRAINER_DOTLIST+=("model.params.use_flow_guidance=$USE_FLOW_GUIDANCE")
fi
if [[ -n "$FLOW_GUIDANCE_STRENGTH" ]]; then
  TRAINER_DOTLIST+=("model.params.flow_guidance_strength=$FLOW_GUIDANCE_STRENGTH")
fi
if [[ -n "$FLOW_BACKEND" ]]; then
  TRAINER_DOTLIST+=("model.params.flow_backend=$FLOW_BACKEND")
fi
if [[ -n "$FLOW_RAFT_VARIANT" ]]; then
  TRAINER_DOTLIST+=("model.params.flow_raft_variant=$FLOW_RAFT_VARIANT")
fi
if [[ -n "$FLOW_RAFT_CKPT" ]]; then
  TRAINER_DOTLIST+=("model.params.flow_raft_ckpt=$FLOW_RAFT_CKPT")
fi
if [[ -n "$DUAL_LORA_TRAIN_MODE" ]]; then
  TRAINER_DOTLIST+=("model.params.dual_lora_train_mode=$DUAL_LORA_TRAIN_MODE")
fi
if [[ -n "$SEMANTIC_START_STEP" ]]; then
  TRAINER_DOTLIST+=("model.params.semantic_start_step=$SEMANTIC_START_STEP")
fi
if [[ -n "$PIXEL_SCALE" ]]; then
  TRAINER_DOTLIST+=("model.params.pixel_scale=$PIXEL_SCALE")
fi
if [[ -n "$SEMANTIC_SCALE" ]]; then
  TRAINER_DOTLIST+=("model.params.semantic_scale=$SEMANTIC_SCALE")
fi

IMAGE_LOGGER_DOTLIST=(
  "lightning.callbacks.image_logger.params.log_images_kwargs.plot_progressive_rows=False"
  "lightning.callbacks.image_logger.params.log_images_kwargs.plot_diffusion_rows=False"
  "lightning.callbacks.image_logger.params.log_images_kwargs.quantize_denoised=False"
)
if [[ "$SAVE_TRAIN_IMAGES" == "0" || "$SAVE_TRAIN_IMAGES" == "false" ]]; then
  IMAGE_LOGGER_DOTLIST+=("lightning.callbacks.image_logger.params.max_images=0")
fi

CMD=("$PYTHON_BIN" -u main.py)
if [[ -z "$RESUME_CKPT" ]]; then
  CMD+=(--base "$CONFIG_PATH")
fi

CMD+=(
  -t
  --no-test
  --gpus "$GPU_IDS"
  --logdir "$LOGDIR"
)

if [[ -n "$RESUME_CKPT" ]]; then
  CMD+=(-r "$RESUME_CKPT")
fi

CMD+=(
  data.params.batch_size="$BATCH_SIZE"
  data.params.num_workers="$NUM_WORKERS"
  lightning.trainer.accumulate_grad_batches="$ACCUM"
  "${TRAINER_DOTLIST[@]}"
  "${IMAGE_LOGGER_DOTLIST[@]}"
  model.params.first_stage_config.params.ckpt_path="$VQ_CKPT"
  model.params.rvrt_root="$RVRT_ROOT"
  model.params.rvrt_ckpt="$RVRT_CKPT"
)

if [[ -z "$RESUME_CKPT" ]]; then
  CMD+=(model.params.ckpt_path="$BASE_LDM_CKPT")
fi

"${CMD[@]}"
