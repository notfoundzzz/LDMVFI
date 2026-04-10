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
CONFIG_PATH="${CONFIG_PATH:-configs/ldm/rvrt-lora-stsr-x4.yaml}"
LOGDIR="${LOGDIR:-$ROOT_DIR/logs}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUM="${ACCUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
VQ_CKPT="${VQ_CKPT:-}"
RVRT_ROOT="${RVRT_ROOT:-/data/Shenzhen/zhahongli/RVRT}"
RVRT_CKPT="${RVRT_CKPT:-$RVRT_ROOT/model_zoo/rvrt/002_RVRT_videosr_bi_Vimeo_14frames.pth}"

if [[ -z "$VQ_CKPT" ]]; then
  echo "VQ_CKPT is required"
  exit 1
fi

mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOGDIR/train_rvrt_lora_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOGDIR/latest_train_rvrt_lora.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
echo "path=$PATH"
echo "gpus=$GPU_IDS"
echo "config=$CONFIG_PATH"
echo "batch_size=$BATCH_SIZE"
echo "accum=$ACCUM"
echo "num_workers=$NUM_WORKERS"
echo "max_epochs=${MAX_EPOCHS:-default}"
echo "vq_ckpt=$VQ_CKPT"
echo "rvrt_root=$RVRT_ROOT"
echo "rvrt_ckpt=$RVRT_CKPT"

TRAINER_DOTLIST=()
if [[ -n "$MAX_EPOCHS" ]]; then
  TRAINER_DOTLIST+=("lightning.trainer.max_epochs=$MAX_EPOCHS")
fi

"$PYTHON_BIN" -u main.py \
  --base "$CONFIG_PATH" \
  -t \
  --gpus "$GPU_IDS" \
  --logdir "$LOGDIR" \
  data.params.batch_size="$BATCH_SIZE" \
  data.params.num_workers="$NUM_WORKERS" \
  lightning.trainer.accumulate_grad_batches="$ACCUM" \
  "${TRAINER_DOTLIST[@]}" \
  model.params.first_stage_config.params.ckpt_path="$VQ_CKPT" \
  model.params.rvrt_root="$RVRT_ROOT" \
  model.params.rvrt_ckpt="$RVRT_CKPT"
