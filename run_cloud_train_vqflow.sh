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
GPU_IDS="${GPU_IDS:-0,}"
DATA_ROOT="${DATA_ROOT:-/data/Shenzhen/zhahongli/datasets/ldmvfi}"
LOGDIR="${LOGDIR:-$ROOT_DIR/logs}"
BATCH_SIZE="${BATCH_SIZE:-10}"
ACCUM="${ACCUM:-1}"

mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOGDIR/train_vqflow_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOGDIR/latest_train_vqflow.log"

echo "root=$ROOT_DIR" | tee -a "$LOG_FILE"
echo "python=$PYTHON_BIN" | tee -a "$LOG_FILE"
echo "gpus=$GPU_IDS" | tee -a "$LOG_FILE"
echo "data_root=$DATA_ROOT" | tee -a "$LOG_FILE"
echo "batch_size=$BATCH_SIZE" | tee -a "$LOG_FILE"
echo "accum=$ACCUM" | tee -a "$LOG_FILE"

"$PYTHON_BIN" -u main.py \
  --base configs/autoencoder/vqflow-f32.yaml \
  -t \
  --gpus "$GPU_IDS" \
  --logdir "$LOGDIR" \
  --data.params.batch_size "$BATCH_SIZE" \
  --lightning.trainer.accumulate_grad_batches "$ACCUM" \
  --data.params.train.params.db_dir "$DATA_ROOT" \
  --data.params.validation.params.db_dir "$DATA_ROOT/vimeo_septuplet" 2>&1 | tee -a "$LOG_FILE"
