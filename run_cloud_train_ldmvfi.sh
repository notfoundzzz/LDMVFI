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
DATA_ROOT="${DATA_ROOT:-/data/Shenzhen/zzff/STVSR/data}"
LOGDIR="${LOGDIR:-$ROOT_DIR/logs}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ACCUM="${ACCUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
VQ_CKPT="${VQ_CKPT:-}"
USE_BVIDVC="${USE_BVIDVC:-auto}"

if [[ -z "$VQ_CKPT" ]]; then
  echo "VQ_CKPT is required"
  exit 1
fi

mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOGDIR/train_ldmvfi_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOGDIR/latest_train_ldmvfi.log"

echo "root=$ROOT_DIR" | tee -a "$LOG_FILE"
echo "python=$PYTHON_BIN" | tee -a "$LOG_FILE"
echo "gpus=$GPU_IDS" | tee -a "$LOG_FILE"
echo "data_root=$DATA_ROOT" | tee -a "$LOG_FILE"
echo "batch_size=$BATCH_SIZE" | tee -a "$LOG_FILE"
echo "accum=$ACCUM" | tee -a "$LOG_FILE"
echo "num_workers=$NUM_WORKERS" | tee -a "$LOG_FILE"
echo "max_epochs=${MAX_EPOCHS:-default}" | tee -a "$LOG_FILE"
echo "vq_ckpt=$VQ_CKPT" | tee -a "$LOG_FILE"
echo "use_bvidvc=$USE_BVIDVC" | tee -a "$LOG_FILE"

TRAIN_DOTLIST=(
  "data.params.train.params.db_dir=$DATA_ROOT"
)

if [[ "$USE_BVIDVC" == "0" || "$USE_BVIDVC" == "false" ]]; then
  TRAIN_DOTLIST=(
    "data.params.train.target=ldm.data.bvi_vimeo.Vimeo90k_triplet"
    "data.params.train.params.db_dir=$DATA_ROOT/vimeo_septuplet"
    "data.params.train.params.train=True"
    "data.params.train.params.crop_sz=[256,256]"
    "data.params.train.params.augment_s=True"
    "data.params.train.params.augment_t=True"
  )
elif [[ "$USE_BVIDVC" == "auto" && ! -d "$DATA_ROOT/bvidvc/quintuplets" ]]; then
  echo "bvidvc/quintuplets not found, falling back to Vimeo-only training" | tee -a "$LOG_FILE"
  TRAIN_DOTLIST=(
    "data.params.train.target=ldm.data.bvi_vimeo.Vimeo90k_triplet"
    "data.params.train.params.db_dir=$DATA_ROOT/vimeo_septuplet"
    "data.params.train.params.train=True"
    "data.params.train.params.crop_sz=[256,256]"
    "data.params.train.params.augment_s=True"
    "data.params.train.params.augment_t=True"
  )
fi

TRAINER_DOTLIST=()
if [[ -n "$MAX_EPOCHS" ]]; then
  TRAINER_DOTLIST+=("lightning.trainer.max_epochs=$MAX_EPOCHS")
fi

"$PYTHON_BIN" -u main.py \
  --base configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml \
  -t \
  --gpus "$GPU_IDS" \
  --logdir "$LOGDIR" \
  data.params.batch_size="$BATCH_SIZE" \
  data.params.num_workers="$NUM_WORKERS" \
  lightning.trainer.accumulate_grad_batches="$ACCUM" \
  "${TRAINER_DOTLIST[@]}" \
  "${TRAIN_DOTLIST[@]}" \
  data.params.validation.params.db_dir="$DATA_ROOT/vimeo_septuplet" \
  model.params.first_stage_config.params.ckpt_path="$VQ_CKPT" 2>&1 | tee -a "$LOG_FILE"
