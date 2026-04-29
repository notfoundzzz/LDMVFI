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

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29592}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/cache/even_corrector_spynet_ddim200}"
TRAIN_SPLITS="${TRAIN_SPLITS:-train}"
VAL_SPLITS="${VAL_SPLITS:-slow_test,medium_test,fast_test}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/experiments/even_residual_corrector_cached}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-32}"
NUM_BLOCKS="${NUM_BLOCKS:-4}"
MAX_RESIDUE="${MAX_RESIDUE:-0.25}"
USE_FLOW_INPUTS="${USE_FLOW_INPUTS:-0}"
CORRECTOR_MODE="${CORRECTOR_MODE:-residual}"
FUSION_INIT_PRED_LOGIT="${FUSION_INIT_PRED_LOGIT:-8.0}"
EDGE_WEIGHT="${EDGE_WEIGHT:-0}"
SSIM_WEIGHT="${SSIM_WEIGHT:-0}"
SSIM_WINDOW="${SSIM_WINDOW:-11}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-1000}"
VAL_INTERVAL="${VAL_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
METRICS="${METRICS:-PSNR,SSIM,LPIPS}"
SEED="${SEED:-1234}"

mkdir -p "$LOG_ROOT" "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/train_even_corrector_cached_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_ROOT/latest_train_even_corrector_cached.log"
exec > >(tee -a "$LOG_FILE") 2>&1

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU"
  exit 1
fi

echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
echo "gpus=$GPU_IDS"
echo "num_gpus=$NUM_GPUS"
echo "master_port=$MASTER_PORT"
echo "cache_root=$CACHE_ROOT"
echo "train_splits=$TRAIN_SPLITS"
echo "val_splits=$VAL_SPLITS"
echo "max_train_samples=$MAX_TRAIN_SAMPLES"
echo "max_val_samples=$MAX_VAL_SAMPLES"
echo "out_dir=$OUT_DIR"
echo "hidden_channels=$HIDDEN_CHANNELS"
echo "num_blocks=$NUM_BLOCKS"
echo "max_residue=$MAX_RESIDUE"
echo "use_flow_inputs=$USE_FLOW_INPUTS"
echo "corrector_mode=$CORRECTOR_MODE"
echo "fusion_init_pred_logit=$FUSION_INIT_PRED_LOGIT"
echo "edge_weight=$EDGE_WEIGHT"
echo "ssim_weight=$SSIM_WEIGHT"
echo "ssim_window=$SSIM_WINDOW"
echo "lr=$LR"
echo "batch_size=$BATCH_SIZE"
echo "max_steps=$MAX_STEPS"
echo "val_interval=$VAL_INTERVAL"
echo "metrics=$METRICS"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

CMD=(
  "$PYTHON_BIN" -m torch.distributed.launch
  --nproc_per_node="$NUM_GPUS"
  --master_port="$MASTER_PORT"
  train_even_residual_corrector_cached.py
  --cache_root "$CACHE_ROOT"
  --train_splits "$TRAIN_SPLITS"
  --val_splits "$VAL_SPLITS"
  --max_train_samples "$MAX_TRAIN_SAMPLES"
  --max_val_samples "$MAX_VAL_SAMPLES"
  --out_dir "$OUT_DIR"
  --hidden_channels "$HIDDEN_CHANNELS"
  --num_blocks "$NUM_BLOCKS"
  --max_residue "$MAX_RESIDUE"
  --use_flow_inputs "$USE_FLOW_INPUTS"
  --corrector_mode "$CORRECTOR_MODE"
  --fusion_init_pred_logit "$FUSION_INIT_PRED_LOGIT"
  --edge_weight "$EDGE_WEIGHT"
  --ssim_weight "$SSIM_WEIGHT"
  --ssim_window "$SSIM_WINDOW"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --max_steps "$MAX_STEPS"
  --val_interval "$VAL_INTERVAL"
  --save_interval "$SAVE_INTERVAL"
  --metrics ${METRICS//,/ }
  --seed "$SEED"
)

echo "==== command ===="
printf '%q ' "${CMD[@]}"
echo
echo "==== start cached training ===="
"${CMD[@]}"
