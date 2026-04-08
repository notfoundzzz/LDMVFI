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
DATA_ROOT="${DATA_ROOT:-/data/Shenzhen/zhahongli/benchmarks}"
CKPT_PATH="${CKPT_PATH:-/data/Shenzhen/zhahongli/models/ldmvfi/ldmvfi-vqflow-f32-c256-concat_max.ckpt}"
CONFIG_PATH="${CONFIG_PATH:-configs/ldm/stsr-x2-resizecond.yaml}"
DATASET="${DATASET:-Ucf_STSR}"
SCALE_FACTOR="${SCALE_FACTOR:-2}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/eval_results_stsr}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs}"

mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/eval_stsr_${DATASET}_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_ROOT/latest_eval_stsr.log"

echo "root=$ROOT_DIR" | tee -a "$LOG_FILE"
echo "python=$PYTHON_BIN" | tee -a "$LOG_FILE"
echo "config=$CONFIG_PATH" | tee -a "$LOG_FILE"
echo "data_root=$DATA_ROOT" | tee -a "$LOG_FILE"
echo "ckpt=$CKPT_PATH" | tee -a "$LOG_FILE"
echo "dataset=$DATASET" | tee -a "$LOG_FILE"
echo "scale_factor=$SCALE_FACTOR" | tee -a "$LOG_FILE"

"$PYTHON_BIN" -u evaluate_stsr.py \
  --config "$CONFIG_PATH" \
  --ckpt "$CKPT_PATH" \
  --dataset "$DATASET" \
  --data_dir "$DATA_ROOT" \
  --out_dir "$OUT_ROOT" \
  --scale_factor "$SCALE_FACTOR" 2>&1 | tee -a "$LOG_FILE"
