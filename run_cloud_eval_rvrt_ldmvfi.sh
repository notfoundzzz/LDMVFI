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
LDM_CONFIG="${LDM_CONFIG:-$ROOT_DIR/configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml}"
LDM_CKPT="${LDM_CKPT:-/data/Shenzhen/zhahongli/models/ldmvfi/ldmvfi-vqflow-f32-c256-concat_max.ckpt}"
DATASET_ROOT="${DATASET_ROOT:-/data/Shenzhen/zhahongli/benchmarks/ucf}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/eval_results_rvrt_ldmvfi}"
SCALE="${SCALE:-4}"
SR_MODE="${SR_MODE:-bicubic}"
RVRT_ROOT="${RVRT_ROOT:-/data/Shenzhen/zhahongli/RVRT}"
RVRT_TASK="${RVRT_TASK:-002_RVRT_videosr_bi_Vimeo_14frames}"
RVRT_CKPT="${RVRT_CKPT:-$RVRT_ROOT/model_zoo/rvrt/${RVRT_TASK}.pth}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs}"

mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/eval_rvrt_ldmvfi_${SR_MODE}_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_ROOT/latest_eval_rvrt_ldmvfi.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
echo "ldm_config=$LDM_CONFIG"
echo "ldm_ckpt=$LDM_CKPT"
echo "dataset_root=$DATASET_ROOT"
echo "out_dir=$OUT_DIR"
echo "scale=$SCALE"
echo "sr_mode=$SR_MODE"
echo "rvrt_root=$RVRT_ROOT"
echo "rvrt_task=$RVRT_TASK"
echo "rvrt_ckpt=$RVRT_CKPT"

"$PYTHON_BIN" -u evaluate_rvrt_ldmvfi.py \
  --ldm_config "$LDM_CONFIG" \
  --ldm_ckpt "$LDM_CKPT" \
  --dataset_root "$DATASET_ROOT" \
  --out_dir "$OUT_DIR" \
  --scale "$SCALE" \
  --sr_mode "$SR_MODE" \
  --rvrt_root "$RVRT_ROOT" \
  --rvrt_task "$RVRT_TASK" \
  --rvrt_ckpt "$RVRT_CKPT" \
  --use_ddim
