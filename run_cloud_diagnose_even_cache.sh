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

GPU_ID="${GPU_ID:-6}"
CACHE_ROOT="${CACHE_ROOT:-/cache/zhahongli/even_corrector_spynet_ddim200_flow_400}"
SPLITS="${SPLITS:-slow_test,medium_test,fast_test}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
METRICS="${METRICS:-PSNR,SSIM,LPIPS}"
MAX_RESIDUES="${MAX_RESIDUES:-0.25,0.5}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/diagnostics/even_corrector_cache}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs}"

mkdir -p "$OUT_DIR" "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/diagnose_even_cache_${STAMP}.log"
SUMMARY_JSON="$OUT_DIR/diagnose_even_cache_${STAMP}.json"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_ROOT/latest_diagnose_even_cache.log"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
echo "gpu_id=$GPU_ID"
echo "cache_root=$CACHE_ROOT"
echo "splits=$SPLITS"
echo "max_samples=$MAX_SAMPLES"
echo "metrics=$METRICS"
echo "max_residues=$MAX_RESIDUES"
echo "log_file=$LOG_FILE"
echo "summary_json=$SUMMARY_JSON"

CMD=(
  "$PYTHON_BIN" -u diagnose_even_corrector_cache.py
  --cache_root "$CACHE_ROOT"
  --splits "$SPLITS"
  --max_samples "$MAX_SAMPLES"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --metrics ${METRICS//,/ }
  --max_residues "$MAX_RESIDUES"
  --summary_json "$SUMMARY_JSON"
)

echo "==== command ===="
printf '%q ' "${CMD[@]}"
echo
echo "==== start diagnosis ===="
"${CMD[@]}"
