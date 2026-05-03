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

GPU_IDS="${GPU_IDS:-0,1,2,3}"
CACHE_ROOT="${CACHE_ROOT:-/cache/zhahongli/even_corrector_spynet_ddim200_flow_400}"
SPLITS="${SPLITS:-slow_test,medium_test,fast_test}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
METRICS="${METRICS:-PSNR,SSIM,LPIPS}"
MAX_RESIDUES="${MAX_RESIDUES:-0.25,0.5}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/diagnostics/even_corrector_cache}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs}"
STAMP="${DIAG_STAMP:-$(date +%Y%m%d_%H%M%S)}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_SHARDS="${#GPUS[@]}"
if [ "$NUM_SHARDS" -lt 1 ]; then
  echo "GPU_IDS must contain at least one GPU id" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_ROOT"
MASTER_LOG="$LOG_ROOT/diagnose_even_cache_${STAMP}_parallel.log"
MERGED_JSON="$OUT_DIR/diagnose_even_cache_${STAMP}_merged.json"
ln -sfn "$(basename "$MASTER_LOG")" "$LOG_ROOT/latest_diagnose_even_cache_parallel.log"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
echo "gpu_ids=$GPU_IDS"
echo "num_shards=$NUM_SHARDS"
echo "cache_root=$CACHE_ROOT"
echo "splits=$SPLITS"
echo "max_samples=$MAX_SAMPLES"
echo "metrics=$METRICS"
echo "max_residues=$MAX_RESIDUES"
echo "out_dir=$OUT_DIR"
echo "log_root=$LOG_ROOT"
echo "stamp=$STAMP"

pids=()
for shard_id in "${!GPUS[@]}"; do
  gpu_id="${GPUS[$shard_id]}"
  echo "==== launch shard=$shard_id/$NUM_SHARDS gpu=$gpu_id ===="
  (
    GPU_ID="$gpu_id" \
    SHARD_ID="$shard_id" \
    NUM_SHARDS="$NUM_SHARDS" \
    DIAG_STAMP="$STAMP" \
    CACHE_ROOT="$CACHE_ROOT" \
    SPLITS="$SPLITS" \
    MAX_SAMPLES="$MAX_SAMPLES" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    METRICS="$METRICS" \
    MAX_RESIDUES="$MAX_RESIDUES" \
    OUT_DIR="$OUT_DIR" \
    LOG_ROOT="$LOG_ROOT" \
    PYTHON_BIN="$PYTHON_BIN" \
    bash run_cloud_diagnose_even_cache.sh
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [ "$failed" -ne 0 ]; then
  echo "At least one diagnosis shard failed. Not merging summaries." >&2
  exit 1
fi

echo "==== merge shard summaries ===="
"$PYTHON_BIN" -u merge_even_diagnosis_summaries.py \
  --inputs "$OUT_DIR/diagnose_even_cache_${STAMP}_shard*of${NUM_SHARDS}.json" \
  --output "$MERGED_JSON"

echo "merged_summary_json=$MERGED_JSON"
