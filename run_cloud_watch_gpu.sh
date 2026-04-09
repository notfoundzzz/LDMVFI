#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

INTERVAL="${INTERVAL:-5}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs}"
QUERY_FIELDS="${QUERY_FIELDS:-index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw}"

mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/gpu_watch_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_ROOT/latest_gpu_watch.log"

echo "root=$ROOT_DIR" | tee -a "$LOG_FILE"
echo "interval=$INTERVAL" | tee -a "$LOG_FILE"
echo "query_fields=$QUERY_FIELDS" | tee -a "$LOG_FILE"
echo "log_file=$LOG_FILE" | tee -a "$LOG_FILE"

while true; do
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "$LOG_FILE"
  nvidia-smi \
    --query-gpu="$QUERY_FIELDS" \
    --format=csv,noheader,nounits | tee -a "$LOG_FILE"
  echo | tee -a "$LOG_FILE"
  sleep "$INTERVAL"
done
