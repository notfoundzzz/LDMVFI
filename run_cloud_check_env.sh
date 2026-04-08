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
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs}"

mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/check_env_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_ROOT/latest_check_env.log"

echo "root=$ROOT_DIR" | tee -a "$LOG_FILE"
echo "python=$PYTHON_BIN" | tee -a "$LOG_FILE"
echo "path=$PATH" | tee -a "$LOG_FILE"

"$PYTHON_BIN" -u check_env_ldmvfi.py 2>&1 | tee -a "$LOG_FILE"
