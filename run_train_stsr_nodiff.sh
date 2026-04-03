#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/Shenzhen/zhahongli/datasets}"
LOGDIR="${LOGDIR:-$REPO_DIR/logs}"

cd "$REPO_DIR"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOGDIR/train_stsr_nodiff_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOGDIR/latest_train_stsr_nodiff.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "repo=$REPO_DIR"
echo "data_root=$DATA_ROOT"
echo "logdir=$LOGDIR"

python -u main.py \
  -t True \
  --base configs/ldm/stsr-x2-nodiff-baseline.yaml \
  --logdir "$LOGDIR" \
  --name stsr_x2_nodiff \
  data.params.train.params.db_dir="$DATA_ROOT" \
  data.params.validation.params.db_dir="$DATA_ROOT/vimeo_septuplet"
