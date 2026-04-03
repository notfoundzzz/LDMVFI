#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/Shenzhen/zhahongli/datasets}"
LOGDIR="${LOGDIR:-$REPO_DIR/logs}"

cd "$REPO_DIR"
python -u main.py \
  -t True \
  --base configs/ldm/stsr-x2-nodiff-baseline.yaml \
  --logdir "$LOGDIR" \
  --name stsr_x2_nodiff \
  data.params.train.params.db_dir="$DATA_ROOT" \
  data.params.validation.params.db_dir="$DATA_ROOT/vimeo_septuplet"
