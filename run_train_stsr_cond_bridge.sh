#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/Shenzhen/zhahongli/datasets}"
VQ_CKPT="${VQ_CKPT:-/data/Shenzhen/zhahongli/models/ldmvfi/vqflow.ckpt}"
LOGDIR="${LOGDIR:-$REPO_DIR/logs}"

cd "$REPO_DIR"
python -u main.py \
  -t True \
  --base configs/ldm/stsr-x2-cond-bridge.yaml \
  --logdir "$LOGDIR" \
  --name stsr_x2_cond_bridge \
  data.params.train.params.db_dir="$DATA_ROOT" \
  data.params.validation.params.db_dir="$DATA_ROOT/vimeo_septuplet" \
  model.params.first_stage_config.params.ckpt_path="$VQ_CKPT"
