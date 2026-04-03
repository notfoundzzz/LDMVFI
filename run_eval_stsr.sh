#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/configs/ldm/stsr-x2-resizecond.yaml}"
CKPT_PATH="${CKPT_PATH:-/data/Shenzhen/zhahongli/models/ldmvfi/ldmvfi-vqflow-f32-c256-concat_max.ckpt}"
DATA_DIR="${DATA_DIR:-/data/Shenzhen/zhahongli/benchmarks}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/eval_results_stsr}"
DATASET="${DATASET:-Ucf_STSR}"

cd "$REPO_DIR"
python -u evaluate_stsr.py \
  --config "$CONFIG_PATH" \
  --ckpt "$CKPT_PATH" \
  --dataset "$DATASET" \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --scale_factor 2
