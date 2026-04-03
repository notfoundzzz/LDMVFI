#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-ldmvfi}"
OUT_DIR="${OUT_DIR:-$PWD/dist}"
ARCHIVE_NAME="${ARCHIVE_NAME:-${ENV_NAME}_env_$(date +%Y%m%d_%H%M%S).tar.gz}"

mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_DIR/pack_${ENV_NAME}_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$OUT_DIR/latest_pack.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "env_name=$ENV_NAME"
echo "out_dir=$OUT_DIR"
echo "archive_name=$ARCHIVE_NAME"

conda run -n "$ENV_NAME" python "$PWD/check_env_ldmvfi.py"
conda run -n "$ENV_NAME" python -m pip uninstall -y clip taming-transformers latent-diffusion || true
conda run -n "$ENV_NAME" python -m pip install --no-deps "git+https://github.com/CompVis/taming-transformers.git@master"
conda run -n "$ENV_NAME" python -m pip install --no-deps "git+https://github.com/openai/CLIP.git@main"
conda run -n "$ENV_NAME" python -m pip install --no-deps "$PWD"
conda pack -n "$ENV_NAME" -o "$OUT_DIR/$ARCHIVE_NAME"

echo "packed_env=$OUT_DIR/$ARCHIVE_NAME"
