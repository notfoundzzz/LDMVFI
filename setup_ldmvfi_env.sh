#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-ldmvfi}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs}"

cd "$ROOT_DIR"
mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/setup_env_${ENV_NAME}_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_ROOT/latest_setup_env.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "root=$ROOT_DIR"
echo "env_name=$ENV_NAME"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "env_exists=$ENV_NAME"
else
  conda create -y -n "$ENV_NAME" python=3.9.13 pip=24.0
fi

conda install -y -n "$ENV_NAME" -c pytorch pytorch=1.11.0 torchvision=0.12.0 cudatoolkit=11.3

# PyTorch 1.11 can break with newer Intel runtime packages and fail with:
# libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
# Pin to a pre-2024.1 Intel runtime that exists on defaults.
conda install -y -n "$ENV_NAME" "mkl=2023.1.0" "intel-openmp=2023.1.0"

conda run -n "$ENV_NAME" python -m pip install --upgrade "pip<24.1"
conda run -n "$ENV_NAME" python -m pip install --upgrade --force-reinstall "setuptools<81"
conda run -n "$ENV_NAME" python -m pip install --constraint "$ROOT_DIR/constraints-pip.txt" -r "$ROOT_DIR/requirements-pip.txt"
conda run -n "$ENV_NAME" python -m pip install --upgrade --force-reinstall "numpy<2" "torchmetrics==0.10.3"

# Install editable repos without re-resolving heavy dependencies.
conda run -n "$ENV_NAME" python -m pip install --no-deps -e "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers"
conda run -n "$ENV_NAME" python -m pip install --no-deps -e "git+https://github.com/openai/CLIP.git@main#egg=clip"
conda run -n "$ENV_NAME" python -m pip install --no-deps -e "$ROOT_DIR"

conda run -n "$ENV_NAME" python "$ROOT_DIR/check_env_ldmvfi.py"

echo "ready_env=$ENV_NAME"
echo "pack_command=bash $ROOT_DIR/pack_ldmvfi_env.sh"
