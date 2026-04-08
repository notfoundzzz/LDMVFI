#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="${PKG_DIR:-$ROOT_DIR/offline_pkgs_rvrt}"
DEFAULT_PYTHON_BIN="/data/Shenzhen/zhahongli/envs/ldmvfi/bin/python"
if [ -x "$DEFAULT_PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs}"

mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/install_offline_rvrt_deps_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_ROOT/latest_install_offline_rvrt_deps.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
echo "pkg_dir=$PKG_DIR"

"$PYTHON_BIN" -m pip install --no-index --find-links "$PKG_DIR" addict ftfy ninja tifffile lazy_loader
"$PYTHON_BIN" -m pip install --no-index --find-links "$PKG_DIR" --no-deps scikit-image==0.24.0

echo "offline RVRT deps installed"
