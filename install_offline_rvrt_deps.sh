#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="${PKG_DIR:-$ROOT_DIR/offline_pkgs_rvrt}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
echo "pkg_dir=$PKG_DIR"

"$PYTHON_BIN" -m pip install --no-index --find-links "$PKG_DIR" addict ftfy ninja tifffile lazy_loader
"$PYTHON_BIN" -m pip install --no-index --find-links "$PKG_DIR" --no-deps scikit-image==0.24.0

echo "offline RVRT deps installed"
