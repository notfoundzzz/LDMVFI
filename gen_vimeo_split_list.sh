#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/Shenzhen/zzff/STVSR/data/vimeo_septuplet}"
LR_ROOT="${LR_ROOT:-$DATA_ROOT/sequences_LR}"
SPLIT="${SPLIT:-slow_test}"
MARKER_FILE="${MARKER_FILE:-im3.png}"
OUT_FILE="${OUT_FILE:-$DATA_ROOT/${SPLIT}_list.txt}"

TARGET_DIR="$LR_ROOT/$SPLIT"

if [ ! -d "$TARGET_DIR" ]; then
  echo "missing split directory: $TARGET_DIR"
  exit 1
fi

mkdir -p "$(dirname "$OUT_FILE")"

find "$TARGET_DIR" -name "$MARKER_FILE" \
  | sed "s#^$TARGET_DIR/##; s#/$MARKER_FILE\$##" \
  | sort > "$OUT_FILE"

COUNT="$(wc -l < "$OUT_FILE" | tr -d ' ')"

echo "data_root=$DATA_ROOT"
echo "lr_root=$LR_ROOT"
echo "split=$SPLIT"
echo "marker_file=$MARKER_FILE"
echo "out_file=$OUT_FILE"
echo "count=$COUNT"
