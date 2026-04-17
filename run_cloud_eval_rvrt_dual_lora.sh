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

GPU_ID="${GPU_ID:-}"
LDM_CONFIG="${LDM_CONFIG:-$ROOT_DIR/configs/ldm/rvrt-dual-lora-stsr-x4.yaml}"
LDM_CKPT="${LDM_CKPT:-}"
DATA_ROOT="${DATA_ROOT:-/data/Shenzhen/zzff/STVSR/data/vimeo_septuplet}"
DATASET_ROOT_HR="${DATASET_ROOT_HR:-$DATA_ROOT/sequences}"
DATASET_ROOT_LR="${DATASET_ROOT_LR:-$DATA_ROOT/sequences_LR}"
SPLIT="${SPLIT:-slow_test}"
SPLITS="${SPLITS:-$SPLIT}"
LIST_FILE="${LIST_FILE:-}"
SCALE="${SCALE:-4}"
SR_MODE="${SR_MODE:-rvrt}"
RVRT_ROOT="${RVRT_ROOT:-/data/Shenzhen/zhahongli/RVRT}"
RVRT_TASK="${RVRT_TASK:-002_RVRT_videosr_bi_Vimeo_14frames}"
RVRT_CKPT="${RVRT_CKPT:-$RVRT_ROOT/model_zoo/rvrt/${RVRT_TASK}.pth}"
PIXEL_LORA_PATTERNS="${PIXEL_LORA_PATTERNS:-}"
SEMANTIC_LORA_PATTERNS="${SEMANTIC_LORA_PATTERNS:-}"
PIXEL_SCALE="${PIXEL_SCALE:-1.0}"
SEMANTIC_SCALE="${SEMANTIC_SCALE:-1.0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/eval_results_rvrt_dual_lora/${SPLIT}}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs}"
SAVE_IMAGES="${SAVE_IMAGES:-0}"
SAVE_SR_IMAGES="${SAVE_SR_IMAGES:-0}"
SAVE_MAX_SAMPLES="${SAVE_MAX_SAMPLES:-0}"

if [[ -z "$LDM_CKPT" ]]; then
  echo "LDM_CKPT is required"
  exit 1
fi

mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/eval_rvrt_dual_lora_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_ROOT/latest_eval_rvrt_dual_lora.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
echo "path=$PATH"
echo "gpu_id=${GPU_ID:-default}"
echo "ldm_config=$LDM_CONFIG"
echo "ldm_ckpt=$LDM_CKPT"
echo "data_root=$DATA_ROOT"
echo "dataset_root_hr=$DATASET_ROOT_HR"
echo "dataset_root_lr=$DATASET_ROOT_LR"
echo "splits=$SPLITS"
echo "pixel_lora_patterns=${PIXEL_LORA_PATTERNS:-default}"
echo "semantic_lora_patterns=${SEMANTIC_LORA_PATTERNS:-default}"
echo "pixel_scale=$PIXEL_SCALE"
echo "semantic_scale=$SEMANTIC_SCALE"
echo "max_samples=$MAX_SAMPLES"

if [ -n "$GPU_ID" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

IFS=',' read -r -a SPLIT_ARRAY <<< "$SPLITS"
SUMMARY_DIR="$OUT_DIR/_summaries"
mkdir -p "$SUMMARY_DIR"

for SPLIT_NAME in "${SPLIT_ARRAY[@]}"; do
  SPLIT_NAME="$(echo "$SPLIT_NAME" | xargs)"
  if [ -z "$SPLIT_NAME" ]; then
    continue
  fi

  CURRENT_LIST_FILE="$LIST_FILE"
  if [ -z "$CURRENT_LIST_FILE" ]; then
    CURRENT_LIST_FILE="$DATA_ROOT/sep_${SPLIT_NAME}list.txt"
  fi

  SPLIT_OUT_DIR="$OUT_DIR/$SPLIT_NAME"
  SUMMARY_JSON="$SUMMARY_DIR/${SPLIT_NAME}.json"
  echo "==== running split=$SPLIT_NAME ===="
  echo "split_out_dir=$SPLIT_OUT_DIR"
  echo "summary_json=$SUMMARY_JSON"

  CMD=(
    "$PYTHON_BIN" -u evaluate_rvrt_dual_lora.py
    --ldm_config "$LDM_CONFIG"
    --ldm_ckpt "$LDM_CKPT"
    --dataset_root_hr "$DATASET_ROOT_HR"
    --dataset_root_lr "$DATASET_ROOT_LR"
    --split "$SPLIT_NAME"
    --out_dir "$SPLIT_OUT_DIR"
    --scale "$SCALE"
    --sr_mode "$SR_MODE"
    --rvrt_root "$RVRT_ROOT"
    --rvrt_task "$RVRT_TASK"
    --rvrt_ckpt "$RVRT_CKPT"
    --pixel_scale "$PIXEL_SCALE"
    --semantic_scale "$SEMANTIC_SCALE"
    --max_samples "$MAX_SAMPLES"
    --summary_json "$SUMMARY_JSON"
    --use_ddim
  )

  if [[ -n "$PIXEL_LORA_PATTERNS" ]]; then
    IFS=',' read -r -a PIX_PAT <<< "$PIXEL_LORA_PATTERNS"
    CMD+=(--pixel_lora_patterns "${PIX_PAT[@]}")
  fi
  if [[ -n "$SEMANTIC_LORA_PATTERNS" ]]; then
    IFS=',' read -r -a SEM_PAT <<< "$SEMANTIC_LORA_PATTERNS"
    CMD+=(--semantic_lora_patterns "${SEM_PAT[@]}")
  fi

  if [ -n "$CURRENT_LIST_FILE" ] && [ -f "$CURRENT_LIST_FILE" ]; then
    CMD+=(--list_file "$CURRENT_LIST_FILE")
  fi
  if [[ "$SAVE_IMAGES" == "1" || "$SAVE_IMAGES" == "true" ]]; then
    CMD+=(--save_images)
  fi
  if [[ "$SAVE_SR_IMAGES" == "1" || "$SAVE_SR_IMAGES" == "true" ]]; then
    CMD+=(--save_sr_images)
  fi
  if [[ "$SAVE_MAX_SAMPLES" != "0" ]]; then
    CMD+=(--save_max_samples "$SAVE_MAX_SAMPLES")
  fi

  "${CMD[@]}"
done

"$PYTHON_BIN" - "$SUMMARY_DIR" <<'PY'
import json
import os
import sys

summary_dir = sys.argv[1]
summary_paths = sorted(
    os.path.join(summary_dir, name)
    for name in os.listdir(summary_dir)
    if name.endswith(".json")
)
if not summary_paths:
    raise SystemExit("No split summary files were produced.")

summaries = []
for path in summary_paths:
    with open(path, "r", encoding="utf-8") as f:
        summaries.append(json.load(f))

metric_names = list(summaries[0]["average"].keys())
total_samples = sum(item["num_samples"] for item in summaries)
overall = {}
for metric in metric_names:
    overall[metric] = round(
        sum(item["average"][metric] * item["num_samples"] for item in summaries) / total_samples,
        3,
    )

print("==== split summaries ====")
for item in summaries:
    print(item["split"], item["average"], f"samples={item['num_samples']}")
print("==== overall average ====")
print(overall, f"samples={total_samples}")
PY
