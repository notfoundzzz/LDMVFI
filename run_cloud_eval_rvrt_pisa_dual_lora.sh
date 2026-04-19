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
LDM_CONFIG="${LDM_CONFIG:-$ROOT_DIR/configs/ldm/rvrt-pisa-dual-lora-stsr-x4.yaml}"
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
PIXEL_LORA_GROUPS="${PIXEL_LORA_GROUPS:-}"
SEMANTIC_LORA_GROUPS="${SEMANTIC_LORA_GROUPS:-}"
LORA_TARGET_SUFFIXES="${LORA_TARGET_SUFFIXES:-}"
PIXEL_SCALE="${PIXEL_SCALE:-1.0}"
SEMANTIC_SCALE="${SEMANTIC_SCALE:-1.0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/eval_results_rvrt_pisa_dual_lora/${SPLIT}}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs}"
SAVE_IMAGES="${SAVE_IMAGES:-0}"
SAVE_SR_IMAGES="${SAVE_SR_IMAGES:-0}"
SAVE_MAX_SAMPLES="${SAVE_MAX_SAMPLES:-0}"
USE_RAW_WEIGHTS="${USE_RAW_WEIGHTS:-0}"
USE_DDIM="${USE_DDIM:-1}"
DDIM_ETA="${DDIM_ETA:-0}"
SEED="${SEED:-1234}"
ALLOW_INCOMPLETE_CKPT="${ALLOW_INCOMPLETE_CKPT:-0}"

if [[ -z "$LDM_CKPT" ]]; then
  echo "LDM_CKPT is required"
  exit 1
fi

mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/eval_rvrt_pisa_dual_lora_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_ROOT/latest_eval_rvrt_pisa_dual_lora.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
echo "path=$PATH"
echo "gpu_id=${GPU_ID:-default}"
echo "ldm_config=$LDM_CONFIG"
echo "ldm_ckpt=$LDM_CKPT"
echo "splits=$SPLITS"
echo "pixel_lora_groups=${PIXEL_LORA_GROUPS:-default}"
echo "semantic_lora_groups=${SEMANTIC_LORA_GROUPS:-default}"
echo "lora_target_suffixes=${LORA_TARGET_SUFFIXES:-default}"
echo "pixel_scale=$PIXEL_SCALE"
echo "semantic_scale=$SEMANTIC_SCALE"
echo "use_raw_weights=$USE_RAW_WEIGHTS"
echo "use_ddim=$USE_DDIM"
echo "ddim_eta=$DDIM_ETA"
echo "seed=$SEED"
echo "allow_incomplete_ckpt=$ALLOW_INCOMPLETE_CKPT"

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
  CMD=(
    "$PYTHON_BIN" -u evaluate_rvrt_pisa_dual_lora.py
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
    --ddim_eta "$DDIM_ETA"
    --seed "$SEED"
  )

  if [[ "$USE_DDIM" == "1" || "$USE_DDIM" == "true" ]]; then
    CMD+=(--use_ddim)
  else
    CMD+=(--use_ddpm)
  fi

  if [[ -n "$PIXEL_LORA_GROUPS" ]]; then
    IFS=',' read -r -a PIX_GROUPS <<< "$PIXEL_LORA_GROUPS"
    CMD+=(--pixel_lora_groups "${PIX_GROUPS[@]}")
  fi
  if [[ -n "$SEMANTIC_LORA_GROUPS" ]]; then
    IFS=',' read -r -a SEM_GROUPS <<< "$SEMANTIC_LORA_GROUPS"
    CMD+=(--semantic_lora_groups "${SEM_GROUPS[@]}")
  fi
  if [[ -n "$LORA_TARGET_SUFFIXES" ]]; then
    IFS=',' read -r -a TARGETS <<< "$LORA_TARGET_SUFFIXES"
    CMD+=(--lora_target_suffixes "${TARGETS[@]}")
  fi
  if [[ "$USE_RAW_WEIGHTS" == "1" || "$USE_RAW_WEIGHTS" == "true" ]]; then
    CMD+=(--use_raw_weights)
  fi
  if [[ "$ALLOW_INCOMPLETE_CKPT" == "1" || "$ALLOW_INCOMPLETE_CKPT" == "true" ]]; then
    CMD+=(--allow_incomplete_ckpt)
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
