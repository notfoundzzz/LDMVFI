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
LDM_CONFIG="${LDM_CONFIG:-$ROOT_DIR/configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml}"
LDM_CKPT="${LDM_CKPT:-/data/Shenzhen/zhahongli/models/ldmvfi/ldmvfi-vqflow-f32-c256-concat_max.ckpt}"
DATA_ROOT="${DATA_ROOT:-/data/Shenzhen/zzff/STVSR/data/vimeo_septuplet}"
DATASET_ROOT_HR="${DATASET_ROOT_HR:-$DATA_ROOT/sequences}"
DATASET_ROOT_LR="${DATASET_ROOT_LR:-$DATA_ROOT/sequences_LR}"
SPLIT="${SPLIT:-slow_test}"
SPLITS="${SPLITS:-$SPLIT}"
LIST_FILE="${LIST_FILE:-}"
SCALE="${SCALE:-4}"
SR_MODE="${SR_MODE:-bicubic}"
EVAL_PIPELINE="${EVAL_PIPELINE:-sr_then_vfi}"
RVRT_ROOT="${RVRT_ROOT:-/data/Shenzhen/zhahongli/RVRT}"
RVRT_TASK="${RVRT_TASK:-002_RVRT_videosr_bi_Vimeo_14frames}"
RVRT_CKPT="${RVRT_CKPT:-$RVRT_ROOT/model_zoo/rvrt/${RVRT_TASK}.pth}"
RVRT_FLOW_MODE="${RVRT_FLOW_MODE:-spynet}"
RVRT_RAFT_VARIANT="${RVRT_RAFT_VARIANT:-large}"
RVRT_RAFT_CKPT="${RVRT_RAFT_CKPT:-}"
RVRT_USE_FLOW_ADAPTER="${RVRT_USE_FLOW_ADAPTER:-}"
RVRT_ADAPTER_CKPT="${RVRT_ADAPTER_CKPT:-}"
RVRT_FLOW_ADAPTER_HIDDEN_CHANNELS="${RVRT_FLOW_ADAPTER_HIDDEN_CHANNELS:-}"
RVRT_FLOW_ADAPTER_ZERO_INIT_LAST="${RVRT_FLOW_ADAPTER_ZERO_INIT_LAST:-}"
RVRT_FLOW_ADAPTER_MAX_RESIDUE="${RVRT_FLOW_ADAPTER_MAX_RESIDUE:-}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/eval_results_rvrt_ldmvfi/${SPLIT}/${SR_MODE}}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs}"
DEFAULT_LORA_CONFIG="$ROOT_DIR/configs/ldm/rvrt-lora-stsr-x4.yaml"
DEFAULT_BASE_CONFIG="$ROOT_DIR/configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml"
SAVE_IMAGES="${SAVE_IMAGES:-0}"
SAVE_SR_IMAGES="${SAVE_SR_IMAGES:-0}"
SAVE_MAX_SAMPLES="${SAVE_MAX_SAMPLES:-0}"
USE_DDIM="${USE_DDIM:-1}"
DDIM_ETA="${DDIM_ETA:-0}"
SEED="${SEED:-1234}"
METRICS="${METRICS:-PSNR,SSIM}"
USE_RAW_WEIGHTS="${USE_RAW_WEIGHTS:-0}"
ALLOW_INCOMPLETE_CKPT="${ALLOW_INCOMPLETE_CKPT:-0}"
USE_FLOW_GUIDANCE="${USE_FLOW_GUIDANCE:-}"
FLOW_BACKEND="${FLOW_BACKEND:-}"
FLOW_CONDITION_MODE="${FLOW_CONDITION_MODE:-}"
FLOW_RAFT_VARIANT="${FLOW_RAFT_VARIANT:-}"
FLOW_RAFT_CKPT="${FLOW_RAFT_CKPT:-}"

if [[ "$LDM_CONFIG" == "$DEFAULT_BASE_CONFIG" ]] && [[ "$LDM_CKPT" == *"rvrt-lora"* ]]; then
  LDM_CONFIG="$DEFAULT_LORA_CONFIG"
fi

mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/eval_rvrt_ldmvfi_${SR_MODE}_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_ROOT/latest_eval_rvrt_ldmvfi.log"
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
echo "list_file=$LIST_FILE"
echo "out_dir=$OUT_DIR"
echo "scale=$SCALE"
echo "sr_mode=$SR_MODE"
echo "eval_pipeline=$EVAL_PIPELINE"
echo "rvrt_root=$RVRT_ROOT"
echo "rvrt_task=$RVRT_TASK"
echo "rvrt_ckpt=$RVRT_CKPT"
echo "rvrt_flow_mode=$RVRT_FLOW_MODE"
echo "rvrt_raft_variant=$RVRT_RAFT_VARIANT"
echo "rvrt_raft_ckpt=${RVRT_RAFT_CKPT:-default}"
echo "rvrt_use_flow_adapter=${RVRT_USE_FLOW_ADAPTER:-default}"
echo "rvrt_adapter_ckpt=${RVRT_ADAPTER_CKPT:-default}"
echo "rvrt_flow_adapter_hidden_channels=${RVRT_FLOW_ADAPTER_HIDDEN_CHANNELS:-default}"
echo "rvrt_flow_adapter_zero_init_last=${RVRT_FLOW_ADAPTER_ZERO_INIT_LAST:-default}"
echo "rvrt_flow_adapter_max_residue=${RVRT_FLOW_ADAPTER_MAX_RESIDUE:-default}"
echo "max_samples=$MAX_SAMPLES"
echo "save_images=$SAVE_IMAGES"
echo "save_sr_images=$SAVE_SR_IMAGES"
echo "save_max_samples=$SAVE_MAX_SAMPLES"
echo "use_ddim=$USE_DDIM"
echo "ddim_eta=$DDIM_ETA"
echo "seed=$SEED"
echo "metrics=$METRICS"
echo "use_raw_weights=$USE_RAW_WEIGHTS"
echo "allow_incomplete_ckpt=$ALLOW_INCOMPLETE_CKPT"
echo "use_flow_guidance=${USE_FLOW_GUIDANCE:-default}"
echo "flow_condition_mode=${FLOW_CONDITION_MODE:-default}"
echo "flow_backend=${FLOW_BACKEND:-default}"
echo "flow_raft_variant=${FLOW_RAFT_VARIANT:-default}"
echo "flow_raft_ckpt=${FLOW_RAFT_CKPT:-default}"

if [[ "$LDM_CONFIG" == "$DEFAULT_BASE_CONFIG" ]] && [[ "$LDM_CKPT" == *"rvrt-lora"* ]]; then
  echo "Refusing to evaluate a LoRA checkpoint with the base LDMVFI config."
  echo "Set LDM_CONFIG=$DEFAULT_LORA_CONFIG or leave the default config selection enabled."
  exit 1
fi

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
    "$PYTHON_BIN" -u evaluate_rvrt_ldmvfi.py
    --ldm_config "$LDM_CONFIG" \
    --ldm_ckpt "$LDM_CKPT" \
    --dataset_root_hr "$DATASET_ROOT_HR" \
    --dataset_root_lr "$DATASET_ROOT_LR" \
    --split "$SPLIT_NAME" \
    --out_dir "$SPLIT_OUT_DIR" \
    --scale "$SCALE" \
    --sr_mode "$SR_MODE" \
    --eval_pipeline "$EVAL_PIPELINE" \
    --rvrt_root "$RVRT_ROOT" \
    --rvrt_task "$RVRT_TASK" \
    --rvrt_ckpt "$RVRT_CKPT" \
    --rvrt_flow_mode "$RVRT_FLOW_MODE" \
    --rvrt_raft_variant "$RVRT_RAFT_VARIANT" \
    --max_samples "$MAX_SAMPLES" \
    --summary_json "$SUMMARY_JSON" \
    --ddim_eta "$DDIM_ETA" \
    --seed "$SEED"
  )

  if [[ "$USE_DDIM" == "1" || "$USE_DDIM" == "true" ]]; then
    CMD+=(--use_ddim)
  else
    CMD+=(--use_ddpm)
  fi

  if [[ -n "$METRICS" ]]; then
    IFS=',' read -r -a METRIC_ARRAY <<< "$METRICS"
    if [[ "${#METRIC_ARRAY[@]}" -gt 0 ]]; then
      CMD+=(--metrics "${METRIC_ARRAY[@]}")
    fi
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

  if [[ "$USE_RAW_WEIGHTS" == "1" || "$USE_RAW_WEIGHTS" == "true" ]]; then
    CMD+=(--use_raw_weights)
  fi

  if [[ "$ALLOW_INCOMPLETE_CKPT" == "1" || "$ALLOW_INCOMPLETE_CKPT" == "true" ]]; then
    CMD+=(--allow_incomplete_ckpt)
  fi
  if [[ -n "$USE_FLOW_GUIDANCE" ]]; then
    CMD+=(--use_flow_guidance "$USE_FLOW_GUIDANCE")
  fi
  if [[ -n "$FLOW_BACKEND" ]]; then
    CMD+=(--flow_backend "$FLOW_BACKEND")
  fi
  if [[ -n "$FLOW_CONDITION_MODE" ]]; then
    CMD+=(--flow_condition_mode "$FLOW_CONDITION_MODE")
  fi
  if [[ -n "$FLOW_RAFT_VARIANT" ]]; then
    CMD+=(--flow_raft_variant "$FLOW_RAFT_VARIANT")
  fi
  if [[ -n "$FLOW_RAFT_CKPT" ]]; then
    CMD+=(--flow_raft_ckpt "$FLOW_RAFT_CKPT")
  fi
  if [[ -n "$RVRT_RAFT_CKPT" ]]; then
    CMD+=(--rvrt_raft_ckpt "$RVRT_RAFT_CKPT")
  fi
  if [[ -n "$RVRT_USE_FLOW_ADAPTER" ]]; then
    CMD+=(--rvrt_use_flow_adapter "$RVRT_USE_FLOW_ADAPTER")
  fi
  if [[ -n "$RVRT_ADAPTER_CKPT" ]]; then
    CMD+=(--rvrt_adapter_ckpt "$RVRT_ADAPTER_CKPT")
  fi
  if [[ -n "$RVRT_FLOW_ADAPTER_HIDDEN_CHANNELS" ]]; then
    CMD+=(--rvrt_flow_adapter_hidden_channels "$RVRT_FLOW_ADAPTER_HIDDEN_CHANNELS")
  fi
  if [[ -n "$RVRT_FLOW_ADAPTER_ZERO_INIT_LAST" ]]; then
    CMD+=(--rvrt_flow_adapter_zero_init_last "$RVRT_FLOW_ADAPTER_ZERO_INIT_LAST")
  fi
  if [[ -n "$RVRT_FLOW_ADAPTER_MAX_RESIDUE" ]]; then
    CMD+=(--rvrt_flow_adapter_max_residue_magnitude "$RVRT_FLOW_ADAPTER_MAX_RESIDUE")
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

total_samples = sum(item["num_samples"] for item in summaries)

print("==== split summaries ====")
if "averages" in summaries[0]:
    groups = list(summaries[0]["averages"].keys())
    for item in summaries:
        print(item["split"], item["averages"], f"samples={item['num_samples']}")

    overall = {}
    for group in groups:
        overall[group] = {}
        metric_names = list(summaries[0]["averages"][group].keys())
        for metric in metric_names:
            weighted_sum = 0.0
            weight_sum = 0
            for item in summaries:
                value = item["averages"][group][metric]
                if value is None:
                    continue
                weight = item.get("metric_frame_counts", {}).get(group, {}).get(metric, item["num_samples"])
                weighted_sum += value * weight
                weight_sum += weight
            overall[group][metric] = round(weighted_sum / weight_sum, 3) if weight_sum else None
else:
    metric_names = list(summaries[0]["average"].keys())
    overall = {}
    for item in summaries:
        print(item["split"], item["average"], f"samples={item['num_samples']}")
    for metric in metric_names:
        overall[metric] = round(
            sum(item["average"][metric] * item["num_samples"] for item in summaries) / total_samples,
            3,
        )
print("==== overall average ====")
print(overall, f"samples={total_samples}")
PY
