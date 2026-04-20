#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON_BIN="/data/Shenzhen/zhahongli/envs/ldmvfi/bin/python"
if [ -x "$DEFAULT_PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
unset LD_LIBRARY_PATH || true
unset CUDA_HOME || true
unset CUDA_PATH || true
PYTHON_DIR="$(dirname "$PYTHON_BIN")"
if [ -d "$PYTHON_DIR" ]; then
  export PATH="$PYTHON_DIR:$PATH"
fi

GPU_ID="${GPU_ID:-}"
LDM_CKPT="${LDM_CKPT:-}"
LDM_CONFIG="${LDM_CONFIG:-$ROOT_DIR/configs/ldm/rvrt-pisa-dual-lora-stsr-x4.yaml}"
DATA_ROOT="${DATA_ROOT:-/data/Shenzhen/zzff/STVSR/data/vimeo_septuplet}"
DATASET_ROOT_LR="${DATASET_ROOT_LR:-$DATA_ROOT/sequences_LR}"
SPLIT="${SPLIT:-slow_test}"
LIST_FILE="${LIST_FILE:-}"
SAMPLE_NAME="${SAMPLE_NAME:-}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
RVRT_ROOT="${RVRT_ROOT:-/data/Shenzhen/zhahongli/RVRT}"
RVRT_TASK="${RVRT_TASK:-002_RVRT_videosr_bi_Vimeo_14frames}"
RVRT_CKPT="${RVRT_CKPT:-$RVRT_ROOT/model_zoo/rvrt/${RVRT_TASK}.pth}"
PIXEL_LORA_GROUPS="${PIXEL_LORA_GROUPS:-}"
SEMANTIC_LORA_GROUPS="${SEMANTIC_LORA_GROUPS:-}"
PIXEL_TARGET_SUFFIXES="${PIXEL_TARGET_SUFFIXES:-}"
SEMANTIC_TARGET_SUFFIXES="${SEMANTIC_TARGET_SUFFIXES:-}"
PIXEL_SCALE_A="${PIXEL_SCALE_A:-1.0}"
SEMANTIC_SCALE_A="${SEMANTIC_SCALE_A:-1.0}"
PIXEL_SCALE_B="${PIXEL_SCALE_B:-1.0}"
SEMANTIC_SCALE_B="${SEMANTIC_SCALE_B:-0.0}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/check_results_rvrt_pisa_dual_lora}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs}"
USE_RAW_WEIGHTS="${USE_RAW_WEIGHTS:-0}"
USE_DDIM="${USE_DDIM:-1}"
DDIM_ETA="${DDIM_ETA:-0}"
SEED="${SEED:-1234}"

if [[ -z "$LDM_CKPT" ]]; then
  echo "LDM_CKPT is required"
  exit 1
fi

mkdir -p "$LOG_ROOT"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/check_rvrt_pisa_dual_lora_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_ROOT/latest_check_rvrt_pisa_dual_lora.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "ldm_ckpt=$LDM_CKPT"
echo "ldm_config=$LDM_CONFIG"
echo "split=$SPLIT"
echo "pixel_lora_groups=${PIXEL_LORA_GROUPS:-default}"
echo "semantic_lora_groups=${SEMANTIC_LORA_GROUPS:-default}"
echo "pixel_target_suffixes=${PIXEL_TARGET_SUFFIXES:-default}"
echo "semantic_target_suffixes=${SEMANTIC_TARGET_SUFFIXES:-default}"
echo "pixel_scale_a=$PIXEL_SCALE_A"
echo "semantic_scale_a=$SEMANTIC_SCALE_A"
echo "pixel_scale_b=$PIXEL_SCALE_B"
echo "semantic_scale_b=$SEMANTIC_SCALE_B"
echo "use_ddim=$USE_DDIM"
echo "ddim_eta=$DDIM_ETA"
echo "seed=$SEED"
echo "use_raw_weights=$USE_RAW_WEIGHTS"

if [ -n "$GPU_ID" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

CMD=(
  "$PYTHON_BIN" -u check_rvrt_pisa_dual_lora.py
  --ldm_ckpt "$LDM_CKPT"
  --ldm_config "$LDM_CONFIG"
  --dataset_root_lr "$DATASET_ROOT_LR"
  --split "$SPLIT"
  --scale 4
  --sr_mode rvrt
  --rvrt_root "$RVRT_ROOT"
  --rvrt_task "$RVRT_TASK"
  --rvrt_ckpt "$RVRT_CKPT"
  --pixel_scale_a "$PIXEL_SCALE_A"
  --semantic_scale_a "$SEMANTIC_SCALE_A"
  --pixel_scale_b "$PIXEL_SCALE_B"
  --semantic_scale_b "$SEMANTIC_SCALE_B"
  --summary_json "$OUT_DIR/summary.json"
  --save_dir "$OUT_DIR"
  --ddim_eta "$DDIM_ETA"
  --seed "$SEED"
)

if [[ "$USE_DDIM" == "1" || "$USE_DDIM" == "true" ]]; then
  CMD+=(--use_ddim)
else
  CMD+=(--use_ddpm)
fi

if [[ -n "$LIST_FILE" ]]; then
  CMD+=(--list_file "$LIST_FILE")
fi
if [[ -n "$PIXEL_LORA_GROUPS" ]]; then
  IFS=',' read -r -a PIX_GROUPS <<< "$PIXEL_LORA_GROUPS"
  CMD+=(--pixel_lora_groups "${PIX_GROUPS[@]}")
fi
if [[ -n "$SEMANTIC_LORA_GROUPS" ]]; then
  IFS=',' read -r -a SEM_GROUPS <<< "$SEMANTIC_LORA_GROUPS"
  CMD+=(--semantic_lora_groups "${SEM_GROUPS[@]}")
fi
if [[ -n "$PIXEL_TARGET_SUFFIXES" ]]; then
  IFS=',' read -r -a PIX_TARGETS <<< "$PIXEL_TARGET_SUFFIXES"
  CMD+=(--pixel_target_suffixes "${PIX_TARGETS[@]}")
fi
if [[ -n "$SEMANTIC_TARGET_SUFFIXES" ]]; then
  IFS=',' read -r -a SEM_TARGETS <<< "$SEMANTIC_TARGET_SUFFIXES"
  CMD+=(--semantic_target_suffixes "${SEM_TARGETS[@]}")
fi
if [[ -n "$SAMPLE_NAME" ]]; then
  CMD+=(--sample_name "$SAMPLE_NAME")
else
  CMD+=(--sample_index "$SAMPLE_INDEX")
fi
if [[ "$USE_RAW_WEIGHTS" == "1" || "$USE_RAW_WEIGHTS" == "true" ]]; then
  CMD+=(--use_raw_weights)
fi

"${CMD[@]}"
