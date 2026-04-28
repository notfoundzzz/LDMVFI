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

GPU_ID="${GPU_ID:-6}"
DATA_ROOT="${DATA_ROOT:-/data/Shenzhen/zzff/STVSR/data/vimeo_septuplet}"
DATASET_ROOT_HR="${DATASET_ROOT_HR:-$DATA_ROOT/sequences}"
DATASET_ROOT_LR="${DATASET_ROOT_LR:-$DATA_ROOT/sequences_LR}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
TRAIN_LIST="${TRAIN_LIST:-$DATA_ROOT/sep_trainlist.txt}"
VAL_SPLITS="${VAL_SPLITS:-slow_test,medium_test,fast_test}"
VAL_LISTS="${VAL_LISTS:-}"
LDM_CONFIG="${LDM_CONFIG:-$ROOT_DIR/configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml}"
LDM_CKPT="${LDM_CKPT:-/data/Shenzhen/zhahongli/models/ldmvfi/ldmvfi-vqflow-f32-c256-concat_max.ckpt}"
RVRT_ROOT="${RVRT_ROOT:-/data/Shenzhen/zhahongli/RVRT_flow_ablate}"
RVRT_TASK="${RVRT_TASK:-002_RVRT_videosr_bi_Vimeo_14frames}"
RVRT_CKPT="${RVRT_CKPT:-/data/Shenzhen/zhahongli/RVRT/model_zoo/rvrt/${RVRT_TASK}.pth}"
RVRT_FLOW_MODE="${RVRT_FLOW_MODE:-spynet}"
RVRT_RAFT_VARIANT="${RVRT_RAFT_VARIANT:-large}"
RVRT_RAFT_CKPT="${RVRT_RAFT_CKPT:-}"
RVRT_USE_FLOW_ADAPTER="${RVRT_USE_FLOW_ADAPTER:-0}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/experiments/even_residual_corrector_spynet}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-32}"
NUM_BLOCKS="${NUM_BLOCKS:-4}"
MAX_RESIDUE="${MAX_RESIDUE:-0.25}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_STEPS="${MAX_STEPS:-1000}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-10}"
VAL_INTERVAL="${VAL_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
USE_DDIM="${USE_DDIM:-1}"
DDIM_STEPS="${DDIM_STEPS:-200}"
DDIM_ETA="${DDIM_ETA:-0}"
METRICS="${METRICS:-PSNR,SSIM,LPIPS}"
SEED="${SEED:-1234}"
USE_RAW_WEIGHTS="${USE_RAW_WEIGHTS:-0}"
ALLOW_INCOMPLETE_CKPT="${ALLOW_INCOMPLETE_CKPT:-0}"

if [[ -z "$VAL_LISTS" ]]; then
  IFS=',' read -r -a VAL_SPLIT_ARRAY <<< "$VAL_SPLITS"
  VAL_LIST_ARRAY=()
  for SPLIT_NAME in "${VAL_SPLIT_ARRAY[@]}"; do
    SPLIT_NAME="$(echo "$SPLIT_NAME" | xargs)"
    if [[ -n "$SPLIT_NAME" ]]; then
      VAL_LIST_ARRAY+=("$DATA_ROOT/sep_${SPLIT_NAME}list.txt")
    fi
  done
  VAL_LISTS="$(IFS=,; echo "${VAL_LIST_ARRAY[*]}")"
fi

mkdir -p "$LOG_ROOT" "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/train_even_corrector_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_ROOT/latest_train_even_corrector.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
echo "gpu_id=$GPU_ID"
echo "data_root=$DATA_ROOT"
echo "train_split=$TRAIN_SPLIT"
echo "train_list=$TRAIN_LIST"
echo "val_splits=$VAL_SPLITS"
echo "val_lists=$VAL_LISTS"
echo "ldm_config=$LDM_CONFIG"
echo "ldm_ckpt=$LDM_CKPT"
echo "rvrt_root=$RVRT_ROOT"
echo "rvrt_ckpt=$RVRT_CKPT"
echo "rvrt_flow_mode=$RVRT_FLOW_MODE"
echo "out_dir=$OUT_DIR"
echo "hidden_channels=$HIDDEN_CHANNELS"
echo "num_blocks=$NUM_BLOCKS"
echo "max_residue=$MAX_RESIDUE"
echo "lr=$LR"
echo "batch_size=$BATCH_SIZE"
echo "max_steps=$MAX_STEPS"
echo "max_val_samples=$MAX_VAL_SAMPLES"
echo "val_interval=$VAL_INTERVAL"
echo "ddim_steps=$DDIM_STEPS"
echo "metrics=$METRICS"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

CMD=(
  "$PYTHON_BIN" -u train_even_residual_corrector.py
  --ldm_config "$LDM_CONFIG"
  --ldm_ckpt "$LDM_CKPT"
  --dataset_root_hr "$DATASET_ROOT_HR"
  --dataset_root_lr "$DATASET_ROOT_LR"
  --train_split "$TRAIN_SPLIT"
  --train_list "$TRAIN_LIST"
  --val_splits "$VAL_SPLITS"
  --val_lists "$VAL_LISTS"
  --max_train_samples "$MAX_TRAIN_SAMPLES"
  --max_val_samples "$MAX_VAL_SAMPLES"
  --rvrt_root "$RVRT_ROOT"
  --rvrt_task "$RVRT_TASK"
  --rvrt_ckpt "$RVRT_CKPT"
  --rvrt_flow_mode "$RVRT_FLOW_MODE"
  --rvrt_raft_variant "$RVRT_RAFT_VARIANT"
  --rvrt_use_flow_adapter "$RVRT_USE_FLOW_ADAPTER"
  --out_dir "$OUT_DIR"
  --hidden_channels "$HIDDEN_CHANNELS"
  --num_blocks "$NUM_BLOCKS"
  --max_residue "$MAX_RESIDUE"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --max_steps "$MAX_STEPS"
  --val_interval "$VAL_INTERVAL"
  --save_interval "$SAVE_INTERVAL"
  --use_ddim "$USE_DDIM"
  --ddim_steps "$DDIM_STEPS"
  --ddim_eta "$DDIM_ETA"
  --metrics ${METRICS//,/ }
  --seed "$SEED"
)

if [[ -n "$RVRT_RAFT_CKPT" ]]; then
  CMD+=(--rvrt_raft_ckpt "$RVRT_RAFT_CKPT")
fi
if [[ "$USE_RAW_WEIGHTS" == "1" || "$USE_RAW_WEIGHTS" == "true" ]]; then
  CMD+=(--use_raw_weights)
fi
if [[ "$ALLOW_INCOMPLETE_CKPT" == "1" || "$ALLOW_INCOMPLETE_CKPT" == "true" ]]; then
  CMD+=(--allow_incomplete_ckpt)
fi

echo "==== command ===="
printf '%q ' "${CMD[@]}"
echo
echo "==== start training ===="
"${CMD[@]}"
