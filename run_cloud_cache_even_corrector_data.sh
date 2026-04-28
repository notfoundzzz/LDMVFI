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

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29591}"
DATA_ROOT="${DATA_ROOT:-/data/Shenzhen/zzff/STVSR/data/vimeo_septuplet}"
DATASET_ROOT_HR="${DATASET_ROOT_HR:-$DATA_ROOT/sequences}"
DATASET_ROOT_LR="${DATASET_ROOT_LR:-$DATA_ROOT/sequences_LR}"
LR_SPLIT_LAYOUT="${LR_SPLIT_LAYOUT:-auto}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
TRAIN_LIST="${TRAIN_LIST:-$DATA_ROOT/sep_trainlist.txt}"
VAL_SPLITS="${VAL_SPLITS:-slow_test,medium_test,fast_test}"
VAL_LISTS="${VAL_LISTS:-}"
CACHE_TRAIN="${CACHE_TRAIN:-1}"
CACHE_VAL="${CACHE_VAL:-1}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-10}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/cache/even_corrector_spynet_ddim200}"
CACHE_DTYPE="${CACHE_DTYPE:-float16}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
LDM_CONFIG="${LDM_CONFIG:-$ROOT_DIR/configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml}"
LDM_CKPT="${LDM_CKPT:-/data/Shenzhen/zhahongli/models/ldmvfi/ldmvfi-vqflow-f32-c256-concat_max.ckpt}"
RVRT_ROOT="${RVRT_ROOT:-/data/Shenzhen/zhahongli/RVRT_flow_ablate}"
RVRT_TASK="${RVRT_TASK:-002_RVRT_videosr_bi_Vimeo_14frames}"
RVRT_CKPT="${RVRT_CKPT:-/data/Shenzhen/zhahongli/RVRT/model_zoo/rvrt/${RVRT_TASK}.pth}"
RVRT_FLOW_MODE="${RVRT_FLOW_MODE:-spynet}"
RVRT_RAFT_VARIANT="${RVRT_RAFT_VARIANT:-large}"
RVRT_RAFT_CKPT="${RVRT_RAFT_CKPT:-}"
RVRT_USE_FLOW_ADAPTER="${RVRT_USE_FLOW_ADAPTER:-0}"
USE_FLOW_INPUTS="${USE_FLOW_INPUTS:-0}"
EVEN_FLOW_BACKEND="${EVEN_FLOW_BACKEND:-farneback}"
EVEN_FLOW_RAFT_VARIANT="${EVEN_FLOW_RAFT_VARIANT:-large}"
EVEN_FLOW_RAFT_CKPT="${EVEN_FLOW_RAFT_CKPT:-}"
USE_DDIM="${USE_DDIM:-1}"
DDIM_STEPS="${DDIM_STEPS:-200}"
DDIM_ETA="${DDIM_ETA:-0}"
SEED="${SEED:-1234}"
USE_RAW_WEIGHTS="${USE_RAW_WEIGHTS:-0}"
ALLOW_INCOMPLETE_CKPT="${ALLOW_INCOMPLETE_CKPT:-0}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs}"

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

mkdir -p "$LOG_ROOT" "$CACHE_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/cache_even_corrector_${STAMP}.log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_ROOT/latest_cache_even_corrector.log"
export PYTHONUNBUFFERED=1
exec > >(tee -a "$LOG_FILE") 2>&1

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU"
  exit 1
fi

echo "root=$ROOT_DIR"
echo "python=$PYTHON_BIN"
echo "log_file=$LOG_FILE"
echo "latest_log=$LOG_ROOT/latest_cache_even_corrector.log"
echo "gpus=$GPU_IDS"
echo "num_gpus=$NUM_GPUS"
echo "master_port=$MASTER_PORT"
echo "data_root=$DATA_ROOT"
echo "cache_root=$CACHE_ROOT"
echo "cache_train=$CACHE_TRAIN"
echo "cache_val=$CACHE_VAL"
echo "max_train_samples=$MAX_TRAIN_SAMPLES"
echo "max_val_samples=$MAX_VAL_SAMPLES"
echo "cache_dtype=$CACHE_DTYPE"
echo "rvrt_flow_mode=$RVRT_FLOW_MODE"
echo "use_flow_inputs=$USE_FLOW_INPUTS"
echo "ddim_steps=$DDIM_STEPS"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

CMD=(
  "$PYTHON_BIN" -u -m torch.distributed.launch
  --nproc_per_node="$NUM_GPUS"
  --master_port="$MASTER_PORT"
  cache_even_corrector_data.py
  --ldm_config "$LDM_CONFIG"
  --ldm_ckpt "$LDM_CKPT"
  --dataset_root_hr "$DATASET_ROOT_HR"
  --dataset_root_lr "$DATASET_ROOT_LR"
  --lr_split_layout "$LR_SPLIT_LAYOUT"
  --train_split "$TRAIN_SPLIT"
  --train_list "$TRAIN_LIST"
  --val_splits "$VAL_SPLITS"
  --val_lists "$VAL_LISTS"
  --cache_train "$CACHE_TRAIN"
  --cache_val "$CACHE_VAL"
  --max_train_samples "$MAX_TRAIN_SAMPLES"
  --max_val_samples "$MAX_VAL_SAMPLES"
  --cache_root "$CACHE_ROOT"
  --cache_dtype "$CACHE_DTYPE"
  --skip_existing "$SKIP_EXISTING"
  --log_interval "$LOG_INTERVAL"
  --rvrt_root "$RVRT_ROOT"
  --rvrt_task "$RVRT_TASK"
  --rvrt_ckpt "$RVRT_CKPT"
  --rvrt_flow_mode "$RVRT_FLOW_MODE"
  --rvrt_raft_variant "$RVRT_RAFT_VARIANT"
  --rvrt_use_flow_adapter "$RVRT_USE_FLOW_ADAPTER"
  --use_flow_inputs "$USE_FLOW_INPUTS"
  --flow_backend "$EVEN_FLOW_BACKEND"
  --flow_raft_variant "$EVEN_FLOW_RAFT_VARIANT"
  --use_ddim "$USE_DDIM"
  --ddim_steps "$DDIM_STEPS"
  --ddim_eta "$DDIM_ETA"
  --seed "$SEED"
)

if [[ -n "$RVRT_RAFT_CKPT" ]]; then
  CMD+=(--rvrt_raft_ckpt "$RVRT_RAFT_CKPT")
fi
if [[ -n "$EVEN_FLOW_RAFT_CKPT" ]]; then
  CMD+=(--flow_raft_ckpt "$EVEN_FLOW_RAFT_CKPT")
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
echo "==== start caching ===="
"${CMD[@]}"
