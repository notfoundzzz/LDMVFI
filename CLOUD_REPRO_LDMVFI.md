# CLOUD_REPRO_LDMVFI

This file is the cloud-first reproduction note for the official LDMVFI project.

Project:

- Paper: https://arxiv.org/pdf/2303.09508
- Code: https://github.com/danier97/LDMVFI


## Goal

The final target is to reproduce LDMVFI on the cloud platform.

The intended execution order is:

1. sync this repo to cloud
2. validate cloud Python/CUDA state
3. run official pretrained evaluation
4. only then attempt stage-1 and stage-2 training


## Recommended cloud layout

Use a stable layout like this:

```text
/data/Shenzhen/zhahongli/
  LDMVFI/
  datasets/ldmvfi/
  benchmarks/
  models/ldmvfi/
  logs/ldmvfi/
```

Suggested paths:

- repo: `/data/Shenzhen/zhahongli/LDMVFI`
- train data root: `/data/Shenzhen/zhahongli/datasets/ldmvfi`
- benchmark root: `/data/Shenzhen/zhahongli/benchmarks`
- checkpoint root: `/data/Shenzhen/zhahongli/models/ldmvfi`
- log root: `/data/Shenzhen/zhahongli/logs/ldmvfi`


## Cloud shell hygiene

Before long runs:

```bash
unset LD_LIBRARY_PATH
unset CUDA_HOME
unset CUDA_PATH
```

If the platform injects conflicting CUDA state, verify Python sees the expected GPU:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu0", torch.cuda.get_device_name(0))
PY
```


## Environment target

Official repo environment:

- Python `3.9.13`
- PyTorch `1.11.0`
- torchvision `0.12.0`
- CUDA toolkit `11.3`
- PyTorch Lightning `1.7.7`

Primary source:

- [environment.yaml](/home/zhahl/LDMVFI/environment.yaml)

Create env on cloud:

```bash
cd /data/Shenzhen/zhahongli/LDMVFI
conda env create -f environment.yaml
conda activate ldmvfi
```

If online install is blocked on cloud, prepare the env or missing wheel packages locally, upload them, then unpack into the target env site-packages.


## Dataset structure

The dataset root must contain lower-case directory names:

```text
/data/Shenzhen/zhahongli/datasets/ldmvfi/
  middlebury_others/
  bvidvc/
    quintuplets/
  vimeo_septuplet/
    sequences/
    sep_trainlist.txt
    sep_testlist.txt

/data/Shenzhen/zhahongli/benchmarks/
  middlebury_others/
  ucf/
  davis90/
  snufilm/
```

Important repo behavior:

- [ldm/data/bvi_vimeo.py](/home/zhahl/LDMVFI/ldm/data/bvi_vimeo.py) loads both `vimeo_septuplet` and `bvidvc` from a shared root
- training configs still contain Windows placeholder paths and must be overridden at runtime


## First milestone: pretrained evaluation

Do this before training.

Put the official released checkpoint at:

```text
/data/Shenzhen/zhahongli/models/ldmvfi/ldmvfi-vqflow-f32-c256-concat_max.ckpt
```

Run:

```bash
cd /data/Shenzhen/zhahongli/LDMVFI
bash run_cloud_eval_ldmvfi.sh
```

This wrapper expects environment variables and provides safe defaults.


## Stage 1: train VQ-FIGAN

Run:

```bash
cd /data/Shenzhen/zhahongli/LDMVFI
bash run_cloud_train_vqflow.sh
```

Adjust these variables if needed:

- `GPU_IDS`
- `DATA_ROOT`
- `LOGDIR`
- `BATCH_SIZE`
- `ACCUM`


## Stage 2: train diffusion U-Net

Run:

```bash
cd /data/Shenzhen/zhahongli/LDMVFI
VQ_CKPT=/data/Shenzhen/zhahongli/models/ldmvfi/vqflow-last.ckpt bash run_cloud_train_ldmvfi.sh
```

Required variable:

- `VQ_CKPT`

Optional variables:

- `GPU_IDS`
- `DATA_ROOT`
- `LOGDIR`
- `BATCH_SIZE`
- `ACCUM`


## Logging

Prefer unbuffered Python plus `tee` logging.

Each wrapper in this repo writes logs into a timestamped file and updates a `latest` symlink.


## Common cloud failure checks

If the traceback does not match local code:

1. verify the actual cloud file contents with `sed -n '1,40p' file.py`
2. verify current branch and commit with `git branch --show-current` and `git log --oneline -n 5`
3. verify imported file path with:

```bash
python -c "import ldm, sys; print(sys.path[:5]); print(ldm.__file__)"
```

4. clear stale caches:

```bash
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
```


## Practical advice

Use cloud to separate three phases:

1. `eval`
   - verify environment, checkpoint loading, dataset layout
2. `stage1`
   - verify autoencoder training stability
3. `stage2`
   - verify diffusion training stability

Do not start with full training until `eval` is already stable.
