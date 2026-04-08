# CODEX_HANDOFF_LDMVFI

## Goal

This repository is being used for an undergraduate thesis project.

The original baseline is:

- `LDMVFI`
- task: video frame interpolation
- input: previous frame + next frame
- output: middle frame

The current thesis target is:

- space-time super-resolution based on `LDMVFI`
- preferred main route from senior guidance:
  - `LR prev + LR next -> video SR frontend -> SR prev + SR next -> LDMVFI -> HR middle`

The preferred frontend is:

- `RVRT`


## Current Repositories

- LDMVFI repo:
  - `/home/zhahl/LDMVFI`
- RVRT repo cloned locally:
  - `/home/zhahl/RVRT`
- RVRT commit used for integration reference:
  - `a5d406c`


## Current Git Status

Important commits already created in the user's fork:

- `69c789c`
  - Add STSR training and evaluation scaffolding
- `e8dcbfb`
  - Add cloud and environment helper scripts
- `0c600c0`
  - Add logfile output to helper scripts
- `200011f`
  - Add RVRT plus LDMVFI pipeline scaffolding

One local commit exists but may or may not be pushed yet, depending on when this file is read:

- `4bde8a6`
  - Add RVRT runtime dependencies to env checks

If needed, verify current remote state with:

```bash
git -C /home/zhahl/LDMVFI log --oneline -n 10
git -C /home/zhahl/LDMVFI remote -v
```


## What Has Been Implemented

### 1. Baseline LDMVFI reproduction support

Added/updated helper files:

- [check_env_ldmvfi.py](/home/zhahl/LDMVFI/check_env_ldmvfi.py)
- [setup_ldmvfi_env.sh](/home/zhahl/LDMVFI/setup_ldmvfi_env.sh)
- [pack_ldmvfi_env.sh](/home/zhahl/LDMVFI/pack_ldmvfi_env.sh)
- [run_pack_ldmvfi_background.sh](/home/zhahl/LDMVFI/run_pack_ldmvfi_background.sh)
- [run_cloud_check_env.sh](/home/zhahl/LDMVFI/run_cloud_check_env.sh)
- [run_cloud_eval_ldmvfi.sh](/home/zhahl/LDMVFI/run_cloud_eval_ldmvfi.sh)
- [run_cloud_train_vqflow.sh](/home/zhahl/LDMVFI/run_cloud_train_vqflow.sh)
- [run_cloud_train_ldmvfi.sh](/home/zhahl/LDMVFI/run_cloud_train_ldmvfi.sh)

All helper scripts write timestamped logs and update `latest_*.log` symlinks.


### 2. Direct STSR-in-LDMVFI branch

This was implemented before the senior guidance clarified the preferred route.
It is still useful as a side branch or ablation, but is no longer the preferred main line.

Files:

- [ldm/data/stsr.py](/home/zhahl/LDMVFI/ldm/data/stsr.py)
- [ldm/data/testsets_stsr.py](/home/zhahl/LDMVFI/ldm/data/testsets_stsr.py)
- [ldm/models/stsr_baseline.py](/home/zhahl/LDMVFI/ldm/models/stsr_baseline.py)
- [ldm/models/diffusion/stsr_ddpm.py](/home/zhahl/LDMVFI/ldm/models/diffusion/stsr_ddpm.py)
- [evaluate_stsr.py](/home/zhahl/LDMVFI/evaluate_stsr.py)

Configs:

- [configs/ldm/stsr-x2-resizecond.yaml](/home/zhahl/LDMVFI/configs/ldm/stsr-x2-resizecond.yaml)
- [configs/ldm/stsr-x2-cond-bridge.yaml](/home/zhahl/LDMVFI/configs/ldm/stsr-x2-cond-bridge.yaml)
- [configs/ldm/stsr-x2-nodiff-baseline.yaml](/home/zhahl/LDMVFI/configs/ldm/stsr-x2-nodiff-baseline.yaml)

Wrappers:

- [run_train_stsr_resizecond.sh](/home/zhahl/LDMVFI/run_train_stsr_resizecond.sh)
- [run_train_stsr_cond_bridge.sh](/home/zhahl/LDMVFI/run_train_stsr_cond_bridge.sh)
- [run_train_stsr_nodiff.sh](/home/zhahl/LDMVFI/run_train_stsr_nodiff.sh)
- [run_eval_stsr.sh](/home/zhahl/LDMVFI/run_eval_stsr.sh)
- [run_cloud_eval_stsr.sh](/home/zhahl/LDMVFI/run_cloud_eval_stsr.sh)


### 3. Preferred main route: RVRT + LDMVFI

This is the current main recommendation.

Files added:

- [RVRT_LDMVFI_PLAN.md](/home/zhahl/LDMVFI/RVRT_LDMVFI_PLAN.md)
- [ldm/models/rvrt_frontend.py](/home/zhahl/LDMVFI/ldm/models/rvrt_frontend.py)
- [ldm/models/rvrt_ldmvfi_pipeline.py](/home/zhahl/LDMVFI/ldm/models/rvrt_ldmvfi_pipeline.py)
- [evaluate_rvrt_ldmvfi.py](/home/zhahl/LDMVFI/evaluate_rvrt_ldmvfi.py)
- [run_cloud_eval_rvrt_ldmvfi.sh](/home/zhahl/LDMVFI/run_cloud_eval_rvrt_ldmvfi.sh)

What this route currently supports:

- `bicubic + LDMVFI`
- `RVRT + LDMVFI`

Current design:

- keep RVRT repo external
- dynamically add RVRT repo root to `sys.path`
- load official RVRT model and official LDMVFI model
- use RVRT to super-resolve the neighboring frames
- feed the restored neighboring frames into original LDMVFI sampling

This is currently an inference-time integration scaffold, not a joint training pipeline.


## Important Code Behavior

### LDMVFI assumptions

Key point:

- original `LDMVFI` decoder is frame-aided
- it strongly benefits from higher-quality neighboring frames
- this is why `RVRT + LDMVFI` is preferred over aggressive end-to-end STSR rewrites

Relevant files:

- [ldm/models/autoencoder.py](/home/zhahl/LDMVFI/ldm/models/autoencoder.py)
- [ldm/models/diffusion/ddpm.py](/home/zhahl/LDMVFI/ldm/models/diffusion/ddpm.py)


### Local dependency path handling

Because cloud installation may be offline, a local path helper was added:

- [ldm/path_setup.py](/home/zhahl/LDMVFI/ldm/path_setup.py)

This is used so the repo can find:

- `src/taming-transformers`
- `src/clip`
- external repo roots when needed

It was necessary because editable installs were unreliable or unavailable on cloud.


## Confirmed Cloud Issues and Fixes

### 1. `taming` import failure

Observed on cloud:

- `ModuleNotFoundError: No module named 'taming'`

Fix used:

- copy local source trees into:
  - `/data/Shenzhen/zhahongli/LDMVFI/src/taming-transformers`
  - `/data/Shenzhen/zhahongli/LDMVFI/src/clip`

The repo now tries to resolve them through `ldm/path_setup.py`.


### 2. LPIPS weight missing

Observed on cloud:

- missing:
  - `metrics/lpips/weights/v0.1/alex.pth`

This file was downloaded locally to:

- `/home/zhahl/LDMVFI/metrics/lpips/weights/v0.1/alex.pth`

Cloud should place it at:

- `/data/Shenzhen/zhahongli/LDMVFI/metrics/lpips/weights/v0.1/alex.pth`


### 3. UCF data naming mismatch

The downloaded UCF package had this naming:

- `frame0.png`
- `frame1.png`
- `frame2.png`
- `frame3.png`
- `framet.png`

But `evaluate.py` class `Ucf` expects:

- `frame_00.png`
- `frame_01_gt.png`
- `frame_02.png`

The working fix was to create symlinks:

- `frame_00.png -> frame1.png`
- `frame_01_gt.png -> framet.png`
- `frame_02.png -> frame2.png`

Why this mapping:

- `ldm/data/testsets_vqm.py`
- class `Ucf101_triplet`
- confirms the meaningful pair is `frame1`, `frame2`, and GT is `framet`


### 4. Cloud shell visibility

Cloud shell only shows about one page, so all helper scripts were updated to log to files.

Use:

```bash
tail -f /data/Shenzhen/zhahongli/LDMVFI/logs/latest_eval.log
tail -f /data/Shenzhen/zhahongli/LDMVFI/logs/latest_check_env.log
tail -f /data/Shenzhen/zhahongli/LDMVFI/logs/latest_eval_stsr.log
tail -f /data/Shenzhen/zhahongli/LDMVFI/logs/latest_eval_rvrt_ldmvfi.log
```


## Current Environment Notes

The `ldmvfi` environment was built around:

- Python `3.9`
- PyTorch `1.11.0`
- torchvision `0.12.0`
- CUDA toolkit `11.3`
- PyTorch Lightning `1.7.7`

Known compatibility pins that mattered:

- `pip<24.1`
- `setuptools<81`
- `numpy<2`
- `torchmetrics==0.10.3`
- `mkl=2023.1.0`
- `intel-openmp=2023.1.0`


### Additional runtime dependencies for RVRT route

These should be present:

- `addict`
- `ftfy`
- `ninja`
- `scikit-image`

Reason:

- `ftfy` is needed by `clip`
- `ninja` helps RVRT compile its CUDA extension on first import
- `addict` and `scikit-image` are RVRT requirements

These were added to:

- [requirements-pip.txt](/home/zhahl/LDMVFI/requirements-pip.txt)
- [check_env_ldmvfi.py](/home/zhahl/LDMVFI/check_env_ldmvfi.py)

If cloud cannot download packages, prepare local files and copy them to cloud.


## Offline Package Advice

Cloud may not allow direct downloads.

In that case, prepare locally:

- `addict`
- `ftfy`
- `ninja`
- `scikit-image`

Suggested local command:

```bash
python -m pip download --dest /home/zhahl/LDMVFI/offline_pkgs_rvrt addict ftfy ninja scikit-image
```

Then copy the folder to cloud, for example:

```text
/data/Shenzhen/zhahongli/LDMVFI/offline_pkgs_rvrt
```

Then install on cloud:

```bash
/data/Shenzhen/zhahongli/envs/ldmvfi/bin/python -m pip install --no-index --find-links ./offline_pkgs_rvrt addict ftfy ninja scikit-image
```


## Cloud Paths Used So Far

Repository:

- `/data/Shenzhen/zhahongli/LDMVFI`

Environment:

- `/data/Shenzhen/zhahongli/envs/ldmvfi`

LDMVFI checkpoint:

- `/data/Shenzhen/zhahongli/models/ldmvfi/ldmvfi-vqflow-f32-c256-concat_max.ckpt`

Suggested RVRT repo:

- `/data/Shenzhen/zhahongli/RVRT`

Suggested RVRT checkpoint:

- `/data/Shenzhen/zhahongli/RVRT/model_zoo/rvrt/002_RVRT_videosr_bi_Vimeo_14frames.pth`


## What Has Been Successfully Confirmed

- baseline `evaluate.py` can load the official `LDMVFI` checkpoint
- cloud GPU environment works on V100
- UCF evaluation path reached actual inference
- remaining failures were around auxiliary assets or naming, not core model loading


## Recommended Next Steps

### Short term

1. ensure the local commit `4bde8a6` is pushed if not already
2. ensure cloud has:
   - `src/taming-transformers`
   - `src/clip`
   - LPIPS `alex.pth`
   - offline-installed RVRT runtime dependencies
3. re-run:
   - official baseline evaluation
4. run:
   - `bicubic + LDMVFI`
5. then run:
   - `RVRT + LDMVFI`


### Experiment order

Recommended order:

1. official `LDMVFI` baseline
2. `bicubic + LDMVFI`
3. `RVRT + LDMVFI`
4. only after that, revisit deeper end-to-end STSR variants if still needed


## Suggested Commands

### Baseline cloud eval

```bash
cd /data/Shenzhen/zhahongli/LDMVFI
unset LD_LIBRARY_PATH CUDA_HOME CUDA_PATH
PYTHON_BIN=/data/Shenzhen/zhahongli/envs/ldmvfi/bin/python DATASET=Ucf DATA_ROOT=/data/Shenzhen/zhahongli/benchmarks bash run_cloud_eval_ldmvfi.sh
```

### RVRT route with bicubic frontend

```bash
cd /data/Shenzhen/zhahongli/LDMVFI
unset LD_LIBRARY_PATH CUDA_HOME CUDA_PATH
PYTHON_BIN=/data/Shenzhen/zhahongli/envs/ldmvfi/bin/python \
SR_MODE=bicubic \
DATASET_ROOT=/data/Shenzhen/zhahongli/benchmarks/ucf \
bash run_cloud_eval_rvrt_ldmvfi.sh
```

### RVRT route with RVRT frontend

```bash
cd /data/Shenzhen/zhahongli/LDMVFI
unset LD_LIBRARY_PATH CUDA_HOME CUDA_PATH
PYTHON_BIN=/data/Shenzhen/zhahongli/envs/ldmvfi/bin/python \
SR_MODE=rvrt \
RVRT_ROOT=/data/Shenzhen/zhahongli/RVRT \
RVRT_CKPT=/data/Shenzhen/zhahongli/RVRT/model_zoo/rvrt/002_RVRT_videosr_bi_Vimeo_14frames.pth \
DATASET_ROOT=/data/Shenzhen/zhahongli/benchmarks/ucf \
bash run_cloud_eval_rvrt_ldmvfi.sh
```


## Final Guidance For The Next Codex

- Do not assume cloud can download packages.
- Prefer local preparation plus upload.
- Prefer verifying actual cloud file contents if behavior looks inconsistent.
- Keep the `RVRT + LDMVFI` route as the thesis main line.
- Treat the direct STSR rewrite branch as secondary or ablation.
- Avoid large architectural rewrites before the bicubic and RVRT frontends are both evaluated.
