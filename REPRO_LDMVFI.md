# LDMVFI Reproduction Notes

This note is a practical reproduction guide for the official LDMVFI paper and code:

- Paper: https://arxiv.org/pdf/2303.09508
- Code: https://github.com/danier97/LDMVFI

It is tailored to the current machine state observed on April 2, 2026:

- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`, `6 GB`
- Active Python env: `/home/zhahl/miniconda3/envs/jit-local`
- Current env status: missing `pytorch_lightning`, `omegaconf`, `cupy`


## What "reproduce" means here

There are three distinct targets:

1. `Official evaluation reproduction`
   - Run the released pretrained checkpoint on the public test sets.
   - This is the most realistic first target on this machine.
2. `Training-path reproduction`
   - Reproduce the official two-stage training pipeline:
     - Stage 1: `VQ-FIGAN`
     - Stage 2: diffusion `U-Net`
3. `Paper-equivalent full training`
   - This is the closest to the paper, but not realistic on a `6 GB` GPU without major compromises.


## Facts confirmed from the paper and code

From the paper:

- Training data: `Vimeo90k-septuplet` + `BVI-DVC`
- Training samples: `64612` Vimeo triplets + `17600` BVI-DVC triplets
- Training crop size: `256 x 256`
- Augmentation: random flip + temporal reverse
- DDIM sampling: `200` steps
- Optimizer:
  - VQ-FIGAN: `Adam`
  - diffusion U-Net: `AdamW`
- Initial learning rate:
  - VQ-FIGAN: `1e-5`
  - U-Net: `1e-6`
- Hardware used in the paper: `NVIDIA RTX 3090`
- Training length:
  - VQ-FIGAN: around `70` epochs
  - U-Net: around `60` epochs

From the official repository:

- Official environment file: [environment.yaml](/home/zhahl/LDMVFI/environment.yaml)
- Training entrypoint: [main.py](/home/zhahl/LDMVFI/main.py)
- Evaluation entrypoint: [evaluate.py](/home/zhahl/LDMVFI/evaluate.py)
- Autoencoder config: [configs/autoencoder/vqflow-f32.yaml](/home/zhahl/LDMVFI/configs/autoencoder/vqflow-f32.yaml)
- Diffusion config: [configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml](/home/zhahl/LDMVFI/configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml)
- Training dataset loader: [ldm/data/bvi_vimeo.py](/home/zhahl/LDMVFI/ldm/data/bvi_vimeo.py)


## Recommended order

Do this in order:

1. Create the official environment as closely as possible.
2. Download the official pretrained checkpoint.
3. Prepare one public test set and run evaluation.
4. Only after that, decide whether to attempt stage-1 and stage-2 training.


## Environment

The official environment is old and specific:

- Python `3.9.13`
- PyTorch `1.11.0`
- torchvision `0.12.0`
- CUDA toolkit `11.3`
- PyTorch Lightning `1.7.7`

Use a dedicated env instead of `jit-local`.

Example:

```bash
cd /home/zhahl/LDMVFI
conda env create -f environment.yaml
conda activate ldmvfi
```

If `conda env create` fails because of `cupy`, install the rest first, then add the CUDA-matched CuPy wheel manually.


## Dataset layout

The repo is strict about lower-case dataset folder names.

Expected root layout:

```text
<data_dir>/
  middlebury_others/
    input/
    gt/
  ucf101/
  davis90/
  snufilm/
  bvidvc/
    quintuplets/
  vimeo_septuplet/
    sequences/
    sep_trainlist.txt
    sep_testlist.txt
```

Important code-level details:

- `Vimeo90k_triplet` expects `im3.png`, `im4.png`, `im5.png` under each sequence folder.
- `BVIDVC_triplet` expects `bvidvc/quintuplets/<id>/quintuplet.png`.
- Training config points `db_dir` to the common root that contains both `vimeo_septuplet/` and `bvidvc/`.


## Official pretrained evaluation

The cleanest first milestone is to evaluate the released pretrained model.

1. Download the official checkpoint from the Google Drive link in [README.md](/home/zhahl/LDMVFI/README.md).
2. Put it somewhere stable, for example:

```text
/home/zhahl/models/ldmvfi/ldmvfi-vqflow-f32-c256-concat_max.ckpt
```

3. Prepare one dataset, for example `Middlebury_others`.
4. Run:

```bash
cd /home/zhahl/LDMVFI
python evaluate.py \
  --config configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml \
  --ckpt /home/zhahl/models/ldmvfi/ldmvfi-vqflow-f32-c256-concat_max.ckpt \
  --dataset Middlebury_others \
  --metrics PSNR SSIM LPIPS \
  --data_dir /path/to/data \
  --out_dir eval_results/ldmvfi-vqflow-f32-c256-concat_max \
  --use_ddim
```

For FloLPIPS after frames are written:

```bash
cd /home/zhahl/LDMVFI
python evaluate_vqm.py \
  --exp ldmvfi-vqflow-f32-c256-concat_max \
  --dataset Middlebury_others \
  --metrics FloLPIPS \
  --data_dir /path/to/data \
  --out_dir eval_results/ldmvfi-vqflow-f32-c256-concat_max
```


## Training pipeline

The project trains in two stages.

### Stage 1: VQ-FIGAN

Official command:

```bash
cd /home/zhahl/LDMVFI
python main.py --base configs/autoencoder/vqflow-f32.yaml -t --gpus 0,
```

What you must override for a real run:

- `data.params.train.params.db_dir`
- `data.params.validation.params.db_dir`

Example:

```bash
cd /home/zhahl/LDMVFI
python main.py \
  --base configs/autoencoder/vqflow-f32.yaml \
  -t \
  --gpus 0, \
  --data.params.train.params.db_dir /path/to/data \
  --data.params.validation.params.db_dir /path/to/data/vimeo_septuplet
```

### Stage 2: diffusion U-Net

Official command:

```bash
cd /home/zhahl/LDMVFI
python main.py --base configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml -t --gpus 0,
```

What you must override for a real run:

- `data.params.train.params.db_dir`
- `data.params.validation.params.db_dir`
- `model.params.first_stage_config.params.ckpt_path`

Example:

```bash
cd /home/zhahl/LDMVFI
python main.py \
  --base configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml \
  -t \
  --gpus 0, \
  --data.params.train.params.db_dir /path/to/data \
  --data.params.validation.params.db_dir /path/to/data/vimeo_septuplet \
  --model.params.first_stage_config.params.ckpt_path /path/to/vqflow-last.ckpt
```


## Current-machine feasibility

This machine is not paper-equivalent.

Main constraint:

- Paper used `RTX 3090`
- This machine has `RTX 3060 Laptop GPU 6 GB`

Practical consequence:

- Official diffusion training config uses `batch_size: 64`
- Official autoencoder training config uses `batch_size: 10`
- These settings are not realistic on `6 GB` VRAM

If you still want a local smoke-training run, start with:

- VQ-FIGAN:
  - `data.params.batch_size=1`
  - `lightning.trainer.accumulate_grad_batches=10`
- diffusion U-Net:
  - `data.params.batch_size=1`
  - `lightning.trainer.accumulate_grad_batches=32`

Example smoke command for stage 1:

```bash
cd /home/zhahl/LDMVFI
python main.py \
  --base configs/autoencoder/vqflow-f32.yaml \
  -t \
  --gpus 0, \
  --data.params.batch_size 1 \
  --lightning.trainer.accumulate_grad_batches 10 \
  --data.params.train.params.db_dir /path/to/data \
  --data.params.validation.params.db_dir /path/to/data/vimeo_septuplet
```

Example smoke command for stage 2:

```bash
cd /home/zhahl/LDMVFI
python main.py \
  --base configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml \
  -t \
  --gpus 0, \
  --data.params.batch_size 1 \
  --lightning.trainer.accumulate_grad_batches 32 \
  --data.params.train.params.db_dir /path/to/data \
  --data.params.validation.params.db_dir /path/to/data/vimeo_septuplet \
  --model.params.first_stage_config.params.ckpt_path /path/to/vqflow-last.ckpt
```

This should be treated as a feasibility run, not a faithful paper reproduction.


## Recommended success criterion

Use this progression:

1. `Pass`
   - env builds
   - official pretrained checkpoint loads
   - one dataset evaluates end-to-end
2. `Better`
   - all public test sets evaluate
   - metrics are close to the paper tables
3. `Expensive`
   - stage-1 training runs stably
   - stage-2 training runs stably
4. `Paper-like`
   - training is moved to a `24 GB` class GPU
   - metrics are compared against the paper tables


## Immediate blockers on this machine

In the current active env `jit-local`, these required packages are missing:

- `pytorch_lightning`
- `omegaconf`
- `cupy`

So do not try to run LDMVFI from `jit-local` as-is.


## Useful source links

- Paper PDF: https://arxiv.org/pdf/2303.09508
- Official repo: https://github.com/danier97/LDMVFI
- Repo README: [README.md](/home/zhahl/LDMVFI/README.md)
- Official env: [environment.yaml](/home/zhahl/LDMVFI/environment.yaml)
- Evaluation entry: [evaluate.py](/home/zhahl/LDMVFI/evaluate.py)
- Dataset loader: [bvi_vimeo.py](/home/zhahl/LDMVFI/ldm/data/bvi_vimeo.py)
