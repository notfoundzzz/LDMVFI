# RVRT + LDMVFI Plan

## Goal

Use a video super-resolution frontend to recover higher-quality neighboring frames before feeding them into the original LDMVFI frame interpolation model.

Pipeline:

`LR prev + LR next -> RVRT -> SR prev + SR next -> LDMVFI -> HR middle`

## Why this route

- LDMVFI's decoder is frame-aided and strongly benefits from high-quality neighboring frames.
- Directly turning LDMVFI into an end-to-end STSR model is higher risk.
- RVRT already provides pretrained Vimeo-based video SR checkpoints, including a Vimeo-to-Vid4 evaluation setting.

## Current implementation

The repository now includes:

- [ldm/models/rvrt_frontend.py](/home/zhahl/LDMVFI/ldm/models/rvrt_frontend.py)
  - `BicubicVideoSR`
  - `RVRTVideoSR`
- [ldm/models/rvrt_ldmvfi_pipeline.py](/home/zhahl/LDMVFI/ldm/models/rvrt_ldmvfi_pipeline.py)
  - `RVRTLDMVFIPipeline`
- [evaluate_rvrt_ldmvfi.py](/home/zhahl/LDMVFI/evaluate_rvrt_ldmvfi.py)
  - evaluation on triplet folders with either `bicubic` or `rvrt` frontend

## External dependency

RVRT is not vendored into this repository. It is expected at a separate path, for example:

```text
/data/Shenzhen/zhahongli/RVRT
```

The code imports RVRT dynamically through its repository root.

## Checkpoint expectation

Recommended RVRT task for the first prototype:

- `002_RVRT_videosr_bi_Vimeo_14frames`

Checkpoint path example:

```text
/data/Shenzhen/zhahongli/RVRT/model_zoo/rvrt/002_RVRT_videosr_bi_Vimeo_14frames.pth
```

LDMVFI checkpoint remains:

```text
/data/Shenzhen/zhahongli/models/ldmvfi/ldmvfi-vqflow-f32-c256-concat_max.ckpt
```

## Notes

- RVRT compiles its `deform_attn` CUDA op on first import via `torch.utils.cpp_extension.load(...)`.
- First import can take time on the cloud machine.
- The current pipeline prototype uses the 2-frame pair directly as RVRT input. This is a pragmatic first step, even though RVRT was trained with longer clips.
- `bicubic` frontend is available as the main baseline against the RVRT frontend.

## Suggested next experiments

1. `bicubic + LDMVFI`
2. `RVRT + LDMVFI`
3. compare on `Vimeo90K test` and `Vid4`
