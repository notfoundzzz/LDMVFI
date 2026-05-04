# Paper Figure Workflow

This note records the reproducible figure workflow for the RVRT + LDMVFI + even-frame corrector experiments.

## 1. Draft numeric figures

Use the built-in draft metrics from the current experiment notes:

```bash
python generate_paper_figures.py --output-dir figures/paper_draft
```

Outputs:

- `fig3_method_framework.{png,svg}`
- `fig4_odd_even_gap.{png,svg}`
- `fig5_corrector_ablation.{png,svg}`
- `fig6_candidate_oracle.{png,svg}`
- `table_full_chain_results.{png,svg,csv,md,tex}`

## 2. Formal numeric figures after full evaluation

After baseline and best-corrector full-chain evaluations finish, regenerate with their `_summaries` directories.
Use explicit diagnosis JSON filenames. Do not choose the latest timestamped file automatically, because RAFT and Farneback runs share the same filename prefix.

For full-validation diagnosis, prefer the parallel launcher:

```bash
cd /data/Shenzhen/zhahongli/LDMVFI && DIAG_STAMP=raft_full_val GPU_IDS=4,5 CACHE_ROOT=/cache/zhahongli/even_corrector_spynet_ddim200_raftlarge_10000_valfull SPLITS=slow_test,medium_test,fast_test MAX_SAMPLES=0 METRICS=PSNR,SSIM,LPIPS bash run_cloud_diagnose_even_cache_parallel.sh
cd /data/Shenzhen/zhahongli/LDMVFI && DIAG_STAMP=farneback_full_val GPU_IDS=6,7 CACHE_ROOT=/cache/zhahongli/even_corrector_spynet_ddim200_farneback_valfull SPLITS=slow_test,medium_test,fast_test MAX_SAMPLES=0 METRICS=PSNR,SSIM,LPIPS bash run_cloud_diagnose_even_cache_parallel.sh

RAFT_DIAG_JSON=/data/Shenzhen/zhahongli/LDMVFI/diagnostics/even_corrector_cache/diagnose_even_cache_raft_full_val_merged.json
FARNE_DIAG_JSON=/data/Shenzhen/zhahongli/LDMVFI/diagnostics/even_corrector_cache/diagnose_even_cache_farneback_full_val_merged.json
```

If the ablation values have been recomputed on full validation, prepare a CSV:

```csv
method,PSNR,SSIM
LDMVFI pred,30.7474,0.9391
Farneback fusion,30.8399,0.9400
RAFT fusion,30.8599,0.9404
RAFT fusion corrector,30.8607,0.9404
```

```bash
python generate_paper_figures.py \
  --output-dir figures/paper \
  --baseline-summary-dir /data/Shenzhen/zhahongli/LDMVFI/eval_full_sr_then_vfi_baseline \
  --corrector-summary-dir /data/Shenzhen/zhahongli/LDMVFI/eval_full_sr_then_vfi_raft_fusion_edge_step9000 \
  --raft-diagnosis-json "$RAFT_DIAG_JSON" \
  --farneback-diagnosis-json "$FARNE_DIAG_JSON" \
  --ablation-results /data/Shenzhen/zhahongli/LDMVFI/experiments/paper_ablation_full_val.csv
```

If a diagnosis JSON or ablation CSV is not available yet, omit that argument. The script will use the built-in draft diagnosis or ablation values.
The main result table only shows metrics that exist in the supplied summaries; LPIPS columns are hidden automatically when LPIPS is not present.

## 3. Qualitative comparison figures

First save prediction images from the evaluation script. The full-chain evaluator writes `output_im1.png` through `output_im7.png`; the key interpolated middle frame is `output_im4.png`.

Baseline qualitative save:

```bash
cd /data/Shenzhen/zhahongli/LDMVFI && GPU_ID=2 OUT_DIR=/data/Shenzhen/zhahongli/LDMVFI/qual_sr_then_vfi_baseline_fast SAVE_IMAGES=1 SAVE_SR_IMAGES=1 SAVE_MAX_SAMPLES=8 SPLITS=fast_test MAX_SAMPLES=8 METRICS=PSNR,SSIM,LPIPS EVAL_PIPELINE=sr_then_vfi RVRT_FLOW_MODE=spynet bash run_cloud_eval_rvrt_ldmvfi.sh
```

Best corrector qualitative save:

```bash
cd /data/Shenzhen/zhahongli/LDMVFI && GPU_ID=3 OUT_DIR=/data/Shenzhen/zhahongli/LDMVFI/qual_sr_then_vfi_raft_edge_step9000_fast SAVE_IMAGES=1 SAVE_SR_IMAGES=1 SAVE_MAX_SAMPLES=8 SPLITS=fast_test MAX_SAMPLES=8 METRICS=PSNR,SSIM,LPIPS EVAL_PIPELINE=sr_then_vfi RVRT_FLOW_MODE=spynet EVEN_CORRECTOR_CKPT=/data/Shenzhen/zhahongli/LDMVFI/experiments/even_corrector_raftlarge_fusion_edge005_10000_valfull/step_009000_even_corrector.pth EVEN_CORRECTOR_HIDDEN_CHANNELS=64 EVEN_CORRECTOR_NUM_BLOCKS=8 EVEN_CORRECTOR_MAX_RESIDUE=0.25 EVEN_CORRECTOR_MODE=fusion EVEN_CORRECTOR_USE_FLOW_INPUTS=1 EVEN_CORRECTOR_FLOW_BACKEND=raft EVEN_CORRECTOR_FLOW_RAFT_VARIANT=large EVEN_CORRECTOR_FLOW_RAFT_CKPT=/data/Shenzhen/zhahongli/models/raft/raft_large_C_T_SKHT_V2-ff5fadd5.pth bash run_cloud_eval_rvrt_ldmvfi.sh
```

Then stitch a paper figure for a selected sample:

```bash
python generate_demo_figure.py \
  --sample-relpath 00001/0625 \
  --split fast_test \
  --dataset-root-lr /data/Shenzhen/zzff/STVSR/data/vimeo_septuplet/sequences_LR \
  --dataset-root-hr /data/Shenzhen/zzff/STVSR/data/vimeo_septuplet/sequences \
  --method "SR-VFI baseline=/data/Shenzhen/zhahongli/LDMVFI/qual_sr_then_vfi_baseline_fast" \
  --method "RAFT fusion corrector=/data/Shenzhen/zhahongli/LDMVFI/qual_sr_then_vfi_raft_edge_step9000_fast" \
  --target-frame 4 \
  --show-error-maps \
  --crop 80,40,96,96 \
  --output figures/paper/qual_fast_00001_0625_im4.png
```

Change `--sample-relpath` and `--crop` after inspecting saved predictions. Prefer fast-motion, edge, and texture-heavy samples.

For fast screening, batch-generate several candidates first:

```bash
python batch_generate_demo_figures.py \
  --split fast_test \
  --dataset-root-lr /data/Shenzhen/zzff/STVSR/data/vimeo_septuplet/sequences_LR \
  --dataset-root-hr /data/Shenzhen/zzff/STVSR/data/vimeo_septuplet/sequences \
  --baseline-root /data/Shenzhen/zhahongli/LDMVFI/qual_sr_then_vfi_baseline_fast \
  --corrector-root /data/Shenzhen/zhahongli/LDMVFI/qual_sr_then_vfi_raft_edge_step9000_fast \
  --output-dir /data/Shenzhen/zhahongli/LDMVFI/figures/paper_demo_candidates \
  --target-frame 4 \
  --max-samples 8 \
  --auto-crop-count 2 \
  --auto-crop-size 96x96 \
  --auto-crop-mode improvement
```

Zoom regions use HR-coordinate crop boxes in `x,y,w,h` format. The same box is applied to GT and all method outputs.
Manual crops can be supplied repeatedly, for example `--crop 80,40,96,96 --crop 220,80,96,96`.

For the final paper qualitative figure, use the compact layout without error maps:

```bash
python generate_paper_qual_demo.py \
  --sample-relpath 00001/0625 \
  --split fast_test \
  --target-frame 4 \
  --dataset-root-hr /data/Shenzhen/zzff/STVSR/data/vimeo_septuplet/sequences \
  --baseline-root /data/Shenzhen/zhahongli/LDMVFI/qual_sr_then_vfi_baseline_fast \
  --corrector-root /data/Shenzhen/zhahongli/LDMVFI/qual_sr_then_vfi_raft_corrector_fast \
  --crop "Pen tip=176,104,112,112" \
  --crop "Little finger=120,20,128,128" \
  --output /data/Shenzhen/zhahongli/LDMVFI/figures/paper/qual_paper_fast_00001_0625_im4.png
```

The selected crops intentionally focus on the pen tip and the little-finger region. The pen-tip crop is visually useful because it shows both local correction and a remaining motion-boundary artifact of the RAFT-based candidate.
