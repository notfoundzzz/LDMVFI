# 2026/04/19

## Summary

- Updated the PiSA Dual-LoRA pipeline to use a PiSA-style `pixel baseline + semantic residual` inference rule.
- Made the PiSA Dual-LoRA comparison scripts deterministic by default with `DDIM + eta=0 + seed`.
- Expanded the PiSA Dual-LoRA module wrapper so configured convolution targets such as `in_layers.2` and `out_layers.3` are no longer skipped silently.
- Changed staged PiSA Dual-LoRA training so pixel and semantic optimizer groups are isolated by stage.

## Files

- `ldm/modules/pisa_dual_lora.py`
- `ldm/models/diffusion/rvrt_pisa_dual_lora_ddpm.py`
- `ldm/models/rvrt_pisa_dual_lora_pipeline.py`
- `evaluate_rvrt_pisa_dual_lora.py`
- `check_rvrt_pisa_dual_lora.py`
- `run_cloud_eval_rvrt_pisa_dual_lora.sh`
- `run_cloud_check_rvrt_pisa_dual_lora.sh`

## Manual Test

1. Run `run_cloud_check_rvrt_pisa_dual_lora.sh` twice with the same `SEED`.
   Expected: the reported output differences are identical.
2. Compare `PIXEL_SCALE_A/SEMANTIC_SCALE_A` against `PIXEL_SCALE_B/SEMANTIC_SCALE_B`.
   Expected: the result changes only with the scale values, not with sampling randomness.
3. Start a staged training run and inspect optimizer group learning rates after `semantic_start_step`.
   Expected: `pixel_lora` uses `0.0` learning rate while `semantic_lora` keeps the configured learning rate.
4. Inspect the printed injection count from a training startup.
   Expected: convolution targets in ResBlocks are included instead of being skipped.

## Expected Result

- Dual-LoRA evaluation now matches the intended PiSA-style mixing rule more closely.
- The Dual-LoRA check script can be used as a deterministic gate before looking at averaged metrics.
- Staged training no longer updates pixel adapters during the semantic-only stage.

# 2026/04/19

## Summary

- Tightened the old `RVRT + LDMVFI` evaluation checkpoint loading so incomplete checkpoints fail fast by default.
- Switched the old evaluation pipeline to prefer the checkpoint-restored `rvrt_frontend` when the selected model wrapper owns one.
- Added `use_ema/raw_weights` and deterministic sampling controls to the old evaluation entrypoint.
- Updated the cloud evaluation wrapper so deterministic comparison can be enabled from environment variables.

## Files

- `ldm/models/rvrt_ldmvfi_pipeline.py`
- `evaluate_rvrt_ldmvfi.py`
- `run_cloud_eval_rvrt_ldmvfi.sh`

## Manual Test

1. Evaluate the same checkpoint twice with the same `SEED`.
   Expected: the metrics and saved images are identical.
2. Evaluate once with default EMA weights and once with `USE_RAW_WEIGHTS=1`.
   Expected: both runs complete and can be compared directly.
3. Evaluate a checkpoint produced with `rvrt_train_mode != frozen`.
   Expected: the log reports `Evaluation RVRT source: checkpoint rvrt_frontend`.
4. Evaluate an intentionally incomplete checkpoint.
   Expected: evaluation fails unless `ALLOW_INCOMPLETE_CKPT=1` is set.

## Expected Result

- Old stitched evaluation is reproducible by default.
- The evaluation path now measures the same RVRT frontend that was stored in the checkpoint whenever that frontend was trainable.
- Wrong or partial checkpoints no longer pass silently in the default path.

# 2026/04/18

## 变更摘要

- 修复 `RVRT + LDMVFI` 与旧 `Dual-LoRA` 评测 pipeline 中的条件帧归一化不一致问题。
- 新增一条独立的 `RVRT + PiSA Dual-LoRA + LDMVFI` 训练线，不复用旧 `dual_lora.py` 的 block 划分。
- 新增独立评测脚本与检查脚本，用于比较不同 `pixel_scale / semantic_scale` 下的输出张量差值。

## 关键文件

- `ldm/models/rvrt_ldmvfi_pipeline.py`
- `ldm/models/rvrt_dual_lora_pipeline.py`
- `ldm/modules/pisa_dual_lora.py`
- `ldm/models/diffusion/rvrt_pisa_dual_lora_ddpm.py`
- `ldm/models/rvrt_pisa_dual_lora_pipeline.py`
- `evaluate_rvrt_pisa_dual_lora.py`
- `check_rvrt_pisa_dual_lora.py`
- `configs/ldm/rvrt-pisa-dual-lora-stsr-x4.yaml`

## 人工测试方法

1. 运行 `run_cloud_eval_rvrt_ldmvfi.sh` 或旧 `run_cloud_eval_rvrt_dual_lora.sh`，确认评测可以正常完成。
2. 使用 `run_cloud_train_rvrt_pisa_dual_lora.sh` 启动新训练线，确认日志中打印出 PiSA Dual-LoRA 注入模块数量。
3. 使用 `run_cloud_eval_rvrt_pisa_dual_lora.sh` 对同一个 checkpoint 做评测。
4. 使用 `run_cloud_check_rvrt_pisa_dual_lora.sh` 对同一个 checkpoint 做检查。

## 预期结果

- 修复后的评测路径应与训练路径使用一致的 RVRT 输出归一化。
- 新 checkpoint 中应同时包含 `pixel_lora_*` 与 `semantic_lora_*` 权重。
- `run_cloud_check_rvrt_pisa_dual_lora.sh` 在不同 scale 设置下，应输出非零的 `mean_abs_diff` 或 `max_abs_diff`。
