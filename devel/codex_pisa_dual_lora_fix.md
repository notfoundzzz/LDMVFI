# 2026/04/22

## 图像重建损失引导
- 在纯 `RVRT + LDMVFI + flow` 训练类中增加可选的图像空间重建损失，默认采用 `L1`。
- 新损失形式为：`loss = diffusion_loss + image_recon_loss_weight * recon_loss`，用于让训练方向更贴近 `PSNR/SSIM`。
- 当前只改纯 flow 主线，不影响 `PiSA / LoRA` 训练分支。

## 人工测试方式

1. 使用更新后的 `configs/ldm/rvrt-flow-guided-stsr-x4.yaml` 启动一条短训练。
2. 观察日志中出现：
   - `Image recon guidance enabled: True`
   - `type=l1, weight=0.100`
3. 训练完成后运行 small eval，对比旧版纯 flow 训练结果。

## 预期结果

- 训练日志中会新增 `train/loss_recon` 与 `val/loss_recon(_ema)`。
- 若该辅助损失有效，纯 flow 版本应比旧版更接近 `baseline`，或在某个 split 上出现小幅正向变化。

# 2026/04/21

## RAFT 在 SR 邻帧上估流

- 调整 `flow_guidance.py` 的正式先验构造逻辑：当 `flow_backend=raft` 时，直接在 `RVRT` 生成的 `prev_sr / next_sr` 上估计双向光流。
- 保留 `Farneback` 的原始低分辨率估流路径不变，仅将 RAFT 专门切到 SR 尺度，原因是 LR 邻帧上存在数值不稳定风险。

## 人工测试方式

1. 先执行 smoke test，确认 `prev_sr / next_sr` 上的 RAFT 光流输出正常，且 `flow_prior.png` 可保存。
2. 再执行正式 small eval，设置：
   `FLOW_BACKEND=raft FLOW_RAFT_VARIANT=large FLOW_RAFT_CKPT=/abs/path/to/raft_large_C_T_SKHT_V2-ff5fadd5.pth`
3. 观察日志不再出现 LR 输入下的 `NaN/Inf` 或 `flow_to_image` 索引异常。

## 预期结果

- RAFT 先验将基于 SR 邻帧构造，而不是低分辨率邻帧。
- 若 RAFT 本身有效，新的 `flow_prior` 应比旧的 LR 估流版本更稳定、更接近中间时刻结构。

# 2026/04/21

## RAFT 光流后端

- 在 `flow_guidance.py` 中新增 `RAFT` 可选后端，保留 `Farneback` 作为回退选项。
- RAFT 通过 `flow_backend=raft`、`flow_raft_variant`、`flow_raft_ckpt` 控制，并要求使用本地权重路径，避免训练脚本触发下载。
- 纯 flow 版和 PiSA flow 版的训练类、pipeline、eval、check 均已接入这组 metadata，保证训练与评测链路一致。

## 人工测试方式

1. 准备本地 RAFT 权重，例如 `raft-large.pth`。
2. 训练时传入：
   `FLOW_BACKEND=raft FLOW_RAFT_VARIANT=large FLOW_RAFT_CKPT=/abs/path/to/raft-large.pth`
3. 观察日志打印：
   `backend=raft, raft_variant=large`
4. 训练完成后跑 small eval，确认 summary 中记录了 `flow_backend=raft`。

## 预期结果

- 光流先验将由更强的 RAFT 替代 Farneback 生成。
- 若 RAFT 权重路径错误，训练/评测会在光流先验构造阶段直接报错，而不是静默回退。

# 2026/04/20

## 纯光流引导版本

- 新增 `rvrt_flow_guided_ddpm.py`，从 `RVRT + LDMVFI` 主线拆出不带 LoRA 的纯光流引导训练类。
- 光流版仍保持 `RVRT` 冻结，只训练 diffusion UNet；额外通过 `flow-guided middle prior` 增强条件分支。
- 新增配置 `configs/ldm/rvrt-flow-guided-stsr-x4.yaml` 与脚本 `run_cloud_train_rvrt_flow_guided.sh`，用于单独验证光流引导本身的效果。
- 评测管线 `rvrt_ldmvfi_pipeline.py` 也补上了 flow prior 注入，这样训练和评测链路一致。

## 人工测试方式

1. 运行 `run_cloud_train_rvrt_flow_guided.sh`。
2. 观察日志中出现 `Flow guidance enabled: True`，且不出现任何 LoRA 注入日志。
3. 使用 `configs/ldm/rvrt-flow-guided-stsr-x4.yaml` 跑 `run_cloud_eval_rvrt_ldmvfi.sh`。
4. 对比纯 `RVRT + pretrained/full-tune LDMVFI` 与 flow-guided 版本的 `PSNR/SSIM`。

## 预期结果

- 训练时只会打印 diffusion UNet 的优化器信息，不会出现 `Injected LoRA` 或 `Injected PiSA Dual-LoRA`。
- 评测时若 checkpoint 带有 flow metadata，链路会自动恢复 `use_flow_guidance` 和 `flow_guidance_strength`。

# 2026/04/20

## Summary

- Added a minimal optical-flow-guided conditioning path for the PiSA Dual-LoRA line.
- The new path builds a `flow-guided middle prior` from LR-neighbor Farneback flow and fuses its encoded features into the original `prev/next` conditioning features.
- Kept the diffusion UNet input dimension unchanged by fusing the flow prior in conditioning space instead of concatenating a third condition latent.
- Added a dedicated flow-guided config file for experiments closer to the thesis topic.

## Files

- `ldm/models/flow_guidance.py`
- `ldm/models/diffusion/rvrt_pisa_dual_lora_ddpm.py`
- `ldm/models/rvrt_pisa_dual_lora_pipeline.py`
- `check_rvrt_pisa_dual_lora.py`
- `evaluate_rvrt_pisa_dual_lora.py`
- `run_cloud_train_rvrt_pisa_dual_lora.sh`
- `configs/ldm/rvrt-pisa-dual-lora-flow-stsr-x4.yaml`

## Manual Test

1. Start training with `configs/ldm/rvrt-pisa-dual-lora-flow-stsr-x4.yaml`.
   Expected: the startup log prints `Flow guidance enabled: True`.
2. Run a deterministic check with the new flow-guided config.
   Expected: `check_results_rvrt_pisa_dual_lora/flow_prior.png` is saved.
3. Compare flow-guided and non-flow-guided runs on the same checkpoint family.
   Expected: the flow-guided path changes the conditioning and may alter both pixel and semantic output differences.

## Expected Result

- The method becomes closer to “optical-flow-guided generative video spatio-temporal super-resolution”.
- Flow guidance is introduced without changing the main diffusion UNet channel dimensions.
- The new path is suitable for a lightweight thesis-oriented ablation.

# 2026/04/20

## Summary

- Split PiSA Dual-LoRA target coverage into independent `pixel_target_suffixes` and `semantic_target_suffixes`.
- Added `semantic_lr_scale` so the semantic optimizer group can use a higher learning rate than the pixel group.
- Updated the default PiSA config to bias semantic training toward `decoder/others` and higher-level attention / FFN targets.
- Updated the check and eval entrypoints to read and override the new pixel / semantic target metadata.
- Made the cloud check wrapper clear CUDA-related environment variables before launching Python.

## Files

- `ldm/modules/pisa_dual_lora.py`
- `ldm/models/diffusion/rvrt_pisa_dual_lora_ddpm.py`
- `check_rvrt_pisa_dual_lora.py`
- `evaluate_rvrt_pisa_dual_lora.py`
- `configs/ldm/rvrt-pisa-dual-lora-stsr-x4.yaml`
- `run_cloud_train_rvrt_pisa_dual_lora.sh`
- `run_cloud_eval_rvrt_pisa_dual_lora.sh`
- `run_cloud_check_rvrt_pisa_dual_lora.sh`

## Manual Test

1. Start a PiSA Dual-LoRA training run with the default config.
   Expected: the log prints different `Pixel target suffixes` and `Semantic target suffixes`, and the semantic optimizer learning rate is higher.
2. Run a deterministic pixel check after a checkpoint is produced.
   Expected: `pixel_scale` changes still produce a clearly non-zero output difference.
3. Run a deterministic semantic check after semantic training has progressed.
   Expected: `semantic_scale` changes produce a larger output difference than before this patch.
4. Launch `run_cloud_check_rvrt_pisa_dual_lora.sh` from a shell polluted by `(base)` conda variables.
   Expected: the wrapper still initializes CUDA successfully because it clears `LD_LIBRARY_PATH`, `CUDA_HOME`, and `CUDA_PATH`.

## Expected Result

- Semantic adapters receive a stronger and more focused optimization signal.
- Pixel and semantic branches no longer share exactly the same target layer set by default.
- The default cloud check path is less fragile under mixed conda environments.

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
# 2026/04/21

## Flow 显式第三条件
- 调整纯 `RVRT + LDMVFI + flow guidance` 主线的条件注入方式：新增 `flow_condition_mode=explicit`。
- 新模式下不再把 `flow_prior` 弱平均到 `prev/next` 条件特征，而是先显式编码 `prev / flow / next` 三路条件。
- 为了保持当前 `UNet in_channels=9` 不变，新增一个 1x1 条件融合层，将三路条件压回原有两路条件通道数。
- 纯 flow 配置 `rvrt-flow-guided-stsr-x4.yaml` 默认切到 `explicit`，训练和 eval wrapper 也增加了 `FLOW_CONDITION_MODE` 入口。

## 人工测试方式

1. 使用纯 flow 配置启动训练：
   `FLOW_CONDITION_MODE=explicit bash run_cloud_train_rvrt_flow_guided.sh`
2. 观察启动日志：
   预期打印 `mode=explicit`，并出现 `Optimizer groups: diffusion+flow_fuser ...`
3. 使用 small eval 对比：
   - no-flow baseline
   - `RAFT + fused`
   - `RAFT + explicit`
4. 检查 summary JSON：
   预期包含 `flow_condition_mode=explicit`

## 预期结果

- `flow_prior` 不再只是对原条件的弱扰动，而是成为更显式的第三路条件信息。
- 在不修改主干 `UNet` 输入通道的前提下，增强模型对高质量光流先验的利用。
# 2026/04/22

## Flow 对齐主输入
- 针对 `RAFT + fused/explicit` 仍未稳定超过 baseline 的现象，新增 `flow_condition_mode=aligned_input`。
- 新模式下不再把光流当作旁路提示，而是先在 `prev_sr / next_sr` 上估双向光流，再将两张 SR 邻帧 warp 到中间时刻，直接替换原始条件输入。
- 训练类与推理 pipeline 均已接入该模式，`fused/explicit` 保留用于旧实验对比。
- 纯 flow 配置 `rvrt-flow-guided-stsr-x4.yaml` 默认切到 `aligned_input`，便于直接做新一轮 small eval。

## 人工测试方式

1. 使用 pure flow 配置运行 small eval：  
   `FLOW_CONDITION_MODE=aligned_input FLOW_BACKEND=raft ... bash run_cloud_eval_rvrt_ldmvfi.sh`
2. 对比三组结果：  
   - no-flow baseline  
   - `RAFT + fused/explicit`  
   - `RAFT + aligned_input`
3. 重点观察 `fast_test / medium_test` 上的 `PSNR / SSIM / LPIPS`。

## 预期结果

- 如果当前瓶颈确实在“光流只做旁路提示”，那么 `aligned_input` 应至少优于 `fused/explicit`。
- 若仍无法接近 baseline，则说明“先对齐再生成”在当前框架下也缺乏足够收益信号。
