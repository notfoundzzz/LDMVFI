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
