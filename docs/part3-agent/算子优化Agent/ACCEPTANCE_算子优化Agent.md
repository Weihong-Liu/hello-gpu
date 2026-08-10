# ACCEPTANCE · 算子优化 Agent

> 实测时间：2026-07-28  
> 机器：`hwj-frp-9070xt-2404` · AMD Radeon RX 9070 XT (gfx1201) / ROCm 7.13  
> 命令：`uv run python -m kernel_optimize --batch chapter15/fixtures/vector_add --max-steps 20`  
> 工作区：`chapter15/logs/tasks/task-20260728-001713`  
> 退出码：0（约 24 分钟）

## 验收对照

| 标准 | 结果 |
|---|---|
| chapter15 fixtures + README | ✓ |
| 非交互 `--batch` 无 TTY 可跑 | ✓ |
| measure_peak / profile_kernel / accept_candidate | ✓ |
| 权威裁决接受 ≥1 轮 | ✓（中位改进 +1.65%） |
| trajectory.jsonl + best.py | ✓ |
| API Key 不进仓库 | ✓（仅 `~/.config/hello-gpu/kernel-agent.env`） |

## 确定性汇总摘要

- 轮次 18，接受 1
- 峰值：带宽 ~551 GB/s；profile：memory-bound（AI≈0.167）
- 因达到 max_steps=20 停止；LLM 未另写长文报告（主循环返回步数上限提示）
- 多次 `compile_error`（向量化 / cache_modifier 等）属搜索噪声，不影响闭环成立

## 产物路径（远端）

- 日志：`~/hello-gpu/code/part3-agent/chapter15/logs/batch-vector-add-20260728-001709.log`
- 轨迹 / best：`~/hello-gpu/code/part3-agent/chapter15/logs/tasks/task-20260728-001713/`
