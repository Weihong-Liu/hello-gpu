# CONSENSUS · 算子优化 Agent

> 状态：已确认（用户 2026-07-28 决策）  
> 上游：`ALIGNMENT_算子优化Agent.md`

---

## 1. 需求描述

基于现有 `kernel_optimize`，对齐 chapter15「读题→生成→评测→反思」闭环，补齐 Skill/教学目录与非交互入口，在 AMD RX 9070 XT 上对 **vector_add** 跑通自动化优化。

## 2. 已确认决策

| 项 | 决策 |
|---|---|
| LLM | 硅基流动；Model `Qwen/Qwen3.6-35B-A3B`；OpenAI 兼容 `https://api.siliconflow.cn/v1` |
| Key | 写入 `~/.config/hello-gpu/kernel-agent.env`（本地+远端），**不进 git** |
| 远端目录 | `~/hello-gpu/code/part3-agent`（新建） |
| 首轮算子 | vector_add |
| 运行方式 | 非交互（预置 task/baseline，跳过 ask_user） |
| 教学目录 | 新建 `chapter15/` |

## 3. 技术方案

- **主干**：完善 `kernel_optimize`（不重写）
- **评测**：复用 `chapter14/evaluate.py`
- **非交互**：`python -m kernel_optimize --batch chapter15/fixtures/vector_add`
- **LLM**：`KERNEL_AGENT_MODEL=openai/Qwen/Qwen3.6-35B-A3B` + `KERNEL_AGENT_API_BASE` + `KERNEL_AGENT_API_KEY`
- **Skill**：补 `skills/rocm-kernel-optimize/references/*.md`

## 4. 验收标准

1. `chapter15/` 含 README + vector_add fixtures（baseline / reference / task.json）
2. `--batch` 可在无 TTY 下跑完优化闭环，产出 `trajectory.jsonl` + best.py + 最终报告
3. baseline 正确性通过；至少完成 measure_peak / profile_kernel / accept_candidate 路径
4. AMD 9070 XT 上实测日志可复现
5. API Key 不出现在仓库文件中

## 5. 边界

- 不重写 chapter14 evaluator
- 不要求首轮必须达到 3–5×（vector_add 优化空间有限）；闭环跑通即可
- rocprof 实测路径可退回 Roofline 模型
