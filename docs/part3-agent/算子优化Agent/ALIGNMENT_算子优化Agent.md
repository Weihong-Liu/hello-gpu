# ALIGNMENT · 算子优化 Agent

> 阶段：Align（对齐）  
> 参考：[第15章 算子优化 Agent 设计](https://datawhalechina.github.io/hello-gpu/part3-agent/chapter15/)  
> 本地代码：`part3-agent 2/kernel_optimize/`（教学用 Reflection/ReAct agent）  
> 目标机器：`hwj-frp-9070xt-2404`（RX 9070 XT / gfx1201）

---

## 1. 项目与任务特性

| 项 | 内容 |
|---|---|
| 项目 | hello-gpu 第三篇（part3-agent）：用 Agent 把「算子优化闭环」讲透并跑通 |
| 本章目标 | 读题 → 生成 → 编译/评测 → 跑分 → 反思改写，形成可迭代 Agent |
| 现有实现 | `kernel_optimize/` 已具备主循环、工具层、intake、峰值测量、提示词与单测 |
| 权威评测 | `chapter14/` evaluator（正确性 + 可信计时 + 配对裁决） |
| 执行语言 | Triton（非 Triton 需 `convert_kernel` 转换） |
| 方法论 | Profiling-Driven / Roofline；正确性先于性能；工具结果即事实 |

## 2. 原始需求

1. 阅读本仓库（`part3-agent 2`）现有代码。
2. 编写/完善一个**算子自动化优化 Agent**（对齐 chapter15 大纲）。
3. 在 AMD 服务器（`hwj-frp-9070xt-2404`，9070 XT）上实测。

## 3. 边界确认（任务范围）

**包含**
- 对齐 chapter15：理解题目 → 初始 kernel → benchmark 反馈 → 反思改写 → 终止条件
- 补齐缺口：`skills/rocm-kernel-optimize/`（references）、非交互跑通能力、AMD 环境部署
- 在 9070 XT 上跑通至少 1 个算子优化闭环（正确性 + 性能裁决 + 报告）
- 按 6A 产出 ALIGNMENT / CONSENSUS / DESIGN / TASK / ACCEPTANCE / FINAL / TODO

**不包含（除非另行确认）**
- 重写 chapter14 evaluator
- 同步教程正文（chapter15 正式文档）
- 完整 rocprof 硬件计数器打磨（可保留 Roofline 模型路径）
- 把旧 `kernel_agent` 重型实现全部删掉/迁移

## 4. 对现有项目的理解

### 4.1 架构现状（已读代码）

```
用户粘贴 kernel
    → ask_user / setup_task（intake）
    → measure_peak / profile_kernel（定方向）
    → LLM 生成候选
    → compile_kernel → bench_kernel → accept_candidate（晋升 + 写轨迹）
    → 反思迭代
    → 最终报告
```

- `agent.py`：Reflection/ReAct 主循环 + 完成护栏（防纯文本假结束）
- `tools.py`：ToolExecutor + 权威/自由工具注册
- `intake.py`：从原始 kernel 生成 reference + task 合同
- `measure.py`：峰值测量 + Roofline 方向
- `prompts.py`：SYSTEM_SOP + INTAKE_GOAL
- 入口：`python -m kernel_optimize` / `python -m kernel_agent`（后者委托前者）

### 4.2 已知缺口（相对 HANDOFF / REFACTOR-PLAN-v2）

| 缺口 | 影响 |
|---|---|
| 缺少 `skills/rocm-kernel-optimize/`（references） | `read_reference` 无内容可读 |
| 本地/远端均无 `kernel-agent.env` | Agent 无法调 LLM |
| 远端 `~/shaowenjie/hello-gpu/code/part3-agent` 几乎空（仅 pyproject/lock） | 需 rsync 本地实现 |
| 交互式 `ask_user` | 远端无 TTY 批量测试困难，需非交互入口 |
| chapter15 目录不存在 | 教程约定路径未落地（可选） |

### 4.3 AMD 服务器现状

- Host：`hwj-frp-9070xt-2404` → 118.145.218.207:7011，用户 `hellogpu`
- GPU：AMD Radeon RX 9070 XT（gfx1201），ROCm 可用
- 系统 Python 无 torch；需用项目 `uv` + ROCm wheel 源安装
- 远端已有 hello-gpu 仓库骨架，part3-agent 代码未同步完整

## 5. 疑问澄清（需用户决策）

### P0 · 阻塞运行

1. **LLM 配置**：Agent 依赖 `KERNEL_AGENT_MODEL` + API Key（写到 `~/.config/hello-gpu/kernel-agent.env`）。  
   请提供要用的模型（如 `anthropic/claude-sonnet-4-5`、`deepseek/deepseek-chat`、`openai/gpt-4o`）及可用 Key/兼容 Base URL。  
   **没有此项无法在真机跑通 Agent。**

2. **远端工作目录**：代码同步到哪里？  
   - A. `~/hello-gpu/code/part3-agent`（新建，属 hellogpu 用户）  
   - B. `~/shaowenjie/hello-gpu/code/part3-agent`（覆盖/补齐已有仓库）  
   - C. 其他路径

### P1 · 验收形态

3. **测试算子**：首轮实测用哪个？  
   - A. `vector_add`（chapter14 fixture，简单、易验证闭环）  
   - B. `masked-softmax`（chapter13 baseline，更贴近优化故事）  
   - C. 你指定的其他 kernel

4. **运行方式**：远端批量测试是否接受「非交互模式」？  
   - 预置 kernel + shape/dtype，跳过 `ask_user`，直接 `setup_task` → 优化闭环  
   - （交互模式仍保留，供本机对话使用）

5. **交付落点**：是否需要新建 `chapter15/` 教学目录（示例 task、README），还是只完善 `kernel_optimize/` 即可？

## 6. 暂定假设（待确认后写入 CONSENSUS）

- 以现有 `kernel_optimize` 为主干完善，不另起炉灶重写 Agent。
- 首轮验收算子默认 `vector_add`，验证闭环正确性与报告；有余力再跑 softmax。
- 增加非交互 CLI，便于 SSH 无 TTY 实测。
- 补齐 `skills/.../references`（optimization-patterns 等）。
- 敏感配置只进 `~/.config/hello-gpu/kernel-agent.env`，不进 git。
