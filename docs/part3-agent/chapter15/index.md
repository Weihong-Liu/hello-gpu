---
title: "第15章 工具封装"
description: "Hello GPU 第15章 · benchmark/profiling/编译包成 Agent 可调用工具"
---

# 第15章 工具封装

## 本章目标、前置知识与产物

> 本章把 Part 1 学过的 benchmark、rocprof 以及编译流程，封装成 Agent 能调用的标准化工具。这是让 Agent「能动手」的前提——没有工具的 Agent 只会空谈。

Part 1 里我们是用人手敲命令的方式做 benchmark 和 profiling 的：写脚本、看输出、记数字。Agent 做不了这件事——它只能产生文本。要让 Agent「能动手」，唯一的办法是**把每一个动作做成输入输出严格定义、输出机器可解析的工具**，Agent 负责决定调哪个工具、怎么调，工具负责执行并给出结构化结果。

本章的素材来自一次真实的多轮 kernel 优化实验（KDA 工作流，见延伸阅读）：同一个任务契约、同一套工具，人工优化做到了 2.87x（fused MLP 对照组，见第 17.5 节），Agent 自动优化贡献接近 0%。本章把那次实验里「工具」这一层的设计拆给你看。

学完本章，你应该能够：

- 解释为什么 Agent 优化 kernel 必须依赖结构化的工具接口；
- 把一个 benchmark 脚本封装成「输入 kernel → 输出 JSON」的标准化工具；
- 把 rocprof 的输出翻译成 Agent 能读懂的瓶颈信号；
- 用 JSON schema 定义工具接口，并处理编译失败、正确性失败等错误路径；
- 理解「工具的输出质量」如何决定 Agent 的上限。

## 15.1 为什么要封装工具

先看一个来自真实实验的教训。在 fused MLP 的对照组实验（HIP / RX 7900 XTX，详见第 17.5 节）里，Agent 共发起 **15 次 API 调用**，其中只有 **1 次**生成的代码能通过编译和正确性测试，其余 14 次全部失败：有的编译报错、有的数值不对、有的性能更差。最终对性能的贡献接近 0%。

这不是 Agent 太笨——而是**每一步「尝试」的反馈都太模糊**。如果 Agent 生成代码后，得到的反馈是「编译失败，第 42 行有 3 个错误」，它至少能缩小问题范围；如果得到的反馈是「跑完了，但结果和参考实现差 0.5」，它能判断数值问题出在归约还是累加；但如果它只能看到「报错」「不对」这类信息，下一轮修改就是盲猜。

所以工具封装的第一原则是：

::: tip 工具封装第一原则
**反馈必须是结构化的、可解析的、分层的。** 编译成功/失败、正确性通过/失败、性能数字，这三层信息必须互相独立、各自明确，Agent 才能知道当前卡在哪一层。
:::

第二原则是**工具要有明确的所有权**：谁负责计时、谁负责检查正确性、谁负责 profiling，都在工具内部完成，Agent 只负责调用和解读。这样 Agent 的 prompt 里不需要塞进 Part 1 的细节（warmup 几次、为什么用 GPU event），那些知识固化在工具里。

第三原则是**可复现**：同一个工具、同一份输入，任何时候跑都得到同一份输出。这是 Part 1 可信计时的延伸——工具封装不能把「量不准」的问题带进自动化。

## 15.2 封装 benchmark 工具

把 Part 1 的计时方法包成一个工具。工具的名字、输入、输出长这样：

```text
工具名：bench_kernel
输入：
  - kernel 源码或编译产物路径
  - 输入形状（M, N, K ...）
  - dtype
  - 重复次数（默认 200）
输出（JSON）：
  {
    "mean_ms": 0.066,
    "median_ms": 0.065,
    "p95_ms": 0.070,
    "bandwidth_gbps": 120.4,
    "tflops": 4.2,
    "std_ms": 0.002
  }
```

注意输出的字段，和 Part 1 的检查清单完全对应：只报 mean 不可信，要带上 median、p95、std；带宽和算力按形状算好，Agent 不需要自己推公式。

看一个真实例子。在 FA Decode（单 token 注意力，第 17 章的主角）的 benchmark 工具里，计时配置是：

```text
warmup = 20 次     # 先热身，丢弃缓存冷启动
rep    = 200 次    # 正式测量 200 次
统计  = median + p95，而不是单次结果
```

这就是 Part 1 第 5 章检查清单的自动化形态。工具内部做热身、重复、同步，Agent 只拿到最终统计。

工具还应该把**参考实现**一起管住：正确性测试和性能测试分离（`test_fa_decode.py` 管正确性、`bench_fa_decode.py` 管性能），但两者共享同一份输入形状和 reference 计算，避免「测的性能和验的正确性不是同一个 kernel」。

## 15.3 封装 profiling 工具

benchmark 告诉你「慢了」，profiling 告诉你「慢在哪」。Agent 需要的是从 profiler 输出里提炼出的**瓶颈信号**，而不是几千行原始 trace。

把 rocprof 包成工具后，输出的核心字段应该是：

```text
{
  "kernel": "fa_decode_split",
  "time_ms": 0.066,
  "grid": [128, 1, 1],
  "vgpr": 244,
  "bandwidth_utilization_pct": 69.5,  // DRAM 带宽利用率
  "bottleneck": "DRAM bandwidth",  // 信号：带宽 / 算力 / 发射 / 启动开销
  "sm_saturation_pct": 68
}
```

真实实验里的一个例子：FA Decode 最终版（split-KV，第 17.3 节的 v3）跑完 profiling 后，工具给出的信号是「DRAM 带宽利用率 69.5%，grid 128 block 填满 68 个 SM」（RTX 3080 上实测；SM 是 NVIDIA 对第 3 章 CU/WGP 的称呼）。Agent 据此判断：还有约 30% 带宽没用上，方向是让访存更饱和，而不是换算法。这个判断和人工分析的结论一致。

profiling 工具封装的关键是**信号提炼规则要写在工具里**：什么样的利用率算访存受限、什么样的算占用率不足，规则固定下来，Agent 每次拿到的是结论而不是数据。

## 15.4 封装编译工具

编译是 Agent 高频动作里最容易失败的一环——上面 15 次调用 14 次失败，绝大多数死在编译和正确性。所以编译工具的输出必须能让 Agent 精确知道失败原因。

```text
工具名：compile_kernel
输入：
  - 源码路径（HIP 或 Triton）
  - 目标平台（gfx1201 / sm86 ...）
输出（JSON）：
  {
    "ok": false,
    "stage": "compile",            // compile / link / run
    "errors": [
      {"line": 42, "msg": "expected a type, found 'float4_2'"}
    ],
    "warnings": []
  }
```

关键设计：**错误要按行列出，而不是整段日志**。Agent 是文本模型，整段编译日志里有 90% 的噪音；按行提取的 error list 让它能直接对应到源码位置。

还有一个实践教训：**编译产物要留档**。在一次 7900XTX 实验里，每轮 Agent 生成的 `.hip`、编译出的 `.bc`/`.s`/`.o` 全部落盘，研究者才能在实验结束后复盘「这一轮为什么失败」。工具负责留档，Agent 不用管。

## 15.5 工具的输入输出 schema

每个工具的接口用 JSON schema 定义，Agent 的 prompt 里只放 schema，不放实现细节。以编译工具为例：

```json
{
  "name": "compile_kernel",
  "description": "编译 HIP/Triton kernel，返回结构化错误",
  "parameters": {
    "type": "object",
    "properties": {
      "source": {"type": "string", "description": "源码文件路径"},
      "arch":  {"type": "string", "enum": ["gfx1201", "gfx1100", "sm86"]}
    },
    "required": ["source"]
  }
}
```

再进一步，把整个任务的要求也写成结构化契约。一次 kernel 优化任务开始前，先写一份 **task contract**，包含：

| 字段 | 内容示例（FA Decode） |
|---|---|
| Objective | 实现单 token 流式 Flash Attention decode kernel，避免物化 N×N 注意力矩阵 |
| Inputs/Outputs | Q(1,H,D)、K_cache(N,H,D)、V_cache(N,H,D) → O(1,H,D)，FP16 |
| Correctness | 与 `torch.einsum` 参考比较，atol=1e-2 |
| Constraints | Triton 3.x；必须用 online softmax，不物化中间矩阵 |
| Validation command | `python3 test_fa_decode.py` |
| Evaluation command | `python3 bench_fa_decode.py` |
| Promotion criteria | 全部正确性测试通过，且 N=2048 ≥3.0x、N=4096 ≥4.0x、N=8192 ≥5.0x |

这份契约就是 Agent 的「题目」。它同时被三个角色复用：Agent 读它来理解任务，工具用它来校验结果，人用它来验收——同一个 schema，三个角色，不会各说各话。

## 15.6 错误处理与重试

工具链运行时的错误分成三类，处理方式不同：

| 错误类型 | 例子 | 回传给 Agent 的信息 | Agent 的下一步 |
|---|---|---|---|
| 环境错误 | GPU 忙、磁盘满、依赖缺失 | 明确错误类型 + 重试建议 | 等待后重试，或换方案 |
| 编译错误 | 语法错、类型错 | 按行的 error list | 定位到行修改源码 |
| 验证错误 | 正确性不通过、性能不达标 | 差异位置 + 数字 | 修改算法，不是改格式 |

关键原则：**工具失败不算任务失败**。工具把错误结构化后返回，Agent 把它当成一次「观测」，触发反思循环：为什么失败 → 改什么 → 再试。这与 KDA 工作流的 promotion rule 一致：

> 只有当候选实现满足任务契约、且有证据表明它提升或保持了目标指标时，才把它提升为当前最优；被拒绝的候选要记录原因，而不是静默丢弃。

失败的记录和成功的记录一样重要——第 17 章会看到，一次「tl.dot 反而更慢」的失败记录，比十次成功更能说明 decode 场景的本质。

## 本章小结

- Agent 优化 kernel 的前提是结构化工具：编译、benchmark、profiling 各有明确输入输出和机器可读反馈。
- benchmark 工具输出 mean/median/p95/std 和带宽、算力，Agent 不需要重复 Part 1 的计时细节。
- profiling 工具输出「瓶颈信号」而不是原始 trace；信号提炼规则固化在工具里。
- 编译工具按行返回错误列表，驱动 Agent 精确定位修改。
- 用 JSON schema 定义工具接口，用 task contract 定义任务契约，三个角色（Agent/工具/人）共享同一份 schema。
- 错误处理的原则是「工具失败不算任务失败」：结构化错误 → 反思 → 重试。

## 延伸阅读

- [kernel-design-agents (MIT HAN Lab)](https://github.com/mit-han-lab/kernel-design-agents) — KDA 工作流与 KernelWiki，本章工具设计的事实来源
- [第 5 章 benchmark 与可信计时](../../part1-profiling/chapter5/index.md) — 工具内部计时的正确性依据
- [第 6 章 用 rocprof 找到慢在哪里](../../part1-profiling/chapter6/index.md) — profiling 信号提炼的底层方法
