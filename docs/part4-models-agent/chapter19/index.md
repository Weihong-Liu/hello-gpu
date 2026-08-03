---
title: "第19章 小模型 LLM 解码 + Agent 自动优化"
description: "Hello GPU 第19章 · Qwen 0.5B/1.8B 量化、decode 算子视角、Agent 优化 KV cache/精度"
---

# 第19章 小模型 LLM 解码 + Agent 自动优化

## 本章导读

> 本章把 Agent 应用到 LLM 解码场景。受 RX 9070 XT 16GB 显存限制，只能量化小模型（Qwen 0.5B/1.8B）。重点是 decode 阶段的算子视角：Agent 如何优化 KV cache 访问、精度选择等。注意：PagedAttention、多卡 TP 等留给 hello-mlsys。

## 19.1 显存约束下的模型选型

说明 RX 9070 XT 16GB 显存为什么选 Qwen 0.5B/1.8B 量化，跑不了 7B+。

## 19.2 LLM 推理流程拆解

一次生成式推理分成两个阶段，瓶颈性质完全不同：

| 阶段 | 计算形态 | 算术强度 | 瓶颈 |
| ---- | ---- | ---- | ---- |
| **prefill** | 整段 prompt 的矩阵乘（GEMM） | 高 | 算力 |
| **decode** | 每生成一个 token 的矩阵×向量（GEMV） | 低 | 带宽 |

decode 为什么是「算子密集型」？因为每生成一个 token，都要把模型**全部权重读一遍**——Qwen 0.5B 有约 1 GB 权重，每 token 从显存读一遍就是 1 GB 流量。而计算量只有一次矩阵×向量：M=1 的 GEMM，即 **GEMV**（General Matrix-Vector multiplication）。

对比算术强度（第 4.4.2 节的定义）：prefill 的 GEMM 每次读取可以复用多次（大 M 分块），算术强度高；decode 的 GEMV 每个权重元素只被一个 token 用一次，读出来用完即弃。所以 decode 的吞吐几乎完全由「每 token 读权重的字节数 ÷ 显存带宽」决定——**这是理解 decode 性能的唯一主线**。

## 19.3 建立 decode baseline

decode 的测量口径：固定 prompt、固定生成长度，测**每 token 时间**（TPOT），通常报告为 tokens/s。测量时要固定生成长度并重复多次——和 Part 1 的检查清单一样，单次结果不可信。

下面是一次真实的 MoE 模型 decode 调参实验（llama.cpp，ROCm 后端，`-p 0 -n 64 -r 5` 即 64 token 生成、5 次重复）。模型是 8.47B 参数、MoE A1B（每 token 只激活 1 个专家）的量化版本（Q4_K_M，4.79 GiB）：

| 配置 | t/s（mean ± std） |
| ---- | ----: |
| 默认（RPB=1, NW=8） | 214.60 ± 0.45 |

这是基线。注意 std 只有 0.45——测量本身很稳，接下来任何配置差异都可以归因于配置而不是噪声。

## 19.4 decode 的算子视角

把 decode 的每步拆成算子：attention（读取 KV cache，第 17 章 FA Decode 的 M=1 场景）、MoE 专家 GEMV（投影）、输出投影。**当模型是 MoE 时，GEMV 是流量主力**——decode 的带宽大部分花在读权重，就为了一次矩阵×向量。

decode 的理论吞吐上限由带宽和「每 token 读多少字节」决定：

$$
理论上限 = \frac{\text{显存带宽}}{\text{每 token 读取量}}
$$

先看一个错误示范：把整个模型权重（4.79 GiB）都算进「每 token 读取量」，用 7900 XTX 的显存带宽 960 GB/s：

$$
\frac{4.79\ \text{GiB} \approx 5.1\ \text{GB}}{960\ \text{GB/s}} \approx 5.4\ \text{ms/token} \approx 190\ \text{t/s}
$$

但实测是 214.6 t/s——比这个「上限」还快。矛盾说明假设错了：**MoE A1B 每层只激活 1 个专家，每 token 读的是共享权重 + 激活专家，远小于 4.79 GiB 总量**。反推：如果 214.6 t/s 真是全量读取，所需带宽是 214.6 × 5.1 GB ≈ 1.1 TB/s，超过 960 GB/s 的物理上限——所以稀疏激活让每 token 的实际读取量明显小于全量权重。这给优化留下了两个方向：压「每 token 读取量」（更强量化、更少激活参数），以及让 GEMV 访存更饱和（下表）。真实实验对 **block 配置**（每块处理的行数 RPB、每块的 warp 数）做了单变量扫描：

**RPB（每块处理行数）扫描**，固定 num_warps=8：

| RPB | t/s | 相对默认 |
| ---- | ----: | ----: |
| 1（默认） | 214.60 ± 0.45 | 1.00x |
| 2 | 276.90 ± 5.73 | 1.29x |
| **4** | **293.82 ± 1.82** | **1.37x** |
| 8 | 285.53 ± 0.49 | 1.33x |

**num_warps 扫描**，固定 RPB=1：

| num_warps | t/s | 相对默认 |
| ---- | ----: | ----: |
| 8（默认） | 214.52 ± 0.33 | 1.00x |
| 4 | 254.58 ± 0.75 | 1.19x |
| **2** | **308.85 ± 2.08** | **1.44x** |
| 1 | 290.17 ± 1.07 | 1.35x |

两个维度的结论一致：**block 配置对 MoE decode 吞吐有 ~1.4x 的影响，且最优不在默认值**。RPB=4 或 num_warps=2 都能把 214 t/s 提到 290-310 t/s，成本为零（只改配置不改算法）。注意两张扫描是两次独立测量，各自带默认配置行，两次默认值（214.60 与 214.52）的微小差异在噪声范围内。

这组数据给 Agent 优化（第 19.5-19.6 节）提供了第一个抓手：decode 性能不是「算法没选对」，而是「访存没吃饱」——优化方向是让 GEMV 的加载更饱和，而不是换更好的 GEMM。这也呼应第 11.12.5 节：decode 形状算术强度低（I < 10），属于访存受限类，tile/num_warps 的选择要以占用率和访存吞吐为先。

## 19.5 Agent 优化 KV cache 访问

让 Agent 分析 KV cache 的访存模式，给出精度/布局优化建议。

## 19.6 Agent 优化精度选择

让 Agent 对比 fp16/int8/int4 在 decode 性能和显存上的权衡。

## 19.7 单卡边界与下一步

明确 PagedAttention、多卡 TP、并发调度等留给 hello-mlsys。

## 本章小结

- decode 与 prefill 瓶颈不同：prefill 是算力受限的 GEMM，decode 是访存受限的 GEMV。
- decode 的吞吐主线是「每 token 读权重的字节数 ÷ 显存带宽」，MoE 量化模型尤其明显。
- block 配置（RPB / num_warps）对 MoE decode 有 ~1.4x 影响，且默认值不是最优——这是零成本的调优空间。
- 量化已经压低了权重体积，decode 的优化方向是让 GEMV 访存更饱和，而不是换算法。
- 本章 19.1 模型选型、19.5-19.7 Agent 优化 KV cache / 精度选择仍在补充中。

## 延伸阅读

- llama.cpp ROCm 构建与 GGUF 量化格式（见附录 A 环境章节）
- [第 17 章 多轮优化实战](../../part3-agent/chapter17/index.md) — FA Decode 的 M=1 注意力优化轨迹，decode 的另一半算子
- [第 11.12 节 tile 形状怎么选](../../part2-kernels/chapter11/index.md) — 访存受限 shape 的配置选择规则
