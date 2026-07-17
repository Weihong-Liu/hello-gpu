---
title: "第11章 Fusion：融合算子"
description: "Hello GPU 第11章 · 以 FlashAttention 为例，学习在线计算、减少中间写回与 IO-aware"
---

# 第11章 Fusion：融合算子

## 本章导读

> 本章把前四章的模式组合起来：矩阵乘产生 Scores，Softmax 做归一化，再与 V 相乘。公共部分先比较物化与在线数据流；HIP/Triton 两篇分别实现教学版前向 FlashAttention，并用完整路径验证减少中间写回的价值。

## 11.1 从普通 Attention 数据流开始

只补本章需要的 Q/K/V、Scores、Softmax 与输出。

## 11.2 物化中间矩阵的代价

画出三段 kernel 与 Scores/P 的全局读写路径。

## 11.3 在线 Softmax 怎样保持精确

手算 running max、normalizer、历史重缩放与输出累加。

## 11.4 固定语义、边界与测量口径

明确 FP16 输入、FP32 累加、causal、尾块和完整时间。

## 11.5 HIP h0/h1：从物化基线到在线融合

先消除完整中间矩阵，再验证正确性。

## 11.6 HIP h2–h4：Wave、query/key 分块与 K/V 复用

每轮只改变一个机制并跟踪资源代价。

## 11.7 Triton t0：物化基线

保持与 HIP 相同的数学语义与计时边界。

## 11.8 Triton t1–t3：在线 query/key tile

解释块级状态、mask 与寄存器/scratch 风险。

## 11.9 完整证据与实现边界

汇总时间、分配字节、dispatch、正确性与资源字段。

## 11.10 从组合到融合的方法总结

回收 Element-Wise、Reduction、Normalization 与 GEMM-Like。

## 本章小结

- 本章当前只保留按新教程结构整理的大纲，旧正文、代码、日志和图片已清除。
- 正式写作时先完成 Radeon RX 9070 XT + ROCm 7.13 + 原生 Ubuntu 24.04 实验，再补命令、输出和性能结论。
- 新实验代码将放在 `code/part2-kernels/chapter11/`。

## 延伸阅读

- 待正式实验与正文完成后补充 FlashAttention 论文与 AMD、ROCm、Triton 官方资料。
