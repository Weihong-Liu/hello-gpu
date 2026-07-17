---
title: "第8章 Reduction：归约算子"
description: "Hello GPU 第8章 · 以 Sum Reduction 为例，学习跨线程协作、LDS 与 Wave Shuffle"
---

# 第8章 Reduction：归约算子

## 本章导读

> 本章撤掉 Element-Wise 中“每个输出彼此独立”的前提：很多输入要共同得到一个结果。公共部分先把串行求和画成并行树；HIP 篇深入 LDS、同步与 Wave Shuffle，Triton 篇用 program partial 和多阶段归约快速表达同一层次。

## 8.1 什么是归约算子

从多输入到少输出的数据依赖出发，区分 sum、max 与 argmax。

## 8.2 从串行求和到并行树

手算一个小数组，对比线性依赖与树形依赖。

## 8.3 固定正确性与测量口径

明确浮点误差、非二次幂长度和完整多阶段计时。

## 8.4 HIP v0：逐元素 atomic baseline

用全局同地址争用建立最短的正确起点。

## 8.5 HIP v1：LDS block 内归约

协作加载、同步并逐轮收缩活动线程。

## 8.6 HIP v2：寄存器局部累加

先让每个线程得到局部和，再进入 LDS。

## 8.7 HIP v3/v4：Wave Shuffle 与多阶段 partial

减少同步，并把跨 block 合并拆成独立阶段。

## 8.8 Triton t0：program partial

说明一个 program 怎样覆盖一段输入并使用 `tl.sum`。

## 8.9 Triton t1：多阶段归约

用 partial buffer 与第二次 dispatch 完成跨 program 合并。

## 8.10 HIP 与 Triton 的归约层次对照

对齐局部和、组内归约与跨组合并。

## 8.11 复跑与练习

覆盖 max、非二次幂、累加 dtype 与负结果分析。

## 本章小结

- 本章当前只保留按新教程结构整理的大纲，旧正文、代码、日志和图片已清除。
- 正式写作时先完成 Radeon RX 9070 XT + ROCm 7.13 + 原生 Ubuntu 24.04 实验，再补命令、输出和性能结论。
- 新实验代码将放在 `code/part2-kernels/chapter8/`。

## 延伸阅读

- 待正式实验与正文完成后补充 AMD、ROCm 与 Triton 官方资料。
