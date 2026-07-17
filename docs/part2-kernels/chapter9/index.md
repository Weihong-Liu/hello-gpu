---
title: "第9章 Normalization：归一化算子"
description: "Hello GPU 第9章 · 以行级 Softmax 为例，学习数值稳定与逐元素/归约融合"
---

# 第9章 Normalization：归一化算子

## 本章导读

> 本章把 Element-Wise 与 Reduction 组合起来：Softmax 既要逐元素取指数，又要两次归约。公共部分先用大数反例理解“减最大值”；HIP 篇观察中间写回与数据驻留，Triton 篇学习一行对应一个 program 的融合表达。

## 9.1 归一化算子解决什么问题

从 logits 到概率，先固定逐行 Softmax 语义。

## 9.2 为什么直接指数会溢出

用小例子推导减最大值，而不是只给公式。

## 9.3 Softmax 由哪些基本模式组成

拆成 max reduction、element-wise exp、sum reduction 与 normalize。

## 9.4 HIP v0：稳定的多 kernel baseline

先保证数值正确，并统计完整路径。

## 9.5 HIP v1/v2：融合 dispatch 与寄存器驻留

区分少一次 launch 与少一次全局写回。

## 9.6 HIP v3：Wave32 收尾

保持数据映射不变，只替换 block 内归约机制。

## 9.7 Triton t0：一行对应一个 program

讲清 mask、next power of 2 与行内 reduction。

## 9.8 Triton t1：block size 与 num warps

只做受控参数实验，并允许参数变大反而变慢。

## 9.9 HIP 与 Triton 的融合边界

比较显式寄存器/LDS 控制与编译器生成映射。

## 9.10 稳定性压力测试与练习

覆盖大正值、大负值、尾行、长行和不同 dtype。

## 本章小结

- 本章当前只保留按新教程结构整理的大纲，旧正文、代码、日志和图片已清除。
- 正式写作时先完成 Radeon RX 9070 XT + ROCm 7.13 + 原生 Ubuntu 24.04 实验，再补命令、输出和性能结论。
- 新实验代码将放在 `code/part2-kernels/chapter9/`。

## 延伸阅读

- 待正式实验与正文完成后补充 AMD、ROCm、Triton 与 Softmax 相关资料。
