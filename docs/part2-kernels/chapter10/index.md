---
title: "第10章 GEMM-Like：矩阵乘类算子"
description: "Hello GPU 第10章 · 以 Matmul 为例，学习分块、数据复用与寄存器累加"
---

# 第10章 GEMM-Like：矩阵乘类算子

## 本章导读

> 本章第一次让同一份输入被多个输出反复使用。公共部分从点积和小矩阵开始画 tile；HIP 篇显式管理 LDS 与线程 fragment，Triton 篇用 program/tile 表达同一复用，并把 autotune 限制为可解释的受控实验。

## 10.1 从点积看矩阵乘

用小矩阵说明输出元素、M/N/K 与 row-major 地址。

## 10.2 为什么朴素实现重复读取

区分算法级计算强度、源码请求字节和物理流量。

## 10.3 Tile 为什么能带来复用

先画块级数据生命周期，再进入代码。

## 10.4 HIP v0：一线程一输出

建立最短的正确点积 baseline。

## 10.5 HIP v1：LDS 分块

协作加载 A/B tile，处理同步、尾块与 bank 风险。

## 10.6 HIP v2/v3：一维与二维寄存器分块

让一个线程计算多个输出，同时跟踪 VGPR 压力。

## 10.7 HIP 进阶实验

只选择一项已实测的 K tile、双缓冲或 WMMA 机制。

## 10.8 Triton t0：用 tl.dot 写 tiled Matmul

解释 program id、M/N tile 与 K 循环。

## 10.9 Triton t1：program 排序与受控 autotune

限制搜索空间，并把选型结果落盘。

## 10.10 HIP 与 Triton 的分块层次对照

比较 block/thread fragment 与 program/tile。

## 10.11 复跑与练习

覆盖非方阵、非整除形状、转置布局与参数反例。

## 本章小结

- 本章当前只保留按新教程结构整理的大纲，旧正文、代码、日志和图片已清除。
- 正式写作时先完成 Radeon RX 9070 XT + ROCm 7.13 + 原生 Ubuntu 24.04 实验，再补命令、输出和性能结论。
- 新实验代码将放在 `code/part2-kernels/chapter10/`。

## 延伸阅读

- 待正式实验与正文完成后补充 AMD、ROCm、Triton 与矩阵乘相关资料。
