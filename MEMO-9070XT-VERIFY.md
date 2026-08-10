# MEMO · RX 9070 XT（书基线）数据复核清单

> 写给同事：全书审查发现 6 处涉及书基线 **RX 9070 XT（gfx1201）+ ROCm 7.13** 的数据问题，需要在这张卡上复核或确认原始实验记录后修正。优先级按从上到下排列。所有改动建议已给出，确认后改起来都很快。

## 1. ⭐ ch4 标称带宽 760 vs 640 GB/s（影响核心结论）

- **位置**：`docs/part0-intro/chapter4/index.md` §4.4.3（约 317-321 行，另见本章小结）
- **现状**：「RX 9070 XT 的标称显存带宽约 $760\ \text{GB/s}$」→ 带宽利用率 583/760 ≈ **77%**，结论「~77% 已经相当不错，优化空间不大」
- **疑点**：AMD 官方规格是 **640 GB/s**（256-bit × 20 Gbps；XFX 页 "up to 640 GB/s"，第三方 644.6）。本书自己的 ch2/ch3 也写「最高约 640 GB/s」——同一张卡两个规格。760 疑似与纹理填充率 760 GT/s 混淆
- **需要**：确认官方规格或实测标称带宽
- **若确认 640**：利用率 = 583/640 ≈ **91%**，§4.4.3 的「~77% 不错、差距来自 launch/缓存」结论需重写为「已接近上限」，并与 ch2/ch3 统一

## 2. ch7 Roofline 参考线 510 GB/s / 10.6 TFLOPS 无出处

- **位置**：`docs/part1-profiling/chapter7/index.md` §7.x（约 71 行、97 行）
- **现状**：「参考线沿用第 3 章独立测得的硬件基线：大数组 copy 的 GDDR6 稳态带宽为 510 GB/s，fp32 matmul 为 10.6 TFLOPS」
- **疑点**：ch3 全文没有 copy 实验，也没有 510 GB/s 或 10.6 TFLOPS（ch3 实测 stride-1 读 571 GB/s；ch5 大 footprint 稳态 568.6 GB/s）。全书同一硬件出现 510/571/568.6/583 四个数字
- **需要**：确认 510/10.6 的原始来源；若丢失，在 9070XT 上补测（大数组 copy + fp32 matmul 微基准），或把参考线改为引用 ch5 的 ~570 GB/s

## 3. ch5 copy 表格时间与带宽口径不一致

- **位置**：`docs/part1-profiling/chapter5/index.md` §5.x copy footprint 表（约 335-337 行，及原始输出块 351-353 行）
- **现状**：8 MiB pair → 0.020 ms、785.1 GB/s；64 MiB → 0.223 ms、572.8 GB/s；256 MiB → 0.900 ms、568.6 GB/s
- **疑点**：按「copy 按 2×footprint 字节」验算，8 MiB 行应得 838.9 GB/s（0.020ms），64 MiB 应得 601.9，256 MiB 应得 596.5——时间列与带宽列对不上。反推带宽列用的是 200 次 launch 窗口的平均时间（0.0214/0.2343/0.9442 ms），时间列标的是 min，两列口径混用
- **需要**：翻原始实验记录确认两列分别是什么统计量；统一口径并让 BW = bytes/t 自洽

## 4. ch6 rocprofv3 的 Grid_Size 列语义

- **位置**：`docs/part1-profiling/chapter6/index.md`（约 170-190 行，另见 ch8:322）
- **现状**：表头「Grid_Size = 一共启动了多少个 work-item」，表中 kernel_coalesced 记 16,777,216、stride=32 记 524,288
- **疑点**：rocprofv3（ROCm 7.13）的 kernel-trace CSV `Grid_Size_X/Y/Z` 文档语义是 thread block 数（block=256 时应为 65,536 / 2,048）；16,777,216/524,288 是旧 rocprof v1 的 work-item 惯例。比例结论（1/32）不受影响，但列含义与文档不符
- **需要**：在 9070XT + ROCm 7.13 上跑一次 ch6 的实验，对照 rocprofv3 实际输出确认；若 block 数，表格数字和表头都改

## 5. ch5 8 MiB 高带宽（785 GB/s）归因 L2 存疑

- **位置**：`docs/part1-profiling/chapter5/index.md`（约 363 行）
- **现状**：「8 MiB 时有效带宽冲到 785.1 GB/s（落在 L2 命中区，数据基本没往返 GDDR6）」
- **疑点**：ch3 自己定义 gfx1201 的 L2 只有 8 MiB——8 MiB 工作集恰好等于 L2 总量（还要与其它流量共享），整集命中不可能；785 GB/s 超过 640 GB/s 标称带宽，更可能命中 64 MiB MALL（Infinity Cache）
- **需要**：确认缓存层级归因（L2 命中区 vs MALL）；若确认 MALL，改表述并解释 gfx1201 的缓存层级

## 6. RDNA4/gfx1201 的 LDS 容量口径（已按官方规格修改，请复核）

- **位置**：`docs/part2-kernels/chapter11/index.md` §11.12.4（约 419-425 行）
- **现状**（已改）：表格 RDNA4 行 LDS 改为「128 KiB/WGP（单 workgroup 上限 64 KB）」，加脚注引用 ROCm gpu-specs 表
- **原问题**：该行原来写 64KB，与 ch3 的 128 KiB 矛盾，且与同表「BLOCK_K=128 最优」互斥
- **需要**：在 9070XT 上复核（rocminfo 或实验）per-WGP 128 KiB / workgroup 上限 64 KB 是否符合实际分配行为

## 背景信息（供参考，非待办）

- ch4 §4.5 与附录 C 的 dispatch 延迟数据（HIP 2.6/20.5 μs vs KFD 2.26/14.96 μs）来自 7900XTX 实测，书中已标注平台；若要在书基线上复测，另行安排
- 全书术语统一（memory-bound 叫法、wavefront 写法、型号写法等）是独立任务，不在本清单
