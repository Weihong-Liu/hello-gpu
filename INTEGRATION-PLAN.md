# dev 分支审计报告 + 实验融入兼容计划

> 生成日期：2026-08-02 ｜ 对象：`dev` 分支（20 章 5 篇，part0=0-4, part1=5-7, part2=8-13, part3=14-17, part4=18-19）
> 基线：Radeon RX 9070 XT (gfx1201) + ROCm 7.13 + 原生 Ubuntu 24.04

---

## 一、宏观审计

### 1.1 篇级平衡：前实后空

| 篇 | 章数 | 内容量（行/章） | 状态 |
|---|---|---|---|
| part0 入门与硬件速通 | 5 | 206–568 | 实心（ch1 厚，ch0 薄） |
| part1 Profiling 实战 | 3 | 227–427 | 中厚（ch7 偏薄） |
| part2 算子实战 | 6 | 200–735 | 实心主力，但内部严重不均 |
| part3 Agent（算子层） | 4 | 44 | **纯骨架**（H2 占位 + 一句话） |
| part4 模型 + Agent | 2 | 48 | **纯骨架** |

全书 20 章中 6 章（30%）是骨架，而骨架恰好集中在本书的差异化卖点（Agent 篇）上——这也是最该投入的地方。

### 1.2 章级 balance：section 太多 vs 内容太少

按「行数 / H2 数」算出每节平均内容密度，判断标准：节均 <30 行 = 标题堆砌；>60 行 = 可能过深。

| 章 | 行数 | H2 | H3 | 代码块 | 图 | 节均行数 | 诊断 |
|---|---|---|---|---|---|---|---|
| ch0 写给读者的话 | 206 | 10 | 0 | 4 | 2 | 21 | 薄，可接受（导读章） |
| ch1 环境准备 | 568 | 11 | 0 | 50 | 1 | 52 | 厚实，OK |
| ch2 GPU 体系结构（上） | 274 | 8 | 0 | 20 | 6 | 34 | 适中 |
| ch3 GPU 体系结构（下） | 395 | 10 | 0 | 26 | 6 | 40 | 适中 |
| ch4 第一个程序 + 性能分析 | 396 | 11 | 4 | 26 | 5 | 36 | 适中 |
| ch5 benchmark 与可信计时 | 427 | 9 | 4 | 14 | 2 | 47 | 适中 |
| ch6 用 rocprof 找慢点 | 304 | 9 | 2 | 26 | 3 | 34 | 适中 |
| ch7 读懂 Roofline | 227 | **13** | 0 | 12 | 4 | **17** | ⚠️ **节太碎**：13 节装 227 行 |
| ch8 Element-Wise | 735 | 12 | **30** | 54 | 9 | 61 | ⚠️ 厚且 H3 深嵌套 |
| ch9 Reduction | 410 | 14 | 4 | 40 | 1 | 29 | 适中 |
| ch10 Softmax | 633 | **17** | **30** | 58 | 1 | 37 | ⚠️ **H2 太多**（17）+ H3 深嵌套 |
| ch11 GEMM-Like | 520 | **17** | 10 | 54 | 1 | 31 | ⚠️ **H2 太多**（17） |
| ch12 Fusion | **200** | 14 | 0 | 18 | 1 | **14** | ❌ **内容太少 + 节太碎** |
| ch13 Fused RMSNorm | **246** | 15 | 0 | 24 | 1 | **16** | ❌ **内容太少 + 节太碎** |
| ch14–17 Agent | 44×4 | 9×4 | 0 | 0 | 0 | ~5 | ❌ 骨架 |

### 1.3 三个结构性问题

1. **孤儿目录**：`docs/part1-hardware-rocm/`（ch3–6 仅剩 images，无内容、无任何引用）——重构残留，应删除。
2. **outline drift**：ch12/ch13 有手动加的 `## 正式实验结果` 节，不在 `outline.mjs` 的 sections 里，`npm run docs:check-outline` 会报漂移。
3. **图严重不足**：ch10/ch11/ch12/ch13 各只有 1 张图。GEMM、Softmax、Fusion 都是数据流密集型章节，图是刚需；对比 ch8 有 9 张。这是最影响可读性的短板。

### 1.4 内容一致性风险

part2 的「正式实验结果」在 7900XTX (gfx1100) / ROCm 7.2 上测得，与书基线 9070XT (gfx1201) / ROCm 7.13 不同。数字引用需注明平台，或在新硬件上复测。

---

## 二、兼容计划：融入 4 个实验

**原则**：不改动 dev 现有 20 章编号与大纲骨架；实验素材以「扩充现有 section / 补充新增小节 / 填实骨架」三种方式融入；所有跨平台数据标注来源硬件；每项给出素材来源与工作量。

### P1：bare-metal-1100 → KFD 裸金属视角 ⭐

**位置**：ch4（第一个程序 + 性能分析）补 1 小节 + 附录 C（可选）
**内容**：
- ch4 新增「dispatch 开销：launch 本身要多久」小节：HIP async dispatch 2.6μs 对比裸金属 KFD 2.26μs（7900XTX 实测）——给「CPU vs GPU」对比补上启动开销维度（对短 kernel 尤其致命）
- 附录 C「裸金属视角：绕过 HIP 直接写 AQL」：KFD ioctl → AQL packet → doorbell 流程，含 ISA 级编码实战（v_perm_b32 等 4 个编码 bug 的发现过程，llvm-mc 验证）——同时填补「ISA 级分析」主题
- ch11 延伸阅读引用 t0-gpu 的 GEMM 对比（4096³：66.7 TF vs rocBLAS 90.8 TF）
**素材**：`/mnt/luyuzhou/hpc/bare-metal-1100/`（experiments/、rdna35_instruction_set_architecture.md、results.*）
**硬件**：7900XTX (gfx1100) —— 需注明与书基线差异，方法可复现
**工作量**：中（ch4 小节 0.5 天；附录 C 1.5 天）

### P2：kernel-agent → part3-agent 四章血肉 ⭐⭐ 优先级最高

**位置**：ch15（工具封装）、ch16（Agent 设计）、ch17（多轮优化实战）
**内容**：
- ch15 工具封装：以 KDA 的 task-contract + bench harness + profile 流程为范本（工具 schema、错误回传触发反思）
- ch16 Agent 设计：KDA 循环架构（读题→draft→plan→实现→验证→profiling→迭代），配 kernel-agent 的 `agent_optimize.py` 循环脚本骨架
- ch17 多轮优化实战：**用 FA Decode 真实轨迹**——naive→split-KV→4.17x/5.13x/5.26x（N=2048/4096/8192），NCU 显示 69.5% DRAM 利用率；附 Paged Attention 3.75x（40.2M→150.9M tok/s）
- **核心教学点（书的差异化卖点）**：如实呈现「纯 Agent 自动优化贡献≈0%（15 次 API 调用仅 1 次通过），3-5x 是人+工具+Agent 协作达成」——与市面夸大 Agent 的教程形成对比；以及反直觉教训：M=1 decode 用 tl.dot 反而更慢（WMMA 16x16 padding 浪费 99.6%）
**素材**：`/mnt/luyuzhou/hpc/kernel-agent/`（README.md 实验总表、task-workspace-fa-decode*/、7900xtx-assm-optimize/experiment-report.md、kernel-design-agents/docs/agent-flow.md）
**硬件**：FA decode 实验在 RTX 3080（CUDA/Triton），fused MLP 在 7900XTX；方法论通用，需标注平台
**工作量**：大（3 章从骨架到实心，各 1.5-2 天），但素材几乎齐备

### P3：tile-optimizer → ch11 GEMM 深化 ⭐

**位置**：ch11 新增「tile 形状怎么选」小节（替换/扩展现有 §11.8 autotune 受控实验）
**内容**：
- tile 敏感性：MoE shape 上最优/最差 tile 差 **12.8x**（16x64x32=7.50 vs 128x256x32=0.59 TFLOPS，RDNA3）
- **反直觉结论「tile 形状 > 面积」**：128x64 与 64x128 同面积差 1.42x
- BLOCK_K=32 跨平台最优；Triton 默认 autotuner 在消费卡上选次优（1024³ spread 11.6x）
- 工具对比：Triton vs Composable Kernel +31%（7.50 TF）——填补「工具对比」缺失主题
- 硬件先验 6 条裁剪规则（剪 95%+ 搜索空间）——可直接对接 part3 Agent 设计（ch16）
**素材**：`/mnt/luyuzhou/hpc/tile-optimizer/`（README.md、experiments/rdna3、rdna3.5、rdna4、benchmarks/ck_gemm_bench、models/rdna3/lambdarank_v2.txt）
**硬件**：RDNA3 (7900XTX) 主力数据 + **RDNA4 (gfx1201) 20 shapes + RDNA3.5 (gfx1151) 20 shapes** —— 后者与书基线（gfx1201）和你本机（gfx1151）直接吻合
**工作量**：中（1-1.5 天，数据现成）

### P4：llama.cpp MoE 解码 sweep → ch19 decode 视角 ⭐

**位置**：ch19（小模型 LLM 解码 + Agent）的「decode 算子视角」小节
**内容**：
- 用 LFM2.5-8B-A1B-Q4_K_M 的 RPB/nwarps 扫描建立「decode = 带宽受限 GEMV」的实证：
  RPB=1/NW=8 → 214.6 t/s，RPB=4 → 293.8 t/s，RPB=8 → 285.5 t/s（≈1.37x，5 次重复带标准差）
- 与 Agent 流程结合：这正是 ch16/17 工具封装的目标场景（对推理框架的发射配置做受控实验）
- 顺带补充 llama.cpp HIP 构建流程（附录 A 环境章节的延伸）
**素材**：`/mnt/luyuzhou/hpc/llama.cpp/results/moe_sweep_20260706_1855.txt`
**硬件**：ROCm 后端实测（GPU 型号需复核——results 头文件误报 CPU 名，预计 7900XTX）
**工作量**：小（0.5 天，数据完整可直接改写）

### 顺带修复（审计项，建议与 P2 一起做）

- 删除 `docs/part1-hardware-rocm/` 孤儿目录
- 把 ch12/ch13 的 `## 正式实验结果` 并入 outline.mjs sections 或降为 H3，消除 drift
- 为 ch10/ch11/ch12/ch13 各补 1-2 张数据流图（ch8 有 9 张，对比明显）
- ch7 Roofline 13 节合并到 8-9 节；ch12 14 节、ch13 15 节合并到 10-11 节（节均 20+ 行）

---

## 三、建议实施顺序

1. **P2（Agent 篇血肉）**——6 章骨架中 4 章直接受益，素材最齐，优先级最高
2. **P3（GEMM tile）**——数据现成且含书基线架构数据，中工作量
3. **P1（KFD + ISA）**——差异化最强（书中完全缺失），但附录 C 工作量略大
4. **P4（decode 实证）**——小工作量，与 ch19 骨架同时填
5. 审计修复项随时穿插

> 注：P2 的 FA decode 实验在 NVIDIA 3080 上，P1/P4 在 AMD 上。书中统一标注平台；若要在书基线（9070XT）上复测，需另行安排跑数时间。
