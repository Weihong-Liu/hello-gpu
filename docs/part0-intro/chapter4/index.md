---
title: "第4章 第一个程序 + 性能分析"
description: "Hello GPU 第4章 · vector add 跑通、baseline benchmark、CPU vs GPU 与带宽利用率分析"
---

# 第4章 第一个程序 + 性能分析

## 本章导读

> 前面几章我们铺开了环境验证和 GPU 体系结构两张地图。现在，地图已经在你手上了——终于到了**写一点代码、量一组数字**的时候。本章会做三件事：跑通第一个手写的 HIP kernel（vector add）、建立一个可复用的 baseline benchmark、再读懂量出来的这组数字——用算术强度和带宽利用率，建立「这个算子离硬件极限有多远」的直觉。vector add 会贯穿整个 Part 1 profiling 篇，所以这章是后面所有实验的起点。

本章对应代码在：

```text
code/part0-intro/
├── pyproject.toml
├── uv.lock
├── activate-rocm.sh
└── chapter4/
    ├── vector_add.hip
    └── benchmark_vector_add.py
```

## 4.1 从已经验证的环境开始

在写第一行 GPU 代码之前，有一件事必须确认：环境是通的。好在这件事[第 1 章](../chapter1/index.md)已经帮你做完了——三道环境验证门（`rocminfo` 能看到 GPU、PyTorch ROCm 能跑 GPU tensor、最小 HIP 程序能编译运行）都已经通过。如果你还没做，请先回去跑完那三道门。

进入本篇环境：

```bash
cd code/part0-intro
uv sync
source ./activate-rocm.sh
```

如果这里无法激活环境，先回到[第 1 章环境准备](../chapter1/index.md)排查 `uv` 环境、ROCm wheel 和 `_rocm_sdk_devel` 初始化问题。

## 4.2 跑通 vector add

环境就绪，开始写代码。一个最小但完整的 HIP 程序通常包含六步：Host 准备数据、Device 分配显存、Host 到 Device 拷贝、启动 kernel、Device 到 Host 拷回、检查结果。这六步构成了所有 GPU 程序的骨架，后面的 Reduction、Softmax、Matmul 无论多复杂，骨架都是这六步。

::: figure fig-vec-add-data-path
```mermaid
flowchart LR
    A[Host 输入] --> B[hipMalloc]
    B --> C[hipMemcpy H2D]
    C --> D[Kernel Launch]
    D --> E[hipMemcpy D2H]
    E --> F[校验结果]
```

最小 HIP Vector Add 程序的数据路径
:::

完整的 `vector_add.hip` 文件在[第 1 章 1.6 节](../chapter1/index.md#_1-6-验证最小-hip-程序)已经作为环境验证出现过，这里不重复贴。**真正值得重点看的是中间那段 kernel**，它才是跑在 GPU 上的代码：

```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}
```

这段 kernel 的映射关系很直接：一个 GPU 线程负责一个元素。`blockIdx.x * blockDim.x + threadIdx.x` 计算出当前线程负责的全局下标，`if (idx < n)` 用来处理最后一个 block 可能越界的情况。

新语法速查：

- `__global__`：告诉编译器「这是一个 GPU 函数，由 CPU 调用、在 GPU 上运行」；
- `<<<blocks, threads>>>`：HIP / CUDA 特有的 kernel 启动语法，`blocks` 是要启动多少组，`threads` 是每组多少个线程；
- `blockIdx.x` / `threadIdx.x`：每个 GPU 线程拿到的「工号」，用它来算自己负责数组里的哪个位置。

换成大白话：`vector_add<<<blocks, threads>>>(...)` 就是在 GPU 上同时叫起 `blocks × threads` 个工人，每个工人执行一次 `vector_add` 函数（如 @fig-block-thread-hierarchy 所示）。

::: figure fig-block-thread-hierarchy
![Block 与 Thread 的层级关系](./images/block-thread-hierarchy.png)

Grid → Block → Thread 的层级，以及 blockIdx/threadIdx 如何算出全局下标
:::

编译并运行：

```bash
cd chapter4
hipcc vector_add.hip -O2 -o vector_add && echo "compile_status: PASS"
./vector_add
```

<details>
<summary>输出：vector add 运行结果（RX 9070 XT）</summary>

```text
device_name: AMD Radeon RX 9070 XT
vector_size: 1048576
blocks: 4096
threads_per_block: 256
max_error: 0
status: PASS
```

</details>

看到 `status: PASS` 后，你已经跑通了第一段真正由自己编译的 GPU kernel——欢迎正式进入 GPU 编程的世界。

## 4.3 建立 baseline benchmark

这一节加入一个很小的 benchmark。它不是为了证明 Vector Add 有多快，而是为了提前建立后续章节会反复使用的习惯：固定输入规模、先 warmup、重复运行多次、记录 mean / median / min。

完整代码在下面的折叠块里。如果你只想先看懂大意，记住三件事（如 @fig-benchmark-warmup-repeat-sync 所示）：

1. 正式计时前先 warmup 5 次，让 GPU 进入比较稳定的状态；
2. 正式跑 30 次，每次用 `torch.cuda.Event` 量 GPU 真正执行完成的时间；
3. 最后用最小值估算带宽，因为最小值更像「没被外部干扰」的那次。

::: figure fig-benchmark-warmup-repeat-sync
![Benchmark 的赛前准备](./images/benchmark-warmup-repeat-sync.png)

准确 benchmark 前先 warmup、多次 repeat，并在同步后计时
:::

<details>
<summary>代码：benchmark_vector_add.py</summary>

```python
import argparse
import statistics
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark torch vector add on CPU and ROCm GPU.")
    parser.add_argument("--size", type=int, default=1 << 24)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    return parser.parse_args()


def benchmark_cpu(size, warmup, repeat):
    a = torch.ones(size, dtype=torch.float32)
    b = torch.full((size,), 2.0, dtype=torch.float32)

    for _ in range(warmup):
        c = a + b
    _ = c.sum().item()

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        c = a + b
        _ = c.sum().item()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return times


def benchmark_gpu(size, warmup, repeat):
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch ROCm backend is not available")

    a = torch.ones(size, device="cuda", dtype=torch.float32)
    b = torch.full((size,), 2.0, device="cuda", dtype=torch.float32)

    for _ in range(warmup):
        c = a + b
    torch.cuda.synchronize()
    _ = c.sum().item()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        c = a + b
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    _ = c.sum().item()
    return times


def summarize(name, times, size):
    mean_ms = statistics.mean(times)
    median_ms = statistics.median(times)
    min_ms = min(times)
    bytes_moved = size * 3 * 4
    bandwidth_gb_s = bytes_moved / (min_ms / 1000) / 1e9

    print(f"{name}_mean_ms: {mean_ms:.6f}")
    print(f"{name}_median_ms: {median_ms:.6f}")
    print(f"{name}_min_ms: {min_ms:.6f}")
    print(f"{name}_bandwidth_gb_s_by_min: {bandwidth_gb_s:.6f}")


def main():
    args = parse_args()
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device_name: {torch.cuda.get_device_name(0)}")
    print(f"vector_size: {args.size}")
    print(f"warmup: {args.warmup}")
    print(f"repeat: {args.repeat}")

    cpu_times = benchmark_cpu(args.size, args.warmup, args.repeat)
    gpu_times = benchmark_gpu(args.size, args.warmup, args.repeat)

    summarize("cpu", cpu_times, args.size)
    summarize("gpu", gpu_times, args.size)
    print("status: PASS")


if __name__ == "__main__":
    main()
```

</details>

其中 GPU 计时最关键的是使用 event 并在每轮后同步：

```python
start.record()
c = a + b
end.record()
torch.cuda.synchronize()
times.append(start.elapsed_time(end))
```

否则 CPU 端可能只是把任务提交出去，计到的不是 GPU 真正执行完成的时间。

运行：

```bash
python benchmark_vector_add.py
```

<details>
<summary>输出：Vector Add baseline benchmark @ AMD Radeon RX 9070 XT + ROCm 7.13</summary>

```text
torch: 2.11.0+rocm7.13.0
cuda_available: True
device_name: AMD Radeon RX 9070 XT
vector_size: 16777216
warmup: 5
repeat: 30
cpu_mean_ms: 8.681129
cpu_median_ms: 8.536416
cpu_min_ms: 7.356311
cpu_bandwidth_gb_s_by_min: 27.367874
gpu_mean_ms: 0.353474
gpu_median_ms: 0.346692
gpu_min_ms: 0.345132
gpu_bandwidth_gb_s_by_min: 583.332163
status: PASS
```

GPU min 延迟 0.345 ms，估算有效带宽约 583 GB/s——比 CPU（27.4 GB/s）快约 21 倍。

</details>

这里的带宽估算使用的是 Vector Add 的最简单数据量模型。一次 Vector Add 要读 `a`、读 `b`、写 `c`，一共经过 3 个数组；每个元素是 `float32`，也就是 4 字节。所以一次完整 Vector Add 搬动的数据量是：

```text
bytes_moved = vector_size × 3 × 4
```

实测 GPU 估算有效带宽约 583 GB/s——下一节就来读懂这组数字，看它离硬件极限有多远。

## 4.4 性能分析：CPU vs GPU 与带宽利用率

跑通和计时都做好了，现在来读懂这组数字。下面的数据全部来自 4.3 的 `benchmark_vector_add.py` 实测输出（16,777,216 个 float32，约 64 MiB/数组，warmup 5 + repeat 30，统计取 min）。

### 4.4.1 CPU vs GPU：到底快在哪

同一个 vector add，CPU 单核和 GPU 的实测对比：

| 指标 | CPU（单核） | GPU（仅 kernel） |
| ---- | ----: | ----: |
| 耗时（min） | ~7.36 ms | ~0.345 ms |
| 有效带宽（按 min 估算） | ~27.4 GB/s | ~583 GB/s |

GPU 快了约 **21 倍**。但注意：这个 21 倍是「数据已经在显存上、只量 kernel 执行」的口径——如果把 Host↔Device 的数据搬运也算进去，差距会小得多（对 vector add 这种简单操作，搬数据的开销可能比计算本身还大）。**GPU 的优势，在数据已经留在显存上、并且有多步计算可以复用它时最明显。**

### 4.4.2 算术强度：vector add 是访存受限

为什么 vector add 这么快、却又「没什么计算量」？看它的**算术强度**（Arithmetic Intensity，记作 $AI$）——每搬 1 Byte 数据做多少次计算：

$$
AI = \frac{\text{FLOPs}}{\text{Bytes}}
$$

对 vector add 的每个 float32 元素：读 `a[i]`（4 Byte）、读 `b[i]`（4 Byte）、写 `c[i]`（4 Byte），共 12 Byte；只做 1 次加法，即 1 FLOP。代入得：

$$
AI = \frac{1\ \text{FLOP}}{12\ \text{Byte}} \approx 0.083\ \text{FLOP/Byte}
$$

0.083 极低——每搬 12 字节才做 1 次加法。这意味着 vector add 是**典型的访存受限（memory-bound）算子**：它的快慢几乎完全取决于显存带宽，而不是算力。把算力堆得再高，对 vector add 也没用，因为瓶颈在搬数据。

### 4.4.3 带宽利用率：离硬件极限有多远

既然是访存受限，那就拿实测带宽和硬件上限比一比。先由 kernel 时间和搬运字节数算**有效带宽** $BW_{effective}$（vector add 读 2 个、写 1 个，共 3 个数组、每元素 4 Byte）：

$$
BW_{effective} = \frac{3 \times N \times 4\ \text{Byte}}{t}
$$

代入 4.3 的实测（$N = 16{,}777{,}216$，GPU min 时间 $t \approx 0.345\ \text{ms}$）：

$$
BW_{effective} = \frac{3 \times 16{,}777{,}216 \times 4\ \text{Byte}}{0.345\ \text{ms}} \approx 583\ \text{GB/s}
$$

RX 9070 XT 的标称显存带宽约 $640\ \text{GB/s}$（厂家规格，记作 $BW_{peak}$，见第 2 章），于是带宽利用率：

$$
\text{利用率} = \frac{BW_{effective}}{BW_{peak}} = \frac{583\ \text{GB/s}}{640\ \text{GB/s}} \approx 91\%
$$

对一个如此简单的 kernel 来说，~91% 已经相当不错——剩下的差距来自 launch 开销、计时边界、缓存与写路径等因素。这也说明 vector add 的访存效率已经很高（完全合并、线性流式），**优化空间不大；想再快，只能提高算术强度**（比如把多个逐元素操作融合成一个 kernel，让搬一次数据做更多计算）。

这套「先看算术强度判断瓶颈在带宽还是算力、再用实测带宽和上限比利用率」的思路，会在 Part 2 每个算子里反复用到：[第 8 章 Element-Wise](../../part2-kernels/chapter8/index.md) 会第一次系统地做这件事，[第 11 章 GEMM-Like](../../part2-kernels/chapter11/index.md)、[第 12 章 Fusion：融合算子](../../part2-kernels/chapter12/index.md) 则是高算术强度、吃算力的另一侧。

### 4.4.4 何时值得用 GPU

| 场景 | CPU 更合适 | GPU 更合适 |
| ---- | :---: | :---: |
| 数据量小（< 1 万元素） | ✅ | ❌ |
| 单次简单操作、数据不在显存 | ✅ | ❌ |
| 大数据、数据已在显存 | ❌ | ✅ |
| 大数据、多步串联计算 | ❌ | ✅ |
| 计算密集型（如矩阵乘） | ❌ | ✅ |

> 完整的 Roofline 读图方法——怎么把工作点画到「算术强度 vs 性能」图上、看它落在带宽斜线还是算力水平线那一侧、据此选排查方向——会在 part1 [第 7 章 读懂 Roofline 图](../../part1-profiling/chapter7/index.md) 系统讲；在那之前，[第 5 章](../../part1-profiling/chapter5/index.md) 和 [第 6 章](../../part1-profiling/chapter6/index.md) 会先教你把时间量准、用 `rocprof` 找到慢在哪个 kernel。

## 4.5 dispatch 开销：launch 本身要多久

前几节量的是 kernel 执行时间。但「调用一个 kernel」这件事本身也花时间——这段开销叫 **dispatch 开销（launch overhead）**。kernel 越短，它在总时间里占比越大。

### 4.5.1 一次 dispatch 发生了什么

从你的代码调用 `hipLaunchKernel` 到 GPU 真正开始执行，中间经过一条链路：

```text
host 端 HIP API
  → AMDGPU 内核驱动（KFD）
  → 写一个 AQL packet 到 queue（描述 grid、kernel 地址、参数）
  → 敲 doorbell（告诉 GPU「有新任务」）
  → GPU 端硬件调度器取 packet，分配到 CU
```

每一跳都有成本：API 调用本身的函数开销、驱动把参数打包进 packet 的开销、写 doorbell 的同步、以及 GPU 侧调度器取任务的时间。这些加起来就是 dispatch 开销。

### 4.5.2 实测：HIP 与裸金属的差距

在 RX 7900 XTX 上有人做过一个对照实验：同一套 AQL dispatch 流程，一边走标准 HIP 运行时，一边绕过 HIP 直接用 KFD ioctl 裸写 AQL packet（见延伸阅读），实测结果：

| 路径 | async（不等结果） | sync（等结果） |
| ---- | ----: | ----: |
| 标准 HIP | 2.6 μs | 20.5 μs |
| 裸金属 KFD | 2.26 μs | 14.96 μs |

两个结论：

1. **async dispatch 的绝对开销在个位数 μs 量级**，其中只有约 13% 是 HIP 运行时带来的（2.6 vs 2.26 μs）——大部分时间花在驱动与硬件的固定流程上，绕开 HIP 也省不掉多少。
2. **sync 的 20.5 μs 里大头是「等」**：sync dispatch 要等 GPU 把活儿干完再返回，所以时间包含任务执行。真正能优化的是 async 路径。

对本章的 vector add（kernel 约 0.345 ms）来说，2.6 μs 的 dispatch 开销占比不到 1%，可以忽略。但**对微 kernel 或者被拆得很碎的 kernel 序列，dispatch 开销会吃掉可观的比例**——第 17 章会看到一个真实例子：split-KV 把一次注意力拆成 160 次 launch，小输入时启动开销反而盖过了并行度收益。

### 4.5.3 对「CPU vs GPU」结论的修正

回看 4.4.1 的 21 倍加速，口径是「只量 kernel 执行」。把 dispatch 和 host 侧准备工作加进去，加速比会下降；数据再小一些，CPU 甚至会反超。所以「这个操作用 GPU 快不快」的正确问法是：**kernel 执行 + 数据搬运 + dispatch 的总账算下来，值不值**。这也是 4.4.4 那张「何时值得用 GPU」表里「数据量小、单次简单操作」选 CPU 的原因之一。

## 4.6 留下实验底稿

跑完前面三段命令，你大概觉得事情已经做完了——其实还没有。**性能工作真正麻烦的一刻，往往不是第一次没跑快，而是过几天回头看时，你自己也说不清当时跑了哪个版本、用了什么输入、那个数字到底是怎么量出来的。** 没留记录的实验，三天后基本等于白做。

所以从这第一个 GPU 程序开始，建议你养成一个小习惯：**每跑完一组实验，顺手把目标、命令、结果记到同一个地方**。在哪里记并不重要——一个 markdown 文件、一份 notebook 都行；重要的是这份记录能在几天后让你（或者别人）一眼看回当初做了什么。

够用的结构其实只有三段：**目标 → 流程 → 结论**。下面这个模板可以直接拿去用：

````markdown
# 实验记录：第一个 GPU 程序

## 实验目标

验证 PyTorch ROCm、最小 HIP kernel 和 Vector Add baseline benchmark 是否能跑通。

## 实测流程

（贴出可以从章节目录直接复制运行的命令）

```bash
cd code/part0-intro
source ./activate-rocm.sh
python chapter1/check_torch_rocm.py
cd chapter4
hipcc vector_add.hip -O2 -o vector_add
./vector_add
python benchmark_vector_add.py
```

## 实测结论

| 项目 | 数值 |
| ---- | ---- |
| 硬件 | AMD Radeon RX 9070 XT（gfx1201）+ ROCm 7.13（原生 Ubuntu 24.04）|
| 输入规模 | 16,777,216 个 float32（≈ 64 MiB/数组）|
| GPU min 延迟 | 0.345 ms |
| GPU 估算带宽 | ~583 GB/s |
| status | PASS |
````

看起来朴素，但半年后你回头翻这些记录，会非常感谢现在的自己。后面这本教程会一路写到 Reduction、Softmax、Matmul、Attention，外加一系列 rocprof 实验——等到 kernel 版本越积越多、benchmark 配置越改越乱时，**能不能一眼看回当初跑过什么**，往往就是「顺利继续」和「回头返工」的分界线。

试一试：把 `--size` 从默认的 `1 << 24` 改成 `1 << 20` 和 `1 << 26`，分别再跑一次 benchmark，把每次的硬件、输入规模、GPU min 延迟和估算带宽随手记到你的实验记录里。先猜一下——GPU 带宽会一直变大、一直变小，还是先升后降？这道题没有标准答案，目的是让你亲手建立"输入规模 vs 性能"的第一感觉，后面 Part 1 会反复用到。

## 本章小结

- 本章在 `part0-intro` 环境里跑通了第一个手写 HIP Vector Add kernel。
- 第一个 HIP kernel 使用"一线程处理一个元素"的最简单映射方式，方便理解 block、thread 和全局下标。
- baseline benchmark 使用 warmup + repeat，并用 GPU event 计时，避免只量到 CPU 提交开销。
- vector add 的算术强度极低（~0.083 FLOP/Byte），是典型的 访存受限算子——性能几乎完全取决于显存带宽，而非算力。
- 实测带宽 ~583 GB/s，约为标称 640 GB/s 的 91%；对这么简单的 kernel 已经不错，想再快只能提高算术强度（融合）。完整的 Roofline 读图方法在 Part 1 第 7 章。
- 下一章进入 Part 1 profiling 篇，先系统学怎么量准数字（benchmark 与可信计时）。

## 延伸阅读

- [ROCm HIP Documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [ROCm System Management Interface](https://rocm.docs.amd.com/projects/rocm_smi_lib/en/latest/)
