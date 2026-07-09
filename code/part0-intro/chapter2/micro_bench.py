"""Ch2 体系结构 micro-benchmark：实测 9070XT 的显存带宽与峰值算力。

跑两组：
  1. bandwidth：大数组 copy（远超 L2），测 GDDR6 平台带宽 B_peak
  2. compute ：torch.matmul fp16 / fp32，测 WMMA / SIMD 算力 P_peak

用法：
    python micro_bench.py
硬件上下文：Radeon RX 9070 XT（gfx1201）+ ROCm 7.13（原生 Ubuntu 24.04）
"""
import argparse

import torch


def bench_copy(mib: int, repeat: int = 50, warmup: int = 10):
    """大数组 copy：读 x + 写 y，搬运 2 × footprint 字节，测 GDDR6 带宽。"""
    n = mib * 1024 * 1024 // 4  # float32 元素个数
    x = torch.empty(n, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    for _ in range(warmup):
        y.copy_(x)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeat):
        y.copy_(x)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / repeat
    gbs = (2 * mib) / ms  # MiB/ms ≈ GB/s
    return ms, gbs


def bench_matmul(dtype, m: int = 4096, repeat: int = 30, warmup: int = 10):
    """torch.matmul：2·m³ FLOP，测峰值算力（fp16 走 WMMA，fp32 走 SIMD FMA）。"""
    a = torch.randn(m, m, dtype=dtype, device="cuda")
    b = torch.randn(m, m, dtype=dtype, device="cuda")
    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeat):
        c = a @ b
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / repeat
    flops = 2 * m**3
    tflops = flops / (ms / 1e3) / 1e12
    return ms, tflops


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matmul-size", type=int, default=4096)
    args = p.parse_args()

    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")
    print("=" * 60)

    print("\n--- 显存带宽（大数组 copy，测 GDDR6 平台 B_peak）---")
    print(f"{'footprint':>12} | {'min_ms':>9} | {'GB/s':>8}")
    print("-" * 38)
    for mib in [64, 256, 512, 1024, 2048]:
        ms, gbs = bench_copy(mib)
        print(f"{mib:>9} MiB | {ms:>7.3f} ms | {gbs:>7.1f}")

    print("\n--- 峰值算力（torch.matmul fp16/fp32，测 P_peak）---")
    print(f"{'dtype':>8} | {'min_ms':>9} | {'TFLOPS':>8}")
    print("-" * 32)
    for dt in [torch.float16, torch.float32]:
        name = {torch.float16: "fp16", torch.float32: "fp32"}[dt]
        ms, tf = bench_matmul(dt, m=args.matmul_size)
        print(f"{name:>8} | {ms:>7.3f} ms | {tf:>7.1f}")


if __name__ == "__main__":
    main()
