"""第 7 章：用 Triton 实现 program partial 与多阶段 reduction。

运行：
    python reduction_triton.py --version all --size 1048576 --block 1024 \
        --warmup 10 --repeat 50

``atomic`` 让每个 Triton program 先用 ``tl.sum`` 得到一个 partial，再把
partial 原子加到标量输出。``multistage`` 则让每个 program 把 partial 写入
中间缓冲区，并在 host 端反复启动相同的 chunk kernel，直到只剩一个值。

BLOCK_SIZE 必须是 2 的幂；本教学脚本进一步限制在 [32, 4096]，避免单个
program 创建过大的 tensor。多阶段版本每一轮都使用 ceil(current / BLOCK_SIZE)
个输出槽，因此会覆盖上一轮的所有 partial，不依赖 partial 数量小于 block。

本文件只提供正确性与 GPU event 计时骨架。性能数据必须在项目规定的
Radeon RX 9070 XT + ROCm 7.13 + 原生 Ubuntu 24.04 环境实测后再引用。
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def reduce_atomic_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Each program reduces one chunk, then performs one global atomic add."""
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    partial = tl.sum(values, axis=0)
    tl.atomic_add(output_ptr, partial)


@triton.jit
def reduce_chunks_kernel(
    input_ptr,
    partial_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce one input chunk into one output partial without atomics."""
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    partial = tl.sum(values, axis=0)
    tl.store(partial_ptr + program_id, partial)


@dataclass(frozen=True)
class TimingResult:
    minimum_ms: float | None
    median_ms: float | None
    mean_ms: float | None
    actual: float
    expected: float
    absolute_error: float
    precheck_correct: bool
    postcheck_correct: bool | None
    timed: bool
    correct: bool


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def num_warps_for(block_size: int) -> int:
    return 4 if block_size <= 1024 else 8


def make_multistage_buffers(
    input_tensor: torch.Tensor, block_size: int
) -> list[torch.Tensor]:
    """Allocate every recursive output once, outside the measured region."""
    buffers: list[torch.Tensor] = []
    current_size = input_tensor.numel()
    while current_size > 1:
        output_size = triton.cdiv(current_size, block_size)
        buffers.append(
            torch.empty(output_size, dtype=input_tensor.dtype, device=input_tensor.device)
        )
        current_size = output_size
    return buffers


def launch_multistage(
    input_tensor: torch.Tensor,
    buffers: list[torch.Tensor],
    block_size: int,
) -> torch.Tensor:
    """Recursively reduce all partials until one scalar remains."""
    source = input_tensor
    current_size = source.numel()
    for destination in buffers:
        reduce_chunks_kernel[(destination.numel(),)](
            source,
            destination,
            current_size,
            BLOCK_SIZE=block_size,
            num_warps=num_warps_for(block_size),
        )
        source = destination
        current_size = destination.numel()
    return source


def benchmark(
    *,
    launch: Callable[[], None],
    prepare: Callable[[], None],
    result_tensor: torch.Tensor,
    expected: float,
    warmup: int,
    repeat: int,
    rtol: float,
    atol: float,
) -> TimingResult:
    # JIT、launch、同步和读回都在计时前完成；预检失败便直接跳过计时。
    prepare()
    launch()
    torch.cuda.synchronize()
    precheck_actual = float(result_tensor.item())
    precheck_error = abs(precheck_actual - expected)
    precheck_correct = math.isclose(
        precheck_actual, expected, rel_tol=rtol, abs_tol=atol
    )
    if not precheck_correct:
        return TimingResult(
            minimum_ms=None,
            median_ms=None,
            mean_ms=None,
            actual=precheck_actual,
            expected=expected,
            absolute_error=precheck_error,
            precheck_correct=False,
            postcheck_correct=None,
            timed=False,
            correct=False,
        )

    for _ in range(warmup):
        prepare()
        launch()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for start, end in zip(starts, ends, strict=True):
        # prepare() 排在 start event 前的同一条 stream 上，不计入 kernel 时间。
        prepare()
        start.record()
        launch()
        end.record()
    torch.cuda.synchronize()

    times = [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
    actual = float(result_tensor.item())
    absolute_error = abs(actual - expected)
    correct = math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol)
    return TimingResult(
        minimum_ms=min(times),
        median_ms=statistics.median(times),
        mean_ms=statistics.fmean(times),
        actual=actual,
        expected=expected,
        absolute_error=absolute_error,
        precheck_correct=True,
        postcheck_correct=correct,
        timed=True,
        correct=precheck_correct and correct,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Triton reduction correctness and GPU-event timing skeleton"
    )
    parser.add_argument(
        "--version",
        choices=("atomic", "multistage", "all"),
        default="all",
        help="implementation to run (default: all)",
    )
    parser.add_argument("--size", type=int, default=1 << 20, help="input elements")
    parser.add_argument(
        "--block",
        type=int,
        default=1024,
        help="power-of-two chunk size in [32, 4096]",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-7,
        help="relative tolerance / 相对容差 (default: 1e-7)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="absolute tolerance / 绝对容差 (default: 1e-3)",
    )
    args = parser.parse_args()

    if args.size <= 0:
        parser.error("--size must be greater than zero")
    if not is_power_of_two(args.block) or not 32 <= args.block <= 4096:
        parser.error("--block must be a power of two in [32, 4096]")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be non-negative and --repeat must be positive")
    if args.rtol < 0.0 or args.atol < 0.0:
        parser.error("--rtol and --atol must be non-negative")
    return args


def print_result(name: str, result: TimingResult, stages: int) -> None:
    if not result.timed:
        print(f"{name:<10} stages={stages} precheck=FAIL timing=SKIPPED")
        print(
            f"           result={result.actual:.9g} "
            f"expected={result.expected:.9g} "
            f"abs_error={result.absolute_error:.6g} correctness=FAIL"
        )
        return

    assert result.minimum_ms is not None
    assert result.median_ms is not None
    assert result.mean_ms is not None
    print(
        f"{name:<10} stages={stages} "
        f"min_ms={result.minimum_ms:.6f} "
        f"median_ms={result.median_ms:.6f} "
        f"mean_ms={result.mean_ms:.6f}"
    )
    print(
        f"           result={result.actual:.9g} "
        f"expected={result.expected:.9g} "
        f"abs_error={result.absolute_error:.6g} "
        f"precheck=OK "
        f"postcheck={'OK' if result.postcheck_correct else 'FAIL'} "
        f"correctness={'OK' if result.correct else 'FAIL'}"
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is unavailable; run this script on the experiment host")

    indices = torch.arange(args.size, dtype=torch.int64, device="cpu")
    host_input = (1 + ((indices + args.seed) & 3)).to(torch.float32)
    del indices
    expected = float(host_input.to(torch.float64).sum().item())
    input_tensor = host_input.to(device="cuda")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch={torch.__version__} triton={triton.__version__} hip={torch.version.hip}")
    print(
        f"size={args.size} block={args.block} warmup={args.warmup} "
        f"repeat={args.repeat}"
    )
    print("block restriction: power-of-two in [32, 4096]")

    all_correct = True
    if args.version in ("atomic", "all"):
        atomic_output = torch.zeros(1, dtype=torch.float32, device="cuda")
        atomic_grid = (triton.cdiv(args.size, args.block),)

        def launch_atomic() -> None:
            reduce_atomic_kernel[atomic_grid](
                input_tensor,
                atomic_output,
                args.size,
                BLOCK_SIZE=args.block,
                num_warps=num_warps_for(args.block),
            )

        atomic_result = benchmark(
            launch=launch_atomic,
            prepare=atomic_output.zero_,
            result_tensor=atomic_output,
            expected=expected,
            warmup=args.warmup,
            repeat=args.repeat,
            rtol=args.rtol,
            atol=args.atol,
        )
        print_result("atomic", atomic_result, stages=1)
        all_correct = all_correct and atomic_result.correct

    if args.version in ("multistage", "all"):
        buffers = make_multistage_buffers(input_tensor, args.block)
        multistage_output = buffers[-1] if buffers else input_tensor

        def launch_recursive() -> None:
            launch_multistage(input_tensor, buffers, args.block)

        multistage_result = benchmark(
            launch=launch_recursive,
            prepare=lambda: None,
            result_tensor=multistage_output,
            expected=expected,
            warmup=args.warmup,
            repeat=args.repeat,
            rtol=args.rtol,
            atol=args.atol,
        )
        print_result("multistage", multistage_result, stages=len(buffers))
        all_correct = all_correct and multistage_result.correct

    return 0 if all_correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
