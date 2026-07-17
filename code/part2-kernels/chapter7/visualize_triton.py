"""Create a small, GPU-free Triton-viz trace for Chapter 7.

Triton-viz executes this teaching kernel with Triton's CPU interpreter. The
trace is for understanding program IDs, offsets, masks, and memory accesses;
it is not a Radeon RX 9070 XT performance experiment.
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl
import triton_viz


@triton_viz.trace("tracer")
@triton.jit
def traced_vector_add_kernel(
    input_a_ptr,
    input_b_ptr,
    output_ptr,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < size
    input_a = tl.load(input_a_ptr + offsets, mask=valid, other=0.0)
    input_b = tl.load(input_b_ptr + offsets, mask=valid, other=0.0)
    tl.store(output_ptr + offsets, input_a + input_b, mask=valid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=13)
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="open the live Triton-viz web UI after checking the trace",
    )
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")
    if args.block <= 0 or args.block & (args.block - 1):
        parser.error("--block must be a positive power of two")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main() -> int:
    args = parse_args()
    input_a = torch.arange(args.size, dtype=torch.float32)
    input_b = torch.arange(args.size, dtype=torch.float32) * 10
    output = torch.empty_like(input_a)

    grid = (triton.cdiv(args.size, args.block),)
    traced_vector_add_kernel[grid](
        input_a,
        input_b,
        output,
        args.size,
        BLOCK_SIZE=args.block,
    )

    reference = input_a + input_b
    if not torch.equal(output, reference):
        raise RuntimeError(
            f"trace execution is incorrect: output={output}, reference={reference}"
        )

    # PyPI triton-viz 3.0 exposes live visualization through launch(). The
    # current development branch also documents trace archives, but save/load
    # are not exported by the 3.0 wheel pinned in this project.
    from triton_viz.core.trace import launches

    if not launches or not launches[-1].records:
        raise RuntimeError("Triton-viz did not record any operations")
    record_count = len(launches[-1].records)
    print(
        "TRACE"
        f" size={args.size}"
        f" block={args.block}"
        f" programs={grid[0]}"
        f" records={record_count}"
        f" triton_viz={triton_viz.__version__}"
        " correctness=OK"
    )

    if args.launch:
        triton_viz.launch(share=False, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
