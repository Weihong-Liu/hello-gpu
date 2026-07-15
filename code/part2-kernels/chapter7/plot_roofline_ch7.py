#!/usr/bin/env python3
"""Plot the measured Chapter 7 Reduction points on the RX 9070 XT Roofline.

The point coordinates and labels come from ``results/summary.csv``.  The CSV is
normally produced by ``summarize_results.py`` after the HIP and Triton runs.

The bandwidth shown for each Reduction implementation is the tutorial's useful
effective-bandwidth metric, ``(4 * N + 4) / event_median_time``.  It is an
algorithm-level comparison metric, not measured physical DRAM, cache, atomic,
or partial-buffer traffic.

Examples:
    python plot_roofline_ch7.py
    python plot_roofline_ch7.py --summary results/summary.csv
    python plot_roofline_ch7.py --out /tmp/roofline-ch7.png
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# Independent reference measurements already used by the repository Roofline.
BW_GDDR6_GB_S = 510.0
P_FP32_TFLOP_S = 10.6

EXPECTED_KEYS = (
    "hip-v0",
    "hip-v1",
    "hip-v2",
    "hip-v3",
    "hip-v4",
    "triton-atomic",
    "triton-multistage",
)

DISPLAY_NAMES = {
    "hip-v0": "HIP v0: per-element atomic",
    "hip-v1": "HIP v1: Local Data Share tree",
    "hip-v2": "HIP v2: grid-stride + Local Data Share",
    "hip-v3": "HIP v3: grid-stride + wave",
    "hip-v4": "HIP v4: staged partials",
    "triton-atomic": "Triton: program sum + atomic",
    "triton-multistage": "Triton: multistage",
}

# Split nearby points across both sides of the shared AI coordinate.  In
# particular, v2/v4 go left while v3/Triton multistage go right, so the four
# fastest implementations form two readable rows instead of one text stack.
LEFT_LABEL_KEYS = {"hip-v1", "hip-v2", "hip-v4"}

REQUIRED_COLUMNS = {
    "backend",
    "implementation",
    "correct",
    "useful_ai_flop_per_byte",
    "useful_bandwidth_gb_s",
    "useful_gflop_s",
}


@dataclass(frozen=True)
class Point:
    key: str
    arithmetic_intensity: float
    performance_tflop_s: float
    useful_bandwidth_gb_s: float


def parse_positive_float(value: str, *, field: str, key: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{key}: {field} is not a number: {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{key}: {field} must be finite and greater than zero")
    return parsed


def parse_correct(value: str, *, key: str) -> None:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return
    if normalized in {"false", "0", "no"}:
        raise ValueError(f"{key}: correctness check failed; refusing to plot it")
    raise ValueError(f"{key}: correct must be true or false, got {value!r}")


def load_points(summary_path: Path) -> list[Point]:
    if not summary_path.is_file():
        raise ValueError(
            f"summary CSV not found: {summary_path}. "
            "Generate it with summarize_results.py or pass --summary."
        )

    try:
        with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"summary CSV has no header: {summary_path}")
            missing_columns = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
            if missing_columns:
                raise ValueError(
                    "summary CSV is missing required columns: "
                    + ", ".join(missing_columns)
                )
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"cannot read summary CSV {summary_path}: {exc}") from exc

    by_key: dict[str, Point] = {}
    for line_number, row in enumerate(rows, start=2):
        backend = (row.get("backend") or "").strip().lower()
        implementation = (row.get("implementation") or "").strip().lower()
        key = f"{backend}-{implementation}"
        if key not in EXPECTED_KEYS:
            raise ValueError(
                f"line {line_number}: unexpected implementation {key!r}; "
                f"expected {', '.join(EXPECTED_KEYS)}"
            )
        if key in by_key:
            raise ValueError(f"line {line_number}: duplicate implementation {key}")

        parse_correct(row["correct"], key=key)
        ai = parse_positive_float(
            row["useful_ai_flop_per_byte"],
            field="useful_ai_flop_per_byte",
            key=key,
        )
        useful_bandwidth = parse_positive_float(
            row["useful_bandwidth_gb_s"],
            field="useful_bandwidth_gb_s",
            key=key,
        )
        useful_gflop_s = parse_positive_float(
            row["useful_gflop_s"], field="useful_gflop_s", key=key
        )

        expected_gflop_s = ai * useful_bandwidth
        if not math.isclose(
            useful_gflop_s, expected_gflop_s, rel_tol=5e-3, abs_tol=1e-9
        ):
            raise ValueError(
                f"{key}: useful metrics disagree: useful_gflop_s="
                f"{useful_gflop_s:.6g}, but AI * useful_bandwidth="
                f"{expected_gflop_s:.6g}"
            )

        by_key[key] = Point(
            key=key,
            arithmetic_intensity=ai,
            performance_tflop_s=useful_gflop_s / 1_000.0,
            useful_bandwidth_gb_s=useful_bandwidth,
        )

    missing_points = [key for key in EXPECTED_KEYS if key not in by_key]
    if missing_points:
        raise ValueError(
            "summary CSV is missing implementations: " + ", ".join(missing_points)
        )
    return [by_key[key] for key in EXPECTED_KEYS]


def spread_label_positions(
    points: list[Point], minimum_gap_decades: float = 0.34
) -> dict[str, float]:
    """Keep labels separated independently on the left and right sides."""
    positions: dict[str, float] = {}
    for side in ("left", "right"):
        previous_log_y: float | None = None
        side_points = [
            point
            for point in points
            if ("left" if point.key in LEFT_LABEL_KEYS else "right") == side
        ]
        for point in sorted(side_points, key=lambda item: item.performance_tflop_s):
            log_y = math.log10(point.performance_tflop_s)
            if previous_log_y is not None:
                log_y = max(log_y, previous_log_y + minimum_gap_decades)
            positions[point.key] = 10**log_y
            previous_log_y = log_y
    return positions


def plot(points: list[Point], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 6.4), dpi=150)

    ai_reference = np.logspace(-2, 3, 600)
    axis.plot(
        ai_reference,
        (BW_GDDR6_GB_S / 1_000.0) * ai_reference,
        color="#2563eb",
        linewidth=1.8,
        label=(
            "GDDR6 (Graphics Double Data Rate 6) reference = "
            f"{BW_GDDR6_GB_S:g} GB/s"
        ),
    )
    axis.axhline(
        P_FP32_TFLOP_S,
        color="#dc2626",
        linewidth=1.8,
        linestyle="--",
        label=(
            "FP32 (32-bit floating-point) reference = "
            f"{P_FP32_TFLOP_S:g} TFLOPS"
        ),
    )

    backend_style = {
        "hip": {"color": "#0f766e", "marker": "o"},
        "triton": {"color": "#c2410c", "marker": "D"},
    }
    for backend, style in backend_style.items():
        backend_points = [point for point in points if point.key.startswith(backend)]
        axis.scatter(
            [point.arithmetic_intensity for point in backend_points],
            [point.performance_tflop_s for point in backend_points],
            color=style["color"],
            marker=style["marker"],
            s=68,
            zorder=5,
            edgecolors="black",
            linewidths=0.55,
            label=f"{backend.upper()} measured points",
        )

    label_positions = spread_label_positions(points)
    shared_ai = max(point.arithmetic_intensity for point in points)
    label_x = {
        "left": shared_ai / 1.4,
        "right": shared_ai * 2.0,
    }
    for point in points:
        color = backend_style[point.key.split("-", 1)[0]]["color"]
        side = "left" if point.key in LEFT_LABEL_KEYS else "right"
        axis.annotate(
            f"{DISPLAY_NAMES[point.key]}\n"
            f"{point.performance_tflop_s * 1_000:.3g} GFLOPS | "
            f"useful bandwidth {point.useful_bandwidth_gb_s:.3g} GB/s",
            xy=(point.arithmetic_intensity, point.performance_tflop_s),
            xytext=(label_x[side], label_positions[point.key]),
            textcoords="data",
            fontsize=7.7,
            color="#111827",
            ha="right" if side == "left" else "left",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.84,
            },
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9},
        )

    minimum_point_y = min(point.performance_tflop_s for point in points)
    maximum_label_y = max(label_positions.values())
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1e-2, 1e3)
    axis.set_ylim(
        min(1e-3, minimum_point_y / 3.0),
        max(30.0, maximum_label_y * 2.0),
    )
    axis.set_xlabel("Useful arithmetic intensity (FLOP / Byte)", fontsize=10.5)
    axis.set_ylabel("Useful performance (TFLOPS, event median)", fontsize=10.5)
    axis.set_title(
        "Roofline: Chapter 7 Reduction on Radeon RX 9070 XT", fontsize=12
    )
    axis.grid(True, which="both", linestyle=":", alpha=0.3)
    axis.legend(loc="upper left", fontsize=7.7, framealpha=0.94)

    figure.text(
        0.5,
        0.036,
        "GFLOPS = giga floating-point operations per second; "
        "TFLOPS = tera floating-point operations per second.",
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#374151",
    )
    figure.text(
        0.5,
        0.012,
        "N = input element count; useful bandwidth = (4N + 4 bytes) / event "
        "median time. This is not measured physical dynamic random-access "
        "memory (DRAM), cache, atomic, or partial-buffer traffic.",
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#374151",
    )
    figure.tight_layout(rect=(0.02, 0.10, 0.99, 0.99))

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight")
    except OSError as exc:
        raise ValueError(f"cannot save plot to {output_path}: {exc}") from exc
    finally:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_output = (
        script_dir.parents[2]
        / "docs"
        / "part2-kernels"
        / "chapter7"
        / "images"
        / "roofline-ch7.png"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=script_dir / "results" / "summary.csv",
        help="Chapter 7 summary CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_output,
        help="output image path (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        points = load_points(args.summary)
        plot(points, args.out)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
