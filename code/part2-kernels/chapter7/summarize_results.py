#!/usr/bin/env python3
"""Build a machine-readable Chapter 7 benchmark and kernel-trace summary."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


IMPLEMENTATION_ORDER = (
    "hip-v0",
    "hip-v1",
    "hip-v2",
    "hip-v3",
    "hip-v4",
    "triton-atomic",
    "triton-multistage",
)

KERNEL_PATTERNS = {
    "hip-v0": ("reduce_v0_atomic",),
    "hip-v1": ("reduce_v1_lds_atomic",),
    "hip-v2": ("reduce_v2_grid_stride_lds_atomic",),
    "hip-v3": ("reduce_v3_wave_atomic",),
    "hip-v4": ("reduce_v4_partials",),
    "triton-atomic": ("reduce_atomic_kernel",),
    "triton-multistage": ("reduce_chunks_kernel",),
}

FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
HIP_RESULT_RE = re.compile(
    rf"^(v[0-4])\s+grid=(\d+)\s+block=(\d+)\s+"
    rf"min_ms=({FLOAT_PATTERN})\s+median_ms=({FLOAT_PATTERN})\s+"
    rf"mean_ms=({FLOAT_PATTERN})$"
)
HIP_FAILED_RE = re.compile(
    r"^(v[0-4])\s+grid=(\d+)\s+block=(\d+)\s+"
    r"precheck=FAIL\s+timing=SKIPPED$"
)
TRITON_RESULT_RE = re.compile(
    rf"^(atomic|multistage)\s+stages=(\d+)\s+"
    rf"min_ms=({FLOAT_PATTERN})\s+median_ms=({FLOAT_PATTERN})\s+"
    rf"mean_ms=({FLOAT_PATTERN})$"
)
TRITON_FAILED_RE = re.compile(
    r"^(atomic|multistage)\s+stages=(\d+)\s+"
    r"precheck=FAIL\s+timing=SKIPPED$"
)


def numeric(value: str | None, *, integer: bool = False) -> int | float | None:
    if value is None or value == "":
        return None
    try:
        return int(value) if integer else float(value)
    except ValueError:
        return None


def first_value(row: dict[str, str], *names: str) -> str | None:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    return None


def load_profile_config(path: Path) -> dict[str, int | str]:
    config: dict[str, int | str] = {}
    if not path.is_file():
        return config
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        config[key.strip()] = int(value) if value.isdigit() else value
    return config


def parse_header(lines: list[str]) -> dict[str, int]:
    for line in lines:
        match = re.match(r"^size=(\d+)\s+block=(\d+)\b", line.strip())
        if match:
            return {"size": int(match.group(1)), "block": int(match.group(2))}
    return {}


def parse_correctness(line: str) -> dict[str, Any]:
    correctness_match = re.search(r"\bcorrectness=(OK|FAIL)\b", line)
    precheck_match = re.search(r"\bprecheck=(OK|FAIL)\b", line)
    postcheck_match = re.search(r"\bpostcheck=(OK|FAIL)\b", line)
    result_match = re.search(rf"\bresult=({FLOAT_PATTERN})\b", line)
    expected_match = re.search(rf"\bexpected=({FLOAT_PATTERN})\b", line)
    error_match = re.search(rf"\babs_error=({FLOAT_PATTERN})\b", line)
    return {
        "correct": (
            correctness_match.group(1) == "OK" if correctness_match else None
        ),
        "precheck_correct": (
            precheck_match.group(1) == "OK" if precheck_match else None
        ),
        "postcheck_correct": (
            postcheck_match.group(1) == "OK" if postcheck_match else None
        ),
        "result": float(result_match.group(1)) if result_match else None,
        "expected": float(expected_match.group(1)) if expected_match else None,
        "absolute_error": float(error_match.group(1)) if error_match else None,
    }


def parse_benchmark_log(
    path: Path, backend: str, warnings: list[str]
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        warnings.append(f"missing benchmark log: {path}")
        return {}

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = parse_header(lines)
    size = header.get("size")
    parsed: dict[str, dict[str, Any]] = {}
    current_key: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if backend == "hip":
            match = HIP_RESULT_RE.match(line)
            failed_match = HIP_FAILED_RE.match(line)
            if match:
                version, grid, block, minimum, median, mean = match.groups()
                current_key = f"hip-{version}"
                parsed[current_key] = {
                    "backend": "hip",
                    "implementation": version,
                    "size": size,
                    "grid": int(grid),
                    "block": int(block),
                    "stages": 2 if version == "v4" else 1,
                    "minimum_ms": float(minimum),
                    "median_ms": float(median),
                    "mean_ms": float(mean),
                    "timed": True,
                    "source_file": str(path),
                }
                continue
            if failed_match:
                version, grid, block = failed_match.groups()
                current_key = f"hip-{version}"
                parsed[current_key] = {
                    "backend": "hip",
                    "implementation": version,
                    "size": size,
                    "grid": int(grid),
                    "block": int(block),
                    "stages": 2 if version == "v4" else 1,
                    "minimum_ms": None,
                    "median_ms": None,
                    "mean_ms": None,
                    "timed": False,
                    "source_file": str(path),
                }
                continue
        else:
            match = TRITON_RESULT_RE.match(line)
            failed_match = TRITON_FAILED_RE.match(line)
            if match:
                version, stages, minimum, median, mean = match.groups()
                current_key = f"triton-{version}"
                parsed[current_key] = {
                    "backend": "triton",
                    "implementation": version,
                    "size": size,
                    "grid": None,
                    "block": header.get("block"),
                    "stages": int(stages),
                    "minimum_ms": float(minimum),
                    "median_ms": float(median),
                    "mean_ms": float(mean),
                    "timed": True,
                    "source_file": str(path),
                }
                continue
            if failed_match:
                version, stages = failed_match.groups()
                current_key = f"triton-{version}"
                parsed[current_key] = {
                    "backend": "triton",
                    "implementation": version,
                    "size": size,
                    "grid": None,
                    "block": header.get("block"),
                    "stages": int(stages),
                    "minimum_ms": None,
                    "median_ms": None,
                    "mean_ms": None,
                    "timed": False,
                    "source_file": str(path),
                }
                continue

        if current_key is not None and "correctness=" in line:
            parsed[current_key].update(parse_correctness(line))
            current_key = None

    for key, result in parsed.items():
        if "correct" not in result:
            result.update(
                {
                    "correct": False if not result["timed"] else None,
                    "precheck_correct": False if not result["timed"] else None,
                    "postcheck_correct": None,
                    "result": None,
                    "expected": None,
                    "absolute_error": None,
                }
            )
            warnings.append(f"missing correctness detail for {key} in {path}")
    return parsed


def aggregate_independent_runs(
    runs: list[dict[str, dict[str, Any]]], warnings: list[str]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for key, result in run.items():
            grouped[key].append(result)

    aggregated: dict[str, dict[str, Any]] = {}
    for key, samples in grouped.items():
        baseline = samples[0]
        for field in ("size", "grid", "block", "stages"):
            values = {sample.get(field) for sample in samples}
            if len(values) > 1:
                warnings.append(
                    f"independent runs disagree for {key} field {field}: "
                    f"{sorted(str(value) for value in values)}"
                )

        minimum_values = [
            sample["minimum_ms"]
            for sample in samples
            if isinstance(sample.get("minimum_ms"), (int, float))
        ]
        median_values = [
            sample["median_ms"]
            for sample in samples
            if isinstance(sample.get("median_ms"), (int, float))
        ]
        mean_values = [
            sample["mean_ms"]
            for sample in samples
            if isinstance(sample.get("mean_ms"), (int, float))
        ]
        median_range = (
            {
                "minimum_ms": min(median_values),
                "maximum_ms": max(median_values),
                "span_ms": max(median_values) - min(median_values),
            }
            if median_values
            else None
        )
        aggregated[key] = {
            "backend": baseline.get("backend"),
            "implementation": baseline.get("implementation"),
            "size": baseline.get("size"),
            "grid": baseline.get("grid"),
            "block": baseline.get("block"),
            "stages": baseline.get("stages"),
            "minimum_ms": (
                statistics.median(minimum_values) if minimum_values else None
            ),
            "median_ms": statistics.median(median_values) if median_values else None,
            "mean_ms": statistics.median(mean_values) if mean_values else None,
            "timed": all(sample.get("timed") is True for sample in samples),
            "correct": all(sample.get("correct") is True for sample in samples),
            "precheck_correct": all(
                sample.get("precheck_correct") is True for sample in samples
            ),
            "postcheck_correct": all(
                sample.get("postcheck_correct") is True for sample in samples
            ),
            "result": baseline.get("result"),
            "expected": baseline.get("expected"),
            "absolute_error": max(
                (
                    sample["absolute_error"]
                    for sample in samples
                    if isinstance(sample.get("absolute_error"), (int, float))
                ),
                default=None,
            ),
            "source_files": [sample["source_file"] for sample in samples],
            "independent_runs": {
                "count": len(samples),
                "minimum_ms_values": minimum_values,
                "median_ms_values": median_values,
                "mean_ms_values": mean_values,
                "median_ms_range": median_range,
                "aggregation": "median of per-process statistics",
            },
        }
    return aggregated


def parse_run_directory(
    path: Path, warnings: list[str]
) -> dict[str, dict[str, Any]]:
    if not path.is_dir():
        return {}
    runs: list[dict[str, dict[str, Any]]] = []
    for log_path in sorted(path.glob("hip*.log")):
        runs.append(parse_benchmark_log(log_path, "hip", warnings))
    for log_path in sorted(path.glob("triton*.log")):
        runs.append(parse_benchmark_log(log_path, "triton", warnings))
    return aggregate_independent_runs(runs, warnings)


def attach_useful_metrics(result: dict[str, Any]) -> None:
    size = result.get("size")
    median_ms = result.get("median_ms")
    if not isinstance(size, int) or size <= 0:
        result["useful_metrics"] = None
        return

    useful_flops = size - 1
    useful_bytes = 4 * size + 4
    arithmetic_intensity = useful_flops / useful_bytes
    metrics: dict[str, float | int | None] = {
        "flops": useful_flops,
        "bytes": useful_bytes,
        "arithmetic_intensity_flop_per_byte": arithmetic_intensity,
        "based_on": "event_median_ms",
        "bandwidth_gb_s": None,
        "gflop_s": None,
    }
    if isinstance(median_ms, (int, float)) and median_ms > 0:
        metrics["bandwidth_gb_s"] = useful_bytes / (median_ms * 1.0e6)
        metrics["gflop_s"] = useful_flops / (median_ms * 1.0e6)
    result["useful_metrics"] = metrics


def read_trace_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def classify_trace(path: Path, rows: list[dict[str, str]]) -> set[str]:
    normalized_name = str(path).lower().replace("_", "-")
    names = "\n".join(row.get("Kernel_Name", "").lower() for row in rows)
    matches: set[str] = set()
    for key, patterns in KERNEL_PATTERNS.items():
        if key in normalized_name or any(pattern.lower() in names for pattern in patterns):
            matches.add(key)
    return matches


def discover_traces(
    profile_dir: Path, warnings: list[str]
) -> dict[str, tuple[Path, list[dict[str, str]], list[Path]]]:
    candidates: dict[str, list[tuple[Path, list[dict[str, str]]]]] = defaultdict(list)
    if not profile_dir.is_dir():
        warnings.append(f"missing profile directory: {profile_dir}")
        return {}

    paths = sorted(
        {
            *profile_dir.rglob("*kernel_trace.csv"),
            *profile_dir.rglob("*kernel-trace.csv"),
        }
    )
    for path in paths:
        try:
            rows = read_trace_csv(path)
        except (OSError, csv.Error) as error:
            warnings.append(f"cannot read trace {path}: {error}")
            continue
        for key in classify_trace(path, rows):
            candidates[key].append((path, rows))

    chosen: dict[str, tuple[Path, list[dict[str, str]], list[Path]]] = {}
    for key, items in candidates.items():
        items.sort(
            key=lambda item: (
                key in str(item[0]).lower().replace("_", "-"),
                item[0].stat().st_mtime_ns,
            ),
            reverse=True,
        )
        selected_path, selected_rows = items[0]
        all_paths = [item[0] for item in items]
        chosen[key] = selected_path, selected_rows, all_paths
        if len(items) > 1:
            warnings.append(
                f"multiple traces match {key}; using canonical/newest "
                f"{selected_path.name}: "
                + ", ".join(path.name for path in all_paths)
            )
    return chosen


def int_column(row: dict[str, str], *names: str) -> int | None:
    value = first_value(row, *names)
    parsed = numeric(value, integer=True)
    return parsed if isinstance(parsed, int) else None


def grid_size(row: dict[str, str]) -> int | None:
    direct = int_column(row, "Grid_Size")
    if direct is not None:
        return direct
    dimensions = [
        int_column(row, "Grid_Size_X"),
        int_column(row, "Grid_Size_Y"),
        int_column(row, "Grid_Size_Z"),
    ]
    present = [value for value in dimensions if value is not None]
    if not present:
        return None
    product = 1
    for value in present:
        product *= value
    return product


def workgroup_size(row: dict[str, str]) -> int | None:
    direct = int_column(row, "Workgroup_Size")
    if direct is not None:
        return direct
    dimensions = [
        int_column(row, "Workgroup_Size_X"),
        int_column(row, "Workgroup_Size_Y"),
        int_column(row, "Workgroup_Size_Z"),
    ]
    present = [value for value in dimensions if value is not None]
    if not present:
        return None
    product = 1
    for value in present:
        product *= value
    return product


def duration_ns(row: dict[str, str]) -> int | None:
    start = int_column(row, "Start_Timestamp", "Start_Timestamp_Ns")
    end = int_column(row, "End_Timestamp", "End_Timestamp_Ns")
    if start is None or end is None or end < start:
        return None
    return end - start


def stats_ms(values_ns: list[int]) -> dict[str, float | int | None]:
    if not values_ns:
        return {
            "samples": 0,
            "minimum_ms": None,
            "median_ms": None,
            "mean_ms": None,
        }
    values_ms = [value / 1.0e6 for value in values_ns]
    return {
        "samples": len(values_ms),
        "minimum_ms": min(values_ms),
        "median_ms": statistics.median(values_ms),
        "mean_ms": statistics.fmean(values_ms),
    }


def summarize_kernel_signatures(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for row in rows:
        signature = (
            row.get("Kernel_Name", ""),
            grid_size(row),
            workgroup_size(row),
            int_column(row, "VGPR_Count"),
            int_column(row, "Accum_VGPR_Count"),
            int_column(row, "SGPR_Count"),
            int_column(row, "LDS_Block_Size", "LDS_Size"),
            int_column(row, "Scratch_Size"),
        )
        duration = duration_ns(row)
        if duration is not None:
            groups[signature].append(duration)
        else:
            groups.setdefault(signature, [])

    summaries: list[dict[str, Any]] = []
    for signature, durations in groups.items():
        (
            name,
            grid,
            workgroup,
            vgpr,
            accum_vgpr,
            sgpr,
            lds_bytes,
            scratch_bytes,
        ) = signature
        summaries.append(
            {
                "kernel_name": name,
                "dispatch_count": len(durations),
                "grid_size": grid,
                "workgroup_size": workgroup,
                "vgpr_count": vgpr,
                "accum_vgpr_count": accum_vgpr,
                "sgpr_count": sgpr,
                "lds_block_size_bytes": lds_bytes,
                "scratch_size_bytes": scratch_bytes,
                "duration": stats_ms(durations),
            }
        )
    summaries.sort(key=lambda item: (item["grid_size"] or 0), reverse=True)
    return summaries


def summarize_trace(
    *,
    key: str,
    source_path: Path,
    candidate_paths: list[Path],
    rows: list[dict[str, str]],
    stages: int,
    warmup: int,
    repeat: int,
    warnings: list[str],
) -> dict[str, Any]:
    patterns = tuple(pattern.lower() for pattern in KERNEL_PATTERNS[key])
    filtered = [
        row
        for row in rows
        if any(pattern in row.get("Kernel_Name", "").lower() for pattern in patterns)
    ]
    filtered.sort(
        key=lambda row: int_column(row, "Start_Timestamp", "Start_Timestamp_Ns")
        or -1
    )

    expected_logical_runs = 1 + warmup + repeat
    expected_dispatches = expected_logical_runs * stages
    if len(filtered) > expected_dispatches:
        warnings.append(
            f"{key} trace has {len(filtered)} matching dispatches; expected "
            f"{expected_dispatches}; using the last {expected_dispatches}"
        )
        selected = filtered[-expected_dispatches:]
    else:
        selected = filtered
    if len(selected) != expected_dispatches:
        warnings.append(
            f"{key} trace has {len(selected)} matching dispatches, expected "
            f"{expected_dispatches} for precheck=1 warmup={warmup} repeat={repeat} "
            f"stages={stages}"
        )

    skip_dispatches = min(len(selected), (1 + warmup) * stages)
    timed_rows = selected[skip_dispatches : skip_dispatches + repeat * stages]
    logical_durations: list[int] = []
    complete_groups = len(timed_rows) // stages if stages > 0 else 0
    for index in range(complete_groups):
        group = timed_rows[index * stages : (index + 1) * stages]
        start = int_column(group[0], "Start_Timestamp", "Start_Timestamp_Ns")
        end = int_column(group[-1], "End_Timestamp", "End_Timestamp_Ns")
        if start is not None and end is not None and end >= start:
            logical_durations.append(end - start)

    return {
        "source_file": str(source_path),
        "candidate_files": [str(path) for path in candidate_paths],
        "matching_dispatch_count": len(filtered),
        "selected_dispatch_count": len(selected),
        "expected_dispatch_count": expected_dispatches,
        "stages_per_logical_run": stages,
        "precheck_logical_runs": 1,
        "warmup_logical_runs": warmup,
        "requested_timed_logical_runs": repeat,
        "timed_logical_duration": stats_ms(logical_durations),
        "kernels": summarize_kernel_signatures(selected),
    }


def relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def write_csv(path: Path, implementations: list[dict[str, Any]]) -> None:
    fieldnames = [
        "backend",
        "implementation",
        "size",
        "grid",
        "block",
        "stages",
        "event_minimum_ms",
        "event_median_ms",
        "event_mean_ms",
        "correct",
        "independent_run_count",
        "event_median_ms_min",
        "event_median_ms_max",
        "useful_ai_flop_per_byte",
        "useful_bandwidth_gb_s",
        "useful_gflop_s",
        "trace_dispatches",
        "trace_minimum_ms",
        "trace_median_ms",
        "trace_mean_ms",
        "trace_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in implementations:
            event = item.get("event") or {}
            useful = event.get("useful_metrics") or {}
            trace = item.get("trace") or {}
            trace_duration = trace.get("timed_logical_duration") or {}
            independent_runs = event.get("independent_runs") or {}
            median_range = independent_runs.get("median_ms_range") or {}
            writer.writerow(
                {
                    "backend": event.get("backend") or item["key"].split("-", 1)[0],
                    "implementation": event.get("implementation")
                    or item["key"].split("-", 1)[1],
                    "size": event.get("size"),
                    "grid": event.get("grid"),
                    "block": event.get("block"),
                    "stages": event.get("stages"),
                    "event_minimum_ms": event.get("minimum_ms"),
                    "event_median_ms": event.get("median_ms"),
                    "event_mean_ms": event.get("mean_ms"),
                    "correct": event.get("correct"),
                    "independent_run_count": independent_runs.get("count"),
                    "event_median_ms_min": median_range.get("minimum_ms"),
                    "event_median_ms_max": median_range.get("maximum_ms"),
                    "useful_ai_flop_per_byte": useful.get(
                        "arithmetic_intensity_flop_per_byte"
                    ),
                    "useful_bandwidth_gb_s": useful.get("bandwidth_gb_s"),
                    "useful_gflop_s": useful.get("gflop_s"),
                    "trace_dispatches": trace.get("matching_dispatch_count"),
                    "trace_minimum_ms": trace_duration.get("minimum_ms"),
                    "trace_median_ms": trace_duration.get("median_ms"),
                    "trace_mean_ms": trace_duration.get("mean_ms"),
                    "trace_file": trace.get("source_file"),
                }
            )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter-dir", type=Path, default=script_dir)
    parser.add_argument("--hip-log", type=Path)
    parser.add_argument("--triton-log", type=Path)
    parser.add_argument("--profiles-dir", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--profile-warmup", type=int)
    parser.add_argument("--profile-repeat", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero if an event result or trace is missing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chapter_dir = args.chapter_dir.resolve()
    hip_log = args.hip_log or chapter_dir / "logs" / "hip_benchmark.log"
    triton_log = args.triton_log or chapter_dir / "logs" / "triton_benchmark.log"
    profiles_dir = args.profiles_dir or chapter_dir / "profiles"
    runs_dir = args.runs_dir or chapter_dir / "logs" / "runs"
    output_path = args.output or chapter_dir / "results" / "summary.json"
    csv_path = args.csv or chapter_dir / "results" / "summary.csv"

    warnings: list[str] = []
    profile_config = load_profile_config(profiles_dir / "profile_config.env")
    profile_warmup = (
        args.profile_warmup
        if args.profile_warmup is not None
        else int(profile_config.get("warmup", 0))
    )
    profile_repeat = (
        args.profile_repeat
        if args.profile_repeat is not None
        else int(profile_config.get("repeat", 10))
    )
    if profile_warmup < 0 or profile_repeat <= 0:
        raise SystemExit("profile warmup must be >= 0 and repeat must be > 0")

    fallback_events = parse_benchmark_log(hip_log, "hip", warnings)
    fallback_events.update(parse_benchmark_log(triton_log, "triton", warnings))
    run_events = parse_run_directory(runs_dir, warnings)
    events = fallback_events
    events.update(run_events)
    for event in events.values():
        attach_useful_metrics(event)

    traces = discover_traces(profiles_dir, warnings)
    implementations: list[dict[str, Any]] = []
    missing: list[str] = []
    for key in IMPLEMENTATION_ORDER:
        event = events.get(key)
        trace_data = traces.get(key)
        trace_summary = None
        if event is None:
            missing.append(f"event:{key}")
        if trace_data is None:
            missing.append(f"trace:{key}")
        else:
            source_path, rows, candidates = trace_data
            if event is not None and isinstance(event.get("stages"), int):
                stages = event["stages"]
            else:
                stages = 2 if key in ("hip-v4", "triton-multistage") else 1
            trace_summary = summarize_trace(
                key=key,
                source_path=source_path,
                candidate_paths=candidates,
                rows=rows,
                stages=stages,
                warmup=profile_warmup,
                repeat=profile_repeat,
                warnings=warnings,
            )
        implementations.append(
            {"key": key, "event": event, "trace": trace_summary}
        )

    if missing:
        warnings.append("missing required data: " + ", ".join(missing))

    for event in events.values():
        if "source_files" in event:
            event["source_files"] = [
                relative_path(Path(path), chapter_dir)
                for path in event["source_files"]
            ]
        elif "source_file" in event:
            event["source_file"] = relative_path(
                Path(event["source_file"]), chapter_dir
            )
    for item in implementations:
        trace = item.get("trace")
        if trace:
            trace["source_file"] = relative_path(
                Path(trace["source_file"]), chapter_dir
            )
            trace["candidate_files"] = [
                relative_path(Path(path), chapter_dir)
                for path in trace["candidate_files"]
            ]

    document = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "measurement_model": {
            "event_statistic_for_useful_metrics": "median_ms",
            "useful_flops": "N - 1",
            "useful_bytes": "4 * N + 4",
            "boundary": (
                "useful bytes are an algorithm-level comparison model, not "
                "measured DRAM, cache, atomic, or partial-buffer traffic"
            ),
        },
        "profile_selection": {
            "precheck_logical_runs": 1,
            "warmup_logical_runs": profile_warmup,
            "timed_logical_runs": profile_repeat,
            "config": profile_config,
        },
        "implementations": implementations,
        "warnings": warnings,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, implementations)
    print(f"wrote {output_path}")
    print(f"wrote {csv_path}")
    for warning in warnings:
        print(f"warning: {warning}")

    return 1 if args.strict and missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
