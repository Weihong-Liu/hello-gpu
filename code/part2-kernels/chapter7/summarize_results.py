"""Aggregate Chapter 7 RESULT records without inventing missing values."""

from __future__ import annotations

import csv
import json
import shlex
import statistics
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
PROFILE_DIR = SCRIPT_DIR / "profiles"
RESULT_DIR = SCRIPT_DIR / "results"
NUMERIC_FIELDS = (
    "min_ms",
    "median_ms",
    "mean_ms",
    "effective_bandwidth_gbs",
    "max_abs_error",
)
RANGE_FIELDS = (
    "median_ms",
    "effective_bandwidth_gbs",
)
TRACE_FIELDS = {
    "Grid_Size_X": "trace_grid_size_x",
    "Workgroup_Size_X": "trace_workgroup_size_x",
    "LDS_Block_Size": "trace_lds_bytes",
    "Scratch_Size": "trace_scratch_bytes",
    "VGPR_Count": "trace_vgpr_count",
    "Accum_VGPR_Count": "trace_accum_vgpr_count",
    "SGPR_Count": "trace_sgpr_count",
}


def parse_result_line(line: str, source: Path) -> dict[str, str] | None:
    if not line.startswith("RESULT "):
        return None
    record: dict[str, str] = {"source": str(source.relative_to(SCRIPT_DIR))}
    for token in shlex.split(line)[1:]:
        key, separator, value = token.partition("=")
        if separator:
            record[key] = value
    return record


def read_records(paths: list[Path]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            record = parse_result_line(line, path)
            if record is not None:
                records.append(record)
    return records


def read_trace_fields(implementation: str) -> dict[str, object]:
    path = PROFILE_DIR / f"{implementation}_kernel_trace.csv"
    if not path.exists():
        return {"trace_dispatches": "NA", **{field: "NA" for field in TRACE_FIELDS.values()}}

    with path.open(encoding="utf-8", newline="") as file:
        target_rows = [
            row
            for row in csv.DictReader(file)
            if "vector_add" in row.get("Kernel_Name", "")
        ]

    fields: dict[str, object] = {"trace_dispatches": len(target_rows)}
    for source, destination in TRACE_FIELDS.items():
        values = {row[source] for row in target_rows if row.get(source)}
        fields[destination] = int(next(iter(values))) if len(values) == 1 else "NA"
    return fields


def aggregate(records: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record["implementation"]].append(record)

    summary: list[dict[str, object]] = []
    for implementation in sorted(grouped):
        group = grouped[implementation]
        first = group[0]
        row: dict[str, object] = {
            "implementation": implementation,
            "runtime": first.get("runtime", "NA"),
            "size": first.get("size", "NA"),
            "dtype": first.get("dtype", "NA"),
            "block": first.get("block", "NA"),
            "grid": first.get("grid", "NA"),
            "correct": "OK"
            if all(item.get("correct") == "OK" for item in group)
            else "FAIL",
            "run_count": len(group),
        }
        row.update(read_trace_fields(implementation))
        for field in NUMERIC_FIELDS:
            values = [
                float(item[field])
                for item in group
                if item.get(field) not in (None, "NA")
            ]
            row[field] = statistics.median(values) if values else "NA"
            if field in RANGE_FIELDS:
                row[f"{field}_run_min"] = min(values) if values else "NA"
                row[f"{field}_run_max"] = max(values) if values else "NA"
        row["sources"] = ";".join(sorted({item["source"] for item in group}))
        summary.append(row)
    return summary


def main() -> None:
    independent_paths = sorted((LOG_DIR / "runs").glob("*.log"))
    records = read_records(independent_paths)
    source_kind = "independent-runs"
    if not records:
        records = read_records(
            [LOG_DIR / "hip_benchmark.log", LOG_DIR / "triton_benchmark.log"]
        )
        source_kind = "main-benchmark"
    if not records:
        raise SystemExit("no RESULT records found; run run_all.sh first")

    summary = aggregate(records)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "implementation",
        "runtime",
        "size",
        "dtype",
        "block",
        "grid",
        "correct",
        "run_count",
        "trace_dispatches",
        *TRACE_FIELDS.values(),
        *NUMERIC_FIELDS,
        "median_ms_run_min",
        "median_ms_run_max",
        "effective_bandwidth_gbs_run_min",
        "effective_bandwidth_gbs_run_max",
        "sources",
    ]
    with (RESULT_DIR / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    payload = {
        "status": "measured",
        "source_kind": source_kind,
        "records": summary,
    }
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(summary)} implementations to {RESULT_DIR}")


if __name__ == "__main__":
    main()
