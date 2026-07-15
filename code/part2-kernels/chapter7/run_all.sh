#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
GPU_ARCH="${GPU_ARCH:-gfx1201}"
SIZE="${SIZE:-1048576}"
HIP_BLOCK="${HIP_BLOCK:-256}"
TRITON_BLOCK="${TRITON_BLOCK:-1024}"
WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
SEED="${SEED:-20260711}"
RUN_EDGE_CASES="${RUN_EDGE_CASES:-1}"
INDEPENDENT_RUNS="${INDEPENDENT_RUNS:-3}"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hello-gpu-ch7.XXXXXX")"
HIP_BINARY="${BUILD_DIR}/reduction_hip"

cleanup() {
    rm -rf "${BUILD_DIR}"
}
trap cleanup EXIT

mkdir -p "${LOG_DIR}"

if [[ ! -f "${PART_DIR}/activate-rocm.sh" ]]; then
    echo "missing ${PART_DIR}/activate-rocm.sh" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${PART_DIR}/activate-rocm.sh"

bash "${SCRIPT_DIR}/collect_environment.sh" \
    2>&1 | tee "${LOG_DIR}/environment.log"

hipcc \
    --offload-arch="${GPU_ARCH}" \
    -O3 \
    -std=c++17 \
    "${SCRIPT_DIR}/reduction_hip.hip" \
    -o "${HIP_BINARY}" \
    2>&1 | tee "${LOG_DIR}/hip_compile.log"

hip_args=(
    --version all
    --size "${SIZE}"
    --block "${HIP_BLOCK}"
    --warmup "${WARMUP}"
    --repeat "${REPEAT}"
    --seed "${SEED}"
)
if [[ -n "${GRID:-}" ]]; then
    hip_args+=(--grid "${GRID}")
fi

"${HIP_BINARY}" "${hip_args[@]}" \
    2>&1 | tee "${LOG_DIR}/hip_benchmark.log"

if [[ "${RUN_EDGE_CASES}" == "1" ]]; then
    hip_partial_grid=$((HIP_BLOCK * 4))
    hip_partial_size=$((HIP_BLOCK * (HIP_BLOCK + 1) + 1))
    hip_correctness_log="${LOG_DIR}/hip_correctness.log"
    : > "${hip_correctness_log}"
    for edge_size in 1 32 $((HIP_BLOCK + 1)); do
        "${HIP_BINARY}" \
            --version all \
            --size "${edge_size}" \
            --block "${HIP_BLOCK}" \
            --warmup 0 \
            --repeat 1 \
            --seed "${SEED}" \
            2>&1 | tee -a "${hip_correctness_log}"
    done

    # 让非零 partial 的数量超过 block，显式覆盖 v4 第二阶段的尾部读取。
    "${HIP_BINARY}" \
        --version v4 \
        --size "${hip_partial_size}" \
        --block "${HIP_BLOCK}" \
        --grid "${hip_partial_grid}" \
        --warmup 0 \
        --repeat 1 \
        --seed "${SEED}" \
        2>&1 | tee -a "${hip_correctness_log}"
fi

python "${SCRIPT_DIR}/reduction_triton.py" \
    --version all \
    --size "${SIZE}" \
    --block "${TRITON_BLOCK}" \
    --warmup "${WARMUP}" \
    --repeat "${REPEAT}" \
    --seed "${SEED}" \
    2>&1 | tee "${LOG_DIR}/triton_benchmark.log"

if [[ "${RUN_EDGE_CASES}" == "1" ]]; then
    triton_partial_size=$((TRITON_BLOCK * (TRITON_BLOCK + 1) + 1))
    triton_correctness_log="${LOG_DIR}/triton_correctness.log"
    : > "${triton_correctness_log}"
    for edge_size in 1 32 $((TRITON_BLOCK + 1)); do
        python "${SCRIPT_DIR}/reduction_triton.py" \
            --version all \
            --size "${edge_size}" \
            --block "${TRITON_BLOCK}" \
            --warmup 0 \
            --repeat 1 \
            --seed "${SEED}" \
            2>&1 | tee -a "${triton_correctness_log}"
    done

    python "${SCRIPT_DIR}/reduction_triton.py" \
        --version multistage \
        --size "${triton_partial_size}" \
        --block "${TRITON_BLOCK}" \
        --warmup 0 \
        --repeat 1 \
        --seed "${SEED}" \
        2>&1 | tee -a "${triton_correctness_log}"
fi

if ((INDEPENDENT_RUNS > 0)); then
    run_log_dir="${LOG_DIR}/runs"
    mkdir -p "${run_log_dir}"
    for run in $(seq 1 "${INDEPENDENT_RUNS}"); do
        "${HIP_BINARY}" "${hip_args[@]}" \
            > "${run_log_dir}/hip_run${run}.log" 2>&1
        python "${SCRIPT_DIR}/reduction_triton.py" \
            --version all \
            --size "${SIZE}" \
            --block "${TRITON_BLOCK}" \
            --warmup "${WARMUP}" \
            --repeat "${REPEAT}" \
            --seed "${SEED}" \
            > "${run_log_dir}/triton_run${run}.log" 2>&1
    done
fi

echo "logs written to ${LOG_DIR}"
