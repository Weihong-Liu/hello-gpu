#!/usr/bin/env bash
set -euo pipefail

print_command() {
    local label="$1"
    shift
    echo "[$label]"
    if "$@"; then
        :
    else
        echo "unavailable (exit=$?)"
    fi
}

echo "[timestamp]"
date -Iseconds

print_command hostname hostname

echo "[os-release]"
if [[ -r /etc/os-release ]]; then
    cat /etc/os-release
else
    echo "unavailable"
fi

print_command uname uname -a

echo "[virtualization]"
if command -v systemd-detect-virt >/dev/null 2>&1; then
    systemd-detect-virt 2>&1 || true
else
    echo "unavailable"
fi
print_command identity id

echo "[gpu-device-nodes]"
shopt -s nullglob
gpu_paths=(/dev/kfd /sys/class/kfd/kfd /dev/dri/card* /dev/dri/renderD*)
if ((${#gpu_paths[@]} == 0)); then
    echo "no GPU device nodes found"
fi
for path in "${gpu_paths[@]}"; do
    if [[ -e "${path}" ]]; then
        ls -ld "${path}"
    else
        echo "missing ${path}"
    fi
done

echo "[rocminfo-summary]"
if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>&1 | awk '
        /^[[:space:]]*Name:[[:space:]]+gfx/ ||
        /^[[:space:]]*Marketing Name:.*(AMD|Radeon)/ ||
        /^[[:space:]]*Wavefront Size:/ ||
        /^[[:space:]]*Max Waves Per CU:/ {
            sub(/^[[:space:]]+/, "")
            print
        }
    '
else
    echo "unavailable"
fi

echo "[amd-smi-asic]"
if command -v amd-smi >/dev/null 2>&1; then
    amd-smi static --asic 2>&1 | awk '
        /^[[:space:]]*(MARKET_NAME|VENDOR_NAME|NUM_COMPUTE_UNITS|TARGET_GRAPHICS_VERSION):/ {
            sub(/^[[:space:]]+/, "")
            print
        }
    ' || true
elif command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname 2>&1 || true
else
    echo "unavailable"
fi

print_command hipcc-version hipcc --version
print_command rocprofv3-version rocprofv3 --version

echo "[uv-version]"
if command -v uv >/dev/null 2>&1; then
    uv --version
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
    "${HOME}/.local/bin/uv" --version
else
    echo "unavailable"
fi

print_command python-version python --version

echo "[python-gpu-stack]"
python - <<'PY'
try:
    import torch
    import triton
except Exception as error:
    print(f"unavailable: {error}")
else:
    print(f"torch={torch.__version__}")
    print(f"triton={triton.__version__}")
    print(f"torch.version.hip={torch.version.hip}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_name={torch.cuda.get_device_name(0)}")
PY
