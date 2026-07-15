// 第 7 章：HIP reduction 的五个渐进版本。
//
// 编译（Radeon RX 9070 XT / gfx1201）：
//   hipcc --offload-arch=gfx1201 -O3 -std=c++17 reduction_hip.hip -o reduction_hip
//
// 运行：
//   ./reduction_hip --version all --size 1048576 --block 256 \
//       --warmup 10 --repeat 50
//
// 限制：
//   - --block 必须是 2 的幂、不得小于设备 wavefront 宽度，且不得超过
//     maxThreadsPerBlock。v1/v2 的二叉 LDS 树依赖这个约束。
//   - --grid 只控制 v2/v3/v4；v0/v1 必须启动 ceil(N / block) 个 block，
//     才能保证每个输入元素恰好被一个线程处理。
//   - v4 的第二阶段由一个 block 以 grid-stride 方式遍历所有 partial，
//     partial_count 可以大于 blockDim.x，不会只归约前 blockDim.x 项。
//
// 本文件只提供正确性与 GPU event 计时骨架。性能数据必须在项目规定的
// Radeon RX 9070 XT + ROCm 7.13 + 原生 Ubuntu 24.04 环境实测后再引用。

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        const hipError_t error = (call);                                       \
        if (error != hipSuccess) {                                             \
            std::fprintf(stderr, "HIP error at %s:%d: %s (%s)\n",             \
                         __FILE__, __LINE__, hipGetErrorString(error), #call);  \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

namespace {

// v0：每个有效线程都争用同一个全局地址。
__global__ void reduce_v0_atomic(const float* __restrict__ input,
                                 float* output,
                                 std::size_t n) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < n) {
        atomicAdd(output, input[index]);
    }
}

// v1：先在 LDS 中完成 block 内二叉树归约，每个 block 只做一次 atomicAdd。
__global__ void reduce_v1_lds_atomic(const float* __restrict__ input,
                                     float* output,
                                     std::size_t n) {
    extern __shared__ float shared[];

    const unsigned int tid = threadIdx.x;
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    shared[tid] = index < n ? input[index] : 0.0f;
    __syncthreads();

    // host 端强制 blockDim.x 为 2 的幂，因此每轮都能成对合并。
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// v2/v3/v4 共用完全相同的 grid-stride 寄存器局部累加。
__device__ __forceinline__ float grid_stride_local_sum(
    const float* __restrict__ input,
    std::size_t n) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;

    float local_sum = 0.0f;
    for (std::size_t i = index; i < n; i += grid_stride) {
        local_sum += input[i];
    }
    return local_sum;
}

// v2：先用 grid-stride 在寄存器中累加，再走完整 LDS 树；每个 block
// 仍然只做一次 atomicAdd。
__global__ void reduce_v2_grid_stride_lds_atomic(
    const float* __restrict__ input,
    float* output,
    std::size_t n) {
    extern __shared__ float shared[];

    const unsigned int tid = threadIdx.x;
    shared[tid] = grid_stride_local_sum(input, n);
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

__device__ __forceinline__ float wave_reduce_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, warpSize);
    }
    return value;
}

// 先在每个 wave 内用寄存器 shuffle 归约，再让第一个 wave 合并各 wave 的结果。
// 返回值只在 threadIdx.x == 0 时有效。
__device__ __forceinline__ float block_reduce_wave(float value,
                                                    float* wave_sums) {
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int wave_id = threadIdx.x / warpSize;
    const unsigned int wave_count =
        (blockDim.x + warpSize - 1) / warpSize;

    value = wave_reduce_sum(value);
    if (lane == 0) {
        wave_sums[wave_id] = value;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (wave_id == 0) {
        block_sum = lane < wave_count ? wave_sums[lane] : 0.0f;
        block_sum = wave_reduce_sum(block_sum);
    }
    return block_sum;
}

// v3：每个线程先在寄存器里处理多个元素，再做 wave reduction；跨 block
// 仍然以每个 block 一次 atomicAdd 合并。
__global__ void reduce_v3_wave_atomic(
    const float* __restrict__ input,
    float* output,
    std::size_t n) {
    extern __shared__ float wave_sums[];

    const float local_sum = grid_stride_local_sum(input, n);
    const float block_sum = block_reduce_wave(local_sum, wave_sums);
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

// v4 阶段 1：每个 block 写一个 partial，不做全局原子操作。
__global__ void reduce_v4_partials(const float* __restrict__ input,
                                   float* partials,
                                   std::size_t n) {
    extern __shared__ float wave_sums[];

    const float local_sum = grid_stride_local_sum(input, n);
    const float block_sum = block_reduce_wave(local_sum, wave_sums);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = block_sum;
    }
}

struct Args {
    std::string version = "all";
    std::size_t size = 1u << 20;
    int block = 256;
    int grid = 0;  // 0 表示按 HIP multiProcessorCount 自动选择，仅用于 v2/v3/v4。
    int warmup = 10;
    int repeat = 50;
    std::uint32_t seed = 20260711u;
    double rtol = 1e-7;  // relative tolerance，相对容差。
    double atol = 1e-3;  // absolute tolerance，绝对容差。
};

[[noreturn]] void usage_and_exit(const char* program, int code) {
    std::FILE* out = code == 0 ? stdout : stderr;
    std::fprintf(
        out,
        "Usage: %s [options]\n"
        "  --version {v0,v1,v2,v3,v4,all}  implementation to run (default: all)\n"
        "  --size N                     input elements (default: 1048576)\n"
        "  --block N                    power of two, >= wavefront width\n"
        "                               (default: 256)\n"
        "  --grid N                     v2/v3/v4 blocks; 0 selects an automatic grid\n"
        "                               (default: 0)\n"
        "  --warmup N                   untimed launches (default: 10)\n"
        "  --repeat N                   timed launches (default: 50)\n"
        "  --seed N                     deterministic input seed\n"
        "  --rtol X                     relative tolerance / 相对容差 "
        "(default: 1e-7)\n"
        "  --atol X                     absolute tolerance / 绝对容差 "
        "(default: 1e-3)\n"
        "  -h, --help                   show this help\n",
        program);
    std::exit(code);
}

std::string next_value(int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", argv[index]);
        usage_and_exit(argv[0], 2);
    }
    return argv[++index];
}

Args parse_args(int argc, char** argv) {
    Args args;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "-h" || option == "--help") {
                usage_and_exit(argv[0], 0);
            } else if (option == "--version") {
                args.version = next_value(argc, argv, i);
            } else if (option == "--size") {
                args.size = std::stoull(next_value(argc, argv, i));
            } else if (option == "--block") {
                args.block = std::stoi(next_value(argc, argv, i));
            } else if (option == "--grid") {
                args.grid = std::stoi(next_value(argc, argv, i));
            } else if (option == "--warmup") {
                args.warmup = std::stoi(next_value(argc, argv, i));
            } else if (option == "--repeat") {
                args.repeat = std::stoi(next_value(argc, argv, i));
            } else if (option == "--seed") {
                args.seed = static_cast<std::uint32_t>(
                    std::stoul(next_value(argc, argv, i)));
            } else if (option == "--rtol") {
                args.rtol = std::stod(next_value(argc, argv, i));
            } else if (option == "--atol") {
                args.atol = std::stod(next_value(argc, argv, i));
            } else {
                std::fprintf(stderr, "unknown option: %s\n", option.c_str());
                usage_and_exit(argv[0], 2);
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "invalid argument: %s\n", error.what());
        usage_and_exit(argv[0], 2);
    }
    return args;
}

bool is_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

std::size_t ceil_div(std::size_t numerator, std::size_t denominator) {
    return numerator / denominator + (numerator % denominator != 0);
}

void validate_args(const Args& args, const hipDeviceProp_t& properties) {
    const bool valid_version = args.version == "all" || args.version == "v0" ||
                               args.version == "v1" || args.version == "v2" ||
                               args.version == "v3" || args.version == "v4";
    if (!valid_version) {
        std::fprintf(stderr,
                     "--version must be v0, v1, v2, v3, v4, or all\n");
        std::exit(EXIT_FAILURE);
    }
    if (args.size == 0) {
        std::fprintf(stderr, "--size must be greater than zero\n");
        std::exit(EXIT_FAILURE);
    }
    if (!is_power_of_two(args.block)) {
        std::fprintf(stderr, "--block must be a power of two\n");
        std::exit(EXIT_FAILURE);
    }
    if (args.block < properties.warpSize) {
        std::fprintf(stderr,
                     "--block (%d) must be at least the device wavefront width (%d)\n",
                     args.block, properties.warpSize);
        std::exit(EXIT_FAILURE);
    }
    if (args.block > properties.maxThreadsPerBlock) {
        std::fprintf(stderr,
                     "--block (%d) exceeds maxThreadsPerBlock (%d)\n",
                     args.block, properties.maxThreadsPerBlock);
        std::exit(EXIT_FAILURE);
    }
    const int waves_per_block = args.block / properties.warpSize;
    if (waves_per_block > properties.warpSize) {
        std::fprintf(
            stderr,
            "--block creates more waves than the first wave can finalize\n");
        std::exit(EXIT_FAILURE);
    }
    if (args.grid < 0 || args.warmup < 0 || args.repeat <= 0) {
        std::fprintf(stderr,
                     "--grid/--warmup must be non-negative and --repeat must be positive\n");
        std::exit(EXIT_FAILURE);
    }
    if (args.rtol < 0.0 || args.atol < 0.0) {
        std::fprintf(stderr, "--rtol and --atol must be non-negative\n");
        std::exit(EXIT_FAILURE);
    }
    if (args.size > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        std::fprintf(stderr, "--size is too large for a float allocation\n");
        std::exit(EXIT_FAILURE);
    }
}

struct TimingResult {
    float minimum_ms = 0.0f;
    float median_ms = 0.0f;
    float mean_ms = 0.0f;
    float actual = 0.0f;
    double absolute_error = 0.0;
    bool precheck_correct = false;
    bool postcheck_correct = false;
    bool timed = false;
    bool correct = false;
};

void read_and_validate(TimingResult& result,
                       const Args& args,
                       float* device_output,
                       double expected,
                       bool precheck) {
    HIP_CHECK(hipMemcpy(&result.actual, device_output, sizeof(float),
                        hipMemcpyDeviceToHost));
    result.absolute_error =
        std::abs(static_cast<double>(result.actual) - expected);
    const bool correct =
        result.absolute_error <= args.atol + args.rtol * std::abs(expected);
    if (precheck) {
        result.precheck_correct = correct;
    } else {
        result.postcheck_correct = correct;
    }
}

template <typename Launch>
TimingResult benchmark(const Args& args,
                       bool reset_output,
                       float* device_output,
                       double expected,
                       Launch launch) {
    TimingResult result;

    // 先做一次完整的正确性预检。预检失败时不进入 warmup 或 timed runs。
    if (reset_output) {
        HIP_CHECK(hipMemsetAsync(device_output, 0, sizeof(float)));
    }
    launch();
    HIP_CHECK(hipDeviceSynchronize());
    read_and_validate(result, args, device_output, expected, true);
    if (!result.precheck_correct) {
        return result;
    }

    for (int i = 0; i < args.warmup; ++i) {
        if (reset_output) {
            HIP_CHECK(hipMemsetAsync(device_output, 0, sizeof(float)));
        }
        launch();
    }
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<hipEvent_t> starts(args.repeat);
    std::vector<hipEvent_t> ends(args.repeat);
    for (int i = 0; i < args.repeat; ++i) {
        HIP_CHECK(hipEventCreate(&starts[i]));
        HIP_CHECK(hipEventCreate(&ends[i]));
    }

    for (int i = 0; i < args.repeat; ++i) {
        // reset 排在 start event 前面的同一条默认 stream 上，不计入 kernel 时间。
        if (reset_output) {
            HIP_CHECK(hipMemsetAsync(device_output, 0, sizeof(float)));
        }
        HIP_CHECK(hipEventRecord(starts[i]));
        launch();
        HIP_CHECK(hipEventRecord(ends[i]));
    }
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> times(args.repeat);
    for (int i = 0; i < args.repeat; ++i) {
        HIP_CHECK(hipEventElapsedTime(&times[i], starts[i], ends[i]));
        HIP_CHECK(hipEventDestroy(starts[i]));
        HIP_CHECK(hipEventDestroy(ends[i]));
    }
    std::sort(times.begin(), times.end());

    const std::size_t middle = times.size() / 2;
    result.minimum_ms = times.front();
    result.median_ms = times.size() % 2 == 0
                           ? 0.5f * (times[middle - 1] + times[middle])
                           : times[middle];
    result.mean_ms = static_cast<float>(
        std::accumulate(times.begin(), times.end(), 0.0) / times.size());
    result.timed = true;
    read_and_validate(result, args, device_output, expected, false);
    result.correct = result.precheck_correct && result.postcheck_correct;
    return result;
}

void print_result(const std::string& version,
                  unsigned int grid,
                  int block,
                  double expected,
                  const TimingResult& result) {
    if (!result.timed) {
        std::printf("%-2s grid=%u block=%d  precheck=FAIL timing=SKIPPED\n",
                    version.c_str(), grid, block);
        std::printf(
            "   result=%.9g expected=%.9g abs_error=%.6g correctness=FAIL\n",
            result.actual, expected, result.absolute_error);
        return;
    }
    std::printf(
        "%-2s grid=%u block=%d  min_ms=%.6f median_ms=%.6f mean_ms=%.6f\n",
        version.c_str(), grid, block, result.minimum_ms, result.median_ms,
        result.mean_ms);
    std::printf(
        "   result=%.9g expected=%.9g abs_error=%.6g "
        "precheck=OK postcheck=%s correctness=%s\n",
        result.actual, expected, result.absolute_error,
        result.postcheck_correct ? "OK" : "FAIL",
        result.correct ? "OK" : "FAIL");
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    validate_args(args, properties);

    const std::size_t v01_grid_size =
        ceil_div(args.size, static_cast<std::size_t>(args.block));
    if (v01_grid_size >
        static_cast<std::size_t>(properties.maxGridSize[0])) {
        std::fprintf(stderr,
                     "ceil(size / block) exceeds the device x-dimension grid limit\n");
        return EXIT_FAILURE;
    }

    const std::size_t automatic_grid = std::min(
        v01_grid_size,
        static_cast<std::size_t>(std::max(1, properties.multiProcessorCount)) * 8u);
    const std::size_t optimized_grid_size =
        args.grid == 0 ? automatic_grid : static_cast<std::size_t>(args.grid);
    if (optimized_grid_size == 0 ||
        optimized_grid_size >
            static_cast<std::size_t>(properties.maxGridSize[0]) ||
        optimized_grid_size > std::numeric_limits<unsigned int>::max()) {
        std::fprintf(stderr,
                     "v2/v3/v4 grid is outside the supported range\n");
        return EXIT_FAILURE;
    }

    const auto v01_grid = static_cast<unsigned int>(v01_grid_size);
    const auto optimized_grid =
        static_cast<unsigned int>(optimized_grid_size);
    const std::size_t lds_shared_bytes = args.block * sizeof(float);
    const std::size_t wave_shared_bytes =
        (args.block / properties.warpSize) * sizeof(float);

    std::vector<float> host_input(args.size);
    double expected = 0.0;
    for (std::size_t i = 0; i < args.size; ++i) {
        // 所有版本共享同一确定性公式，尾部漏项会产生至少 1 的误差。
        const float value = static_cast<float>(
            1u + ((static_cast<std::uint64_t>(i) + args.seed) & 3u));
        host_input[i] = value;
        expected += static_cast<double>(value);
    }

    float* device_input = nullptr;
    float* device_output = nullptr;
    float* device_partials = nullptr;
    HIP_CHECK(hipMalloc(&device_input, args.size * sizeof(float)));
    HIP_CHECK(hipMalloc(&device_output, sizeof(float)));
    HIP_CHECK(hipMalloc(&device_partials,
                        optimized_grid_size * sizeof(float)));
    HIP_CHECK(hipMemcpy(device_input, host_input.data(),
                        args.size * sizeof(float), hipMemcpyHostToDevice));

    std::printf(
        "GPU: %s (arch=%s, wavefront=%d, hip_multi_processor_count=%d)\n",
        properties.name, properties.gcnArchName, properties.warpSize,
        properties.multiProcessorCount);
    std::printf(
        "size=%zu block=%d v01_grid=%u optimized_grid=%u "
        "warmup=%d repeat=%d\n",
        args.size, args.block, v01_grid, optimized_grid, args.warmup,
        args.repeat);
    std::printf("block restriction: power-of-two and >= wavefront width\n");

    bool all_correct = true;
    auto wants = [&](const char* version) {
        return args.version == "all" || args.version == version;
    };

    if (wants("v0")) {
        const auto result = benchmark(args, true, device_output, expected, [&] {
            hipLaunchKernelGGL(reduce_v0_atomic, dim3(v01_grid), dim3(args.block),
                               0, 0, device_input, device_output, args.size);
            HIP_CHECK(hipGetLastError());
        });
        print_result("v0", v01_grid, args.block, expected, result);
        all_correct = all_correct && result.correct;
    }

    if (wants("v1")) {
        const auto result = benchmark(args, true, device_output, expected, [&] {
            hipLaunchKernelGGL(reduce_v1_lds_atomic, dim3(v01_grid),
                               dim3(args.block), lds_shared_bytes, 0,
                               device_input, device_output, args.size);
            HIP_CHECK(hipGetLastError());
        });
        print_result("v1", v01_grid, args.block, expected, result);
        all_correct = all_correct && result.correct;
    }

    if (wants("v2")) {
        const auto result = benchmark(args, true, device_output, expected, [&] {
            hipLaunchKernelGGL(reduce_v2_grid_stride_lds_atomic,
                               dim3(optimized_grid), dim3(args.block),
                               lds_shared_bytes, 0,
                               device_input, device_output, args.size);
            HIP_CHECK(hipGetLastError());
        });
        print_result("v2", optimized_grid, args.block, expected, result);
        all_correct = all_correct && result.correct;
    }

    if (wants("v3")) {
        const auto result = benchmark(args, true, device_output, expected, [&] {
            hipLaunchKernelGGL(reduce_v3_wave_atomic,
                               dim3(optimized_grid), dim3(args.block),
                               wave_shared_bytes, 0,
                               device_input, device_output, args.size);
            HIP_CHECK(hipGetLastError());
        });
        print_result("v3", optimized_grid, args.block, expected, result);
        all_correct = all_correct && result.correct;
    }

    if (wants("v4")) {
        const auto result = benchmark(args, false, device_output, expected, [&] {
            hipLaunchKernelGGL(reduce_v4_partials, dim3(optimized_grid),
                               dim3(args.block), wave_shared_bytes, 0,
                               device_input, device_partials, args.size);
            HIP_CHECK(hipGetLastError());
            // 同一个 kernel 以单 block 处理第二阶段。它的 grid-stride 循环会
            // 遍历全部 partial，而不是只读取前 blockDim.x 个。
            hipLaunchKernelGGL(reduce_v4_partials, dim3(1), dim3(args.block),
                               wave_shared_bytes, 0, device_partials,
                               device_output, optimized_grid_size);
            HIP_CHECK(hipGetLastError());
        });
        print_result("v4", optimized_grid, args.block, expected, result);
        all_correct = all_correct && result.correct;
    }

    HIP_CHECK(hipFree(device_partials));
    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_input));
    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
