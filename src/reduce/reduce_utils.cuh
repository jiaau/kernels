#pragma once

// #include <cub/cub.cuh>
#include <math_constants.h>

namespace reduce {

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a + b; }
};

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return max(a, b); }
};

template <typename T>
__inline__ __device__ T Inf();

template <>
__inline__ __device__ float Inf<float>() {
    return CUDART_INF_F;
}

template <template <typename> class ReductionOp, typename T, int thread_group_width = 32>
__inline__ __device__ T warp_all_reduce(T val) {
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <template <typename> class ReductionOp, typename T>
__inline__ __device__ T get_default_val() {
    if constexpr (std::is_same_v<ReductionOp<T>, MaxOp<T>>) {
        return -Inf<T>();
    } else {
        return static_cast<T>(0);
    }
}

template <template <typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T block_all_reduce(T val) {
    __shared__ T shared_mem[32];
    __shared__ T result_broadcast;

    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    T warp_result = warp_all_reduce<ReductionOp, T>(val);
    if (lane == 0) shared_mem[wid] = warp_result;
    __syncthreads();

    T block_result =
        (threadIdx.x < block_size / 32) ? shared_mem[lane] : get_default_val<ReductionOp, T>();
    block_result = warp_all_reduce<ReductionOp, T>(block_result);

    if (threadIdx.x == 0) { result_broadcast = block_result; }
    __syncthreads();
    return result_broadcast;
}

// template <template <typename> class ReductionOp, typename T, int block_size>
// __inline__ __device__ T block_all_reduce(T val) {
//     typedef cub::BlockReduce<T, block_size> BlockReduce;
//     __shared__ typename BlockReduce::TempStorage temp_storage;
//     __shared__ T result_broadcast;
//     T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
//     if (threadIdx.x == 0) { result_broadcast = result; }
//     __syncthreads();
//     return result_broadcast;
// }

inline cudaError_t
GetNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves, int *num_blocks) {
    int dev;
    {
        cudaError_t err = cudaGetDevice(&dev);
        if (err != cudaSuccess) { return err; }
    }
    int sm_count;
    {
        cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        if (err != cudaSuccess) { return err; }
    }
    int tpm;
    {
        cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        if (err != cudaSuccess) { return err; }
    }
    *num_blocks =
        std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
    return cudaSuccess;
}

} // namespace reduce