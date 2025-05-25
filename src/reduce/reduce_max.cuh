#pragma once

#include "utils.cuh"

namespace reduce {

template <typename T>
__inline__ __device__ T threadMax(T a, T b) {
    return (a > b) ? a : b;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        x = threadMax<T>(x, __shfl_down_sync(0xFFFFFFFF, x, offset));
    return x;
}

template <typename T>
__inline__ __device__ T blockReduceMax(T x) {
    static __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    x = warpReduceMax<T>(x);
    if (lane == 0) shared[wid] = x;
    __syncthreads();
    x = (threadIdx.x < blockDim.x / 32) ? shared[lane] : T(-INFINITY);
    if (wid == 0) { x = warpReduceMax<T>(x); }
    return x;
}

// gridDim (n, 1, 1),   blockDim (head_len, 1, 1)
static __global__ void max_kernel(float *in, float *out, const int head_len) {
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < head_len; i += blockDim.x) {
        int offset = OFFSET(blockIdx.x, i, head_len);
        local_max = threadMax(local_max, in[offset]);
    }
    local_max = blockReduceMax(local_max);

    if (threadIdx.x == 0) out[blockIdx.x] = local_max;
}

} // namespace reduce
