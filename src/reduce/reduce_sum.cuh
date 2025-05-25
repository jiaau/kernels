#pragma once

#include "utils.cuh"

namespace reduce {

template <typename T>
__inline__ __device__ T warpReduceSum(T x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        x += __shfl_down_sync(0xFFFFFFFF, x, offset);
    return x;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T x) {
    static __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    x = warpReduceSum<T>(x);
    if (lane == 0) shared[wid] = x;
    __syncthreads();
    x = (threadIdx.x < blockDim.x / 32) ? shared[lane] : T(0.);
    if (wid == 0) { x = warpReduceSum<T>(x); }
    return x;
}

// gridDim (n, 1, 1),   blockDim (head_len, 1, 1)
static __global__ void sum_kernel(float *in, float *out, const int head_len) {
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < head_len; i += blockDim.x) {
        int offset = OFFSET(blockIdx.x, i, head_len);
        local_sum += in[offset];
    }
    local_sum = blockReduceSum(local_sum);

    if (threadIdx.x == 0) out[blockIdx.x] = local_sum;
}

} // namespace reduce
