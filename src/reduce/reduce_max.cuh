#pragma once

#include "reduce_utils.cuh"
#include "utils.cuh"

namespace reduce {

// gridDim (rows, 1, 1),   blockDim (block_size, 1, 1)
template <typename T, int block_size>
static __global__ void max_kernel(const T *in, T *out, const size_t cols) {
    T thread_max = -Inf<T>();
    for (size_t i = threadIdx.x; i < cols; i += block_size) {
        size_t offset = OFFSET(blockIdx.x, i, cols);
        thread_max = MaxOp<T>()(thread_max, in[offset]);
    }
    T row_max = block_all_reduce<MaxOp, T, block_size>(thread_max);

    if (threadIdx.x == 0) out[blockIdx.x] = row_max;
}

template <typename T, int block_size>
void gpu_max(const size_t rows, const size_t cols, const T *in, T *out) {
    max_kernel<T, block_size><<<rows, block_size>>>(in, out, cols);
}

} // namespace reduce
