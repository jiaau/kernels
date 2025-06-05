#pragma once

#include "reduce_utils.cuh"
#include "utils.cuh"

namespace reduce {

// gridDim (rows, 1, 1),   blockDim (block_size, 1, 1)
template <typename T, int block_size>
static __global__ void sum_kernel(const T *in, T *out, const size_t cols) {
    T thread_sum = 0.0;
    for (size_t i = threadIdx.x; i < cols; i += block_size) {
        size_t offset = OFFSET(blockIdx.x, i, cols);
        thread_sum = SumOp<T>()(thread_sum, in[offset]);
    }
    T row_sum = block_all_reduce<SumOp, T, block_size>(thread_sum);

    if (threadIdx.x == 0) out[blockIdx.x] = row_sum;
}

template <typename T, int block_size>
void gpu_sum(const size_t rows, const size_t cols, const T *in, T *out) {
    sum_kernel<T, block_size><<<rows, block_size>>>(in, out, cols);
}

} // namespace reduce
