#pragma once

#include "reduce_utils.cuh"
#include "utils.cuh"

namespace reduce {

// gridDim (grid_dim_x, 1, 1),   blockDim (block_size, 1, 1)
template <typename T, int block_size>
static __global__ void sum_kernel(const T *in, T *out, const int64_t rows, const int64_t cols) {
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        T thread_sum = 0;
        for (int64_t i = threadIdx.x; i < cols; i += block_size) {
            int64_t offset = OFFSET(row, i, cols);
            thread_sum = SumOp<T>()(thread_sum, in[offset]);
        }
        T row_sum = block_all_reduce<SumOp, T, block_size>(thread_sum);

        if (threadIdx.x == 0) out[row] = row_sum;
    }
}

template <typename T, int block_size>
void gpu_sum(const T *in, T *out, const int64_t rows, const int64_t cols) {
    constexpr int waves = 32;
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
        if (err != cudaSuccess) { return; }
    }
    sum_kernel<T, block_size><<<grid_dim_x, block_size>>>(in, out, rows, cols);
}

} // namespace reduce
