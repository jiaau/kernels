#pragma once
#include "softmax_utils.cuh"
#include "utils.cuh"

namespace softmax {
// gridDim(grid_dim_x, 1, 1), block(block_size, 1, 1)
template <typename T, int block_size>
__global__ void softmax_naive_kernel(const int rows, const int cols, const T *in, T *out) {
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        T thread_max = -Inf<T>();
        for (int i = threadIdx.x; i < cols; i += block_size) {
            thread_max = MaxOp<T>()(thread_max, in[OFFSET(row, i, cols)]);
        }
        T row_max = block_all_reduce<MaxOp, T, block_size>(thread_max);

        for (int i = threadIdx.x; i < cols; i += block_size) {
            out[OFFSET(row, i, cols)] = exp(in[OFFSET(row, i, cols)] - row_max);
        }

        T thread_sum = 0;
        for (int i = threadIdx.x; i < cols; i += block_size) {
            thread_sum += out[OFFSET(row, i, cols)];
        }
        T row_sum = block_all_reduce<SumOp, T, block_size>(thread_sum);

        for (int i = threadIdx.x; i < cols; i += block_size) {
            out[OFFSET(row, i, cols)] /= row_sum;
        }
    }
}

template <typename T, int block_size>
void softmax_naive(const int rows, const int cols, const T *in, T *out) {
    constexpr int waves = 32;
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
        if (err != cudaSuccess) { return; }
    }
    softmax_naive_kernel<T, block_size><<<grid_dim_x, block_size>>>(rows, cols, in, out);
}

} // namespace softmax