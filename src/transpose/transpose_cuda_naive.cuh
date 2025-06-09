#pragma once

#include "transpose_utils.cuh"
#include "utils.cuh"

namespace transpose {

// gridDim (n/TILE_N, m/TILE_M, 1),   blockDim (BLOCK_M, BLOCK_N, 1)
__global__ void
transpose_naive_kernel(float *out, const float *in, const int64_t M, const int64_t N) {
    int m = blockIdx.y * TILE_M + threadIdx.y;
    int n = blockIdx.x * TILE_N + threadIdx.x;

    for (int64_t j = 0; j < TILE_M; j += blockDim.y)
        out[OFFSET(n, m + j, M)] = in[OFFSET(m + j, n, N)]; // read by row, write by column
}

template <unsigned BLOCK_M = 32, unsigned BLOCK_N = 8>
void transpose_naive(float *out, const float *in, const int64_t M, const int64_t N) {
    dim3 gridDim(N / TILE_N, M / TILE_M, 1);
    dim3 blockDim(BLOCK_M, BLOCK_N, 1);
    transpose_naive_kernel<<<gridDim, blockDim>>>(out, in, M, N);
}

} // namespace transpose