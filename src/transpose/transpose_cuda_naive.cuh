#pragma once

#include "utils.cuh"

namespace transpose {

// gridDim (n/TILE_N, m/TILE_M, 1),   blockDim (THREAD_X, THREAD_Y, 1)
template <unsigned THREAD_X = 32, unsigned THREAD_Y = 8>
__global__ void transpose_naive(float *out, const float *in, const int m, const int n) {
    const int TILE_M = 32;
    const int TILE_N = 32;

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    for (int j = 0; j < TILE_M; j += THREAD_Y)
        out[OFFSET(col, row + j, m)] = in[OFFSET(row + j, col, n)];
}

} // namespace transpose