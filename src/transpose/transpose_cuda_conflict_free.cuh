#pragma once

#include <cstdint>
#include "swizzle.cuh"
#include "utils.cuh"

namespace transpose {

// gridDim (n/TILE_N, m/TILE_M, 1),   blockDim (THREAD_X, THREAD_Y, 1)
template <unsigned THREAD_X = 32, unsigned THREAD_Y = 8>
__global__ void
transpose_cuda_conflict_free(float *out, const float *in, const int m, const int n) {
    const int TILE_M = 32;
    const int TILE_N = 32;

    __shared__ float tile[1024]; // 32 * 32

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    for (int j = 0; j < TILE_M; j += THREAD_Y) {
        uint32_t logical_addr = OFFSET(threadIdx.x, threadIdx.y + j, 32);
        uint32_t physical_addr = swizzle<5, 5, 0>(logical_addr);
        tile[physical_addr] = in[OFFSET(row + j, col, n)]; // read by row, write by column
    }

    __syncthreads();

    int n_col = blockIdx.y * TILE_M + threadIdx.x;
    int n_row = blockIdx.x * TILE_N + threadIdx.y;

    for (int j = 0; j < TILE_M; j += THREAD_Y) {
        uint32_t logical_addr = OFFSET(threadIdx.y + j, threadIdx.x, 32);
        uint32_t physical_addr = swizzle<5, 5, 0>(logical_addr);
        out[OFFSET(n_row + j, n_col, m)] = tile[physical_addr]; // read by row, write by row
    }
}

} // namespace transpose