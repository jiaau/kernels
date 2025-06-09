#pragma once

#include "transpose_utils.cuh"
#include "utils.cuh"

namespace transpose {

// gridDim (n/TILE_N, m/TILE_M, 1),   blockDim (BLOCK_M, BLOCK_N, 1)
__global__ void
transpose_cuda_coalesced_kernel(float *out, const float *in, const int64_t M, const int64_t N) {

    __shared__ float tile[TILE_M][TILE_N];

    int row_in = blockIdx.y * TILE_M + threadIdx.y;
    int col_in = blockIdx.x * TILE_N + threadIdx.x;

    for (int j = 0; j < TILE_M; j += blockDim.y)
        tile[threadIdx.x][threadIdx.y + j] =
            in[OFFSET(row_in + j, col_in, N)]; // read by row, write by column

    __syncthreads();

    int row_out = blockIdx.x * TILE_N + threadIdx.y;
    int col_out = blockIdx.y * TILE_M + threadIdx.x;

    for (int j = 0; j < TILE_M; j += blockDim.y)
        out[OFFSET(row_out + j, col_out, M)] =
            tile[threadIdx.y + j][threadIdx.x]; // read by row, write by row
}

template <unsigned BLOCK_M = 32, unsigned BLOCK_N = 8>
void transpose_cuda_coalesced(float *out, const float *in, const int64_t M, const int64_t N) {
    dim3 gridDim(N / TILE_N, M / TILE_M, 1);
    dim3 blockDim(BLOCK_M, BLOCK_N, 1);
    transpose_cuda_coalesced_kernel<<<gridDim, blockDim>>>(out, in, M, N);
}

} // namespace transpose