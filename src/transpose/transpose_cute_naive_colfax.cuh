#pragma once

#include <cute/tensor.hpp>

#include "transpose_utils.cuh"

namespace transpose {

using namespace cute;

// gridDim (n/TILE_N, m/TILE_M, 1),   blockDim (THREAD_X, THREAD_Y, 1)
template <class TensorS, class TensorD, class ThreadLayoutS, class ThreadLayoutD>
__global__ void
transpose_cute_naive_colfax_kernel(TensorS S, TensorD D, ThreadLayoutS tS, ThreadLayoutD tD) {
    Tensor gS = S(make_coord(_, _), blockIdx.y, blockIdx.x); // (tileM, tileN)
    Tensor gD = D(make_coord(_, _), blockIdx.y, blockIdx.x); // (tileN, tileM)

    auto tid = threadIdx.y * blockDim.x + threadIdx.x;

    Tensor tSgS = local_partition(gS, tS, tid); // (ThrValM, ThrValN)
    Tensor tDgD = local_partition(gD, tD, tid);

    Tensor rmem = make_tensor_like(tSgS);

    copy(tSgS, rmem);
    copy(rmem, tDgD);
}

template <typename T = float, unsigned BLOCK_M = 32, unsigned BLOCK_N = 8>
void transpose_cute_naive_colfax(T *out, const T *in, const int64_t M, const int64_t N) {
    // Make Tensors
    auto shape = make_shape(M, N);
    auto gmemLayoutS = make_layout(shape, LayoutRight{});
    Tensor tensor_s = make_tensor(make_gmem_ptr(in), gmemLayoutS);

    auto gmemLayoutDT = make_layout(shape, GenColMajor{});
    Tensor tensor_dt = make_tensor(make_gmem_ptr(out), gmemLayoutDT);

    // Tile tensors
    using tileM = Int<TILE_M>;
    using tileN = Int<TILE_N>;

    auto block_shape = make_shape(tileM{}, tileN{}); // (tileM, tileN)

    Tensor tiled_tensor_s = tiled_divide(tensor_s, block_shape); // ((tileM, tileN), m', n')
    Tensor tiled_tensor_dt = tiled_divide(tensor_dt, block_shape); // ((tileM, tileN), m', n')

    auto threadLayoutS =
        make_layout(make_shape(Int<BLOCK_N>{}, Int<BLOCK_M>{}), LayoutRight{}); 
    auto threadLayoutDT =
        make_layout(make_shape(Int<BLOCK_N>{}, Int<BLOCK_M>{}), LayoutRight{});

    dim3 gridDim(N / TILE_N, M / TILE_M, 1);
    dim3 blockDim(BLOCK_M, BLOCK_N, 1);
    transpose_cute_naive_colfax_kernel<<<gridDim, blockDim>>>(
        tiled_tensor_s, tiled_tensor_dt, threadLayoutS, threadLayoutDT);
}

} // namespace transpose