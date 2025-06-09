#pragma once

#include <cute/tensor.hpp>

#include "transpose_utils.cuh"

namespace transpose {

using namespace cute;

// gridDim (n/TILE_N, m/TILE_M, 1),   blockDim (THREAD_X, THREAD_Y, 1)
template <class TensorS, class TensorD, class ThreadLayoutS, class ThreadLayoutD>
__global__ void
transpose_cute_naive_kernel(TensorS S, TensorD D, ThreadLayoutS tS, ThreadLayoutD tD) {
    Tensor gS = S(make_coord(_, _), blockIdx.y, blockIdx.x); // (tileM, tileN)
    Tensor gD = D(make_coord(_, _), blockIdx.x, blockIdx.y); // (tileN, tileM)

    auto tid = threadIdx.y * blockDim.x + threadIdx.x;

    Tensor tSgS = local_partition(gS, tS, tid); // (ThrValM, ThrValN)
    Tensor tDgD = local_partition(gD, tD, tid);

    Tensor rmem = make_tensor_like(tSgS);

    copy(tSgS, rmem);
    copy(rmem, tDgD);
}

template <typename T = float, unsigned BLOCK_M = 32, unsigned BLOCK_N = 8>
void transpose_cute_naive(T *out, const T *in, const int64_t M, const int64_t N) {
    // Make Tensors
    auto s_shape = make_shape(M, N);
    auto d_shape = make_shape(N, M);

    auto gmemLayoutS = make_layout(s_shape, LayoutRight{});
    Tensor tensor_s = make_tensor(make_gmem_ptr(in), gmemLayoutS);

    auto gmemLayoutD = make_layout(d_shape, LayoutRight{});
    Tensor tensor_d = make_tensor(make_gmem_ptr(out), gmemLayoutD);

    // Tile tensors
    using tileM = Int<TILE_M>;
    using tileN = Int<TILE_N>;

    auto block_shape_s = make_shape(tileM{}, tileN{});  // (tileM, tileN)
    auto block_shape_d = make_shape(tileN{}, tileM{}); // (tileN, tileM)

    Tensor tiled_tensor_s = tiled_divide(tensor_s, block_shape_s);  // ((tileM, tileN), m', n')
    Tensor tiled_tensor_d = tiled_divide(tensor_d, block_shape_d); // ((tileN, tileM), n', m')

    auto threadLayoutS =
        make_layout(make_shape(Int<BLOCK_N>{}, Int<BLOCK_M>{}), LayoutRight{}); // read by row
    auto threadLayoutD =
        make_layout(make_shape(Int<BLOCK_M>{}, Int<BLOCK_N>{}), GenColMajor{}); // write by column

    dim3 gridDim(N / TILE_N, M / TILE_M, 1);
    dim3 blockDim(BLOCK_M, BLOCK_N, 1);
    transpose_cute_naive_kernel<<<gridDim, blockDim>>>(
        tiled_tensor_s, tiled_tensor_d, threadLayoutS, threadLayoutD);
}

} // namespace transpose