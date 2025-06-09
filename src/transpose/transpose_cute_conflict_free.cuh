#pragma once

#include <cute/tensor.hpp>
#include <cutlass/detail/layout.hpp>
#include "cute/arch/copy_sm80.hpp"
#include "cute/layout.hpp"
#include "cute/stride.hpp"
#include "cute/swizzle.hpp"

#include "transpose_utils.cuh"

namespace transpose {

using namespace cute;

// gridDim (n/TILE_N, m/TILE_M, 1),   blockDim (BLOCK_M, BLOCK_N, 1)
template <typename T,
          class TensorS,
          class TensorD,
          class SmemLayoutS,
          class ThreadLayoutS,
          class SmemLayoutD,
          class ThreadLayoutD>
__global__ void transpose_cute_conflict_free_kernel(TensorS const S,
                                                TensorD const D,
                                                SmemLayoutS const smemLayoutS,
                                                ThreadLayoutS const tS,
                                                SmemLayoutD const smemLayoutD,
                                                ThreadLayoutD const tD) {
    __shared__ T tile[TILE_M][TILE_N];
    using CuteArray = array_aligned<T, cosize_v<decltype(smemLayoutS)>>;
    CuteArray &smem = *reinterpret_cast<CuteArray *>(tile);
    auto sS = make_tensor(make_smem_ptr(smem.data()), smemLayoutS);
    auto sD = make_tensor(make_smem_ptr(smem.data()), smemLayoutD);

    // ------------------------------------------------------------------------------------

    Tensor gS = S(make_coord(_, _), blockIdx.y, blockIdx.x); //(TILE_M, TILE_N)
    Tensor gD = D(make_coord(_, _), blockIdx.x, blockIdx.y); //(TILE_N, TILE_M)

    auto tid = threadIdx.y * blockDim.x + threadIdx.x;

    Tensor tSsS = local_partition(sS, tS, tid);
    Tensor tDsD = local_partition(sD, tD, tid);

    Tensor tSgS = local_partition(gS, tS, tid); // (ThrValM, ThrValN)
    Tensor tDgD = local_partition(gD, tD, tid);

    copy(tSgS, tSsS); // read by row, write by column, leads to store bank conflicts

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    copy(tDsD, tDgD); // read by row, write by row
}

template <typename T = float, unsigned BLOCK_M = 32, unsigned BLOCK_N = 8>
void transpose_cute_conflict_free(T *out, const T *in, const int64_t M, const int64_t N) {
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

    auto block_shape_s = make_shape(tileM{}, tileN{}); // (tileM, tileN)
    auto block_shape_d = make_shape(tileN{}, tileM{}); // (tileN, tileM)

    Tensor tiled_tensor_s = tiled_divide(tensor_s, block_shape_s); // ((tileM, tileN), m', n')
    Tensor tiled_tensor_d = tiled_divide(tensor_d, block_shape_d); // ((tileN, tileM), n', m')

    // ? Sequential swapping does not affect the correctness of the result, but does
    // ? affect the number of bank conflicts
    auto threadLayoutS = make_layout(make_shape(Int<BLOCK_N>{}, Int<BLOCK_M>{}), LayoutRight{});
    auto threadLayoutD = make_layout(make_shape(Int<BLOCK_N>{}, Int<BLOCK_M>{}), LayoutRight{});

    // key point:
    auto smemLayoutS = composition(Swizzle<5, 0, 5>{},make_layout(block_shape_s, GenColMajor{}));
    auto smemLayoutD = composition(smemLayoutS,make_layout(block_shape_d, GenRowMajor{}));
    // ? why this works? or:
    // auto smemLayoutS = make_layout(block_shape_s, LayoutRight{});
    // auto smemLayoutD =
    //     composition(smemLayoutS, make_layout(block_shape_d, LayoutRight{}));

    dim3 gridDim(N / TILE_N, M / TILE_M, 1);
    dim3 blockDim(BLOCK_N, BLOCK_M, 1);
    transpose_cute_conflict_free_kernel<T><<<gridDim, blockDim>>>(
        tiled_tensor_s, tiled_tensor_d, smemLayoutS, threadLayoutS, smemLayoutD, threadLayoutD);
}

} // namespace transpose