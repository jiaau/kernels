#pragma once

#include <cute/tensor.hpp>
#include <cutlass/detail/layout.hpp>
#include "cute/arch/copy_sm80.hpp"
#include "cute/layout.hpp"
#include "cute/stride.hpp"

namespace transpose {

using namespace cute;

// gridDim (n/TILE_N, m/TILE_M, 1),   blockDim (THREAD_X, THREAD_Y, 1)
template <unsigned threadX = 32, unsigned threadY = 8>
__global__ void transpose_cute_coalesced(float *out, const float *in, const int m, const int n) {
    // Make Tensors
    auto in_shape = make_shape(m, n);
    auto gemem_layout_in = make_layout(in_shape, LayoutRight{});
    Tensor tensor_in = make_tensor(make_gmem_ptr(in), gemem_layout_in);

    auto out_shape = make_shape(n, m);
    auto gemem_layout_out = make_layout(out_shape, LayoutRight{});
    Tensor tensor_out = make_tensor(make_gmem_ptr(out), gemem_layout_out);

    // Tile tensors
    using TILE_M = Int<32>;
    using TILE_N = Int<32>;

    auto block_shape_in = make_shape(TILE_M{}, TILE_N{});  // (TILE_M, TILE_N)
    auto block_shape_out = make_shape(TILE_N{}, TILE_M{}); // (TILE_N, TILE_M)

    Tensor tiled_tensor_in = tiled_divide(tensor_in, block_shape_in); // ((TILE_M, TILE_N), m', n')
    Tensor tiled_tensor_out =
        tiled_divide(tensor_out, block_shape_out); // ((TILE_N, TILE_M), n', m')

    // key point:
    auto smem_layout_in = make_layout(block_shape_in, GenColMajor{});
    auto smem_layout_out = make_layout(block_shape_out, GenRowMajor{});
    // ? why this works? or:
    // auto smem_layout_in = make_layout(block_shape_in, LayoutRight{});
    // auto smem_layout_out =
    //     composition(smem_layout_in, make_layout(block_shape_out, LayoutRight{}));

    // ? Sequential swapping does not affect the correctness of the result, but does
    // ? affect the number of bank conflicts
    auto thread_layout_in = make_layout(make_shape(Int<threadY>{}, Int<threadX>{}), LayoutRight{});
    auto thread_layout_out = make_layout(make_shape(Int<threadY>{}, Int<threadX>{}), LayoutRight{});

    __shared__ float tile[32][32];
    using CuteArray = array_aligned<float, cosize_v<decltype(smem_layout_in)>>;
    CuteArray &smem = *reinterpret_cast<CuteArray *>(tile);
    auto smem_in = make_tensor(make_smem_ptr(smem.data()), smem_layout_in);
    auto smem_out = make_tensor(make_smem_ptr(smem.data()), smem_layout_out);

    // ------------------------------------------------------------------------------------

    Tensor gmem_in = tiled_tensor_in(make_coord(_, _), blockIdx.y, blockIdx.x);   //(TILE_M, TILE_N)
    Tensor gmem_out = tiled_tensor_out(make_coord(_, _), blockIdx.x, blockIdx.y); //(TILE_N, TILE_M)

    auto tid = threadIdx.y * blockDim.x + threadIdx.x;

    Tensor t_in_s_in = local_partition(smem_in, thread_layout_in, tid);
    Tensor t_out_s_out = local_partition(smem_out, thread_layout_out, tid);

    Tensor t_in_g_in = local_partition(gmem_in, thread_layout_in, tid); // (ThrValM, ThrValN)
    Tensor t_out_g_out = local_partition(gmem_out, thread_layout_out, tid);

    copy(t_in_g_in, t_in_s_in); // read by row, write by column, leads to store bank conflicts

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    copy(t_out_s_out, t_out_g_out); // read by row, write by row
}

} // namespace transpose