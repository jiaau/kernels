#pragma once

#include <cute/tensor.hpp>

namespace transpose {

using namespace cute;

// gridDim (n/TILE_N, m/TILE_M, 1),   blockDim (THREAD_X, THREAD_Y, 1)
template <unsigned threadX = 32, unsigned threadY = 8>
__global__ void transpose_cute_naive_colfax(float *out, const float *in, const int m, const int n) {
    //
    // Make Tensors
    //
    auto tensor_shape = make_shape(m, n);
    auto gemem_layout_in = make_layout(tensor_shape, LayoutRight{});
    Tensor tensor_in = make_tensor(make_gmem_ptr(in), gemem_layout_in);

    // Make a transposed view of the output
    auto gemem_layout_out_trans = make_layout(tensor_shape, GenColMajor{});
    Tensor tensor_out_trans = make_tensor(make_gmem_ptr(out), gemem_layout_out_trans);

    //
    // Tile tensors
    //
    using tileM = Int<32>;
    using tileN = Int<32>;

    auto block_shape = make_shape(tileM{}, tileN{}); // (tileM, tileN)
    // auto block_shape = make_shape(blockDim.x, blockDim.x);

    Tensor tiled_tensor_in = tiled_divide(tensor_in, block_shape); // ((tileM, tileN), m', n')
    Tensor tiled_tensor_out_trans =
        tiled_divide(tensor_out_trans, block_shape); // ((tileM, tileN), m', n')

    Tensor g_in = tiled_tensor_in(make_coord(_, _), blockIdx.y, blockIdx.x); // (tileM, tileN)
    Tensor g_out_trans =
        tiled_tensor_out_trans(make_coord(_, _), blockIdx.y, blockIdx.x); // (tileM, tileN)

    auto thread_layout = make_layout(make_shape(Int<threadY>{}, Int<threadX>{}), LayoutRight{});

    Tensor t_g_in = local_partition(g_in,
                                    thread_layout,
                                    threadIdx.y * blockDim.x + threadIdx.x); // (ThrValM, ThrValN)
    Tensor t_g_out_trans =
        local_partition(g_out_trans, thread_layout, threadIdx.y * blockDim.x + threadIdx.x);

    Tensor rmem = make_tensor_like(t_g_in);

    copy(t_g_in, rmem);
    copy(rmem, t_g_out_trans);
}

} // namespace transpose