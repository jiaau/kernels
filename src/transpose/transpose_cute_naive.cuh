#pragma once

#include <cute/tensor.hpp>

namespace transpose {

using namespace cute;

// gridDim (n/TILE_N, m/TILE_M, 1),   blockDim (THREAD_X, THREAD_Y, 1)
template <unsigned threadX = 32, unsigned threadY = 8>
__global__ void transpose_cute_naive(float *out, const float *in, const int m, const int n) {
    // Make Tensors
    auto in_shape = make_shape(m, n);
    auto out_shape = make_shape(n, m);

    auto gemem_layout_in = make_layout(in_shape, LayoutRight{});
    Tensor tensor_in = make_tensor(make_gmem_ptr(in), gemem_layout_in);

    // Make a transposed view of the output
    auto gemem_layout_out = make_layout(out_shape, LayoutRight{});
    Tensor tensor_out = make_tensor(make_gmem_ptr(out), gemem_layout_out);

    // Tile tensors
    using tileM = Int<32>;
    using tileN = Int<32>;

    auto block_shape_in = make_shape(tileM{}, tileN{});  // (tileM, tileN)
    auto block_shape_out = make_shape(tileN{}, tileM{}); // (tileN, tileM)

    Tensor tiled_tensor_in = tiled_divide(tensor_in, block_shape_in);    // ((tileM, tileN), m', n')
    Tensor tiled_tensor_out = tiled_divide(tensor_out, block_shape_out); // ((tileN, tileM), n', m')

    auto thread_layout_in =
        make_layout(make_shape(Int<threadY>{}, Int<threadX>{}), LayoutRight{}); // read by row
    auto thread_layout_out =
        make_layout(make_shape(Int<threadX>{}, Int<threadY>{}), GenColMajor{}); // write by column

    // ------------------------------------------------------------------------------------

    Tensor g_in = tiled_tensor_in(make_coord(_, _), blockIdx.y, blockIdx.x);   // (tileM, tileN)
    Tensor g_out = tiled_tensor_out(make_coord(_, _), blockIdx.x, blockIdx.y); // (tileN, tileM)

    auto tid = threadIdx.y * blockDim.x + threadIdx.x;

    Tensor t_g_in = local_partition(g_in, thread_layout_in, tid); // (ThrValM, ThrValN)
    Tensor t_g_out = local_partition(g_out, thread_layout_out, tid);

    Tensor rmem = make_tensor_like(t_g_in);

    copy(t_g_in, rmem);
    copy(rmem, t_g_out);
}

} // namespace transpose