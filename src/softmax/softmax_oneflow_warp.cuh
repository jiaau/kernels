#pragma once

#include <assert.h>

#include "softmax_utils.cuh"

namespace softmax {

template <typename LOAD,
          typename STORE,
          typename ComputeType,
          int pack_size,
          int cols_per_thread,
          int thread_group_width,
          int rows_per_access,
          bool padding,
          Algorithm algorithm>
__global__ void SoftmaxWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    static_assert(cols_per_thread % pack_size == 0, "");
    constexpr int num_packs = cols_per_thread / pack_size;
    static_assert(thread_group_width <= WARP_SIZE, "");
    static_assert(WARP_SIZE % thread_group_width == 0, "");
    assert(cols <= cols_per_thread * thread_group_width);
    ComputeType buf[rows_per_access][cols_per_thread];
    const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int num_global_thread_group = gridDim.x * blockDim.y;
    const int lane_id = threadIdx.x;
    const int64_t step = num_global_thread_group * rows_per_access;

    // gmem to register, get thread max
    for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {
        ComputeType thread_max[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            thread_max[row_id] = -Inf<ComputeType>();
            ComputeType *row_buf = buf[row_id];
            // pack_id 表示这是第几次 pack
#pragma unroll
            for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
                const int pack_offset = pack_id * pack_size;
                // row 和 col 采取相同的索引方式（col 的 pack_id 对应 row 的 blockIdx.x）
                // 然后乘以每个 id 负责的 size，row 为 rows_per_access，col 为 pack_size
                // 同一线程负责的 packs 不是连续的，跨越了 thread_group_width
                const int col = (pack_id * thread_group_width + lane_id) * pack_size;
                if (!padding || col < cols) {
                    load.template load<pack_size>(row_buf + pack_offset, row + row_id, col);
#pragma unroll
                    for (int i = 0; i < pack_size; ++i) {
                        thread_max[row_id] = max(thread_max[row_id], row_buf[pack_offset + i]);
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < pack_size; ++i) {
                        row_buf[pack_offset + i] = -Inf<ComputeType>();
                    }
                }
            }
        }

        // get row max
        ComputeType warp_max[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            warp_max[row_id] =
                warp_all_reduce<MaxOp, ComputeType, thread_group_width>(thread_max[row_id]);
        }

        // get thread sum
        ComputeType thread_sum[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            thread_sum[row_id] = 0;
            ComputeType *row_buf = buf[row_id];
#pragma unroll
            for (int i = 0; i < cols_per_thread; ++i) {
                if (algorithm == Algorithm::kSoftmax) {
                    row_buf[i] = exp(row_buf[i] - warp_max[row_id]);
                    thread_sum[row_id] += row_buf[i];
                } else if (algorithm == Algorithm::kLogSoftmax) {
                    row_buf[i] -= warp_max[row_id];
                    thread_sum[row_id] += exp(row_buf[i]);
                } else {
                    __trap();
                }
            }
        }

        // get row sum
        ComputeType warp_sum[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            warp_sum[row_id] =
                warp_all_reduce<SumOp, ComputeType, thread_group_width>(thread_sum[row_id]);
        }

        // register to gmem
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            ComputeType *row_buf = buf[row_id];
#pragma unroll
            for (int i = 0; i < cols_per_thread; ++i) {
                if (algorithm == Algorithm::kSoftmax) {
                    row_buf[i] /= warp_sum[row_id];
                } else if (algorithm == Algorithm::kLogSoftmax) {
                    row_buf[i] -= log(warp_sum[row_id]);
                } else {
                    __trap();
                }
            }
#pragma unroll
            for (int i = 0; i < num_packs; ++i) {
                const int col = (i * thread_group_width + lane_id) * pack_size;
                if (!padding || col < cols) {
                    store.template store<pack_size>(row_buf + i * pack_size, row + row_id, col);
                }
            }
        }
    }
}

template <typename LOAD,
          typename STORE,
          typename ComputeType,
          int pack_size,
          int cols_per_thread,
          int thread_group_width,
          int rows_per_access,
          bool padding,
          Algorithm algorithm>
inline cudaError_t
LaunchSoftmaxWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    constexpr int block_size = 128;
    constexpr int waves = 32;
    static_assert(block_size % thread_group_width == 0, "");
    constexpr int thread_groups_per_block = block_size / thread_group_width;
    dim3 block_dim(thread_group_width, thread_groups_per_block);
    const int64_t num_blocks =
        (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, num_blocks, waves, &grid_dim_x);
        if (err != cudaSuccess) { return err; }
    }
    SoftmaxWarpImpl<LOAD,
                    STORE,
                    ComputeType,
                    pack_size,
                    cols_per_thread,
                    thread_group_width,
                    rows_per_access,
                    padding,
                    algorithm><<<grid_dim_x, block_dim>>>(load, store, rows, cols);
    return cudaPeekAtLastError();
}

template <typename LOAD,
          typename STORE,
          typename ComputeType,
          int pack_size,
          int cols_per_thread,
          int thread_group_width,
          int rows_per_access,
          Algorithm algorithm>
inline cudaError_t
DispatchSoftmaxWarpImplPadding(LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    if (cols == cols_per_thread * thread_group_width) {
        return LaunchSoftmaxWarpImpl<LOAD,
                                     STORE,
                                     ComputeType,
                                     pack_size,
                                     cols_per_thread,
                                     thread_group_width,
                                     rows_per_access,
                                     false,
                                     algorithm>(load, store, rows, cols);
    } else {
        return LaunchSoftmaxWarpImpl<LOAD,
                                     STORE,
                                     ComputeType,
                                     pack_size,
                                     cols_per_thread,
                                     thread_group_width,
                                     rows_per_access,
                                     true,
                                     algorithm>(load, store, rows, cols);
    }
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
typename std::enable_if<pack_size == 1, cudaError_t>::type
DispatchSoftmaxWarpImplCols(LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    if (cols <= 0) { return cudaErrorInvalidValue; }
#define ELIF(thread_group_width)                                                                   \
    else if (cols <= (thread_group_width) * pack_size) {                                           \
        if (rows % 2 == 0) {                                                                       \
            return DispatchSoftmaxWarpImplPadding<LOAD,                                            \
                                                  STORE,                                           \
                                                  ComputeType,                                     \
                                                  pack_size,                                       \
                                                  pack_size,                                       \
                                                  thread_group_width,                              \
                                                  2,                                               \
                                                  algorithm>(load, store, rows, cols);             \
        } else {                                                                                   \
            return DispatchSoftmaxWarpImplPadding<LOAD,                                            \
                                                  STORE,                                           \
                                                  ComputeType,                                     \
                                                  pack_size,                                       \
                                                  pack_size,                                       \
                                                  thread_group_width,                              \
                                                  1,                                               \
                                                  algorithm>(load, store, rows, cols);             \
        }                                                                                          \
    }
    ELIF(1)
    ELIF(2)
    ELIF(4)
    ELIF(8)
    ELIF(16)
    ELIF(32)
#undef ELIF
#define ELIF(col)                                                                                  \
    else if (cols <= (col) * WARP_SIZE) {                                                          \
        return DispatchSoftmaxWarpImplPadding<LOAD,                                                \
                                              STORE,                                               \
                                              ComputeType,                                         \
                                              pack_size,                                           \
                                              col,                                                 \
                                              WARP_SIZE,                                           \
                                              1,                                                   \
                                              algorithm>(load, store, rows, cols);                 \
    }
    ELIF(2)
    ELIF(3)
    ELIF(4)
    ELIF(5)
    ELIF(6)
    ELIF(7)
    ELIF(8)
    ELIF(9)
    ELIF(10)
    ELIF(11)
    ELIF(12)
    ELIF(13)
    ELIF(14)
    ELIF(15)
    ELIF(16)
    ELIF(17)
    ELIF(18)
    ELIF(19)
    ELIF(20)
    ELIF(21)
    ELIF(22)
    ELIF(23)
    ELIF(24)
    ELIF(25)
    ELIF(26)
    ELIF(27)
    ELIF(28)
    ELIF(29)
    ELIF(30)
    ELIF(31)
    ELIF(32)
#undef ELIF
    else { return cudaErrorInvalidValue; }
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
typename std::enable_if<pack_size == 2, cudaError_t>::type
DispatchSoftmaxWarpImplCols(LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    if (cols <= 0) { return cudaErrorInvalidValue; }
#define ELIF(thread_group_width)                                                                   \
    else if (cols <= (thread_group_width) * pack_size) {                                           \
        if (rows % 2 == 0) {                                                                       \
            return DispatchSoftmaxWarpImplPadding<LOAD,                                            \
                                                  STORE,                                           \
                                                  ComputeType,                                     \
                                                  pack_size,                                       \
                                                  pack_size,                                       \
                                                  thread_group_width,                              \
                                                  2,                                               \
                                                  algorithm>(load, store, rows, cols);             \
        } else {                                                                                   \
            return DispatchSoftmaxWarpImplPadding<LOAD,                                            \
                                                  STORE,                                           \
                                                  ComputeType,                                     \
                                                  pack_size,                                       \
                                                  pack_size,                                       \
                                                  thread_group_width,                              \
                                                  1,                                               \
                                                  algorithm>(load, store, rows, cols);             \
        }                                                                                          \
    }
    ELIF(1)
    ELIF(2)
    ELIF(4)
    ELIF(8)
    ELIF(16)
    ELIF(32)
#undef ELIF
#define ELIF(col)                                                                                  \
    else if (cols <= (col) * WARP_SIZE) {                                                          \
        return DispatchSoftmaxWarpImplPadding<LOAD,                                                \
                                              STORE,                                               \
                                              ComputeType,                                         \
                                              pack_size,                                           \
                                              col,                                                 \
                                              WARP_SIZE,                                           \
                                              1,                                                   \
                                              algorithm>(load, store, rows, cols);                 \
    }
    ELIF(4)
    ELIF(6)
    ELIF(8)
    ELIF(10)
    ELIF(12)
    ELIF(14)
    ELIF(16)
    ELIF(18)
    ELIF(20)
    ELIF(22)
    ELIF(24)
    ELIF(26)
    ELIF(28)
    ELIF(30)
    ELIF(32)
#undef ELIF
    else { return cudaErrorInvalidValue; }
}

template <typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
struct DispatchSoftmaxWarpImplPackSize {
    cudaError_t operator()(LOAD load, STORE store, const int64_t rows, const int64_t cols) {
        if (cols % 2 == 0) {
            return DispatchSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 2, algorithm>(
                load, store, rows, cols);
        } else {
            return DispatchSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 1, algorithm>(
                load, store, rows, cols);
        }
    }
};

template <typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t
DispatchSoftmaxWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    return DispatchSoftmaxWarpImplPackSize<LOAD, STORE, ComputeType, algorithm>()(
        load, store, rows, cols);
}

} // namespace softmax