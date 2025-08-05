#pragma once

#include <assert.h>

#include "softmax_utils.cuh"

namespace softmax {

template <typename LOAD,
          typename STORE,
          typename ComputeType,
          int pack_size,
          int block_size,
          Algorithm algorithm>
__global__ void
SoftmaxBlockSMemImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto *buf = reinterpret_cast<ComputeType *>(shared_buf);
    const int tid = threadIdx.x;
    assert(cols % pack_size == 0);
    const int num_packs = cols / pack_size;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        ComputeType thread_max = -Inf<ComputeType>();
        // 这里 pack_id 的含义和 softmax_oneflow_warp.cuh 中的 pack_id 不同
        // 这里 pack_id 表示这是第几个 pack，比如 8 cols，pack_size 为 2，那么 pack_id 为 0, 1, 2, 3
        // softmax_oneflow_warp.cuh
        // 中的索引方式适用于双层索引模型，外层索引块，内存索引块内具体的工作 这里如果要使用
        // softmax_oneflow_warp.cuh 中的索引方式，需要修改： num_packs = cols / pack_size; ->
        // num_packs = cols / (thread_group_width * pack_size);
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            ComputeType pack[pack_size];
            load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
            // for (int i = 0; i < pack_size; ++i) {
            //     buf[i * num_packs + pack_id] = pack[i];
            //     thread_max = max(thread_max, pack[i]);
            // }
            for (int j = 0; j < pack_size; ++j) {
                // int lane = tid % 32;
                // int new_j = j;
                // if (j == 0) {
                //     if (lane < 15)
                //         new_j = 0;
                //     else
                //         new_j = 1;
                // } else {
                //     if (lane < 15)
                //         new_j = 1;
                //     else
                //         new_j = 0;
                // }
                buf[pack_id * pack_size + j] = pack[j];
                thread_max = max(thread_max, pack[j]);
            }
        }
        const ComputeType row_max = block_all_reduce<MaxOp, ComputeType, block_size>(thread_max);
        ComputeType thread_sum = 0;
        for (int col = tid; col < cols; col += block_size) {
            if (algorithm == Algorithm::kSoftmax) {
                const ComputeType exp_x = exp(buf[col] - row_max);
                buf[col] = exp_x;
                thread_sum += exp_x;
            } else {
                const ComputeType x = buf[col] - row_max;
                buf[col] = x;
                thread_sum += exp(x);
            }
        }
        const ComputeType row_sum = block_all_reduce<SumOp, ComputeType, block_size>(thread_sum);
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            ComputeType pack[pack_size];
#pragma unroll
            // for (int i = 0; i < pack_size; ++i) {
            //     if (algorithm == Algorithm::kSoftmax) {
            //         pack[i] = buf[i * num_packs + pack_id] / row_sum;
            //     } else if (algorithm == Algorithm::kLogSoftmax) {
            //         pack[i] = buf[i * num_packs + pack_id] - log(row_sum);
            //     } else {
            //         __trap();
            //     }
            // }
            for (int j = 0; j < pack_size; ++j) {
                if (algorithm == Algorithm::kSoftmax) {
                    // int lane = tid % 32;
                    // int new_j = j;
                    // if (j == 0) {
                    //     if (lane < 15)
                    //         new_j = 0;
                    //     else
                    //         new_j = 1;
                    // } else {
                    //     if (lane < 15)
                    //         new_j = 1;
                    //     else
                    //         new_j = 0;
                    // }
                    pack[j] = buf[pack_id * pack_size + j] / row_sum;
                } else if (algorithm == Algorithm::kLogSoftmax) {
                    pack[j] = buf[pack_id * pack_size + j] - log(row_sum);
                } else {
                    __trap();
                }
            }
            store.template store<pack_size>(pack, row, pack_id * pack_size);
        }
    }
}

template <typename LOAD,
          typename STORE,
          typename ComputeType,
          int pack_size,
          int block_size,
          Algorithm algorithm>
inline cudaError_t LaunchSoftmaxBlockSMemImpl(
    LOAD load, STORE store, int smem, const int64_t rows, const int64_t cols) {
    constexpr int waves = 32;
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
        if (err != cudaSuccess) { return err; }
    }
    SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size, algorithm>
        <<<grid_dim_x, block_size, smem>>>(load, store, rows, cols);
    return cudaPeekAtLastError();
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxBlockSMemImplBlockSize(
    LOAD load, STORE store, const int64_t rows, const int64_t cols, bool *success) {
    constexpr int block_size_conf_1 = 128;
    constexpr int block_size_conf_2 = 256;
    constexpr int block_size_conf_3 = 512;
    constexpr int block_size_conf_4 = 1024;
    const size_t smem = cols * sizeof(ComputeType);
    int max_active_blocks_conf_1;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_1,
            SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1, algorithm>,
            block_size_conf_1,
            smem);
        if (err != cudaSuccess) { return err; }
    }
    if (max_active_blocks_conf_1 <= 0) {
        *success = false;
        return cudaSuccess;
    }
    int max_active_blocks_conf_4;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_4,
            SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4, algorithm>,
            block_size_conf_4,
            smem);
        if (err != cudaSuccess) { return err; }
    }
    if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
        *success = true;
        return LaunchSoftmaxBlockSMemImpl<LOAD,
                                          STORE,
                                          ComputeType,
                                          pack_size,
                                          block_size_conf_4,
                                          algorithm>(load, store, smem, rows, cols);
    }
    int max_active_blocks_conf_3;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_3,
            SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3, algorithm>,
            block_size_conf_3,
            smem);
        if (err != cudaSuccess) { return err; }
    }
    if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
        *success = true;
        return LaunchSoftmaxBlockSMemImpl<LOAD,
                                          STORE,
                                          ComputeType,
                                          pack_size,
                                          block_size_conf_3,
                                          algorithm>(load, store, smem, rows, cols);
    }
    int max_active_blocks_conf_2;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_2,
            SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2, algorithm>,
            block_size_conf_2,
            smem);
        if (err != cudaSuccess) { return err; }
    }
    if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
        *success = true;
        return LaunchSoftmaxBlockSMemImpl<LOAD,
                                          STORE,
                                          ComputeType,
                                          pack_size,
                                          block_size_conf_2,
                                          algorithm>(load, store, smem, rows, cols);
    }
    *success = true;
    return LaunchSoftmaxBlockSMemImpl<LOAD,
                                      STORE,
                                      ComputeType,
                                      pack_size,
                                      block_size_conf_1,
                                      algorithm>(load, store, smem, rows, cols);
}

template <typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
struct TryDispatchSoftmaxBlockSMemImplPackSize {
    cudaError_t
    operator()(LOAD load, STORE store, const int64_t rows, const int64_t cols, bool *success) {
        if (cols % 2 == 0) {
            return TryDispatchSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 2, algorithm>(
                load, store, rows, cols, success);
        } else {
            return TryDispatchSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 1, algorithm>(
                load, store, rows, cols, success);
        }
    }
};

template <typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxBlockSMemImpl(
    LOAD load, STORE store, const int64_t rows, const int64_t cols, bool *success) {
    return TryDispatchSoftmaxBlockSMemImplPackSize<LOAD, STORE, ComputeType, algorithm>()(
        load, store, rows, cols, success);
}

} // namespace softmax