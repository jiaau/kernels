#include <cuda_fp16.h>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#include <iostream>

#include "softmax_oneflow_block_smem.cuh"
#include "softmax_oneflow_warp.cuh"

namespace softmax {

template <typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchSoftmax(LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    // 优先使用不需要同步的实现
    if (cols <= 1024) {
        return DispatchSoftmaxWarpImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(
            load, store, rows, cols);
    } else {
        bool dispatch_smem_impl_success;
        {
            cudaError_t err =
                TryDispatchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(
                    load, store, rows, cols, &dispatch_smem_impl_success);
            if (err != cudaSuccess) { return err; }
        }
        if (!dispatch_smem_impl_success) { return cudaErrorInvalidValue; }
        return cudaSuccess;
    }
}

template <typename T>
struct DefaultComputeType {
    using type = T;
};

template <>
struct DefaultComputeType<half> {
    using type = float;
};

#if CUDA_VERSION >= 11000
template <>
struct DefaultComputeType<nv_bfloat16> {
    using type = float;
};
#endif // CUDA_VERSION >= 11000

#define CUDA_CHECK(condition)                                                                      \
    for (cudaError_t _cuda_check_status = (condition); _cuda_check_status != cudaSuccess;)         \
    std::cout << "Check failed: " #condition " : " << cudaGetErrorString(_cuda_check_status)       \
              << " (" << _cuda_check_status << ") "

template <Algorithm algorithm, typename T>
void softmax_oneflow(const int rows, const int cols, const T *x, T *y) {
    using ComputeType = typename DefaultComputeType<T>::type;
    DirectLoad<T, ComputeType> load(x, cols);
    DirectStore<ComputeType, T> store(y, cols);
    if constexpr (algorithm == Algorithm::kSoftmax) {
        CUDA_CHECK((DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
            load, store, rows, cols)));
    } else {
        throw std::runtime_error("UNIMPLEMENTED");
    }
}

} // namespace softmax