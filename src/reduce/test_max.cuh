#pragma once

#include <chrono>
#include "utils.cuh"

namespace reduce {

template <typename T>
void cpu_max(const T *in, T *out, const int64_t rows, const int64_t cols) {
    for (int64_t i = 0; i < rows; i++) {
        T pmax = -INFINITY;
        for (int64_t j = 0; j < cols; j++) {
            pmax = std::max(pmax, in[OFFSET(i, j, cols)]);
        }
        out[i] = pmax;
    }
}

/**
 * \brief Test function for row-wise reduction max on a 1024x2048 matrix
 * \note Computes row-wise max of a 1024x2048 matrix, resulting in a 1024-element vector
 * \note Validates results against CPU reference implementation
 */
template <int64_t rows = 1024, int64_t cols = 2048>
bool test_max(void (*gpu_max)(const float *, float *, const int64_t, const int64_t),
              bool bench = false,
              int times = 3) {
    const int64_t n = rows * cols;
    float *in_h = (float *)malloc(n * sizeof(float));
    float *out_h = (float *)malloc(rows * sizeof(float));

    fill_data(in_h, n);

    float *in_d, *out_d;
    cudaMalloc(&in_d, n * sizeof(float));
    cudaMalloc(&out_d, rows * sizeof(float));

    cudaMemcpy(in_d, in_h, n * sizeof(float), cudaMemcpyHostToDevice);

    gpu_max(in_d, out_d, rows, cols);

    cudaDeviceSynchronize();
    cudaMemcpy(out_h, out_d, rows * sizeof(float), cudaMemcpyDeviceToHost);

    float *out_h_ref = (float *)malloc(rows * sizeof(float));

    cpu_max(in_h, out_h_ref, rows, cols);

    bool ret = !diff(out_h, out_h_ref, rows);

    if (ret && bench) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < times; i++) {
            gpu_max(in_d, out_d, rows, cols);
            cudaDeviceSynchronize();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::string line(90, '-');
        std::cout << line << std::endl;
        std::cout
            << "reduce time: "
            << std::chrono::duration_cast<std::chrono::microseconds>((t1 - t0) / times).count()
            << "us" << std::endl;
    }

    free(in_h);
    free(out_h);
    free(out_h_ref);
    cudaFree(in_d);
    cudaFree(out_d);
    return ret;
}

} // namespace reduce