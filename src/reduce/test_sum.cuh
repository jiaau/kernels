#pragma once

#include <chrono>
#include "utils.cuh"

namespace reduce {

template <typename T>
void cpu_sum(const size_t rows, const size_t cols, const T *in, T *out) {
    for (size_t i = 0; i < rows; i++) {
        T psum = 0.0;
        for (size_t j = 0; j < cols; j++) {
            psum += in[OFFSET(i, j, cols)];
        }
        out[i] = psum;
    }
}

/**
 * \brief Test function for row-wise reduction sum on a 1024x2048 matrix
 * \note Computes row-wise sum of a 1024x2048 matrix, resulting in a 1024-element vector
 * \note Validates results against CPU reference implementation
 */
bool test_sum_1024x2048(void (*gpu_sum)(const size_t, const size_t, const float *, float *),
                        bool bench = false,
                        int times = 3) {
    const size_t rows = 1024;
    const size_t cols = 2048;
    const size_t n = rows * cols;
    float *in_h = (float *)malloc(n * sizeof(float));
    float *out_h = (float *)malloc(rows * sizeof(float));

    fill_data(in_h, n);

    float *in_d, *out_d;
    cudaMalloc(&in_d, n * sizeof(float));
    cudaMalloc(&out_d, rows * sizeof(float));

    cudaMemcpy(in_d, in_h, n * sizeof(float), cudaMemcpyHostToDevice);

    gpu_sum(rows, cols, in_d, out_d);

    cudaDeviceSynchronize();
    cudaMemcpy(out_h, out_d, rows * sizeof(float), cudaMemcpyDeviceToHost);

    float *out_h_ref = (float *)malloc(rows * sizeof(float));

    cpu_sum(rows, cols, in_h, out_h_ref);

    bool ret = !diff(out_h, out_h_ref, rows);

    if (ret && bench) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < times; i++) {
            gpu_sum(rows, cols, in_d, out_d);
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