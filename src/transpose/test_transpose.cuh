#pragma once

#include <chrono>
#include "utils.cuh"

void cpu_transpose(float *out, const float *in, const int m, const int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            out[OFFSET(j, i, m)] = in[OFFSET(i, j, n)];
        }
    }
}

/**
 * \brief Test function for Transpose with 1024x2048 matrices
 * \note Validates results against CPU reference implementation
 */
template <unsigned THREAD_X = 32, unsigned THREAD_Y = 8>
bool test_transpose_1024x2048(void (*transpose)(float *, const float *, const int, const int),
                              bool bench = false,
                              int times = 3) {
    const int m = 1024, n = 2048;
    float *in_h = (float *)malloc(m * n * sizeof(float));
    float *out_h = (float *)malloc(n * m * sizeof(float));

    fill_data(in_h, m * n);

    float *in_d, *out_d;
    cudaMalloc(&in_d, m * n * sizeof(float));
    cudaMalloc(&out_d, n * m * sizeof(float));

    cudaMemcpy(in_d, in_h, m * n * sizeof(float), cudaMemcpyHostToDevice);

    const int TIME_M = 32;
    const int TIME_N = 32;

    dim3 gridDim(n / TIME_N, m / TIME_M, 1);
    dim3 blockDim(THREAD_X, THREAD_Y, 1);
    transpose<<<gridDim, blockDim>>>(out_d, in_d, m, n);

    cudaDeviceSynchronize();
    cudaMemcpy(out_h, out_d, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    float *out_h_ref = (float *)malloc(n * m * sizeof(float));

    cpu_transpose(out_h_ref, in_h, m, n);

    bool ret = !diff(out_h, out_h_ref, n * m);

    if (ret && bench) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < times; i++) {
            transpose<<<gridDim, blockDim>>>(out_d, in_d, m, n);
            cudaDeviceSynchronize();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::string line(100, '-');
        std::cout << line << std::endl;
        std::cout
            << "transpose time: "
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