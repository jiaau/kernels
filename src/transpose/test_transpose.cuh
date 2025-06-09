#pragma once

#include <chrono>
#include <cstdint>
#include "utils.cuh"

namespace transpose {

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
template <int64_t M = 1024, int64_t N = 2048>
bool test_transpose(void (*gpu_transpose)(float *, const float *, const int64_t, const int64_t),
                    bool bench = false,
                    int times = 3) {
    float *in_h = (float *)malloc(M * N * sizeof(float));
    float *out_h = (float *)malloc(N * M * sizeof(float));

    fill_data(in_h, M * N);

    float *in_d, *out_d;
    cudaMalloc(&in_d, M * N * sizeof(float));
    cudaMalloc(&out_d, N * M * sizeof(float));

    cudaMemcpy(in_d, in_h, M * N * sizeof(float), cudaMemcpyHostToDevice);

    gpu_transpose(out_d, in_d, M, N);

    cudaDeviceSynchronize();
    cudaMemcpy(out_h, out_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    float *out_h_ref = (float *)malloc(N * M * sizeof(float));

    cpu_transpose(out_h_ref, in_h, M, N);

    bool ret = !diff(out_h, out_h_ref, N * M);

    if (ret && bench) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < times; i++) {
            gpu_transpose(out_d, in_d, M, N);
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

} // namespace transpose