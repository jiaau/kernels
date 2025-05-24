#pragma once

#include <chrono>
#include "utils.cuh"

void cpu_sgemm(float *a, float *b, float *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

/**
 * \brief Test function for SGEMM (Single-precision General Matrix Multiplication) with
 * 512x512 matrices
 * \note Computes C = A * B for 512x512 matrices with K=512
 * \note Validates results against CPU reference implementation
 */
bool test_sgemm_512x512_x512(
    void (*kernel)(float *, float *, float *, const int, const int, const int),
    bool bench = false,
    int times = 3) {
    const int n = 512 * 512 * 512;
    float *a_h = (float *)malloc(n * sizeof(float));
    float *b_h = (float *)malloc(n * sizeof(float));
    float *c_h = (float *)malloc(n * sizeof(float));

    fill_data(a_h, n);
    fill_data(b_h, n);

    float *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, n * sizeof(float));
    cudaMalloc(&b_d, n * sizeof(float));
    cudaMalloc(&c_d, n * sizeof(float));

    cudaMemcpy(a_d, a_h, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, n * sizeof(float), cudaMemcpyHostToDevice);

    const int M = 512, N = 512, K = 512;
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    kernel<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);

    cudaDeviceSynchronize();
    cudaMemcpy(c_h, c_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    float *c_ref = (float *)malloc(n * sizeof(float));

    cpu_sgemm(a_h, b_h, c_ref, M, N, K);

    bool ret = !diff(c_h, c_ref, n);

    if (ret && bench) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < times; i++) {
            kernel<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);
            cudaDeviceSynchronize();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "sgemm time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>((t1 - t0) /
                                                                           times)
                         .count()
                  << "us" << std::endl;
    }

    free(a_h);
    free(b_h);
    free(c_h);
    free(c_ref);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    return ret;
}