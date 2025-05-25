#pragma once

#include <chrono>
#include "utils.cuh"

template <typename T>
void cpu_max(T *in, T *out, const int row, const int col) {
    for (int i = 0; i < row; i++) {
        T pmax = -INFINITY;
        for (int j = 0; j < col; j++) {
            pmax = std::max(pmax, in[OFFSET(i, j, col)]);
        }
        out[i] = pmax;
    }
}

/**
 * \brief Test function for row-wise reduction max on a 512x512 matrix
 * \note Computes row-wise max of a 512x512 matrix, resulting in a 512-element vector
 * \note Validates results against CPU reference implementation
 */
bool test_max_512x512(void (*kernel)(float *, float *, const int),
                      bool bench = false,
                      int times = 3) {
    const int row = 512;
    const int col = 512;
    const int n = row * col;
    float *in_h = (float *)malloc(n * sizeof(float));
    float *out_h = (float *)malloc(row * sizeof(float));

    fill_data(in_h, n);

    float *in_d, *out_d;
    cudaMalloc(&in_d, n * sizeof(float));
    cudaMalloc(&out_d, row * sizeof(float));

    cudaMemcpy(in_d, in_h, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridDim(row);
    dim3 blockDim(round_up_thread(col));
    kernel<<<gridDim, blockDim>>>(in_d, out_d, col);

    cudaDeviceSynchronize();
    cudaMemcpy(out_h, out_d, row * sizeof(float), cudaMemcpyDeviceToHost);

    float *out_h_ref = (float *)malloc(row * sizeof(float));

    cpu_max(in_h, out_h_ref, row, col);

    bool ret = !diff(out_h, out_h_ref, row);

    if (ret && bench) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < times; i++) {
            kernel<<<gridDim, blockDim>>>(in_d, out_d, col);
            cudaDeviceSynchronize();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::string line(90, '-');
        std::cout << line << std::endl;
        std::cout << "reduce time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>((t1 - t0) /
                                                                           times)
                         .count()
                  << "us" << std::endl;
    }

    free(in_h);
    free(out_h);
    free(out_h_ref);
    cudaFree(in_d);
    cudaFree(out_d);
    return ret;
}