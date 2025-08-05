#pragma once
#include <chrono>
#include <cmath>
#include <cstring>
#include "utils.cuh"

namespace softmax {

void cpu_softmax(const int rows, const int cols, const float *in, float *out) {
    float *row_max = (float *)malloc(rows * sizeof(float));
    memset(row_max, 0, rows * sizeof(float));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            row_max[i] = std::max(row_max[i], in[OFFSET(i, j, cols)]);
        }
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[OFFSET(i, j, cols)] = exp(in[OFFSET(i, j, cols)] - row_max[i]);
        }
    }

    float *row_sum = (float *)malloc(rows * sizeof(float));
    memset(row_sum, 0, rows * sizeof(float));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            row_sum[i] += out[OFFSET(i, j, cols)];
        }
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[OFFSET(i, j, cols)] /= row_sum[i];
        }
    }

    free(row_max);
    free(row_sum);
}

/**
 * \brief Test function for row-wise softmax on a 2048x1024 matrix
 */
template <int rows = 2048, int cols = 1024>
bool test_softmax(void (*gpu_softmax)(const int, const int, const float *, float *),
                  bool bench = false,
                  int times = 3) {
    const int n = rows * cols;
    float *in_h = (float *)malloc(n * sizeof(float));
    float *out_h = (float *)malloc(n * sizeof(float));

    fill_data(in_h, n);

    float *in_d, *out_d;
    cudaMalloc(&in_d, n * sizeof(float));
    cudaMalloc(&out_d, n * sizeof(float));

    cudaMemcpy(in_d, in_h, n * sizeof(float), cudaMemcpyHostToDevice);

    gpu_softmax(rows, cols, in_d, out_d);

    cudaDeviceSynchronize();
    cudaMemcpy(out_h, out_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    float *out_h_ref = (float *)malloc(n * sizeof(float));

    cpu_softmax(rows, cols, in_h, out_h_ref);

    bool ret = !diff(out_h, out_h_ref, n);

    if (ret && bench) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < times; i++) {
            gpu_softmax(rows, cols, in_d, out_d);
            cudaDeviceSynchronize();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::string line(90, '-');
        std::cout << line << std::endl;
        std::cout
            << "softmax time: "
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
} // namespace softmax