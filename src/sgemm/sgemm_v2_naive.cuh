#pragma once

#include "utils.cuh"

namespace sgemm {
__global__ void sgemm_v2_naive(float *__restrict__ a,
                               float *__restrict__ b,
                               float *__restrict__ c,
                               const int M,
                               const int N,
                               const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BK][BM];
    __shared__ float s_b[BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        if (tid % 2) {
            if (1 <= tid && tid <= 31) {
                load_a_smem_m = (tid + 32) >> 1;
            } else if (33 <= tid && tid <= 63) {
                load_a_smem_m = (tid - 32) >> 1;
            } else if (65 <= tid && tid <= 95) {
                load_a_smem_m = (tid + 32) >> 1;
            } else if (97 <= tid && tid <= 127) {
                load_a_smem_m = (tid - 32) >> 1;
            } else if (129 <= tid && tid <= 159) {
                load_a_smem_m = (tid + 32) >> 1;
            } else if (161 <= tid && tid <= 191) {
                load_a_smem_m = (tid - 32) >> 1;
            } else if (193 <= tid && tid <= 223) {
                load_a_smem_m = (tid + 32) >> 1;
            } else if (225 <= tid && tid <= 255) {
                load_a_smem_m = (tid - 32) >> 1;
            }
        }

        s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            if (tk < 4) {
                FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2]);
                FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);
            } else {
                int y1 = ty * TM / 2;
                int y2 = ty * TM / 2 + BM / 2;
                if (0 <= y1 && y1 < 16) {
                    y1 += 16;
                } else if (16 <= y1 && y1 < 32) {
                    y1 -= 16;
                } else if (32 <= y1 && y1 < 48) {
                    y1 += 16;
                } else if (48 <= y1 && y1 < 64) {
                    y1 -= 16;
                } else if (64 <= y1 && y1 < 80) {
                    y1 += 16;
                } else if (80 <= y1 && y1 < 96) {
                    y1 -= 16;
                } else if (96 <= y1 && y1 < 112) {
                    y1 += 16;
                } else if (112 <= y1 && y1 < 128) {
                    y1 -= 16;
                }
                if (0 <= y2 && y2 < 16) {
                    y2 += 16;
                } else if (16 <= y2 && y2 < 32) {
                    y2 -= 16;
                } else if (32 <= y2 && y2 < 48) {
                    y2 += 16;
                } else if (48 <= y2 && y2 < 64) {
                    y2 -= 16;
                } else if (64 <= y2 && y2 < 80) {
                    y2 += 16;
                } else if (80 <= y2 && y2 < 96) {
                    y2 -= 16;
                } else if (96 <= y2 && y2 < 112) {
                    y2 += 16;
                } else if (112 <= y2 && y2 < 128) {
                    y2 -= 16;
                }
                FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][y1]);
                FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][y2]);
            }
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}

} // namespace sgemm