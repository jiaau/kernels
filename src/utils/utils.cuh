#pragma once
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mma.h>

// #define DEBUG

#define HALF2(val) (*reinterpret_cast<half2 *>(&val))

#define OFFSET(m, n, ld) ((m) * (ld) + (n))
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

// vector load 128 bit, will be compiled to .128 instructions
__device__ __forceinline__ void ld_st_128bit(void *dst, void *src) {
    *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<float4 *>(src);
}

/**
 * \brief Fill data with random values
 * \tparam T data type, should be fp type
 */
template <typename T>
void fill_data(T *data, int n) {
    srand(time(0));
    for (int i = 0; i < n; i++) {
#ifdef DEBUG
        data[i] = (float)(i % 16);
#else
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
#endif
    }
}

/**
 * \brief Diff 2 arrays of type fp
 */
template <typename Atype, typename Btype>
bool diff(Atype a, Btype b, int n, float atol = 1e-2) {
    for (int i = 0; i < n; i++) {
        auto error = (float)a[i] - (float)b[i];
        // check abosulte error
        if (std::abs(error) > atol) {
            std::cout << " \033[1;31mDifference found at index " << i << ": a[" << i
                      << "] = " << (float)a[i] << ", b[" << i << "] = " << (float)b[i]
                      << "\033[0m\n";
            return true;
        }
    }
    return false;
}

#define CHECK_TEST(...)                                                                  \
    if (!(__VA_ARGS__)) {                                                                \
        std::cout << "\033[1;31m" << #__VA_ARGS__ << " failed\033[0m\n";                 \
        return 1;                                                                        \
    } else {                                                                             \
        std::cout << "\033[1;32m" << #__VA_ARGS__ << " passed\033[0m\n";                 \
    }

#define MAX_NUM_THREADS 1024

template <typename T>
inline T round_up_thread(T m) {
    T d = 32, limit = MAX_NUM_THREADS;
    T x = m > limit ? limit : m;
    return ((x + d - 1) / d) * d;
}