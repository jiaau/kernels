#include <cstring>
#include "test_transpose.cuh"
#include "transpose_cuda_coalesced.cuh"
#include "transpose_cuda_conflict_free.cuh"
#include "transpose_cuda_naive.cuh"
#include "transpose_cute_coalesced.cuh"
#include "transpose_cute_conflict_free.cuh"
#include "transpose_cute_naive.cuh"
#include "transpose_cute_naive_colfax.cuh"
#include "utils.cuh"

void print_usage() {
    std::cout << "Usage: ./transpose [--bench] [--times N]" << std::endl;
    std::cout << "  --bench: Enable benchmark mode" << std::endl;
    std::cout << "  --times N: Number of benchmark iterations (default: 3)" << std::endl;
}

int main(int argc, char **argv) {
    bool bench = false;
    int times = 3;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) {
            bench = true;
        } else if (strcmp(argv[i], "--times") == 0 && i + 1 < argc) {
            times = atoi(argv[++i]);
            if (times <= 0) {
                std::cerr << "Error: times must be positive" << std::endl;
                print_usage();
                return 1;
            }
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            print_usage();
            return 1;
        }
    }

    using namespace transpose;
    // cuda

    // naive
    CHECK_TEST(test_transpose(transpose_naive<32, 8>, bench, times));
    CHECK_TEST(test_transpose(transpose_naive<32, 32>, bench, times));
    
    // coalesced
    CHECK_TEST(test_transpose(transpose_cuda_coalesced<32, 8>, bench, times));
    CHECK_TEST(test_transpose(transpose_cuda_coalesced<32, 32>, bench, times));

    // conflict free
    CHECK_TEST(test_transpose(transpose_cuda_conflict_free<32, 8>, bench, times));
    CHECK_TEST(test_transpose(transpose_cuda_conflict_free<32, 32>, bench, times));

    // cute

    // colfax
    CHECK_TEST(test_transpose(transpose_cute_naive_colfax<float, 32, 8>, bench, times));
    CHECK_TEST(test_transpose(transpose_cute_naive_colfax<float, 32, 32>, bench, times));

    // naive
    CHECK_TEST(test_transpose(transpose_cute_naive<float, 32, 8>, bench, times));
    CHECK_TEST(test_transpose(transpose_cute_naive<float, 32, 32>, bench, times));

    // coalesced
    CHECK_TEST(test_transpose(transpose_cute_coalesced<float, 32, 8>, bench, times));
    CHECK_TEST(test_transpose(transpose_cute_coalesced<float, 32, 32>, bench, times));

    // conflict free
    CHECK_TEST(test_transpose(transpose_cute_conflict_free<float, 32, 8>, bench, times));
    CHECK_TEST(test_transpose(transpose_cute_conflict_free<float, 32, 32>, bench, times));
    return 0;
}
