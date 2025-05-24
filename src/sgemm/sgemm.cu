#include <cstring>
#include "sgemm_v1.cuh"
#include "sgemm_v2.cuh"
#include "sgemm_v2_bitop.cuh"
#include "sgemm_v2_naive.cuh"
#include "sgemm_v3.cuh"
#include "test_sgemm.cuh"
#include "utils.cuh"

void print_usage() {
    std::cout << "Usage: ./sgemm [--bench] [--times N]" << std::endl;
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

    CHECK_TEST(test_sgemm_512x512_x512(sgemm::sgemm_v1, bench, times));
    CHECK_TEST(test_sgemm_512x512_x512(sgemm::sgemm_v2, bench, times));
    CHECK_TEST(test_sgemm_512x512_x512(sgemm::sgemm_v2_naive, bench, times));
    CHECK_TEST(test_sgemm_512x512_x512(sgemm::sgemm_v2_bitop, bench, times));
    CHECK_TEST(test_sgemm_512x512_x512(sgemm::sgemm_v3, bench, times));
    return 0;
}
