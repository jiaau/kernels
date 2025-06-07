#include <cstring>
#include "reduce_max.cuh"
#include "reduce_sum.cuh"
#include "test_max.cuh"
#include "test_sum.cuh"
#include "utils.cuh"

void print_usage() {
    std::cout << "Usage: ./reduce [--bench] [--times N]" << std::endl;
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

    using namespace reduce;
    CHECK_TEST(test_sum(gpu_sum<float, 512>, bench, times));
    CHECK_TEST(test_max(gpu_max<float, 512>, bench, times));
    return 0;
}
