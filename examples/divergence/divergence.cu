#include <iostream>
#include <cuda_runtime.h>
#include <bitset>

#include "test_divergence.cuh"

int main() {
    uint32_t *d_out;
    cudaMalloc(&d_out, 32 * sizeof(uint32_t));
    
    get_active_thread<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    
    uint32_t *h_out = new uint32_t[32];
    
    cudaMemcpy(h_out, d_out, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // convert to 32-bit binary string
    for (int i = 0; i < 32; i++) {
        std::cout << std::bitset<32>(h_out[i]).to_string() << std::endl;
    }
    
    delete[] h_out;
    cudaFree(d_out);
    
    return 0;
}