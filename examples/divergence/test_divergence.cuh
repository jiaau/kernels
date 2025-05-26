#pragma once

__global__ void get_active_thread(uint32_t *out) {
    if (threadIdx.x % 2 == 0) {
        out[threadIdx.x] = __activemask();
    } else {
        out[threadIdx.x] = __activemask();
    }
}