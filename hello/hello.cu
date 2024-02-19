#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello World from GPU!\n");
}

int main() {
    // Launch a kernel on the GPU with one thread
    helloFromGPU<<<1, 1>>>();
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    return 0;
}

