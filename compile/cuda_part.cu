// cuda_part.cu
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void cudaKernel() {
    printf("Hello from the GPU!\n");
}

extern "C" void runCudaPart() {
    printf("This is the GPU part of the code.\n");
    cudaKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
