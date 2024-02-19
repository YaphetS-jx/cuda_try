#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <assert.h>
#include "simpleC.h"
#include "simpleCUDA.h"


// Error handling macro
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        printf("CUDA error code %d\n", err); \
        MPI_Abort(MPI_COMM_WORLD, err);}


int getMyGPU(void)
{
    int mpiRank, numCPUs;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numCPUs);
    int numGPUs;
    cudaError_t cErr;

    while ((cErr = cudaGetDeviceCount(&numGPUs)) != cudaSuccess) continue;
    assert(cErr == cudaSuccess);
    // if (!mpiRank) printf("number of CPU %d, number of GPU %d\n", numCPUs, numGPUs);

    return mpiRank % numGPUs;
}

// Device code
// Very simple GPU Kernel that computes square roots of input numbers
__global__ void simpleMPIKernel(double *input, double *output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    output[tid] = sqrt(input[tid]);
}


// CUDA computation on each node
// No MPI here, only CUDA
void computeGPU(double *hostData, int blockSize, int gridSize)
{
    int myGPU = getMyGPU();
	cudaError_t cuE = cudaSetDevice(myGPU);
	assert(cudaSuccess == cuE);

    int dataSize = blockSize * gridSize;

    // Allocate data on GPU memory
    double *deviceInputData = NULL;
    CUDA_CHECK(cudaMalloc((void **)&deviceInputData, dataSize * sizeof(double)));

    double *deviceOutputData = NULL;
    CUDA_CHECK(cudaMalloc((void **)&deviceOutputData, dataSize * sizeof(double)));

    // Copy to GPU memory
    CUDA_CHECK(cudaMemcpy(deviceInputData, hostData, dataSize * sizeof(double), cudaMemcpyHostToDevice));

    // Run kernel
    simpleMPIKernel<<<gridSize, blockSize>>>(deviceInputData, deviceOutputData);

    // Copy data back to CPU memory
    CUDA_CHECK(cudaMemcpy(hostData, deviceOutputData, dataSize *sizeof(double), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(deviceInputData));
    CUDA_CHECK(cudaFree(deviceOutputData));
}