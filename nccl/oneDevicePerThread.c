#include <stdio.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>


int main(int argc, char* argv[])
{
    int size = 1;
    int myRank, nRanks, localRank = 0;


    //initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    int numGPUs;
    CUDACHECK(cudaGetDeviceCount(&numGPUs));
    if (!myRank) printf("# cores %d, #GPUs %d\n", nRanks, numGPUs);

    //calculating localRank based on hostname which is used in selecting a GPU
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);

    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p=0; p<nRanks; p++) {
        if (p == myRank) break;
        if (hostHashs[p] == hostHashs[myRank]) localRank++;
    }

    ncclUniqueId id;
    ncclComm_t comm;
    float *sendbuff, *recvbuff;
    cudaStream_t s;


    //get NCCL unique ID at rank 0 and broadcast it to all others
    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    printf("rank %d, Hostname: %s, localRank %d, id %d\n", myRank, hostname, localRank, id);

    float *send_h = (float *) malloc(sizeof(float) *size);
    float *recv_h = (float *) malloc(sizeof(float) *size);
    for (int i = 0; i < size; i++) send_h[i] = myRank;

    //picking a GPU based on localRank, allocate device buffers
    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaMalloc((void*)&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc((void*)&recvbuff, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));

    CUDACHECK(cudaMemcpy(sendbuff,send_h,size * sizeof(float),cudaMemcpyHostToDevice));

    //initializing NCCL
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));


    //communicating using NCCL
    NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum, comm, s));


    //completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));

    CUDACHECK(cudaMemcpy(recv_h,recvbuff,size * sizeof(float),cudaMemcpyDeviceToHost));

    printf("send %f, result %f\n", send_h[0], recv_h[0]);

    //free device buffers
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));


    //finalizing NCCL
    ncclCommDestroy(comm);


    //finalizing MPI
    MPICHECK(MPI_Finalize());


    // printf("[MPI Rank %d] Success \n", myRank);
    return 0;
}