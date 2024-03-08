#include <stdio.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "communication.h"

#define MPICHECK(cmd) do {                          \
    int e = cmd;                                      \
    if( e != MPI_SUCCESS ) {                          \
        printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
        exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
        printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
        exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
    ncclResult_t r = cmd;                             \
    if (r!= ncclSuccess) {                            \
        printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
        exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)

static uint64_t getHostHash(const char* string) {
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++){
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}


static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i< maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

int num_hosts() {
#define length 1024 
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char *hostname = (char *) malloc(sizeof(char) * length * size);
    getHostName(hostname+rank*length, length);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostname, length, MPI_CHAR, MPI_COMM_WORLD);

    int unique_nodes = 1; // Start with 1 for the root node
    for (int i = length; i < size * length; i += length) {
        int j;
        for (j = 0; j < i; j += length) {
            if (strncmp(&hostname[i], &hostname[j], length) == 0) {
                break; // Found a matching hostname earlier in the list
            }
        }
        if (j == i) {
            unique_nodes++; // No match found, this is a unique node
        }
    }
    free(hostname);
    return unique_nodes;
#undef length
}


int main(int argc, char* argv[])
{
    int DMnx = 134;
    int DMny = 78;
    int DMnz = 232;
    int reps = 10;

    if (argc == 4) {
        DMnx = atoi(argv[1]);
        DMny = atoi(argv[2]);
        DMnz = atoi(argv[3]);
    }

    //initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));

    int rank, size;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    if (!rank) printf("DM grid %d %d %d\n", DMnx, DMny, DMnz);
    ////////////////////////////////////////////////////////////////////////
    // Ask MPI to decompose our processes in a 2D cartesian grid for us
    int dims[3] = {0};
    MPI_Dims_create(size, 3, dims);
    if (!rank) {
        printf("dims ");
        for (int i = 0; i < 3; i++)
            printf("%d  ", dims[i]);
        printf("\n");
    }

    // Make both dimensions non-periodic
    int periods[3];
    for (int i = 0; i < 3; i++) periods[i] = 1;

    // Create a communicator with a cartesian topology.
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);
    ////////////////////////////////////////////////////////////////////////

    int numGPUs; // per node
	cudaError_t cErr;
	while ((cErr = cudaGetDeviceCount(&numGPUs)) != cudaSuccess) continue;
	assert(cErr == cudaSuccess);

    int numHost = num_hosts();
    // check if #GPUs == #CPUs
    if (size != numGPUs * numHost) {
        if (!rank) printf("# of CPU threads: %d, # of devices: %d. Please use CPU:GPU = 1.\n", size, numGPUs*numHost);
        exit(EXIT_FAILURE);
    }

    int myGPU = rank % numGPUs;
    cudaError_t cuE;
    cuE = cudaSetDevice(myGPU);
	assert(cudaSuccess == cuE);

    int neighbor[6];
    cudaStream_t stream;
    ncclComm_t comm;
    create_NCCL_comm(cart_comm, neighbor, &stream, &comm);
    ////////////////////////////////////////////////////////////////////////

    int len = 2 * (DMnx*DMny + DMny*DMnz + DMnx*DMnz);
    
    double *d_send, *d_recv; 
    double *send, *recv, *recv_gpu;
    send = (double *) malloc(sizeof(double) * len);
    for (int i = 0; i < len; i++) send[i] = drand48();
    recv = (double *) malloc(sizeof(double) * len);
    recv_gpu = (double *) malloc(sizeof(double) * len);
    
    int sendcounts[6], sdispls[6];
    int recvcounts[6], rdispls[6];
    // set up parameters for MPI_Ineighbor_alltoallv
    // TODO: do this in Initialization to save computation time!
    sendcounts[0] = sendcounts[1] = recvcounts[0] = recvcounts[1] = DMny * DMnz;
    sendcounts[2] = sendcounts[3] = recvcounts[2] = recvcounts[3] = DMnx * DMnz;
    sendcounts[4] = sendcounts[5] = recvcounts[4] = recvcounts[5] = DMnx * DMny;
    
    rdispls[0] = sdispls[0] = 0;
    rdispls[1] = sdispls[1] = sdispls[0] + sendcounts[0];
    rdispls[2] = sdispls[2] = sdispls[1] + sendcounts[1];
    rdispls[3] = sdispls[3] = sdispls[2] + sendcounts[2];
    rdispls[4] = sdispls[4] = sdispls[3] + sendcounts[3];
    rdispls[5] = sdispls[5] = sdispls[4] + sendcounts[4];

    while ((cuE = cudaMalloc((void **) &d_send, sizeof(double) * len)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaMalloc((void **) &d_recv, sizeof(double) * len)) != cudaSuccess) continue; assert(cudaSuccess == cuE);

    while ((cuE = cudaMemcpy(d_send, send, sizeof(double) * len, cudaMemcpyHostToDevice)) != cudaSuccess) continue; assert(cudaSuccess == cuE);

    double t1, t2;

    MPI_Barrier(MPI_COMM_WORLD);
    // t1 = MPI_Wtime();
    NLCC_Neighbor_alltoallv(d_send, sendcounts, sdispls, 
                    d_recv, recvcounts, rdispls, neighbor, comm, stream);
    MPI_Barrier(MPI_COMM_WORLD);
    // t2 = MPI_Wtime();
    // if (!rank) printf("NLCC communication time %.3f ms\n", (t2-t1)*1e3);

    while ((cuE = cudaMemcpy(recv_gpu, d_recv, sizeof(double) * len, cudaMemcpyDeviceToHost)) != cudaSuccess) continue; assert(cudaSuccess == cuE);

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for (int i = 0; i < reps; i++) {
        MPI_Neighbor_alltoallv(send, sendcounts, sdispls, MPI_DOUBLE, 
                                    recv, recvcounts, rdispls, MPI_DOUBLE, cart_comm);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    if (!rank) printf("MPI  communication time %.3f ms\n", (t2-t1)*1e3);

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for (int i = 0; i < reps; i++) {
        NLCC_Neighbor_alltoallv(d_send, sendcounts, sdispls, 
                        d_recv, recvcounts, rdispls, neighbor, comm, stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    if (!rank) printf("NLCC communication time %.3f ms\n", (t2-t1)*1e3);

    double err = 0; 
    for (int i = 0; i < len; i++) {
        err += (recv[i] - recv_gpu[i]);
    }
    MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    assert(err < 1E-16);

    while ((cuE = cudaFree(d_send)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaFree(d_recv)) != cudaSuccess) continue; assert(cudaSuccess == cuE);

    free(send);
    free(recv);
    free(recv_gpu);
    
    //finalizing MPI
    MPICHECK(MPI_Finalize());
    //finalizing nlcc
    free_NCCL_comm(&comm);

    return 0;
}



