#include <stdio.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

#include "communication.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void change_value(double* vector) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        vector[0] = 1;
    }
}


__global__ void print_vector(double *vec, int len) {
    for (int i = 0; i < len; i++) {
        printf("%f\t", vec[i]);
    }
    printf("\n");
}


void print_vector_gpu(double *vec, int len) {
    print_vector<<<1,1>>>(vec, len);
}

// assuming already bind GPU to CPU with 1:1 ration 
void create_NCCL_comm(MPI_Comm cart_comm, int *neighbor, cudaStream_t *stream, ncclComm_t *comm)
{
    if (cart_comm == MPI_COMM_NULL) return;
    int rank, size;
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Comm_size(cart_comm, &size);
    
    ncclUniqueId id;
    
    //get NCCL unique ID at rank 0 and broadcast it to all others
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, cart_comm);

    //picking a GPU based on localRank, allocate device buffers
    CUDACHECK(cudaStreamCreate(stream));
    NCCLCHECK(ncclCommInitRank(comm, size, id, rank));

    int dims[3], periods[3], coords[3];
    MPI_Cart_get(cart_comm, 3, dims, periods, coords);

    for (int dir = 0; dir < 3; dir ++) {
        MPI_Cart_shift(cart_comm, dir, 1, neighbor + dir*2, neighbor + dir*2 + 1);
    }
}

void free_NCCL_comm(ncclComm_t *comm)
{
    NCCLCHECK(ncclCommDestroy(*comm));
}

void NLCC_Neighbor_alltoallv(double *d_send, int *sendcounts, int *sdispls, 
                        double *d_recv, int *recvcounts, int *rdispls, int *neighbor, ncclComm_t comm, cudaStream_t stream)
{
    NCCLCHECK(ncclGroupStart());
    for (int d = 0; d < 3; d++) {
        // -1 dir 
        ncclSend(d_send + sdispls[d*2], sendcounts[d*2], ncclDouble, neighbor[d*2], comm, stream);
        ncclRecv(d_recv + rdispls[d*2], recvcounts[d*2], ncclDouble, neighbor[d*2], comm, stream);
        // +1 dir 
        ncclSend(d_send + sdispls[d*2+1], sendcounts[d*2+1], ncclDouble, neighbor[d*2+1], comm, stream);
        ncclRecv(d_recv + rdispls[d*2+1], recvcounts[d*2+1], ncclDouble, neighbor[d*2+1], comm, stream);
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(cudaStreamSynchronize(stream));
}

#ifdef __cplusplus
}
#endif