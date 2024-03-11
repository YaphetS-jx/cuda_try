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
void create_NCCL_comm(MPI_Comm cart_comm, cudaStream_t *stream, ncclComm_t *comm)
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
}


void find_neighbor(MPI_Comm cart_comm, int *neighbor)
{
    if (cart_comm == MPI_COMM_NULL) return;
    int rank, size;
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Comm_size(cart_comm, &size);

    int dims[3], periods[3], coords[3];
    MPI_Cart_get(cart_comm, 3, dims, periods, coords);

    for (int dir = 0; dir < 3; dir ++) {
        MPI_Cart_shift(cart_comm, dir, 1, neighbor + dir*2, neighbor + dir*2 + 1);
    }

    // int neighbor2[6];
    // for (int dir = 0; dir < 3; dir ++) {
        // int coords_shift[3] = {coords[0], coords[1], coords[2]};
        // coords_shift[dir] = (coords[dir] + 1) % dims[dir];
        // MPI_Cart_rank(cart_comm, coords_shift, neighbor2 + dir*2 + 1);
        // coords_shift[dir] = (coords[dir] - 1 + dims[dir]) % dims[dir];
        // MPI_Cart_rank(cart_comm, coords_shift, neighbor2 + dir*2);
    // }
}

void find_neighbor_dist_graph(MPI_Comm dist_graph_comm, int *neighbor)
{
    if (dist_graph_comm == MPI_COMM_NULL) return;
    int rank, size;
    MPI_Comm_rank(dist_graph_comm, &rank);
    MPI_Comm_size(dist_graph_comm, &size);

    int srcw[26], destw[26], src[26], dest[26];
    MPI_Dist_graph_neighbors(dist_graph_comm, 26, src, srcw, 26, dest, destw);

    // printf("rank %d, src  ", rank);
    // for (int i = 0; i < 26; i++) printf("%d ", src[i]);
    // printf("\n");
    // printf("rank %d, dest ", rank);
    // for (int i = 0; i < 26; i++) printf("%d ", dest[i]);
    // printf("\n");
    for (int i = 0; i < 26; i++) neighbor[i] = src[i];
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
        if (neighbor[d*2] >= 0) {
            ncclSend(d_send + sdispls[d*2], sendcounts[d*2], ncclDouble, neighbor[d*2], comm, stream);
            ncclRecv(d_recv + rdispls[d*2], recvcounts[d*2], ncclDouble, neighbor[d*2], comm, stream);
        }
        // +1 dir 
        if (neighbor[d*2+1] >= 0) {
            ncclSend(d_send + sdispls[d*2+1], sendcounts[d*2+1], ncclDouble, neighbor[d*2+1], comm, stream);
            ncclRecv(d_recv + rdispls[d*2+1], recvcounts[d*2+1], ncclDouble, neighbor[d*2+1], comm, stream);
        }
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(cudaStreamSynchronize(stream));
}

void NLCC_Neighbor_alltoallv_dist_graph(double *d_send, int *sendcounts, int *sdispls,
                        double *d_recv, int *recvcounts, int *rdispls, int *neighbor, ncclComm_t comm, cudaStream_t stream)
{
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < 26; i++) {
        if (neighbor[i] >= 0) {
            ncclSend(d_send + sdispls[i], sendcounts[i], ncclDouble, neighbor[i], comm, stream);
            ncclRecv(d_recv + rdispls[i], recvcounts[i], ncclDouble, neighbor[i], comm, stream);
        }
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(cudaStreamSynchronize(stream));
}

#ifdef __cplusplus
}
#endif