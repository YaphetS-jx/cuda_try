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


#ifdef __cplusplus
extern "C" {
#endif

void create_NCCL_comm(MPI_Comm cart_comm, cudaStream_t *stream, ncclComm_t *comm);

void find_neighbor(MPI_Comm cart_comm, int *neighbor);

void find_neighbor_dist_graph(MPI_Comm dist_graph_comm, int *neighbor);

void free_NCCL_comm(ncclComm_t *comm);

void NLCC_Neighbor_alltoallv(double *d_send, int *sendcounts, int *sdispls, 
                        double *d_recv, int *recvcounts, int *rdispls, int *neighbor, ncclComm_t comm, cudaStream_t stream);

void NLCC_Neighbor_alltoallv_dist_graph(double *d_send, int *sendcounts, int *sdispls,
                        double *d_recv, int *recvcounts, int *rdispls, int *neighbor, ncclComm_t comm, cudaStream_t stream);

__global__ void change_value(double* vector);

__global__ void print_vector(double *vec, int len);

void print_vector_gpu(double *vec, int len);

#ifdef __cplusplus
}
#endif