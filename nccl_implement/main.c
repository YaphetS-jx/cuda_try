#include <stdio.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

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

static inline int block_decompose(const int n, const int p, const int rank)
{
    return n / p + ((rank < n % p) ? 1 : 0);
}

void test_cartesian(int DMnx, int DMny, int DMnz, int reps, MPI_Comm cart_comm, cudaStream_t stream, ncclComm_t comm);

void test_cartesian_nonorth(int DMnx, int DMny, int DMnz, int reps, MPI_Comm dist_graph_comm, cudaStream_t stream, ncclComm_t comm);

void create_dist_graph(MPI_Comm cart_comm, MPI_Comm *dist_graph_comm);


int main(int argc, char* argv[])
{
    //initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));

    int rank, size;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    time_t t;
    srand((unsigned) time(&t));
    
    int Nx = rand()%1000+12;
    int Ny = rand()%1000+12;
    int Nz = rand()%1000+12;
    int reps = 10;
    int periods[3] = {rand()%2, rand()%2, rand()%2};
    // int periods[3] = {1, 1, 1};

    if (argc == 4) {
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
        Nz = atoi(argv[3]);
    } else {
        MPI_Bcast(&Nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&Ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&Nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(periods, 3, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (!rank) printf("DM grid %d %d %d, period %d %d %d\n", Nx, Ny, Nz, periods[0], periods[1], periods[2]);
    ////////////////////////////////////////////////////////////////////////
    // creating 3D dims for cartesian topology 
    int dims[3] = {0};
    MPI_Dims_create(size, 3, dims);
    if (!rank) {
        printf("dims ");
        for (int i = 0; i < 3; i++)
            printf("%d  ", dims[i]);
        printf("\n");
    }

    // Create a communicator with a cartesian topology.
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);

    int rank_cart, coords[3];
    MPI_Comm_rank(cart_comm, &rank_cart);
    MPI_Cart_coords(cart_comm, rank_cart, 3, coords);
    int DMnx = block_decompose(Nx, dims[0], coords[0]);
    int DMny = block_decompose(Ny, dims[1], coords[1]);
    int DMnz = block_decompose(Nz, dims[2], coords[2]);

    // create Distributed graph
    MPI_Comm dist_graph_comm;
    create_dist_graph(cart_comm, &dist_graph_comm);

    ////////////////////////////////////////////////////////////////////////

    int numGPUs; // per node
	cudaError_t cuE;
	while ((cuE = cudaGetDeviceCount(&numGPUs)) != cudaSuccess) continue;
	assert(cuE == cudaSuccess);

    int numHost = num_hosts();
    // check if #GPUs == #CPUs
    if (size != numGPUs * numHost) {
        if (!rank) printf("# of CPU threads: %d, # of devices: %d. Please use CPU:GPU = 1.\n", size, numGPUs*numHost);
        exit(EXIT_FAILURE);
    }

    int myGPU = rank % numGPUs;
    cuE = cudaSetDevice(myGPU);
	assert(cudaSuccess == cuE);

    cudaStream_t stream;
    ncclComm_t comm;
    create_NCCL_comm(cart_comm, &stream, &comm);
    ////////////////////////////////////////////////////////////////////////

    // tests 
    test_cartesian(DMnx, DMny, DMnz, reps, cart_comm, stream, comm);

    test_cartesian_nonorth(DMnx, DMny, DMnz, reps, dist_graph_comm, stream, comm);
    
    //finalizing MPI
    MPICHECK(MPI_Finalize());
    //finalizing nlcc
    free_NCCL_comm(&comm);

    return 0;
}


void test_cartesian(int DMnx, int DMny, int DMnz, int reps, MPI_Comm cart_comm, cudaStream_t stream, ncclComm_t comm)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int neighbor[6];
    find_neighbor(cart_comm, neighbor);

    int len = 2 * (DMnx*DMny + DMny*DMnz + DMnx*DMnz);
    
    double *send, *recv, *recv_gpu;
    send = (double *) malloc(sizeof(double) * len);
    for (int i = 0; i < len; i++) send[i] = drand48();
    recv = (double *) malloc(sizeof(double) * len);
    for (int i = 0; i < len; i++) recv[i] = -1.;
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

    cudaError_t cuE;

    double *d_send, *d_recv; 
    while ((cuE = cudaMalloc((void **) &d_send, sizeof(double) * len)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaMalloc((void **) &d_recv, sizeof(double) * len)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaMemcpy(d_send, send, sizeof(double) * len, cudaMemcpyHostToDevice)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaMemcpy(d_recv, recv, sizeof(double) * len, cudaMemcpyHostToDevice)) != cudaSuccess) continue; assert(cudaSuccess == cuE);

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
    if (!rank) printf("MPI %d times communication time %.3f ms\n", reps, (t2-t1)*1e3);

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for (int i = 0; i < reps; i++) {
        NLCC_Neighbor_alltoallv(d_send, sendcounts, sdispls, 
                        d_recv, recvcounts, rdispls, neighbor, comm, stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    if (!rank) printf("NLCC %d times communication time %.3f ms\n", reps, (t2-t1)*1e3);

    double err = 0; 
    for (int i = 0; i < len; i++) {
        err += (recv[i] - recv_gpu[i]);
    }
    MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    if (!rank) printf("orth test err %e\n", err);
    // assert(err < 1E-16);

    while ((cuE = cudaFree(d_send)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaFree(d_recv)) != cudaSuccess) continue; assert(cudaSuccess == cuE);

    free(send);
    free(recv);
    free(recv_gpu);
}


void test_cartesian_nonorth(int DMnx, int DMny, int DMnz, int reps, MPI_Comm dist_graph_comm, cudaStream_t stream, ncclComm_t comm)
{
    if (dist_graph_comm == MPI_COMM_NULL) return;
    int rank, size;
    MPI_Comm_rank(dist_graph_comm, &rank);
    MPI_Comm_size(dist_graph_comm, &size);

    int neighbor[26];
    find_neighbor_dist_graph(dist_graph_comm, neighbor);

    int len = 2 * (4 + 2 * (DMnx + DMny + DMnz) + (DMnx * DMny + DMnx * DMnz + DMny * DMnz) );
    
    double *send, *recv, *recv_gpu;
    send = (double *) malloc(sizeof(double) * len);
    for (int i = 0; i < len; i++) send[i] = drand48();
    recv = (double *) malloc(sizeof(double) * len);
    for (int i = 0; i < len; i++) recv[i] = -1.;
    recv_gpu = (double *) malloc(sizeof(double) * len);
    
    int sendcounts[26], sdispls[26], recvcounts[26], rdispls[26];
    // set up parameters for MPI_Ineighbor_alltoallv
    sendcounts[0] = sendcounts[2] = sendcounts[6] = sendcounts[8] = sendcounts[17] = sendcounts[19] = sendcounts[23] = sendcounts[25] = 1;
    sendcounts[1] = sendcounts[7] = sendcounts[18] = sendcounts[24] = DMnx;
    sendcounts[3] = sendcounts[5] = sendcounts[20] = sendcounts[22] = DMny;
    sendcounts[4] = sendcounts[21] = DMnx * DMny;
    sendcounts[9] = sendcounts[11] = sendcounts[14] = sendcounts[16] = DMnz;
    sendcounts[10] = sendcounts[15] = DMnx * DMnz;
    sendcounts[12] = sendcounts[13] = DMny * DMnz;
    
    for(int i = 0; i < 26; i++){
        recvcounts[i] = sendcounts[i];
    }    
    
    rdispls[0] = sdispls[0] = 0;
    for(int i = 1; i < 26;i++){
        rdispls[i] = sdispls[i] = sdispls[i-1] + sendcounts[i-1]; 
    }

    cudaError_t cuE;

    double *d_send, *d_recv; 
    while ((cuE = cudaMalloc((void **) &d_send, sizeof(double) * len)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaMalloc((void **) &d_recv, sizeof(double) * len)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaMemcpy(d_send, send, sizeof(double) * len, cudaMemcpyHostToDevice)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaMemcpy(d_recv, recv, sizeof(double) * len, cudaMemcpyHostToDevice)) != cudaSuccess) continue; assert(cudaSuccess == cuE);

    double t1, t2;

    MPI_Barrier(dist_graph_comm);
    // t1 = MPI_Wtime();
    NLCC_Neighbor_alltoallv_dist_graph(d_send, sendcounts, sdispls, 
                    d_recv, recvcounts, rdispls, neighbor, comm, stream);
    MPI_Barrier(dist_graph_comm);
    // t2 = MPI_Wtime();
    // if (!rank) printf("NLCC communication time %.3f ms\n", (t2-t1)*1e3);

    while ((cuE = cudaMemcpy(recv_gpu, d_recv, sizeof(double) * len, cudaMemcpyDeviceToHost)) != cudaSuccess) continue; assert(cudaSuccess == cuE);

    MPI_Barrier(dist_graph_comm);
    t1 = MPI_Wtime();
    for (int i = 0; i < reps; i++) {
        MPI_Neighbor_alltoallv(send, sendcounts, sdispls, MPI_DOUBLE, 
                                    recv, recvcounts, rdispls, MPI_DOUBLE, dist_graph_comm);
    }

    MPI_Barrier(dist_graph_comm);
    t2 = MPI_Wtime();
    if (!rank) printf("MPI %d times communication time %.3f ms\n", reps, (t2-t1)*1e3);

    MPI_Barrier(dist_graph_comm);
    t1 = MPI_Wtime();
    for (int i = 0; i < reps; i++) {
        NLCC_Neighbor_alltoallv_dist_graph(d_send, sendcounts, sdispls, 
                    d_recv, recvcounts, rdispls, neighbor, comm, stream);
    }
    MPI_Barrier(dist_graph_comm);
    t2 = MPI_Wtime();
    if (!rank) printf("NLCC %d times communication time %.3f ms\n", reps, (t2-t1)*1e3);

    double err = 0; 
    for (int i = 0; i < len; i++) {
        err += (recv[i] - recv_gpu[i]);
    }
    MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_SUM, dist_graph_comm);
    if (!rank) printf("nonorth test err %e\n", err);

    while ((cuE = cudaFree(d_send)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaFree(d_recv)) != cudaSuccess) continue; assert(cudaSuccess == cuE);

    free(send);
    free(recv);
    free(recv_gpu);
}


void create_dist_graph(MPI_Comm cart_comm, MPI_Comm *dist_graph_comm)
{
    int rank, size;
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Comm_size(cart_comm, &size);

    int dims[3], periods[3], coord_dmcomm[3];
    MPI_Cart_get(cart_comm, 3, dims, periods, coord_dmcomm);

    int ncoords[3];
    int nnproc = 1; // No. of neighboring processors in each direction
    int nneighb = 26, rank_chk;
    int neighb[26];
    int count = 0;

    for(int k = -nnproc; k <= nnproc; k++){
        for(int j = -nnproc; j <= nnproc; j++){
            for(int i = -nnproc; i <= nnproc; i++){
                int tmp = 0;
                if(i == 0 && j == 0 && k == 0){
                    continue;
                } else{
                    ncoords[0] = coord_dmcomm[0] + i;
                    ncoords[1] = coord_dmcomm[1] + j;
                    ncoords[2] = coord_dmcomm[2] + k;
                    for(int dir = 0; dir < 3; dir++){
                        if(periods[dir]){
                            if(ncoords[dir] < 0)
                                ncoords[dir] += dims[dir];
                            else if(ncoords[dir] >= dims[dir])
                                ncoords[dir] -= dims[dir];
                            tmp = 1;
                        } else{
                            if(ncoords[dir] < 0 || ncoords[dir] >= dims[dir]){
                                rank_chk = MPI_PROC_NULL;
                                tmp = 0;
                                break;
                            }
                            else
                                tmp = 1;
                        }
                    }
                    //TODO: For dirchlet give rank = MPI_PROC_NULL for out of bounds coordinates
                    if(tmp == 1)
                        MPI_Cart_rank(cart_comm,ncoords,&rank_chk); // proc rank corresponding to ncoords_mapped

                    neighb[count] = rank_chk;
                    count++;
                }
            }
        }
    }
    MPI_Dist_graph_create_adjacent(cart_comm,nneighb,neighb,(int *)MPI_UNWEIGHTED,nneighb,neighb,(int *)MPI_UNWEIGHTED,MPI_INFO_NULL,0,dist_graph_comm); // creates a distributed graph topology (adjacent, cartesian cubical)
}