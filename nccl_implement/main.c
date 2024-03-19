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
#include "tests.h"

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

    test_cartesian_Complex(DMnx, DMny, DMnz, reps, cart_comm, stream, comm);

    test_cartesian_nonorth(DMnx, DMny, DMnz, reps, dist_graph_comm, stream, comm);

    test_cartesian_nonorth_complex(DMnx, DMny, DMnz, reps, dist_graph_comm, stream, comm);
    
    //finalizing MPI
    MPICHECK(MPI_Finalize());
    //finalizing nlcc
    free_NCCL_comm(&comm);

    return 0;
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