#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <math.h>

#include "simpleCUDA.h"
#include "simpleC.h"

int main(int argc, char *argv[])
{
    int reps = 100;

    // Dimensions of the dataset
    int blockSize = 256;
    int gridSize = 10000;
    int dataSizePerNode = gridSize * blockSize;

    // Initialize MPI state
    MPI_Init(&argc, &argv);

    // Get our MPI node number and node count
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Generate some random numbers on the root node (node 0)
    int dataSizeTotal = dataSizePerNode * size;
    double *dataRoot = NULL;

    if (rank == 0)  // Are we the root node?
    {
        dataRoot = (double*) malloc(sizeof(double) * dataSizeTotal);
        initData(dataRoot, dataSizeTotal);
    }

    // Allocate a buffer on each node
    double *dataNode = (double *) malloc(sizeof(double) * dataSizePerNode);

    // Dispatch a portion of the input data to each node
    MPI_Scatter(dataRoot, dataSizePerNode, MPI_DOUBLE, dataNode,
                dataSizePerNode, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) free(dataRoot);    

    double *dataNode_CPU = (double *) malloc(sizeof(double) * dataSizePerNode);
    memcpy(dataNode_CPU, dataNode, sizeof(double) * dataSizePerNode);

    double t1, t2;
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    // compute on CPU
    for (int rep = 0; rep < reps; rep++) {
        simpleMPI(dataNode_CPU, dataNode_CPU, dataSizePerNode);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    if (rank == 0) printf("CPU time %.3e ms\n", (t2-t1)*1e3);

    MPI_Barrier(MPI_COMM_WORLD);
    
    // On each node, run computation on GPU
    for (int rep = 0; rep < reps; rep++) {
        computeGPU(dataNode, blockSize, gridSize);
    }
    t1 = MPI_Wtime();
    if (rank == 0) printf("GPU time %.3e ms\n", (t1-t2)*1e3);

    // Reduction to the root node, computing the sum of output elements
    double sumNode = sum(dataNode, dataSizePerNode);
    double sumNode_CPU = sum(dataNode_CPU, dataSizePerNode);
    double sumRoot, sumRoot_CPU;

    // printf("rank %d, sum %.6f, %.6f\n", rank, sumNode_CPU, sumNode);

    MPI_Reduce(&sumNode, &sumRoot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sumNode_CPU, &sumRoot_CPU, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("GPU vs CPU err = %.3e\n", fabs(sumRoot-sumRoot_CPU));

    // Cleanup
    free(dataNode);
    free(dataNode_CPU);
    MPI_Finalize();
    return 0;
}