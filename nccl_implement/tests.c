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
#include <complex.h>
#include <cuComplex.h>

#include "communication.h"
#include "tests.h"

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

void test_cartesian_Complex(int DMnx, int DMny, int DMnz, int reps, MPI_Comm cart_comm, cudaStream_t stream, ncclComm_t comm)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int neighbor[6];
    find_neighbor(cart_comm, neighbor);

    int len = 2 * (DMnx*DMny + DMny*DMnz + DMnx*DMnz);
    
    double _Complex *send, *recv, *recv_gpu;
    send = (double _Complex*) malloc(sizeof(double _Complex) * len);
    for (int i = 0; i < len; i++) send[i] = drand48() + I * drand48();
    recv = (double _Complex*) malloc(sizeof(double _Complex) * len);
    for (int i = 0; i < len; i++) recv[i] = -1.;
    recv_gpu = (double _Complex*) malloc(sizeof(double _Complex) * len);
    
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

    int d_sendcounts[6], d_sdispls[6];
    int d_recvcounts[6], d_rdispls[6];
    for (int i = 0; i < 6; i++) d_sendcounts[i] = 2*sendcounts[i];
    for (int i = 0; i < 6; i++) d_sdispls[i] = 2*sdispls[i];
    for (int i = 0; i < 6; i++) d_recvcounts[i] = 2*recvcounts[i];
    for (int i = 0; i < 6; i++) d_rdispls[i] = 2*rdispls[i];
    

    cudaError_t cuE;

    cuDoubleComplex *d_send, *d_recv; 
    while ((cuE = cudaMalloc((void **) &d_send, sizeof(cuDoubleComplex) * len)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaMalloc((void **) &d_recv, sizeof(cuDoubleComplex) * len)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaMemcpy(d_send, send, sizeof(cuDoubleComplex) * len, cudaMemcpyHostToDevice)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaMemcpy(d_recv, recv, sizeof(cuDoubleComplex) * len, cudaMemcpyHostToDevice)) != cudaSuccess) continue; assert(cudaSuccess == cuE);

    double t1, t2;

    MPI_Barrier(MPI_COMM_WORLD);
    // t1 = MPI_Wtime();
    NLCC_Neighbor_alltoallv((double *)d_send, d_sendcounts, d_sdispls, 
                    (double *)d_recv, d_recvcounts, d_rdispls, neighbor, comm, stream);
    MPI_Barrier(MPI_COMM_WORLD);
    // t2 = MPI_Wtime();
    // if (!rank) printf("NLCC communication time %.3f ms\n", (t2-t1)*1e3);

    while ((cuE = cudaMemcpy(recv_gpu, d_recv, sizeof(cuDoubleComplex) * len, cudaMemcpyDeviceToHost)) != cudaSuccess) continue; assert(cudaSuccess == cuE);

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for (int i = 0; i < reps; i++) {
        MPI_Neighbor_alltoallv(send, sendcounts, sdispls, MPI_DOUBLE_COMPLEX, 
                                    recv, recvcounts, rdispls, MPI_DOUBLE_COMPLEX, cart_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    if (!rank) printf("MPI %d times communication time %.3f ms\n", reps, (t2-t1)*1e3);

    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for (int i = 0; i < reps; i++) {
        NLCC_Neighbor_alltoallv((double *)d_send, d_sendcounts, d_sdispls, 
                    (double *)d_recv, d_recvcounts, d_rdispls, neighbor, comm, stream);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    if (!rank) printf("NLCC %d times communication time %.3f ms\n", reps, (t2-t1)*1e3);

    double err = 0; 
    for (int i = 0; i < len; i++) {
        err += fabs(creal(recv[i] - recv_gpu[i])) + fabs(cimag(recv[i] - recv_gpu[i]));
    }
    MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    if (!rank) printf("orth complex test err %e\n", err);
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


void test_cartesian_nonorth_complex(int DMnx, int DMny, int DMnz, int reps, MPI_Comm dist_graph_comm, cudaStream_t stream, ncclComm_t comm)
{
    if (dist_graph_comm == MPI_COMM_NULL) return;
    int rank, size;
    MPI_Comm_rank(dist_graph_comm, &rank);
    MPI_Comm_size(dist_graph_comm, &size);

    int neighbor[26];
    find_neighbor_dist_graph(dist_graph_comm, neighbor);

    int len = 2 * (4 + 2 * (DMnx + DMny + DMnz) + (DMnx * DMny + DMnx * DMnz + DMny * DMnz) );
    
    double _Complex *send, *recv, *recv_gpu;
    send = (double _Complex*) malloc(sizeof(double _Complex) * len);
    for (int i = 0; i < len; i++) send[i] = drand48() + I * drand48();
    recv = (double _Complex*) malloc(sizeof(double _Complex) * len);
    for (int i = 0; i < len; i++) recv[i] = -1.+ I * 1.;
    recv_gpu = (double _Complex*) malloc(sizeof(double _Complex) * len);
    
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

    int d_sendcounts[26], d_sdispls[26], d_recvcounts[26], d_rdispls[26];
    for (int i = 0; i < 26; i++) d_sendcounts[i] = 2*sendcounts[i];
    for (int i = 0; i < 26; i++) d_sdispls[i] = 2*sdispls[i];
    for (int i = 0; i < 26; i++) d_recvcounts[i] = 2*recvcounts[i];
    for (int i = 0; i < 26; i++) d_rdispls[i] = 2*rdispls[i];

    cudaError_t cuE;

    cuDoubleComplex *d_send, *d_recv; 
    while ((cuE = cudaMalloc((void **) &d_send, sizeof(cuDoubleComplex) * len)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaMalloc((void **) &d_recv, sizeof(cuDoubleComplex) * len)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaMemcpy(d_send, send, sizeof(cuDoubleComplex) * len, cudaMemcpyHostToDevice)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaMemcpy(d_recv, recv, sizeof(cuDoubleComplex) * len, cudaMemcpyHostToDevice)) != cudaSuccess) continue; assert(cudaSuccess == cuE);

    double t1, t2;

    MPI_Barrier(dist_graph_comm);
    // t1 = MPI_Wtime();
    NLCC_Neighbor_alltoallv_dist_graph((double *)d_send, d_sendcounts, d_sdispls, 
                    (double *)d_recv, d_recvcounts, d_rdispls, neighbor, comm, stream);
    MPI_Barrier(dist_graph_comm);
    // t2 = MPI_Wtime();
    // if (!rank) printf("NLCC communication time %.3f ms\n", (t2-t1)*1e3);

    while ((cuE = cudaMemcpy(recv_gpu, d_recv, sizeof(cuDoubleComplex) * len, cudaMemcpyDeviceToHost)) != cudaSuccess) continue; assert(cudaSuccess == cuE);

    MPI_Barrier(dist_graph_comm);
    t1 = MPI_Wtime();
    for (int i = 0; i < reps; i++) {
        MPI_Neighbor_alltoallv(send, sendcounts, sdispls, MPI_DOUBLE_COMPLEX, 
                                    recv, recvcounts, rdispls, MPI_DOUBLE_COMPLEX, dist_graph_comm);
    }

    MPI_Barrier(dist_graph_comm);
    t2 = MPI_Wtime();
    if (!rank) printf("MPI %d times communication time %.3f ms\n", reps, (t2-t1)*1e3);

    MPI_Barrier(dist_graph_comm);
    t1 = MPI_Wtime();
    for (int i = 0; i < reps; i++) {
        NLCC_Neighbor_alltoallv_dist_graph((double *)d_send, d_sendcounts, d_sdispls, 
                    (double *)d_recv, d_recvcounts, d_rdispls, neighbor, comm, stream);
    }
    MPI_Barrier(dist_graph_comm);
    t2 = MPI_Wtime();
    if (!rank) printf("NLCC %d times communication time %.3f ms\n", reps, (t2-t1)*1e3);

    double err = 0; 
    for (int i = 0; i < len; i++) {
        err += fabs(creal(recv[i] - recv_gpu[i])) + fabs(cimag(recv[i] - recv_gpu[i]));
    }
    MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_SUM, dist_graph_comm);
    if (!rank) printf("nonorth complex test err %e\n", err);

    while ((cuE = cudaFree(d_send)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
    while ((cuE = cudaFree(d_recv)) != cudaSuccess) continue; assert(cudaSuccess == cuE);

    free(send);
    free(recv);
    free(recv_gpu);
}