#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <time.h>
#include <assert.h>
#include <math.h>

#include "headers.h"




int main(int argc, char **argv)
{
    int M, N, K;
    M = N = K = 10;    
    if(argc == 4){
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    
    double *A = (double *) malloc(sizeof(double) * M*K);
    double *B = (double *) malloc(sizeof(double) * K*N);
    double *C = (double *) malloc(sizeof(double) * M*N);
    double *C_CUDA = (double *) malloc(sizeof(double) * M*N);
    rand_vec(A, M*K); rand_vec(B, K*N);

    double alpha = 1.0, beta = 0.0;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, M);

    CUBLAS_DGEMM_(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C_CUDA, M);

    // Print the result
    // print_matrix(A, M, K, 'H');
    // print_matrix(B, K, N, 'H');
    // print_matrix(C, M, N, 'H');
    // print_matrix(C_CUDA, M, N, 'H');
    printf("error between BLAS and CUBLAS %.2e\n", find_err(C, C_CUDA, M*N));

    free(A);
    free(B);
    free(C);
    free(C_CUDA);
    return 0;
}


void print_matrix(double *A, int nrow, int ncol, char ACC)
{
    assert(ACC == 'H' || ACC == 'L');
    printf("\n");
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            if (ACC == 'H')
                printf("%10.6f  ", A[i+j*nrow]);
            else
                printf("%7.3f  ", A[i+j*nrow]);
        }
        printf("\n");
    }
}


void rand_vec(double *vec, int len)
{
    for (int i = 0; i < len; i++) {
        vec[i] = drand48();
    }
}


double find_err(double *A, double *B, int len) 
{
    double err = 0;
    for (int i = 0; i < len; i++) {
        err += fabs(A[i] - B[i]);
    }
    return err;
}