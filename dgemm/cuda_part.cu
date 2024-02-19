#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <assert.h>
#include <mkl.h>

#include "headers.h"


extern "C" void CUBLAS_DGEMM_(const CBLAS_LAYOUT 	 layout,
                   const CBLAS_TRANSPOSE TRANSA,
                   const CBLAS_TRANSPOSE TRANSB,
                   const int 	              m,
                   const int 	              n,
                   const int 	              k,
                   const double           alpha,
                   const double              *A,
                   const int 	            lda,
                   const double              *B,
                   const int 	            ldb,
                   const double            beta,
                   double                    *C,
                   const int 	            ldc)
{
	cublasHandle_t handle;
	cublasOperation_t transa, transb;
	cublasStatus_t cubSt;
	cudaError_t cuE;

	double *d_A, *d_B, *d_C;
	switch (TRANSA)
	{
		case CblasNoTrans:   transa = CUBLAS_OP_N; break;
		case CblasTrans:     transa = CUBLAS_OP_T; break;
		case CblasConjTrans: transa = CUBLAS_OP_C; break;
		default: printf("cublas_dgemm: Fatal Error. TRANSA not recognized.\n"); exit(0);
	}
	
	switch (TRANSB)
	{
		case CblasNoTrans:   transb = CUBLAS_OP_N; break;
		case CblasTrans:     transb = CUBLAS_OP_T; break;
		case CblasConjTrans: transb = CUBLAS_OP_C; break;
		default: printf("cublas_dgemm: Fatal Error. TRANSB not recognized.\n"); exit(0);
	}

	while ((cuE = cudaMalloc((void **) &d_A, sizeof(double) * m * k)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
	while ((cuE = cudaMalloc((void **) &d_B, sizeof(double) * k * n)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
	while ((cuE = cudaMalloc((void **) &d_C, sizeof(double) * m * n)) != cudaSuccess) continue; assert(cudaSuccess == cuE);

	while ((cubSt = cublasCreate(&handle)) != CUBLAS_STATUS_SUCCESS) continue; assert(CUBLAS_STATUS_SUCCESS == cubSt);
	while ((cubSt = cublasSetMatrix(m, k, sizeof(*A), A, m, d_A, m)) != CUBLAS_STATUS_SUCCESS) continue; assert(CUBLAS_STATUS_SUCCESS == cubSt);
	while ((cubSt = cublasSetMatrix(k, n, sizeof(*B), B, k, d_B, k)) != CUBLAS_STATUS_SUCCESS) continue; assert(CUBLAS_STATUS_SUCCESS == cubSt);
	
	while ((cubSt = cublasDgemm(handle, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc)) != CUBLAS_STATUS_SUCCESS) continue;
	assert(CUBLAS_STATUS_SUCCESS == cubSt);

	while ((cubSt = cublasGetMatrix(m, n, sizeof(*C), d_C, m, C, m)) != CUBLAS_STATUS_SUCCESS) continue; assert(CUBLAS_STATUS_SUCCESS == cubSt);

	while ((cuE = cudaFree(d_A)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
	while ((cuE = cudaFree(d_B)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
	while ((cuE = cudaFree(d_C)) != cudaSuccess) continue; assert(cudaSuccess == cuE);
	
	while ((cubSt = cublasDestroy(handle)) != CUBLAS_STATUS_SUCCESS) continue; assert(CUBLAS_STATUS_SUCCESS == cubSt);
 	while ((cuE = cudaDeviceSynchronize()) != cudaSuccess) continue; assert(cudaSuccess == cuE);
}