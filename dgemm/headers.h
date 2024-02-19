#include <stdio.h>
#include <mkl.h>

void print_matrix(double *A, int nrow, int ncol, char ACC);

void rand_vec(double *vec, int len);

double find_err(double *A, double *B, int len);

#ifndef __cplusplus
void CUBLAS_DGEMM_(const CBLAS_LAYOUT 	 layout,
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
                   const int 	            ldc);

#endif // __cplusplus