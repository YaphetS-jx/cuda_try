

void test_cartesian(int DMnx, int DMny, int DMnz, int reps, MPI_Comm cart_comm, cudaStream_t stream, ncclComm_t comm);

void test_cartesian_Complex(int DMnx, int DMny, int DMnz, int reps, MPI_Comm cart_comm, cudaStream_t stream, ncclComm_t comm);

void test_cartesian_nonorth(int DMnx, int DMny, int DMnz, int reps, MPI_Comm dist_graph_comm, cudaStream_t stream, ncclComm_t comm);

void test_cartesian_nonorth_complex(int DMnx, int DMny, int DMnz, int reps, MPI_Comm dist_graph_comm, cudaStream_t stream, ncclComm_t comm);