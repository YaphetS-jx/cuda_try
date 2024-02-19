#ifdef __cplusplus
extern "C" {
#endif
    int getMyGPU(void);
    void computeGPU(double *hostData, int blockSize, int gridSize);
    void my_abort(int err);
#ifdef __cplusplus
}
#endif