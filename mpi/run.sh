export CUDA_MPS_PIPE_DIRECTORY=$(pwd)/mps/nvidia-mps # Set to the same location as the MPS control daemon
export CUDA_MPS_LOG_DIRECTORY=$(pwd)/mps/nvidia-log # Set to the same location as the MPS control daemon

nvidia-cuda-mps-control -d # Start the daemon.

srun ./test 

echo quit | nvidia-cuda-mps-control