mkdir -p $HOME/tmp
export CUDA_MPS_LOG_DIRECTORY=$HOME/tmp
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps

nvidia-cuda-mps-control -d

srun ./test

echo quit | nvidia-cuda-mps-control