# Makefile for compiling CUDA and NCCL program

# Specify the compiler
CC = mpicc
NVCC = /usr/local/pace-apps/manual/packages/nvhpc/Linux_x86_64/22.11/cuda/bin/nvcc

# Specify the flags for the compiler
CFLAGS = -std=gnu99 -O3
CUFLAGS = -arch=sm_70 -ccbin=g++ -Xcompiler=-pthread

# Include directories and path 
CUDAROOT = /usr/local/pace-apps/manual/packages/nvhpc/Linux_x86_64/22.11/cuda
CPPFLAGS += -I$(CUDAROOT)/include
LDFLAGS += -L$(CUDAROOT)/lib64
LIBS += -lcudart

NCCLROOT = /usr/local/pace-apps/manual/packages/nvhpc/Linux_x86_64/22.11/comm_libs/nccl
CPPFLAGS += -I$(NCCLROOT)/include
LDFLAGS += -L$(NCCLROOT)/lib
LIBS += -lnccl

# Object files
OBJECTS = main.o communication.o

# Executable name
all: test

test: $(OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o test $^ $(LIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

%.o : %.cu
	$(NVCC) -c $(CUFLAGS) $(CPPFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) test