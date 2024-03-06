# Makefile for compiling CUDA and NCCL program

# Specify the compiler
CC = mpicc

# Specify the flags for the compiler
CCFLAGS = -std=gnu99 -O3

# Include directories and path 
CUDAROOT = /usr/local/pace-apps/manual/packages/nvhpc/Linux_x86_64/22.11/cuda
INCLUDES += -I$(CUDAROOT)/include
LDFLAGS += -L$(CUDAROOT)/lib64
LIBS += -lcudart

NCCLROOT = /usr/local/pace-apps/manual/packages/nvhpc/Linux_x86_64/22.11/comm_libs/nccl/
INCLUDES += -I$(NCCLROOT)/include
LDFLAGS += -L$(NCCLROOT)/lib
LIBS += -lnccl

# Source files
# SOURCES = oneProcessMultipleDevice.c
# SOURCES = oneDevicePerThread.c
SOURCES = multipleDevicePerThread.c

# Object files
OBJECTS = $(SOURCES:.c=.o)

# Executable name
all: test

test: $(SOURCES)
	$(CC) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -o test $(SOURCES) $(LIBS)

clean:
	rm -f $(OBJECTS) test