CC=mpicc
CXX=nvcc

# Source files
C_SOURCES=$(wildcard *.c)
CU_SOURCES=$(wildcard *.cu)

CFLAGS = -std=gnu99 -O3 -fopenmp
CPPFLAGS = 
LDLIBS = -lrt -lm
OBJSC=$(C_SOURCES:.c=.o)

# -std=c++14 will fail the compilation
CXXFLAGS = -arch=sm_70 -ccbin=g++ -Xcompiler=-pthread

CUDAROOT = /usr/local/pace-apps/manual/packages/nvhpc/Linux_x86_64/22.11/cuda
INCLUDES += -I$(CUDAROOT)/include
LDFLAGS += -L$(CUDAROOT)/lib64
LDLIBS += -lcudart

CUBLASROOT = /usr/local/pace-apps/manual/packages/nvhpc/Linux_x86_64/22.11/math_libs/11.8/targets/x86_64-linux
INCLUDES += -I$(CUBLASROOT)/include
LDFLAGS += -L$(CUBLASROOT)/lib
LDLIBS += -lcublas -lcusolver

OBJSCU=$(CU_SOURCES:.cu=.o)

LIBBASE = ./test

all: test

# Note the implicit rule to compile '.c' files into '.o' files is
# %.o : %.c
# 	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

test: $(OBJSC) $(OBJSCU)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(LIBBASE) $^ $(LDLIBS)

%.o : %.cu
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -f  $(OBJSC) $(OBJSCU) $(LIBBASE)
