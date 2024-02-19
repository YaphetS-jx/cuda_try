all: test

CC = icc
CFLAGS = -g -DMKL_ILP64 -mkl=sequential -std=gnu99 -O3 -fopenmp
NVCC = nvcc
CUFLAGS = -lcudart -lcublas


OBJS = main.o cuda_part.o

test: $(OBJS) 
	$(CC) $(CFLAGS) $(CUFLAGS) -o $@ $^ 
	
%.o: %.c
	$(CC) $(CFLAGS) -c $<

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $<

clean:
	rm -f *.o test
