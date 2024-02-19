CC=gcc
NVCC=nvcc
CFLAGS=-Wall -g
LDFLAGS=-lcudart

# Source files
C_SOURCES=$(wildcard *.c)
CU_SOURCES=$(wildcard *.cu)

# Object files
C_OBJS=$(C_SOURCES:.c=.o)
CU_OBJS=$(CU_SOURCES:.cu=.o)

# Executable
EXEC=app

all: $(EXEC)

$(EXEC): $(C_OBJS) $(CU_OBJS)
	$(CC) -o $(EXEC) $(C_OBJS) $(CU_OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

%.o: %.cu
	$(NVCC) -c $<

clean:
	rm -f $(EXEC) $(C_OBJS) $(CU_OBJS)
