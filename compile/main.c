// main.c
#include <stdio.h>

// Declaration of a function defined in CUDA code
void runCudaPart();

int main() {
    printf("This is the CPU part of the code.\n");
    
    // Call a function defined in the CUDA code
    runCudaPart();

    return 0;
}
