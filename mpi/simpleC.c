#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "simpleC.h"


void simpleMPI(double *input, double *output, int len)
{
    for (int i = 0; i < len; i++) {
        output[i] = sqrt(input[i]);
    }
}

// Initialize an array with random data (between 0 and 1)
void initData(double *data, int dataSize)
{
    for (int i = 0; i < dataSize; i++)
    {
        data[i] = (double)rand() / RAND_MAX;
    }
}


double sum(double *data, int size)
{
    double accum = 0;
    for (int i = 0; i < size; i++) {
        accum += data[i];
    }
    return accum;
}