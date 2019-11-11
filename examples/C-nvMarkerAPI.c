#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>

#include <likwid-gpumarker.h>

extern int cuda_function(int gpu, size_t size);


int main(int argc, char* argv[])
{
    int i = 0;
    int numDevices = 1;

    LIKWID_GPUMARKER_INIT;

    LIKWID_GPUMARKER_START("matmul");
    // You can read the environment variable LIKWID_GPUS to determine list of GPUs
    for (i = 0; i < numDevices; i++)
        int err = cuda_function(0, 3200);

    LIKWID_GPUMARKER_STOP("matmul");

    LIKWID_GPUMARKER_CLOSE;

    return 0;
}
