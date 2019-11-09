#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <matmul.h>

#include <likwid-gpumarker.h>


int main(int argc, char* argv[])
{
    int count = 0;
    uint64_t *run1 = NULL;
    uint64_t *run2 = NULL;
    int err = 0;
    int gid = 0;
    LIKWID_GPUMARKER_INIT;

    LIKWID_GPUMARKER_START("matmul");
    matmul(0, 3200);
    //matmul(2, 3200);
    LIKWID_GPUMARKER_STOP("matmul");

    LIKWID_GPUMARKER_CLOSE;

    return 0;
}
