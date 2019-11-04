#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <likwid-marker.h>

int main(int argc, char* argv[])
{
    int i, j;
    int size;
    int iters = 10;
    double* vector;
    if (argc != 2)
    {
        printf("Number of elements for vector must be given on command line.\n");
        return 1;
    }

    size = atoi(argv[1]);
    vector = (double*) malloc(size * sizeof(double));
    if (!vector)
        return 2;

    LIKWID_MARKER_INIT;

    LIKWID_MARKER_START("init");
    for (i=0;i<size;i++)
        vector[i] = 2.0;
    LIKWID_MARKER_STOP("init");
    printf("Initialized %d elements\n", size);


    LIKWID_MARKER_START("pow");
    for (j=0;j<iters;j++)
    {
        for (i=0;i<size;i++)
            vector[i] = vector[i] * vector[i];
    }
    LIKWID_MARKER_STOP("pow");
    printf("Calculated power of %d for all %d elements\n", iters*2, size);

    LIKWID_MARKER_CLOSE;

    free(vector);
    return 0;



}
