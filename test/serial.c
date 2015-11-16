#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <likwid.h>

int main(int argc, char* argv[])
{
    int i, j;
    int size;
    double* vector;
    if (argc != 2)
        return 1;

    size = atoi(argv[1]);
    vector = (double*) malloc(size * sizeof(double));
    if (!vector)
        return 2;

    LIKWID_MARKER_INIT;

    LIKWID_MARKER_START("init");
    for (i=0;i<size;i++)
        vector[i] = 2.0;
    LIKWID_MARKER_STOP("init");


    LIKWID_MARKER_START("pow");
    for (j=0;j<10;j++)
    {
        for (i=0;i<size;i++)
            vector[i] = vector[i] * vector[i];
    }
    LIKWID_MARKER_STOP("pow");

    LIKWID_MARKER_CLOSE;

    free(vector);
    return 0;



}
