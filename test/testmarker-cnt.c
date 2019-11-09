#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include <likwid-marker.h>

#define SIZE 1000000

double sum = 0, a[SIZE], b[SIZE], c[SIZE];

int main(int argc, char* argv[])
{
    int i, j ;
    double alpha = 3.14;

    /* Initialize */
    for (i=0; i<SIZE; i++)
    {
        a[i] = 1.0/(double) i;
        b[i] = 1.0;
        c[i] = (double) i;
    }
    LIKWID_MARKER_INIT;

//    likwid_pinProcess(2);
    printf("Main running on core %d\n", likwid_getProcessorId());


/****************************************************/
#pragma omp parallel
    {
        LIKWID_MARKER_THREADINIT;
        char* label = malloc(40*sizeof(char));
        int threadId = omp_get_thread_num();
//        likwid_pinThread(threadId);
        printf("Thread running on core %d\n", likwid_getProcessorId());

        for (int counter=1; counter< 3; counter++)
        {
            sprintf(label,"plain-%d",counter);
#pragma omp barrier
            LIKWID_MARKER_START(label);
            for (j = 0; j < counter * threadId; j++)
            {
                for (i = 0; i < SIZE; i++)
                {
                    a[i] = b[i] + alpha * c[i];
                    sum += a[i];
                }
            }
#pragma omp barrier
            LIKWID_MARKER_STOP(label);
            printf("Flops performed thread %d region %s: %g\n",threadId, label,(double)counter*threadId*SIZE*3);
        }
        free(label);
    }
/****************************************************/


    LIKWID_MARKER_CLOSE;
    printf( "OK, dofp result = %e\n", sum);
}
