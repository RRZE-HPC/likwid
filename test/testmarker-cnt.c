#include <stdio.h>
#include <omp.h>

#ifdef PERFMON
#include <likwid.h>
#endif

#define SIZE 1000000

double sum = 0, a[SIZE], b[SIZE], c[SIZE];

main()
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

#ifdef PERFMON
    printf("Using likwid\n");
    likwid_markerInit();
#endif

/****************************************************/
#pragma omp parallel
    {
        char* label = malloc(40*sizeof(char));
        int threadId = omp_get_thread_num();

        for (int counter=1; counter< 3; counter++)
        {
            sprintf(label,"plain-%d",counter);
#ifdef PERFMON
#pragma omp barrier
            likwid_markerStartRegion(label);
#endif
            for (j = 0; j < counter * threadId; j++)
            {
                for (i = 0; i < SIZE; i++) 
                {
                    a[i] = b[i] + alpha * c[i];
                    sum += a[i];
                }
            }
#ifdef PERFMON
#pragma omp barrier
            likwid_markerStopRegion(label);
#endif
            printf("Flops performed thread %d region %s: %g\n",threadId, label,(double)counter*threadId*SIZE*3);
        }
        free(label);
    }
/****************************************************/


#ifdef PERFMON
    likwid_markerClose();
#endif
    printf( "OK, dofp result = %e\n", sum);
}
