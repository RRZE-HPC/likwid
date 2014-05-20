#include <stdio.h>
#include <omp.h>

#ifdef PERFMON
#include <likwid.h>
#endif

#define SIZE 1000000

double sum = 0, a[SIZE], b[SIZE], c[SIZE];

main()
{
    double alpha = 3.14;

    /* Initialize */
    for (int i=0; i<SIZE; i++)
    {
        a[i] = 1.0/(double) i;
        b[i] = 1.0;
        c[i] = (double) i;
    }

#ifdef PERFMON
    printf("Using likwid\n");
    likwid_markerInit();
#endif

#pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        /****************************************************/
#pragma omp for
        for (int j = 0; j < 10; j++)
        {

#ifdef PERFMON
        likwid_markerStartRegion("plain");
#endif
            for (int k = 0; k < (threadId+1); k++)  {
                for (int i = 0; i < SIZE; i++) 
                {
                    a[i] = b[i] + alpha * c[i];
                    sum += a[i];
                }
            }

#ifdef PERFMON
        likwid_markerStopRegion("plain");
#endif
        }
        printf("Flops performed plain: %g\n",(double)10*SIZE*3);
        /****************************************************/
    }


#ifdef PERFMON
    likwid_markerClose();
#endif
    printf( "OK, dofp result = %e\n", sum);
}
