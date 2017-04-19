#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <unistd.h>

#include <likwid.h>

#define SIZE 1000000

double sum = 0, a[SIZE], b[SIZE], c[SIZE];

int main(int argc, char* argv[])
{
    double alpha = 3.14;

    /* Initialize */
    for (int i=0; i<SIZE; i++)
    {
        a[i] = 1.0/(double) i;
        b[i] = 1.0;
        c[i] = (double) i;
    }

    LIKWID_MARKER_INIT;

#pragma omp parallel
    {
        LIKWID_MARKER_THREADINIT;

        LIKWID_MARKER_START("time");
        sleep(2);
        LIKWID_MARKER_STOP("time");

        int threadId = omp_get_thread_num();
        /****************************************************/
#pragma omp for
        for (int j = 0; j < 10; j++)
        {

            LIKWID_MARKER_START("plain");
            for (int k = 0; k < (threadId+1); k++)  {
                for (int i = 0; i < SIZE; i++) 
                {
                    a[i] = b[i] + alpha * c[i];
                    sum += a[i];
                }
            }

            LIKWID_MARKER_STOP("plain");
        }
        printf("Flops performed plain: %g\n",(double)10*SIZE*3);
        /****************************************************/
    }


    LIKWID_MARKER_CLOSE;
    printf( "OK, dofp result = %e\n", sum);
}
