#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sched.h>

#include <likwid-marker.h>

#define SIZE 1000000
#define N1   1000
#define N2   300
#define N3   500

double sum = 0, a[SIZE], b[SIZE], c[SIZE];

int main(int argc, char* argv[])
{
    double alpha = 3.14;
    int num_dels = 1;

    /* Initialize */
    for (int i=0; i<SIZE; i++)
    {
        a[i] = 1.0/(double) i;
        b[i] = 1.0;
        c[i] = (double) i;
    }

    LIKWID_MARKER_INIT;
    int num_threads = omp_get_num_threads();

#pragma omp parallel for num_threads(num_tasks)
    for (int task = 0; task<num_dels+1; task++) {
        if (task==0) {
#pragma omp parallel for num_threads(num_threads)
            for (int thread_num=0; thread_num < num_threads; thread_num++) {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(thread_num, &set);
                sched_setaffinity(0, sizeof(cpu_set_t), &set);

                /**... work ...**/
            } //barrier

        /**...unmeasured work **/
        } else { //task==1,2
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(0, &set);
            sched_setaffinity(0, sizeof(cpu_set_t), &set);

            /**... work... **/
        }
    }

    LIKWID_MARKER_CLOSE;
    printf( "OK, dofp result = %e\n", sum);
}
