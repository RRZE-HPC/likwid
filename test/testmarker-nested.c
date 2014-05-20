#include <stdio.h>
#include <omp.h>

#ifdef PERFMON
#include <likwid.h>
#endif

#define SIZE 1000000
#define N1   1000
#define N2   300
#define N3   500

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
    likwid_markerInit(numberOfThreads,numberOfRegions);
#endif

#pragma omp parallel for num_threads(num_tasks) 
        for (int task = 0; task<num_dels+1; task++) { 
            if (task==0) { 
#pragma omp parallel for num_threads(num_threads) 
                for (thread_num=0; thread_num < num_threads; thread_num++) { 
                    cpu_set_t set; 
                    CPU_ZERO(&set); 
                    CPU_SET(...) 
                    sched_setaffinity(0, sizeof(cpu_set_t), &set); 

                    struct marker_t m = markerStart(); 
                    /**... work ...**/ 
                    markerStop(m, regionId); 
                } //barrier 

                /**...unmeasured work **/ 
            } else { //task==1,2 
                cpu_set_t set; 
                CPU_ZERO(&set); 
                CPU_SET(...) 
                sched_setaffinity(0, sizeof(cpu_set_t), &set); 

                struct marker_t m = markerStart(); 
                /**... work... **/ 
                markerStop(m, regionId); 
            } 
        } 

#ifdef PERFMON
    likwid_markerClose();
#endif
    printf( "OK, dofp result = %e\n", sum);
}
