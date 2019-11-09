#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <stdint.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>


#define ITER 10
#define SIZE 40000000

#define gettid() syscall(SYS_gettid)
#include <likwid-marker.h>
#define HLINE "-------------------------------------------------------------\n"

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif

typedef struct {
    struct timeval before;
    struct timeval after;
} TimeData;


void time_start(TimeData* time)
{
    gettimeofday(&(time->before),NULL);
}


void time_stop(TimeData* time)
{
    gettimeofday(&(time->after),NULL);
}

double time_print(TimeData* time)
{
    long int sec;
    double timeDuration;

    sec = time->after.tv_sec - time->before.tv_sec;
    timeDuration = ((double)((sec*1000000)+time->after.tv_usec) - (double) time->before.tv_usec);

    return (timeDuration/1000000);
}

static int nprocessors = 0;

static int
getProcessorID(cpu_set_t* cpu_set)
{
    int processorId;

    for (processorId=0;processorId<nprocessors;processorId++)
    {
	if (CPU_ISSET(processorId,cpu_set))
	{
	    break;
	}
    }
    return processorId;
}


int  threadProcessorId()
{
    cpu_set_t  cpu_set;
    CPU_ZERO(&cpu_set);
    sched_getaffinity(gettid(),sizeof(cpu_set_t), &cpu_set);

    return getProcessorID(&cpu_set);
}

void allocate_vector(double** ptr, uint64_t size)
{
    int errorCode;

    errorCode = posix_memalign((void**) ptr, 64, size*sizeof(double));

    if (errorCode)
    {
	if (errorCode == EINVAL)
	{
	    fprintf(stderr,
		    "Alignment parameter is not a power of two\n");
	    exit(EXIT_FAILURE);
	}
	if (errorCode == ENOMEM)
	{
	    fprintf(stderr,
		    "Insufficient memory to fulfill the request\n");
	    exit(EXIT_FAILURE);
	}
    }
}



static int get_nworkers()
{
    return __cilkrts_get_nworkers();
}
static int get_totalworkers()
{
    return __cilkrts_get_total_workers();
}

static int show_thread()
{
    int ID = __cilkrts_get_worker_number();
    printf("Thread %d TID %lu CPU %d\n", ID, gettid(), sched_getcpu());
    return 0;
}

int main(){
    int i, k;
    int nworkers, totalworkers;
    char cpuCount[20];
    double *a, *b, *c, *d;
    double sums[2000];
    cpu_set_t cpuset;
    TimeData timer;
    double triad_time, copy_time, total = 0;

    nprocessors = sysconf(_SC_NPROCESSORS_CONF);

    nworkers = cilk_spawn get_nworkers();
    totalworkers = cilk_spawn get_totalworkers();

    for (i=0;i<nworkers;i++)
    {
        sums[i] = 0;
    }

    LIKWID_MARKER_INIT;

    cilk_spawn allocate_vector(&a, SIZE);
    cilk_spawn allocate_vector(&b, SIZE);
    cilk_spawn allocate_vector(&c, SIZE);
    cilk_spawn allocate_vector(&d, SIZE);
    cilk_sync;

    for (i=0; i<SIZE; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
        d[i] = 1.0;
    }

    time_start(&timer);
    for (k=0; k<ITER; k++)
    {
        for (i=0;i<nworkers;i++)
        {
            cilk_spawn LIKWID_MARKER_START("copy");
        }
        cilk_sync;
        cilk_for(i=0;i<SIZE;i++)
        {
            c[i] = a[i];
        }
        for (i=0;i<nworkers;i++)
        {
            cilk_spawn LIKWID_MARKER_STOP("copy");
        }
        cilk_sync;
    }
    time_stop(&timer);
    copy_time = time_print(&timer)/(double)ITER;

    time_start(&timer);
    for (k=0; k<ITER; k++)
    {
        for (i=0;i<nworkers;i++)
        {
            cilk_spawn LIKWID_MARKER_START("triad");
        }
        cilk_sync;
        cilk_for(i=0;i<SIZE;i++)
        {
            a[i] = b[i] +  c[i] * d[i];
        }
        for (i=0;i<nworkers;i++)
        {
            cilk_spawn LIKWID_MARKER_STOP("triad");
        }
        cilk_sync;
    }
    time_stop(&timer);
    triad_time = time_print(&timer)/(double)ITER;

    printf("Processed %.1f Mbyte at copy benchmark in %.4f seconds: %.2f MByte/s\n",
                        1E-6*(2*SIZE*sizeof(double)),
                        copy_time,
                        1E-6*((2*SIZE*sizeof(double))/copy_time));
    printf("Processed %.1f Mbyte at triad benchmark in %.4f seconds: %.2f MByte/s\n",
                        1E-6*(4*SIZE*sizeof(double)),
                        triad_time,
                        1E-6*((4*SIZE*sizeof(double))/triad_time));

    printf("Main PID %d\n",getpid());
    for (i=0;i<nworkers;i++)
    {
        cilk_spawn show_thread();
    }
    cilk_sync;

    LIKWID_MARKER_CLOSE;
}
