#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/syscall.h>
#ifdef _OPENMP
#include <omp.h>
# endif
#include <stdint.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include <pthread.h>

#define ITER 10
#define SIZE 40000000

#define gettid() syscall(SYS_gettid)
#include <likwid-cpumarker.h>
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

static int
getProcessorID(cpu_set_t* cpu_set)
{
    int processorId;

    for (processorId=0;processorId<128;processorId++)
    {
        if (CPU_ISSET(processorId,cpu_set))
        {
            break;
        }
    }
    return processorId;
}

int  threadGetProcessorId()
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


int main(int argn, char** argc)
{
    double *a,*b,*c,*d;
    double scalar = 3.0;
    TimeData timer;
    double triad_time, copy_time, scale_time, add_time;

    allocate_vector(&a, SIZE);
    allocate_vector(&b, SIZE);
    allocate_vector(&c, SIZE);
    allocate_vector(&d, SIZE);

#ifdef LIKWID_PERFMON
    printf("Using likwid\n");
#endif

    LIKWID_MARKER_INIT;

#ifdef _OPENMP
    printf(HLINE);
#pragma omp parallel
    {
#pragma omp master
    {
        printf ("Number of Threads requested = %i\n",omp_get_num_threads());
    }
    printf ("Thread %d running on processor %d ....\n",omp_get_thread_num(),sched_getcpu());
    }
    printf(HLINE);
#endif

#pragma omp parallel for
    for (int j=0; j<SIZE; j++) {
        a[j] = 1.0;
        b[j] = 2.0;
        c[j] = 0.0;
        d[j] = 1.0;
    }

    time_start(&timer);
#pragma omp parallel
    {
        for (int k=0; k<ITER; k++)
        {
            LIKWID_MARKER_START("copy");
#pragma simd
#pragma omp for
            for (int j=0; j<SIZE; j++)
            {
                c[j] = a[j];
            }
            LIKWID_MARKER_STOP("copy");
        }
    }
    time_stop(&timer);
    copy_time = time_print(&timer)/(double)ITER;

    time_start(&timer);
#pragma omp parallel
    {
        for (int k=0; k<ITER; k++)
        {

            LIKWID_MARKER_START("scale");
#pragma simd
#pragma omp for
            for (int j=0; j<SIZE; j++)
            {
                b[j] = scalar * c[j];
            }
            LIKWID_MARKER_STOP("scale");
        }
    }
    time_stop(&timer);
    scale_time = time_print(&timer)/(double)ITER;

    time_start(&timer);
#pragma omp parallel
    {
        for (int k=0; k<ITER; k++)
        {

            LIKWID_MARKER_START("add");
#pragma simd
#pragma omp for
            for (int j=0; j<SIZE; j++)
            {
                c[j] = a[j] + b[j];
            }
            LIKWID_MARKER_STOP("add");
        }
    }
    time_stop(&timer);
    add_time = time_print(&timer)/(double)ITER;

    time_start(&timer);
#pragma omp parallel
    {
        LIKWID_MARKER_START("triad_total");
        for (int k=0; k<ITER; k++)
        {

            LIKWID_MARKER_START("triad");
#pragma simd
#pragma omp for
            for (int j=0; j<SIZE; j++)
            {

                a[j] = b[j] +  c[j] * d[j];
            }
            LIKWID_MARKER_STOP("triad");
        }
        LIKWID_MARKER_STOP("triad_total");
    }
    time_stop(&timer);
    triad_time = time_print(&timer)/(double)ITER;


    printf("Processed %.1f Mbyte at copy benchmark in %.4f seconds: %.2f MByte/s\n",
                        1E-6*(2*SIZE*sizeof(double)),
                        copy_time,
                        1E-6*((2*SIZE*sizeof(double))/copy_time));
    printf("Processed %.1f Mbyte at scale benchmark in %.4f seconds: %.2f MByte/s %.2f MFLOP/s\n",
                        1E-6*(2*SIZE*sizeof(double)),
                        scale_time,
                        1E-6*((2*SIZE*sizeof(double))/scale_time),
                        1E-6*(SIZE/scale_time));
    printf("Processed %.1f Mbyte at add benchmark in %.4f seconds: %.2f MByte/s %.2f MFLOP/s\n",
                        1E-6*(3*SIZE*sizeof(double)),
                        add_time,
                        1E-6*((3*SIZE*sizeof(double))/add_time),
                        1E-6*(SIZE/add_time));
    printf("Processed %.1f Mbyte at triad benchmark in %.4f seconds: %.2f MByte/s %.2f MFLOP/s\n",
                        1E-6*(4*SIZE*sizeof(double)),
                        triad_time,
                        1E-6*((4*SIZE*sizeof(double))/triad_time),
                        1E-6*((2*SIZE)/triad_time));


    LIKWID_MARKER_CLOSE;
    free(a);
    free(b);
    free(c);
    free(d);
    return 0;
}

