#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <atomic>
#include <thread>
#include <likwid-cpumarker.h>
#include <sched.h>
#include <syscall.h>
#include <sys/time.h>

#define gettid() syscall(SYS_gettid)
#define ITER 10
#define SIZE 40000000
#ifdef __GNUG__
#define RESTRICT __restrict__
#else
#define RESTRICT restrict
#endif
using namespace std;

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
    if (CPU_COUNT(&cpu_set) > 1)
    {
        return sched_getcpu();
    }
    else
    {
        return getProcessorID(&cpu_set);
    }
    return -1;
}


double copy_times[CPU_SETSIZE];
double triad_times[CPU_SETSIZE];

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



int calc_thread(double* RESTRICT a, double* RESTRICT b, double* RESTRICT c, double* RESTRICT d, int id, int all)
{
    int i;
    int start;
    int end;
    TimeData timer;
    start = id*(SIZE/all);
    end = start+(SIZE/all);

    LIKWID_MARKER_THREADINIT;

    printf ("Thread %d running on processor %d ....\n", id, threadGetProcessorId());

    time_start(&timer);
    for (int k=0; k<ITER; k++)
    {
        LIKWID_MARKER_START("copy");
        #pragma simd
        for(i=start;i<end;i++)
        {
            c[i] = a[i];
        }
        LIKWID_MARKER_STOP("copy");
    }
    time_stop(&timer);
    copy_times[id] = time_print(&timer);

    time_start(&timer);
    for (int k=0; k<ITER; k++)
    {
        LIKWID_MARKER_START("triad");
        #pragma simd
        for(i=start;i<end;i++)
        {
            a[i] = b[i] +  c[i] * d[i];
        }
        LIKWID_MARKER_STOP("triad");
    }
    time_stop(&timer);
    triad_times[id] = time_print(&timer);
    return 0;
}

int
main(int argc, char ** argv)
{
    cpu_set_t cpuset;
    sched_getaffinity(getpid(),sizeof(cpu_set_t), &cpuset);
    std::thread t[CPU_SETSIZE];
    double *a,*b,*c,*d;
    double copy_time = 0.0;
    double triad_time = 0.0;
    int num_threads = 0;
    int id = 0;

    for (int i=0;i<CPU_SETSIZE; i++)
    {
        if (CPU_ISSET(i, &cpuset))
        {
            num_threads++;
        }
        copy_times[i] = 0.0;
        triad_times[i] = 0.0;
    }

    printf ("Number of Threads requested = %i\n",num_threads);

    allocate_vector(&a, SIZE);
    allocate_vector(&b, SIZE);
    allocate_vector(&c, SIZE);
    allocate_vector(&d, SIZE);
    LIKWID_MARKER_INIT;

    #pragma ivdep
    for (int j=0; j<SIZE; ++j) {
        a[j] = 1.0;
        b[j] = 2.0;
        c[j] = 0.0;
        d[j] = 1.0;
    }

    for (int i=0;i<CPU_SETSIZE; i++)
    {
        if (CPU_ISSET(i, &cpuset))
        {
            t[i] = std::thread( calc_thread, a, b, c, d, id, num_threads);
            id++;
            if (id >= num_threads)
                break;
        }
    }
    id = 0;
    for (int i=0;i<CPU_SETSIZE; i++)
    {
        if (CPU_ISSET(i, &cpuset))
        {
            t[i].join();
            copy_time += copy_times[id]/(double)ITER;
            triad_time += triad_times[id]/(double)ITER;
            id++;
            if (id >= num_threads)
                break;
        }
    }

    printf("Processed %.1f Mbyte at copy benchmark in %.4f seconds: %.2f MByte/s\n",
                        1E-6*(2*SIZE*sizeof(double)),
                        copy_time,
                        1E-6*((2*SIZE*sizeof(double))/copy_time));
    printf("Processed %.1f Mbyte at triad benchmark in %.4f seconds: %.2f MByte/s\n",
                        1E-6*(4*SIZE*sizeof(double)),
                        triad_time,
                        1E-6*((4*SIZE*sizeof(double))/triad_time));

    LIKWID_MARKER_CLOSE;

    return 0;
}
