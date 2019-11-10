/*
 * =======================================================================================
 *
 *      Filename:  streamAPI.c
 *
 *      Description:  Copy of the STREAM benchmark (only copy and triad) with hardware
 *                    performance measurement instrumentation using LIKWID
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */

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

#define ITER 100
#define SIZE 40000000
#define DATATYPE float

#define gettid() syscall(SYS_gettid)
#include <likwid.h>
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

void allocate_vector(DATATYPE** ptr, uint64_t size)
{
    int errorCode;

    errorCode = posix_memalign((void**) ptr, 64, size*sizeof(DATATYPE));

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
    int err, i ,j;
    int numCPUs = 0;
    int gid;
    DATATYPE *a,*b,*c,*d;
    TimeData timer;
    double triad_time, copy_time, scale_time, stream_time;
    char estr[1024];
    double result, scalar = 3.0;
    char* ptr;

    if (argn != 3)
    {
        printf("Usage: %s <cpustr> <events>\n", argc[0]);
        return 1;
    }

    strcpy(estr, argc[2]);

    allocate_vector(&a, SIZE);
    allocate_vector(&b, SIZE);
    allocate_vector(&c, SIZE);
    allocate_vector(&d, SIZE);

    err = topology_init();
    if (err < 0)
    {
        printf("Failed to initialize LIKWID's topology module\n");
        return 1;
    }
    CpuTopology_t topo = get_cpuTopology();
    affinity_init();
    int* cpus = (int*)malloc(topo->numHWThreads * sizeof(int));
    if (!cpus)
        return 1;
    numCPUs = cpustr_to_cpulist(argc[1], cpus, topo->numHWThreads);
    omp_set_num_threads(numCPUs);
    err = perfmon_init(numCPUs, cpus);
    if (err < 0)
    {
        printf("Failed to initialize LIKWID's performance monitoring module\n");
        affinity_finalize();
        topology_finalize();
        return 1;
    }
    gid = perfmon_addEventSet(estr);
    if (gid < 0)
    {
        printf("Failed to add event string %s to LIKWID's performance monitoring module\n", estr);
        perfmon_finalize();
        affinity_finalize();
        topology_finalize();
        return 1;
    }

    err = perfmon_setupCounters(gid);
    if (err < 0)
    {
        printf("Failed to setup group %d in LIKWID's performance monitoring module\n", gid);
        perfmon_finalize();
        affinity_finalize();
        topology_finalize();
        return 1;
    }

#ifdef _OPENMP
    printf(HLINE);
#pragma omp parallel
    {
#pragma omp master
    {
        printf ("Number of Threads requested = %i\n",omp_get_num_threads());
    }
    likwid_pinThread(cpus[omp_get_thread_num()]);
    printf ("Thread %d running on processor %d ....\n",omp_get_thread_num(),sched_getcpu());
    }
#endif

#pragma omp parallel for
    for (int j=0; j<SIZE; j++) {
        a[j] = 1.0;
        b[j] = 2.0;
        c[j] = 0.0;
        d[j] = 1.0;
    }

    err = perfmon_startCounters();
    if (err < 0)
    {
        printf("Failed to start counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }
    time_start(&timer);
#pragma omp parallel
    {
        for (int k=0; k<ITER; k++)
        {
            LIKWID_MARKER_START("copy");
#pragma omp for
            for (int j=0; j<SIZE; j++)
            {
                c[j] = a[j];
            }
            LIKWID_MARKER_STOP("copy");
        }
    }
    time_stop(&timer);
    err = perfmon_stopCounters();
    copy_time = time_print(&timer)/(double)ITER;
    if (err < 0)
    {
        printf("Failed to stop counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }

    printf("Processed %.1f Mbyte at copy benchmark in %.4f seconds: %.2f MByte/s\n",
                        1E-6*(2*SIZE*sizeof(DATATYPE)),
                        copy_time,
                        1E-6*((2*SIZE*sizeof(DATATYPE))/copy_time));

    ptr = strtok(estr,",");
    j = 0;
    while (ptr != NULL)
    {
        for (i = 0;i < numCPUs; i++)
        {
            result = perfmon_getResult(gid, j, cpus[i]);
            printf("Measurement result for event set %s at CPU %d: %f\n", ptr, cpus[i], result);
        }
        ptr = strtok(NULL,",");
        j++;
    }
    strcpy(estr, argc[2]);
    perfmon_setupCounters(gid);

    err = perfmon_startCounters();
    if (err < 0)
    {
        printf("Failed to start counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }
    time_start(&timer);
#pragma omp parallel
    {
        for (int k=0; k<ITER; k++)
        {
            LIKWID_MARKER_START("scale");
#pragma omp for
            for (int j=0; j<SIZE; j++)
            {
                b[j] = scalar*c[j];
            }
            LIKWID_MARKER_STOP("scale");
        }
    }
    time_stop(&timer);
    err = perfmon_stopCounters();
    scale_time = time_print(&timer)/(double)ITER;
    if (err < 0)
    {
        printf("Failed to stop counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }

    printf("Processed %.1f Mbyte at scale benchmark in %.4f seconds: %.2f MByte/s\n",
                        1E-6*(2*SIZE*sizeof(DATATYPE)),
                        copy_time,
                        1E-6*((2*SIZE*sizeof(DATATYPE))/copy_time));

    ptr = strtok(estr,",");
    j = 0;
    while (ptr != NULL)
    {
        for (i = 0;i < numCPUs; i++)
        {
            result = perfmon_getResult(gid, j, cpus[i]);
            printf("Measurement result for event set %s at CPU %d: %f\n", ptr, cpus[i], result);
        }
        ptr = strtok(NULL,",");
        j++;
    }
    strcpy(estr, argc[2]);
    perfmon_setupCounters(gid);
    err = perfmon_startCounters();
    if (err < 0)
    {
        printf("Failed to start counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }
    time_start(&timer);
#pragma omp parallel
    {
        for (int k=0; k<ITER; k++)
        {
            LIKWID_MARKER_START("stream");
#pragma omp for
            for (int j=0; j<SIZE; j++)
            {
                c[j] = a[j] + b[j];
            }
            LIKWID_MARKER_STOP("stream");
        }
    }
    time_stop(&timer);
    err = perfmon_stopCounters();
    stream_time = time_print(&timer)/(double)ITER;
    if (err < 0)
    {
        printf("Failed to stop counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }

    printf("Processed %.1f Mbyte at stream benchmark in %.4f seconds: %.2f MByte/s\n",
                        1E-6*(2*SIZE*sizeof(DATATYPE)),
                        copy_time,
                        1E-6*((2*SIZE*sizeof(DATATYPE))/copy_time));

    ptr = strtok(estr,",");
    j = 0;
    while (ptr != NULL)
    {
        for (i = 0;i < numCPUs; i++)
        {
            result = perfmon_getResult(gid, j, cpus[i]);
            printf("Measurement result for event set %s at CPU %d: %f\n", ptr, cpus[i], result);
        }
        ptr = strtok(NULL,",");
        j++;
    }
    strcpy(estr, argc[2]);
    perfmon_setupCounters(gid);
    err = perfmon_startCounters();
    if (err < 0)
    {
        printf("Failed to start counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }
    time_start(&timer);
#pragma omp parallel
    {
        for (int k=0; k<ITER; k++)
        {

            LIKWID_MARKER_START("triad");
#pragma omp for
            for (int j=0; j<SIZE; j++)
            {
                a[j] = b[j] +  c[j] * scalar;
            }
            LIKWID_MARKER_STOP("triad");
        }
    }
    time_stop(&timer);
    err = perfmon_stopCounters();
    triad_time = time_print(&timer)/(double)ITER;
    if (err < 0)
    {
        printf("Failed to stop counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }



    printf("Processed %.1f Mbyte at triad benchmark in %.4f seconds: %.2f MByte/s\n",
                        1E-6*(4*SIZE*sizeof(DATATYPE)),
                        triad_time,
                        1E-6*((4*SIZE*sizeof(DATATYPE))/triad_time));
    ptr = strtok(estr,",");
    j = 0;
    while (ptr != NULL)
    {
        for (i = 0;i < numCPUs; i++)
        {
            result = perfmon_getResult(gid, j, cpus[i]);
            printf("Measurement result for event set %s at CPU %d: %f\n", ptr, cpus[i], result);
        }
        ptr = strtok(NULL,",");
        j++;
    }

    perfmon_finalize();
    affinity_finalize();
    topology_finalize();
    return 0;
}

