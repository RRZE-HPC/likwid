/*
 * ===========================================================================
 *
 *      Filename:  likwid-bench.c
 *
 *      Description:  A flexible and extensible benchmarking toolbox
 *
 *      Version:  <VERSION>
 *      Created:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Company:  RRZE Erlangen
 *      Project:  likwid
 *      Copyright:  Copyright (c) 2010, Jan Treibig
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License, v2, as
 *      published by the Free Software Foundation
 *     
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *     
 *      You should have received a copy of the GNU General Public License
 *      along with this program; if not, write to the Free Software
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * ===========================================================================
 */

#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>

#include <bstrlib.h>
#include <types.h>
#include <cpuid.h>
#include <affinity.h>
#include <timer.h>
#include <threads.h>
#include <barrier.h>
#include <testcases.h>
#include <strUtil.h>
#include <allocator.h>

#include <likwid.h>

extern void* runTest(void* arg);

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define HELP_MSG \
printf("Threaded Memory Hierarchy Benchmark --  Version 1.0\n"); \
printf("\n"); \
printf("Supported Options:\n"); \
printf("-h\t Help message\n"); \
printf("-a\t list available benchmarks \n"); \
printf("-p\t list available thread domains \n"); \
printf("-l <TEST>\t list properties of benchmark \n"); \
printf("-i <INT>\t number of iterations \n"); \
printf("-g <INT>\t number of workgroups (mandatory)\n"); \
printf("-t <TEST>\t type of test \n"); \
printf("-w\t <thread_domain>:<size>[:<num_threads>]-<streamId>:<domain_id>[:<offset>], size in kB, MB or GB  (mandatory)\n"); \
printf("Usage: likwid-bench -t copy -i 1000 -g 1 -w S0:100kB \n"); \
exit(0);

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE  ############ */

void
copyThreadData(ThreadUserData* src,ThreadUserData* dst)
{
    uint32_t i;

    *dst = *src;
    dst->processors = (int*) malloc(src->numberOfThreads*sizeof(int));
    dst->streams = (void**) malloc(src->test->streams*sizeof(void*));

    for (i=0; i<  src->test->streams; i++)
    {
        dst->streams[i] = src->streams[i];
    }

    for (i=0; i<src->numberOfThreads; i++)
    {
        dst->processors[i] = src->processors[i];
    }
}



/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int main(int argc, char** argv)
{
    int iter = 100;
    uint32_t i;
    uint32_t j;
    int globalNumberOfThreads = 0;
    int optPrintDomains = 0;
    int c;
    ThreadUserData myData;
    bstring testcase = bfromcstr("none");
    uint32_t numberOfWorkgroups = 0;
    int tmp = 0;
    double time;
    const TestCase* test = NULL;
    Workgroup* currentWorkgroup = NULL;
    Workgroup* groups = NULL;

    cpuid_init();
    affinity_init();

    /* Handling of command line options */
    if (argc ==  1) { HELP_MSG }

    while ((c = getopt (argc, argv, "g:w:t:i:l:aph")) != -1) {
        switch (c)
        {
            case 'h':
                HELP_MSG
                exit (EXIT_SUCCESS);    
            case 'a':
                printf(TESTS"\n");
                exit (EXIT_SUCCESS);    
            case 'w':
                tmp--;

                if (tmp == -1)
                {
                    fprintf (stderr, "More workgroups configured than allocated!\n");
                    return EXIT_FAILURE;
                }
                if (!test)
                {
                    fprintf (stderr, "You need to specify a test case first!\n");
                    return EXIT_FAILURE;
                }
                testcase = bfromcstr(optarg);
                currentWorkgroup = groups+tmp;  /*FIXME*/
                bstr_to_workgroup(currentWorkgroup, testcase, test->type, test->streams);
                bdestroy(testcase);

                for (i=0; i<  test->streams; i++)
                {
                    if (currentWorkgroup->streams[i].offset%test->stride)
                    {
                        fprintf (stderr, "Stream %d: offset is not a multiple of stride!\n",i);
                        return EXIT_FAILURE;
                    }

                    allocator_allocateVector(&(currentWorkgroup->streams[i].ptr),
                            PAGE_ALIGNMENT,
                            currentWorkgroup->size,
                            currentWorkgroup->streams[i].offset,
                            test->type,
                            currentWorkgroup->streams[i].domain);
                }

                break;
            case 'i':
                iter =  atoi(optarg);
                break;
            case 'l':
                testcase = bfromcstr(optarg);
                for (i=0; i<NUMKERNELS; i++)
                {
                    if (biseqcstr(testcase, kernels[i].name))
                    {
                        test = kernels+i;
                        break;
                    }
                }

                if (biseqcstr(testcase,"none"))
                {
                    fprintf (stderr, "Unknown test case %s\n",optarg);
                    return EXIT_FAILURE;
                }
                else
                {
                    printf("Name: %s\n",test->name);
                    printf("Number of streams: %d\n",test->streams);
                    printf("Loop stride: %d\n",test->stride);
                    printf("Flops: %d\n",test->flops);
                    printf("Bytes: %d\n",test->bytes);
                    switch (test->type)
                    {
                        case SINGLE:
                            printf("Data Type: Single precision float\n");
                            break;
                        case DOUBLE:
                            printf("Data Type: Double precision float\n");
                            break;
                    }
                }
                bdestroy(testcase);
                exit (EXIT_SUCCESS);    

                break;
            case 'p':
                optPrintDomains = 1;
                break;
            case 'g':
                numberOfWorkgroups =  atoi(optarg);
                allocator_init(numberOfWorkgroups * MAX_STREAMS);
                tmp = numberOfWorkgroups;
                groups = (Workgroup*) malloc(numberOfWorkgroups*sizeof(Workgroup));
                break;
            case 't':
                testcase = bfromcstr(optarg);

                for (i=0; i<NUMKERNELS; i++)
                {
                    if (biseqcstr(testcase, kernels[i].name))
                    {
                        test = kernels+i;
                        break;
                    }
                }

                if (biseqcstr(testcase,"none"))
                {
                    fprintf (stderr, "Unknown test case %s\n",optarg);
                    return EXIT_FAILURE;
                }
                bdestroy(testcase);
                break;
            case '?':
                if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
                return EXIT_FAILURE;
            default:
                HELP_MSG
        }
    }


    if (optPrintDomains)
    {
        affinity_printDomains();
        exit (EXIT_SUCCESS);    
    }
    timer_init();

 /* :WARNING:05/04/2010 08:58:05 AM:jt: At the moment the thread
  * module only allows equally sized thread groups*/
    for (i=0; i<numberOfWorkgroups; i++)
    {
        globalNumberOfThreads += groups[i].numberOfThreads;
    }

    threads_init(globalNumberOfThreads);
    threads_createGroups(numberOfWorkgroups);

    /* we configure global barriers only */
    barrier_init(1);
    barrier_registerGroup(globalNumberOfThreads);

#ifdef PERFMON
    printf("Using likwid\n");
    likwid_markerInit(globalNumberOfThreads,1);
    likwid_markerRegisterRegion("Bench");
#endif


    /* initialize data structures for threads */
    for (i=0; i<numberOfWorkgroups; i++)
    {
        myData.iter = iter;
        myData.size = groups[i].size;
        myData.test = test;
        myData.numberOfThreads = groups[i].numberOfThreads;
        myData.processors = (int*) malloc(myData.numberOfThreads * sizeof(int));
        myData.streams = (void**) malloc(test->streams * sizeof(void*));

        for (j=0; j<groups[i].numberOfThreads; j++)
        {
            myData.processors[j] = groups[i].processorIds[j];
        }

        for (j=0; j<  test->streams; j++)
        {
            myData.streams[j] = groups[i].streams[j].ptr;
        }

        threads_registerDataGroup(i, &myData, copyThreadData);

        free(myData.processors);
        free(myData.streams);
    }

    printf(HLINE);
    printf("LIKWID MICRO BENCHMARK\n"); 
    printf("Test: %s\n",test->name); 
    printf(HLINE);
    printf("Using %d work groups\n",numberOfWorkgroups);
    printf("Using %d threads\n",globalNumberOfThreads);
    printf(HLINE);

    threads_create(runTest); 
    threads_destroy();
    allocator_finalize();

    time = (double) threads_data[0].cycles / (double) timer_getCpuClock();
    printf("Cycles: %llu \n", LLU_CAST threads_data[0].cycles);
    printf("Iterations: %llu \n", LLU_CAST iter);
    printf("Size: %d \n",  currentWorkgroup->size );
    printf("Vectorlength: %d \n", threads_data[0].data.size);
    printf("Time: %e sec\n", time);
    printf("MFlops/s:\t%.2f\n",
            1.0E-06 * ((double) numberOfWorkgroups * iter * currentWorkgroup->size *  test->flops/  time));
    printf("MByte/s:\t%.2f\n",
            1.0E-06 * ( (double) numberOfWorkgroups * iter * currentWorkgroup->size *  test->bytes/ time));
    printf("Cycles per update:\t%f\n",
            ((double) threads_data[0].cycles / (double) (iter * threads_data[0].data.size)));

	switch ( test->type )
    {
        case SINGLE:
    printf("Cycles per cacheline:\t%f\n",
            (16.0 * (double) threads_data[0].cycles / (double) (iter * threads_data[0].data.size)));
            break;
        case DOUBLE:
    printf("Cycles per cacheline:\t%f\n",
            (8.0 * (double) threads_data[0].cycles / (double) (iter * threads_data[0].data.size)));
            break;
    }

    printf(HLINE);
#ifdef PERFMON
   likwid_markerClose();
#endif

    return EXIT_SUCCESS;
}

