/*
 * =======================================================================================
 *
 *      Filename:  likwid-bench.c
 *
 *      Description:  A flexible and extensible benchmarking toolbox
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
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
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>

#include <bstrlib.h>
#include <types.h>
#include <error.h>
#include <cpuid.h>
#include <numa.h>
#include <affinity.h>
#include <timer.h>
#include <threads.h>
#include <barrier.h>
#include <testcases.h>
#include <strUtil.h>
#include <allocator.h>

#include <likwid.h>
#ifdef PAPI
#include <papi.h>
#include <omp.h>
#endif

extern void* runTest(void* arg);

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define HELP_MSG \
    fprintf(stdout, "Threaded Memory Hierarchy Benchmark --  Version  %d.%d \n\n",VERSION,RELEASE); \
    fprintf(stdout, "\n"); \
    fprintf(stdout, "Supported Options:\n"); \
    fprintf(stdout, "-h\t Help message\n"); \
    fprintf(stdout, "-v\t Version information\n"); \
    fprintf(stdout, "-q\t Silent without output\n"); \
    fprintf(stdout, "-a\t list available benchmarks \n"); \
    fprintf(stdout, "-p\t list available thread domains\n"); \
    fprintf(stdout, "-l <TEST>\t list properties of benchmark \n"); \
    fprintf(stdout, "-i <INT>\t number of iterations \n"); \
    fprintf(stdout, "-g <INT>\t number of workgroups (mandatory)\n"); \
    fprintf(stdout, "-t <TEST>\t type of test \n"); \
    fprintf(stdout, "-w\t <thread_domain>:<size>[:<num_threads>[:<chunk size>:<stride>][-<streamId>:<domain_id>[:<offset>]], size in kB, MB or GB  (mandatory)\n"); \
    fprintf(stdout, "Processors are in compact ordering. Optionally every stream can be placed. Either no stream or all streams must be placed. Multiple streams are separated by commas.\n"); \
    fprintf(stdout, "Usage: likwid-bench -t copy -i 1000 -g 1 -w S0:100kB:10:1:2 \n"); \
    fprintf(stdout, "\tRun 10 threads on socket 0 using physical cores only (presuming SMT2 system).\n"); \
    fprintf(stdout, "Example with data placement: likwid-bench -t copy -i 1000 -g 1 -w S0:100kB:20-0:S1,1:S1 \n"); \
    fprintf(stdout, "\tRun 20 threads on socket 0 and place both arrays of the copy test case on socket 1.\n"); \
    fflush(stdout);

#define VERSION_MSG \
    fprintf(stdout, "likwid-bench   %d.%d \n\n",VERSION,RELEASE); \
    fflush(stdout);

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE  ############ */

void copyThreadData(ThreadUserData* src,ThreadUserData* dst)
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
    FILE* OUTSTREAM = stdout;

    if (cpuid_init() == EXIT_FAILURE)
    {
        ERROR_PLAIN_PRINT(Unsupported processor!);
    }
    numa_init();
    affinity_init();

    /* Handling of command line options */
    if (argc ==  1)
    {
        HELP_MSG;
        affinity_finalize();
        exit(EXIT_SUCCESS);
    }
    opterr = 0;
    while ((c = getopt (argc, argv, "g:w:t:i:l:aphvq")) != -1) {
        switch (c)
        {
            case 'h':
                HELP_MSG;
                affinity_finalize();
                if (groups)
                {
                    free(groups);
                }
                exit (EXIT_SUCCESS);
            case 'v':
                VERSION_MSG;
                affinity_finalize();
                if (groups)
                {
                    free(groups);
                }
                exit (EXIT_SUCCESS);
            case 'a':
                if (OUTSTREAM)
                {
                    fprintf(OUTSTREAM, TESTS"\n");
                    fflush(OUTSTREAM);
                }
                affinity_finalize();
                if (groups)
                {
                    free(groups);
                }
                exit (EXIT_SUCCESS);
            case 'q':
                OUTSTREAM = NULL;
                break;
            case 'w':
                tmp--;

                if (tmp == -1)
                {
                    fprintf (stderr, "More workgroups configured than allocated!\n"
                        "Did you forget to set the number of workgroups with -g?\n");
                    affinity_finalize();
                    if (groups)
                    {
                        free(groups);
                    }
                    return EXIT_FAILURE;
                }
                if (!test)
                {
                    fprintf (stderr, "You need to specify a test case first!\n");
                    affinity_finalize();
                    if (groups)
                    {
                        free(groups);
                    }
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
                        affinity_finalize();
                        if (groups)
                        {
                            free(groups);
                        }
                        return EXIT_FAILURE;
                    }

                    allocator_allocateVector(OUTSTREAM,
                            &(currentWorkgroup->streams[i].ptr),
                            PAGE_ALIGNMENT,
                            currentWorkgroup->size,
                            currentWorkgroup->streams[i].offset,
                            test->type,
                            currentWorkgroup->streams[i].domain);
                }

                break;
            case 'i':
                iter =  atoi(optarg);
                if (iter <= 0)
                {
                    fprintf(stderr, "Iterations must be greater than 0.\n");
                    exit(EXIT_FAILURE);
                }
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

                if (biseqcstr(testcase,"none") || !test)
                {
                    fprintf (stderr, "Unknown test case %s\n",optarg);
                    if (OUTSTREAM)
                    {
                        fprintf(OUTSTREAM, "Available test cases:\n");
                        fprintf(OUTSTREAM, TESTS"\n");
                        fflush(OUTSTREAM);
                    }
                    affinity_finalize();
                    if (groups)
                    {
                        free(groups);
                    }
                    return EXIT_FAILURE;
                }
                else
                {
                    if (OUTSTREAM)
                    {
                        fprintf(OUTSTREAM, "Name: %s\n",test->name);
                        fprintf(OUTSTREAM, "Number of streams: %d\n",test->streams);
                        fprintf(OUTSTREAM, "Loop stride: %d\n",test->stride);
                        fprintf(OUTSTREAM, "Flops: %d\n", (int) test->flops);
                        fprintf(OUTSTREAM, "Bytes: %d\n",test->bytes);
                        switch (test->type)
                        {
                            case SINGLE:
                                fprintf(OUTSTREAM, "Data Type: Single precision float\n");
                                break;
                            case DOUBLE:
                                fprintf(OUTSTREAM, "Data Type: Double precision float\n");
                                break;
                        }
                        fflush(OUTSTREAM);
                    }
                }
                bdestroy(testcase);
                affinity_finalize();
                if (groups)
                {
                    free(groups);
                }
                exit (EXIT_SUCCESS);

                break;
            case 'p':
                optPrintDomains = 1;
                break;
            case 'g':
                numberOfWorkgroups =  atoi(optarg);
                if (numberOfWorkgroups <= 0)
                {
                    fprintf(stderr, "Number of Workgroups must be 1 or greater.\n");
                    exit(EXIT_FAILURE);
                }
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
                    affinity_finalize();
                    if (groups)
                    {
                        free(groups);
                    }
                    return EXIT_FAILURE;
                }
                bdestroy(testcase);
                break;
            case '?':
                if (optopt == 'l' || optopt == 'g' || optopt == 'w' || 
                        optopt == 't' || optopt == 'i')
                    fprintf (stderr, "Option `-%c' requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
                affinity_finalize();
                if (groups)
                {
                    free(groups);
                }
                return EXIT_FAILURE;
            default:
                HELP_MSG;
        }
    }

    if (numberOfWorkgroups == 0 && !optPrintDomains)
    {
        fprintf(stderr, "Number of Workgroups must be 1 or greater.\n");
        affinity_finalize();
        allocator_finalize();
        if (groups)
        {
            free(groups);
        }
        exit(EXIT_FAILURE);
    }
    if (tmp > 0 && iter > 0)
    {
        fprintf(stderr, "%d workgroups requested but only %d given on commandline\n",numberOfWorkgroups,numberOfWorkgroups-tmp);
        affinity_finalize();
        allocator_finalize();
        if (groups)
        {
            free(groups);
        }
        exit(EXIT_FAILURE);
    }
    if (iter <= 0)
    {
        fprintf(stderr,"Iterations must be greater than 0\n");
        affinity_finalize();
        allocator_finalize();
        if (groups)
        {
            free(groups);
        }
        exit(EXIT_FAILURE);
    }
    if (test && !(currentWorkgroup || groups))
    {
        fprintf(stderr, "Workgroups must be set on commandline\n");
        affinity_finalize();
        allocator_finalize();
        if (groups)
        {
            free(groups);
        }
        exit(EXIT_FAILURE);
    }

    if (optPrintDomains)
    {
        affinity_printDomains(OUTSTREAM);
        affinity_finalize();
        allocator_finalize();
        if (groups)
        {
            free(groups);
        }
        exit (EXIT_SUCCESS);
    }
    timer_init();

    /* :WARNING:05/04/2010 08:58:05 AM:jt: At the moment the thread
     * module only allows equally sized thread groups*/
    for (i=0; i<numberOfWorkgroups; i++)
    {
        globalNumberOfThreads += groups[i].numberOfThreads;
    }

    threads_init(OUTSTREAM, globalNumberOfThreads);
    threads_createGroups(numberOfWorkgroups);

    /* we configure global barriers only */
    barrier_init(1);
    barrier_registerGroup(globalNumberOfThreads);

#ifdef PERFMON
    if (OUTSTREAM)
    {
        fprintf(OUTSTREAM, "Using likwid\n");
        fflush(OUTSTREAM);
    }
    likwid_markerInit();
#endif
#ifdef PAPI
    if (OUTSTREAM)
    {
        fprintf(OUTSTREAM, "Using PAPI\n");
    }
    PAPI_library_init (PAPI_VER_CURRENT);
    PAPI_thread_init((unsigned long (*)(void))(omp_get_thread_num));
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

    if (OUTSTREAM)
    {
        fprintf(OUTSTREAM, HLINE);
        fprintf(OUTSTREAM, "LIKWID MICRO BENCHMARK\n");
        fprintf(OUTSTREAM, "Test: %s\n",test->name);
        fprintf(OUTSTREAM, HLINE);
        fprintf(OUTSTREAM, "Using %d work groups\n",numberOfWorkgroups);
        fprintf(OUTSTREAM, "Using %d threads\n",globalNumberOfThreads);
        fprintf(OUTSTREAM, HLINE);
        fflush(OUTSTREAM);
    }

    threads_create(runTest);
    threads_join();
    allocator_finalize();

    uint32_t realSize = 0;
    uint64_t realCycles = 0;
    int current_id = 0;

    if (OUTSTREAM)
    {
        fprintf(OUTSTREAM, HLINE);
        for(j=0;j<numberOfWorkgroups;j++)
        {
            current_id = j*groups[j].numberOfThreads;
            realCycles += threads_data[current_id].cycles;
            realSize += groups[j].numberOfThreads * threads_data[current_id].data.size;
        }
        time = (double) realCycles / (double) timer_getCpuClock();
        fprintf(OUTSTREAM, "Cycles: %llu \n", LLU_CAST realCycles);
        fprintf(OUTSTREAM, "Iterations: %llu \n", LLU_CAST iter);
        fprintf(OUTSTREAM, "Size %d \n",  realSize );
        fprintf(OUTSTREAM, "Vectorlength: %llu \n", LLU_CAST threads_data[current_id].data.size);
        fprintf(OUTSTREAM, "Time: %e sec\n", time);
        fprintf(OUTSTREAM, "Number of Flops: %llu \n", LLU_CAST (iter * realSize *  test->flops));
        fprintf(OUTSTREAM, "MFlops/s: %.2f\n",
                1.0E-06 * ((double) iter * realSize *  test->flops/  time));
        fprintf(OUTSTREAM, "MByte/s: %.2f\n",
                1.0E-06 * ( (double) iter * realSize *  test->bytes/ time));
        fprintf(OUTSTREAM, "Cycles per update: %f\n",
                ((double) realCycles / (double) (iter * numberOfWorkgroups * threads_data[current_id].numberOfThreads *  threads_data[current_id].data.size)));

        switch ( test->type )
        {
            case SINGLE:
                fprintf(OUTSTREAM, "Cycles per cacheline: %f\n",
                        (16.0 * (double) realCycles / (double) (iter * realSize)));
                break;
            case DOUBLE:
                fprintf(OUTSTREAM, "Cycles per cacheline: %f\n",
                        (8.0 * (double) realCycles / (double) (iter * realSize)));
                break;
        }

        fprintf(OUTSTREAM, HLINE);
        fflush(OUTSTREAM);
    }
    threads_destroy(numberOfWorkgroups);
    barrier_destroy();
    
    affinity_finalize();
#ifdef PERFMON
    likwid_markerClose();
#endif

    return EXIT_SUCCESS;
}

