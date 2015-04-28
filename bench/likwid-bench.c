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
 *      Copyright (C) 2013 Jan Treibig 
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
#include <inttypes.h>

#include <bstrlib.h>
#include <errno.h>
#include <threads.h>
#include <barrier.h>
#include <testcases.h>
#include <strUtil.h>
#include <allocator.h>

#include <likwid.h>

extern void* runTest(void* arg);
extern void* getIter(void* arg);

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define HELP_MSG \
    printf("Threaded Memory Hierarchy Benchmark --  Version  %d.%d \n\n",VERSION,RELEASE); \
printf("\n"); \
printf("Supported Options:\n"); \
printf("-h\t Help message\n"); \
printf("-a\t List available benchmarks \n"); \
printf("-d\t Delimiter used for physical core list (default ,) \n"); \
printf("-p\t List available thread domains\n\t or the physical ids of the cores selected by the -c expression \n"); \
printf("-s <TIME>\t Seconds to run the test minimally (default 1)\n");\
printf("\t If resulting iteration count is below 10, it is normalized to 10.\n");\
printf("-l <TEST>\t list properties of benchmark \n"); \
printf("-t <TEST>\t type of test \n"); \
printf("-w\t <thread_domain>:<size>[:<num_threads>[:<chunk size>:<stride>]-<streamId>:<domain_id>[:<offset>], size in kB, MB or GB  (mandatory)\n"); \
printf("\n"); \
printf("Usage: likwid-bench -t copy -w S0:100kB:1 \n")

#define VERSION_MSG \
    printf("likwid-bench   %d.%d \n\n",VERSION,RELEASE)

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
    uint64_t iter = 100;
    uint32_t i;
    uint32_t j;
    int globalNumberOfThreads = 0;
    int optPrintDomains = 0;
    int c;
    ThreadUserData myData;
    bstring testcase = bfromcstr("none");
    uint64_t numberOfWorkgroups = 0;
    int tmp = 0;
    double time;
    double cycPerUp = 0.0;
    const TestCase* test = NULL;
    uint64_t realSize = 0;
    uint64_t realIter = 0;
    uint64_t cpuClock = 0;
    uint64_t demandIter = 0;
    TimerData itertime;
    Workgroup* currentWorkgroup = NULL;
    Workgroup* groups = NULL;
    uint32_t min_runtime = 1; /* 1s */
    bstring HLINE;
    binsertch(HLINE, 0, 80, '-');
    int (*ownprintf)(const char *format, ...);
    ownprintf = &printf;

    /* Handling of command line options */
    if (argc ==  1)
    {
        HELP_MSG;
        exit(EXIT_SUCCESS);
    }

    while ((c = getopt (argc, argv, "w:t:s:l:aphv")) != -1) {
        switch (c)
        {
            case 'h':
                HELP_MSG;
                exit (EXIT_SUCCESS);
            case 'v':
                VERSION_MSG;
                exit (EXIT_SUCCESS);
            case 'a':
                printf(TESTS"\n");
                exit (EXIT_SUCCESS);
            case 'w':
                numberOfWorkgroups++;
                break;
            case 's':
                min_runtime = atoi(optarg);
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
                numberOfWorkgroups = LLU_CAST atol(optarg);
                
                tmp = numberOfWorkgroups;
                
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
                HELP_MSG;
        }
    }


    if (topology_init() != EXIT_SUCCESS)
    {
        fprintf(stderr, "Unsupported processor!\n");
        exit(EXIT_FAILURE);
    }
    numa_init();
    affinity_init();
    timer_init();

    allocator_init(numberOfWorkgroups * MAX_STREAMS);
    groups = (Workgroup*) malloc(numberOfWorkgroups*sizeof(Workgroup));
    tmp = 0;

    if (optPrintDomains)
    {
        AffinityDomains_t affinity = get_affinityDomains();
        printf("Number of Domains %d\n",affinity->numberOfAffinityDomains);
        for (i=0; i < affinity->numberOfAffinityDomains; i++ )
        {
            printf("Domain %d:\n",i);
            printf("\tTag %s:",bdata(affinity->domains[i].tag));

            for ( uint32_t j=0; j < affinity->domains[i].numberOfProcessors; j++ )
            {
                printf(" %d",affinity->domains[i].processorList[j]);
            }
            printf("\n");
        }
        exit (EXIT_SUCCESS);
    }

    optind = 0;
    while ((c = getopt (argc, argv, "w:t:s:l:i:aphv")) != -1) 
    {
        switch (c)
        {
            case 'w':
                currentWorkgroup = groups+tmp;
                testcase = bfromcstr(optarg);
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
                tmp++;
                break;
            default:
                continue;
                break;
        }
    }

    /* :WARNING:05/04/2010 08:58:05 AM:jt: At the moment the thread
     * module only allows equally sized thread groups*/
    for (i=0; i<numberOfWorkgroups; i++)
    {
        globalNumberOfThreads += groups[i].numberOfThreads;
    }

    ownprintf(bdata(HLINE));
    printf("LIKWID MICRO BENCHMARK\n");
    printf("Test: %s\n",test->name);
    ownprintf(bdata(HLINE));
    printf("Using %" PRIu64 " work groups\n",numberOfWorkgroups);
    printf("Using %d threads\n",globalNumberOfThreads);
    ownprintf(bdata(HLINE));


    threads_init(globalNumberOfThreads);
    threads_createGroups(numberOfWorkgroups);

    /* we configure global barriers only */
    barrier_init(1);
    barrier_registerGroup(globalNumberOfThreads);
    cpuClock = timer_getCpuClock();

#ifdef PERFMON
    //printf("Using likwid\n");
    likwid_markerInit();
#endif


    /* initialize data structures for threads */
    for (i=0; i<numberOfWorkgroups; i++)
    {
        myData.iter = iter;
        if (demandIter > 0)
        {
            myData.iter = demandIter;
        }
        myData.min_runtime = min_runtime;
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

    if (demandIter == 0)
    {
        threads_create(getIter);
        threads_join();
    }
    for (i=0; i<numberOfWorkgroups; i++)
    {
        iter = threads_updateIterations(i, demandIter);
    }
    threads_create(runTest);
    threads_join();
    allocator_finalize();

    for (int i=0; i<globalNumberOfThreads; i++)
    {
        realSize += threads_data[i].data.size;
        realIter += threads_data[i].data.iter;
    }



    time = (double) threads_data[0].cycles / (double) cpuClock;
    ownprintf(bdata(HLINE));
    printf("Cycles:\t\t\t%llu\n", LLU_CAST threads_data[0].cycles);
    printf("Iterations:\t\t%llu\n", LLU_CAST realIter);
    printf("Iterations per thread:\t%llu\n",LLU_CAST threads_data[0].data.iter);
    printf("Size:\t\t\t%" PRIu64 "\n",  realSize*test->bytes );
    printf("Size per thread:\t%llu\n", LLU_CAST threads_data[0].data.size*test->bytes);
    printf("Time:\t\t\t%e sec\n", time);
    printf("Number of Flops:\t%llu\n", LLU_CAST (numberOfWorkgroups * iter * realSize *  test->flops));
    printf("MFlops/s:\t\t%.2f\n",
            1.0E-06 * ((double) numberOfWorkgroups * iter * realSize *  test->flops/  time));
    printf("Data volume (Byte):\t%llu\n", LLU_CAST (numberOfWorkgroups * iter * realSize *  test->bytes));
    printf("MByte/s:\t\t%.2f\n",
            1.0E-06 * ( (double) numberOfWorkgroups * iter * realSize *  test->bytes/ time));
    cycPerUp = ((double) threads_data[0].cycles / (double) (threads_data[0].data.iter * threads_data[0].data.size));
    printf("Cycles per update:\t%f\n", cycPerUp);

    switch ( test->type )
    {
        case SINGLE:
            printf("Cycles per cacheline:\t%f\n", (16.0 * cycPerUp));
            break;
        case DOUBLE:
            printf("Cycles per cacheline:\t%f\n", (8.0 * cycPerUp));
            break;
    }

    ownprintf(bdata(HLINE));
    threads_destroy(numberOfWorkgroups);
    
#ifdef PERFMON
    likwid_markerClose();
#endif

    return EXIT_SUCCESS;
}

