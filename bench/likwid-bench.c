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
extern void* getIterSingle(void* arg);

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define HELP_MSG printf("Threaded Memory Hierarchy Benchmark --  Version  %d.%d \n\n",VERSION,RELEASE); \
    printf("\n"); \
    printf("Supported Options:\n"); \
    printf("-h\t\t Help message\n"); \
    printf("-a\t\t List available benchmarks \n"); \
    printf("-d\t\t Delimiter used for physical core list (default ,) \n"); \
    printf("-p\t\t List available thread domains\n"); \
    printf("\t\t or the physical ids of the cores selected by the -c expression \n"); \
    printf("-s <TIME>\t Seconds to run the test minimally (default 1)\n");\
    printf("\t\t If resulting iteration count is below 10, it is normalized to 10.\n");\
    printf("-i <ITERS>\t Specify the number of iterations per thread manually. \n"); \
    printf("-l <TEST>\t list properties of benchmark \n"); \
    printf("-t <TEST>\t type of test \n"); \
    printf("-w\t\t <thread_domain>:<size>[:<num_threads>[:<chunk size>:<stride>]-<streamId>:<domain_id>[:<offset>]\n"); \
    printf("\t\t <size> in kB, MB or GB  (mandatory)\n"); \
    printf("\n"); \
    printf("Usage: \n"); \
    printf("# Run the store benchmark on all CPUs of the system with a vector size of 1 GB\n"); \
    printf("likwid-bench -t store -w S0:1GB\n"); \
    printf("# Run the copy benchmark on one CPU at CPU socket 0 with a vector size of 100kB\n"); \
    printf("likwid-bench -t copy -w S0:100kB:1\n"); \
    printf("# Run the copy benchmark on one CPU at CPU socket 0 with a vector size of 100MB but place one stream on CPU socket 1\n"); \
    printf("likwid-bench -t copy -w S0:100MB:1-0:S0,1:S1\n"); \

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
    uint64_t maxCycles = 0;
    uint64_t minCycles = UINT64_MAX;
    uint64_t cyclesClock = 0;
    uint64_t demandIter = 0;
    TimerData itertime;
    Workgroup* currentWorkgroup = NULL;
    Workgroup* groups = NULL;
    uint32_t min_runtime = 1; /* 1s */
    bstring HLINE = bfromcstr("");
    binsertch(HLINE, 0, 80, '-');
    binsertch(HLINE, 80, 1, '\n');
    int (*ownprintf)(const char *format, ...);
    ownprintf = &printf;

    /* Handling of command line options */
    if (argc ==  1)
    {
        HELP_MSG;
        exit(EXIT_SUCCESS);
    }

    while ((c = getopt (argc, argv, "w:t:s:l:aphvi:")) != -1) {
        switch (c)
        {
            case 'h':
                HELP_MSG;
                exit (EXIT_SUCCESS);
            case 'v':
                VERSION_MSG;
                exit (EXIT_SUCCESS);
            case 'a':
                ownprintf(TESTS"\n");
                exit (EXIT_SUCCESS);
            case 'w':
                numberOfWorkgroups++;
                break;
            case 's':
                min_runtime = atoi(optarg);
                break;
            case 'i':
                demandIter = strtoul(optarg, NULL, 10);
                if (demandIter <= 0)
                {
                    fprintf (stderr, "Error: Iterations must be greater than 0\n");
                    return EXIT_FAILURE;
                }
                break;
            case 'l':
                bdestroy(testcase);
                testcase = bfromcstr(optarg);
                for (i=0; i<NUMKERNELS; i++)
                {
                    if (biseqcstr(testcase, kernels[i].name))
                    {
                        test = kernels+i;
                        break;
                    }
                }

                if (test == NULL)
                {
                    fprintf (stderr, "Error: Unknown test case %s\n",optarg);
                    return EXIT_FAILURE;
                }
                else
                {
                    ownprintf("Name: %s\n",test->name);
                    ownprintf("Number of streams: %d\n",test->streams);
                    ownprintf("Loop stride: %d\n",test->stride);
                    ownprintf("Flops: %d\n",test->flops);
                    ownprintf("Bytes: %d\n",test->bytes);
                    switch (test->type)
                    {
                        case SINGLE:
                            ownprintf("Data Type: Single precision float\n");
                            break;
                        case DOUBLE:
                            ownprintf("Data Type: Double precision float\n");
                            break;
                    }
                    if (test->loads >= 0)
                    {
                        ownprintf("Load Ops: %d\n",test->loads);
                    }
                    if (test->stores >= 0)
                    {
                        ownprintf("Store Ops: %d\n",test->stores);
                    }
                    if (test->branches >= 0)
                    {
                        ownprintf("Branches: %d\n",test->branches);
                    }
                    if (test->instr_const >= 0)
                    {
                        ownprintf("Constant instructions: %d\n",test->instr_const);
                    }
                    if (test->instr_loop >= 0)
                    {
                        ownprintf("Loop instructions: %d\n",test->instr_loop);
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

                if (test == NULL)
                {
                    fprintf (stderr, "Error: Unknown test case %s\n",optarg);
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
    if ((numberOfWorkgroups == 0) && (!optPrintDomains))
    {
        fprintf(stderr, "Error: At least one workgroup (-w) must be set on commandline\n");
        exit (EXIT_FAILURE);
    }

    if (topology_init() != EXIT_SUCCESS)
    {
        fprintf(stderr, "Error: Unsupported processor!\n");
        exit(EXIT_FAILURE);
    }

    if ((test == NULL) && (!optPrintDomains))
    {
        fprintf(stderr, "Unknown test case. Please check likwid-bench -a for available tests\n");
        fprintf(stderr, "and select one using the -t commandline option\n");
        exit(EXIT_FAILURE);
    }

    numa_init();
    affinity_init();
    timer_init();

    if (optPrintDomains)
    {
        AffinityDomains_t affinity = get_affinityDomains();
        ownprintf("Number of Domains %d\n",affinity->numberOfAffinityDomains);
        for (i=0; i < affinity->numberOfAffinityDomains; i++ )
        {
            ownprintf("Domain %d:\n",i);
            ownprintf("\tTag %s:",bdata(affinity->domains[i].tag));

            for ( uint32_t j=0; j < affinity->domains[i].numberOfProcessors; j++ )
            {
                ownprintf(" %d",affinity->domains[i].processorList[j]);
            }
            ownprintf("\n");
        }
        exit (EXIT_SUCCESS);
    }

    allocator_init(numberOfWorkgroups * MAX_STREAMS);
    groups = (Workgroup*) malloc(numberOfWorkgroups*sizeof(Workgroup));
    tmp = 0;

    optind = 0;
    while ((c = getopt (argc, argv, "w:t:s:l:i:aphv")) != -1)
    {
        switch (c)
        {
            case 'w':
                currentWorkgroup = groups+tmp;
                bstring groupstr = bfromcstr(optarg);
                i = bstr_to_workgroup(currentWorkgroup, groupstr, test->type, test->streams);
                bdestroy(groupstr);
                if (i == 0)
                {
                    for (i=0; i<  test->streams; i++)
                    {
                        if (currentWorkgroup->streams[i].offset%test->stride)
                        {
                            fprintf (stderr, "Error: Stream %d: offset is not a multiple of stride!\n",i);
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
                }
                else
                {
                    exit(EXIT_FAILURE);
                }
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
    ownprintf("LIKWID MICRO BENCHMARK\n");
    ownprintf("Test: %s\n",test->name);
    ownprintf(bdata(HLINE));
    ownprintf("Using %" PRIu64 " work groups\n",numberOfWorkgroups);
    ownprintf("Using %d threads\n",globalNumberOfThreads);
    ownprintf(bdata(HLINE));


    threads_init(globalNumberOfThreads);
    threads_createGroups(numberOfWorkgroups);

    /* we configure global barriers only */
    barrier_init(1);
    barrier_registerGroup(globalNumberOfThreads);
    cyclesClock = timer_getCycleClock();

#ifdef LIKWID_PERFMON
    if (getenv("LIKWID_FILEPATH") != NULL)
    {
        ownprintf("Using Likwid Marker API\n");
    }
    LIKWID_MARKER_INIT;
    ownprintf(bdata(HLINE));
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
        myData.cycles = 0;
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
        getIterSingle((void*) &threads_data[0]);
        for (i=0; i<numberOfWorkgroups; i++)
        {
            iter = threads_updateIterations(i, demandIter);
        }
    }
#ifdef DEBUG_LIKWID
    else
    {
        ownprintf("Using manually selected iterations per thread\n");
    }
#endif

    threads_create(runTest);
    threads_join();

    for (int i=0; i<globalNumberOfThreads; i++)
    {
        realSize += threads_data[i].data.size;
        realIter += threads_data[i].data.iter;
        if (threads_data[i].cycles > maxCycles)
        {
            maxCycles = threads_data[i].cycles;
        }
        if (threads_data[i].cycles < minCycles)
        {
            minCycles = threads_data[i].cycles;
        }
    }



    time = (double) maxCycles / (double) cyclesClock;
    ownprintf(bdata(HLINE));
    ownprintf("Cycles:\t\t\t%" PRIu64 "\n", maxCycles);
    ownprintf("CPU Clock:\t\t%" PRIu64 "\n", timer_getCpuClock());
    ownprintf("Cycle Clock:\t\t%" PRIu64 "\n", cyclesClock);
    ownprintf("Time:\t\t\t%e sec\n", time);
    ownprintf("Iterations:\t\t%" PRIu64 "\n", realIter);
    ownprintf("Iterations per thread:\t%" PRIu64 "\n",threads_data[0].data.iter);
    ownprintf("Inner loop executions:\t%.0f\n", ((double)realSize)/((double)test->stride));
    ownprintf("Size:\t\t\t%" PRIu64 "\n",  realSize*test->bytes );
    ownprintf("Size per thread:\t%" PRIu64 "\n", threads_data[0].data.size*test->bytes);
    ownprintf("Number of Flops:\t%" PRIu64 "\n", (threads_data[0].data.iter * realSize *  test->flops));
    ownprintf("MFlops/s:\t\t%.2f\n",
            1.0E-06 * ((double) threads_data[0].data.iter * realSize *  test->flops/  time));
    
    ownprintf("Data volume (Byte):\t%llu\n", LLU_CAST (threads_data[0].data.iter * realSize *  test->bytes));
    ownprintf("MByte/s:\t\t%.2f\n",
            1.0E-06 * ( (double) threads_data[0].data.iter * realSize *  test->bytes/ time));

    cycPerUp = ((double) maxCycles / (double) (threads_data[0].data.iter * realSize));
    ownprintf("Cycles per update:\t%f\n", cycPerUp);

    switch ( test->type )
    {
        case SINGLE:
            ownprintf("Cycles per cacheline:\t%f\n", (16.0 * cycPerUp));
            break;
        case DOUBLE:
            ownprintf("Cycles per cacheline:\t%f\n", (8.0 * cycPerUp));
            break;
    }
    ownprintf("Loads per update:\t%" PRIu64 "\n", test->loads );
    ownprintf("Stores per update:\t%" PRIu64 "\n", test->stores );
    if ((test->loads > 0) && (test->stores > 0))
    {
        ownprintf("Load/store ratio:\t%.2f\n", ((double)test->loads)/((double)test->stores) );
    }
    if ((test->instr_loop > 0) && (test->instr_const > 0))
    {
        ownprintf("Instructions:\t\t%" PRIu64 "\n", LLU_CAST ((double)realSize/test->stride)*test->instr_loop*threads_data[0].data.iter + test->instr_const );
    }
    if (test->uops > 0)
    {
        ownprintf("UOPs:\t\t\t%" PRIu64 "\n", LLU_CAST ((double)realSize/test->stride)*test->uops*threads_data[0].data.iter);
    }

    ownprintf(bdata(HLINE));
    threads_destroy(numberOfWorkgroups);
    allocator_finalize();

#ifdef LIKWID_PERFMON
    if (getenv("LIKWID_FILEPATH") != NULL)
    {
        ownprintf("Writing Likwid Marker API results to file %s\n", getenv("LIKWID_FILEPATH"));
    }
    LIKWID_MARKER_CLOSE;
#endif

    return EXIT_SUCCESS;
}

