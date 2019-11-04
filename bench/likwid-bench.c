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
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>

#include <bstrlib.h>
#include <errno.h>
#include <threads.h>
#include <barrier.h>
#include <testcases.h>
#include <strUtil.h>
#include <allocator.h>
#include <ptt2asm.h>

#include <likwid.h>
#include <likwid-marker.h>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

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
    printf("-W\t\t <thread_domain>:<size>[:<num_threads>[:<chunk size>:<stride>]]\n"); \
    printf("\t\t <size> in kB, MB or GB (mandatory)\n"); \
    printf("For dynamically loaded benchmarks\n"); \
    printf("-f <PATH>\t Specify a folder for the temporary files. default: /tmp\n"); \
    printf("\n"); \
    printf("Difference between -w and -W :\n"); \
    printf("-w allocates the streams in the thread_domain with one thread and support placement of streams\n"); \
    printf("-W allocates the streams chunk-wise by each thread in the thread_domain\n"); \
    printf("\n"); \
    printf("Usage: \n"); \
    printf("# Run the store benchmark on all CPUs of the system with a vector size of 1 GB\n"); \
    printf("likwid-bench -t store -w S0:1GB\n"); \
    printf("# Run the copy benchmark on one CPU at CPU socket 0 with a vector size of 100kB\n"); \
    printf("likwid-bench -t copy -w S0:100kB:1\n"); \
    printf("# Run the copy benchmark on one CPU at CPU socket 0 with a vector size of 100MB but place one stream on CPU socket 1\n"); \
    printf("likwid-bench -t copy -w S0:100MB:1-0:S0,1:S1\n"); \
/*    printf("-c <COMP_LIST>\t Specify a list of compilers that should be searched for. default: gcc,icc,pgcc\n"); \*/
/*    printf("-f <COMP_FLAGS>\t Specify compiler flags. Use \". default: \"-shared -fPIC\"\n"); \*/

#define VERSION_MSG \
    printf("likwid-bench -- Version %d.%d.%d\n",VERSION,RELEASE,MINORVERSION); \

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


void illhandler(int signum, siginfo_t *info, void *ptr)
{
    fprintf(stderr, "ERROR: Illegal instruction\n");
    fprintf(stderr, "This happens if you want to run a kernel that uses instructions not available on your system.\n");
    exit(EXIT_FAILURE);
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
    double cycPerCL = 0.0;
    TestCase* test = NULL;
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
    int clsize = sysconf (_SC_LEVEL1_DCACHE_LINESIZE);
    char compilers[512] = "gcc,icc,pgcc";
    char defcompilepath[512] = "/tmp";
    char compilepath[513] = "";
    char compileflags[512] = "-shared -fPIC";
    ownprintf = &printf;
    struct sigaction sig;

    /* Handling of command line options */
    if (argc ==  1)
    {
        HELP_MSG;
        exit(EXIT_SUCCESS);
    }

    while ((c = getopt (argc, argv, "W:w:t:s:l:aphvi:f:")) != -1) {
        switch (c)
        {
            case 'f':
                tmp = snprintf(compilepath, 512, "%s", optarg);
                if (tmp > 0)
                {
                    compilepath[tmp] = '\0';
                }
                break;
            default:
                break;
        }
    }
    optind = 0;

    while ((c = getopt (argc, argv, "W:w:t:s:l:aphvi:f:")) != -1) {
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

                struct bstrList* l = dynbench_getall();
                if (l)
                {
                    ownprintf("\nUser benchmarks:\n");
                    for (i = 0; i < l->qty; i++)
                    {
                        if (dynbench_test(l->entry[i]))
                        {
                            TestCase* t = NULL;
                            int err = dynbench_load(l->entry[i], &t, NULL, NULL, NULL);
                            if (!err && t)
                            {
                                printf("%s - %s\n", t->name, t->desc);
                                dynbench_close(t, NULL);
                            }
                        }
                    }
                    bstrListDestroy(l);
                }
                exit (EXIT_SUCCESS);
            case 'w':
            case 'W':
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
                int builtin = 1;
                for (i=0; i<NUMKERNELS; i++)
                {
                    if (biseqcstr(testcase, kernels[i].name))
                    {
                        test = (TestCase*)kernels+i;
                        break;
                    }
                }

                if (test == NULL && dynbench_test(testcase))
                {
                    dynbench_load(testcase, &test, NULL, NULL, NULL);
                    builtin = 0;
                }

                if (test == NULL)
                {
                    fprintf (stderr, "Error: Unknown test case %s\n",optarg);
                    return EXIT_FAILURE;
                }
                else
                {
                    ownprintf("Name: %s\n",test->name);
                    ownprintf("Description: %s\n",test->desc);
                    ownprintf("Number of streams: %d\n",test->streams);
                    ownprintf("Loop stride: %d\n",test->stride);
                    switch (test->type)
                    {
                        case INT:
                            ownprintf("Data Type: Integer\n");
                            break;
                        case SINGLE:
                            ownprintf("Data Type: Single precision float\n");
                            break;
                        case DOUBLE:
                            ownprintf("Data Type: Double precision float\n");
                            break;
                    }
                    ownprintf("Flops per element: %d\n",test->flops);
                    ownprintf("Bytes per element: %d\n",test->bytes);
                    if (test->loads > 0 && test->stores > 0)
                    {
                        double ratio = (double)test->loads/(double)(test->stores+test->loads);
                        double load_bytes = ((double)test->bytes) * ratio;
                        ownprintf("Load bytes per element: %.0f\n", load_bytes);
                        ownprintf("Store bytes per element: %.0f\n",((double)test->bytes) - load_bytes);
                    }
                    else if (test->loads >= 0 && test->stores == 0)
                    {
                        ownprintf("Load bytes per element: %d\n",test->bytes);
                    }
                    else if (test->loads == 0 && test->stores > 0)
                    {
                        ownprintf("Store bytes per element: %d\n",test->bytes);
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
                    if (test->uops >= 0)
                    {
                        ownprintf("Loop micro Ops (\u03BCOPs): %d\n",test->uops);
                    }
                }
                bdestroy(testcase);
                if (!builtin)
                {
                    dynbench_close(test, NULL);
                }
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
                bdestroy(testcase);
                testcase = bfromcstr(optarg);

                for (i=0; i<NUMKERNELS; i++)
                {
                    if (biseqcstr(testcase, kernels[i].name))
                    {
                        test = (TestCase*)kernels+i;
                        break;
                    }
                }

                if (test == NULL && dynbench_test(testcase))
                {
                    if (strlen(compilepath) == 0)
                    {
                        int ret = snprintf(compilepath, 512, "%s", defcompilepath);
                        if (ret > 0) compilepath[ret] = '\0';
                    }
                    dynbench_load(testcase, &test, compilepath, compilers, compileflags);
                }

                if (test == NULL)
                {
                    fprintf (stderr, "Error: Unknown test case %s\n",optarg);
                    return EXIT_FAILURE;
                }
                bdestroy(testcase);
                break;
            case 'f':
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

    memset(&sig, 0, sizeof(struct sigaction));
    sig.sa_sigaction = illhandler;
    sig.sa_flags = SA_SIGINFO;
    sigaction(SIGILL, &sig, NULL);

    if (optPrintDomains)
    {
        bdestroy(testcase);
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
    memset(groups, 0, numberOfWorkgroups*sizeof(Workgroup));
    tmp = 0;

    optind = 0;
    while ((c = getopt (argc, argv, "W:w:t:s:l:i:aphv")) != -1)
    {
        switch (c)
        {
            case 'w':
            case 'W':
                currentWorkgroup = groups+tmp;
                bstring groupstr = bfromcstr(optarg);
                if (c == 'W')
                {
                    currentWorkgroup->init_per_thread = 1;
                }
                i = bstr_to_workgroup(currentWorkgroup, groupstr, test->type, test->streams);
                bdestroy(groupstr);
                size_t newsize = 0;
                size_t stride = test->stride;
                int nrThreads = currentWorkgroup->numberOfThreads;
                size_t orig_size = currentWorkgroup->size;
                if (i == 0)
                {
                    int warn_once = 1;
                    for (i=0; i<  test->streams; i++)
                    {
                        if (currentWorkgroup->streams[i].offset%test->stride)
                        {
                            fprintf (stderr, "Error: Stream %d: offset is not a multiple of stride!\n",i);
                            return EXIT_FAILURE;
                        }
                        if ((int)(floor(orig_size/currentWorkgroup->numberOfThreads)) % test->stride)
                        {
                            int typesize = allocator_dataTypeLength(test->type);
                            newsize = (((int)(floor(orig_size/nrThreads))/stride)*(stride))*nrThreads;
                            if (newsize > 0 && warn_once)
                            {
                                fprintf (stderr, "Warning: Sanitizing vector length to a multiple of the loop stride %d and thread count %d from %d elements (%d bytes) to %d elements (%d bytes)\n",stride, nrThreads, orig_size, orig_size*typesize, newsize, newsize*typesize);
                                warn_once = 0;
                            }
                            else if (newsize == 0)
                            {
                                int given = currentWorkgroup->size*test->streams*typesize;
                                int each_iter = test->stride*test->bytes;
                                // For the case that one stream is used for loading and storing
                                // Cases are daxpy and update
                                if (test->streams*typesize*test->stride < each_iter)
                                {
                                    each_iter = test->streams*typesize*test->stride;
                                }
                                fprintf(stderr, "Error: The given vector length of %dB is too small to fit %d threads because each loop iteration of kernel '%s' requires %d Bytes (%d x %dB = %dB). So the minimal selectable size for the kernel is %dB.\n", given, nrThreads, test->name, each_iter, nrThreads, each_iter, each_iter*nrThreads, each_iter*nrThreads);
                                allocator_finalize();
                                workgroups_destroy(&groups, numberOfWorkgroups, test->streams);
                                exit(EXIT_FAILURE);
                            }
                        }
                        else
                        {
                            newsize = orig_size;
                        }
                        allocator_allocateVector(&(currentWorkgroup->streams[i].ptr),
                                                    PAGE_ALIGNMENT,
                                                    newsize,
                                                    currentWorkgroup->streams[i].offset,
                                                    test->type,
                                                    test->stride,
                                                    currentWorkgroup->streams[i].domain,
                                                    currentWorkgroup->init_per_thread && nrThreads > 1);
                    }
                    tmp++;
                }
                else
                {
                    exit(EXIT_FAILURE);
                }
                if (newsize != currentWorkgroup->size)
                {
                    currentWorkgroup->size = newsize;
                }
                if (nrThreads > 1)
                {
                    if (currentWorkgroup->init_per_thread)
                    {
                        printf("Initialization: Each thread in domain initializes its own stream chunks\n");
                    }
                    else
                    {
                        printf("Initialization: First thread in domain initializes the whole stream\n");
                    }
                }
                break;
            default:
                continue;
                break;
        }
    }
    if (numberOfWorkgroups > 1)
    {
        int g0_numberOfThreads = groups[0].numberOfThreads;
        int g0_size = groups[0].size;
        for (i = 1; i < numberOfWorkgroups; i++)
        {
            if (g0_numberOfThreads != groups[i].numberOfThreads)
            {
                fprintf (stderr, "Warning: Multiple workgroups with different thread counts are not recommended. Use with case!\n");
                break;
            }
        }
        for (i = 1; i < numberOfWorkgroups; i++)
        {
            if (g0_size != groups[i].size)
            {
                fprintf (stderr, "Warning: Multiple workgroups with different sizes are not recommended. Use with case!\n");
                break;
            }
        }
    }

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
    threads_createGroups(numberOfWorkgroups, groups);

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
        myData.init_per_thread = groups[i].init_per_thread;
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

    timer_start(&itertime);
    threads_create(runTest);
    threads_join();
    timer_stop(&itertime);

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

    if (cyclesClock > 0)
    {
        time = (double) maxCycles / (double) cyclesClock;
    }
    else
    {
        time = timer_print(&itertime);
    }

/*#if defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_8A)*/
/*    if (maxCycles > 0)*/
/*    {*/
/*        ownprintf("WARNING: The cycle based values are calculated using the current but fixed CPU frequency on ARMv7/8\n");*/
/*    }*/
/*    else*/
/*    {*/
/*        ownprintf(bdata(HLINE));*/
/*        ownprintf("WARNING: All cycle based values will be zero on ARMv7!\n");*/
/*        ownprintf("WARNING: The cycle count cannot be calculated because the clock frequency is not fixed.\n");*/
/*    }*/
/*#endif*/
    int datatypesize = allocator_dataTypeLength(test->type);
    uint64_t size_per_thread = threads_data[0].data.size;
    uint64_t iters_per_thread = threads_data[0].data.iter;
    uint64_t datavol = iters_per_thread * realSize * test->bytes;
    uint64_t allIters = realIter * (int)(((double)realSize)/((double)test->stride*globalNumberOfThreads));
    ownprintf(bdata(HLINE));
    ownprintf("Cycles:\t\t\t%" PRIu64 "\n", maxCycles);
    ownprintf("CPU Clock:\t\t%" PRIu64 "\n", timer_getCpuClock());
    ownprintf("Cycle Clock:\t\t%" PRIu64 "\n", cyclesClock);
    ownprintf("Time:\t\t\t%e sec\n", time);
    ownprintf("Iterations:\t\t%" PRIu64 "\n", realIter);
    ownprintf("Iterations per thread:\t%" PRIu64 "\n",iters_per_thread);
    ownprintf("Inner loop executions:\t%d\n", (int)(((double)realSize)/((double)test->stride*globalNumberOfThreads)));
    ownprintf("Size (Byte):\t\t%" PRIu64 "\n",  realSize * datatypesize * test->streams);
    ownprintf("Size per thread:\t%" PRIu64 "\n", size_per_thread * datatypesize * test->streams);
    ownprintf("Number of Flops:\t%" PRIu64 "\n", (iters_per_thread * realSize *  test->flops));
    ownprintf("MFlops/s:\t\t%.2f\n",
            1.0E-06 * ((double) (iters_per_thread * realSize *  test->flops) /  time));
    ownprintf("Data volume (Byte):\t%llu\n",
            LLU_CAST (datavol));
    ownprintf("MByte/s:\t\t%.2f\n",
            1.0E-06 * ( (double) (iters_per_thread * realSize * test->bytes) / time));

    size_t destsize = 0;
    size_t datasize = 0;
    double perUpFactor = 0.0;
    switch (test->type)
    {
        case INT:
            datasize = test->bytes/sizeof(int);
            destsize = test->bytes/test->streams;
            perUpFactor = (clsize/sizeof(int));
            break;
        case SINGLE:
            datasize = test->bytes/sizeof(float);
            destsize = test->bytes/test->streams;
            perUpFactor = (clsize/sizeof(float));
            break;
        case DOUBLE:
            datasize = test->bytes/sizeof(double);
            destsize = test->bytes/test->streams;
            perUpFactor = (clsize/sizeof(double));
            break;
    }

    cycPerCL = (double) maxCycles/((double)datavol/(clsize*datasize));
    ownprintf("Cycles per update:\t%f\n", cycPerCL/perUpFactor);
    ownprintf("Cycles per cacheline:\t%f\n", cycPerCL);
    ownprintf("Loads per update:\t%ld\n", test->loads );
    ownprintf("Stores per update:\t%ld\n", test->stores );
    if (test->loads > 0 && test->stores > 0)
    {
        double ratio = (double)test->loads/(double)(test->stores+test->loads);
        double load_bytes = ((double)test->bytes) * ratio;
        ownprintf("Load bytes per element:\t%.0f\n", load_bytes);
        ownprintf("Store bytes per elem.:\t%.0f\n",((double)test->bytes) - load_bytes);
    }
    else if (test->loads >= 0 && test->stores == 0)
    {
        ownprintf("Load bytes per element:\t%d\n",test->bytes);
        ownprintf("Store bytes per elem.:\t0\n");
    }
    else if (test->loads == 0 && test->stores > 0)
    {
        ownprintf("Load bytes per element:\t0\n");
        ownprintf("Store bytes per elem.:\t%d\n",test->bytes);
    }
    if ((test->loads > 0) && (test->stores > 0))
    {
        ownprintf("Load/store ratio:\t%.2f\n", ((double)test->loads)/((double)test->stores) );
    }
    if ((test->instr_loop > 0) && (test->instr_const > 0))
    {
        ownprintf("Instructions:\t\t%" PRIu64 "\n",
                LLU_CAST ((double)realSize/test->stride)*test->instr_loop*threads_data[0].data.iter + test->instr_const );
    }
    if (test->uops > 0)
    {
        ownprintf("UOPs:\t\t\t%" PRIu64 "\n",
                LLU_CAST ((double)realSize/test->stride)*test->uops*threads_data[0].data.iter);
    }

    ownprintf(bdata(HLINE));
    threads_destroy(numberOfWorkgroups, test->streams);
    allocator_finalize();
    workgroups_destroy(&groups, numberOfWorkgroups, test->streams);

#ifdef LIKWID_PERFMON
    if (getenv("LIKWID_FILEPATH") != NULL)
    {
        ownprintf("Writing Likwid Marker API results to file %s\n", getenv("LIKWID_FILEPATH"));
    }
    LIKWID_MARKER_CLOSE;
#endif

    if (test->dlhandle != NULL)
    {
        dynbench_close(test, compilepath);
    }

    bdestroy(HLINE);
    return EXIT_SUCCESS;
}
