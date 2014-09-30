/*
 * =======================================================================================
 *
 *      Filename:  likwid-pin.c
 *
 *      Description:  An application to pin a program including threads
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
/* #####   HEADER FILE INCLUDES   ######################################### */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sched.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>

#include <error.h>
#include <types.h>
#include <bstrlib.h>
#include <cpuid.h>
#include <affinity.h>
#include <numa.h>
#include <memsweep.h>
#include <strUtil.h>

#ifdef COLOR
#include <textcolor.h>
#endif

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */
#define HELP_MSG \
    fprintf(stdout, "likwid-pin --  Version %d.%d \n\n",VERSION,RELEASE); \
    fprintf(stdout, "\n"); \
    fprintf(stdout, "Supported Options:\n"); \
    fprintf(stdout, "-h\t Help message\n"); \
    fprintf(stdout, "-v\t Version information\n"); \
    fprintf(stdout, "-i\t Set numa interleave policy with all involved numa nodes\n"); \
    fprintf(stdout, "-S\t Sweep memory in involved numa nodes\n"); \
    fprintf(stdout, "-c\t comma separated list of processor ids or expression\n"); \
    fprintf(stdout, "-s\t bitmask with threads to skip\n"); \
    fprintf(stdout, "-p\t Print available domains with mapping on physical ids\n"); \
    fprintf(stdout, "  \t If used together with -c option outputs a physical processor ids.\n"); \
    fprintf(stdout, "-d\t Delimiter used for using -p to output physical processor list, default is comma.\n\n"); \
    fprintf(stdout, "-q\t Silent without output\n\n"); \
    fprintf(stdout, "There are three possibilities to provide a thread to processor list:\n\n"); \
    fprintf(stdout, "1. Thread list with physical or logical thread numberings and physical cores first.\n"); \
    fprintf(stdout, "Example usage thread list: likwid-pin -c N:0,4-6 ./myApp\n"); \
    fprintf(stdout, "You can pin with the following numberings:\n");  \
    fprintf(stdout, "\t1. Physical numbering of OS.\n");  \
    fprintf(stdout, "\t2. Logical numbering inside node. e.g. -c N:0-3\n");  \
    fprintf(stdout, "\t3. Logical numbering inside socket. e.g. -c S0:0-3\n");  \
    fprintf(stdout, "\t4. Logical numbering inside last level cache group. e.g. -c C0:0-3\n");  \
    fprintf(stdout, "\t5. Logical numbering inside NUMA domain. e.g. -c M0:0-3\n");  \
    fprintf(stdout, "\tYou can also mix domains separated by  @, e.g. -c S0:0-3@S1:0-3 \n\n");  \
    fprintf(stdout, "2. Expressions based thread list generation with compact processor numbering.\n"); \
    fprintf(stdout, "Example usage expression: likwid-pin -c E:N:8 ./myApp\n"); \
    fprintf(stdout, "This will generate a compact list of thread to processor mapping for the node domain with eight threads.\n");  \
    fprintf(stdout, "The following syntax variants are available:\n");  \
    fprintf(stdout, "\t1. -c E:<thread domain>:<number of threads>\n");  \
    fprintf(stdout, "\t2. -c E:<thread domain>:<number of threads>:<chunk size>:<stride>\n");  \
    fprintf(stdout, "\t   For two SMT threads per core on a SMT 4 machine use e.g. -c E:N:122:2:4\n\n");  \
    fprintf(stdout, "3. Scatter policy among thread domain type.\n"); \
    fprintf(stdout, "Example usage scatter: likwid-pin -c M:scatter ./myApp\n"); \
    fprintf(stdout, "This will generate a thread to processor mapping scattered among all memory domains with physical cores first.\n\n");  \
    fprintf(stdout, "If you ommit the -c option likwid will use all processors available on the node\n"); \
    fprintf(stdout, "with physical cores first. likwid-pin will also set OMP_NUM_THREADS with as many\n"); \
    fprintf(stdout, "threads as specified in your pin expression if OMP_NUM_THREADS is not present\n"); \
    fprintf(stdout, "in your environment.\n\n"); \
    fflush(stdout);

#define VERSION_MSG \
    fprintf(stdout, "likwid-pin   %d.%d \n\n",VERSION,RELEASE); \
    fflush(stdout);

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */
    static void
pinPid(int cpuid, int silent)
{
    int status;
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET(cpuid, &cpuset);

    status = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

    if (status == -1) 
    {
        fprintf(stderr, "sched_setaffinity failed : %s \n",strerror(errno));
    }
    else 
    {
        if(!silent)
        {
#ifdef COLOR
            color_on(BRIGHT, COLOR);
#endif
            fprintf(stdout, "[likwid-pin] Main PID -> core %d - OK",  cpuid);
#ifdef COLOR
            color_reset();
#endif
            fprintf(stdout, "\n");
            fflush(stdout);
        }
    }
}


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int main (int argc, char** argv)
{
    int i;
    int c;
    int skipMask = -1;
    int optInterleaved = 0;
    int optMemSweep = 0;
    int optPrintDomains = 0;
    int optSilent = 0;
    int hasAffinity = 0;
    bstring  pinString;
    bstring  skipString;
    bstring  argString;
    int numThreads=0;
    int threads[MAX_NUM_THREADS];
    char delimiter = ',';
    threads[0] = 0;

    if (argc ==  1) {
        HELP_MSG;
        exit (EXIT_SUCCESS);
    }

    if (cpuid_init() == EXIT_SUCCESS)
    {
        numa_init();
        affinity_init();
        hasAffinity = 1;
    }

    while ((c = getopt (argc, argv, "+c:d:hipqs:Sv")) != -1)
    {
        switch (c)
        {
            case 'c':
                CHECK_OPTION_STRING;
                if (hasAffinity)
                {
                    numThreads = bstr_to_cpuset(threads, argString);
                }
                else
                {
                    numThreads = bstr_to_cpuset_physical((uint32_t*) threads, argString);
                }

                if(!numThreads)
                {
                    ERROR_PLAIN_PRINT(Failed to parse cpu list.);
                }
                break;
            case 'd':
                delimiter = optarg[0];
                break;
            case 'h':
                HELP_MSG;
                exit (EXIT_SUCCESS);
            case 'i':
                optInterleaved = 1;
                break;
            case 'p':
                if (!hasAffinity)
                {
                    fprintf(stderr, "Option -p is not supported for unknown processor!\n");
                    exit(EXIT_SUCCESS);
                }
                optPrintDomains = 1;
                break;
            case 'q':
                optSilent = 1;
                setenv("LIKWID_SILENT","true", 1);
                break;
            case 's':
                CHECK_OPTION_STRING;
                skipMask = strtoul((char*) argString->data,NULL,16);
                break;
            case 'S':
                if (!hasAffinity)
                {
                    fprintf(stderr, "Option -S is not supported for unknown processor!\n");
                    exit(EXIT_SUCCESS);
                }
                optMemSweep = 1;
                break;
            case 'v':
                VERSION_MSG;
                exit (EXIT_SUCCESS);
            default:
                HELP_MSG;
                exit(EXIT_FAILURE);
        }
    }
    if (optind == argc && !optPrintDomains)
    {
    	fprintf(stderr,"Executable must be given on commandline\n");
    	exit(EXIT_FAILURE);
    }

    if (optPrintDomains && numThreads)
    {
        fprintf(stdout, "%d",threads[0]);

        for ( i=1; i< numThreads; i++)
        {
            fprintf(stdout, "%c%d",delimiter,threads[i]);
        }
        fprintf(stdout, "\n");
        fflush(stdout);
        exit (EXIT_SUCCESS);
    }
    else if ( optPrintDomains )
    {
        affinity_printDomains();
        exit (EXIT_SUCCESS);
    }

    if (!numThreads)
    {
        argString = bformat("N:0-%u", cpuid_topology.numHWThreads-1);
        numThreads = bstr_to_cpuset(threads, argString);
    }

    /* CPU List:
     * pthread (default): pin main pid + all thread tids
     *
     * OpenMP: Pin OMP_NUM_THREADS
     * intel openmp: pin main pid + all thread tids (skip thread 1)
     * gcc openmp: pin main pid + all thread tids (one less)
     */

    if (optInterleaved)
    {
        fprintf(stdout, "Set mem_policy to interleaved\n");
        fflush(stdout);
        numa_setInterleaved(threads, numThreads);
    }

    if (optMemSweep)
    {
        fprintf(stdout, "Sweeping memory\n");
        fflush(stdout);
        memsweep_threadGroup(threads, numThreads);
    }

    if ( getenv("OMP_NUM_THREADS") == NULL )
    {
        argString = bformat("%d",numThreads);
        setenv("OMP_NUM_THREADS",(char*) argString->data , 0);
    }

    if (numThreads > 1)
    {
        bstring ldPreload = bfromcstr(getenv("LD_PRELOAD"));

        pinString = bformat("%d",threads[1]);

        for (i=2; i < numThreads;i++)
        {
            bformata(pinString,",%d",threads[i]);
        }

        bformata(pinString,",%d",threads[0]);

        if (skipMask >= 0)
        {
            skipString = bformat("%d",skipMask);
            setenv("LIKWID_SKIP",(char*) bdata(skipString) , 1);
        }

        setenv("KMP_AFFINITY", "disabled", 1);
        setenv("LIKWID_PIN",(char*) bdata(pinString) , 1);


        if (ldPreload == NULL)
        {
            setenv("LD_PRELOAD",TOSTRING(LIBLIKWIDPIN), 1);
        }
        else
        {
            bconchar(ldPreload, ':');
            bcatcstr(ldPreload, TOSTRING(LIBLIKWIDPIN));
            setenv("LD_PRELOAD", bdata(ldPreload), 1);
        }
    }

    pinPid(threads[0], optSilent);
    fflush(stdout);

    argv +=  optind;
    execvp(argv[0], argv);
    perror("execvp");
    fprintf(stderr,"failed to execute %s\n", argv[0]);

    return EXIT_SUCCESS;
}

