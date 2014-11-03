/*
 * =======================================================================================
 *
 *      Filename:  likwid-perfctr.c
 *
 *      Description:  An application to read out performance counter registers
 *                  on x86 processors
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
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>
#include <signal.h>

#include <error.h>
#include <types.h>
#include <bitUtil.h>
#include <accessClient.h>
#include <msr.h>
#include <timer.h>
#include <cpuid.h>
#include <affinity.h>
#include <cpuFeatures.h>
#include <perfmon.h>
#include <daemon.h>
#include <bstrlib.h>
#include <numa.h>
#include <strUtil.h>


/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */
#define HELP_MSG \
fprintf(stdout, "likwid-perfctr --  Version  %d.%d \n\n",VERSION,RELEASE); \
fprintf(stdout, "\n"); \
fprintf(stdout, "Example Usage: likwid-perfctr -C 2  ./a.out \n"); \
fprintf(stdout, "Supported Options:\n"); \
fprintf(stdout, "-h\t Help message\n"); \
fprintf(stdout, "-v\t Version information\n"); \
fprintf(stdout, "-V\t verbose output\n"); \
fprintf(stdout, "-g\t performance group or event set string\n"); \
fprintf(stdout, "-H\t Get group help (together with -g switch) \n"); \
fprintf(stdout, "-t\t timeline mode with frequency in s or ms, e.g. 300ms\n"); \
fprintf(stdout, "-S\t stethoscope mode with duration in s\n"); \
fprintf(stdout, "-m\t use markers inside code \n"); \
fprintf(stdout, "-s\t bitmask with threads to skip\n"); \
fprintf(stdout, "-o\t Store output to file, with output conversation according to file suffix\n"); \
fprintf(stdout, "\t Conversation scripts can be supplied in %s\n",TOSTRING(LIKWIDFILTERPATH)); \
fprintf(stdout, "-O\t Output easily parseable CSV instead of fancy tables\n"); \
fprintf(stdout, "-M\t set how MSR registers are accessed: 0=direct, 1=msrd\n"); \
fprintf(stdout, "-a\t list available performance groups\n"); \
fprintf(stdout, "-e\t list available counters and events\n"); \
fprintf(stdout, "-i\t print cpu info\n"); \
fprintf(stdout, "-c\t processor ids to measure (required), e.g 0,3-4,8\n"); \
fprintf(stdout, "-C\t processor ids to measure (this variant also cares for pinning of process/threads)\n"); \
fprintf(stdout, "\t\t for -c and -C, see likwid-pin -h for details\n"); \
fflush(stdout);


#define VERSION_MSG \
fprintf(stdout, "likwid-perfctr  %d.%d \n\n",VERSION,RELEASE); \
fflush(stdout);

/* To be able to give useful error messages instead of just dieing without a
 * comment. Mainly happens because we get a SIGPIPE if the daemon drops us. */
static void Signal_Handler(int sig)
{
    fprintf(stderr, "ERROR - [%s:%d] Signal %d caught\n", __FILE__, __LINE__, sig);
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */
int main (int argc, char** argv)
{
    int optInfo = 0;
    int optPrintGroups = 0;
    int optPrintGroupHelp = 0;
    int optPrintEvents = 0;
    int optUseMarker = 0;
    int optReport = 0;
    int optTimeline = 0;
    int optStethoscope = 0;
    int optPin = 0;
    int c;
    bstring eventString = bfromcstr("_NOGROUP");
    bstring  argString;
    bstring  pinString;
    bstring  skipString;
    bstring  filterScript = bfromcstr("NO");
    int skipMask = -1;
    BitMask counterMask;
    bstring filepath = bformat("/tmp/likwid_%u.txt", (uint32_t) getpid());
    int numThreads = 0;
    int threads[MAX_NUM_THREADS];
    threads[0] = 0;
    int i,j;
    FILE* OUTSTREAM = stdout;
    struct timespec interval;

    if (argc ==  1)
    {
        HELP_MSG;
        bdestroy(filepath);
        bdestroy(eventString);
        exit (EXIT_SUCCESS);
    }

    if (cpuid_init() == EXIT_FAILURE)
    {
        ERROR_PLAIN_PRINT(Unsupported processor!);
    }
    numa_init();
    affinity_init();

    while ((c = getopt (argc, argv, "+ac:C:d:eg:hHimM:o:OPs:S:t:vV")) != -1)
    {
        switch (c)
        {
            case 'a':
                numThreads = 1; /*to get over the error message */
                threads[0] = 0;
                optPrintGroups = 1;
                break;
            case 'C':
                optPin = 1;
                CHECK_OPTION_STRING;
                numThreads = bstr_to_cpuset(threads, argString);

                if(!numThreads)
                {
                    ERROR_PLAIN_PRINT(Failed to parse cpu list.);
                }

                break;
            case 'c':
                CHECK_OPTION_STRING;
                numThreads = bstr_to_cpuset(threads, argString);
                if(!numThreads)
                {
                    ERROR_PLAIN_PRINT(Failed to parse cpu list.);
                }

                break;
            case 'd':
                fprintf(stdout, "Option -d for daemon mode is deprecated. Daemon mode has be renamed to timeline mode (Option -t)!\n");
                fflush(stdout);
                break;
            case 'e':
                numThreads=1; /*to get over the error message */
                threads[0]=0;
                optPrintEvents = 1;
                break;
            case 'g':
                CHECK_OPTION_STRING;
                eventString = bstrcpy(argString);
                break;
            case 'h':
                HELP_MSG;
                cpuid_print();
                bdestroy(filepath);
                bdestroy(eventString);
                exit (EXIT_SUCCESS);
            case 'H':
                numThreads=1; /*to get over the error message */
                threads[0]=0;
                optPrintGroupHelp = 1;
                break;
            case 'i':
                numThreads=1; /*to get over the error message */
                threads[0]=0;
                optInfo = 1;
                perfmon_verbose = 1;
                break;
            case 'm':
                optUseMarker = 1;
                break;
            case 'M':  /* Set MSR Access mode */
                CHECK_OPTION_STRING;
                accessClient_setaccessmode(str2int((char*) argString->data));
                break;
            case 'o':
                CHECK_OPTION_STRING;
                OUTSTREAM = bstr_to_outstream(argString, filterScript);

                if(!OUTSTREAM)
                {
                    ERROR_PLAIN_PRINT(Failed to parse out file pattern.);
                }
                break;
            case 'O':
                perfmon_setCSVMode(1);
                break;
            case 's':
                CHECK_OPTION_STRING;
                skipMask = strtoul((char*) argString->data,NULL,16);
                break;
            case 'S':
                CHECK_OPTION_STRING;
                optStethoscope = str2int((char*) argString->data);
                if (optStethoscope <= 0)
                {
                    fprintf(stderr, "The measurement time must be larger than 0\n\n");
                    HELP_MSG;
                    exit(EXIT_FAILURE);
                }
                break;
            case 't':
                CHECK_OPTION_STRING;
                bstr_to_interval(argString, &interval);
                optTimeline = 1;
                break;
            case 'v':
                VERSION_MSG;
                bdestroy(filepath);
                bdestroy(eventString);
                exit (EXIT_SUCCESS);
            case 'V':
                perfmon_verbose = 1;
                break;
            case '?':
                if (optopt == 'S'||optopt == 't'||optopt == 'c'||optopt == 'C'||
                    optopt == 'o'||optopt == 'M'||optopt == 'g')
                {

                }
                else if (isprint (optopt))
                {
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                }
                else
                {
                    fprintf (stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
                }
                return EXIT_FAILURE;
            default:
                HELP_MSG;
                bdestroy(filepath);
                bdestroy(eventString);
                exit (EXIT_SUCCESS);
        }
    }

    if (!numThreads)
    {
        fprintf (stderr, "ERROR: Required -c. You must specify at least one processor.\n");
        HELP_MSG;
        exit(EXIT_FAILURE);
    }

    if (optPin)
    {

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

            if (skipMask > 0)
            {
                skipString = bformat("%d",skipMask);
                setenv("LIKWID_SKIP",(char*) skipString->data , 1);
            }
            setenv("KMP_AFFINITY", "disabled", 1);
            setenv("LIKWID_PIN",(char*) pinString->data , 1);

            setenv("LIKWID_SILENT","true", 1);
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

        affinity_pinProcess(threads[0]);
    }


    for (i = 0; i< numThreads;i++)
    {
        for (j = 0; j< numThreads;j++)
        {
            if(i != j && threads[i] == threads[j])
            {
                fprintf (stderr, "ERROR: Processor list (%d",threads[0]);
                for (c=1;c<numThreads;c++)
                {
                    fprintf (stderr, ",%d",threads[c]);
                }
                fprintf (stderr, ") is not unique.\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    { /* Init signal handler */
        struct sigaction sia;
        sia.sa_handler = Signal_Handler;
        sigemptyset(&sia.sa_mask);
        sia.sa_flags = 0;
        sigaction(SIGPIPE, &sia, NULL);
    }

    perfmon_init(numThreads, threads, OUTSTREAM);

    if (perfmon_verbose)
    {
        fprintf(OUTSTREAM,"CPU family:\t%u \n",cpuid_info.family);
        fprintf(OUTSTREAM,"CPU model:\t%u \n", cpuid_info.model);
        fprintf(OUTSTREAM,"CPU stepping:\t%u \n", cpuid_info.stepping);
        fprintf(OUTSTREAM,"CPU features:\t%s \n", cpuid_info.features);

        if( cpuid_info.family == P6_FAMILY && cpuid_info.perf_version)
        {
            fprintf(OUTSTREAM,HLINE);
            fprintf(OUTSTREAM,"PERFMON version:\t%u \n",cpuid_info.perf_version);
            fprintf(OUTSTREAM,"PERFMON number of counters:\t%u \n",cpuid_info.perf_num_ctr);
            fprintf(OUTSTREAM,"PERFMON width of counters:\t%u \n",cpuid_info.perf_width_ctr);
            fprintf(OUTSTREAM,"PERFMON number of fixed counters:\t%u \n",cpuid_info.perf_num_fixed_ctr);
        }
    }
    fprintf(OUTSTREAM,HLINE);
    fflush(OUTSTREAM);

    if (optInfo)
    {
        exit (EXIT_SUCCESS);
    }
    if (optPrintGroups)
    {
        perfmon_printAvailableGroups();
        exit (EXIT_SUCCESS);
    }
    if (optPrintGroupHelp)
    {
        perfmon_printGroupHelp(eventString);
        exit (EXIT_SUCCESS);
    }
    if (optPrintEvents)
    {
        perfmon_printCounters();
        perfmon_printEvents();
        exit (EXIT_SUCCESS);
    }
    if ((!optTimeline && !optStethoscope) && (optind == argc))
    {
        fprintf(OUTSTREAM,"NOTICE: You have to specify a program to measure as argument!\n");
        exit (EXIT_SUCCESS);
    }
    argv +=  optind;
    bstring exeString = bfromcstr(argv[0]);
    for (i=1; i<(argc-optind); i++)
        {
            bconchar(exeString, ' ');
            bcatcstr(exeString, argv[i]);
        }
    if (blength(exeString) == 0 && !optStethoscope)
    {
        fprintf(OUTSTREAM, "Executable must be given on commandline\n");
        fflush(OUTSTREAM);
        exit(EXIT_FAILURE);
    }
    if (biseqcstr(eventString,"_NOGROUP"))
    {
        fprintf(OUTSTREAM,"NOTICE: You have to specify a group or event set to measure using the -g option.\n");
        fprintf(OUTSTREAM,"        Use likwid-perfctr -a to get a list of available groups and likwid-perfctr -e for supported events.\n\n");
        exit (EXIT_SUCCESS);
    }

    timer_init();

    fprintf(OUTSTREAM,HLINE);
    fprintf(OUTSTREAM,"CPU type:\t%s \n",cpuid_info.name);
    fprintf(OUTSTREAM,"CPU clock:\t%3.2f GHz \n",  (float) timer_getCpuClock() * 1.E-09);
    fflush(OUTSTREAM);

    fprintf(OUTSTREAM,HLINE);
    fflush(OUTSTREAM);

    if (optStethoscope)
    {
        perfmon_setupEventSet(eventString, &counterMask);
        perfmon_startCounters();
        sleep(optStethoscope);
        perfmon_stopCounters();
        perfmon_printCounterResults();
    }
    else if (optTimeline)
    {
        fprintf(OUTSTREAM,"CORES: %d", threads[0]);
        for (int i=1; i<numThreads; i++)
        {
            fprintf(OUTSTREAM," %d", threads[i]);
        }
        fprintf(OUTSTREAM," \n");
        fflush(OUTSTREAM);

        daemon_start(eventString, interval);
        if (system(bdata(exeString)) == EOF)
        {
            fprintf(stderr, "Failed to execute %s!\n", bdata(exeString));
            exit(EXIT_FAILURE);
        }
        daemon_stop(SIGINT);
    }
    else
    {
        if (perfmon_verbose)
        {
            fprintf(OUTSTREAM,"Executing: %s \n",bdata(exeString));
            fflush(OUTSTREAM);
        }

        if (optReport)
        {
            //        multiplex_start();
        }
        else if (!optUseMarker && !optTimeline)
        {
            perfmon_setupEventSet(eventString, &counterMask);
            perfmon_startCounters();
        }
        else
        {
            if (getenv("LIKWID_FILEPATH") == NULL)
                setenv("LIKWID_FILEPATH",(char*) filepath->data, 1);
            perfmon_setupEventSet(eventString, &counterMask);
            char* modeStr = (char*) malloc(40 * sizeof(char));
            sprintf(modeStr,"%d",accessClient_mode);
            setenv("LIKWID_MODE", modeStr, 1);
            bitMask_toString(modeStr,counterMask);
            setenv("LIKWID_MASK", modeStr, 1);
            free(modeStr);

            perfmon_startCounters();
        }

        if (system(bdata(exeString)) == EOF)
        {
            fprintf(stderr, "Failed to execute %s!\n", bdata(exeString));
            exit(EXIT_FAILURE);
        }

        if (optReport)
        {
            //        multiplex_stop();
            //        perfmon_printReport(&set);
        }
        else
        {
            if (optUseMarker)
            {
                perfmon_stopCounters();
                perfmon_printMarkerResults(filepath);
            }
            else
            {
                perfmon_stopCounters();
                perfmon_printCounterResults();
            }
        }
    }

    bdestroy(filepath);
    bdestroy(exeString);
    perfmon_finalize();
    fflush(OUTSTREAM);
    fclose(OUTSTREAM);
    /* call filterscript if specified */
    if (!biseqcstr(filterScript,"NO"))
    {
        bcatcstr(filterScript, " perfctr");
        if (system(bdata(filterScript)) == EOF)
        {
            fprintf(stderr, "Failed to execute filter %s!\n", bdata(filterScript));
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}

