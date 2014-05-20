/*
 * ===========================================================================
 *
 *      Filename:  likwid-perfCtr.c
 *
 *      Description:  An application to read out performance counter registers
 *                  on x86 processors
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
#include <stdio.h>
#include <string.h>
#include <sched.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>

#include <error.h>
#include <types.h>
#include <msr.h>
#include <timer.h>
#include <cpuid.h>
#include <affinity.h>
#include <cpuFeatures.h>
#include <perfmon.h>
#include <daemon.h>
#include <bstrlib.h>
#include <strUtil.h>


#define HELP_MSG \
printf("likwid-perfCtr --  Version  %d.%d \n\n",VERSION,RELEASE); \
printf("\n"); \
printf("Example Usage: likwid-perfCtr -c 2  likwid-pin -c 2 ./a.out \n"); \
printf("Supported Options:\n"); \
printf("-h\t Help message\n"); \
printf("-H\t Get group help (together with -g switch) \n"); \
printf("-v\t Version information\n"); \
printf("-V\t verbose output\n"); \
printf("-i\t print cpu info\n"); \
printf("-m\t use markers inside code \n"); \
printf("-g\t performance group  or event set string\n"); \
printf("-d\t daemon mode with frequency in seconds\n"); \
printf("-a\t list available performance groups\n"); \
printf("-e\t list available counters and events\n"); \
printf("-c\t processor ids to measure (required), e.g. 1,2-4,8\n\n")


#define VERSION_MSG \
printf("likwid-perfCtr  %d.%d \n\n",VERSION,RELEASE);

int main (int argc, char** argv)
{ 
    int optInfo = 0;
    int optPrintGroups = 0;
    int optPrintGroupHelp = 0;
    int optPrintEvents = 0;
    int optUseMarker = 0;
    int optReport = 0;
    int optDaemon = 0;
    int c;
    bstring eventString = bformat("FLOPS_DP");
    bstring  argString;
    bstring filepath = bformat("/tmp/likwid_%u.txt",(uint32_t) getpid());
    int numThreads=0;
    int threads[MAX_NUM_THREADS];
    threads[0] = 0;
    int i,j;

    if (argc ==  1) 
    {
        HELP_MSG;
        bdestroy(filepath);
        bdestroy(eventString);
        exit (EXIT_SUCCESS);    
    }

    cpuid_init();
    affinity_init();

    while ((c = getopt (argc, argv, "+c:g:hd:HvmVaeir")) != -1)
    {
        switch (c)
        {
            case 'h':
                HELP_MSG;
                bdestroy(filepath);
                bdestroy(eventString);
                exit (EXIT_SUCCESS);    
            case 'H':
                numThreads=1; /*to get over the error message */
                threads[0]=0;
                optPrintGroupHelp = 1;
                break;
            case 'v':
                VERSION_MSG;
                bdestroy(filepath);
                bdestroy(eventString);
                exit (EXIT_SUCCESS);    
            case 'g':
                if (! (argString = bSecureInput(200,optarg)))
                {
                    ERROR_MSG(Failed to read argument string!);
                }

                eventString = bstrcpy(argString);
                bdestroy(argString);
                break;
            case 'c':
                if (! (argString = bSecureInput(200,optarg)))
                {
                    ERROR_MSG(Failed to read argument string!);
                }

                numThreads = bstr_to_cpuset(threads, argString);

                if(!numThreads)
                {
                    ERROR_MSG(Failed to parse cpu list.);
                }

                bdestroy(argString);
                break;
            case 'd':
                optDaemon = atoi(optarg);
                break;
            case 'V':
                perfmon_verbose = 1;
                break;
            case 'a':
                numThreads=1; /*to get over the error message */
                threads[0]=0;
                optPrintGroups = 1;
                break;
            case 'e':
                numThreads=1; /*to get over the error message */
                threads[0]=0;
                optPrintEvents = 1;
                break;
            case 'm':
                optUseMarker = 1;
                break;
            case 'r':
                eventString = bfromcstr("REPORT");
                optReport = 1;
                break;
            case 'i':
                numThreads=1; /*to get over the error message */
                threads[0]=0;
                optInfo = 1;
                perfmon_verbose = 1;
                break;
            case '?':
                if (isprint (optopt))
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
        exit(EXIT_FAILURE);
    }

    for (i = 0; i< numThreads;i++)
    {
        for (j = 0; j< numThreads;j++)
        {
            if(i != j && threads[i] == threads[j])
            {
                fprintf (stderr, "ERROR: Processor list is not unique.\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    perfmon_init(numThreads, threads);

    if (perfmon_verbose) 
    {
        printf("CPU family:\t%u \n",cpuid_info.family);
        printf("CPU model:\t%u \n", cpuid_info.model);
        printf("CPU stepping:\t%u \n", cpuid_info.stepping);
        printf("CPU features:\t%s \n", cpuid_info.features);

        if( cpuid_info.family == P6_FAMILY && cpuid_info.perf_version) 
        {
            printf(HLINE);
            printf("PERFMON version:\t%u \n",cpuid_info.perf_version);
            printf("PERFMON number of counters:\t%u \n",cpuid_info.perf_num_ctr);
            printf("PERFMON width of counters:\t%u \n",cpuid_info.perf_width_ctr);
            printf("PERFMON number of fixed counters:\t%u \n",cpuid_info.perf_num_fixed_ctr);
        }
    }
    printf(HLINE);

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
    if ((!optDaemon) && (optind == argc)) 
    {
        printf("NOTICE: You have to specify a program to measure as argument!\n");
        exit (EXIT_SUCCESS);
    }

    msr_check();
    timer_init();

    if( cpuid_info.family == P6_FAMILY ) 
        cpuFeatures_init(0);
    {

        if (cpuFeatureFlags.speedstep)
        {
            fprintf (stderr, "Speedstep is enabled!\nThis produces inaccurate timing measurements.\n");
            fprintf (stderr, "For reliable clock measurements disable speedstep.\n");
        }
    }

    printf(HLINE);
    printf("CPU type:\t%s \n",cpuid_info.name);
    printf("CPU clock:\t%3.2f GHz \n",  (float) timer_getCpuClock() * 1.E-09);

    perfmon_setupEventSet(eventString);
    printf(HLINE);

    if (optDaemon)
    {
        daemon_init(eventString);
        daemon_start(optDaemon);
        
    }

	argv +=  optind;
    bstring exeString = bfromcstr(argv[0]);

    for (i=1; i<(argc-optind); i++)
    {
        bconchar(exeString, ' ');
        bcatcstr(exeString, argv[i]);
    }
    if (perfmon_verbose) printf("Executing: %s \n",bdata(exeString));

    if (optReport)
    {
//        multiplex_start();
    }
    else if (!optUseMarker)
    {
        perfmon_startCounters();
    }
    else
    {
        setenv("LIKWID_FILEPATH",(char*) filepath->data, 1);
    }

        printf("%s\n",bdata(exeString));
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
		if (!optUseMarker)
		{
			perfmon_stopCounters();
			perfmon_printCounterResults();
		}
		else
		{
			perfmon_printMarkerResults(filepath);
		}
	}

    bdestroy(filepath);
    bdestroy(exeString);
    perfmon_finalize();

    return EXIT_SUCCESS;
}

