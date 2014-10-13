/*
 * =======================================================================================
 *
 *      Filename:  likwid-powermeter.c
 *
 *      Description:  An application to get information about power 
 *      consumption on architectures implementing the RAPL interface.
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
#include <stdio.h>
#include <string.h>
#include <sched.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>

#include <types.h>
#include <strUtil.h>
#include <error.h>
#include <lock.h>
#include <timer.h>
#include <cpuid.h>
#include <numa.h>
#include <accessClient.h>
#include <msr.h>
#include <affinity.h>
#include <perfmon.h>
#include <power.h>
#include <thermal.h>

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define HELP_MSG \
fprintf(stdout, "\nlikwid-powermeter --  Version  %d.%d \n\n",VERSION,RELEASE); \
fprintf(stdout, "A tool to print Power and Clocking information on Intel SandyBridge CPUS.\n"); \
fprintf(stdout, "Options:\n"); \
fprintf(stdout, "-h\t\t Help message\n"); \
fprintf(stdout, "-v\t\t Version information\n"); \
fprintf(stdout, "-M <0|1>\t set how MSR registers are accessed: 0=direct, 1=msrd \n"); \
fprintf(stdout, "-c <list>\t specify sockets to measure\n"); \
fprintf(stdout, "-i\t\t print information from MSR_PKG_POWER_INFO register and Turbo Mode\n"); \
fprintf(stdout, "-s <duration>\t set measure duration in sec. (default 2s) \n"); \
fprintf(stdout, "-p\t\t print dynamic clocking and CPI values (requires executable)\n\n");   \
fprintf(stdout, "Usage: likwid-powermeter -s 4 -c 1 \n");  \
fprintf(stdout, "Alternative as wrapper: likwid-powermeter -c 1 ./a.out\n"); \
fflush(stdout);

#define VERSION_MSG \
fprintf(stdout, "likwid-powermeter  %d.%d \n\n",VERSION,RELEASE); \
fflush(stdout);


int main (int argc, char** argv)
{
    int socket_fd = -1;
    int optInfo = 0;
    int optClock = 0;
    int optStethoscope = 0;
    int optSockets = 0;
    double runtime;
    int hasDRAM = 0;
    int hasPP0 = 0;
    int c;
    bstring argString;
    bstring eventString = bfromcstr("CLOCK");
    int numSockets=1;
    int numThreads=0;
    int threadsSockets[MAX_NUM_NODES*2];
    int threads[MAX_NUM_THREADS];
    threadsSockets[0] = 0;

    if (argc == 1)
    {
    	HELP_MSG;
    	exit (EXIT_SUCCESS);
    }

    while ((c = getopt (argc, argv, "+c:hiM:ps:v")) != -1)
    {
        switch (c)
        {
            case 'c':
                CHECK_OPTION_STRING;
                numSockets = bstr_to_cpuset_physical((uint32_t*) threadsSockets, argString);
                bdestroy(argString);
                optSockets = 1;
                break;

            case 'h':
                HELP_MSG;
                exit (EXIT_SUCCESS);
            case 'i':
                optInfo = 1;
                break;
            case 'M':  /* Set MSR Access mode */
                CHECK_OPTION_STRING;
                accessClient_setaccessmode(str2int((char*) argString->data));
                bdestroy(argString);
                break;
            case 'p':
                optClock = 1;
                break;
            case 's':
                CHECK_OPTION_STRING;
                optStethoscope = str2int((char*) argString->data);
                bdestroy(argString);
                break;
            case 'v':
                VERSION_MSG;
                exit (EXIT_SUCCESS);
            case '?':
            	if (optopt == 's' || optopt == 'M' || optopt == 'c')
            	{
            		HELP_MSG;
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
                exit( EXIT_FAILURE);
            default:
                HELP_MSG;
                exit (EXIT_SUCCESS);
        }
    }

    if (!lock_check())
    {
        fprintf(stderr,"Access to performance counters is locked.\n");
        exit(EXIT_FAILURE);
    }
    if (optClock && optind == argc)
    {
    	fprintf(stderr,"Commandline option -p requires an executable.\n");
    	exit(EXIT_FAILURE);
    }
    if (optSockets && !optStethoscope && optind == argc)
    {
        fprintf(stderr,"Commandline option -c requires an executable if not used in combination with -s.\n");
        exit(EXIT_FAILURE);
    }
    if (optStethoscope == 0 && optind == argc && !optInfo)
    {
        fprintf(stderr,"Either -s <seconds> or executable must be given on commandline.\n");
        exit(EXIT_FAILURE);
    }

    if (cpuid_init() == EXIT_FAILURE)
    {
        fprintf(stderr, "CPU not supported\n");
        exit(EXIT_FAILURE);
    }
    
    if (numSockets > cpuid_topology.numSockets)
    {
    	fprintf(stderr, "System has only %d sockets but %d are given on commandline.\n",
    			cpuid_topology.numSockets, numSockets);
    	exit(EXIT_FAILURE);
    }

    numa_init(); /* consider NUMA node as power unit for the moment */
    accessClient_init(&socket_fd);
    msr_init(socket_fd);
    timer_init();

    /* check for supported processors */
    if ((cpuid_info.model == SANDYBRIDGE_EP) ||
            (cpuid_info.model == SANDYBRIDGE) ||
            (cpuid_info.model == IVYBRIDGE) ||
            (cpuid_info.model == IVYBRIDGE_EP) ||
            (cpuid_info.model == HASWELL) ||
            (cpuid_info.model == HASWELL_EX) ||
            (cpuid_info.model == NEHALEM_BLOOMFIELD) ||
            (cpuid_info.model == NEHALEM_LYNNFIELD) ||
            (cpuid_info.model == NEHALEM_WESTMERE) ||
            (cpuid_info.model == ATOM_SILVERMONT))
    {
        if (numSockets == 0)
        {
            numSockets = numa_info.numberOfNodes;
        }
        for(int i=0; i<numSockets; i++)
        {
            power_init(numa_info.nodes[threadsSockets[i]].processors[0]);
        }
    }
    else
    {
        fprintf (stderr, "Query Turbo Mode only supported on Intel Nehalem/Westmere/SandyBridge/IvyBridge/Haswell processors!\n");
        exit(EXIT_FAILURE);
    }

    double clock = (double) timer_getCpuClock();

    fprintf(stdout, HLINE);
    fprintf(stdout, "CPU name:\t%s \n",cpuid_info.name);
    fprintf(stdout, "CPU clock:\t%3.2f GHz \n",  (float) clock * 1.E-09);
    fprintf(stdout, HLINE);
    fflush(stdout);

    if (optInfo)
    {
        if (power_info.turbo.numSteps != 0)
        {
            fprintf(stdout, "Base clock:\t%.2f MHz \n",  power_info.baseFrequency );
            fprintf(stdout, "Minimal clock:\t%.2f MHz \n",  power_info.minFrequency );
            fprintf(stdout, "Turbo Boost Steps:\n");
            for (int i=0; i < power_info.turbo.numSteps; i++ )
            {
                fprintf(stdout, "C%d %.2f MHz \n",i+1,  power_info.turbo.steps[i] );
            }
        }
        fprintf(stdout, HLINE);
        fflush(stdout);
    }

    if ((cpuid_info.model == SANDYBRIDGE_EP) ||
        (cpuid_info.model == IVYBRIDGE_EP) ||
        (cpuid_info.model == HASWELL_EX))
    {
        hasDRAM = 1;
    }
    if (cpuid_info.model != ATOM_SILVERMONT)
    {
        hasPP0 = 1;
    }
    if ((cpuid_info.model != SANDYBRIDGE) &&
        (cpuid_info.model != SANDYBRIDGE_EP)  &&
        (cpuid_info.model != IVYBRIDGE)  &&
        (cpuid_info.model != IVYBRIDGE_EP)  &&
        (cpuid_info.model != HASWELL) &&
        (cpuid_info.model != HASWELL_EX) &&
        (cpuid_info.model != ATOM_SILVERMONT))
    {
        fprintf (stderr, "RAPL not supported on this processor!\n");
        exit(EXIT_FAILURE);
    }

    if (optInfo)
    {
        fprintf(stdout, "Thermal Spec Power: %g Watts \n", power_info.tdp );
        fprintf(stdout, "Minimum  Power: %g Watts \n", power_info.minPower);
        fprintf(stdout, "Maximum  Power: %g Watts \n", power_info.maxPower);
        fprintf(stdout, "Maximum  Time Window: %g micro sec \n", power_info.maxTimeWindow);
        fprintf(stdout, HLINE);
        fflush(stdout);
        exit(EXIT_SUCCESS);
    }

    if (optClock)
    {
        affinity_init();
        argString = bformat("S%u:0-%u", threadsSockets[0], cpuid_topology.numCoresPerSocket-1);
        for (int i=1; i<numSockets; i++)
        {
            bstring tExpr = bformat("@S%u:0-%u", threadsSockets[i], cpuid_topology.numCoresPerSocket-1);
            bconcat(argString, tExpr);
        }
        numThreads = bstr_to_cpuset(threads, argString);
        bdestroy(argString);
        perfmon_init(numThreads, threads, stdout);
        perfmon_setupEventSet(eventString, NULL);
    }

    {
        PowerData pDataPkg[MAX_NUM_NODES*2];
        PowerData pDataDram[MAX_NUM_NODES*2];
        PowerData pDataPP0[MAX_NUM_NODES*2];
        fprintf(stdout, "Measure on sockets: %d", threadsSockets[0]);
        for (int i=1; i<numSockets; i++)
        {
            fprintf(stdout, ", %d", threadsSockets[i]);
        }
        fprintf(stdout, "\n");
        fflush(stdout);

        if (optStethoscope)
        {
            if (optClock)
            {
                perfmon_startCounters();
            }
            else
            {
                for (int i=0; i<numSockets; i++)
                {
                    int cpuId = numa_info.nodes[threadsSockets[i]].processors[0];
                    if (hasDRAM) power_start(pDataDram+i, cpuId, DRAM);
                    if (hasPP0) power_start(pDataPP0+i, cpuId, PP0);
                    power_start(pDataPkg+i, cpuId, PKG);
                }
            }
            sleep(optStethoscope);

            if (optClock)
            {
                perfmon_stopCounters();
                perfmon_printCounterResults();
                perfmon_finalize();
            }
            else
            {
                for (int i=0; i<numSockets; i++)
                {
                    int cpuId = numa_info.nodes[threadsSockets[i]].processors[0];
                    power_stop(pDataPkg+i, cpuId, PKG);
                    if (hasPP0) power_stop(pDataPP0+i, cpuId, PP0);
                    if (hasDRAM) power_stop(pDataDram+i, cpuId, DRAM);
                }
            }
            runtime = (double) optStethoscope;
        }
        else
        {
            TimerData time;
            argv +=  optind;
            bstring exeString = bfromcstr(argv[0]);

            for (int i=1; i<(argc-optind); i++)
            {
                bconchar(exeString, ' ');
                bcatcstr(exeString, argv[i]);
            }
            fprintf(stdout, "Executing: %s\n",bdata(exeString));
            fflush(stdout);


            if (optClock)
            {
                perfmon_startCounters();
            }
            else
            {
                for (int i=0; i<numSockets; i++)
                {
                    int cpuId = numa_info.nodes[threadsSockets[i]].processors[0];
                    if (hasDRAM) power_start(pDataDram+i, cpuId, DRAM);
                    if (hasPP0) power_start(pDataPP0+i, cpuId, PP0);
                    power_start(pDataPkg+i, cpuId, PKG);
                }

                timer_start(&time);
            }

            if (system(bdata(exeString)) == EOF)
            {
                fprintf(stderr, "Failed to execute %s!\n", bdata(exeString));
                exit(EXIT_FAILURE);
            }

            if (optClock)
            {
                perfmon_stopCounters();
                perfmon_printCounterResults();
                perfmon_finalize();
            }
            else
            {
                timer_stop(&time);

                for (int i=0; i<numSockets; i++)
                {
                    int cpuId = numa_info.nodes[threadsSockets[i]].processors[0];
                    power_stop(pDataPkg+i, cpuId, PKG);
                    if (hasDRAM) power_stop(pDataDram+i, cpuId, DRAM);
                    if (hasPP0) power_stop(pDataPP0+i, cpuId, PP0);
                }
                runtime = timer_print(&time);
            }
        }

        if (!optClock)
        {
            fprintf(stdout, "Runtime: %g second \n",runtime);
            fprintf(stdout, HLINE);
            for (int i=0; i<numSockets; i++)
            {
                fprintf(stdout, "Socket %d\n",threadsSockets[i]);
                fprintf(stdout, "Domain: PKG \n");
                fprintf(stdout, "Energy consumed: %g Joules \n", power_printEnergy(pDataPkg+i));
                fprintf(stdout, "Power consumed: %g Watts \n", power_printEnergy(pDataPkg+i) / runtime );
                if (hasDRAM)
                {
                    fprintf(stdout, "Domain: DRAM \n");
                    fprintf(stdout, "Energy consumed: %g Joules \n", power_printEnergy(pDataDram+i));
                    fprintf(stdout, "Power consumed: %g Watts \n", power_printEnergy(pDataDram+i) / runtime );
                }
                if (hasPP0)
                {
                    fprintf(stdout, "Domain: PP0 \n");
                    fprintf(stdout, "Energy consumed: %g Joules \n", power_printEnergy(pDataPP0+i));
                    fprintf(stdout, "Power consumed: %g Watts \n", power_printEnergy(pDataPP0+i) / runtime );
                }
                fprintf(stdout, "\n");
            }
            fflush(stdout);
        }
    }

#if 0
    if ( cpuid_hasFeature(TM2) )
    {
        thermal_init(0);
        printf("Current core temperatures:\n");

        for (uint32_t i = 0; i < cpuid_topology.numCoresPerSocket; i++ )
        {
            printf("Core %d: %u C\n",
                    numa_info.nodes[socketId].processors[i],
                    thermal_read(numa_info.nodes[socketId].processors[i]));
        }
    }
#endif

    msr_finalize();
    return EXIT_SUCCESS;
}

