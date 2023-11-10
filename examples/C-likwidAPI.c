/*
 * =======================================================================================
 *
 *      Filename:  C-likwidAPI.c
 *
 *      Description:  Example how to use the LIKWID API in C/C++ applications
 *
 *      Version:   5.3
 *      Released:  10.11.2023
 *
 *      Author:  Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2023 RRZE, University Erlangen-Nuremberg
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
#include <unistd.h>

#include <likwid.h>


int main(int argc, char* argv[])
{
    int i, j;
    int err;
    int* cpus;
    int gid;
    double result = 0.0;
    char estr[] = "L2_LINES_IN_ALL:PMC0,L2_TRANS_L2_WB:PMC1";
    char* enames[2] = {"L2_LINES_IN_ALL:PMC0","L2_TRANS_L2_WB:PMC1"};
    int n = sizeof(enames) / sizeof(enames[0]);
    //perfmon_setVerbosity(3);
    // Load the topology module and print some values.
    err = topology_init();
    if (err < 0)
    {
        printf("Failed to initialize LIKWID's topology module\n");
        return 1;
    }
    // CpuInfo_t contains global information like name, CPU family, ...
    CpuInfo_t info = get_cpuInfo();
    // CpuTopology_t contains information about the topology of the CPUs.
    CpuTopology_t topo = get_cpuTopology();
    // Create affinity domains. Commonly only needed when reading Uncore counters
    affinity_init();

    printf("Likwid example on a %s with %d CPUs\n", info->name, topo->numHWThreads);

    cpus = (int*)malloc(topo->numHWThreads * sizeof(int));
    if (!cpus)
        return 1;

    for (i=0;i<topo->numHWThreads;i++)
    {
        cpus[i] = topo->threadPool[i].apicId;
    }

    // Must be called before perfmon_init() but only if you want to use another
    // access mode as the pre-configured one. For direct access (0) you have to
    // be root.
    //accessClient_setaccessmode(0);

    // Initialize the perfmon module.
    err = perfmon_init(topo->numHWThreads, cpus);
    if (err < 0)
    {
        printf("Failed to initialize LIKWID's performance monitoring module\n");
        topology_finalize();
        return 1;
    }

    // Add eventset string to the perfmon module.
    gid = perfmon_addEventSet(estr);
    if (gid < 0)
    {
        printf("Failed to add event string %s to LIKWID's performance monitoring module\n", estr);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }

    // Setup the eventset identified by group ID (gid).
    err = perfmon_setupCounters(gid);
    if (err < 0)
    {
        printf("Failed to setup group %d in LIKWID's performance monitoring module\n", gid);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }
    // Start all counters in the previously set up event set.
    err = perfmon_startCounters();
    if (err < 0)
    {
        printf("Failed to start counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }


    // Perform some work.
    sleep(2);

    // Read and record current event counts.
    err = perfmon_readCounters();
    if (err < 0)
    {
        printf("Failed to read counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }

    // Print the result of every thread/CPU for all events in estr, counting from last read/startCounters().
    printf("Work task 1/2 measurements:\n");
    for (j=0; j<n; j++)
    {
        for (i = 0;i < topo->numHWThreads; i++)
        {
            result = perfmon_getLastResult(gid, j, i);
            printf("- event set %s at CPU %d: %f\n", enames[j], cpus[i], result);
        }
    }


    // Perform another piece of work
    sleep(2);

    // Read and record current event counts.
    err = perfmon_readCounters();
    if (err < 0)
    {
        printf("Failed to read counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }

    // Print the result of every thread/CPU for all events in estr, counting between the 
    // previous two calls of perfmon_readCounters().
    printf("Work task 2/2 measurements:\n");
    for (j=0; j<n; j++)
    {
        for (i = 0;i < topo->numHWThreads; i++)
        {
            result = perfmon_getLastResult(gid, j, i);
            printf("- event set %s at CPU %d: %f\n", enames[j], cpus[i], result);
        }
    }



    // Stop all counters in the currently-active event set.
    err = perfmon_stopCounters();
    if (err < 0)
    {
        printf("Failed to stop counters for group %d for thread %d\n",gid, (-1*err)-1);
        perfmon_finalize();
        topology_finalize();
        return 1;
    }

    // Print the result of every thread/CPU for all events in estr, counting since counters first started.
    printf("Total sum measurements:\n");
    for (j=0; j<n; j++)
    {
        for (i = 0;i < topo->numHWThreads; i++)
        {
            result = perfmon_getResult(gid, j, i);
            printf("- event set %s at CPU %d: %f\n", enames[j], cpus[i], result);
        }
    }


    free(cpus);
    // Uninitialize the perfmon module.
    perfmon_finalize();
    affinity_finalize();
    // Uninitialize the topology module.
    topology_finalize();
    return 0;
}
