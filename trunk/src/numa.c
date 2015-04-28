/*
 * =======================================================================================
 *
 *      Filename:  numa.c
 *
 *      Description:  Implementation of Linux NUMA interface
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
/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <error.h>
#include <dirent.h>
#ifdef HAS_MEMPOLICY
#include <linux/mempolicy.h>
#endif
#include <topology.h>

#include <configuration.h>

#include <error.h>
#include <bstrlib.h>
//#include <strUtil.h>

#include <numa.h>
#include <numa_proc.h>

#ifdef LIKWID_USE_HWLOC
#include <hwloc.h>
#include <topology_hwloc.h>
#include <numa_hwloc.h>
#endif


/* #####   EXPORTED VARIABLES   ########################################### */
NumaTopology numa_info = {0,NULL};
int maxIdConfiguredNode = 0;

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */
/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */
/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */
/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */
int str2int(const char* str)
{
    char* endptr;
    errno = 0;
    unsigned long val;
    val = strtoul(str, &endptr, 10);

    if ((errno == ERANGE && val == LONG_MAX)
        || (errno != 0 && val == 0))
    {
        fprintf(stderr, "Value in string out of range\n");
        return -EINVAL;
    }

    if (endptr == str)
    {
        fprintf(stderr, "No digits were found\n");
        return -EINVAL;
    }

    return (int) val;
}

int
empty_numa_init()
{
    printf("MEMPOLICY NOT supported in kernel!\n");
    return 0;
}

void 
empty_numa_setInterleaved(int* processorList, int numberOfProcessors)
{
    printf("MEMPOLICY NOT supported in kernel!\n");
    return;
}

void
empty_numa_membind(void* ptr, size_t size, int domainId)
{
    printf("MBIND NOT supported in kernel!\n");
    return;
}


const struct numa_functions numa_funcs = {
#ifndef HAS_MEMPOLICY
    .numa_init = empty_numa_init,
    .numa_setInterleaved = empty_numa_setInterleaved,
    .numa_membind = empty_numa_membind
#else
#ifdef LIKWID_USE_HWLOC
    .numa_init = hwloc_numa_init,
#else
    .numa_init = proc_numa_init,
#endif
    .numa_setInterleaved = proc_numa_setInterleaved,
    .numa_membind = proc_numa_membind
#endif
};


int numa_init(void)
{
    const struct numa_functions funcs = numa_funcs;

    if (init_config == 0)
    {
        init_configuration();
    }

    if (access(config.topologyCfgFileName, R_OK) && numa_info.numberOfNodes <= 0)
    {
        cpu_set_t cpuSet;
        CPU_ZERO(&cpuSet);
        sched_getaffinity(0,sizeof(cpu_set_t), &cpuSet);
        if (cpuid_topology.activeHWThreads < cpuid_topology.numHWThreads)
        {
            return proc_numa_init();
        }
        return funcs.numa_init();
    }
    return 0;
}

void numa_setInterleaved(int* processorList, int numberOfProcessors)
{
    const struct numa_functions funcs = numa_funcs;
    return funcs.numa_setInterleaved(processorList, numberOfProcessors);
}

void numa_membind(void* ptr, size_t size, int domainId)
{
    const struct numa_functions funcs = numa_funcs;
    return funcs.numa_membind(ptr, size, domainId);
}

#ifndef HAS_MEMPOLICY
void numa_finalize(void)
{
    return;
}
#else
void numa_finalize(void)
{
    int i;
    for(i=0;i<numa_info.numberOfNodes;i++)
    {
        if (numa_info.nodes[i].processors)
        {
            free(numa_info.nodes[i].processors);
        }
        if (numa_info.nodes[i].distances)
        {
            free(numa_info.nodes[i].distances);
        }
    }
    if (numa_info.nodes)
    {
        free(numa_info.nodes);
    }
    return;
}
#endif
