/*
 * =======================================================================================
 *
 *      Filename:  numa.c
 *
 *      Description:  Implementation of Linux NUMA interface. Selects between hwloc and
 *                    procfs/sysfs backends.
 *
 *      Version:   5.1
 *      Released:  16.11.2020
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
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

#include <numa.h>
#include <numa_proc.h>

#ifdef LIKWID_USE_HWLOC
#include <hwloc.h>
#include <topology_hwloc.h>
#include <numa_hwloc.h>
#endif


/* #####   EXPORTED VARIABLES   ########################################### */

NumaTopology numa_info = {0,NULL};

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

int numaInitialized = 0;

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
str2int(const char* str)
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

uint64_t
proc_getFreeSysMem(void)
{
    FILE *fp;
    uint64_t free = 0;
    bstring freeString  = bformat("MemFree:");
    int i;

    if (!access("/proc/meminfo", R_OK))
    {
        if (NULL != (fp = fopen ("/proc/meminfo", "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            struct bstrList* tokens = bsplit(src,(char) '\n');
            for (i=0;i<tokens->qty;i++)
            {
                if (binstr(tokens->entry[i],0,freeString) != BSTR_ERR)
                {
                     bstring tmp = bmidstr (tokens->entry[i], 10, blength(tokens->entry[i])-10  );
                     bltrimws(tmp);
                     struct bstrList* subtokens = bsplit(tmp,(char) ' ');
                     free = str2int(bdata(subtokens->entry[0]));
                     bdestroy(tmp);
                     bstrListDestroy(subtokens);
                }
            }
            bstrListDestroy(tokens);
            bdestroy(src);
            fclose(fp);
        }
    }
    else
    {
        bdestroy(freeString);
        ERROR;
    }
    bdestroy(freeString);
    return free;
}

uint64_t
proc_getTotalSysMem(void)
{
    int i;
    FILE *fp;
    uint64_t total = 0;
    bstring totalString  = bformat("MemTotal:");

    if (!access("/proc/meminfo", R_OK))
    {
        if (NULL != (fp = fopen ("/proc/meminfo", "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            struct bstrList* tokens = bsplit(src,(char) '\n');
            for (i=0;i<tokens->qty;i++)
            {
                if (binstr(tokens->entry[i],0,totalString) != BSTR_ERR)
                {
                     bstring tmp = bmidstr (tokens->entry[i], 10, blength(tokens->entry[i])-10  );
                     bltrimws(tmp);
                     struct bstrList* subtokens = bsplit(tmp,(char) ' ');
                     total = str2int(bdata(subtokens->entry[0]));
                     bdestroy(tmp);
                     bstrListDestroy(subtokens);
                }
            }
            bstrListDestroy(tokens);
            bdestroy(src);
            fclose(fp);
        }
    }
    else
    {
        bdestroy(totalString);
        ERROR;
    }

    bdestroy(totalString);
    return total;
}

int
empty_numa_init()
{
    printf("MEMPOLICY NOT supported in kernel!\n");
    return virtual_numa_init();
}

void
empty_numa_setInterleaved(const int* processorList, int numberOfProcessors)
{
    printf("MEMPOLICY NOT supported in kernel!\n");
    if (!processorList || numberOfProcessors < 0)
        printf("Invalid options\n");
    return;
}

void
empty_numa_setMembind(const int* processorList, int numberOfProcessors)
{
    printf("MEMPOLICY NOT supported in kernel!\n");
    if (!processorList || numberOfProcessors < 0)
        printf("Invalid options\n");
    return;
}

void
empty_numa_membind(void* ptr, size_t size, int domainId)
{
    printf("MBIND NOT supported in kernel!\n");
    if (!ptr || size <= 0 || domainId < 0)
        printf("Invalid options\n");
    return;
}

const struct numa_functions numa_funcs = {
#ifndef HAS_MEMPOLICY
    .numa_init = empty_numa_init,
    .numa_setInterleaved = empty_numa_setInterleaved,
    .numa_membind = empty_numa_membind,
    .numa_setMembind = empty_numa_setMembind,
#else
#ifdef LIKWID_USE_HWLOC
#if defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_8A__) || defined(__ARM_ARCH_8A) || defined(_ARCH_PPC64)
    .numa_init = proc_numa_init,
#else
    .numa_init = hwloc_numa_init,
#endif
#else
    .numa_init = proc_numa_init,
#endif
    .numa_setInterleaved = proc_numa_setInterleaved,
    .numa_membind = proc_numa_membind,
    .numa_setMembind = proc_numa_setMembind,
#endif
};

int
numa_init(void)
{
    const struct numa_functions funcs = numa_funcs;
    int ret = 0;
    if (init_config == 0)
    {
        init_configuration();
    }
    if (numaInitialized == 1)
    {
        return 0;
    }

    if ((config.topologyCfgFileName != NULL) && (!access(config.topologyCfgFileName, R_OK)) && (numa_info.nodes != NULL))
    {
        /* If we read in the topology file, the NUMA related stuff is already initialized */
        numaInitialized = 1;
        return 0;
    }
    else
    {
        cpu_set_t cpuSet;
        CPU_ZERO(&cpuSet);
        sched_getaffinity(0,sizeof(cpu_set_t), &cpuSet);
        if (cpuid_topology.activeHWThreads < cpuid_topology.numHWThreads &&
            getenv("HWLOC_FSROOT") == NULL)
        {
            ret = proc_numa_init();
        }
        else
        {
            ret = funcs.numa_init();
        }
        if (ret == 0)
            numaInitialized = 1;
    }
    return ret;
}

void
numa_setInterleaved(const int* processorList, int numberOfProcessors)
{
    const struct numa_functions funcs = numa_funcs;
    funcs.numa_setInterleaved(processorList, numberOfProcessors);
    return;
}

void
numa_membind(void* ptr, size_t size, int domainId)
{
    const struct numa_functions funcs = numa_funcs;
    funcs.numa_membind(ptr, size, domainId);
    return;
}

void
numa_setMembind(const int* processorList, int numberOfProcessors)
{
    const struct numa_functions funcs = numa_funcs;
    funcs.numa_setMembind(processorList, numberOfProcessors);
    return;
}

#ifndef HAS_MEMPOLICY
void
numa_finalize(void)
{
    return;
}
#else
void
numa_finalize(void)
{
    int i = 0;
    if (!numaInitialized)
    {
        return;
    }
    for(i = 0; i < (int)numa_info.numberOfNodes; i++)
    {
        if (numa_info.nodes[i].processors)
        {
            free(numa_info.nodes[i].processors);
        }
        if (numa_info.nodes[i].distances)
        {
            free(numa_info.nodes[i].distances);
        }
        numa_info.nodes[i].id = 0;
        numa_info.nodes[i].totalMemory = 0;
        numa_info.nodes[i].freeMemory = 0;
        numa_info.nodes[i].numberOfProcessors = 0;
        numa_info.nodes[i].numberOfDistances = 0;
    }
    if (numa_info.nodes)
    {
        free(numa_info.nodes);
    }
    numa_info.numberOfNodes = 0;
    numaInitialized = 0;
    return;
}

int
likwid_getNumberOfNodes()
{
    if (numaInitialized)
    {
        return numa_info.numberOfNodes;
    }
    return 0;
}
#endif
