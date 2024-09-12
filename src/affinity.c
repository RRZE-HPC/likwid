/*
 * =======================================================================================
 *
 *      Filename:  affinity.c
 *
 *      Description:  Implementation of affinity module.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com,
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
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
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <pthread.h>
#include <math.h>

#include <types.h>
#include <error.h>
#include <likwid.h>
#include <numa.h>
#include <affinity.h>
#include <lock.h>
#include <tree.h>
#include <topology.h>
#include <topology_hwloc.h>

/* #####   EXPORTED VARIABLES   ########################################### */

int *affinity_thread2core_lookup = NULL;
int *affinity_thread2die_lookup = NULL;
int *affinity_thread2socket_lookup = NULL;
int *affinity_thread2numa_lookup = NULL;
int *affinity_thread2sharedl3_lookup = NULL;

int *socket_lock = NULL;
int *die_lock = NULL;
int *core_lock = NULL;
int *tile_lock = NULL;
int *numa_lock = NULL;
int *sharedl2_lock = NULL;
int *sharedl3_lock = NULL;
/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define gettid() syscall(SYS_gettid)

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int  affinity_numberOfDomains = 0;
static AffinityDomain*  domains;
static int affinity_initialized = 0;

AffinityDomains affinityDomains;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int
getProcessorID(cpu_set_t* cpu_set)
{
    int processorId;
    topology_init();
    CpuTopology_t cputopo = get_cpuTopology();

    for ( processorId = 0; processorId < cputopo->numHWThreads; processorId++ )
    {
        if ( CPU_ISSET(processorId,cpu_set) )
        {
            break;
        }
    }
    return processorId;
}

static int
treeFillNextEntries(
    TreeNode* tree,
    int* processorIds,
    int startidx,
    int socketId,
    int coreOffset,
    int coreSpan,
    int numberOfEntries)
{
    int counter = numberOfEntries;
    int skip = 0;
    int c, t, c_count = 0;
    TreeNode* node = tree;
    TreeNode* thread;
    CpuTopology_t cputopo = get_cpuTopology();

    node = tree_getChildNode(node);

    /* get socket node */
    for (int i=0; i<socketId; i++)
    {
        node = tree_getNextNode(node);
        if ( node == NULL )
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Cannot find socket %d in topology tree, i);
        }
    }

    node = tree_getChildNode(node);
    /* skip offset cores */
    for (int i=0; i<coreOffset; i++)
    {
        node = tree_getNextNode(node);

        if ( node == NULL )
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Cannot find core %d in topology tree, i);
        }
    }

    /* Traverse horizontal */
    while ( node != NULL && c_count < coreSpan)
    {
        if ( !counter ) break;

        thread = tree_getChildNode(node);

        while ( thread != NULL && (numberOfEntries-counter) < numberOfEntries )
        {
            if (cputopo->threadPool[thread->id].inCpuSet)
            {
                processorIds[startidx+(numberOfEntries-counter)] = thread->id;
                thread = tree_getNextNode(thread);
                counter--;
            }
            else
            {
                thread = tree_getNextNode(thread);
            }
        }
        c_count++;
        node = tree_getNextNode(node);
    }
    return numberOfEntries-counter;
}

static int get_id_of_type(hwloc_obj_t base, hwloc_obj_type_t type)
{
    hwloc_obj_t walker = base->parent;
    while (walker && walker->type != type)
        walker = walker->parent;
    if (walker && walker->type == type)
        return walker->os_index;
    return -1;
}

#define AFF_FREE_AND_RESET(ptr) if (ptr != NULL) { free(ptr); ptr = NULL; }

static int create_lookups()
{
    int err = 0;
    int do_cache = 1;
    int cachelimit = 0;
    int cacheIdx = -1;
    int * tmp = NULL;
    int num_tmp = 0;
    err = topology_init();
    if (err != 0)
    {
        return err;
    }
    err = numa_init();
    if (err != 0)
    {
        return err;
    }
    CpuTopology_t cputopo = get_cpuTopology();
    NumaTopology_t ntopo = get_numaTopology();
    if (!affinity_thread2core_lookup)
    {
        affinity_thread2core_lookup = malloc(cputopo->numHWThreads * sizeof(int));
        if (!affinity_thread2core_lookup)
        {
            return -ENOMEM;
        }
        memset(affinity_thread2core_lookup, -1, cputopo->numHWThreads*sizeof(int));
    }
    if (!affinity_thread2socket_lookup)
    {
        affinity_thread2socket_lookup = malloc(cputopo->numHWThreads * sizeof(int));
        if (!affinity_thread2socket_lookup)
        {
            AFF_FREE_AND_RESET(affinity_thread2core_lookup);
            return -ENOMEM;
        }
        memset(affinity_thread2socket_lookup, -1, cputopo->numHWThreads*sizeof(int));
    }
    if (!affinity_thread2sharedl3_lookup)
    {
        affinity_thread2sharedl3_lookup = malloc(cputopo->numHWThreads * sizeof(int));
        if (!affinity_thread2sharedl3_lookup)
        {
            AFF_FREE_AND_RESET(affinity_thread2socket_lookup);
            AFF_FREE_AND_RESET(affinity_thread2core_lookup);
            return -ENOMEM;
        }
        memset(affinity_thread2sharedl3_lookup, -1, cputopo->numHWThreads*sizeof(int));
    }
    if (!affinity_thread2numa_lookup)
    {
        affinity_thread2numa_lookup = malloc(cputopo->numHWThreads * sizeof(int));
        if (!affinity_thread2numa_lookup)
        {
            AFF_FREE_AND_RESET(affinity_thread2sharedl3_lookup);
            AFF_FREE_AND_RESET(affinity_thread2socket_lookup);
            AFF_FREE_AND_RESET(affinity_thread2core_lookup);
            return -ENOMEM;
        }
        memset(affinity_thread2numa_lookup, -1, cputopo->numHWThreads*sizeof(int));
    }
    if (!affinity_thread2die_lookup)
    {
        affinity_thread2die_lookup = malloc(cputopo->numHWThreads * sizeof(int));
        if (!affinity_thread2die_lookup)
        {
            AFF_FREE_AND_RESET(affinity_thread2numa_lookup);
            AFF_FREE_AND_RESET(affinity_thread2sharedl3_lookup);
            AFF_FREE_AND_RESET(affinity_thread2socket_lookup);
            AFF_FREE_AND_RESET(affinity_thread2core_lookup);
            return -ENOMEM;
        }
        memset(affinity_thread2die_lookup, -1, cputopo->numHWThreads*sizeof(int));
    }
    if (!socket_lock)
    {
        socket_lock = malloc(cputopo->numHWThreads * sizeof(int));
        if (!socket_lock)
        {
            AFF_FREE_AND_RESET(affinity_thread2die_lookup);
            AFF_FREE_AND_RESET(affinity_thread2numa_lookup);
            AFF_FREE_AND_RESET(affinity_thread2sharedl3_lookup);
            AFF_FREE_AND_RESET(affinity_thread2socket_lookup);
            AFF_FREE_AND_RESET(affinity_thread2core_lookup);
            return -ENOMEM;
        }
        memset(socket_lock, LOCK_INIT, cputopo->numHWThreads*sizeof(int));
    }
    if (!die_lock)
    {
        die_lock = malloc(cputopo->numHWThreads * sizeof(int));
        if (!die_lock)
        {
            AFF_FREE_AND_RESET(socket_lock);
            AFF_FREE_AND_RESET(affinity_thread2die_lookup);
            AFF_FREE_AND_RESET(affinity_thread2numa_lookup);
            AFF_FREE_AND_RESET(affinity_thread2sharedl3_lookup);
            AFF_FREE_AND_RESET(affinity_thread2socket_lookup);
            AFF_FREE_AND_RESET(affinity_thread2core_lookup);
            return -ENOMEM;
        }
        memset(die_lock, LOCK_INIT, cputopo->numHWThreads*sizeof(int));
    }
    if (!numa_lock)
    {
        numa_lock = malloc(cputopo->numHWThreads * sizeof(int));
        if (!numa_lock)
        {
            AFF_FREE_AND_RESET(die_lock);
            AFF_FREE_AND_RESET(socket_lock);
            AFF_FREE_AND_RESET(affinity_thread2die_lookup);
            AFF_FREE_AND_RESET(affinity_thread2numa_lookup);
            AFF_FREE_AND_RESET(affinity_thread2sharedl3_lookup);
            AFF_FREE_AND_RESET(affinity_thread2socket_lookup);
            AFF_FREE_AND_RESET(affinity_thread2core_lookup);
            return -ENOMEM;
        }
        memset(numa_lock, LOCK_INIT, cputopo->numHWThreads*sizeof(int));
    }
    if (!core_lock)
    {
        core_lock = malloc(cputopo->numHWThreads * sizeof(int));
        if (!core_lock)
        {
            AFF_FREE_AND_RESET(numa_lock);
            AFF_FREE_AND_RESET(die_lock);
            AFF_FREE_AND_RESET(socket_lock);
            AFF_FREE_AND_RESET(affinity_thread2die_lookup);
            AFF_FREE_AND_RESET(affinity_thread2numa_lookup);
            AFF_FREE_AND_RESET(affinity_thread2sharedl3_lookup);
            AFF_FREE_AND_RESET(affinity_thread2socket_lookup);
            AFF_FREE_AND_RESET(affinity_thread2core_lookup);
            return -ENOMEM;
        }
        memset(core_lock, LOCK_INIT, cputopo->numHWThreads*sizeof(int));
    }
    if (!tile_lock)
    {
        tile_lock = malloc(cputopo->numHWThreads * sizeof(int));
        if (!tile_lock)
        {
            AFF_FREE_AND_RESET(core_lock);
            AFF_FREE_AND_RESET(numa_lock);
            AFF_FREE_AND_RESET(die_lock);
            AFF_FREE_AND_RESET(socket_lock);
            AFF_FREE_AND_RESET(affinity_thread2die_lookup);
            AFF_FREE_AND_RESET(affinity_thread2numa_lookup);
            AFF_FREE_AND_RESET(affinity_thread2sharedl3_lookup);
            AFF_FREE_AND_RESET(affinity_thread2socket_lookup);
            AFF_FREE_AND_RESET(affinity_thread2core_lookup);
            return -ENOMEM;
        }
        memset(tile_lock, LOCK_INIT, cputopo->numHWThreads*sizeof(int));
    }
    if (!sharedl2_lock)
    {
        sharedl2_lock = malloc(cputopo->numHWThreads * sizeof(int));
        if (!sharedl2_lock)
        {
            AFF_FREE_AND_RESET(tile_lock);
            AFF_FREE_AND_RESET(core_lock);
            AFF_FREE_AND_RESET(numa_lock);
            AFF_FREE_AND_RESET(die_lock);
            AFF_FREE_AND_RESET(socket_lock);
            AFF_FREE_AND_RESET(affinity_thread2die_lookup);
            AFF_FREE_AND_RESET(affinity_thread2numa_lookup);
            AFF_FREE_AND_RESET(affinity_thread2sharedl3_lookup);
            AFF_FREE_AND_RESET(affinity_thread2socket_lookup);
            AFF_FREE_AND_RESET(affinity_thread2core_lookup);
            return -ENOMEM;
        }
        memset(sharedl2_lock, LOCK_INIT, cputopo->numHWThreads*sizeof(int));
    }
    if (!sharedl3_lock)
    {
        sharedl3_lock = malloc(cputopo->numHWThreads * sizeof(int));
        if (!sharedl3_lock)
        {
            AFF_FREE_AND_RESET(sharedl2_lock);
            AFF_FREE_AND_RESET(tile_lock);
            AFF_FREE_AND_RESET(core_lock);
            AFF_FREE_AND_RESET(numa_lock);
            AFF_FREE_AND_RESET(die_lock);
            AFF_FREE_AND_RESET(socket_lock);
            AFF_FREE_AND_RESET(affinity_thread2die_lookup);
            AFF_FREE_AND_RESET(affinity_thread2numa_lookup);
            AFF_FREE_AND_RESET(affinity_thread2sharedl3_lookup);
            AFF_FREE_AND_RESET(affinity_thread2socket_lookup);
            AFF_FREE_AND_RESET(affinity_thread2core_lookup);
            return -ENOMEM;
        }
        memset(sharedl3_lock, LOCK_INIT, cputopo->numHWThreads*sizeof(int));
    }
    tmp = malloc(cputopo->numHWThreads * sizeof(int));
    if (!tmp)
    {
        AFF_FREE_AND_RESET(sharedl3_lock);
        AFF_FREE_AND_RESET(sharedl2_lock);
        AFF_FREE_AND_RESET(tile_lock);
        AFF_FREE_AND_RESET(core_lock);
        AFF_FREE_AND_RESET(numa_lock);
        AFF_FREE_AND_RESET(die_lock);
        AFF_FREE_AND_RESET(socket_lock);
        AFF_FREE_AND_RESET(affinity_thread2die_lookup);
        AFF_FREE_AND_RESET(affinity_thread2numa_lookup);
        AFF_FREE_AND_RESET(affinity_thread2sharedl3_lookup);
        AFF_FREE_AND_RESET(affinity_thread2socket_lookup);
        AFF_FREE_AND_RESET(affinity_thread2core_lookup);
        return -ENOMEM;
    }

    int num_pu = cputopo->numHWThreads;
    if (cputopo->numCacheLevels == 0)
    {
        do_cache = 0;
    }
    if (do_cache)
    {
        cachelimit = cputopo->cacheLevels[cputopo->numCacheLevels-1].threads;
        cacheIdx = -1;
    }
    for (int pu_idx = 0; pu_idx < num_pu; pu_idx++)
    {
        HWThread* t = &cputopo->threadPool[pu_idx];
        int found = 0;
        for (int j = 0; j < num_tmp; j++)
        {
            if (tmp[j] == t->packageId)
            {
                found = 1;
                break;
            }
        }
        if (!found)
        {
            tmp[num_tmp++] = t->packageId;
        }
    }
    for (int pu_idx = 0; pu_idx < num_pu; pu_idx++)
    {
        HWThread* t = &cputopo->threadPool[pu_idx];
        int hwthreadid = t->apicId;
        int coreid = t->coreId;
        int dieid = t->dieId;
        int sockid = t->packageId;
        int dies_per_socket = MAX(cputopo->numDies/cputopo->numSockets, 1);
        affinity_thread2core_lookup[hwthreadid] = coreid;
        affinity_thread2socket_lookup[hwthreadid] = sockid;
        for (int j = 0; j < num_tmp; j++)
        {
            if (affinity_thread2socket_lookup[hwthreadid] == tmp[j])
            {
                if (affinity_thread2socket_lookup[hwthreadid] != j)
                {
                    affinity_thread2socket_lookup[hwthreadid] = j;
                    sockid = j;
                }
            }
        }
        affinity_thread2die_lookup[hwthreadid] = (sockid * dies_per_socket) + dieid;
        int memid = 0;
        for (int n = 0; n < ntopo->numberOfNodes; n++)
        {
            for (int i = 0; i < ntopo->nodes[n].numberOfProcessors; i++)
            {
                if (ntopo->nodes[n].processors[i] == hwthreadid)
                {
                    memid = n;
                    break;
                }
            }
        }
        affinity_thread2numa_lookup[hwthreadid] = memid;
        if (do_cache && cachelimit > 0)
        {
            int numberOfCoresPerCache = cachelimit/cputopo->numThreadsPerCore;
            affinity_thread2sharedl3_lookup[hwthreadid] = coreid / numberOfCoresPerCache;
        }
        DEBUG_PRINT(DEBUGLEV_DEVELOP, T %d T2C %d T2S %d T2D %d T2LLC %d T2M %d, hwthreadid,
                                        affinity_thread2core_lookup[hwthreadid],
                                        affinity_thread2socket_lookup[hwthreadid],
                                        affinity_thread2die_lookup[hwthreadid],
                                        affinity_thread2sharedl3_lookup[hwthreadid],
                                        affinity_thread2numa_lookup[hwthreadid]);
    }
    free(tmp);
    return 0;
}

static int affinity_getPoolId(int cpuId)
{
    CpuTopology_t cputopo = get_cpuTopology();
    for (int i = 0; i < cputopo->numHWThreads; i++)
    {
        if (cputopo->threadPool[i].apicId == cpuId)
        {
            return i;
        }
    }
    return -1;
}

static int affinity_countSocketCores(int len, int* hwthreads, int* helper)
{
    CpuTopology_t cputopo = get_cpuTopology();
    int hidx = 0;
    for (int i = 0; i < len; i++)
    {
        int pid = affinity_getPoolId(hwthreads[i]);
        if (pid >= 0)
        {
            int found = 0;
            for (int k = 0; k < hidx; k++)
            {
                if (helper[k] == cputopo->threadPool[pid].coreId)
                {
                    found = 1;
                }
            }
            if (!found)
            {
                helper[hidx++] = cputopo->threadPool[pid].coreId;
            }
        }
    }
    return hidx;
}

static int affinity_addNodeDomain(AffinityDomain* domain, int* help)
{
    CpuTopology_t cputopo = get_cpuTopology();
    if (!domain)
    {
        return -EINVAL;
    }
    if (cputopo)
    {
        domain->numberOfProcessors = cputopo->activeHWThreads;
        domain->processorList = malloc(cputopo->activeHWThreads * sizeof(int));
        if (!domain->processorList)
        {
            return -ENOMEM;
        }
        int offset = 0;
        int cores = 0;
        for (int i = 0; i < MAX(cputopo->numSockets, 1); i++)
        {
            int tmp = treeFillNextEntries(cputopo->topologyTree,
                                          domain->processorList, offset,
                                          i, 0,
                                          cputopo->numCoresPerSocket,
                                          cputopo->numCoresPerSocket*cputopo->numThreadsPerCore);
            cores += affinity_countSocketCores(tmp, &domain->processorList[offset], help);
            offset += tmp;
        }
        domain->numberOfProcessors = offset;
        domain->numberOfCores = cores;
        domain->tag = bformat("N");
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity domain N: %d HW threads on %d cores, domain->numberOfProcessors, domain->numberOfCores);
        return 0;
    }
    return -EINVAL;
}

static int affinity_addSocketDomain(int socket, AffinityDomain* domain, int* help)
{
    CpuTopology_t cputopo = get_cpuTopology();
    if (!domain)
    {
        return -EINVAL;
    }
    if (cputopo)
    {
        domain->numberOfProcessors = cputopo->numCoresPerSocket * cputopo->numThreadsPerCore;
        domain->processorList = malloc(cputopo->activeHWThreads * sizeof(int));
        if (!domain->processorList)
        {
            return -ENOMEM;
        }
        int tmp = treeFillNextEntries(cputopo->topologyTree,
                                      domain->processorList,
                                      0,
                                      socket, 0,
                                      cputopo->numCoresPerSocket,
                                      domain->numberOfProcessors);
        tmp = MIN(tmp, domain->numberOfProcessors);
        domain->numberOfProcessors = tmp;
        domain->numberOfCores = affinity_countSocketCores(domain->numberOfProcessors, domain->processorList, help);
        domain->tag = bformat("S%d", socket);
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity domain S%d: %d HW threads on %d cores, socket, domain->numberOfProcessors, domain->numberOfCores);
        return 0;
    }
    return -EINVAL;
}

static int affinity_addDieDomain(int socket, int die, AffinityDomain* domain, int* help)
{
    CpuTopology_t cputopo = get_cpuTopology();
    if (!domain)
    {
        return -EINVAL;
    }
    if (cputopo)
    {
        int numDiesPerSocket = MAX(cputopo->numDies/cputopo->numSockets, 1);
        int dieId = (socket * numDiesPerSocket) + die;
        int numCoresPerDie = cputopo->numCoresPerSocket/numDiesPerSocket;
        int numThreadsPerDie = numCoresPerDie * cputopo->numThreadsPerCore;
        int coreOffset = die * numCoresPerDie;
        domain->processorList = malloc(numThreadsPerDie * sizeof(int));
        if (!domain->processorList)
        {
            return -ENOMEM;
        }
        domain->numberOfProcessors = numThreadsPerDie;
        domain->numberOfCores = numCoresPerDie;
        int tmp = treeFillNextEntries(cputopo->topologyTree,
                                      domain->processorList,
                                      0,
                                      socket,
                                      die * numCoresPerDie,
                                      numCoresPerDie,
                                      domain->numberOfProcessors);
        domain->numberOfProcessors = tmp;
        domain->numberOfCores = affinity_countSocketCores(domain->numberOfProcessors, domain->processorList, help);
        domain->tag = bformat("D%d", dieId);
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity domain D%d: %d HW threads on %d cores, dieId, domain->numberOfProcessors, domain->numberOfCores);
        return 0;
    }
    return -EINVAL;
}

static int affinity_addCacheDomain(int socket, int cacheId, AffinityDomain* domain, int* help)
{
    CpuTopology_t cputopo = get_cpuTopology();
    if (!domain)
    {
        return -EINVAL;
    }
    if (cputopo && cputopo->numCacheLevels > 0)
    {
        int numThreadsPerCache = cputopo->cacheLevels[cputopo->numCacheLevels-1].threads;
        int numCoresPerCache = numThreadsPerCache / cputopo->numThreadsPerCore;
        int numCachesPerSocket = cputopo->numCoresPerSocket / numCoresPerCache;
        int cid = (socket * numCachesPerSocket) + cacheId;
        domain->processorList = malloc(numThreadsPerCache * sizeof(int));
        if (!domain->processorList)
        {
            return -ENOMEM;
        }
        domain->numberOfProcessors = numThreadsPerCache;
        domain->numberOfCores = numCoresPerCache;
        int tmp = treeFillNextEntries(cputopo->topologyTree,
                                      domain->processorList,
                                      0,
                                      socket,
                                      cacheId * numCoresPerCache,
                                      numCoresPerCache,
                                      domain->numberOfProcessors);
        domain->numberOfProcessors = tmp;
        domain->numberOfCores = affinity_countSocketCores(domain->numberOfProcessors, domain->processorList, help);
        domain->tag = bformat("C%d", cid);
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity domain C%d: %d HW threads on %d cores, cid, domain->numberOfProcessors, domain->numberOfCores);
        return 0;
    }
    return -EINVAL;
}

static int _affinity_addMemoryDomain(int nodeId, AffinityDomain* domain, int* help)
{
    CpuTopology_t cputopo = get_cpuTopology();
    NumaTopology_t numatopo = get_numaTopology();
    int num_hwthreads = 0;
    if ((nodeId < 0) || (!domain))
    {
        return -EINVAL;
    }
    if (cputopo && numatopo)
    {
        if (nodeId >= 0 && nodeId < numatopo->numberOfNodes)
        {
            domain->processorList = malloc(numatopo->nodes[nodeId].numberOfProcessors * sizeof(int));
            if (!domain->processorList)
            {
                return -ENOMEM;
            }
            for (int i = 0; i < numatopo->nodes[nodeId].numberOfProcessors; i++)
            {
                for (int j = 0; j < cputopo->numHWThreads; j++)
                {
                    if (cputopo->threadPool[j].apicId == numatopo->nodes[nodeId].processors[i] && cputopo->threadPool[j].inCpuSet == 1)
                    {
                        domain->processorList[num_hwthreads++] = (int)numatopo->nodes[nodeId].processors[i];
                    }
                }
            }
            domain->numberOfProcessors = num_hwthreads;
            domain->numberOfCores = affinity_countSocketCores(domain->numberOfProcessors, domain->processorList, help);
            domain->tag = bformat("M%d", nodeId);
            
            return 0;
        }
    }
    return -EINVAL;
}

static int affinity_addMemoryDomain(int nodeId, AffinityDomain* domain, int* help)
{
    int err = _affinity_addMemoryDomain(nodeId, domain, help);
    if (err == 0)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity domain M%d: %d HW threads on %d cores, nodeId, domain->numberOfProcessors, domain->numberOfCores);
    }
    return err;
}

#ifdef LIKWID_WITH_NVMON
static int affinity_addCudaDomain(int nodeId, AffinityDomain* domain, int offset, int* help)
{
    int err = 0;
    CpuTopology_t cputopo = get_cpuTopology();
    NumaTopology_t numatopo = get_numaTopology();
    CudaTopology_t cudatopo = get_cudaTopology();

    if ((nodeId < 0) || (!domain))
    {
        return -EINVAL;
    }
    if (cputopo && numatopo && cudatopo)
    {
        if (nodeId >= 0 && nodeId < cudatopo->numDevices)
        {
            CudaDevice* cudadev = &cudatopo->devices[nodeId];
            if (cudadev->numaNode >= 0)
            {
                err = _affinity_addMemoryDomain(cudadev->numaNode, domain, help);
                if (err == 0)
                {
                    bdestroy(domain->tag);
                    domain->tag = bformat("G%d", nodeId+offset);
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity domain %s: %d HW threads on %d cores, bdata(domain->tag), domain->numberOfProcessors, domain->numberOfCores);
                    return 0;
                }
            }
        }
    }
    return -EINVAL;
}
#endif

#ifdef LIKWID_WITH_ROCMON
static int affinity_addRocmDomain(int nodeId, AffinityDomain* domain, int offset, int* help)
{
    int err = 0;
    CpuTopology_t cputopo = get_cpuTopology();
    NumaTopology_t numatopo = get_numaTopology();
    RocmTopology_t rocmtopo = get_rocmTopology();

    if ((nodeId < 0) || (!domain))
    {
        return -EINVAL;
    }
    if (cputopo && numatopo && rocmtopo)
    {
        if (nodeId >= 0 && nodeId < rocmtopo->numDevices)
        {
            RocmDevice* rocmdev = &rocmtopo->devices[nodeId];
            if (rocmdev->numaNode >= 0)
            {
                err = _affinity_addMemoryDomain(rocmdev->numaNode, domain, help);
                if (err == 0)
                {
                    bdestroy(domain->tag);
                    domain->tag = bformat("G%d", nodeId+offset);
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity domain %s: %d HW threads on %d cores, bdata(domain->tag), domain->numberOfProcessors, domain->numberOfCores);
                    return 0;
                }
            }
            else
            {
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Skipping affinity domain G%d because NUMA node unknown, nodeId);
            }
        }
    }
    return -EINVAL;
}
#endif

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
affinity_init()
{
    int err = 0;
    int numberOfDomains = 1; /* all systems have the node domain */
    int finalNumberOfDomains = 0;
    int currentDomain;
    int subCounter = 0;
    int offset = 0;
    int domid = 0;
    int tmp = 0;
    if (affinity_initialized == 1)
    {
        return 0;
    }
    err = topology_init();
    if (err != 0)
    {
        return err;
    }
    CpuTopology_t cputopo = get_cpuTopology();
    CpuInfo_t cpuinfo = get_cpuInfo();
    err = numa_init();
    if (err != 0)
    {
        return err;
    }
    NumaTopology_t numatopo = get_numaTopology();

#ifdef LIKWID_WITH_NVMON
    int numCudaDomains = 0;
    CudaTopology_t cudatopo = NULL;
    err = topology_cuda_init();
    if (err != 0)
    {
        return err;
    }
#endif
#ifdef LIKWID_WITH_ROCMON
    int numRocmDomains = 0;
    RocmTopology_t rocmtopo = NULL;
    err = topology_rocm_init();
    if (err != 0)
    {
        return err;
    }
#endif

    int doCacheDomains = 1;
    int numberOfCacheDomains = 0;
    int numberOfCoresPerCache = 0;
    int numberOfProcessorsPerCache = 0;

    /* check system and remove domains if needed */
    if (cpuinfo->vendor == APPLE_M1 && cpuinfo->model == APPLE_M1_STUDIO)
    {
        doCacheDomains = 0;
    }

    /* determine total number of domains */
    numberOfDomains = 1;
    numberOfDomains += cputopo->numSockets;
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity: Socket domains %d, cputopo->numSockets);
    numberOfDomains += (cputopo->numDies > 0 ? cputopo->numDies : cputopo->numSockets);
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity: CPU die domains %d, (cputopo->numDies > 0 ? cputopo->numDies : cputopo->numSockets));
    if (doCacheDomains && cputopo->numCacheLevels > 0)
    {
        numberOfProcessorsPerCache = cputopo->cacheLevels[cputopo->numCacheLevels-1].threads;
        numberOfCoresPerCache = numberOfProcessorsPerCache / cputopo->numThreadsPerCore;
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity: CPU cores per LLC %d, numberOfCoresPerCache);
        int numCachesPerSocket = cputopo->numCoresPerSocket / numberOfCoresPerCache;
        numberOfCacheDomains = cputopo->numSockets * MAX(numCachesPerSocket, 1);
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity: Cache domains %d, numberOfCacheDomains);
        numberOfDomains += numberOfCacheDomains;
    }
    numberOfDomains += numatopo->numberOfNodes;
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity: NUMA domains %d, numatopo->numberOfNodes);
#if defined(LIKWID_WITH_NVMON) || defined(LIKWID_WITH_ROCMON)
    int gpuDomains = 0;
#ifdef LIKWID_WITH_NVMON
    cudatopo = get_cudaTopology();
    for (int i = 0; i < cudatopo->numDevices; i++)
    {
        CudaDevice* dev = &cudatopo->devices[i];
        if (dev->numaNode >= 0)
        {
            numCudaDomains++;
        }
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity: CUDA domains %d (%d device(s)), numCudaDomains, cudatopo->numDevices);
    gpuDomains += numCudaDomains;
#endif
#ifdef LIKWID_WITH_ROCMON
    rocmtopo = get_rocmTopology();
    for (int i = 0; i < rocmtopo->numDevices; i++)
    {
        RocmDevice* dev = &rocmtopo->devices[i];
        if (dev->numaNode >= 0)
        {
            numRocmDomains++;
        }
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity: ROCm domains %d (%d device(s)), numRocmDomains, rocmtopo->numDevices);
    gpuDomains += numRocmDomains;
#endif
    numberOfDomains += gpuDomains;
#endif
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity: All domains %d, numberOfDomains);

    domains = (AffinityDomain*) malloc(numberOfDomains * sizeof(AffinityDomain));
    if (!domains)
    {
        fprintf(stderr,"No more memory for %ld bytes for array of affinity domains\n",numberOfDomains * sizeof(AffinityDomain));
        return -ENOMEM;
    }
    memset(domains, 0, numberOfDomains * sizeof(AffinityDomain));
    int* helper = malloc(cputopo->numHWThreads * sizeof(int));
    if (!helper)
    {
        free(domains);
        return -ENOMEM;
    }

    /* Node domain */
    err = affinity_addNodeDomain(&domains[domid], helper);
    if (!err)
    {
        domid++;
        finalNumberOfDomains++;
    }
    /* Socket domains */
    for (int i = 0; i < cputopo->numSockets; i++)
    {
        err = affinity_addSocketDomain(i, &domains[domid], helper);
        if (!err)
        {
            domid++;
            finalNumberOfDomains++;
        }
    }
    /* CPU die domains */
    for (int i = 0; i < cputopo->numSockets; i++)
    {
        int numDiesPerSocket = 1;
        if (cputopo->numDies > 0)
        {
            numDiesPerSocket = cputopo->numDies/cputopo->numSockets;
        }
        for (int j = 0; j < numDiesPerSocket; j++)
        {
            err = affinity_addDieDomain(i, j, &domains[domid], helper);
            if (!err)
            {
                domid++;
                finalNumberOfDomains++;
            }
        }
    }
    /* Last level cache domains */
    if (doCacheDomains && cputopo->numCacheLevels > 0)
    {
        for (int i = 0; i < cputopo->numSockets; i++)
        {
            int numThreadPerCache = cputopo->cacheLevels[cputopo->numCacheLevels-1].threads;
            int numCoresPerCache = numThreadPerCache / cputopo->numThreadsPerCore;
            int numCachesPerSocket = cputopo->numCoresPerSocket / numCoresPerCache;
            for (int j = 0; j < MAX(numCachesPerSocket, 1); j++)
            {
                err = affinity_addCacheDomain(i, j, &domains[domid], helper);
                if (!err)
                {
                    domid++;
                    finalNumberOfDomains++;
                }
            }
        }
    }
    /* Memory / NUMA domains */
    for (int i = 0; i < numatopo->numberOfNodes; i++)
    {
        err = affinity_addMemoryDomain(i, &domains[domid], helper);
        if (!err)
        {
            domid++;
            finalNumberOfDomains++;
        }
    }
#ifdef LIKWID_WITH_NVMON
    for (int i = 0; i < cudatopo->numDevices; i++)
    {
        err = affinity_addCudaDomain(i, &domains[domid], 0, helper);
        if (!err)
        {
            domid++;
            finalNumberOfDomains++;
        }
    }
#endif
#ifdef LIKWID_WITH_ROCMON
    int gpuOffset = 0;
#ifdef LIKWID_WITH_NVMON
    gpuOffset = numCudaDomains;
#endif
    for (int i = 0; i < rocmtopo->numDevices; i++)
    {
        err = affinity_addRocmDomain(i, &domains[domid], gpuOffset, helper);
        if (!err)
        {
            domid++;
            finalNumberOfDomains++;
        }
    }
#endif
    free(helper);

    affinity_numberOfDomains = numberOfDomains;
    affinityDomains.numberOfAffinityDomains = numberOfDomains;
    affinityDomains.numberOfSocketDomains = cputopo->numSockets;
    affinityDomains.numberOfNumaDomains = numatopo->numberOfNodes;
    affinityDomains.numberOfProcessorsPerSocket = cputopo->numCoresPerSocket * cputopo->numThreadsPerCore;
    affinityDomains.numberOfCacheDomains = numberOfCacheDomains;
    affinityDomains.numberOfCoresPerCache = numberOfCoresPerCache;
    affinityDomains.numberOfProcessorsPerCache = numberOfProcessorsPerCache;
    affinityDomains.numberOfCudaDomains = 0;
#ifdef LIKWID_WITH_NVMON
    affinityDomains.numberOfCudaDomains = numCudaDomains;
#endif
    affinityDomains.numberOfRocmDomains = 0;
#ifdef LIKWID_WITH_ROCMON
    affinityDomains.numberOfRocmDomains = numRocmDomains;
#endif
    affinityDomains.domains = domains;

    create_lookups();

    affinity_initialized = 1;
    return 0;
}

void
affinity_finalize()
{
    if (affinity_initialized == 0)
    {
        return;
    }
    if (!affinityDomains.domains)
    {
        return;
    }
    for ( int i=0; i < affinityDomains.numberOfAffinityDomains; i++ )
    {
        if (affinityDomains.domains[i].tag)
            bdestroy(affinityDomains.domains[i].tag);
        if (affinityDomains.domains[i].processorList != NULL)
        {
            free(affinityDomains.domains[i].processorList);
        }
        affinityDomains.domains[i].processorList = NULL;
    }
    if (affinityDomains.domains != NULL)
    {
        free(affinityDomains.domains);
        affinityDomains.domains = NULL;
    }
    if (affinity_thread2core_lookup)
    {
        free(affinity_thread2core_lookup);
        affinity_thread2core_lookup = NULL;
    }
    if (affinity_thread2socket_lookup)
    {
        free(affinity_thread2socket_lookup);
        affinity_thread2socket_lookup = NULL;
    }
    if (affinity_thread2sharedl3_lookup)
    {
        free(affinity_thread2sharedl3_lookup);
        affinity_thread2sharedl3_lookup = NULL;
    }
    if (affinity_thread2numa_lookup)
    {
        free(affinity_thread2numa_lookup);
        affinity_thread2numa_lookup = NULL;
    }
    if (affinity_thread2die_lookup)
    {
        free(affinity_thread2die_lookup);
        affinity_thread2die_lookup = NULL;
    }
    if (socket_lock)
    {
        free(socket_lock);
        socket_lock = NULL;
    }
    if (die_lock)
    {
        free(die_lock);
        die_lock = NULL;
    }
    if (numa_lock)
    {
        free(numa_lock);
        numa_lock = NULL;
    }
    if (tile_lock)
    {
        free(tile_lock);
        tile_lock = NULL;
    }
    if (core_lock)
    {
        free(core_lock);
        core_lock = NULL;
    }
    if (sharedl2_lock)
    {
        free(sharedl2_lock);
        sharedl2_lock = NULL;
    }
    if (sharedl3_lock)
    {
        free(sharedl3_lock);
        sharedl3_lock = NULL;
    }


    affinityDomains.domains = NULL;
    affinity_numberOfDomains = 0;
    affinityDomains.numberOfAffinityDomains = 0;
    affinityDomains.numberOfSocketDomains = 0;
    affinityDomains.numberOfNumaDomains = 0;
    affinityDomains.numberOfProcessorsPerSocket = 0;
    affinityDomains.numberOfCacheDomains = 0;
    affinityDomains.numberOfCoresPerCache = 0;
    affinityDomains.numberOfProcessorsPerCache = 0;
    affinity_initialized = 0;
}

int
affinity_processGetProcessorId()
{
    int ret;
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    ret = sched_getaffinity(getpid(),sizeof(cpu_set_t), &cpu_set);

    if (ret < 0)
    {
        ERROR;
    }

    return getProcessorID(&cpu_set);
}

int
affinity_threadGetProcessorId()
{
    cpu_set_t  cpu_set;
    CPU_ZERO(&cpu_set);
    sched_getaffinity(gettid(),sizeof(cpu_set_t), &cpu_set);

    return getProcessorID(&cpu_set);
}

#ifdef HAS_SCHEDAFFINITY
void
affinity_pinThread(int processorId)
{
    cpu_set_t cpuset;
    pthread_t thread;

    thread = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(processorId, &cpuset);
    pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
}
#else
void
affinity_pinThread(int processorId)
{
}
#endif

void
affinity_pinProcess(int processorId)
{
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET(processorId, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}

void
affinity_pinProcesses(int cpu_count, const int* processorIds)
{
    int i;
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    for(i=0;i<cpu_count;i++)
    {
        CPU_SET(processorIds[i], &cpuset);
    }
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}

const AffinityDomain*
affinity_getDomain(bstring domain)
{

    for ( int i=0; i < affinity_numberOfDomains; i++ )
    {
        if ( biseq(domain, domains[i].tag) )
        {
            return domains+i;
        }
    }

    return NULL;
}

void
affinity_printDomains()
{
    for ( int i=0; i < affinity_numberOfDomains; i++ )
    {
        printf("Domain %d:\n",i);
        printf("\tTag %s:",bdata(domains[i].tag));

        for ( uint32_t j=0; j < domains[i].numberOfProcessors; j++ )
        {
            printf(" %d",domains[i].processorList[j]);
        }
        printf("\n");
    }
}

AffinityDomains_t
get_affinityDomains(void)
{
    return &affinityDomains;
}
