/*
 * =======================================================================================
 *
 *      Filename:  affinity.c
 *
 *      Description:  Implementation of affinity module.
 *
 *      Version:   4.2
 *      Released:  22.12.2016
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com,
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
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
#include <tree.h>
#include <topology.h>

/* #####   EXPORTED VARIABLES   ########################################### */

int affinity_core2node_lookup[MAX_NUM_THREADS];
int affinity_thread2core_lookup[MAX_NUM_THREADS];

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

    for ( processorId = 0; processorId < MAX_NUM_THREADS; processorId++ )
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
            if (cpuid_topology.threadPool[thread->id].inCpuSet)
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
/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
affinity_init()
{
    int numberOfDomains = 1; /* all systems have the node domain */
    int currentDomain;
    int subCounter = 0;
    int offset = 0;
    int tmp;
    if (affinity_initialized == 1)
    {
        return;
    }
    topology_init();
    int numberOfSocketDomains = cpuid_topology.numSockets;
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity: Socket domains %d, numberOfSocketDomains);
    numa_init();
    int numberOfNumaDomains = numa_info.numberOfNodes;
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity: NUMA domains %d, numberOfNumaDomains);
    int numberOfProcessorsPerSocket =
        cpuid_topology.numCoresPerSocket * cpuid_topology.numThreadsPerCore;
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity: CPUs per socket %d, numberOfProcessorsPerSocket);
    int numberOfCacheDomains;

    int numberOfCoresPerCache =
        cpuid_topology.cacheLevels[cpuid_topology.numCacheLevels-1].threads/
        cpuid_topology.numThreadsPerCore;
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity: CPU cores per LLC %d, numberOfCoresPerCache);

    int numberOfProcessorsPerCache =
        cpuid_topology.cacheLevels[cpuid_topology.numCacheLevels-1].threads;
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity: CPUs per LLC %d, numberOfProcessorsPerCache);
    /* for the cache domain take only into account last level cache and assume
     * all sockets to be uniform. */

    /* determine how many last level shared caches exist per socket */
    numberOfCacheDomains = cpuid_topology.numSockets *
        (cpuid_topology.numCoresPerSocket/numberOfCoresPerCache);
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity: Cache domains %d, numberOfCacheDomains);
    /* determine total number of domains */
    numberOfDomains += numberOfSocketDomains + numberOfCacheDomains + numberOfNumaDomains;
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity: All domains %d, numberOfDomains);
    domains = (AffinityDomain*) malloc(numberOfDomains * sizeof(AffinityDomain));
    if (!domains)
    {
        fprintf(stderr,"No more memory for %ld bytes for array of affinity domains\n",numberOfDomains * sizeof(AffinityDomain));
        return;
    }

    /* Node domain */
    domains[0].numberOfProcessors = cpuid_topology.activeHWThreads;
    domains[0].numberOfCores = cpuid_topology.numSockets * cpuid_topology.numCoresPerSocket;
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity domain N: %d HW threads on %d cores, domains[0].numberOfProcessors, domains[0].numberOfCores);
    domains[0].tag = bformat("N");
    domains[0].processorList = (int*) malloc(cpuid_topology.numHWThreads*sizeof(int));
    if (!domains[0].processorList)
    {
        fprintf(stderr,"No more memory for %ld bytes for processor list of affinity domain %s\n",
                cpuid_topology.numHWThreads*sizeof(int), 
                bdata(domains[0].tag));
        return;
    }
    offset = 0;

    if (numberOfSocketDomains > 1)
    {
        for (int i=0; i<numberOfSocketDomains; i++)
        {
          tmp = treeFillNextEntries(cpuid_topology.topologyTree,
                                    domains[0].processorList, offset,
                                    i, 0,
                                    cpuid_topology.numCoresPerSocket,
                                    numberOfProcessorsPerSocket);
          offset += tmp;
        }
    }
    else
    {
        tmp = treeFillNextEntries(cpuid_topology.topologyTree,
                                  domains[0].processorList, 0,
                                  0, 0,
                                  domains[0].numberOfCores,
                                  domains[0].numberOfProcessors);
        domains[0].numberOfProcessors = tmp;
    }

    /* Socket domains */
    currentDomain = 1;
    for (int i=0; i < numberOfSocketDomains; i++ )
    {
        domains[currentDomain + i].numberOfProcessors = numberOfProcessorsPerSocket;
        domains[currentDomain + i].numberOfCores =  cpuid_topology.numCoresPerSocket;
        domains[currentDomain + i].tag = bformat("S%d", i);
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity domain S%d: %d HW threads on %d cores, i, domains[currentDomain + i].numberOfProcessors, domains[currentDomain + i].numberOfCores);
        domains[currentDomain + i].processorList = (int*) malloc( domains[currentDomain + i].numberOfProcessors * sizeof(int));
        if (!domains[currentDomain + i].processorList)
        {
            fprintf(stderr,"No more memory for %ld bytes for processor list of affinity domain %s\n",
                    domains[currentDomain + i].numberOfProcessors * sizeof(int),
                    bdata(domains[currentDomain + i].tag));
            return;
        }

        tmp = treeFillNextEntries(cpuid_topology.topologyTree,
                                  domains[currentDomain + i].processorList, 0,
                                  i, 0, cpuid_topology.numCoresPerSocket,
                                  domains[currentDomain + i].numberOfProcessors);
        tmp = MIN(tmp, domains[currentDomain + i].numberOfProcessors);
        for ( int j = 0; j < tmp; j++ )
        {
            affinity_core2node_lookup[domains[currentDomain + i].processorList[j]] = i;
        }
        domains[currentDomain + i].numberOfProcessors = tmp;
    }

    /* Cache domains */
    currentDomain += numberOfSocketDomains;
    subCounter = 0;
    for (int i=0; i < numberOfSocketDomains; i++ )
    {
        offset = 0;

        for ( int j=0; j < (numberOfCacheDomains/numberOfSocketDomains); j++ )
        {
            domains[currentDomain + subCounter].numberOfProcessors = numberOfProcessorsPerCache;
            domains[currentDomain + subCounter].numberOfCores =  numberOfCoresPerCache;
            domains[currentDomain + subCounter].tag = bformat("C%d", subCounter);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Affinity domain C%d: %d HW threads on %d cores, subCounter, domains[currentDomain + subCounter].numberOfProcessors, domains[currentDomain + subCounter].numberOfCores);
            domains[currentDomain + subCounter].processorList = (int*) malloc(numberOfProcessorsPerCache*sizeof(int));
            if (!domains[currentDomain + subCounter].processorList)   
            {
                fprintf(stderr,"No more memory for %ld bytes for processor list of affinity domain %s\n",
                        numberOfProcessorsPerCache*sizeof(int),
                        bdata(domains[currentDomain + subCounter].tag));
                return;
            }

            tmp = treeFillNextEntries(cpuid_topology.topologyTree,
                                      domains[currentDomain + subCounter].processorList, 0,
                                      i, offset, numberOfCoresPerCache,
                                      domains[currentDomain + subCounter].numberOfProcessors);

            domains[currentDomain + subCounter].numberOfProcessors = tmp;
            offset += (tmp < numberOfCoresPerCache ? tmp : numberOfCoresPerCache);
            subCounter++;
        }
    }
    /* Memory domains */
    currentDomain += numberOfCacheDomains;
    subCounter = 0;
    if ((numberOfNumaDomains >= numberOfSocketDomains) && (numberOfNumaDomains > 1))
    {
        for (int i=0; i < numberOfSocketDomains; i++ )
        {
            offset = 0;
            for ( int j=0; j < (int)ceil((double)(numberOfNumaDomains)/numberOfSocketDomains); j++ )
            {
                domains[currentDomain + subCounter].numberOfProcessors =
                                numa_info.nodes[subCounter].numberOfProcessors;

                domains[currentDomain + subCounter].numberOfCores =
                                numa_info.nodes[subCounter].numberOfProcessors/cpuid_topology.numThreadsPerCore;

                domains[currentDomain + subCounter].tag = bformat("M%d", subCounter);

                DEBUG_PRINT(DEBUGLEV_DEVELOP,
                        Affinity domain M%d: %d HW threads on %d cores,
                        subCounter, domains[currentDomain + subCounter].numberOfProcessors,
                        domains[currentDomain + subCounter].numberOfCores);

                domains[currentDomain + subCounter].processorList =
                                (int*) malloc(numa_info.nodes[subCounter].numberOfProcessors*sizeof(int));

                if (!domains[currentDomain + subCounter].processorList)
                {
                    fprintf(stderr,"No more memory for %ld bytes for processor list of affinity domain %s\n",
                            numa_info.nodes[subCounter].numberOfProcessors*sizeof(int),
                            bdata(domains[currentDomain + subCounter].tag));
                    return;
                }

                tmp = treeFillNextEntries(cpuid_topology.topologyTree,
                                          domains[currentDomain + subCounter].processorList, 0,
                                          i, offset, domains[currentDomain + subCounter].numberOfCores,
                                          domains[currentDomain + subCounter].numberOfProcessors);
                domains[currentDomain + subCounter].numberOfProcessors = tmp;
                offset += domains[currentDomain + subCounter].numberOfCores;
                subCounter++;
            }
        }
    }
    else
    {
        offset = 0;
        int NUMAthreads = numberOfProcessorsPerSocket * numberOfSocketDomains;
        domains[currentDomain + subCounter].numberOfProcessors = NUMAthreads;
        domains[currentDomain + subCounter].numberOfCores =  NUMAthreads/cpuid_topology.numThreadsPerCore;
        domains[currentDomain + subCounter].tag = bformat("M%d", subCounter);

        DEBUG_PRINT(DEBUGLEV_DEVELOP,
                Affinity domain M%d: %d HW threads on %d cores,
                subCounter, domains[currentDomain + subCounter].numberOfProcessors,
                domains[currentDomain + subCounter].numberOfCores);

        domains[currentDomain + subCounter].processorList = (int*) malloc(NUMAthreads*sizeof(int));

        if (!domains[currentDomain + subCounter].processorList)
        {
            fprintf(stderr,"No more memory for %ld bytes for processor list of affinity domain %s\n",
                    NUMAthreads*sizeof(int),
                    bdata(domains[currentDomain + subCounter].tag));
            return;
        }
        tmp = 0;
        for (int i=0; i < numberOfSocketDomains; i++ )
        {
            tmp += treeFillNextEntries(
                cpuid_topology.topologyTree,
                domains[currentDomain + subCounter].processorList, tmp,
                i, 0, domains[currentDomain + subCounter].numberOfCores,
                numberOfProcessorsPerSocket);
            offset += numberOfProcessorsPerSocket;
        }
        domains[currentDomain + subCounter].numberOfProcessors = tmp;
    }

    affinity_numberOfDomains = numberOfDomains;
    affinityDomains.numberOfAffinityDomains = numberOfDomains;
    affinityDomains.numberOfSocketDomains = numberOfSocketDomains;
    affinityDomains.numberOfNumaDomains = numberOfNumaDomains;
    affinityDomains.numberOfProcessorsPerSocket = numberOfProcessorsPerSocket;
    affinityDomains.numberOfCacheDomains = numberOfCacheDomains;
    affinityDomains.numberOfCoresPerCache = numberOfCoresPerCache;
    affinityDomains.numberOfProcessorsPerCache = numberOfProcessorsPerCache;
    affinityDomains.domains = domains;
    affinity_initialized = 1;
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

