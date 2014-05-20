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
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <pthread.h>

#include <error.h>
#include <types.h>
#include <numa.h>
#include <affinity.h>
#include <cpuid.h>
#include <tree.h>

/* #####   EXPORTED VARIABLES   ########################################### */

int affinity_core2node_lookup[MAX_NUM_THREADS];

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define gettid() syscall(SYS_gettid)


/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int  affinity_numberOfDomains = 0;
static AffinityDomain*  domains;

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

static void
treeFillNextEntries(
    TreeNode* tree,
    int* processorIds,
    int socketId,
    int offset,
    int numberOfEntries )
{
  int counter = numberOfEntries;
  TreeNode* node = tree;
  TreeNode* thread;

  node = tree_getChildNode(node);

  /* get socket node */
  for (int i=0; i<socketId; i++)
  {
    node = tree_getNextNode(node);

    if ( node == NULL )
    {
      printf("ERROR: Socket %d not existing!",i);
      exit(EXIT_FAILURE);
    }
  }

  node = tree_getChildNode(node);
  /* skip offset cores */
  for (int i=0; i<offset; i++)
  {
    node = tree_getNextNode(node);

    if ( node == NULL )
    {
      printf("ERROR: Core %d not existing!",i);
      exit(EXIT_FAILURE);
    }
  }

  /* Traverse horizontal */
  while ( node != NULL )
  {
    if ( !counter ) break;

    thread = tree_getChildNode(node);

    while ( thread != NULL )
    {
      processorIds[numberOfEntries-counter] = thread->id;
      thread = tree_getNextNode(thread);
      counter--;
    }
    node = tree_getNextNode(node);
  }
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
affinity_init()
{
    int numberOfDomains = 1; /* all systems have the node domain */
    int currentDomain;
    int subCounter = 0;
    int offset = 0;
    int numberOfSocketDomains = cpuid_topology.numSockets;;
    int numberOfNumaDomains = numa_info.numberOfNodes;
    int numberOfProcessorsPerSocket =
        cpuid_topology.numCoresPerSocket * cpuid_topology.numThreadsPerCore;
    int numberOfCacheDomains;

    int numberOfCoresPerCache =
        cpuid_topology.cacheLevels[cpuid_topology.numCacheLevels-1].threads/
        cpuid_topology.numThreadsPerCore;

    int numberOfProcessorsPerCache =
        cpuid_topology.cacheLevels[cpuid_topology.numCacheLevels-1].threads;

    /* for the cache domain take only into account last level cache and assume
     * all sockets to be uniform. */

    /* determine how many last level shared caches exist per socket */
    numberOfCacheDomains = cpuid_topology.numSockets *
        (cpuid_topology.numCoresPerSocket/numberOfCoresPerCache);

    /* determine total number of domains */
    numberOfDomains += numberOfSocketDomains + numberOfCacheDomains + numberOfNumaDomains;

    domains = (AffinityDomain*) malloc(numberOfDomains * sizeof(AffinityDomain));

    /* Node domain */
    domains[0].numberOfProcessors = cpuid_topology.numHWThreads;
    domains[0].numberOfCores = cpuid_topology.numSockets * cpuid_topology.numCoresPerSocket;
    domains[0].processorList = (int*) malloc(cpuid_topology.numHWThreads*sizeof(int));
    domains[0].tag = bformat("N");
    offset = 0;

    for (int i=0; i<numberOfSocketDomains; i++)
    {
      treeFillNextEntries(
          cpuid_topology.topologyTree,
          domains[0].processorList + offset,
          i, 0, numberOfProcessorsPerSocket);

      offset += numberOfProcessorsPerSocket;
    }

    /* Socket domains */
    currentDomain = 1;

    for (int i=0; i < numberOfSocketDomains; i++ )
    {
      domains[currentDomain + i].numberOfProcessors = numberOfProcessorsPerSocket;
      domains[currentDomain + i].numberOfCores =  cpuid_topology.numCoresPerSocket;
      domains[currentDomain + i].processorList = (int*) malloc( domains[currentDomain + i].numberOfProcessors * sizeof(int));
      domains[currentDomain + i].tag = bformat("S%d", i);

      treeFillNextEntries(
          cpuid_topology.topologyTree,
          domains[currentDomain + i].processorList,
          i, 0, domains[currentDomain + i].numberOfProcessors);
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
        domains[currentDomain + subCounter].processorList = (int*) malloc(numberOfProcessorsPerCache*sizeof(int));
        domains[currentDomain + subCounter].tag = bformat("C%d", subCounter);

        treeFillNextEntries(
            cpuid_topology.topologyTree,
            domains[currentDomain + subCounter].processorList,
            i, offset, domains[currentDomain + subCounter].numberOfProcessors);

        offset += numberOfCoresPerCache;
        subCounter++;
      }
    }

    /* Memory domains */
    currentDomain += numberOfCacheDomains;
    subCounter = 0;

    for (int i=0; i < numberOfSocketDomains; i++ )
    {
      offset = 0;

      for ( int j=0; j < (numberOfNumaDomains/numberOfSocketDomains); j++ )
      {
        domains[currentDomain + subCounter].numberOfProcessors = numberOfProcessorsPerCache;
        domains[currentDomain + subCounter].numberOfCores =  numberOfCoresPerCache;
        domains[currentDomain + subCounter].processorList = (int*) malloc(numa_info.nodes[subCounter].numberOfProcessors*sizeof(int));
        domains[currentDomain + subCounter].tag = bformat("M%d", subCounter);

        treeFillNextEntries(
            cpuid_topology.topologyTree,
            domains[currentDomain + subCounter].processorList,
            i, offset, domains[currentDomain + subCounter].numberOfProcessors);

        offset += numberOfCoresPerCache;
        subCounter++;
      }
    }

    /* This is redundant ;-). Create thread to node lookup */
    for ( uint32_t i = 0; i < numa_info.numberOfNodes; i++ )
    {
        for ( int j = 0; j < numa_info.nodes[i].numberOfProcessors; j++ )
        {
            affinity_core2node_lookup[numa_info.nodes[i].processors[j]] = i;
        }
    }

    affinity_numberOfDomains = numberOfDomains;
}


void
affinity_finalize()
{
    for ( int i=0; i < affinity_numberOfDomains; i++ )
    {
        free(domains[i].processorList);
    }
    free(domains);
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

