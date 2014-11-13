/*
 * =======================================================================================
 *
 *      Filename:  libperfctr.c
 *
 *      Description:  Marker API interface of module perfmon
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
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <inttypes.h>

#include <likwid.h>
#include <bitUtil.h>
#include <lock.h>
#include <tree.h>
#include <timer.h>
#include <hashTable.h>
#include <registers.h>
#include <error.h>


#include <perfmon_core2_counters.h>
#include <perfmon_haswell_counters.h>
#include <perfmon_interlagos_counters.h>
#include <perfmon_kabini_counters.h>
#include <perfmon_k10_counters.h>
#include <perfmon_nehalem_counters.h>
#include <perfmon_phi_counters.h>
#include <perfmon_pm_counters.h>
#include <perfmon_sandybridge_counters.h>
#include <perfmon_ivybridge_counters.h>
#include <perfmon_westmereEX_counters.h>
#include <perfmon_silvermont_counters.h>


/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

int socket_lock[MAX_NUM_NODES];
static int hasPCICounters = 0;
static int likwid_init = 0;
static BitMask counterMask;
static int numberOfGroups = 0;
static int activeGroup = 0;
static uint64_t regTypeMask = 0;
static int threads2Cpu[MAX_NUM_THREADS];
static int num_cpus = 0;


/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define gettid() syscall(SYS_gettid)

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

void str2BitMask(const char* str, BitMask* mask)
{
    char* endptr;
    errno = 0;
    struct bstrList* tokens;
    bstring q = bfromcstralloc (60, str);
    tokens = bsplit(q,' ');

    for (int i=0; i<tokens->qty; i++)
    {
        uint64_t val =  strtoull((char*) tokens->entry[i]->data, &endptr, 16);
        if ((errno == ERANGE && val == LONG_MAX ) || (errno != 0 && val == 0))
        {
          ERROR;
        }

        if (endptr == str)
        {
          ERROR_PLAIN_PRINT(No digits were found);
        }

        mask->mask[i] = val;
    }

    bstrListDestroy(tokens);
    bdestroy(q);
}

static int getProcessorID(cpu_set_t* cpu_set)
{
    int processorId;

    for (processorId=0;processorId<MAX_NUM_THREADS;processorId++)
    {
        if (CPU_ISSET(processorId,cpu_set))
        {
            break;
        }
    }
    return processorId;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void likwid_markerInit(void)
{
    int i;
    int groupId;
    int verbosity;
    bstring bThreadStr;
    struct bstrList* threadTokens;
    int cpu_id = likwid_getProcessorId();
    char* modeStr = getenv("LIKWID_MODE");
    char* maskStr = getenv("LIKWID_MASK");
    char* eventStr = getenv("LIKWID_EVENTS");
    char* cThreadStr = getenv("LIKWID_THREADS");
    char* groupStr = getenv("LIKWID_GROUPS");
    /* Dirty hack to avoid nonnull warnings */
    int (*ownatoi)(const char*);
    ownatoi = &atoi;

    if ((modeStr != NULL) && (maskStr != NULL) && (eventStr != NULL) && (cThreadStr != NULL) && (groupStr != NULL))
    {
        likwid_init = 1;
    }
    else
    {
        fprintf(stderr, "Cannot initalize LIKWID marker API, environment variables are not set\n");
        fprintf(stderr, "You have to set the -m commandline switch for likwid-perfctr\n");
        return;
    }
    sscanf(getenv("LIKWID_COUNTERMASK"), "%" PRIx64, &regTypeMask);
    verbosity = atoi(getenv("LIKWID_DEBUG"));
    numberOfGroups = atoi(getenv("LIKWID_GROUPS"));
    if (!lock_check())
    {
        fprintf(stderr,"Access to performance counters is locked.\n");
        exit(EXIT_FAILURE);
    }

    topology_init();
    numa_init();
    affinity_init();
    hashTable_init();

    for(int i=0; i<MAX_NUM_THREADS; i++) thread_sockets[i] = -1;
    for(int i=0; i<MAX_NUM_NODES; i++) socket_lock[i] = LOCK_INIT;

    accessClient_setaccessmode(atoi(modeStr));
    perfmon_verbosity = verbosity;
    /*perfmon_init(num_cpus, threads2Cpu);

    activeGroup = perfmon_addEventSet(eventStr);
    groupSet->activeGroup = activeGroup;*/
    
    
    str2BitMask(maskStr, &counterMask);

    bThreadStr = bfromcstr(cThreadStr);
    threadTokens = bstrListCreate();
    threadTokens = bsplit(bThreadStr,',');
    int threadsToCpu[threadTokens->qty];
    for (i=0; i<threadTokens->qty; i++)
    {
        threadsToCpu[i] = ownatoi(bdata(threadTokens->entry[i]));
        if (accessClient_mode != DAEMON_AM_DIRECT)
        {
            accessClient_init(&thread_sockets[threadsToCpu[i]]);
        }
    }
    perfmon_init(threadTokens->qty, threadsToCpu);
    bdestroy(bThreadStr);
    bstrListDestroy(threadTokens);

    groupId = perfmon_addEventSet(eventStr);
    accessClient_finalize(socket_fd);
    socket_fd = -1;
    groupSet->activeGroup = groupId;
    for(int i=0;i<groupSet->groups[activeGroup].numberOfEvents;i++)
    {
        for(int j=0;j<groupSet->numberOfThreads;j++)
        {
            groupSet->groups[activeGroup].events[i].threadCounter[j].init = TRUE;
        }
    }
}

void likwid_markerThreadInit(void)
{
    if ( ! likwid_init )
    {
        return;
    }

    int cpu_id = likwid_getProcessorId();

    for(int i=0; i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
    {
        groupSet->groups[groupSet->activeGroup].events[i].threadCounter[cpu_id].init = TRUE;
    }
}

/* File format
 * 1 numberOfThreads numberOfRegions
 * 2 regionID:regionTag0
 * 3 regionID:regionTag1
 * 4 regionID threadID countersvalues(space separated)
 * 5 regionID threadID countersvalues
 */
void likwid_markerClose(void)
{
    FILE *file = NULL;
    LikwidResults* results = NULL;
    int numberOfThreads;
    int numberOfRegions;
    if ( ! likwid_init )
    {
        return;
    }
    numberOfGroups = atoi(getenv("LIKWID_GROUPS"));
    hashTable_finalize(&numberOfThreads, &numberOfRegions, &results);

    file = fopen(getenv("LIKWID_FILEPATH"),"w");

    if (file != NULL)
    {
        fprintf(file,"%d %d %d\n",numberOfThreads,numberOfRegions, numberOfGroups);
        for (int i=0; i<numberOfRegions; i++)
        {
            fprintf(file,"%d:%s\n",i,bdata(results[i].tag));
        }
        for (int i=0; i<numberOfRegions; i++)
        {
            for (int j=0; j<numberOfThreads; j++)
            {
                fprintf(file,"%d ",i);
                fprintf(file,"%d ",j);
                fprintf(file,"%d ",threads2Cpu[j]);
                fprintf(file,"%u ",results[i].count[j]);
                fprintf(file,"%e ",results[i].time[j]);

                for (int k=0; k<groupSet->groups[activeGroup].numberOfEvents; k++)
                {
                    fprintf(file,"%e ",results[i].counters[j][k]);
                }
                fprintf(file,"\n");
            }
        }
        fclose(file);
    }

    for (int i=0;i<numberOfRegions; i++)
    {
        for (int j=0;j<numberOfThreads; j++)
        {
            free(results[i].counters[j]);
        }
        free(results[i].time);
        bdestroy(results[i].tag);
        free(results[i].count);
        free(results[i].counters);
    }

    if (results != NULL)
    {
        free(results);
    }

    msr_finalize();
    pci_finalize();

    for (int i=0; i<MAX_NUM_THREADS; i++)
    {
        accessClient_finalize(thread_sockets[i]);
        thread_sockets[i] = -1;
    }
}


int likwid_markerStartRegion(const char* regionTag)
{
    if ( ! likwid_init )
    {
        return -EFAULT;
    }
    bstring tag = bfromcstralloc(100, regionTag);
    LikwidThreadResults* results;
    int ret;
    int threadId;
    uint64_t res;
    uint64_t tmp, counter_result;
    char groupSuffix[10];
    sprintf(groupSuffix, "-%d", activeGroup);
    bcatcstr(tag, groupSuffix);
    int cpu_id = hashTable_get(tag, &results);
    bdestroy(tag);
    int socket_fd = thread_sockets[cpu_id];

    if (accessClient_mode != DAEMON_AM_DIRECT)
    {
        for(int i=0;i<groupSet->numberOfThreads;i++)
        {
            if (cpu_id == groupSet->threads[i].processorId)
            {
                threadId = i;
                break;
            }
        }
    }

    socket_fd = thread_sockets[cpu_id];

    perfmon_readCountersCpu(cpu_id);

    for(int i=0;i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, START [%s] READ EVENT [%d] ID %d VALUE %llu , regionTag, cpu_id, i,
                        LLU_CAST groupSet->groups[groupSet->activeGroup].events[i].threadCounter[cpu_id].counterData);
        groupSet->groups[groupSet->activeGroup].events[i].threadCounter[cpu_id].startData =
            groupSet->groups[groupSet->activeGroup].events[i].threadCounter[cpu_id].counterData;
    }

    timer_start(&(results->startTime));
}

#define READ_END_MEM_CHANNEL(channel, reg, cid)                      \
    counter_result = pci_tread(socket_fd, cpu_id, channel, reg##_A); \
    counter_result = (counter_result<<32) +                          \
    pci_tread(socket_fd, cpu_id, channel, reg##_B);                  \
    results->PMcounters[cid] += (uint64_t) counter_result - results->StartPMcounters[cid]


/* TODO: Readout hash at the end. Compute result at the end of the function to
 * keep overhead in region low */

int likwid_markerStopRegion(const char* regionTag)
{
    if (! likwid_init)
    {
        return -EFAULT;
    }

    TimerData timestamp;
    timer_stop(&timestamp);
    int cpu_id = likwid_getProcessorId();
    int ret;
    uint64_t res;
    uint64_t tmp, counter_result;
    int socket_fd = thread_sockets[cpu_id];
    double PMcounters[NUM_PMC];
    bstring tag = bfromcstr(regionTag);
    char groupSuffix[100];
    LikwidThreadResults* results;
    sprintf(groupSuffix, "-%d", activeGroup);
    bcatcstr(tag, groupSuffix);
    
    hashTable_get(tag, &results);
    results->startTime.stop = timestamp.stop;
    results->time += timer_print(&(results->startTime));
    bdestroy(tag);
    
    perfmon_readCountersCpu(cpu_id);
    for(int i=0;i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, STOP [%s] READ EVENT [%d] ID %d VALUE %llu, regionTag, cpu_id, i,
                        LLU_CAST groupSet->groups[groupSet->activeGroup].events[i].threadCounter[cpu_id].counterData);
        results->PMcounters[groupSet->groups[groupSet->activeGroup].events[i].index] += perfmon_getResult(groupSet->activeGroup, i, cpu_id);

    }
}

void likwid_switchMarkerGroup(int groupId)
{
    perfmon_switchActiveGroup(groupId);
    activeGroup = groupId;
}

int  likwid_getProcessorId()
{
    cpu_set_t  cpu_set;
    CPU_ZERO(&cpu_set);
    sched_getaffinity(gettid(),sizeof(cpu_set_t), &cpu_set);

    return getProcessorID(&cpu_set);
}

#ifdef HAS_SCHEDAFFINITY
int  likwid_pinThread(int processorId)
{
    int ret;
    cpu_set_t cpuset;
    pthread_t thread;

    thread = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(processorId, &cpuset);
    ret = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);

    if (ret != 0)
    {
        ERROR;
        return FALSE;
    }

    return TRUE;
}
#endif


int  likwid_pinProcess(int processorId)
{
    int ret;
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET(processorId, &cpuset);
    ret = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

    if (ret < 0)
    {
        ERROR;
        return FALSE;
    }

    return TRUE;
}


