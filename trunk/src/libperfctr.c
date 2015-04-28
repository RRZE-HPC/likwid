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
 *      Authors:  Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
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
#include <access.h>

#include <perfmon.h>
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
static int likwid_init = 0;
static int numberOfGroups = 0;
static int* groups;
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

static int getThreadID(int cpu_id)
{
    int i;
    for(i=0;i<groupSet->numberOfThreads;i++)
    {
        if (cpu_id == groupSet->threads[i].processorId)
        {
            return i;
        }
    }
    return -1;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void likwid_markerInit(void)
{
    int i;
    int verbosity;
    bstring bThreadStr;
    bstring bEventStr;
    struct bstrList* threadTokens;
    struct bstrList* eventStrings;
    char* modeStr = getenv("LIKWID_MODE");
    char* eventStr = getenv("LIKWID_EVENTS");
    char* cThreadStr = getenv("LIKWID_THREADS");
    char* filepath = getenv("LIKWID_FILEPATH");
    /* Dirty hack to avoid nonnull warnings */
    int (*ownatoi)(const char*);
    ownatoi = &atoi;

    if ((modeStr != NULL) && (filepath != NULL) && (eventStr != NULL) && (cThreadStr != NULL))
    {
        likwid_init = 1;
    }
    else if (likwid_init == 0)
    {
        fprintf(stderr, "Cannot initalize LIKWID marker API, environment variables are not set\n");
        fprintf(stderr, "You have to set the -m commandline switch for likwid-perfctr\n");
        return;
    }
    else
    {
        return;
    }

    verbosity = atoi(getenv("LIKWID_DEBUG"));
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


    bThreadStr = bfromcstr(cThreadStr);
    threadTokens = bstrListCreate();
    threadTokens = bsplit(bThreadStr,',');
    num_cpus = threadTokens->qty;
    for (i=0; i<num_cpus; i++)
    {
        threads2Cpu[i] = ownatoi(bdata(threadTokens->entry[i]));
    }
    perfmon_init(num_cpus, threads2Cpu);
    bdestroy(bThreadStr);
    bstrListDestroy(threadTokens);

    bEventStr = bfromcstr(eventStr);
    eventStrings = bstrListCreate();
    eventStrings = bsplit(bEventStr,'|');
    numberOfGroups = eventStrings->qty;
    groups = malloc(numberOfGroups * sizeof(int));
    if (!groups)
    {
        fprintf(stderr,"Cannot allocate space for group handling.\n");
        bstrListDestroy(eventStrings);
        exit(EXIT_FAILURE);
    }
    for (i=0; i<eventStrings->qty; i++)
    {
        groups[i] = perfmon_addEventSet(bdata(eventStrings->entry[i]));
    }
    bstrListDestroy(eventStrings);

    groupSet->activeGroup = 0;
}

void likwid_markerThreadInit(void)
{
    if ( !likwid_init )
    {
        return;
    }

    int cpu_id = likwid_getProcessorId();
    int thread_id = getThreadID(cpu_id);

    HPMaddThread(cpu_id);
    initThreadArch(cpu_id);

    for(int i=0; i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
    {
        groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].init = TRUE;
    }
}

void likwid_markerNextGroup(void)
{
    int i;
    int next_group;

    if (!likwid_init)
    {
        return;
    }

    next_group = (groupSet->activeGroup + 1) % numberOfGroups;
    if (next_group != groupSet->activeGroup)
    {
        i = perfmon_switchActiveGroup(next_group);
    }
    return;
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

    hashTable_finalize(&numberOfThreads, &numberOfRegions, &results);

    file = fopen(getenv("LIKWID_FILEPATH"),"w");

    if (file != NULL)
    {
        fprintf(file,"%d %d %d\n",numberOfThreads, numberOfRegions, numberOfGroups);
        for (int i=0; i<numberOfRegions; i++)
        {
            if (results[i].count[0] == 0)
            {
                continue;
            }
            fprintf(file,"%d:%s\n",i,bdata(results[i].tag));
        }
        for (int i=0; i<numberOfRegions; i++)
        {
            if (results[i].count[0] == 0)
            {
                continue;
            }
            for (int j=0; j<numberOfThreads; j++)
            {
                fprintf(file,"%d ",i);
                fprintf(file,"%d ",results[i].groupID);
                fprintf(file,"%d ",j);
                fprintf(file,"%u ",results[i].count[j]);
                fprintf(file,"%e ",results[i].time[j]);
                fprintf(file,"%d ",groupSet->groups[results[i].groupID].numberOfEvents);

                for (int k=0; k<groupSet->groups[results[i].groupID].numberOfEvents; k++)
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
    likwid_init = 0;
    HPMfinalize();
}


int likwid_markerStartRegion(const char* regionTag)
{
    if ( ! likwid_init )
    {
        return -EFAULT;
    }
    bstring tag = bfromcstralloc(100, regionTag);
    LikwidThreadResults* results;
    char groupSuffix[10];
    sprintf(groupSuffix, "-%d", groupSet->activeGroup);
    bcatcstr(tag, groupSuffix);
    int cpu_id = hashTable_get(tag, &results);
    int thread_id = getThreadID(cpu_id);

    perfmon_readCountersCpu(cpu_id);

    for(int i=0;i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, START [%s] READ EVENT [%d=%d] EVENT %d VALUE %llu , regionTag, thread_id, cpu_id, i,
                        LLU_CAST groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData);
        groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].startData =
                groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData;
    }
    results->groupID = groupSet->activeGroup;
    timer_start(&(results->startTime));
    bdestroy(tag);
    return 0;
}


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
    int cpu_id;
    int thread_id;
    bstring tag = bfromcstr(regionTag);
    char groupSuffix[100];
    LikwidThreadResults* results;
    sprintf(groupSuffix, "-%d", groupSet->activeGroup);
    bcatcstr(tag, groupSuffix);

    cpu_id = hashTable_get(tag, &results);
    thread_id = getThreadID(cpu_id);
    results->startTime.stop.int64 = timestamp.stop.int64;
    results->time += timer_print(&(results->startTime));
    results->count++;
    bdestroy(tag);

    perfmon_readCountersCpu(cpu_id);
    for(int i=0;i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, STOP [%s] READ EVENT [%d=%d] EVENT %d VALUE %llu, regionTag, thread_id, cpu_id, i,
                        LLU_CAST groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData);
        results->PMcounters[i] += perfmon_getResult(groupSet->activeGroup, i, thread_id);
    }
    return 0;
}

void likwid_markerPrintRegion(const char* regionTag)
{
    if (! likwid_init)
    {
        return;
    }
    int cpu_id;
    int thread_id;
    bstring tag = bfromcstr(regionTag);
    char groupSuffix[100];
    LikwidThreadResults* results;
    sprintf(groupSuffix, "-%d", groupSet->activeGroup);
    bcatcstr(tag, groupSuffix);

    cpu_id = hashTable_get(tag, &results);
    thread_id = getThreadID(cpu_id);
    printf("%d %s %d %f", cpu_id, bdata(tag), results->count, results->time);
    for(int i=0;i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
    {
        printf(" %f", results->PMcounters[i]);
    }
    printf("\n");
    return;
}

void likwid_markerGetRegion(const char* regionTag, int nr_events, double* events, double *time, int *count)
{
    if (! likwid_init)
    {
        return;
    }
    int length = 0;
    int cpu_id;
    int thread_id;
    bstring tag = bfromcstr(regionTag);
    char groupSuffix[100];
    LikwidThreadResults* results;
    sprintf(groupSuffix, "-%d", groupSet->activeGroup);
    bcatcstr(tag, groupSuffix);

    cpu_id = hashTable_get(tag, &results);
    thread_id = getThreadID(cpu_id);
    *count = results->count;
    *time = results->time;
    length = MIN(groupSet->groups[groupSet->activeGroup].numberOfEvents, nr_events);
    for(int i=0;i<length;i++)
    {
        events[i] = results->PMcounters[i];
    }
    return;
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


