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

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

int socket_lock[MAX_NUM_NODES];
static int likwid_init = 0;
static int numberOfGroups = 0;
static int* groups;
static int threads2Cpu[MAX_NUM_THREADS];
static pthread_t threads2Pthread[MAX_NUM_THREADS];
static int realThreads2Cpu[MAX_NUM_THREADS] = { [ 0 ... (MAX_NUM_THREADS-1)] = -1};
static int num_cpus = 0;
static int registered_cpus = 0;
static pthread_mutex_t globalLock = PTHREAD_MUTEX_INITIALIZER;
static int use_locks = 0;
static pthread_mutex_t threadLocks[MAX_NUM_THREADS] = { [ 0 ... (MAX_NUM_THREADS-1)] = PTHREAD_MUTEX_INITIALIZER};

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

    if (!lock_check())
    {
        fprintf(stderr,"Access to performance counters is locked.\n");
        exit(EXIT_FAILURE);
    }

    topology_init();
    numa_init();
    affinity_init();
    hashTable_init();

    for(int i=0; i<MAX_NUM_NODES; i++) socket_lock[i] = LOCK_INIT;

    HPMmode(atoi(modeStr));

    if (getenv("LIKWID_DEBUG") != NULL)
    {
        perfmon_verbosity = atoi(getenv("LIKWID_DEBUG"));
        verbosity = perfmon_verbosity;
    }

    if (getenv("LIKWID_PIN") != NULL)
    {
        bThreadStr = bfromcstr(cThreadStr);
        threadTokens = bstrListCreate();
        threadTokens = bsplit(bThreadStr,',');
        num_cpus = threadTokens->qty;
        for (i=0; i<num_cpus; i++)
        {
            threads2Cpu[i] = ownatoi(bdata(threadTokens->entry[i]));
            HPMaddThread(threads2Cpu[i]);
        }
        bdestroy(bThreadStr);
        bstrListDestroy(threadTokens);
        likwid_pinThread(threads2Cpu[0]);
        if (getenv("OMP_NUM_THREADS") != NULL)
        {
            if (ownatoi(getenv("OMP_NUM_THREADS")) > num_cpus)
            {
                use_locks = 1;
            }
        }
        if (getenv("CILK_NWORKERS") != NULL)
        {
            if (ownatoi(getenv("CILK_NWORKERS")) > num_cpus)
            {
                use_locks = 1;
            }
        }
    }
    else
    {
        use_locks = 1;
        for (i=0;i<cpuid_topology.activeHWThreads; i++)
        {
            if (cpuid_topology.threadPool[i].inCpuSet)
            {
                threads2Cpu[num_cpus++] = cpuid_topology.threadPool[i].apicId;
                HPMaddThread(threads2Cpu[num_cpus-1]);
            }
        }
    }

    i = perfmon_init(num_cpus, threads2Cpu);
    if (i<0)
    {
        fprintf(stderr,"Failed to initialize LIKWID perfmon library.\n");
        return;
    }

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
    bdestroy(bEventStr);

    for (i=0; i<num_cpus; i++)
    {
        
        initThreadArch(threads2Cpu[i]);
        hashTable_initThread(threads2Cpu[i]);
        for(int j=0; j<groupSet->groups[groups[0]].numberOfEvents;j++)
        {
            groupSet->groups[groups[0]].events[j].threadCounter[i].init = TRUE;
            if (groupSet->groups[groups[0]].events[j].type == PERF)
            {
                setup_perf_event(threads2Cpu[i], &(groupSet->groups[groups[0]].events[j].event));
            }
        }
    }

    groupSet->activeGroup = 0;
}

void likwid_markerThreadInit(void)
{
    int myID;
    int cpu_id;
    if ( !likwid_init )
    {
        return;
    }
    
    if (use_locks == 1)
    {
        pthread_mutex_lock(&globalLock);
    }
    myID = registered_cpus++;
    threads2Pthread[myID] = pthread_self();
    if (use_locks == 1)
    {
        pthread_mutex_unlock(&globalLock);
    }

    if (getenv("LIKWID_PIN") != NULL)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        sched_getaffinity(gettid(), sizeof(cpu_set_t), &cpuset);
        if ((CPU_COUNT(&cpuset) > 1) || (likwid_getProcessorId() != threads2Cpu[myID % num_cpus]))
        {
            likwid_pinThread(threads2Cpu[myID % num_cpus]);
            cpu_id = threads2Cpu[myID % num_cpus];
            DEBUG_PRINT(DEBUGLEV_DEVELOP, "Pin thread %lu to CPU %d\n", gettid(), cpu_id);
        }
        else
        {
            cpu_id = likwid_getProcessorId();
        }
    }
    else
    {
        cpu_id = likwid_getProcessorId();
    }
    realThreads2Cpu[myID % num_cpus] = cpu_id;
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
    int numberOfThreads = 0;
    int numberOfRegions = 0;
    char* markerfile = NULL;
    if ( ! likwid_init )
    {
        fprintf(stdout, "LIKWID not properly initialized\n");
        return;
    }
    hashTable_finalize(&numberOfThreads, &numberOfRegions, &results);
    if ((numberOfThreads == 0)||(numberOfThreads == 0))
    {
        fprintf(stdout, "No threads or regions defined in hash table\n");
        return;
    }
    markerfile = getenv("LIKWID_FILEPATH");
    if (markerfile == NULL)
    {
        fprintf(stdout, "Is the application executed with LIKWID wrapper? No file path for the Marker API output defined.\n");
        return;
    }
    file = fopen(markerfile,"w");

    if (file != NULL)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "Creating Marker file %s with %d regions, %d groups and %d threads\n", markerfile, numberOfRegions, numberOfGroups, numberOfThreads);
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
                if (results[i].count[j % num_cpus] == 0) continue;
                fprintf(file,"%d ",i);
                fprintf(file,"%d ",results[i].groupID);
                fprintf(file,"%d ",j);
                fprintf(file,"%u ",results[i].count[j]);
                fprintf(file,"%e ",results[i].time[j]);
                fprintf(file,"%d ",groupSet->groups[results[i].groupID].numberOfEvents);

                for (int k=0; k<groupSet->groups[results[i].groupID].numberOfEvents; k++)
                {
                    if (realThreads2Cpu[j % num_cpus] != -1)
                    {
                        fprintf(file,"%e ",results[i].counters[j][k]);
                    }
                    else
                    {
                        fprintf(file,"0 ");
                    }
                }
                fprintf(file,"\n");
            }
        }
        fclose(file);
        fprintf(stderr, "Wrote LIKWID Marker API output to file %s\n", markerfile);
    }
    else
    {
        fprintf(stderr, "Cannot open file %s\n", markerfile);
        fprintf(stderr, "%s", strerror(errno));
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

int likwid_markerRegisterRegion(const char* regionTag)
{
    if ( ! likwid_init )
    {
        return -EFAULT;
    }
    TimerData timer;
    bstring tag = bfromcstralloc(100, regionTag);
    LikwidThreadResults* results;
    char groupSuffix[10];
    sprintf(groupSuffix, "-%d", groupSet->activeGroup);
    bcatcstr(tag, groupSuffix);
    int cpu_id = hashTable_get(tag, &results);
    bdestroy(tag);
    return 0;
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
    
    bdestroy(tag);
    timer_start(&(results->startTime));
    return 0;
}



int likwid_markerStopRegion(const char* regionTag)
{
    if (! likwid_init)
    {
        return -EFAULT;
    }

    TimerData timestamp;
    timer_stop(&timestamp);
    double result = 0.0;
    int cpu_id;
    int myCPU = likwid_getProcessorId();
    int thread_id;
    bstring tag = bfromcstr(regionTag);
    char groupSuffix[100];
    LikwidThreadResults* results;
    sprintf(groupSuffix, "-%d", groupSet->activeGroup);
    bcatcstr(tag, groupSuffix);
    if (use_locks == 1)
    {
        pthread_mutex_lock(&threadLocks[myCPU]);
    }
    
    cpu_id = hashTable_get(tag, &results);
    thread_id = getThreadID(cpu_id);
    results->groupID = groupSet->activeGroup;
    results->startTime.stop.int64 = timestamp.stop.int64;
    results->time += timer_print(&(results->startTime));
    results->count++;
    bdestroy(tag);
    
    perfmon_readCountersCpu(cpu_id);
    
    for(int i=0;i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, STOP [%s] READ EVENT [%d=%d] EVENT %d VALUE %llu, regionTag, thread_id, cpu_id, i,
                        LLU_CAST groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData);
        result = perfmon_getResult(groupSet->activeGroup, i, thread_id);
        results->PMcounters[i] += result;
        //groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].startData = 0x0ULL;
        //groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData = 0x0ULL;
    }
    if (use_locks == 1)
    {
        pthread_mutex_unlock(&threadLocks[myCPU]);
    }
    return 0;
}


void likwid_markerGetRegion(const char* regionTag, int* nr_events, double* events, double *time, int *count)
{
    if (! likwid_init)
    {
        return;
    }
    int length = 0;
    int cpu_id;
    int myCPU = likwid_getProcessorId();
    int thread_id;
    bstring tag = bfromcstr(regionTag);
    char groupSuffix[100];
    LikwidThreadResults* results;
    sprintf(groupSuffix, "-%d", groupSet->activeGroup);
    bcatcstr(tag, groupSuffix);

    cpu_id = hashTable_get(tag, &results);
    thread_id = getThreadID(myCPU);
    *count = results->count;
    *time = results->time;
    length = MIN(groupSet->groups[groupSet->activeGroup].numberOfEvents, *nr_events);
    for(int i=0;i<length;i++)
    {
        events[i] = results->PMcounters[i];
    }
    *nr_events = length;
    return;
}


int  likwid_getProcessorId()
{
    int i;
    cpu_set_t  cpu_set;
    CPU_ZERO(&cpu_set);
    sched_getaffinity(gettid(),sizeof(cpu_set_t), &cpu_set);
    if (CPU_COUNT(&cpu_set) > 1)
    {
        return sched_getcpu();
    }
    else
    {
        return getProcessorID(&cpu_set);
    }
    return -1;
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
        ERROR_PRINT("ERROR: Pinning of thread to CPU %d failed\n", processorId);
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
        ERROR_PRINT("ERROR: Pinning of process to CPU %d failed\n", processorId);
        return FALSE;
    }

    return TRUE;
}


