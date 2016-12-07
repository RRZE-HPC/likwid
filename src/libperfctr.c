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
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <inttypes.h>
#include <signal.h>

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
int core_lock[MAX_NUM_THREADS];
static int likwid_init = 0;
static int numberOfGroups = 0;
static int* groups;
static int threads2Cpu[MAX_NUM_THREADS];
static pthread_t mainThread;
static int realThreads2Cpu[MAX_NUM_THREADS] = { [ 0 ... (MAX_NUM_THREADS-1)] = -1};
static int num_cpus = 0;
static int registered_cpus = 0;
static pthread_mutex_t globalLock = PTHREAD_MUTEX_INITIALIZER;
static int use_locks = 0;
static pthread_mutex_t threadLocks[MAX_NUM_THREADS] = { [ 0 ... (MAX_NUM_THREADS-1)] = PTHREAD_MUTEX_INITIALIZER};
static bstring lastStartedRegions[MAX_NUM_THREADS];
static int ovf_secs = 0;
static int ovf_usecs = 50000;
static int ovf_started = 0;
static int nest_level[MAX_NUM_THREADS] = { [ 0 ... (MAX_NUM_THREADS-1)] = 0};
static LikwidThreadResults ovf_data;
static int ovf_checks = 0;

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define gettid() syscall(SYS_gettid)

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

void likwid_overflowRead(int sig);


static int
getProcessorID(cpu_set_t* cpu_set)
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

static int
getThreadID(int cpu_id)
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

static double
calculateMarkerResult(RegisterIndex index, uint64_t start, uint64_t stop, int overflows)
{
    double result = 0.0;

    if (overflows == 0)
    {
        result = (double) (stop - start);
    }
    else if (overflows > 0)
    {
        result += (double) ((perfmon_getMaxCounterValue(counter_map[index].type) - start) + stop);
        result += (double) ((overflows-1) * perfmon_getMaxCounterValue(counter_map[index].type));
    }
    if (counter_map[index].type == POWER)
    {
        result *= power_getEnergyUnit(getCounterTypeOffset(index));
    }
    else if (counter_map[index].type == THERMAL)
    {
        result = (double)stop;
    }
    return result;
}

static unsigned int
usalarm (unsigned int secs, unsigned int usecs)
{
    struct itimerval old, new;
    new.it_interval.tv_usec = 0;
    new.it_interval.tv_sec = 0;
    new.it_value.tv_usec = (long int) usecs;
    new.it_value.tv_sec = (long int) secs;
    if (setitimer (ITIMER_REAL, &new, &old) < 0)
        return 0;
    else
        return old.it_value.tv_sec;
}

void
likwid_overflowRead(int sig)
{
    if (! likwid_init)
    {
        return;
    }

    perfmon_readCounters();

    ovf_checks++;
    signal(SIGALRM, likwid_overflowRead);
    usalarm(ovf_secs, ovf_usecs);
    return;
}

static void
measure_ovfl_int()
{
    for(int i=0;i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
        ovf_data.PMcounters[i] = 0.0;
#ifdef _ARCH_PPC
    double mem[4000000];
    for (int i=0;i<4000000;i++)
        mem[i] = i;
    if (mem[2000000] < 0)
        return;
    perfmon_readCountersCpu(threads2Cpu[0]);
    for(int i=0;i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
    {
        ovf_data.StartPMcounters[i] = groupSet->groups[groupSet->activeGroup].events[i].threadCounter[0].counterData;
    }
    timer_start(&ovf_data.startTime);
    likwid_overflowRead(0);
    likwid_overflowRead(0);
    likwid_overflowRead(0);
    likwid_overflowRead(0);
    usalarm(0,0);
    timer_stop(&ovf_data.startTime);
    perfmon_readCountersCpu(threads2Cpu[0]);
    for(int i=0;i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
    {
        ovf_data.PMcounters[i] = (double)(groupSet->groups[groupSet->activeGroup].events[i].threadCounter[0].counterData - ovf_data.StartPMcounters[i]);
    }
#endif
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
likwid_markerInit(void)
{
    int i;
    int verbosity;
    int setinit = 0;
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

    if ((modeStr != NULL) && (filepath != NULL) && (eventStr != NULL) && (cThreadStr != NULL) && likwid_init == 0)
    {
        setinit = 1;
    }
    else if (likwid_init == 0)
    {
        fprintf(stderr, "Running without Marker API. Activate Marker API with -m on commandline.\n");
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
    mainThread = pthread_self();

    for(int i=0; i<MAX_NUM_NODES; i++) socket_lock[i] = LOCK_INIT;

    HPMmode(atoi(modeStr));

    if (getenv("LIKWID_DEBUG") != NULL)
    {
        perfmon_verbosity = atoi(getenv("LIKWID_DEBUG"));
        verbosity = perfmon_verbosity;
    }

    bThreadStr = bfromcstr(cThreadStr);
    threadTokens = bsplit(bThreadStr,',');
    num_cpus = threadTokens->qty;
    for (i=0; i<num_cpus; i++)
    {
        threads2Cpu[i] = ownatoi(bdata(threadTokens->entry[i]));
        realThreads2Cpu[i] = threads2Cpu[i];
    }
    bdestroy(bThreadStr);
    bstrListDestroy(threadTokens);

    if (getenv("LIKWID_PIN") != NULL)
    {
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

    i = perfmon_init(num_cpus, threads2Cpu);
    if (i<0)
    {
        //fprintf(stderr,"Failed to initialize LIKWID perfmon library.\n");
        return;
    }

    bEventStr = bfromcstr(eventStr);
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
        hashTable_initThread(threads2Cpu[i]);
        for(int j=0; j<groupSet->groups[groups[0]].numberOfEvents;j++)
        {
            groupSet->groups[groups[0]].events[j].threadCounter[i].init = TRUE;
            groupSet->groups[groups[0]].state = STATE_START;
        }
    }
    if (setinit)
    {
        likwid_init = 1;
    }
    groupSet->activeGroup = 0;
    measure_ovfl_int();
}

void
likwid_markerThreadInit(void)
{
    int myID;
    if ( !likwid_init )
    {
        return;
    }

    pthread_mutex_lock(&globalLock);
    myID = registered_cpus++;
    pthread_mutex_unlock(&globalLock);

    if (getenv("LIKWID_PIN") != NULL)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        sched_getaffinity(gettid(), sizeof(cpu_set_t), &cpuset);
        if ((CPU_COUNT(&cpuset) > 1) || (likwid_getProcessorId() != threads2Cpu[myID % num_cpus]))
        {
            likwid_pinThread(threads2Cpu[myID % num_cpus]);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, "Pin thread %lu to CPU %d\n", gettid(), threads2Cpu[myID % num_cpus]);
        }
    }
}

void
likwid_markerNextGroup(void)
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
    measure_ovfl_int();
    return;
}

/* File format
 * 1 numberOfThreads numberOfRegions
 * 2 regionID:regionTag0
 * 3 regionID:regionTag1
 * 4 regionID threadID countersvalues(space separated)
 * 5 regionID threadID countersvalues
 */
void
likwid_markerClose(void)
{
    FILE *file = NULL;
    LikwidResults* results = NULL;
    int numberOfThreads = 0;
    int numberOfRegions = 0;
    char* markerfile = NULL;
    int lineidx = 0;
    char line[1024];

    if ( ! likwid_init )
    {
        return;
    }
    hashTable_finalize(&numberOfThreads, &numberOfRegions, &results);
    if ((numberOfThreads == 0)||(numberOfThreads == 0))
    {
        fprintf(stderr, "No threads or regions defined in hash table\n");
        return;
    }
    markerfile = getenv("LIKWID_FILEPATH");
    if (markerfile == NULL)
    {
        fprintf(stderr,
                "Is the application executed with LIKWID wrapper? No file path for the Marker API output defined.\n");
        return;
    }
    file = fopen(markerfile,"w");

    if (file != NULL)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP,
                Creating Marker file %s with %d regions %d groups and %d threads,
                markerfile, numberOfRegions, numberOfGroups, numberOfThreads);
        fprintf(file,"%d %d %d\n",numberOfThreads, numberOfRegions, numberOfGroups);
        DEBUG_PRINT(DEBUGLEV_DEVELOP, %d %d %d, numberOfThreads, numberOfRegions, numberOfGroups);

        for (int i=0; i<numberOfRegions; i++)
        {
            fprintf(file,"%d:%s\n",i,bdata(results[i].tag));
            DEBUG_PRINT(DEBUGLEV_DEVELOP, %d:%s, i,bdata(results[i].tag));
        }
        for (int i=0; i<numberOfRegions; i++)
        {
            for (int j=0; j<numberOfThreads; j++)
            {
                fprintf(file,"%d ",i);
                fprintf(file,"%d ",results[i].groupID);
                fprintf(file,"%d ",results[i].cpulist[j]);
                fprintf(file,"%u ",results[i].count[j]);
                fprintf(file,"%e ",results[i].time[j]);
                fprintf(file,"%d ",groupSet->groups[results[i].groupID].numberOfEvents);
                lineidx = sprintf(&(line[0]), "%d %d %d %u %e %d ",
                        i, results[i].groupID,results[i].cpulist[j],results[i].count[j],
                        results[i].time[j],groupSet->groups[results[i].groupID].numberOfEvents);

                for (int k=0; k<groupSet->groups[results[i].groupID].numberOfEvents; k++)
                {
                    fprintf(file,"%e ",results[i].counters[j][k]);
                    lineidx += sprintf(&(line[lineidx]), "%e ", results[i].counters[j][k]);
                }
                fprintf(file,"\n");
                DEBUG_PRINT(DEBUGLEV_DEVELOP, %s,line);
            }
        }
        fclose(file);
    }
    else
    {
        fprintf(stderr, "Cannot open file %s\n", markerfile);
        fprintf(stderr, "%s", strerror(errno));
    }

}

void __attribute__((destructor (101))) likwid_markerCloseDestruct(void)
{
    LikwidResults* results = NULL;
    int numberOfThreads = 0;
    int numberOfRegions = 0;
    if (!likwid_init)
        return;
    hashTable_finalize(&numberOfThreads, &numberOfRegions, &results);
    if ((numberOfThreads == 0)||(numberOfThreads == 0))
    {
        return;
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
        free(results[i].cpulist);
        free(results[i].counters);
    }

    if (results != NULL)
    {
        free(results);
    }
    likwid_init = 0;
    HPMfinalize();
}

int
likwid_markerRegisterRegion(const char* regionTag)
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
    return HPMaddThread(cpu_id);
}



int
likwid_markerStartRegion(const char* regionTag)
{
    if ( ! likwid_init )
    {
        return -EFAULT;
    }
    int myCPU = likwid_getProcessorId();
    if (getThreadID(myCPU) < 0)
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
    results->cpuID = cpu_id;
    results->state = REGION_RUNNING;
    lastStartedRegions[thread_id] = bstrcpy(tag);
    for(int i=0;i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, START [%s] READ EVENT [%d=%d] EVENT %d VALUE %llu,
                regionTag, thread_id, cpu_id, i,
                LLU_CAST groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData);

        results->StartPMcounters[i] = groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData;
        results->LastPMcounters[i] = groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData;
        results->StartOverflows[i] = 0; 
    }


    if (!ovf_started && pthread_equal(mainThread, pthread_self()))
    {
        ovf_started = 1;
        ovf_checks = 0;
        signal(SIGALRM, likwid_overflowRead);
        usalarm(ovf_secs, ovf_usecs);
    }
    nest_level[thread_id]++;

    bdestroy(tag);
    timer_start(&(results->startTime));
    return 0;
}



int
likwid_markerStopRegion(const char* regionTag)
{
    if (! likwid_init)
    {
        return -EFAULT;
    }

    TimerData timestamp;
    timer_stop(&timestamp);
    double result = 0.0;
    int cpu_id = 0, ovf_scale = ovf_checks;
    int myCPU = likwid_getProcessorId();
    if (getThreadID(myCPU) < 0)
    {
        return -EFAULT;
    }
    usalarm(0, 0);
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
    if (thread_id == 0)
    {
        results->time -= (((double)ovf_scale)/4) * timer_print(&(ovf_data.startTime));
    }
    results->count++;
    results->state = REGION_STOPPED;
    nest_level[thread_id]--;
    if (nest_level[thread_id] == 0 && pthread_equal(mainThread, pthread_self()))
    {
        usalarm(0,0);
        ovf_started = 0;
        ovf_checks = 0;
    }

    perfmon_readCountersCpu(cpu_id);

    for(int i=0;i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, STOP [%s] READ EVENT [%d=%d] EVENT %d VALUE %llu, regionTag, thread_id, cpu_id, i,
                        LLU_CAST groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData);
        if (groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData < results->LastPMcounters[i])
            results->StopOverflows[i]++;
        int overflows = results->StopOverflows[i] - results->StartOverflows[i];
        result = calculateMarkerResult(groupSet->groups[groupSet->activeGroup].events[i].index, results->StartPMcounters[i],
                                        groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData,
                                        overflows);
        if ( overflows > 0)
        {
            hashTable_updateOverflows(cpu_id, i, overflows, tag);
        }
        if (counter_map[groupSet->groups[groupSet->activeGroup].events[i].index].type == THERMAL)
        {
            results->PMcounters[i] = result;
        }
        else if (thread_id == 0)
        {
            if (ovf_scale > 0 && result - ((((double)ovf_scale)/4) * ovf_data.PMcounters[i]) > 0)
                results->PMcounters[i] += result - ((((double)ovf_scale)/4) * ovf_data.PMcounters[i]);
            else
                results->PMcounters[i] += result;
        }
        else
        {
            results->PMcounters[i] += result;
        }
    }
    bdestroy(tag);
    if (use_locks == 1)
    {
        pthread_mutex_unlock(&threadLocks[myCPU]);
    }
    return 0;
}

void
likwid_markerGetRegion(
        const char* regionTag,
        int* nr_events,
        double* events,
        double *time,
        int *count)
{
    if (! likwid_init)
    {
        *nr_events = 0;
        *time = 0;
        *count = 0;
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
    bdestroy(tag);
    return;
}

int
likwid_getProcessorId()
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
int
likwid_pinThread(int processorId)
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

int
likwid_pinProcess(int processorId)
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

