/*
 * =======================================================================================
 *
 *      Filename:  libperfctr.c
 *
 *      Description:  Marker API interface of module perfmon
 *
 *      Version:   5.1.0
 *      Released:  20.11.2020
 *
 *      Authors:  Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2020 RRZE, University Erlangen-Nuremberg
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
#include <math.h>

#include <likwid.h>
#include <bitUtil.h>
#include <lock.h>
#include <tree.h>
#include <timer.h>
#include <hashTable.h>
#include <registers.h>
#include <error.h>
#include <access.h>
#include <affinity.h>
#include <perfmon.h>
#include <bstrlib.h>
#include <voltage.h>

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

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
    uint64_t maxValue = 0ULL;
    if (overflows == 0)
    {
        result = (double) (stop - start);
    }
    else if (overflows > 0)
    {
        maxValue = perfmon_getMaxCounterValue(counter_map[index].type);
        result += (double) ((maxValue - start) + stop);
        if (overflows > 1)
        {
            result += (double) ((overflows-1) * maxValue);
        }
    }
    if (counter_map[index].type == POWER)
    {
        result *= power_getEnergyUnit(getCounterTypeOffset(index));
    }
    else if ((counter_map[index].type == THERMAL) ||
             (counter_map[index].type == MBOX0TMP))
    {
        result = (double)stop;
    }
    else if (counter_map[index].type == VOLTAGE)
    {
        result = voltage_value(stop);
    }
    return result;
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
    char* perfpid = getenv("LIKWID_PERF_EXECPID");
    char* debugStr = getenv("LIKWID_DEBUG");
    char* pinStr = getenv("LIKWID_PIN");
    char execpid[20];
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

#ifndef LIKWID_USE_PERFEVENT
    HPMmode(atoi(modeStr));
#endif
    if (debugStr != NULL)
    {
        perfmon_verbosity = atoi(debugStr);
        verbosity = perfmon_verbosity;
    }

    bThreadStr = bfromcstr(cThreadStr);
    threadTokens = bsplit(bThreadStr,',');
    num_cpus = threadTokens->qty;
    for (i=0; i<num_cpus; i++)
    {
        threads2Cpu[i] = ownatoi(bdata(threadTokens->entry[i]));
    }
    bdestroy(bThreadStr);
    bstrListDestroy(threadTokens);

    if (pinStr != NULL)
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
#ifdef LIKWID_USE_PERFEVENT
    if (perfpid != NULL)
    {
        snprintf(execpid, 19, "%d", getpid());
        setenv("LIKWID_PERF_PID", execpid, 1);
        char* perfflags = getenv("LIKWID_PERF_FLAGS");
        if (perfflags)
        {
            setenv("LIKWID_PERF_FLAGS", getenv("LIKWID_PERF_FLAGS"), 1);
        }
    }
#endif

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
    threads2Pthread[registered_cpus] = pthread_self();
    registered_cpus++;

    groupSet->activeGroup = 0;

    perfmon_setupCounters(groupSet->activeGroup);
    perfmon_startCounters();
}

void
likwid_markerThreadInit(void)
{
    int myID = 0, i = 0;
    pthread_t t;
    if ( !likwid_init )
    {
        return;
    }
    char* pinStr = getenv("LIKWID_PIN");

    pthread_mutex_lock(&globalLock);
    t = pthread_self();
    for (i=0; i<registered_cpus; i++)
    {
        if (pthread_equal(t, threads2Pthread[i]))
        {
            t = 0;
        }
    }
    if (t != 0)
    {
        threads2Pthread[registered_cpus] = t;
        myID = registered_cpus++;
    }
    pthread_mutex_unlock(&globalLock);

    if (pinStr != NULL)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        sched_getaffinity(gettid(), sizeof(cpu_set_t), &cpuset);
        if ((CPU_COUNT(&cpuset) > 1) || (likwid_getProcessorId() != threads2Cpu[myID % num_cpus]))
        {
            likwid_pinThread(threads2Cpu[myID % num_cpus]);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Pin thread %lu to CPU %d currently %d, gettid(), threads2Cpu[myID % num_cpus], sched_getcpu());
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
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Switch from group %d to group %d, groupSet->activeGroup, next_group);
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
void
likwid_markerClose(void)
{
    FILE *file = NULL;
    LikwidResults* results = NULL;
    int numberOfThreads = 0;
    int numberOfRegions = 0;
    char* markerfile = NULL;
    int* validRegions = NULL;

    if ( ! likwid_init )
    {
        return;
    }
    hashTable_finalize(&numberOfThreads, &numberOfRegions, &results);
    if ((numberOfThreads == 0)||(numberOfRegions == 0))
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
    validRegions = (int*)malloc(numberOfRegions*sizeof(int));
    if (!validRegions)
    {
        return;
    }
    for (int i=0; i<numberOfRegions; i++)
    {
        validRegions[i] = 0;
    }
    file = fopen(markerfile,"w");

    if (file != NULL)
    {
        int newNumberOfRegions = 0;
        int newRegionID = 0;
        for (int i=0; i<numberOfRegions; i++)
        {
            for (int j=0; j<numberOfThreads; j++)
            {
                validRegions[i] += results[i].count[j];
            }
            if (validRegions[i] > 0)
                newNumberOfRegions++;
            else
                fprintf(stderr, "WARN: Skipping region %s for evaluation.\n", bdata(results[i].tag));
        }
        if (newNumberOfRegions < numberOfRegions)
        {
            fprintf(stderr, "WARN: Regions are skipped because:\n");
            fprintf(stderr, "      - The region was only registered\n");
            fprintf(stderr, "      - The region was started but never stopped\n");
            fprintf(stderr, "      - The region was never started but stopped\n");
        }
        DEBUG_PRINT(DEBUGLEV_DEVELOP,
                Creating Marker file %s with %d regions %d groups and %d threads,
                markerfile, newNumberOfRegions, numberOfGroups, numberOfThreads);
        bstring thread_regs_grps = bformat("%d %d %d", numberOfThreads, newNumberOfRegions, numberOfGroups);
        fprintf(file,"%s\n", bdata(thread_regs_grps));
        DEBUG_PRINT(DEBUGLEV_DEVELOP, %s, bdata(thread_regs_grps));
        bdestroy(thread_regs_grps);

        for (int i=0; i<numberOfRegions; i++)
        {
            if (validRegions[i] == 0)
                continue;
            bstring tmp = bformat("%d:%s", newRegionID, bdata(results[i].tag));
            fprintf(file,"%s\n", bdata(tmp));
            DEBUG_PRINT(DEBUGLEV_DEVELOP, %s, bdata(tmp));
            bdestroy(tmp);
            newRegionID++;
        }
        newRegionID = 0;
        for (int i=0; i<numberOfRegions; i++)
        {
            if (validRegions[i] == 0)
                continue;
            int nevents = groupSet->groups[results[i].groupID].numberOfEvents;
            for (int j=0; j<numberOfThreads; j++)
            {
                bstring l = bformat("%d %d %d %u %e %d ", newRegionID,
                                                          results[i].groupID,
                                                          results[i].cpulist[j],
                                                          results[i].count[j],
                                                          results[i].time[j],
                                                          nevents);

                for (int k=0; k < MIN(nevents, NUM_PMC); k++)
                {
                    bstring tmp = bformat("%e ", results[i].counters[j][k]);
                    bconcat(l, tmp);
                    bdestroy(tmp);
                }
                fprintf(file,"%s\n", bdata(l));
                DEBUG_PRINT(DEBUGLEV_DEVELOP, %s, bdata(l));
                bdestroy(l);
            }
            newRegionID++;
        }
        fclose(file);
    }
    else
    {
        fprintf(stderr, "Cannot open file %s\n", markerfile);
        fprintf(stderr, "%s", strerror(errno));
    }
    if (validRegions)
    {
        free(validRegions);
    }
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
    int ret = 0;
    uint64_t tmp = 0x0ULL;
    bstring tag = bfromcstralloc(100, regionTag);
    LikwidThreadResults* results;
    char groupSuffix[10];
    sprintf(groupSuffix, "-%d", groupSet->activeGroup);
    bcatcstr(tag, groupSuffix);
    int cpu_id = hashTable_get(tag, &results);
    bdestroy(tag);

#ifndef LIKWID_USE_PERFEVENT
    // Add CPU to access layer if ACCESSMODE is direct or accessdaemon
    ret =  HPMaddThread(cpu_id);
    // Perform one access to fully initialize connection to access daemon
    uint32_t reg = counter_map[groupSet->groups[groups[0]].events[0].index].counterRegister;
    HPMread(cpu_id, MSR_DEV, reg, &tmp);
#endif
    return ret;
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
    if (results->state == MARKER_STATE_START)
    {
        fprintf(stderr, "WARN: Region %s was already started\n", regionTag);
    }
    perfmon_readCountersCpu(cpu_id);
    results->cpuID = cpu_id;
    for(int i=0;i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
    {
        if (groupSet->groups[groupSet->activeGroup].events[i].type != NOTYPE)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, START [%s] READ EVENT [%d=%d] EVENT %d VALUE %llu,
                    regionTag, thread_id, cpu_id, i,
                    LLU_CAST groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData);
            //groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].startData =
            //        groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData;

            results->StartPMcounters[i] = groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData;
            results->StartOverflows[i] = groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].overflows;
        }
        else
        {
            results->StartPMcounters[i] = NAN;
            results->StartOverflows[i] = -1;
        }
    }
    results->state = MARKER_STATE_START;

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
    int cpu_id;
    int myCPU = likwid_getProcessorId();
    if (getThreadID(myCPU) < 0)
    {
        return -EFAULT;
    }
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
    if (results->state != MARKER_STATE_START)
    {
        fprintf(stderr, "WARN: Stopping an unknown/not-started region %s\n", regionTag);
        return -EFAULT;
    }
    results->groupID = groupSet->activeGroup;
    results->startTime.stop.int64 = timestamp.stop.int64;
    results->time += timer_print(&(results->startTime));
    results->count++;
    bdestroy(tag);

    perfmon_readCountersCpu(cpu_id);

    for(int i=0;i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
    {
        if (groupSet->groups[groupSet->activeGroup].events[i].type != NOTYPE)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, STOP [%s] READ EVENT [%d=%d] EVENT %d VALUE %llu, regionTag, thread_id, cpu_id, i,
                            LLU_CAST groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData);
            result = calculateMarkerResult(groupSet->groups[groupSet->activeGroup].events[i].index, results->StartPMcounters[i],
                                            groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData,
                                            groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].overflows -
                                            results->StartOverflows[i]);
            if ((counter_map[groupSet->groups[groupSet->activeGroup].events[i].index].type != THERMAL) &&
                (counter_map[groupSet->groups[groupSet->activeGroup].events[i].index].type != VOLTAGE) &&
                (counter_map[groupSet->groups[groupSet->activeGroup].events[i].index].type != MBOX0TMP))
            {
                results->PMcounters[i] += result;
            }
            else
            {
                results->PMcounters[i] = result;
            }
        }
        else
        {
            results->PMcounters[i] = NAN;
        }
    }
    results->state = MARKER_STATE_STOP;
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
    if (count != NULL)
    {
        *count = results->count;
    }
    if (time != NULL)
    {
        *time = results->time;
    }
    if (nr_events != NULL && events != NULL && *nr_events > 0)
    {
        length = MIN(groupSet->groups[groupSet->activeGroup].numberOfEvents, *nr_events);
        for(int i=0;i<length;i++)
        {
            events[i] = results->PMcounters[i];
        }
        *nr_events = length;
    }
    bdestroy(tag);
    return;
}


int
likwid_markerResetRegion(const char* regionTag)
{
    if (! likwid_init)
    {
        return -EFAULT;
    }
    int cpu_id;
    int myCPU = likwid_getProcessorId();
    if (getThreadID(myCPU) < 0)
    {
        return -EFAULT;
    }
    bstring tag = bfromcstr(regionTag);
    char groupSuffix[100];
    LikwidThreadResults* results;
    sprintf(groupSuffix, "-%d", groupSet->activeGroup);
    bcatcstr(tag, groupSuffix);

    cpu_id = hashTable_get(tag, &results);
    if (results->state != MARKER_STATE_STOP)
    {
        fprintf(stderr, "ERROR: Can only reset stopped regions\n");
        return -EFAULT;
    }

    memset(results->StartPMcounters, 0, groupSet->groups[groupSet->activeGroup].numberOfEvents*sizeof(double));
    memset(results->PMcounters, 0, groupSet->groups[groupSet->activeGroup].numberOfEvents*sizeof(double));
    memset(results->StartOverflows, 0, groupSet->groups[groupSet->activeGroup].numberOfEvents*sizeof(double));
    results->count = 0;
    results->time = 0;
    timer_reset(&results->startTime);
    return 0;
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
