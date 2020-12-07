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
 *      Authors:  Thomas Gruber (tr), thomas.roehl@googlemail.com
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

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <syscall.h>

#include <likwid.h>
#include <lock.h>
#include <registers_types.h>
#include <map.h>
#include <error.h>
#include <bstrlib.h>
#include <perfmon.h>

#include <libperfctr_types.h>

#define gettid() syscall(SYS_gettid)


/* Global flag whether MarkerAPI is active */
static int likwid_init = 0;

/* Store group information */
static int activeGroup = -1;
static int numberOfGroups = 0;
static int* groups = NULL;

/* Store thread information */
static int num_cpus = 0;
static int *threads2Cpu = NULL;
static int *cpus2threads = NULL;

typedef struct {
    int thread_id;
    int cpu_id;
    off_t last;
    LikwidThreadResults* last_res;
    Map_t regions;
    uint64_t _padding[4];
} GroupThreadsMap;

typedef struct {
    int numberOfThreads;
    GroupThreadsMap *threads;
} GroupThreads;

typedef struct {
    int numberOfGroups;
    GroupThreads *groups;
} MarkerGroups;

static MarkerGroups* mgroups = NULL;
#define GET_THREAD_MAP(groupID, cpuID) mgroups->groups[(groupID)].threads[cpus2threads[(cpuID)]].regions
#define GET_THREAD_LAST(groupID, cpuID) mgroups->groups[(groupID)].threads[cpus2threads[(cpuID)]].last
#define GET_THREAD_LASTRES(groupID, cpuID) mgroups->groups[(groupID)].threads[cpus2threads[(cpuID)]].last_res

/* Some locking data */
static pthread_mutex_t globalLock = PTHREAD_MUTEX_INITIALIZER;
static int use_locks = 0;
static pthread_mutex_t* threadLocks = NULL;

static int listsplit(char* list, int** outlist, char splitchar)
{
    int ret = 0;
    bstring blist = bfromcstr(list);
    struct bstrList* items = bsplit(blist, splitchar);
    int* out = malloc(items->qty * sizeof(int));
    if (!out)
    {
        return -1;
    }
    for (int i = 0; i < items->qty; i++)
    {
        out[i] = atoi(bdata(items->entry[i]));
    }
    ret = items->qty;
    bdestroy(blist);
    bstrListDestroy(items);
    *outlist = out;
    return ret;
}

static int allocate_markerGroups(int numberOfGroups, int numberOfThreads)
{
    if (!mgroups)
    {
        mgroups = malloc(sizeof(MarkerGroups));
        if (mgroups)
        {
            mgroups->groups = malloc(numberOfGroups * sizeof(GroupThreads));
            if (mgroups->groups)
            {
                mgroups->numberOfGroups = numberOfGroups;
                for (int i = 0; i < numberOfGroups; i++)
                {
                    mgroups->groups[i].threads = malloc(numberOfThreads * sizeof(GroupThreadsMap));
                    if (!mgroups->groups[i].threads)
                    {
                        for (int j = 0; j < i; j++)
                        {
                            free(mgroups->groups[j].threads);
                        }
                        free(mgroups->groups);
                        free(mgroups);
                        mgroups = NULL;
                        return -ENOMEM;
                    }
                    mgroups->groups[i].numberOfThreads = numberOfThreads;
                    memset(mgroups->groups[i].threads, 0, numberOfThreads * sizeof(GroupThreadsMap));
                    for (int j = 0; j < numberOfThreads; j++)
                    {
                        init_smap(&mgroups->groups[i].threads[j].regions);
                    }
                }
            }
        }
    }
    return 0;
}

static void destroy_markerGroups()
{
    if (mgroups)
    {
        for (int i = 0; i < mgroups->numberOfGroups; i++)
        {
            for (int j = 0; j < mgroups->groups[i].numberOfThreads; j++)
            {
                Map_t m = GET_THREAD_MAP(i, j);
                destroy_smap(m);
            }
            free(mgroups->groups[i].threads);
        }
        free(mgroups->groups);
        free(mgroups);
        mgroups = NULL;
    }
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
    else if (counter_map[index].type == THERMAL)
    {
        result = (double)stop;
    }
    return result;
}

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

void
likwid_markerInit(void)
{
    int i = 0;
    int ret = 0;
    int setinit = 0;
    char* modeStr = getenv("LIKWID_MODE");
    char* eventStr = getenv("LIKWID_EVENTS");
    char* cThreadStr = getenv("LIKWID_THREADS");
    char* filepath = getenv("LIKWID_FILEPATH");
    char execpid[30];

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
        setinit = 0;
        return;
    }

    char* perfpid = getenv("LIKWID_PERF_EXECPID");
    char* debugStr = getenv("LIKWID_DEBUG");
    char* pinStr = getenv("LIKWID_PIN");

#ifndef LIKWID_USE_PERFEVENT
    HPMmode(atoi(modeStr));
#endif
    if (debugStr != NULL)
    {
        perfmon_verbosity = atoi(debugStr);
/*        verbosity = perfmon_verbosity;*/
    }

    num_cpus = listsplit(cThreadStr, &threads2Cpu, ',');
    if (num_cpus <= 0)
    {
        fprintf(stderr, "Failed to read threads 2 CPU string: %s\n", cThreadStr);
        setinit = 0;
        return;
    }

    if (pinStr != NULL)
    {
        likwid_pinThread(threads2Cpu[0]);
        if (getenv("OMP_NUM_THREADS") != NULL)
        {
            if (atoi(getenv("OMP_NUM_THREADS")) > num_cpus)
            {
                use_locks = 1;
            }
        }
        if (getenv("CILK_NWORKERS") != NULL)
        {
            if (atoi(getenv("CILK_NWORKERS")) > num_cpus)
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
    ret = perfmon_init(num_cpus, threads2Cpu);
    if (ret < 0)
    {
        free(threads2Cpu);
        threads2Cpu = NULL;
        num_cpus = 0;
    }

    bstring bEventStr = bfromcstr(eventStr);
    struct bstrList* eventStrings = bsplit(bEventStr,'|');
    numberOfGroups = eventStrings->qty;
    groups = malloc(numberOfGroups * sizeof(int));
    if (!groups)
    {
        fprintf(stderr,"Cannot allocate space for group handling.\n");
        bstrListDestroy(eventStrings);
        bdestroy(bEventStr);
        free(threads2Cpu);
        threads2Cpu = NULL;
        num_cpus = 0;
        return;
    }
    for (i=0; i<eventStrings->qty; i++)
    {
        groups[i] = perfmon_addEventSet(bdata(eventStrings->entry[i]));
    }
    bstrListDestroy(eventStrings);
    bdestroy(bEventStr);

    cpus2threads = malloc(num_cpus * sizeof(int));
    if (!cpus2threads)
    {
        free(threads2Cpu);
        threads2Cpu = NULL;
        num_cpus = 0;
        return;
    }

    for (i = 0; i < num_cpus; i++)
    {
        cpus2threads[threads2Cpu[i]] = i;
    }

    ret = allocate_markerGroups(numberOfGroups, num_cpus);
    if (ret < 0)
    {
        free(cpus2threads);
        cpus2threads = NULL;
        free(threads2Cpu);
        threads2Cpu = NULL;
        num_cpus = 0;
        return;
    }

    if (setinit)
    {
        likwid_init = 1;
    }

    activeGroup = groups[0];
    perfmon_setupCounters(activeGroup);
    perfmon_startCounters();
}

int
likwid_markerRegisterRegion(const char* regionTag)
{
    if ( ! likwid_init )
    {
        return -EFAULT;
    }
    LikwidThreadResults* results = NULL;
    int cpu_id = sched_getcpu();
    Map_t m = GET_THREAD_MAP(activeGroup, cpu_id);

    if (get_smap_by_key(m, (char*)regionTag, (void**)&results) < 0)
    {
        results = malloc(sizeof(LikwidThreadResults));
        if (results)
        {
            results->label = bfromcstr(regionTag);
            results->time = 0.0;
            results->count = 0;
            results->cpuID = cpu_id;
            results->state = MARKER_STATE_NEW;
            for (int i=0; i< NUM_PMC; i++)
            {
                results->PMcounters[i] = 0.0;
                results->StartPMcounters[i] = 0.0;
            }
            add_smap(m, (char*)regionTag, results);
            mgroups->groups[(activeGroup)].threads[cpus2threads[(cpu_id)]].last = (off_t)regionTag;
            mgroups->groups[(activeGroup)].threads[cpus2threads[(cpu_id)]].last_res = results;
        }
    }
    else
    {
        //fprintf(stderr, "Region '%s' already registered\n", regionTag);
    }
    return 0;
}

int
likwid_markerStartRegion(const char* regionTag)
{
    if ( ! likwid_init )
    {
        return -EFAULT;
    }
    TimerData start;
    LikwidThreadResults* results = NULL;
    int cpu_id = sched_getcpu();
    int thread_id = cpus2threads[cpu_id];
    Map_t m = GET_THREAD_MAP(activeGroup, cpu_id);
    off_t lastTag = GET_THREAD_LAST(activeGroup, cpu_id);
    if (lastTag == (off_t)regionTag)
    {
        results = GET_THREAD_LASTRES(activeGroup, cpu_id);
    }
    else
    {
        int ret = get_smap_by_key(m, (char*)regionTag, (void**)&results);
        if (ret < 0)
        {
            likwid_markerRegisterRegion(regionTag);
            get_smap_by_key(m, (char*)regionTag, (void**)&results);
        }
    }

    if (results->state == MARKER_STATE_START)
    {
        fprintf(stderr, "WARN: Region %s was already started\n", regionTag);
        return -EFAULT;
    }

    timer_start(&start);
    perfmon_readCountersCpu(cpu_id);
    timer_stop(&start);
    //printf("MARKER START %f\n", timer_print(&start));
    for(int i = 0; i < perfmon_getNumberOfEvents(activeGroup); i++)
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
    mgroups->groups[(activeGroup)].threads[cpus2threads[(cpu_id)]].last = (off_t)regionTag;
    mgroups->groups[(activeGroup)].threads[cpus2threads[(cpu_id)]].last_res = results;
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
    TimerData stop;
    timer_stop(&timestamp);

    LikwidThreadResults* results = NULL;
    int cpu_id = sched_getcpu();
    int thread_id = cpus2threads[cpu_id];
    Map_t m = GET_THREAD_MAP(activeGroup, cpu_id);
    off_t lastTag = GET_THREAD_LAST(activeGroup, cpu_id);
    if (lastTag == (off_t)regionTag)
    {
        results = GET_THREAD_LASTRES(activeGroup, cpu_id);
    }
    else
    {
        int ret = get_smap_by_key(m, (char*)regionTag, (void**)&results);
        if (ret < 0)
        {
            fprintf(stderr, "WARN: Stopping an unknown region '%s'\n", regionTag);
            return -EFAULT;
        }
    }
    // if (get_smap_by_key(m, (char*)regionTag, (void**)&results) < 0)
    // {
    //     fprintf(stderr, "WARN: Stopping an unknown region '%s'\n", regionTag);
    //     return -EFAULT;
    // }
    if (results->state != MARKER_STATE_START)
    {
        fprintf(stderr, "WARN: Stopping an not-started region '%s'\n", regionTag);
        return -EFAULT;
    }
    results->startTime.stop.int64 = timestamp.stop.int64;
    results->time += timer_print(&(results->startTime));
    results->count++;
    timer_start(&stop);
    perfmon_readCountersCpu(cpu_id);
    timer_stop(&stop);
    //printf("MARKER STOP %f\n", timer_print(&stop));
    for(int i = 0; i < perfmon_getNumberOfEvents(activeGroup); i++)
    {
        if (groupSet->groups[groupSet->activeGroup].events[i].type != NOTYPE)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, STOP [%s] READ EVENT [%d=%d] EVENT %d VALUE %llu, regionTag, thread_id, cpu_id, i,
                            LLU_CAST groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData);
            double result = calculateMarkerResult(groupSet->groups[groupSet->activeGroup].events[i].index, results->StartPMcounters[i],
                                            groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData,
                                            groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].overflows -
                                            results->StartOverflows[i]);
            if (counter_map[groupSet->groups[groupSet->activeGroup].events[i].index].type != THERMAL)
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
    mgroups->groups[(activeGroup)].threads[cpus2threads[(cpu_id)]].last = (off_t)regionTag;
    mgroups->groups[(activeGroup)].threads[cpus2threads[(cpu_id)]].last_res = results;
    results->state = MARKER_STATE_STOP;
    return 0;
}

void
likwid_markerClose(void)
{
    FILE *file = NULL;
    LikwidResults* results = NULL;
    char* markerfile = NULL;
    int* validRegions = NULL;

    if ( ! likwid_init )
    {
        return;
    }
    markerfile = getenv("LIKWID_FILEPATH");
    if (markerfile == NULL)
    {
        fprintf(stderr, "Is the application executed with LIKWID wrapper? No file path for the Marker API output defined.\n");
        return;
    }

    int numberOfRegions = 0;
    int numberOfThreads = num_cpus;
    for (int i = 0; i < num_cpus; i++)
    {
        for (int j = 0; j < numberOfGroups; j++)
        {
            int s = get_map_size(GET_THREAD_MAP(j, threads2Cpu[i]));
            numberOfRegions = (numberOfRegions > s ? numberOfRegions : s );
        }
    }

    validRegions = malloc(numberOfRegions * numberOfGroups * sizeof(int));
    if (!validRegions)
    {
        return;
    }
    memset(validRegions, 0, numberOfRegions * numberOfGroups * sizeof(int));

    for (int i = 0; i < num_cpus; i++)
    {
        for (int j = 0; j < numberOfGroups; j++)
        {
            Map_t m = GET_THREAD_MAP(j, threads2Cpu[i]);
            for (int k = 0; k < numberOfRegions; k++)
            {
                LikwidThreadResults* results = NULL;
                if (get_smap_by_idx(m, k, (void**)&results) == 0)
                {
                    validRegions[(j*numberOfGroups) + k] += results->count;
                }
                else
                {
                    fprintf(stderr, "No region defined for group %d and Thread %d (CPU %d)\n", j, i, threads2Cpu[i]);
                }
            }
        }
    }

    int newNumberOfRegions = 0;
    for (int i = 0; i < numberOfRegions * numberOfGroups; i++)
    {
        if (validRegions[i] > 0) newNumberOfRegions++;
    }
    if (newNumberOfRegions < numberOfRegions)
    {
        fprintf(stderr, "WARN: Regions are skipped because:\n");
        fprintf(stderr, "      - The region was only registered\n");
        fprintf(stderr, "      - The region was started but never stopped\n");
        fprintf(stderr, "      - The region was never started but stopped\n");
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Creating Marker file %s with %d regions %d groups and %d threads,
                                markerfile, newNumberOfRegions, numberOfGroups, numberOfThreads);
    bstring out = bformat("%d %d %d\n", numberOfThreads, newNumberOfRegions, numberOfGroups);
    int newRegionID = 0;
    for (int i = 0; i < numberOfRegions; i++)
    {
        if (validRegions[i] == 0)
        {
            continue;
        }
        Map_t m = GET_THREAD_MAP(0, threads2Cpu[0]);
        if (get_smap_by_idx(m, i, (void**)&results) == 0)
        {
            bstring tmp = bformat("%d:%s-%d\n", newRegionID, bdata(results->tag), results->groupID);
            bconcat(out, tmp);
            bdestroy(tmp);
            newRegionID++;
        }
    }
    newRegionID = 0;
    for (int i = 0; i < numberOfRegions; i++)
    {
        for (int j = 0; j < numberOfGroups; j++)
        {
            int nevents = perfmon_getNumberOfEvents(groups[j]);
            for (int k = 0; k < numberOfThreads; k++)
            {
                Map_t m = GET_THREAD_MAP(j, threads2Cpu[k]);
                LikwidThreadResults* tresults = NULL;
                if (get_smap_by_idx(m, k, (void**)&tresults) == 0)
                {
                    if (tresults->count == 0)
                    {
                        continue;
                    }
                    bstring la = bformat("%d %d %d %u %e %d ", newRegionID,
                                                              groups[j],
                                                              threads2Cpu[k],
                                                              tresults->count,
                                                              tresults->time,
                                                              nevents);
                    for (int l=0; l < MIN(nevents, NUM_PMC); l++)
                    {
                        bstring tmp = bformat("%e ", tresults->PMcounters[l]);
                        bconcat(la, tmp);
                        bdestroy(tmp);
                    }
                    bconchar(la, '\n');
                    bconcat(out, la);
                    bdestroy(la);
                }
            }
            newRegionID++;
        }
    }
    file = fopen(markerfile,"w");
    if (file)
    {
        if (perfmon_verbosity == DEBUGLEV_DEVELOP)
            printf("%s\n", bdata(out));
        fprintf(file,"%s", bdata(out));
        fclose(file);
    }
    else
    {
        fprintf(stderr, "Cannot open file %s\n", markerfile);
        fprintf(stderr, "%s", strerror(errno));
    }
    free(validRegions);
    destroy_markerGroups();
    return;
}

int
likwid_markerResetRegion(const char* regionTag)
{
    if ( ! likwid_init )
    {
        return -EFAULT;
    }
    LikwidThreadResults* results = NULL;
    int cpu_id = sched_getcpu();
    int thread_id = cpus2threads[cpu_id];
    Map_t m = GET_THREAD_MAP(activeGroup, cpu_id);
    if (get_smap_by_key(m, (char*)regionTag, (void**)&results) < 0)
    {
        fprintf(stderr, "WARN: Resetting an unknown region '%s'\n", regionTag);
        return -EFAULT;
    }
    if (results->state != MARKER_STATE_STOP)
    {
        fprintf(stderr, "WARN: Resetting a started region '%s'\n", regionTag);
        return -EFAULT;
    }
    results->time = 0.0;
    results->count = 0;
    results->cpuID = cpu_id;
    results->state = MARKER_STATE_NEW;
    for (int i=0; i< NUM_PMC; i++)
    {
        results->PMcounters[i] = 0.0;
        results->StartPMcounters[i] = 0.0;
    }
    return 0;
}

int
likwid_getProcessorId()
{
    cpu_set_t  cpu_set;
    CPU_ZERO(&cpu_set);
    sched_getaffinity(gettid(),sizeof(cpu_set_t), &cpu_set);
    if (CPU_COUNT(&cpu_set) > 1)
        return sched_getcpu();
    else
        getProcessorID(&cpu_set);
    return -1;
}

void
likwid_markerNextGroup(void)
{
    if ( ! likwid_init )
    {
        return;
    }

    int new_activeGroup = (activeGroup + 1);
    if (new_activeGroup >= numberOfGroups)
        new_activeGroup = 0;

    if (new_activeGroup != activeGroup)
    {
        perfmon_stopCounters();
        perfmon_setupCounters(activeGroup);
        perfmon_startCounters();
        activeGroup = new_activeGroup;
    }

    return;
}

void
likwid_markerThreadInit(void)
{
    return;
}

void
likwid_markerGetRegion(
        const char* regionTag,
        int* nr_events,
        double* events,
        double *time,
        int *count)
{
    if ( ! likwid_init )
    {
        return;
    }
    LikwidThreadResults* results = NULL;
    int cpu_id = sched_getcpu();
    int thread_id = cpus2threads[cpu_id];
    Map_t m = GET_THREAD_MAP(activeGroup, cpu_id);
    if (get_smap_by_key(m, (char*)regionTag, (void**)&results) < 0)
    {
        fprintf(stderr, "WARN: Getting data for an unknown region '%s'\n", regionTag);
        return;
    }
    if (results->state != MARKER_STATE_STOP)
    {
        fprintf(stderr, "WARN: Getting data for a started region '%s'\n", regionTag);
        return;
    }
    if (nr_events != NULL && events != NULL && *nr_events > 0)
    {
        int nevents = perfmon_getNumberOfEvents(results->groupID);
        for (int i=0; i< MIN(*nr_events, nevents); i++)
        {
            events[i] = results->PMcounters[i];
        }
        *nr_events = MIN(*nr_events, nevents);
    }
    if (time != NULL) *time = results->time;
    if (count != NULL) *count = results->count;
    return;
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
