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
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <inttypes.h>
#include <math.h>

#include <cwisstable/declare.h>
#include <cwisstable/policy.h>

#include <likwid.h>
#include <bitUtil.h>
#include <lock.h>
#include <tree.h>
#include <timer.h>
#include <registers.h>
#include <error.h>
#include <access.h>
#include <affinity.h>
#include <perfmon.h>
#include <bstrlib.h>
#include <voltage.h>
#include <lw_alloc.h>

#define LIKWID_MARKER_MAX_REGION_NAME 1024

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define gettid() syscall(SYS_gettid)



/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

typedef struct {
    char* region;
    int group;
} LikwidThreadKey;

typedef struct {
    LikwidThreadKey* k;
    LikwidThreadResults* v;
} _libperfctr_result_map_Entry;

static inline void libperfctr_result_copy(void* dst, const void* src) {
    const _libperfctr_result_map_Entry* s = (const _libperfctr_result_map_Entry*)src;
    _libperfctr_result_map_Entry* d = (_libperfctr_result_map_Entry*)dst;

    // Copy key
    d->k = lw_malloc(sizeof(LikwidThreadKey));
    size_t len = (strlen(s->k->region) < LIKWID_MARKER_MAX_REGION_NAME-1 ? strlen(s->k->region) : LIKWID_MARKER_MAX_REGION_NAME-1);
    d->k->region = lw_malloc(len + 1);
    memcpy(d->k->region, s->k->region, len + 1);
    d->k->group = s->k->group;

    // Copy value
    d->v = lw_malloc(sizeof(LikwidThreadResults));
    d->v->label = bstrcpy(s->v->label);
    d->v->index = s->v->index;
    d->v->time = s->v->time;
    d->v->groupID = s->v->groupID;
    d->v->cpuID = s->v->cpuID;
    d->v->count = s->v->count;
    d->v->state = s->v->state;
    d->v->startTime.start.int64 = s->v->startTime.start.int64;
    d->v->startTime.stop.int64 = s->v->startTime.stop.int64;
    memcpy(s->v->StartPMcounters, d->v->StartPMcounters, NUM_PMC * sizeof(double));
    memcpy(s->v->StartOverflows, d->v->StartOverflows, NUM_PMC * sizeof(int));
    memcpy(s->v->PMcounters, d->v->PMcounters, NUM_PMC * sizeof(double));
}

static inline void libperfctr_result_dtor(void* val) {
    _libperfctr_result_map_Entry* e = (_libperfctr_result_map_Entry*)val;

    // Destroy key
    free(e->k->region);
    free(e->k);

    // Destroy value
    bdestroy(e->v->label);
    free(e->v);
}


static inline size_t libperfctr_result_hash(const void* val) {
    LikwidThreadKey * const e = *(LikwidThreadKey * const*)val;
    size_t len = strlen(e->region);
    CWISS_FxHash_State state = 0;
    CWISS_FxHash_Write(&state, e->region, len);
    CWISS_FxHash_Write(&state, &e->group, sizeof(int));
    return state;
}

static inline bool libperfctr_result_eq(const void* a, const void* b) {
    const LikwidThreadKey* ap = *(const LikwidThreadKey* const*)a;
    const LikwidThreadKey* bp = *(const LikwidThreadKey* const*)b;
    return strcmp(ap->region, bp->region) == 0 && ap->group == bp->group;
}


CWISS_DECLARE_NODE_MAP_POLICY(libperfctr_result_policy, LikwidThreadKey*, LikwidThreadResults*,
                              (obj_copy, libperfctr_result_copy),
                              (obj_dtor, libperfctr_result_dtor),
                              (key_hash, libperfctr_result_hash),
                              (key_eq, libperfctr_result_eq));

CWISS_DECLARE_HASHMAP_WITH(libperfctr_result_map, LikwidThreadKey*, LikwidThreadResults*, libperfctr_result_policy);

typedef struct {
    int id;
    int process_id;
    int hwthread_id;
    int thread_id;
    int active_group;
    pthread_t pthread;
    libperfctr_result_map regions;
} LikwidMarkerThread;

typedef struct {
    uint64_t k;
    LikwidMarkerThread* v;
} _libperfctr_thread_map_Entry;

static inline void libperfctr_thread_copy(void* dst, const void* src) {
    const _libperfctr_thread_map_Entry* s = (const _libperfctr_thread_map_Entry*)src;
    _libperfctr_thread_map_Entry* d = (_libperfctr_thread_map_Entry*)dst;

    d->k = s->k;

    d->v = lw_malloc(sizeof(LikwidMarkerThread));
    d->v->id = s->v->id;
    d->v->process_id = s->v->process_id;
    d->v->hwthread_id = s->v->hwthread_id;
    d->v->thread_id = s->v->thread_id;
    d->v->active_group = s->v->active_group;
    d->v->pthread = s->v->pthread;
    d->v->regions = libperfctr_result_map_dup(&s->v->regions);
}

static inline void libperfctr_thread_dtor(void* val) {
    _libperfctr_thread_map_Entry* e = (_libperfctr_thread_map_Entry*)val;
    libperfctr_result_map_destroy(&e->v->regions);
    free(e->v);
}

static inline size_t libperfctr_thread_hash(const void* val) {
    const uint64_t e = *(const uint64_t*)val;
    CWISS_FxHash_State state = 0;
    CWISS_FxHash_Write(&state, &e, sizeof(const uint64_t));
    return state;
}

static inline bool libperfctr_thread_eq(const void* a, const void* b) {
    const uint64_t* ap = (const uint64_t*)a;
    const uint64_t* bp = (const uint64_t*)b;
    return *ap == *bp;
}

CWISS_DECLARE_NODE_MAP_POLICY(libperfctr_thread_policy, uint64_t, LikwidMarkerThread*,
                              (obj_copy, libperfctr_thread_copy),
                              (obj_dtor, libperfctr_thread_dtor),
                              (key_hash, libperfctr_thread_hash),
                              (key_eq, libperfctr_thread_eq));

CWISS_DECLARE_HASHMAP_WITH(libperfctr_thread_map, uint64_t, LikwidMarkerThread*, libperfctr_thread_policy);


static libperfctr_thread_map _libperfctr_thread_map;
static int _libperfctr_num_hwthreads = 0;
static int *_libperfctr_thread_2_hwthread = NULL;
static int _libperfctr_init = 0;
static int _libperfctr_verbosity = DEBUGLEV_ONLY_ERROR;
static int _libperfctr_num_groups = 0;
static int *_libperfctr_groups;
static int _libperfctr_use_locks = 0;
static pthread_mutex_t _libperfctr_lock = PTHREAD_MUTEX_INITIALIZER;


static double
calculateMarkerResult(RegisterIndex index, uint64_t start, uint64_t stop, int overflows)
{
    double result = 0.0;
    uint64_t maxValue = 0ULL;
    if (start > stop)
    {
        overflows++;
    }
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


int libperfctr_get_thread(uint64_t id, LikwidMarkerThread** thread) {
    pthread_mutex_lock(&_libperfctr_lock);
    DEBUG_PRINT(DEBUGLEV_DEVELOP, "GET Query for thread %d in group %d", thread->process_id, thread->active_group);
    if (!libperfctr_thread_map_contains(&_libperfctr_thread_map, &id)) {
        LikwidMarkerThread t = {
            .id = id,
            .process_id = id,
            .hwthread_id = sched_getcpu(),
            .thread_id = libperfctr_thread_map_size(&_libperfctr_thread_map),
            .pthread = pthread_self(),
            .regions = libperfctr_result_map_new(10),
            .active_group = 0,
        };
        for (int i = 0; i < _libperfctr_num_hwthreads; i++) {
            if (_libperfctr_thread_2_hwthread[i] == t.hwthread_id) {
                t.thread_id = i;
                break;
            }
        }
        libperfctr_thread_map_Entry entry = {
            .key = id,
            .val = &t,
        };
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "Adding thread %ld to threadmap", id);
        libperfctr_thread_map_Insert ins = libperfctr_thread_map_insert(&_libperfctr_thread_map, &entry);
        if (!ins.inserted) {
            ERROR_PRINT("ERROR: Failed to insert thread %ld\n", id);
        }
        libperfctr_result_map_destroy(&t.regions);
    }
    if (thread) {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "Getting thread %ld from threadmap", id);
        libperfctr_thread_map_Iter iter = libperfctr_thread_map_find(&_libperfctr_thread_map, &id);
        libperfctr_thread_map_Entry *entry = NULL;
        do {
            entry = libperfctr_thread_map_Iter_get(&iter);
            if (entry->key == id) {
                DEBUG_PRINT(DEBUGLEV_DEVELOP, "Found thread %ld in threadmap", id);
                *thread = entry->val;
                break;
            }
        } while (entry);
    }
    pthread_mutex_unlock(&_libperfctr_lock);
    return 0;
}

int libperfctr_add_region(LikwidMarkerThread* thread, const char* regionTag, LikwidThreadResults** results) {
    LikwidThreadKey key = {
        .region = (char*)regionTag,
        .group = thread->active_group,
    };
    LikwidThreadKey* const keyptr = &key;

    DEBUG_PRINT(DEBUGLEV_DEVELOP, "ADD Query for region %s group %d in thread %d", regionTag, thread->active_group, thread->process_id);
    if (!libperfctr_result_map_contains(&thread->regions, &keyptr)) {
        LikwidThreadResults res = {
            .label = bfromcstr(regionTag),
            .count = 0,
            .index = 0,
            .time = 0,
            .groupID = groupSet->activeGroup,
            .state = MARKER_STATE_NEW,
            .cpuID = thread->hwthread_id,
            .startTime.start.int64 = 0,
            .startTime.stop.int64 = 0,
        };
        libperfctr_result_map_Entry entry = {
            .key = &key,
            .val = &res,
        };
        libperfctr_result_map_Insert ins = libperfctr_result_map_insert(&thread->regions, &entry);
        if (!ins.inserted) {
            ERROR_PRINT("ERROR: Failed to insert region '%s' for thread %d", regionTag, thread->process_id);
        } else {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, "Inserted region %s for thread %d", regionTag, thread->process_id);
        }
        bdestroy(res.label);
    }
    if (results) {
        int found = 0;
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "ADD Searching for region %s for thread %d", regionTag, thread->process_id);
        libperfctr_result_map_Iter iter = libperfctr_result_map_find(&thread->regions, &keyptr);
        for (libperfctr_result_map_Entry *re = libperfctr_result_map_Iter_get(&iter); re != NULL; re = libperfctr_result_map_Iter_next(&iter)) {
            *results = re->val;
            DEBUG_PRINT(DEBUGLEV_DEVELOP, "Found region %s for thread %d", regionTag, thread->process_id);
            found = 1;
            break;
        }
        if (!found) {
            errno = ENOENT;
            ERROR_PRINT("Unable to find region %s for thread %d", regionTag, thread->process_id);
            return -ENOENT;
        }
    }
    return 0;
}

int libperfctr_get_region(LikwidMarkerThread* thread, const char* regionTag, LikwidThreadResults** results) {
    DEBUG_PRINT(DEBUGLEV_DEVELOP, "GET Searching for region %s for thread %d", regionTag, thread->process_id);
    LikwidThreadKey key = {
        .region = (char*)regionTag,
        .group = thread->active_group,
    };
    LikwidThreadKey* const keyptr = &key;
    libperfctr_result_map_Iter iter = libperfctr_result_map_find(&thread->regions, &keyptr);
    int found = 0;
    for (libperfctr_result_map_Entry *re = libperfctr_result_map_Iter_get(&iter); re != NULL; re = libperfctr_result_map_Iter_next(&iter)) {
        *results = re->val;
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "Found region %s for thread %d", regionTag, thread->process_id);
        found = 1;
        break;
    }
    if (!found) {
        errno = ENOENT;
        ERROR_PRINT("Unable to find region %s for thread %d", regionTag, thread->process_id);
        return -ENOENT;
    }
    return 0;
}

int get_omp_threads() {
    char* v = NULL;
    if (getenv("OMP_NUM_THREADS") != NULL) {
        v = getenv("OMP_NUM_THREADS");
    }
    return atoi(v);
}

int get_cilk_workers() {
    char* v = NULL;
    if (getenv("CILK_NWORKERS") != NULL) {
        v = getenv("CILK_NWORKERS");
    }
    return atoi(v);
}

void set_perf_event_pid() {
    char execpid[40];
    int ret = snprintf(execpid, sizeof(execpid), "%d", getpid());
    if (ret < 0) {
        return;
    }
    ret = setenv("LIKWID_PERF_PID", execpid, 1);
    if (ret < 0) {
        return;
    }
    return;
}


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ########### */

int likwid_getProcessorId(void) {
    return sched_getcpu();
}

int likwid_pinProcess(int processorId) {
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


void likwid_markerInit(void) {
    int ret = 0;
    int setinit = 0;
    char* modeStr = getenv("LIKWID_MODE");
    char* eventStr = getenv("LIKWID_EVENTS");
    char* cThreadStr = getenv("LIKWID_THREADS");
    char* filepath = getenv("LIKWID_FILEPATH");
    char* debugStr = getenv("LIKWID_DEBUG");
    char* pinStr = getenv("LIKWID_PIN");

    if ((modeStr != NULL) && (filepath != NULL) && (eventStr != NULL) && (cThreadStr != NULL) && (_libperfctr_init == 0)) {
        setinit = 1;
    } else if (_libperfctr_init == 0) {
        fprintf(stderr, "Running without Marker API. Activate Marker API with -m on commandline.\n");
        return;
    } else {
        return;
    }

    if (!lock_check()) {
        fprintf(stderr,"Access to performance counters is locked.\n");
        return;
    }

    if (debugStr != NULL) {
        int verbosity = atoi(debugStr);
        perfmon_setVerbosity(verbosity);
        _libperfctr_verbosity = verbosity;
    }

    bstring bThreadStr = bfromcstr(cThreadStr);
    struct bstrList* threadTokens = bsplit(bThreadStr, ',');
    _libperfctr_thread_2_hwthread = lw_malloc(threadTokens->qty * sizeof(int));
    for (int i = 0; i < threadTokens->qty; i++) {
        char* v = NULL;
        if (threadTokens->entry[i]) {
            v = bdata(threadTokens->entry[i]);
        }
        _libperfctr_thread_2_hwthread[i] = atoi(v);
    }
    _libperfctr_num_hwthreads = threadTokens->qty;
    bdestroy(bThreadStr);
    bstrListDestroy(threadTokens);

    bstring bEventStr = bfromcstr(eventStr);
    struct bstrList* eventStrings = bsplit(bEventStr,'|');
    _libperfctr_groups = lw_malloc(eventStrings->qty * sizeof(int));

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (pinStr != NULL) {
        // If LIKWID should pin the threads, we check whether we are already
        // in a restricted cpuset with a single hwthread. This happens
        // when an OpenMP region is executed before LIKWID_MARKER_INIT.
        // When the first thread starts up, LIKWID pins the master process
        // and the thread. But we want to get the topology without the restriction
        // here. We don't need to set it back because the first operation
        // after getting the topology is the pinning of the master process.
        ret = sched_getaffinity(0, sizeof(cpu_set_t), &cpuset);
        if (ret == 0) {
            int count = CPU_COUNT(&cpuset);
            if (count == 1 && count != _libperfctr_num_hwthreads) {
                cpu_set_t newcpuset;
                CPU_ZERO(&newcpuset);
                for (int i = 0; i < _libperfctr_num_hwthreads; i++) {
                    CPU_SET(_libperfctr_thread_2_hwthread[i], &newcpuset);
                }
                ret = sched_setaffinity(0, sizeof(cpu_set_t), &newcpuset);
                if (ret < 0) {
                    ERROR_PRINT("Failed to set cpuset for master thread.\n");
                }
            }
        } else {
            ERROR_PRINT("Failed to get cpuset for master thread.\n");
        }
    }

    topology_init();
    numa_init();
    affinity_init();
    _libperfctr_thread_map = libperfctr_thread_map_new(_libperfctr_num_hwthreads);

#ifndef LIKWID_USE_PERFEVENT
    HPMmode(atoi(modeStr));
    set_perf_event_pid();
#endif

    if (pinStr != NULL) {
        likwid_pinThread(_libperfctr_thread_2_hwthread[0]);
        if (get_omp_threads() > _libperfctr_num_hwthreads) {
            _libperfctr_use_locks = 1;
        }
        if (get_cilk_workers() > _libperfctr_num_hwthreads) {
            _libperfctr_use_locks = 1;
        }
    }

    ret = perfmon_init(_libperfctr_num_hwthreads, _libperfctr_thread_2_hwthread);
    if (ret < 0) {
        ERROR_PRINT("Failed to initialize LIKWID perfmon library.\n");
        return;
    }

    for (int i = 0; i < eventStrings->qty; i++) {
        int grp = perfmon_addEventSet(bdata(eventStrings->entry[i]));
        if (grp >= 0) {
            _libperfctr_groups[_libperfctr_num_groups++] = grp;
        }
    }
    bstrListDestroy(eventStrings);
    bdestroy(bEventStr);

    for (int i = 0; i < _libperfctr_num_hwthreads; i++) {
        for(int j=0; j<groupSet->groups[_libperfctr_groups[0]].numberOfEvents;j++) {
            groupSet->groups[_libperfctr_groups[0]].events[j].threadCounter[i].init = TRUE;
            groupSet->groups[_libperfctr_groups[0]].state = STATE_START;
        }
    }

/*    LikwidMarkerThread* thread = NULL;*/
/*    libperfctr_get_thread(gettid(), &thread);*/

    groupSet->activeGroup = 0;

    perfmon_setupCounters(groupSet->activeGroup);
    perfmon_startCounters();

    if (setinit) {
        DEBUG_PRINT(DEBUGLEV_DETAIL, "MarkerAPI initialized\n");
        _libperfctr_init = 1;
    }

}

void likwid_markerThreadInit(void) {
    if (!_libperfctr_init) {
        return;
    }
    int ret = 0;
    int tid = gettid();
    int hwthread_id = sched_getcpu();
    LikwidMarkerThread* thread = NULL;
    ret = libperfctr_get_thread(tid, &thread);
    if (ret < 0) {
        fprintf(stderr, "ERROR: Failed to get thread data for thread %d\n", tid);
        return;
    }
    thread->hwthread_id = hwthread_id;

    if (getenv("LIKWID_PIN") != NULL) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        sched_getaffinity(tid, sizeof(cpu_set_t), &cpuset);
        if (CPU_COUNT(&cpuset) > 1) {
            int hwt = _libperfctr_thread_2_hwthread[thread->thread_id % _libperfctr_num_hwthreads];
            likwid_pinThread(hwt);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, "Pin thread %d to HW thread %d", tid, hwt);
            thread->hwthread_id = hwt;
        }
    }
}

void likwid_markerNextGroup(void) {
    if (!_libperfctr_init) {
        return;
    }
    int next_group = (groupSet->activeGroup + 1) % _libperfctr_num_groups;
    if (next_group != groupSet->activeGroup)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "Switch from group %d to group %d", groupSet->activeGroup, next_group);
        perfmon_switchActiveGroup(next_group);
    }
    return;
}

int likwid_markerRegisterRegion(const char *regionTag) {
    int ret = 0;
    LikwidMarkerThread *thread = NULL;
    LikwidThreadResults* results = NULL;
    if (!_libperfctr_init) {
        ERROR_PRINT("MarkerAPI not initialized\n");
        return -EFAULT;
    }
    uint64_t tid = gettid();
    int hwthread_id = sched_getcpu();

    ret = libperfctr_get_thread(tid, &thread);
    if (ret < 0) {
        fprintf(stderr, "ERROR: Failed to get thread data for thread %ld\n", tid);
        return ret;
    }
    thread->hwthread_id = hwthread_id;
    ret = libperfctr_add_region(thread, regionTag, &results);
    if (ret < 0) {
        fprintf(stderr, "ERROR: Failed to get region data for region %s and thread %ld\n", regionTag, tid);
        return ret;
    }

#ifndef LIKWID_USE_PERFEVENT
    // Add CPU to access layer if ACCESSMODE is direct or accessdaemon
    ret =  HPMaddThread(thread->hwthread_id);
    // Perform one access to fully initialize connection to access daemon
    uint64_t tmp = 0x0ULL;
    uint32_t reg = counter_map[groupSet->groups[_libperfctr_groups[0]].events[0].index].counterRegister;
    HPMread(thread->hwthread_id, MSR_DEV, reg, &tmp);
#endif

    return 0;
}

int likwid_markerStartRegion(const char *regionTag) {
    int ret = 0;
    LikwidMarkerThread *thread = NULL;
    LikwidThreadResults* results = NULL;
    if (!_libperfctr_init) {
        ERROR_PRINT("MarkerAPI not initialized\n");
        return -EFAULT;
    }
    uint64_t tid = gettid();
    int hwthread_id = sched_getcpu();

    DEBUG_PRINT(DEBUGLEV_DEVELOP, "Get thread %ld on HW thread %d", tid, hwthread_id);
    ret = libperfctr_get_thread(tid, &thread);
    if (ret < 0) {
        ERROR_PRINT("ERROR: Failed to get thread data for thread %ld", tid);
        return ret;
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, "Get region %s for thread %ld (thread %p)", regionTag, tid, thread);
    ret = libperfctr_add_region(thread, regionTag, &results);
    if (ret < 0) {
        ERROR_PRINT("ERROR: Failed to get region data for region %s and thread %ld", regionTag, tid);
        return ret;
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, "Checking region %s for thread %ld (results %p)", regionTag, tid, results);
    if (results->state == MARKER_STATE_START) {
        fprintf(stderr, "WARN: Region %s was already started\n", regionTag);
    }
    if (thread->hwthread_id != hwthread_id) {
        fprintf(stderr, "WARN: Region %s was registered running on HW thread %d but moved to %d\n", regionTag, thread->hwthread_id, hwthread_id);
    }
    thread->hwthread_id = hwthread_id;
    DEBUG_PRINT(DEBUGLEV_DEVELOP, "Read counters on HW thread %d", thread->hwthread_id);
    perfmon_readCountersCpu(thread->hwthread_id);
    results->cpuID = thread->hwthread_id;

    // Read data
    DEBUG_PRINT(DEBUGLEV_DEVELOP, "Reading data from group %d", groupSet->activeGroup);
    PerfmonEventSet *data = &groupSet->groups[groupSet->activeGroup];
    for(int i = 0; i < data->numberOfEvents;i++)
    {
        if (data->events[i].type != NOTYPE)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, "START [%s] READ EVENT [%d=%d] EVENT %d VALUE %llu",
                    regionTag, thread->thread_id, thread->hwthread_id, i,
                    LLU_CAST data->events[i].threadCounter[thread->thread_id].counterData);
            //groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].startData =
            //        groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].counterData;

            results->StartPMcounters[i] = data->events[i].threadCounter[thread->thread_id].counterData;
            results->StartOverflows[i] = data->events[i].threadCounter[thread->thread_id].overflows;
        }
        else
        {
            results->StartPMcounters[i] = NAN;
            results->StartOverflows[i] = -1;
        }
    }

    results->state = MARKER_STATE_START;
    timer_start(&(results->startTime));
    return 0;
}

int likwid_markerStopRegion(const char *regionTag) {
    int ret = 0;
    LikwidMarkerThread *thread = NULL;
    LikwidThreadResults* results = NULL;
    if (!_libperfctr_init) {
        ERROR_PRINT("MarkerAPI not initialized\n");
        return -EFAULT;
    }
    TimerData timestamp;
    timer_stop(&timestamp);
    uint64_t tid = gettid();
    int hwthread_id = sched_getcpu();

    ret = libperfctr_get_thread(tid, &thread);
    if (ret < 0) {
        fprintf(stderr, "ERROR: Failed to get thread data for thread %ld\n", tid);
        return ret;
    }
    ret = libperfctr_get_region(thread, regionTag, &results);
    if (ret == -EEXIST) {
        fprintf(stderr, "ERROR: Region %s does not exist for thread %ld\n", regionTag, tid);
        return ret;
    }

    if (results->state != MARKER_STATE_START) {
        fprintf(stderr, "WARN: Stopping an unknown/not-started region %s\n", regionTag);
        return -EFAULT;
    }
    if (thread->hwthread_id != hwthread_id) {
        fprintf(stderr, "WARN: Region %s was registered running on HW thread %d but moved to %d\n", regionTag, thread->hwthread_id, hwthread_id);
    }

    results->state = MARKER_STATE_STOP;
    results->startTime.stop.int64 = timestamp.stop.int64;
    results->time += timer_print(&(results->startTime));
    results->count++;

    perfmon_readCountersCpu(thread->hwthread_id);

    // Read data
    PerfmonEventSet *data = &groupSet->groups[groupSet->activeGroup];
    for(int i = 0; i < data->numberOfEvents;i++)
    {
        if (data->events[i].type != NOTYPE)
        {
            double result = calculateMarkerResult(data->events[i].index, results->StartPMcounters[i],
                                            data->events[i].threadCounter[thread->thread_id].counterData,
                                            data->events[i].threadCounter[thread->thread_id].overflows -
                                            results->StartOverflows[i]);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, "STOP [%s] READ EVENT [%d=%d] EVENT %d VALUE %llu DIFF %f",
                            regionTag, thread->thread_id, thread->hwthread_id, i,
                            LLU_CAST data->events[i].threadCounter[thread->thread_id].counterData, result);
            if ((counter_map[data->events[i].index].type != THERMAL) &&
                (counter_map[data->events[i].index].type != VOLTAGE) &&
                (counter_map[data->events[i].index].type != MBOX0TMP))
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

    return 0;
}

int likwid_markerResetRegion(const char *regionTag) {
    int ret = 0;
    LikwidMarkerThread *thread = NULL;
    LikwidThreadResults* results = NULL;
    if (!_libperfctr_init) {
        ERROR_PRINT("MarkerAPI not initialized\n");
        return -EFAULT;
    }
    uint64_t tid = gettid();

    ret = libperfctr_get_thread(tid, &thread);
    if (ret < 0) {
        fprintf(stderr, "ERROR: Failed to get thread data for thread %ld\n", tid);
        return ret;
    }
    ret = libperfctr_get_region(thread, regionTag, &results);
    if (ret == -EEXIST) {
        fprintf(stderr, "ERROR: Region %s does not exist for thread %ld\n", regionTag, tid);
        return ret;
    }

    if (results->state != MARKER_STATE_STOP) {
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

void likwid_markerGetRegion(const char *regionTag, int *nr_events,
                                   double *events, double *time, int *count) {
    int ret = 0;
    LikwidMarkerThread *thread = NULL;
    LikwidThreadResults* results = NULL;
    if (!_libperfctr_init) {
        ERROR_PRINT("MarkerAPI not initialized\n");
        return;
    }
    uint64_t tid = gettid();

    ret = libperfctr_get_thread(tid, &thread);
    if (ret < 0) {
        fprintf(stderr, "ERROR: Failed to get thread data for thread %ld\n", tid);
        return;
    }
    ret = libperfctr_get_region(thread, regionTag, &results);
    if (ret == -EEXIST) {
        fprintf(stderr, "ERROR: Region %s does not exist for thread %ld\n", regionTag, tid);
        return;
    }
    if (count != NULL) {
        *count = results->count;
    }
    if (time != NULL) {
        *time = results->time;
    }
    if (nr_events != NULL && events != NULL && *nr_events > 0) {
        int length = MIN(groupSet->groups[groupSet->activeGroup].numberOfEvents, *nr_events);
        for(int i = 0; i < length; i++) {
            events[i] = results->PMcounters[i];
        }
        *nr_events = length;
    }
    return;
}



int
likwid_markerWriteFile(const char* markerfile)
{
    if (markerfile == NULL) {
        fprintf(stderr, "File can not be NULL.\n");
        return -EFAULT;
    }

    FILE *file = NULL;
    int* validRegions = NULL;

    int numberOfThreads = perfmon_getNumberOfThreads();
    int numberOfRegions = perfmon_getNumberOfRegions();
    int numberOfGroups = perfmon_getNumberOfGroups();

    if (!_libperfctr_init)
    {
        return -EFAULT;
    }
    if ((numberOfThreads == 0)||(numberOfRegions == 0))
    {
        fprintf(stderr, "No threads or regions defined in hash table\n");
        return -EFAULT;
    }

    file = fopen(markerfile,"w");
    if (!file)
    {
        fprintf(stderr, "Unable open marker file for writing: %s\n", markerfile);
        perror("fopen");
        return -errno;
    }

    int newNumberOfRegions = 0;
    int newRegionID = 0;
    validRegions = (int*)malloc(numberOfRegions*sizeof(int));
    if (!validRegions)
    {
        return -EFAULT;
    }
    for (int i=0; i<numberOfRegions; i++)
    {
        validRegions[i] = 0;
    }
    for (int i=0; i<numberOfRegions; i++)
    {
        for (int j=0; j<numberOfThreads; j++)
        {
            validRegions[i] += perfmon_getCountOfRegion(i, j);
        }
        if (validRegions[i] > 0)
            newNumberOfRegions++;
        else
            fprintf(stderr, "WARN: Skipping region %s for evaluation.\n", perfmon_getTagOfRegion(i));
    }
    if (newNumberOfRegions < numberOfRegions)
    {
        fprintf(stderr, "WARN: Regions are skipped because:\n");
        fprintf(stderr, "      - The region was only registered\n");
        fprintf(stderr, "      - The region was started but never stopped\n");
        fprintf(stderr, "      - The region was never started but stopped\n");
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP,
            "Creating Marker file %s with %d regions %d groups and %d threads",
            markerfile, newNumberOfRegions, numberOfGroups, numberOfThreads);
    bstring thread_regs_grps = bformat("%d %d %d", numberOfThreads, newNumberOfRegions, numberOfGroups);
    fprintf(file,"%s\n", bdata(thread_regs_grps));
    DEBUG_PRINT(DEBUGLEV_DEVELOP, "%s", bdata(thread_regs_grps));
    bdestroy(thread_regs_grps);

    for (int i=0; i<numberOfRegions; i++)
    {
        if (validRegions[i] == 0)
            continue;
        bstring tmp = bformat("%d:%s", newRegionID, perfmon_getTagOfRegion(i));
        fprintf(file,"%s\n", bdata(tmp));
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "%s", bdata(tmp));
        bdestroy(tmp);
        newRegionID++;
    }
    int *cpulist = (int*) malloc(numberOfThreads * sizeof(int));
    if (cpulist == NULL)
    {
        fprintf(stderr, "Failed to allocate %lu bytes for the cpulist storage\n",
                    numberOfThreads * sizeof(int));
        free(validRegions);
        return -EFAULT;
    }
    newRegionID = 0;
    for (int i=0; i<numberOfRegions; i++)
    {
        if (validRegions[i] == 0)
            continue;
        int groupID = perfmon_getGroupOfRegion(i);
        int nevents = groupSet->groups[groupID].numberOfEvents;
        perfmon_getCpulistOfRegion(i, numberOfThreads, cpulist);
        for (int j=0; j<numberOfThreads; j++)
        {
            int count = perfmon_getCountOfRegion(i, j);
            double time = perfmon_getTimeOfRegion(i, j);
            bstring l = bformat("%d %d %d %u %e %d ", newRegionID,
                                                      groupID,
                                                      cpulist[j],
                                                      count,
                                                      time,
                                                      nevents);

            for (int k=0; k < MIN(nevents, NUM_PMC); k++)
            {
                bstring tmp = bformat("%e ", perfmon_getResultOfRegionThread(i, k, j));
                bconcat(l, tmp);
                bdestroy(tmp);
            }
            fprintf(file,"%s\n", bdata(l));
            DEBUG_PRINT(DEBUGLEV_DEVELOP, "%s", bdata(l));
            bdestroy(l);
        }
        newRegionID++;
    }
    fclose(file);
    free(validRegions);
    free(cpulist);
    return 0;
}

void likwid_markerClose(void) {
    if (!_libperfctr_init) {
        ERROR_PRINT("MarkerAPI not initialized\n");
        return;
    }

    char* markerFilePath = getenv("LIKWID_FILEPATH");
    if (markerFilePath == NULL) {
        fprintf(stderr, "Is the application executed with LIKWID wrapper? No file path for the MarkerAPI output defined.\n");
        return;
    }

    int numberOfThreads = 0;
    int numberOfRegions = 0;
    libperfctr_thread_map_Iter iter = libperfctr_thread_map_iter(&_libperfctr_thread_map);
    for (libperfctr_thread_map_Entry* p = libperfctr_thread_map_Iter_get(&iter); p != NULL; p = libperfctr_thread_map_Iter_next(&iter)) {
        LikwidMarkerThread* thread = (LikwidMarkerThread*)p->val;
        int regions = libperfctr_result_map_size(&thread->regions);
        numberOfRegions = (regions > numberOfRegions ? regions : numberOfRegions);
        numberOfThreads++;
    }

    int num_res = 0;
    LikwidResults *res = lw_malloc(numberOfRegions * sizeof(LikwidResults));
    for (int i = 0; i < numberOfRegions; i++) {
        res[i].time = lw_malloc(numberOfThreads * sizeof(double));
        memset(res[i].time, 0, numberOfThreads * sizeof(double));
        res[i].count = lw_malloc(numberOfThreads * sizeof(uint32_t));
        memset(res[i].count, 0, numberOfThreads * sizeof(uint32_t));
        res[i].cpulist = lw_malloc(numberOfThreads * sizeof(int));
        memset(res[i].cpulist, 0, numberOfThreads * sizeof(int));
        res[i].counters = lw_malloc(numberOfThreads * sizeof(double*));
        for (int j = 0; j < numberOfThreads; j++) {
            res[i].counters[j] = lw_malloc(NUM_PMC * sizeof(double));
            memset(res[i].counters[j], 0, NUM_PMC * sizeof(double));
        }
    }

    libperfctr_thread_map_Iter titer = libperfctr_thread_map_iter(&_libperfctr_thread_map);
    for (libperfctr_thread_map_Entry* te = libperfctr_thread_map_Iter_get(&titer); te != NULL; te = libperfctr_thread_map_Iter_next(&titer)) {
        LikwidMarkerThread* thread = (LikwidMarkerThread*)te->val;
        libperfctr_result_map_Iter riter = libperfctr_result_map_iter(&thread->regions);
        for (libperfctr_result_map_Entry *re = libperfctr_result_map_Iter_get(&riter); re != NULL; re = libperfctr_result_map_Iter_next(&riter)) {
            LikwidThreadResults* tdata = (LikwidThreadResults*)re->val;
            int ridx = -1;
            for (int k = 0; k < num_res; k++) {
                LikwidResults * rres = &res[k];
                bstring teststr = bformat("%s-%d", bdata(tdata->label), tdata->groupID);
                if (bstrcmp(rres->tag, teststr) == BSTR_OK) {
                    ridx = k;
                }
                bdestroy(teststr);
            }
            if (ridx < 0) {
                // new region result
                LikwidResults * rres = &res[num_res];
                rres->tag = bformat("%s-%d", bdata(tdata->label), tdata->groupID);
                rres->groupID = tdata->groupID;
                rres->eventCount = groupSet->groups[tdata->groupID].numberOfEvents;
                ridx = num_res;
                num_res++;
            }
            if (ridx >= 0 && ridx < numberOfRegions) {
                LikwidResults * rres = &res[ridx];
                rres->time[thread->thread_id] = tdata->time;
                rres->count[thread->thread_id] = tdata->count;
                rres->cpulist[thread->thread_id] = tdata->cpuID;
                rres->threadCount++;
                memcpy(rres->counters[thread->thread_id], tdata->PMcounters, NUM_PMC * sizeof(double));
            }
        }
    }

    markerResults = res;
    markerRegions = numberOfRegions;

    likwid_markerWriteFile(markerFilePath);

    libperfctr_thread_map_destroy(&_libperfctr_thread_map);

    numa_finalize();
    affinity_finalize();

    return;
}
