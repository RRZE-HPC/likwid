#include <likwid.h>

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <error.h>
#include <rocmon_types.h>
#include <util.h>
#include <bstrlib.h>
#include <perfgroup.h>

// TODO replace hip names with rocm names (if applicable)
// TODO insert missing checks for gettid == main_tid
// TODO change function order to make more sense

// glibc only added a real gettid() function in version 2.30 (2019),
// so manually call the syscall.
#define gettid() syscall(SYS_gettid)
#define DUMMY_TID -1

// The ENOSPC we determine from fprintf is just a guess.
// I'm not sure if there is a way to do this correctly.
#define chk_fprintf(filePtr, formatStr, ...) \
    do { \
        fprintf(filePtr, formatStr, ##__VA_ARGS__); \
        if (ferror(filePtr)) { \
            err = -ENOSPC; \
            goto cleanup; \
        } \
    } while (0)

// TODO remove this once we use cwisstables and no longer need to create
// a string combining both group ID and region ID
#define LABEL_MAX_SIZE 256

static pthread_mutex_t rocmarker_init_mutex = PTHREAD_MUTEX_INITIALIZER;
static RocmarkerContext *rocmarker_ctx;

static int get_gpu_idx(int hipDeviceId, size_t *idx) {
    for (size_t i = 0; i < rocmarker_ctx->numHipDeviceIds; i++) {
        if (rocmarker_ctx->hipDeviceIds[i] == hipDeviceId) {
            *idx = i;
            return 0;
        }
    }

    return -EINVAL;
}

static int get_group_idx(int groupId, size_t *idx) {
    for (size_t i = 0; i < rocmarker_ctx->numGroups; i++) {
        if (rocmarker_ctx->groups[i].groupId == groupId) {
            *idx = i;
            return 0;
        }
    }

    return -EINVAL;
}

static void label_fmt(char *buf, size_t size, const char *regionTag, int groupId) {
    snprintf(buf, size, "%s-%d", regionTag, groupId);
}

static void rocmarker_group_free(RocmarkerGroup *group) {
    if (!group)
        return;

    if (group->eventNames) {
        for (size_t i = 0; i < group->numEventNames; i++)
            free(group->eventNames[i]);
        free(group->eventNames);
    }

    if (group->metrics) {
        for (size_t i = 0; i < group->numMetrics; i++) {
            free(group->metrics[i].name);
            free(group->metrics[i].formula);
        }
        free(group->metrics);
    }

    free(group);
}

static void rocmarker_ctx_free(void) {
    if (!rocmarker_ctx)
        return;

    free(rocmarker_ctx->hipDeviceIds);
    destroy_smap(rocmarker_ctx->regions);

    if (rocmarker_ctx->groups) {
        for (size_t i = 0; i < rocmarker_ctx->numGroups; i++)
            rocmarker_group_free(&rocmarker_ctx->groups[i]);
        free(rocmarker_ctx->groups);
    }

    free(rocmarker_ctx);
    rocmarker_ctx = NULL;
}

static int gpulist_from_str(const char *gpustring, size_t *numGpus, int **gpus) {
    bstring bgpustring = bfromcstr(gpustring);
    if (!bgpustring)
        return -ENOMEM;

    int err = 0;
    struct bstrList *gpustrings = bsplit(bgpustring, ',');
    if (!gpustrings) {
        err = -ENOMEM;
        goto cleanup;
    }

    int *newGpus = calloc(gpustrings->qty, sizeof(*newGpus));
    if (!newGpus) {
        err = -errno;
        goto cleanup;
    }

    *numGpus = gpustrings->qty;
    *gpus = newGpus;

    for (int i = 0; i < gpustrings->qty; i++)
        newGpus[i] = atoi(bdata(gpustrings->entry[i]));

cleanup:
    if (err < 0)
        free(newGpus);

    bstrListDestroy(gpustrings);
    bdestroy(bgpustring);
    return err;
}

static int eventsets_init(const char *eventStr) {
    bstring eventStrCopy = bfromcstr(eventStr);
    if (!eventStrCopy)
        return -ENOMEM;

    int err = 0;
    struct bstrList *eventsForGroups = bsplit(eventStrCopy, '|');
    if (!eventsForGroups) {
        err = -ENOMEM;
        goto cleanup;
    }

    rocmarker_ctx->groups = calloc(eventsForGroups->qty, sizeof(*rocmarker_ctx->groups));
    if (rocmarker_ctx->groups) {
        err = -errno;
        goto cleanup;
    }

    rocmarker_ctx->numGroups = eventsForGroups->qty;
    rocmarker_ctx->activeGroupIdx = 0;

    for (int i = 0; i < eventsForGroups->qty; i++) {
        RocmarkerGroup *group = &rocmarker_ctx->groups[i];

        const char *eventStr = bdata(eventsForGroups->entry[i]);
        err = rocmon_addEventSet(eventStr);
        if (err < 0)
            goto cleanup;

        group->groupId = err;
        group->numEventNames = rocmon_getNumberOfEvents(err);
        group->eventNames = calloc(group->numEventNames, sizeof(*group->eventNames));
        if (!group->eventNames) {
            err = -errno;
            goto cleanup;
        }

        for (size_t i = 0; i < group->numEventNames; i++) {
            group->eventNames[i] = strdup(eventStr);
            if (!group->eventNames[i]) {
                err = -errno;
                goto cleanup;
            }
        }

        err = rocmon_getNumberOfMetrics(group->groupId);
        if (err < 0)
            goto cleanup;

        group->numMetrics = (size_t)err;
        group->metrics = calloc(group->numMetrics, sizeof(*group->metrics));
        if (!group->metrics) {
            err = -errno;
            goto cleanup;
        }

        for (size_t j = 0; j < group->numMetrics; j++) {
            const char *metricName;
            err = rocmon_getMetricName(group->groupId, (int)j, &metricName);
            if (err < 0)
                goto cleanup;

            group->metrics[j].name = strdup(metricName);
            if (!group->metrics[j].name) {
                err = -errno;
                goto cleanup;
            }

            const char *metricFormula;
            err = rocmon_getMetricFormula(group->groupId, (int)j, &metricFormula);
            if (err < 0)
                goto cleanup;

            group->metrics[j].formula = strdup(metricFormula);
            if (group->metrics[j].formula) {
                err = -errno;
                goto cleanup;
            }
        }
    }

cleanup:
    bstrListDestroy(eventsForGroups);
    bdestroy(eventStrCopy);
    return err;
}

static void region_free(RocmarkerRegion *region);

static void region_free_vptr(void *region) {
    region_free(region);
}

int rocmon_markerInit(void) {
    const char *eventStr = getenv("LIKWID_ROCMON_EVENTS");
    const char *gpuStr = getenv("LIKWID_ROCMON_GPUS");
    const char *gpuFileStr = getenv("LIKWID_ROCMON_FILEPATH");
    const char *verbosityStr = getenv("LIKWID_ROCMON_VERBOSITY");
    const char *debugStr = getenv("LIKWID_DEBUG");

    if (!eventStr || !gpuFileStr) {
        fprintf(stderr, "Running without GPU Marker API. Activate GPU Marker API with -m, -G and -W on commandline.\n");
        return -EINVAL;
    }

    pthread_mutex_lock(&rocmarker_init_mutex);

    int err = 0;

    if (rocmarker_ctx) {
        err = -EEXIST;
        goto unlock_err;
    }

    rocmarker_ctx = calloc(1, sizeof(&rocmarker_ctx));
    if (!rocmarker_ctx) {
        err = -errno;
        goto unlock_err;
    }

    rocmarker_ctx->main_tid = gettid();

    if (verbosityStr)
        rocmon_setVerbosity(atoi(verbosityStr));

    if (debugStr)
        perfmon_setVerbosity(atoi(debugStr));

    int *gpuIds = NULL;
    size_t numGpuIds;
    err = gpulist_from_str(gpuStr, &numGpuIds, &gpuIds);
    if (err < 0)
        goto unlock_err;

    err = rocmon_init(numGpuIds, gpuIds);
    free(gpuIds);
    if (err < 0)
        goto unlock_err;

    // If the user inputs 0, NULL for the GPU list, explicitly query
    // the number of GPUs that were autodetected.
    rocmarker_ctx->numHipDeviceIds = (size_t)rocmon_getNumberOfGPUs();
    rocmarker_ctx->hipDeviceIds = calloc(rocmarker_ctx->numHipDeviceIds, sizeof(*rocmarker_ctx->hipDeviceIds));
    if (!rocmarker_ctx->hipDeviceIds) {
        err = -errno;
        goto unlock_err;
    }

    for (size_t i = 0; i < rocmarker_ctx->numHipDeviceIds; i++)
        rocmarker_ctx->hipDeviceIds[i] = rocmon_getIdOfGPU((int)i);

    err = eventsets_init(eventStr);
    if (err < 0)
        goto unlock_err;

    // Setup initial event set (usually 0)
    err = rocmon_setupCounters(rocmarker_ctx->groups[rocmarker_ctx->activeGroupIdx].groupId);
    if (err < 0)
        goto unlock_err;

    err = rocmon_startCounters();
    if (err < 0)
        goto unlock_err;

    err = init_map(&rocmarker_ctx->regions, MAP_KEY_TYPE_STR, 0, region_free_vptr);
    if (err < 0)
        goto unlock_err;

    pthread_mutex_unlock(&rocmarker_init_mutex);
    return 0;

unlock_err:
    rocmarker_ctx_free();
    pthread_mutex_unlock(&rocmarker_init_mutex);
    return err;
}

void rocmon_markerClose(void) {
    pthread_mutex_lock(&rocmarker_init_mutex);

    // init must occur before finalize
    assert(rocmarker_ctx);
    assert(gettid() == rocmarker_ctx->main_tid);

    rocmon_stopCounters();

    // TODO: Why is a result file mandatory?
    const char *resultFile = getenv("LIKWID_ROCMON_FILEPATH");
    if (!resultFile) {
        ERROR_PRINT("Is the application executed with LIKWID wrapper? "
                "No file path for the Rocmon Marker API output defined.\n");
    } else {
        rocmon_markerWriteFile(resultFile);
    }

    rocmarker_ctx_free();
    rocmon_finalize();

    pthread_mutex_unlock(&rocmarker_init_mutex);
}

static int gpu_results_init(RocmarkerGpuResultList *resultList, int groupId) {
    const int numCounters = rocmon_getNumberOfEvents(groupId);
    assert(numCounters >= 0);

    resultList->counterValues = calloc((size_t)numCounters, sizeof(*resultList->counterValues));
    if (!resultList->counterValues)
        return -errno;

    resultList->numCounterValues = (size_t)numCounters;
    return 0;
}

static void gpu_results_fini(RocmarkerGpuResultList *resultList) {
    if (!resultList)
        return;

    free(resultList->counterValues);
    memset(resultList, 0, sizeof(*resultList));
}

static int region_init(RocmarkerRegion *region, const char *regionTag, int groupId) {
    region->tag = strdup(regionTag);
        return -errno;

    region->groupId = groupId;
    region->started = false;
    region->execCount = 0;
    region->lastStartTime = 0;
    region->lastStopTime = 0;
    region->totalTime = 0;

    region->gpuResults = calloc(rocmarker_ctx->numHipDeviceIds, sizeof(*region->gpuResults));
    if (!region->gpuResults)
        return -errno;

    for (size_t i = 0; i < rocmarker_ctx->numHipDeviceIds; i++) {
        int err = gpu_results_init(&region->gpuResults[i], groupId);
        if (err < 0)
            return err;
    }

    return 0;
}

static void region_fini(RocmarkerRegion *region) {
    if (!region)
        return;

    for (size_t i = 0; i < rocmarker_ctx->numHipDeviceIds; i++)
        gpu_results_fini(&region->gpuResults[i]);

    free(region->gpuResults);
    free(region->tag);
    memset(region, 0, sizeof(*region));
}

static void region_free(RocmarkerRegion *region) {
    region_fini(region);
    free(region);
}

static bool valid_region_tag(const char *regionTag) {
    while (*regionTag) {
        if (!isalnum(*regionTag) && *regionTag != '_')
            return false;
        regionTag++;
    }
    return true;
}

int rocmon_markerRegisterRegion(const char *regionTag) {
    if (!rocmarker_ctx)
        return -EFAULT;

    // check if regionTag contains illegal characters
    if (!valid_region_tag(regionTag))
        return -EINVAL;

    assert(gettid() == rocmarker_ctx->main_tid);

    const int activeGroup = rocmarker_ctx->groups[rocmarker_ctx->activeGroupIdx].groupId;

    char regionLabel[LABEL_MAX_SIZE];
    label_fmt(regionLabel, sizeof(regionLabel), regionTag, activeGroup);

    // Does the marker exist already?
    RocmarkerRegion *region = NULL;
    int err = get_smap_by_key(rocmarker_ctx->regions, regionLabel, (void **)&region);
    if (err == 0) {
        return -EEXIST;
    } else if (err != -ENOENT && err < 0) {
        return err;
    }

    // Marker doens't yet exist, so create a new one
    region = calloc(1, sizeof(*region));
    if (!region)
        return -errno;

    // Insert it
    err = add_smap(rocmarker_ctx->regions, regionLabel, region);
    if (err < 0) {
        free(region);
        return err;
    }

    err = region_init(region, regionTag, activeGroup);
    if (err < 0)
        goto cleanup;

    return 0;

cleanup:
    region_fini(region);
    free(region);

    return err;
}

int rocmon_markerStartRegion(const char *regionTag) {
    if (!rocmarker_ctx)
        return -EFAULT;

    assert(gettid() == rocmarker_ctx->main_tid);

    const int activeGroup = rocmarker_ctx->groups[rocmarker_ctx->activeGroupIdx].groupId;

    char regionLabel[LABEL_MAX_SIZE];
    label_fmt(regionLabel, sizeof(regionLabel), regionTag, activeGroup);
    
    // Check if regionTag already exists. If not create it first.
    RocmarkerRegion *region;
    int err = get_smap_by_key(rocmarker_ctx->regions, regionLabel, (void **)&region);
    if (err == -ENOENT) {
        err = rocmon_markerRegisterRegion(regionTag);
        if (err < 0)
            return err;
    } else if (err < 0) {
        return err;
    }

    if (region->started) {
        ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Cannot start region '%s', which is already started.", regionTag);
        return -EBUSY;
    }

    if (activeGroup != region->groupId)
        return -EINVAL;

    err = rocmon_readCounters();
    if (err < 0)
        return err;

    err = rocmon_getTimestampOfLastReadOfGroup(activeGroup, &region->lastStartTime);
    if (err < 0)
        return err;

    for (size_t i = 0; i < rocmarker_ctx->numHipDeviceIds; i++) {
        RocmarkerGpuResultList *resultList = &region->gpuResults[i];
        const int hipDeviceId = rocmarker_ctx->hipDeviceIds[i];

        for (size_t j = 0; j < resultList->numCounterValues; j++)
            resultList->counterValues[j].lastValue = rocmon_getLastResult(hipDeviceId, activeGroup, j);
    }

    region->started = true;
    return 0;
}

int rocmon_markerStopRegion(const char *regionTag) {
    if (!rocmarker_ctx)
        return -EFAULT;

    assert(gettid() == rocmarker_ctx->main_tid);

    const int activeGroup = rocmarker_ctx->groups[rocmarker_ctx->activeGroupIdx].groupId;

    char regionLabel[LABEL_MAX_SIZE];
    label_fmt(regionLabel, sizeof(regionLabel), regionTag, activeGroup);

    RocmarkerRegion *region;
    int err = get_smap_by_key(rocmarker_ctx->regions, regionLabel, (void **)&region);
    if (err < 0)
        return err;

    if (!region->started) {
        ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Cannot stop region '%s', which is not started.", regionTag);
        return -EBUSY;
    }

    err = rocmon_readCounters();
    if (err < 0)
        return err;

    // No need to store stopTime in the RocmarkerRegion struct, since it is defacto
    // only a temporary to accumulate the total 
    err = rocmon_getTimestampOfLastReadOfGroup(activeGroup, &region->lastStopTime);
    if (err < 0)
        return err;

    region->totalTime += region->lastStopTime - region->lastStartTime;
    region->execCount++;

    for (size_t i = 0; i < rocmarker_ctx->numHipDeviceIds; i++) {
        RocmarkerGpuResultList *resultList = &region->gpuResults[i];
        const int hipDeviceId = rocmarker_ctx->hipDeviceIds[i];

        for (size_t j = 0; j < resultList->numCounterValues; j++) {
            // TODO The legacy rocmon differentiated the accumulation between RPR and RSMI events.
            // I don't see any reason why that would make a difference, but if we run into bad results,
            // we should check if that's the cause.
            const double stopValue = rocmon_getLastResult(hipDeviceId, activeGroup, j);
            const double startValue = resultList->counterValues[j].lastValue;
            resultList->counterValues[j].fullValue += stopValue - startValue;
        }
    }

    region->started = false;
    return 0;
}

int rocmon_markerResetRegion(const char *regionTag) {
    if (!rocmarker_ctx)
        return -EFAULT;

    assert(gettid() == rocmarker_ctx->main_tid);

    const int activeGroup = rocmarker_ctx->groups[rocmarker_ctx->activeGroupIdx].groupId;

    char regionLabel[LABEL_MAX_SIZE];
    label_fmt(regionLabel, sizeof(regionLabel), regionTag, activeGroup);

    RocmarkerRegion *region;
    int err = get_smap_by_key(rocmarker_ctx->regions, regionLabel, (void **)&region);
    if (err < 0)
        return err;

    region->totalTime = 0;
    region->execCount = 0;

    for (size_t i = 0; i < rocmarker_ctx->numHipDeviceIds; i++) {
        RocmarkerGpuResultList *resultList = &region->gpuResults[i];

        for (size_t j = 0; j < resultList->numCounterValues; j++) {
            resultList->counterValues[j].fullValue = 0.0;
            resultList->counterValues[j].lastValue = 0.0;
        }
    }

    return 0;
}

void rocmon_markerNextGroup(void) {
    if (!rocmarker_ctx || rocmarker_ctx->main_tid != gettid())
        return;

    if (rocmarker_ctx->numGroups == 0 || rocmarker_ctx->numGroups == 1)
        return;

    rocmarker_ctx->activeGroupIdx = (rocmarker_ctx->activeGroupIdx + 1) %
        rocmarker_ctx->numGroups;

    rocmon_switchActiveGroup(
            rocmarker_ctx->groups[rocmarker_ctx->activeGroupIdx].groupId
            );
}

int rocmon_markerGetGpuIds(int **gpuIds, size_t *numGpuIds) {
    if (!rocmarker_ctx)
        return -EFAULT;

    int *newGpuIds = calloc(rocmarker_ctx->numHipDeviceIds, sizeof(*newGpuIds));
    if (!newGpuIds)
        return -errno;

    for (size_t i = 0; i < rocmarker_ctx->numHipDeviceIds; i++)
        newGpuIds[i] = rocmarker_ctx->hipDeviceIds[i];

    *gpuIds = newGpuIds;
    *numGpuIds = rocmarker_ctx->numHipDeviceIds;
    return 0;
}

int rocmon_markerGetGroupIds(int **groupIds, size_t *numGroupIds) {
    if (!rocmarker_ctx)
        return -EFAULT;

    int *newGroupIds = calloc(rocmarker_ctx->numHipDeviceIds, sizeof(*newGroupIds));
    if (!newGroupIds)
        return -errno;

    for (size_t i = 0; i < rocmarker_ctx->numGroups; i++)
        newGroupIds[i] = rocmarker_ctx->groups[i].groupId;

    *groupIds = newGroupIds;
    *numGroupIds = rocmarker_ctx->numGroups;
    return 0;
}

int rocmon_markerGetGroupInfo(int groupId, char ***eventNames, size_t *numEventNames, char ***metricNames, char ***metricFormulas, size_t *numMetrics) {
    if (!rocmarker_ctx)
        return -EFAULT;

    size_t groupIdx;
    int err = get_group_idx(groupId, &groupIdx);
    if (err < 0)
        return err;

    const RocmarkerGroup *group = &rocmarker_ctx->groups[groupIdx];

    char **newMetricNames = NULL;
    char **newMetricFormulas = NULL;
    char **newEventNames = calloc(group->numEventNames, sizeof(*newEventNames));
    if (!newEventNames)
        return -errno;

    for (size_t i = 0; i < group->numEventNames; i++) {
        newEventNames[i] = strdup(group->eventNames[i]);
        if (!newEventNames[i]) {
            err = -errno;
            goto cleanup;
        }
    }

    newMetricNames = calloc(group->numMetrics, sizeof(*newMetricNames));
    if (!newMetricNames)
        return -errno;

    for (size_t i = 0; i < group->numMetrics; i++) {
        newMetricNames[i] = strdup(group->metrics[i].name);
        if (newMetricNames[i]) {
            err = -errno;
            goto cleanup;
        }
    }

    newMetricFormulas = calloc(group->numMetrics, sizeof(*newMetricFormulas));
    if (!newMetricFormulas) {
        err = -errno;
        goto cleanup;
    }

    for (size_t i = 0; i < group->numMetrics; i++) {
        newMetricFormulas[i] = strdup(group->metrics[i].formula);
        if (newMetricFormulas[i]) {
            err = -errno;
            goto cleanup;
        }
    }

    *metricNames = newMetricNames;
    *metricFormulas = newMetricFormulas;
    *numMetrics = group->numMetrics;
    *eventNames = newEventNames;
    *numEventNames = group->numEventNames;
    return 0;

cleanup:
    if (newEventNames) {
        for (size_t i = 0; i < group->numEventNames; i++)
            free(newEventNames[i]);
        free(newEventNames);
    }
    if (newMetricNames) {
        for (size_t i = 0; i < group->numMetrics; i++)
            free(newMetricNames[i]);
        free(newMetricNames);
    }
    if (newMetricFormulas) {
        for (size_t i = 0; i < group->numMetrics; i++)
            free(newMetricFormulas[i]);
        free(newMetricFormulas);
    }
    return err;
}

int rocmon_markerGetRegionCounters(const char *regionTag, int groupId, int gpuId, size_t *numCounters, double **counters, size_t *numMetrics, double **metrics) {
    if (!rocmarker_ctx)
        return -EFAULT;

    // Make label and lookup the region
    char regionLabel[LABEL_MAX_SIZE];
    label_fmt(regionLabel, sizeof(regionLabel), regionTag, groupId);

    RocmarkerRegion *region;
    int err = get_smap_by_key(rocmarker_ctx->regions, regionLabel, (void **)&region);
    if (err < 0)
        return err;

    // Lookup GPU ID and Group ID
    size_t gpuIdx;
    err = get_gpu_idx(gpuId, &gpuIdx);
    if (err < 0)
        return err;

    size_t groupIdx;
    err = get_group_idx(groupId, &groupIdx);
    if (err < 0)
        return err;

    // Create counter value list
    RocmarkerGpuResultList *result = &region->gpuResults[gpuIdx];

    double *newCounters = calloc(result->numCounterValues, sizeof(*newCounters));
    if (!newCounters)
        return -errno;

    for (size_t i = 0; i < result->numCounterValues; i++)
        newCounters[i] = result->counterValues[i].fullValue;

    // Create metric value list
    RocmarkerGroup *group = &rocmarker_ctx->groups[groupIdx];

    double *newMetrics = calloc(group->numMetrics, sizeof(*newMetrics));
    if (!newMetrics) {
        err = -errno;
        goto cleanup;
    }

    // clist doesn't return proper error codes, so we just ignore it for now
    CounterList clist;
    init_clist(&clist);

    assert(group->numEventNames == result->numCounterValues);

    for (size_t i = 0; i < result->numCounterValues; i++)
        add_to_clist(&clist, group->eventNames[i], result->counterValues[i].fullValue);

    add_to_clist(&clist, "time", (double)region->totalTime / 1e9);
    add_to_clist(&clist, "true", 1);
    add_to_clist(&clist, "false", 0);

    for (size_t i = 0; i < group->numMetrics; i++) {
        err = calc_metric(group->metrics[i].formula, &clist, &newMetrics[i]);
        if (err < 0) {
            ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Cannot calculate formula: %s", group->metrics[i].formula);
            goto cleanup;
        }
    }

    *counters = newCounters;
    *numCounters = result->numCounterValues;
    *metrics = newMetrics;
    *numMetrics = group->numMetrics;
    return 0;

cleanup:
    free(newCounters);
    free(newMetrics);
    return err;
}

int rocmon_markerGetRegionStats(const char *regionTag, int groupId,
        size_t *execCount, double *execTime) {
    if (!rocmarker_ctx)
        return -EFAULT;

    // Make label and lookup the region
    char regionLabel[LABEL_MAX_SIZE];
    label_fmt(regionLabel, sizeof(regionLabel), regionTag, groupId);

    RocmarkerRegion *region;
    int err = get_smap_by_key(rocmarker_ctx->regions, regionLabel, (void **)&region);
    if (err < 0)
        return err;

    // Return results
    *execTime = (double)region->totalTime / 1e9;
    *execCount = region->execCount;
    return 0;
}

int rocmon_markerGetRegionTags(char ***regionTags, int **regionGroupIds, size_t *numRegions) {
    if (!rocmarker_ctx)
        return -EFAULT;

    int newNumRegions = get_map_size(rocmarker_ctx->regions);
    if (newNumRegions < 0)
        return newNumRegions;

    char **newRegionTags = calloc((size_t)newNumRegions, sizeof(*newRegionTags));
    if (!newRegionTags)
        return -errno;

    int err = 0;
    int *newRegionGroupIds = calloc((size_t)newNumRegions, sizeof(*newRegionGroupIds));
    if (!newRegionGroupIds) {
        err = -errno;
        goto cleanup;
    }

    for (int i = 0; i < newNumRegions; i++) {
        RocmarkerRegion *region;
        err = get_smap_by_idx(rocmarker_ctx->regions, i, (void **)&region);
        if (err < 0)
            return err;

        newRegionTags[i] = strdup(region->tag);
        if (!newRegionTags[i]) {
            err = -errno;
            goto cleanup;
        }

        newRegionGroupIds[i] = region->groupId;
    }

    *regionTags = newRegionTags;
    *regionGroupIds = newRegionGroupIds;
    *numRegions = (size_t)newNumRegions;

    return 0;

cleanup:
    if (newRegionTags) {
        for (size_t i = 0; i < (size_t)newNumRegions; i++)
            free(newRegionTags[i]);
        free(newRegionTags);
    }
    free(newRegionGroupIds);
    return err;
}

int rocmon_markerWriteFile(const char *markerfile) {
    if (!rocmarker_ctx)
        return -EFAULT;

    FILE *fp = fopen(markerfile, "w");
    if (!fp)
        return -errno;

    int err = get_map_size(rocmarker_ctx->regions);
    if (err < 0)
        goto cleanup;

    const size_t numRegions = (size_t)err;

    /* File format:
     * numGpus numRegions numGroups
     * GPU hipDeviceId
     * ... ('numGpus' number of lines)
     * GROUP groupId numEvents eventA eventB eventC numMetrics metricNameA metricFormulaA metricNameB metricFormulaB
     * ... ('numGroups' number of GROUP lines)
     * REGION regionTag groupId execCount execTime ; 42.4 8.24 -1.0 ; 1337 0.0 0.12e5
     * ... ('numRegions' number of REGION lines)
     *     ('numGpus' groups of results, separated by ';')
     */

    /* Checking for errors with fprintf really doens't work, if we don't know the
     * non-truncated number of characters to be written. However, I'm lazy and I don't
     * want to allocate something, format it, to then check if it's all written. */

    chk_fprintf(fp, "ROCMON_MARKER_FILE %zu %zu %zu\n", rocmarker_ctx->numHipDeviceIds,
            numRegions,
            rocmarker_ctx->numGroups);

    // Write hip device IDs
    for (size_t i = 0; i < rocmarker_ctx->numHipDeviceIds; i++)
        chk_fprintf(fp, "GPU %d\n", rocmarker_ctx->hipDeviceIds[i]);

    // Write groups
    for (size_t i = 0; i < rocmarker_ctx->numGroups; i++) {
        RocmarkerGroup *group = &rocmarker_ctx->groups[i];
        chk_fprintf(fp, "GROUP %d %zu", group->groupId, group->numEventNames);
        for (size_t j = 0; j < group->numEventNames; j++)
            chk_fprintf(fp, " %s", group->eventNames[j]);
        chk_fprintf(fp, " %zu", group->numMetrics);
        for (size_t j = 0; j < group->numMetrics; j++) {
            // Do not allow escape character "'"
            if (strchr(group->metrics[j].name, '\'') || strchr(group->metrics[j].formula, '\'')) {
                err = -EINVAL;
                goto cleanup;
            }

            chk_fprintf(fp, " '%s' '%s'", group->metrics[j].name, group->metrics[j].formula);
        }
        chk_fprintf(fp, "\n");
    }

    // Write region info
    for (size_t ir = 0; ir < numRegions; ir++) {
        RocmarkerRegion *region;
        err = get_smap_by_idx(rocmarker_ctx->regions, ir, (void **)&region);
        if (err < 0)
            goto cleanup;

        chk_fprintf(fp, "REGION %s %d %zu %f", region->tag, region->groupId, region->execCount, (double)region->totalTime / 1e9);

        for (size_t ig = 0; ig < rocmarker_ctx->numHipDeviceIds; ig++) {
            RocmarkerGpuResultList *result = &region->gpuResults[ig];

            chk_fprintf(fp, " ;");

            for (size_t iv = 0; iv < result->numCounterValues; iv++)
                chk_fprintf(fp, " %f", result->counterValues[iv].fullValue);
        }

        chk_fprintf(fp, "\n");
    }

cleanup:
    fclose(fp);
    return err;
}

int rocmon_markerInitResultsFromFile(const char *markerfile) {
    pthread_mutex_lock(&rocmarker_init_mutex);

    int err = 0;
    if (rocmarker_ctx) {
        err = -EEXIST;
        goto unlock_err;
    }

    // What we do here is provide a very terrible API.
    // Load the marker results will modify the internal state of the marker
    // API, but the marker API will not be usable, because the rocmon is not initialized.
    // We try to prevent that by using a 'main_tid', which is always invalid to prevent
    // the user calling the normal functions.
    
    rocmarker_ctx = calloc(1, sizeof(&rocmarker_ctx));
    if (!rocmarker_ctx) {
        err = -errno;
        goto unlock_err;
    }

    rocmarker_ctx->main_tid = DUMMY_TID;

    FILE *fp = fopen(markerfile, "w");
    if (!fp) {
        err = -errno;
        goto unlock_err;
    }

    size_t numRegions;

    // read line: 'numGpus numRegions numGroups'
    if (fscanf(fp, "ROCMON_MARKER_FILE %zu %zu %zu", &rocmarker_ctx->numHipDeviceIds, &numRegions, &rocmarker_ctx->numGroups) != 3) {
        ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Cannot parse marker header");
        err = -EINVAL;
        goto unlock_err;
    }

    rocmarker_ctx->hipDeviceIds = calloc(rocmarker_ctx->numHipDeviceIds, sizeof(*rocmarker_ctx->hipDeviceIds));
    if (!rocmarker_ctx->hipDeviceIds) {
        err = -errno;
        goto unlock_err;
    }

    // read multiple lines: 'gpuIdx hipDeviceId'
    for (size_t i = 0; i < rocmarker_ctx->numHipDeviceIds; i++) {
        if (fscanf(fp, "GPU %d", &rocmarker_ctx->hipDeviceIds[i]) != 1) {
            ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Invalid GPU line");
            err = -EINVAL;
            goto unlock_err;
        }
    }

    err = init_map(&rocmarker_ctx->regions, MAP_KEY_TYPE_STR, 0, region_free_vptr);
    if (err < 0)
        goto unlock_err;

    rocmarker_ctx->groups = calloc(rocmarker_ctx->numGroups, sizeof(*rocmarker_ctx->groups));
    if (!rocmarker_ctx->groups) {
        err = -errno;
        goto unlock_err;
    }

    // Read multiple lines: 'GROUP groupId eventA eventB eventC'
    for (size_t i = 0; i < rocmarker_ctx->numGroups; i++) {
        RocmarkerGroup *group = &rocmarker_ctx->groups[i];

        if (fscanf(fp, "GROUP %d %zu", &group->groupId, &group->numEventNames) != 2) {
            ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Invalid GROUP line");
            err = -EINVAL;
            goto unlock_err;
        }

        group->eventNames = calloc(group->numEventNames, sizeof(*group->eventNames));
        if (!group->eventNames) {
            err = -errno;
            goto unlock_err;
        }

        for (size_t j = 0; j < group->numEventNames; j++) {
            if (fscanf(fp, "%ms", &group->eventNames[j]) != 1) {
                err = -EINVAL;
                goto unlock_err;
            }
        }

        if (fscanf(fp, "%zu", &group->numMetrics) != 1) {
            ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Invalid GROUP line (nmetrics)");
            err = -EINVAL;
            goto unlock_err;
        }

        for (size_t j = 0; j < group->numMetrics; j++) {
            RocmarkerMetric *metric = &group->metrics[j];
            if (fscanf(fp, " '%m[^'] '%m[^']", &metric->name, &metric->formula) != 2) {
                ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Invalid GROUP line (metrics)");
                err = -EINVAL;
                goto unlock_err;
            }
        }
    }

    // Read regions: 'regionIdx groupId regionTag'
    for (size_t ir = 0; ir < numRegions; ir++) {
        RocmarkerRegion *region = calloc(1, sizeof(*region));
        if (!region) {
            err = -errno;
            goto unlock_err;
        }

        double execTime;
        if (fscanf(fp, "REGION %ms %d %zu %lf",
                    &region->tag, &region->groupId, &region->execCount, &execTime) != 5) {
            ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Invalid REGION line");
            free(region);
            err = -EINVAL;
            goto unlock_err;
        }
        region->totalTime = (double)(execTime * 1e9);

        char label[LABEL_MAX_SIZE];
        label_fmt(label, sizeof(label), region->tag, region->groupId);

        err = add_smap(rocmarker_ctx->regions, label, region);
        if (err < 0) {
            region_free(region);
            goto unlock_err;
        }

        region->gpuResults = calloc(rocmarker_ctx->numHipDeviceIds, sizeof(*region->gpuResults));
        if (!region->gpuResults) {
            err = -errno;
            goto unlock_err;
        }

        for (size_t ig = 0; ig < rocmarker_ctx->numHipDeviceIds; ig++) {
            RocmarkerGpuResultList *result = &region->gpuResults[ig];

            if (fscanf(fp, ";") != 0) {
                ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Invalid REGION line (sep)");
                err = -EINVAL;
                goto unlock_err;
            }

            size_t groupIdx;
            err = get_group_idx(region->groupId, &groupIdx);
            if (err < 0)
                goto unlock_err;

            result->numCounterValues = rocmarker_ctx->groups[groupIdx].numEventNames;

            for (size_t iv = 0; iv < result->numCounterValues; iv++) {
                if (fscanf(fp, "%lf", &result->counterValues[iv].fullValue) != 1) {
                    ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Invalid REGION line (val)");
                    err = -EINVAL;
                    goto unlock_err;
                }
            }
        }
    }

    pthread_mutex_unlock(&rocmarker_init_mutex);
    return 0;

unlock_err:
    if (fp)
        fclose(fp);
    rocmarker_ctx_free();
    pthread_mutex_unlock(&rocmarker_init_mutex);
    return err;
}

void rocmon_markerDestroyResults(void) {
    pthread_mutex_lock(&rocmarker_init_mutex);

    assert(rocmarker_ctx);
    assert(rocmarker_ctx->main_tid == DUMMY_TID);

    rocmarker_ctx_free();

    pthread_mutex_unlock(&rocmarker_init_mutex);
}
