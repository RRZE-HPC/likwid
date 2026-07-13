/*
 * =======================================================================================
 *
 *      Filename:  rocmon_marker.c
 *
 *      Description:  Marker API interface of module rocmon
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Authors:  Thomas Gruber (tg), thomas.roehl@googlemail.com
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
#ifdef LIKWID_WITH_ROCMON

#include <syscall.h>

#include <lock.h>
#include <bstrlib.h>
#include <error.h>
#include <map.h>
#include <perfgroup.h>
#include <types.h>

#include <likwid.h>
#include <rocmon_types.h>
#include <lw_alloc.h>

#define gettid() syscall(SYS_gettid)

#ifndef NAN
#define NAN (0.0/0.0)
#endif

#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif

static int rocmon_marker_initialized = 0;
static pid_t main_tid = -1;

static int num_groups = 0;
static int* gpu_groups = NULL;
static int active_group = -1;

static int num_gpus = 0;
static int* gpu_ids = NULL;
static Map_t* gpu_maps = NULL;

typedef enum {
    ROCMON_MARKER_STATE_NEW,
    ROCMON_MARKER_STATE_START,
    ROCMON_MARKER_STATE_STOP
} LikwidRegionState;

typedef struct {
    bstring label;

    int gpuId;
    int groupId;

    uint32_t count;
    double timeActive;
    TimerData startTime;

    LikwidRegionState state;
    RocmonEventResultList groupResults;
} RocmonRegionResults;

static int
_rocmon_parse_gpustr(char* gpuStr, int* numGpus, int** gpuIds)
{
    // Create bstring
    bstring bGpuStr = bfromcstr(gpuStr);
    
    // Parse list
    struct bstrList* gpuTokens = bsplit(bGpuStr,',');
    int tmpNumGpus = gpuTokens->qty;

    // Allocate gpuId list
    int* tmpGpuIds = malloc(tmpNumGpus * sizeof(int));
    if (!tmpGpuIds)
    {
        fprintf(stderr,"Cannot allocate space for GPU list.\n");
        bdestroy(bGpuStr);
        bstrListDestroy(gpuTokens);
        return -EXIT_FAILURE;
    }

    // Parse ids to int
    for (int i = 0; i < tmpNumGpus; i++)
    {
        char* tmp = NULL;
        if (bdata(gpuTokens->entry[i]) != NULL) {
            tmp = bdata(gpuTokens->entry[i]);
        } else {
            free(tmpGpuIds);
            bstrListDestroy(gpuTokens);
            bdestroy(bGpuStr);
            return -EXIT_FAILURE;
        }
        tmpGpuIds[i] = atoi(tmp);
    }

    // Copy data
    *numGpus = tmpNumGpus;
    *gpuIds = tmpGpuIds;

    // Destroy bstring
    bdestroy(bGpuStr);
    bstrListDestroy(gpuTokens);

    return -EINVAL;
}

static void label_fmt(char *buf, size_t size, const char *regionTag, int groupId)
{
    snprintf(buf, size, "%s-%d", regionTag, groupId);
}

static void rocmarker_group_fini(RocmarkerGroup *group)
{
    if (!group)
        return;

    if (group->events) {
        for (size_t i = 0; i < group->numEvents; i++) {
            free(group->events[i].eventName);
            free(group->events[i].counterName);
        }
        free(group->events);
    }

    if (group->metrics) {
        for (size_t i = 0; i < group->numMetrics; i++) {
            free(group->metrics[i].name);
            free(group->metrics[i].formula);
        }
        free(group->metrics);
    }
}

static void rocmarker_ctx_free(void)
{
    if (!rocmarker_ctx)
        return;

    free(rocmarker_ctx->hipDeviceIds);
    destroy_smap(rocmarker_ctx->regions);

    if (rocmarker_ctx->groups) {
        for (size_t i = 0; i < rocmarker_ctx->numGroups; i++)
            rocmarker_group_fini(&rocmarker_ctx->groups[i]);
        free(rocmarker_ctx->groups);
    }

    free(rocmarker_ctx);
    rocmarker_ctx = NULL;
}

static int gpulist_from_str(const char *gpustring, size_t *numGpus, int **gpus)
{
    bstring bgpustring = bfromcstr(gpustring);
    if (!bgpustring)
        return -ENOMEM;

    int err                     = 0;
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
    *gpus    = newGpus;

    for (int i = 0; i < gpustrings->qty; i++) {
        char* s = NULL;
        if (bdata(gpustrings->entry[i]) != NULL) {
            s = bdata(gpustrings->entry[i]);
        }
        newGpus[i] = atoi(s);
    }

cleanup:
    if (err < 0)
        free(newGpus);

    bstrListDestroy(gpustrings);
    bdestroy(bgpustring);
    return err;
}

static int eventsets_init(const char *eventStr)
{
    bstring eventStrCopy = bfromcstr(eventStr);
    if (!eventStrCopy)
        return -ENOMEM;

    int err                          = 0;
    struct bstrList *eventsForGroups = bsplit(eventStrCopy, '|');
    if (!eventsForGroups) {
        err = -ENOMEM;
        goto cleanup;
    }

    rocmarker_ctx->groups = calloc(eventsForGroups->qty, sizeof(*rocmarker_ctx->groups));
    if (!rocmarker_ctx->groups) {
        err = -errno;
        goto cleanup;
    }

    rocmarker_ctx->numGroups      = eventsForGroups->qty;
    rocmarker_ctx->activeGroupIdx = 0;

    for (int i = 0; i < eventsForGroups->qty; i++) {
        RocmarkerGroup *group = &rocmarker_ctx->groups[i];

        err = rocmon_addEventSet(bdata(eventsForGroups->entry[i]));
        if (err < 0)
            goto cleanup;

        group->groupId   = err;
        group->numEvents = rocmon_getNumberOfEvents(err);
        group->events    = calloc(group->numEvents, sizeof(*group->events));
        if (!group->events) {
            err = -errno;
            goto cleanup;
        }

        for (size_t i = 0; i < group->numEvents; i++) {
            const char *eventName;
            err = rocmon_getEventName(group->groupId, (int)i, &eventName);
            if (err < 0)
                goto cleanup;

            const char *counterName;
            err = rocmon_getCounterName(group->groupId, (int)i, &counterName);
            if (err < 0)
                goto cleanup;

            group->events[i].eventName = strdup(eventName);
            if (!group->events[i].eventName) {
                err = -errno;
                goto cleanup;
            }

            group->events[i].counterName = strdup(counterName);
            if (!group->events[i].counterName) {
                err = -errno;
                goto cleanup;
            }
        }

        err = rocmon_getNumberOfMetrics(group->groupId);
        if (err < 0)
            goto cleanup;

        group->numMetrics = (size_t)err;
        group->metrics    = calloc(group->numMetrics, sizeof(*group->metrics));
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
            if (!group->metrics[j].formula) {
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

static void region_free_vptr(void *region)
{
    region_free(region);
}

int rocmon_markerInit(void)
{
    const char *eventStr     = getenv("LIKWID_ROCMON_EVENTS");
    const char *gpuStr       = getenv("LIKWID_ROCMON_GPUS");
    const char *verbosityStr = getenv("LIKWID_ROCMON_VERBOSITY");
    const char *debugStr     = getenv("LIKWID_DEBUG");

    if (!eventStr || !gpuStr) {
        ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR,
            "Running without GPU Marker API. Activate GPU Marker API with -m, -G and -W on "
            "commandline.");
        return -EINVAL;
    }

    pthread_mutex_lock(&rocmarker_init_mutex);

    int err = 0;

    if (rocmarker_ctx) {
        err = -EEXIST;
        goto unlock_err;
    }

    rocmarker_ctx = calloc(1, sizeof(*rocmarker_ctx));
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
    size_t numGpuIds = 0;
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
    rocmarker_ctx->hipDeviceIds =
        calloc(rocmarker_ctx->numHipDeviceIds, sizeof(*rocmarker_ctx->hipDeviceIds));
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
}

static void
_rocmon_saveToFile(const char* markerfile)
{
    /* File format
     * 1 numberOfGPUs numberOfRegions numberOfGpuGroups
     * 2 regionID:regionTag0
     * 3 regionID:regionTag1
     * 4 regionID groupID gpuID callCount timeActive numEvents countersvalues(space separated)
     * 5 regionID groupID gpuID callCount timeActive numEvents countersvalues(space separated)
     */

    // Verify there is something to output
    int numberOfRegions = get_map_size(gpu_maps[0]);
    int numberOfGPUs = rocmon_getNumberOfGPUs();
    if ((numberOfGPUs == 0) || (numberOfRegions == 0))
    {
        fprintf(stderr, "No GPUs or regions defined in hash table\n");
        return;
    }

    // Open file in write mode
    FILE* file = fopen(markerfile,"w");
    if (file == NULL)
    {
        fprintf(stderr, "Cannot open file %s\n", markerfile);
        fprintf(stderr, "%s", strerror(errno));
        return;
    }

    // Write header: numberOfGPUs numberOfRegions numberOfGpuGroups
    bstring thread_regs_grps = bformat("%d %d %d", numberOfGPUs, numberOfRegions, num_groups);
    fprintf(file,"%s\n", bdata(thread_regs_grps));
    bdestroy(thread_regs_grps);

    // Write region tags
    for (int j = 0; j < numberOfRegions; j++)
    {
        RocmonRegionResults* results = NULL;
        int ret = get_smap_by_idx(gpu_maps[0], j, (void**)&results);
        if (ret != 0)
        {
            continue;
        }

        // Write region tags: regionID:regionTag0
        bstring tmp = bformat("%d:%s", j, bdata(results->label));
        fprintf(file,"%s\n", bdata(tmp));
        bdestroy(tmp);
    }

    // Write counter values for each region
    for (int j = 0; j < numberOfRegions; j++)
    {
        for (int i = 0; i < numberOfGPUs; i++)
        {
            RocmonRegionResults* results = NULL;
            int ret = get_smap_by_idx(gpu_maps[i], j, (void**)&results);
            if (ret != 0)
            {
                continue;
            }

            // Write: regionID groupID gpuID callCount timeActive numEvents countersvalues(space separated)
            bstring l = bformat("%d %d %d %u %e %d ", 
                            j, results->groupId, gpu_ids[results->gpuId], results->count, 
                            results->timeActive, results->groupResults.numResults);
            for (int k = 0; k < results->groupResults.numResults; k++)
            {
                bstring tmp = bformat("%e ", results->groupResults.results[k].fullValue);
                bconcat(l, tmp);
                bdestroy(tmp);
            }
            fprintf(file,"%s\n", bdata(l));
            bdestroy(l);
        }
    }
}

static void
_rocmon_finalize(void)
{
#define FREE_IF_NOT_NULL(x) if (x != NULL) { free(x); x = NULL; }

    // Ensure markers were initialized
    if (!rocmon_marker_initialized)
    {
        return;
    }

    FREE_IF_NOT_NULL(gpu_ids);
    FREE_IF_NOT_NULL(gpu_groups);

    // Free each map
    for (int i = 0; i < num_gpus; i++)
    {
        destroy_smap(gpu_maps[i]);
    }
    
    rocmon_finalize();
}


void
rocmon_markerInit(void)
{
    int ret;

    // Check if rocmon markers are already initialized
    if (rocmon_marker_initialized)
    {
        return;
    }

    // Get environment variables
    char* eventStr = getenv("LIKWID_ROCMON_EVENTS");
    char* gpuStr = getenv("LIKWID_ROCMON_GPUS");
    char* gpuFileStr = getenv("LIKWID_ROCMON_FILEPATH");
    char* verbosityStr = getenv("LIKWID_ROCMON_VERBOSITY");
    char* debugStr = getenv("LIKWID_DEBUG");

    // Validate environment variables are set
    if ((eventStr == NULL) || (gpuStr == NULL) || (gpuFileStr == NULL))
    {
        fprintf(stderr, "Running without GPU Marker API. Activate GPU Marker API with -m, -G and -W on commandline.\n");
        return;
    }
    if (verbosityStr != NULL) {
        int v = atoi(verbosityStr);
        rocmon_setVerbosity(v);
    }
    if (debugStr != NULL)
    {
        int v = atoi(debugStr);
        perfmon_setVerbosity(v);
    }

    // Init timer module
    timer_init();
    
    // Save current thread id
    main_tid = gettid();

    // Parse GPU list
    ret = _rocmon_parse_gpustr(gpuStr, &num_gpus, &gpu_ids);
    if (ret < 0)
    {
        fprintf(stderr, "Error parsing GPU string.\n");
        exit(ret);
    }

    // Allocate GPU Hashmaps
    gpu_maps = malloc(num_gpus * sizeof(Map_t));
    if (!gpu_maps)
    {
        fprintf(stderr,"Cannot allocate space for results.\n");
        free(gpu_ids);
        exit(-EXIT_FAILURE);
    }

    // Parse event string
    bstring bGeventStr = bfromcstr(eventStr);
    struct bstrList* gEventStrings = bsplit(bGeventStr,'|');
    num_groups = gEventStrings->qty;

    // Allocate space for event group ids
    gpu_groups = malloc(num_groups * sizeof(int));
    if (!gpu_groups)
    {
        fprintf(stderr,"Cannot allocate space for group handling.\n");
        bstrListDestroy(gEventStrings);
        free(gpu_ids);
        free(gpu_maps);
        bdestroy(bGeventStr);
        exit(-EXIT_FAILURE);
    }

    // Initialize rocmon
    ret = rocmon_init(num_gpus, gpu_ids);
    if (ret < 0)
    {
        fprintf(stderr,"Error init Rocmon Marker API.\n");
        free(gpu_ids);
        free(gpu_maps);
        free(gpu_groups);
        bstrListDestroy(gEventStrings);
        bdestroy(bGeventStr);
        exit(-EXIT_FAILURE);
    }

    // Add event sets
    for (int i = 0; i < gEventStrings->qty; i++)
    {
        ret = rocmon_addEventSet(bdata(gEventStrings->entry[i]), &gpu_groups[i]);
        if (ret < 0)
        {
            fprintf(stderr,"Error setting up Rocmon Marker API.\n");
            free(gpu_ids);
            free(gpu_maps);
            free(gpu_groups);
            exit(-EXIT_FAILURE);
        }
    }
    bstrListDestroy(gEventStrings);
    bdestroy(bGeventStr);
    active_group = 0;

    // Init GPU maps
    for (int i = 0; i < num_gpus; i++)
    {
        init_smap(&gpu_maps[i]);
    }

    // Setup counters
    ret = rocmon_setupCounters(gpu_groups[active_group]);
    if (ret)
    {
        fprintf(stderr,"Error setting up Rocmon Marker API.\n");
        free(gpu_ids);
        free(gpu_maps);
        free(gpu_groups);
        rocmon_finalize();
        exit(-EXIT_FAILURE);
    }

    // Start counters
    ret = rocmon_startCounters();
    if (ret)
    {
        fprintf(stderr,"Error starting up Rocmon Marker API.\n");
        free(gpu_ids);
        free(gpu_maps);
        free(gpu_groups);
        rocmon_finalize();
        exit(-EXIT_FAILURE);
    }

    rocmon_marker_initialized = 1;
}


void
rocmon_markerClose(void)
{
    // Ensure markers were initialized
    if (!rocmon_marker_initialized)
    {
        return;
    }

    // Verify that we are on the same thread
    if (gettid() != main_tid)
    {
        return;
    }

    // Stop counters
    rocmon_stopCounters();

    // Get markerfile path from environment
    char* markerfile = getenv("LIKWID_ROCMON_FILEPATH");
    if (markerfile == NULL)
    {
        fprintf(stderr, "Is the application executed with LIKWID wrapper? No file path for the Rocmon Marker API output defined.\n");
        return;
    }
    else
    {
        _rocmon_saveToFile(markerfile);
    }

    _rocmon_finalize();
}


int
rocmon_markerWriteFile(const char* markerfile)
{
    if (!markerfile)
    {
        return -EINVAL;
    }
    _rocmon_saveToFile(markerfile);
    return 0;
}

int
rocmon_markerRegisterRegion(const char* regionTag)
{
    // Ensure markers were initialized
    if (!rocmon_marker_initialized)
    {
        return -EFAULT;
    }

    // Verify that we are on the same thread
    if (gettid() != main_tid)
    {
        return 0;
    }

    // Add region results to each gpu map
    for (int i = 0; i < num_gpus; i++)
    {
        // Allocate memory for region results
        RocmonRegionResults* results = malloc(sizeof(RocmonRegionResults));
        if (results == NULL)
        {
            fprintf(stderr, "Failed to register region %s\n", regionTag);
            return -ENOMEM;
        }

        // Initialize struct
        results->label = bformat("%s-%d", regionTag, active_group);
        results->timeActive = 0;
        results->count = 0;
        results->gpuId = gpu_ids[i];
        results->groupId = gpu_groups[active_group];
        results->state = ROCMON_MARKER_STATE_NEW;
        
        // Get number of events in active group
        int numEvents = rocmon_getNumberOfEvents(active_group);
        
        // Allocate memory for event results
        RocmonEventResult* tmpResults = malloc(numEvents * sizeof(RocmonEventResult));
        if (tmpResults == NULL)
        {
            fprintf(stderr, "Failed to allocate event results for region %s\n", regionTag);
            free(results);
            return -ENOMEM;
        }
        results->groupResults.results = tmpResults;
        results->groupResults.numResults = numEvents;

        // Initialize event results
        for (int j = 0; j < numEvents; j++)
        {
            RocmonEventResult* res = &results->groupResults.results[j];
            res->lastValue = 0.0;
            res->fullValue = 0.0;
        }

        // Add region results to map
        add_smap(gpu_maps[i], bdata(results->label), results);
    }

    return 0;
}


int
rocmon_markerStartRegion(const char* regionTag)
{
    // Ensure markers were initialized
    if (!rocmon_marker_initialized)
    {
        return -EFAULT;
    }

    // Verify that we are on the same thread
    if (gettid() != main_tid)
    {
        return 0;
    }

    // Read counters (for all devices)
    TimerData timestamp;
    ROCMON_DEBUG_PRINT(DEBUGLEV_DETAIL, "START REGION '%s' (group %d)", regionTag, active_group);
    timer_start(&timestamp);
    rocmon_readCounters();

    // Copy values for each device
    bstring tag = bformat("%s-%d", regionTag, active_group);
    for (int i = 0; i < num_gpus; i++)
    {
        // Get results from map
        RocmonRegionResults* results = NULL;
        int ret = get_smap_by_key(gpu_maps[i], bdata(tag), (void**) &results);
        if (ret < 0)
        {
            fprintf(stderr, "WARN: Starting an unknown region %s\n", regionTag);
            return -EFAULT;
        }

        // Check region state
        if (results->state == ROCMON_MARKER_STATE_START)
        {
            fprintf(stderr, "WARN: Starting an already-started region %s\n", regionTag);
            return -EFAULT;
        }

        // Update timer information
        results->startTime.start = timestamp.start;

        // Copy values for each event
        for (int j = 0; j < results->groupResults.numResults; j++)
        {
            RocmonEventResult* res = &results->groupResults.results[j];
            res->lastValue = rocmon_getResult(results->gpuId, results->groupId, j);
        }

        results->state = ROCMON_MARKER_STATE_START;
    }

    bdestroy(tag);
    return 0;
}


int
rocmon_markerStopRegion(const char* regionTag)
{
    // Ensure markers were initialized
    if (!rocmon_marker_initialized)
    {
        return -EFAULT;
    }

    // Verify that we are on the same thread
    if (gettid() != main_tid)
    {
        return 0;
    }

    // Read counters (for all devices)
    TimerData timestamp;
    ROCMON_DEBUG_PRINT(DEBUGLEV_DETAIL, "STOP REGION '%s' (group %d)", regionTag, active_group);
    timer_stop(&timestamp);
    rocmon_readCounters();

    // Copy values for each device
    bstring tag = bformat("%s-%d", regionTag, active_group);
    for (int i = 0; i < num_gpus; i++)
    {
        // Get results from map
        RocmonRegionResults* results = NULL;
        int ret = get_smap_by_key(gpu_maps[i], bdata(tag), (void**) &results);
        if (ret < 0)
        {
            fprintf(stderr, "WARN: Stopping an unknown region %s\n", regionTag);
            return -EFAULT;
        }

        // Check region state
        if (results->state != ROCMON_MARKER_STATE_START)
        {
            fprintf(stderr, "WARN: Stopping an not-started region %s\n", regionTag);
            return -EFAULT;
        }

        // Update timer and count information
        results->startTime.stop = timestamp.stop;
        results->timeActive += timer_print(&results->startTime);
        results->count++;

        // Copy values for each event
        for (int j = 0; j < results->groupResults.numResults; j++)
        {
            RocmonEventResult* res = &results->groupResults.results[j];
            if (rocmon_getEventName(results->groupId, j)[1] == 'S')
            {   // ROCm SMI event
                res->fullValue += rocmon_getLastResult(results->gpuId, results->groupId, j);
            }
            else
            {   // ROC-Profiler event
                res->fullValue += rocmon_getResult(results->gpuId, results->groupId, j) - res->lastValue;
            }
        }

        results->state = ROCMON_MARKER_STATE_STOP;
    }

    bdestroy(tag);
    return 0;
}


void
rocmon_markerGetRegion(
        const char* regionTag,
        int* nr_gpus,
        int* nr_events,
        double** events,
        double** time,
        int **count)
{
    // Ensure markers were initialized
    if (!rocmon_marker_initialized)
    {
        return;
    }

    // TODO: implement this function
    fprintf(stderr, "WARN: Function 'rocmon_markerGetRegion' is not implemented.\n");

    (void)regionTag;
    
    *nr_gpus = 0;
    *nr_events = 0;
    *time = NULL;
    *events = NULL;
    *count = NULL;
}


int
rocmon_markerResetRegion(const char* regionTag)
{
    // Ensure markers were initialized
    if (!rocmon_marker_initialized)
    {
        return -EFAULT;
    }

    // Verify that we are on the same thread
    if (gettid() != main_tid)
    {
        return 0;
    }

    // Reset values for each device
    bstring tag = bformat("%s-%d", regionTag, active_group);
    for (int i = 0; i < num_gpus; i++)
    {
        // Get results from map
        RocmonRegionResults* results = NULL;
        int ret = get_smap_by_key(gpu_maps[i], bdata(tag), (void**) &results);
        if (ret < 0)
        {
            fprintf(stderr, "WARN: Stopping an unknown region %s\n", regionTag);
            return -EFAULT;
        }

        // Update timer and count information
        timer_reset(&results->startTime);
        results->timeActive = 0;
        results->count = 0;

        // Reset values for each event
        for (int j = 0; j < results->groupResults.numResults; j++)
        {
            RocmonEventResult* res = &results->groupResults.results[j];
            res->lastValue = 0;
            res->fullValue = 0;
        }
    }

    return 0;
}


void
rocmon_markerNextGroup(void)
{
    // Ensure markers were initialized
    if (!rocmon_marker_initialized)
    {
        return;
    }

    // Verify that we are on the same thread
    if (gettid() != main_tid)
    {
        return;
    }

    int nextGroup = (active_group + 1) % num_groups;
    if (nextGroup != active_group)
    {
        rocmon_switchActiveGroup(nextGroup);
    }
}


LikwidRocmResults* rocmMarkerResults = NULL;
int rocmMarkerRegions = 0;

int
rocmon_readMarkerFile(const char* filename)
{
    int ret = 0;
    FILE* fp = NULL;
    char buf[2048];
    buf[0] = '\0';
    char *ptr = NULL;
    int gpus = 0, groups = 0, regions = 0;
    int nr_regions = 0;

    if (filename == NULL)
    {
        return -EINVAL;
    }
    if (access(filename, R_OK))
    {
        return -EINVAL;
    }
    fp = fopen(filename, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Error opening file %s\n", filename);
    }
    ptr = fgets(buf, sizeof(buf), fp);
    ret = sscanf(buf, "%d %d %d", &gpus, &regions, &groups);
    if (ret != 3)
    {
        fprintf(stderr, "ROCMMarker file missformatted.\n");
        return -EINVAL;
    }
    rocmMarkerResults = realloc(rocmMarkerResults, regions * sizeof(LikwidRocmResults));
    if (rocmMarkerResults == NULL)
    {
        fprintf(stderr, "Failed to allocate %lu bytes for the marker results storage\n", regions * sizeof(LikwidRocmResults));
        return -ENOMEM;
    }
    int* regionGPUs = (int*)malloc(regions * sizeof(int));
    if (regionGPUs == NULL)
    {
        fprintf(stderr, "Failed to allocate %lu bytes for temporal gpu count storage\n", regions * sizeof(int));
        return -ENOMEM;
    }
    rocmMarkerRegions = regions;
    for ( int i=0; i < regions; i++ )
    {
        regionGPUs[i] = 0;
        rocmMarkerResults[i].gpuCount = gpus;
        rocmMarkerResults[i].time = (double*) malloc(gpus * sizeof(double));
        if (!rocmMarkerResults[i].time)
        {
            fprintf(stderr, "Failed to allocate %lu bytes for the time storage\n", gpus * sizeof(double));
            break;
        }
        rocmMarkerResults[i].count = (uint32_t*) malloc(gpus * sizeof(uint32_t));
        if (!rocmMarkerResults[i].count)
        {
            fprintf(stderr, "Failed to allocate %lu bytes for the count storage\n", gpus * sizeof(uint32_t));
            break;
        }
        rocmMarkerResults[i].gpulist = (int*) malloc(gpus * sizeof(int));
        if (!rocmMarkerResults[i].gpulist)
        {
            fprintf(stderr, "Failed to allocate %lu bytes for the gpulist storage\n", gpus * sizeof(int));
            break;
        }
        rocmMarkerResults[i].counters = (double**) malloc(gpus * sizeof(double*));
        if (!rocmMarkerResults[i].counters)
        {
            fprintf(stderr, "Failed to allocate %lu bytes for the counter result storage\n", gpus * sizeof(double*));
            break;
        }
    }
    while (fgets(buf, sizeof(buf), fp))
    {
        if (strchr(buf,':'))
        {
            int regionid = 0, groupid = -1;
            char regiontag[100];
            char* ptr = NULL;
            char* colonptr = NULL;
            regiontag[0] = '\0';
            ret = sscanf(buf, "%d:%s", &regionid, regiontag);

            ptr = strrchr(regiontag,'-');
            colonptr = strchr(buf,':');
            if (ret != 2 || ptr == NULL || colonptr == NULL)
            {
                fprintf(stderr, "Line %s not a valid region description\n", buf);
                continue;
            }
            groupid = atoi(ptr+1);
            snprintf(regiontag, strlen(regiontag)-strlen(ptr)+1, "%s", &(buf[colonptr-buf+1]));
            rocmMarkerResults[regionid].groupID = groupid;
            rocmMarkerResults[regionid].tag = bfromcstr(regiontag);
            nr_regions++;
        }
        else
        {
            int regionid = 0, groupid = 0, gpu = 0, count = 0, nevents = 0;
            int gpuidx = 0, eventidx = 0;
            double time = 0;
            char remain[1024];
            remain[0] = '\0';
            ret = sscanf(buf, "%d %d %d %d %lf %d %[^\t\n]", &regionid, &groupid, &gpu, &count, &time, &nevents, remain);
            if (ret != 7)
            {
                fprintf(stderr, "Line %s not a valid region values line\n", buf);
                continue;
            }
            if (gpu >= 0)
            {
                gpuidx = regionGPUs[regionid];
                rocmMarkerResults[regionid].gpulist[gpuidx] = gpu;
                rocmMarkerResults[regionid].eventCount = nevents;
                rocmMarkerResults[regionid].time[gpuidx] = time;
                rocmMarkerResults[regionid].count[gpuidx] = count;
                rocmMarkerResults[regionid].counters[gpuidx] = lw_calloc(nevents, sizeof(double));

                eventidx = 0;
                ptr = strtok(remain, " ");
                while (ptr != NULL && eventidx < nevents)
                {
                    sscanf(ptr, "%lf", &(rocmMarkerResults[regionid].counters[gpuidx][eventidx]));
                    ptr = strtok(NULL, " ");
                    eventidx++;
                }
                regionGPUs[regionid]++;
            }
        }
    }
    for ( int i=0; i < regions; i++ )
    {
        rocmMarkerResults[i].gpuCount = regionGPUs[i];
    }
    free(regionGPUs);
    fclose(fp);
    return nr_regions;
}

void
rocmon_destroyMarkerResults()
{
    int i = 0, j = 0;
    if (rocmMarkerResults != NULL)
    {
        for (i = 0; i < rocmMarkerRegions; i++)
        {
            free(rocmMarkerResults[i].time);
            free(rocmMarkerResults[i].count);
            free(rocmMarkerResults[i].gpulist);
            for (j = 0; j < rocmMarkerResults[i].gpuCount; j++)
            {
                free(rocmMarkerResults[i].counters[j]);
            }
            free(rocmMarkerResults[i].counters);
            bdestroy(rocmMarkerResults[i].tag);
        }
        free(rocmMarkerResults);
        rocmMarkerResults = NULL;
        rocmMarkerRegions = 0;
    }
}


int
rocmon_getCountOfRegion(int region, int gpu)
{
    if (rocmMarkerResults == NULL)
    {
        ERROR_PRINT("Rocmon module not properly initialized");
        return -EINVAL;
    }
    if (region < 0 || region >= rocmMarkerRegions)
    {
        return -EINVAL;
    }
    if (gpu < 0 || gpu >= rocmMarkerResults[region].gpuCount)
    {
        return -EINVAL;
    }
    if (rocmMarkerResults[region].count == NULL)
    {
        return 0;
    }
    return rocmMarkerResults[region].count[gpu];
}

double
rocmon_getTimeOfRegion(int region, int gpu)
{
    if (rocmMarkerResults == NULL)
    {
        ERROR_PRINT("Rocmon module not properly initialized");
        return -EINVAL;
    }
    if (region < 0 || region >= rocmMarkerRegions)
    {
        return -EINVAL;
    }
    if (gpu < 0 || gpu >= rocmMarkerResults[region].gpuCount)
    {
        return -EINVAL;
    }
    if (rocmMarkerResults[region].time == NULL)
    {
        return 0.0;
    }
    return rocmMarkerResults[region].time[gpu];
}

int
rocmon_getGpulistOfRegion(int region, int count, int* gpulist)
{
    int i;
    if (rocmMarkerResults == NULL)
    {
        ERROR_PRINT("Rocmon module not properly initialized");
        return -EINVAL;
    }
    if (region < 0 || region >= rocmMarkerRegions)
    {
        return -EINVAL;
    }
    if (gpulist == NULL)
    {
        return -EINVAL;
    }
    for (i=0; i< MIN(count, rocmMarkerResults[region].gpuCount); i++)
    {
        gpulist[i] = rocmMarkerResults[region].gpulist[i];
    }
    return MIN(count, rocmMarkerResults[region].gpuCount);
}

int
rocmon_getGpusOfRegion(int region)
{
    if (rocmMarkerResults == NULL)
    {
        ERROR_PRINT("Rocmon module not properly initialized");
        return -EINVAL;
    }
    if (region < 0 || region >= rocmMarkerRegions)
    {
        return -EINVAL;
    }
    return rocmMarkerResults[region].gpuCount;
}

int
rocmon_getMetricsOfRegion(int region)
{
    if (rocmMarkerResults == NULL)
    {
        ERROR_PRINT("Rocmon module not properly initialized");
        return -EINVAL;
    }
    if (region < 0 || region >= rocmMarkerRegions)
    {
        return -EINVAL;
    }
    return rocmon_getNumberOfMetrics(rocmMarkerResults[region].groupID);
}

int
rocmon_getNumberOfRegions()
{
    if (rocmMarkerResults == NULL)
    {
        ERROR_PRINT("Rocmon module not properly initialized");
        return -EINVAL;
    }
    return rocmMarkerRegions;
}

int
rocmon_getGroupOfRegion(int region)
{
    if (rocmMarkerResults == NULL)
    {
        ERROR_PRINT("Rocmon module not properly initialized");
        return -EINVAL;
    }
    if (region < 0 || region >= rocmMarkerRegions)
    {
        return -EINVAL;
    }
    return rocmMarkerResults[region].groupID;
}

char*
rocmon_getTagOfRegion(int region)
{
    if (rocmMarkerResults == NULL)
    {
        ERROR_PRINT("Rocmon module not properly initialized");
        return NULL;
    }
    if (region < 0 || region >= rocmMarkerRegions)
    {
        return NULL;
    }
    return bdata(rocmMarkerResults[region].tag);
}

int
rocmon_getEventsOfRegion(int region)
{
    if (rocmMarkerResults == NULL)
    {
        ERROR_PRINT("Rocmon module not properly initialized");
        return -EINVAL;
    }
    if (region < 0 || region >= rocmMarkerRegions)
    {
        return -EINVAL;
    }
    return rocmMarkerResults[region].eventCount;
}

double
rocmon_getResultOfRegionGpu(int region, int eventId, int gpuId)
{
    if (rocmMarkerResults == NULL)
    {
        ERROR_PRINT("Rocmon module not properly initialized");
        return -EINVAL;
    }
    if (region < 0 || region >= rocmMarkerRegions)
    {
        return -EINVAL;
    }
    if (gpuId < 0 || gpuId >= rocmMarkerResults[region].gpuCount)
    {
        return -EINVAL;
    }
    if (eventId < 0 || eventId >= rocmMarkerResults[region].eventCount)
    {
        return -EINVAL;
    }
    if (rocmMarkerResults[region].counters[gpuId] == NULL)
    {
        return 0.0;
    }
    return rocmMarkerResults[region].counters[gpuId][eventId];
}

double
rocmon_getMetricOfRegionGpu(int region, int metricId, int gpuId)
{
    int e = 0, err = 0;
    double result = 0.0;
    CounterList clist;
    if (rocmMarkerResults == NULL)
    {
        ERROR_PRINT("Rocmon module not properly initialized");
        return NAN;
    }
    if (region < 0 || region >= rocmMarkerRegions)
    {
        return NAN;
    }
    if (rocmMarkerResults == NULL)
    {
        return NAN;
    }
    if (gpuId < 0 || gpuId >= rocmMarkerResults[region].gpuCount)
    {
        return NAN;
    }
    GroupInfo* ginfo = &rocmon_context->groups[rocmMarkerResults[region].groupID];
    if (metricId < 0 || metricId >= ginfo->nmetrics)
    {
        return NAN;
    }
    char *f = ginfo->metricformulas[metricId];
    timer_init();
    init_clist(&clist);
    for (e = 0; e < rocmMarkerResults[region].eventCount; e++)
    {
        double res = rocmon_getResultOfRegionGpu(region, e, gpuId);
        char* ctr = ginfo->counters[e];
        add_to_clist(&clist, ctr, res);
    }
    add_to_clist(&clist, "time", rocmon_getTimeOfRegion(rocmMarkerResults[region].groupID, gpuId));
    add_to_clist(&clist, "inverseClock", 1.0/timer_getCycleClock());
    add_to_clist(&clist, "true", 1);
    add_to_clist(&clist, "false", 0);

    err = calc_metric(f, &clist, &result);
    if (err < 0)
    {
        ERROR_PRINT("Cannot calculate formula %s", f);
        return NAN;
    }
    destroy_clist(&clist);
    return result;
}

int rocmon_markerGetRegionStats(
    const char *regionTag, int groupId, size_t *execCount, double *execTime)
{
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
    *execTime  = (double)region->totalTime / 1e9;
    *execCount = region->execCount;
    return 0;
}

int rocmon_markerGetRegionTags(char ***regionTags, int **regionGroupIds, size_t *numRegions)
{
    if (!rocmarker_ctx)
        return -EFAULT;

    int newNumRegions = get_map_size(rocmarker_ctx->regions);
    if (newNumRegions < 0)
        return newNumRegions;

    char **newRegionTags = calloc((size_t)newNumRegions, sizeof(*newRegionTags));
    if (!newRegionTags)
        return -errno;

    int err                = 0;
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

    *regionTags     = newRegionTags;
    *regionGroupIds = newRegionGroupIds;
    *numRegions     = (size_t)newNumRegions;

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

int rocmon_markerWriteFile(const char *markerfile)
{
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
     * GROUP groupId numEvents eventA counterA eventB counterB eventC counterC numMetrics metricNameA metricFormulaA metricNameB metricFormulaB
     * ... ('numGroups' number of GROUP lines)
     * REGION regionTag groupId execCount execTime ; 42.4 8.24 -1.0 ; 1337 0.0 0.12e5
     * ... ('numRegions' number of REGION lines)
     *     ('numGpus' groups of results, separated by ';')
     */

    /* Checking for errors with fprintf really doens't work, if we don't know the
     * non-truncated number of characters to be written. However, I'm lazy and I don't
     * want to allocate something, format it, to then check if it's all written. */

    chk_fprintf(fp,
        "ROCMON_MARKER_FILE %zu %zu %zu\n",
        rocmarker_ctx->numHipDeviceIds,
        numRegions,
        rocmarker_ctx->numGroups);

    // Write hip device IDs
    for (size_t i = 0; i < rocmarker_ctx->numHipDeviceIds; i++)
        chk_fprintf(fp, "GPU %d\n", rocmarker_ctx->hipDeviceIds[i]);

    // Write groups
    for (size_t i = 0; i < rocmarker_ctx->numGroups; i++) {
        RocmarkerGroup *group = &rocmarker_ctx->groups[i];
        chk_fprintf(fp, "GROUP %d %zu", group->groupId, group->numEvents);
        for (size_t j = 0; j < group->numEvents; j++)
            chk_fprintf(fp, " %s %s", group->events[j].eventName, group->events[j].counterName);
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

        chk_fprintf(fp,
            "REGION %s %d %zu %f",
            region->tag,
            region->groupId,
            region->execCount,
            (double)region->totalTime / 1e9);

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

int rocmon_markerInitResultsFromFile(const char *markerfile)
{
    FILE *fp = NULL;
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

    rocmarker_ctx = calloc(1, sizeof(*rocmarker_ctx));
    if (!rocmarker_ctx) {
        err = -errno;
        goto unlock_err;
    }

    rocmarker_ctx->main_tid = DUMMY_TID;

    fp = fopen(markerfile, "r");
    if (!fp) {
        err = -errno;
        goto unlock_err;
    }

    size_t numRegions;

    // read line: 'numGpus numRegions numGroups'
    if (fscanf(fp,
            "ROCMON_MARKER_FILE %zu %zu %zu\n",
            &rocmarker_ctx->numHipDeviceIds,
            &numRegions,
            &rocmarker_ctx->numGroups) != 3) {
        ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Cannot parse marker header");
        err = -EINVAL;
        goto unlock_err;
    }

    rocmarker_ctx->hipDeviceIds =
        calloc(rocmarker_ctx->numHipDeviceIds, sizeof(*rocmarker_ctx->hipDeviceIds));
    if (!rocmarker_ctx->hipDeviceIds) {
        err = -errno;
        goto unlock_err;
    }

    // read multiple lines: 'gpuIdx hipDeviceId'
    for (size_t i = 0; i < rocmarker_ctx->numHipDeviceIds; i++) {
        if (fscanf(fp, "GPU %d\n", &rocmarker_ctx->hipDeviceIds[i]) != 1) {
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

        if (fscanf(fp, "GROUP %d %zu", &group->groupId, &group->numEvents) != 2) {
            ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Invalid GROUP line");
            err = -EINVAL;
            goto unlock_err;
        }

        group->events = calloc(group->numEvents, sizeof(*group->events));
        if (!group->events) {
            err = -errno;
            goto unlock_err;
        }

        for (size_t j = 0; j < group->numEvents; j++) {
            if (fscanf(
                    fp, " %ms %ms", &group->events[j].eventName, &group->events[j].counterName) !=
                2) {
                ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Invalid GROUP line (event/counter name)");
                err = -EINVAL;
                goto unlock_err;
            }
        }

        if (fscanf(fp, " %zu", &group->numMetrics) != 1) {
            ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Invalid GROUP line (nmetrics)");
            err = -EINVAL;
            goto unlock_err;
        }

        group->metrics = calloc(group->numMetrics, sizeof(*group->metrics));
        if (!group->metrics) {
            err = -errno;
            goto unlock_err;
        }

        for (size_t j = 0; j < group->numMetrics; j++) {
            RocmarkerMetric *metric = &group->metrics[j];
            if (fscanf(fp, " '%m[^']' '%m[^']'", &metric->name, &metric->formula) != 2) {
                ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Invalid GROUP line (metrics)");
                err = -EINVAL;
                goto unlock_err;
            }
        }

        int d = fscanf(fp, "\n");
        d++;
    }

    // Read regions: 'regionIdx groupId regionTag'
    for (size_t ir = 0; ir < numRegions; ir++) {
        RocmarkerRegion *region = calloc(1, sizeof(*region));
        if (!region) {
            err = -errno;
            goto unlock_err;
        }

        double execTime;
        if (fscanf(fp,
                "REGION %ms %d %zu %lf",
                &region->tag,
                &region->groupId,
                &region->execCount,
                &execTime) != 4) {
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

            if (fscanf(fp, " ;") != 0) {
                ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Invalid REGION line (sep)");
                err = -EINVAL;
                goto unlock_err;
            }

            size_t groupIdx;
            err = get_group_idx(region->groupId, &groupIdx);
            if (err < 0)
                goto unlock_err;

            result->numCounterValues = rocmarker_ctx->groups[groupIdx].numEvents;
            result->counterValues =
                calloc(result->numCounterValues, sizeof(*result->counterValues));
            if (!result->counterValues) {
                err = -errno;
                goto unlock_err;
            }

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

void rocmon_markerDestroyResults(void)
{
    pthread_mutex_lock(&rocmarker_init_mutex);

    assert(rocmarker_ctx);
    assert(rocmarker_ctx->main_tid == DUMMY_TID);

    rocmarker_ctx_free();

    pthread_mutex_unlock(&rocmarker_init_mutex);
}
