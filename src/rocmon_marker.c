/*
 * =======================================================================================
 *
 *      Filename:  libnvctr.c
 *
 *      Description:  Marker API interface of module nvmon
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

#include <likwid.h>
#include <rocmon_types.h>

#define gettid() syscall(SYS_gettid)

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
        tmpGpuIds[i] = atoi(bdata(gpuTokens->entry[i]));
    }

    // Copy data
    *numGpus = tmpNumGpus;
    *gpuIds = tmpGpuIds;

    // Destroy bstring
    bdestroy(bGpuStr);
    bstrListDestroy(gpuTokens);

    return 0;
}

static void
_rocmon_saveToFile(void)
{
    /* File format
     * 1 numberOfGPUs numberOfRegions numberOfGpuGroups
     * 2 regionID:regionTag0
     * 3 regionID:regionTag1
     * 4 regionID groupID gpuID callCount timeActive numEvents countersvalues(space separated)
     * 5 regionID groupID gpuID callCount timeActive numEvents countersvalues(space separated)
     */

    // Get markerfile path from environment
    char* markerfile = getenv("LIKWID_ROCMON_FILEPATH");
    if (markerfile == NULL)
    {
        fprintf(stderr, "Is the application executed with LIKWID wrapper? No file path for the Rocmon Marker API output defined.\n");
        return;
    }

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

    // Validate environment variables are set
    if ((eventStr == NULL) || (gpuStr == NULL) || (gpuFileStr == NULL))
    {
        fprintf(stderr, "Running without GPU Marker API. Activate GPU Marker API with -m, -G and -W on commandline.\n");
        return;
    }

    // TODO: Verbosity

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

    _rocmon_saveToFile();
    _rocmon_finalize();
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
        results->gpuId = i;
        results->groupId = active_group;
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
            res->lastValue = rocmon_getLastResult(i, results->groupId, j);
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
            res->fullValue = rocmon_getLastResult(i, results->groupId, j) - res->lastValue;
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

#endif /* LIKWID_WITH_ROCMON */
