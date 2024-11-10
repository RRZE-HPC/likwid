/*
 * =======================================================================================
 *
 *      Filename:  libnvctr.c
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
#include <rocmon.h>
#include <rocmon_common_types.h>
#include <rocmon_v1_types.h>
#ifdef LIKWID_ROCPROF_SDK
#include <rocmon_sdk_types.h>
#endif
#include <rocmon_smi_types.h>

#ifndef FREE_IF_NOT_NULL
#define FREE_IF_NOT_NULL(x) if (x != NULL) { free(x); x = NULL; }
#endif

#ifndef gettid
#define gettid() syscall(SYS_gettid)
#endif

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
    printf("rocmon_markerInit\n");

    // Get environment variables
    char* eventStr = getenv("LIKWID_ROCMON_EVENTS");
    char* gpuStr = getenv("LIKWID_ROCMON_GPUS");
    char* gpuFileStr = getenv("LIKWID_ROCMON_FILEPATH");
    char* verbosityStr = getenv("LIKWID_ROCMON_VERBOSITY");
    char* debugStr = getenv("LIKWID_DEBUG");

    // Validate environment variables are set
    if ((eventStr == NULL) || (gpuStr == NULL) || (gpuFileStr == NULL))
    {
        fprintf(stderr, "Running without Rocmon Marker API. Activate Rocmon Marker API with -m, -I and -R on commandline.\n");
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
        fprintf(stderr,"Error initializing Rocmon Marker API with %d\n", ret);
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
            fprintf(stderr,"Error setting up Rocmon Marker API: %d\n", ret);
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
        fprintf(stderr,"Error setting up Rocmon Marker API: %d\n", ret);
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
        fprintf(stderr,"Error starting up Rocmon Marker API: %d\n", ret);
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
        printf("Saving ROCMON MarkerAPI results to %s\n", markerfile);
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
    ROCMON_DEBUG_PRINT(DEBUGLEV_DETAIL, START REGION '%s' (group %d), regionTag, active_group);
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
    ROCMON_DEBUG_PRINT(DEBUGLEV_DETAIL, STOP REGION '%s' (group %d), regionTag, active_group);
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
    int ret = 0, i = 0;
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
    printf("# %s\n", buf);
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
    for ( uint32_t i=0; i < regions; i++ )
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
        printf("# %s\n", buf);
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
                rocmMarkerResults[regionid].counters[gpuidx] = malloc(nevents * sizeof(double));

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
    for ( uint32_t i=0; i < regions; i++ )
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
        ERROR_PLAIN_PRINT(Rocmon module not properly initialized);
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
        ERROR_PLAIN_PRINT(Rocmon module not properly initialized);
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
        ERROR_PLAIN_PRINT(Rocmon module not properly initialized);
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
        ERROR_PLAIN_PRINT(Rocmon module not properly initialized);
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
        ERROR_PLAIN_PRINT(Rocmon module not properly initialized);
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
        ERROR_PLAIN_PRINT(Rocmon module not properly initialized);
        return -EINVAL;
    }
    return rocmMarkerRegions;
}

int
rocmon_getGroupOfRegion(int region)
{
    if (rocmMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Rocmon module not properly initialized);
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
        ERROR_PLAIN_PRINT(Rocmon module not properly initialized);
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
        ERROR_PLAIN_PRINT(Rocmon module not properly initialized);
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
        ERROR_PLAIN_PRINT(Rocmon module not properly initialized);
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
        ERROR_PLAIN_PRINT(Rocmon module not properly initialized);
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
    GroupInfo* ginfo = rocmon_get_group(rocmMarkerResults[region].groupID);
    if ((!ginfo) || (metricId < 0) || (metricId >= ginfo->nmetrics))
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
        ERROR_PRINT(Cannot calculate formula %s, f);
        return NAN;
    }
    destroy_clist(&clist);
    return result;
}

#endif /* LIKWID_WITH_ROCMON */
