/*
 * =======================================================================================
 *
 *      Filename:  appDaemon.c
 *
 *      Description:  Implementation a interface library to hook into applications
 *                    using the GOTCHA library
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
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


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <bstrlib.h>
#include <stdbool.h>

#include <likwid.h>
#include <error.h>

typedef void(*appdaemon_exit_func)(void);
#define APPDAEMON_MAX_EXIT_FUNCS 2
static appdaemon_exit_func appdaemon_exit_funcs[APPDAEMON_MAX_EXIT_FUNCS];
static int appdaemon_num_exit_funcs = 0;

static struct tagbstring daemon_name = bsStatic("likwid-appDaemon.so");
static FILE* output_file = NULL;

// Timeline mode
static bool timelineStop = false;
static bool timelineRunning = false;
static pthread_t timelineTid;

static int appdaemon_register_exit(appdaemon_exit_func f)
{
    if (appdaemon_num_exit_funcs < APPDAEMON_MAX_EXIT_FUNCS)
    {
        appdaemon_exit_funcs[appdaemon_num_exit_funcs] = f;
        appdaemon_num_exit_funcs++;
    }
}

static void prepare_ldpreload()
{
    char *ldpreload = getenv("LD_PRELOAD");
    if (!ldpreload)
        return;

    bstring bldpre = bfromcstr(ldpreload);
    bstring new_bldpre = bfromcstr("");
    struct bstrList *liblist = bsplit(bldpre, ':');
    for (int i = 0; i < liblist->qty; i++)
    {
        if (binstr(liblist->entry[i], 0, &daemon_name) == BSTR_ERR)
        {
            bconcat(new_bldpre, liblist->entry[i]);
            bconchar(new_bldpre, ':');
        }
    }
    setenv("LD_PRELOAD", bdata(new_bldpre), 1);
    bstrListDestroy(liblist);
    bdestroy(new_bldpre);
    bdestroy(bldpre);
}

static int parse_gpustr(char* gpuStr, int* numGpus, int** gpuIds)
{
    // Create bstring
    bstring bGpuStr = bfromcstr(gpuStr);
    int (*ownatoi)(const char*) = atoi;
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
        tmpGpuIds[i] = ownatoi(bdata(gpuTokens->entry[i]));
    }

    // Copy data
    *numGpus = tmpNumGpus;
    *gpuIds = tmpGpuIds;

    // Destroy bstring
    bdestroy(bGpuStr);
    bstrListDestroy(gpuTokens);

    return 0;
}

typedef struct {
    int numDevices;
    int *devices;
    int numGroups;
    int *groups;
    double (*getTime)(int group);
    int (*numEvents)(int group);
    double (*getResult)(int gpu, int group, int event);
} appdaemon_output_data;

static int appdaemon_write_output_file(const char* markerfile, appdaemon_output_data* data) {

    /* MarkerAPI File format
     * 1 numberOfGPUs numberOfRegions numberOfGpuGroups
     * 2 regionID:regionTag0
     * 3 regionID:regionTag1
     * 4 regionID groupID gpuID callCount timeActive numEvents countersvalues(space separated)
     * 5 regionID groupID gpuID callCount timeActive numEvents countersvalues(space separated)
    */
    /* Here we use it to hand over the results to likwid-perfctr */

    // Open file in write mode
    FILE* file = fopen(markerfile,"w");
    if (file == NULL)
    {
        int ret = errno;
        fprintf(stderr, "Cannot open file %s\n", markerfile);
        fprintf(stderr, "%s", strerror(errno));
        return -ret;
    }
    fprintf(file,"%d 1 %d\n", data->numDevices, data->numGroups);
    int regionId = 0;
    for (int i = 0; i < data->numGroups; i++) {
        fprintf(file, "%d:appdaemon-%d\n", regionId, data->groups[i]);
    }
    for (int i = 0; i < data->numGroups; i++) {
        int groupId = data->groups[i];
        int numEvents = data->numEvents(groupId);
        double time = data->getTime(groupId);
        for (int j = 0; j < data->numDevices; j++) {
            fprintf(file, "%d %d %d %u %e %d ", regionId, groupId, data->devices[j], 1, time, numEvents);
            for (int k = 0; k < numEvents; k++) {
                fprintf(file, "%e ", data->getResult(groupId, k, j));
            }
            fprintf(file, "\n");
        }
    }
    fflush(file);
    fclose(file);
    return 0;
}


/*
Nvmon
*/
#ifdef LIKWID_NVMON
static int  nvmon_initialized = 0;
static int* nvmon_gpulist = NULL;
static int  nvmon_numgpus = 0;
static int* nvmon_gids = NULL;
static int  nvmon_numgids = 0;

static int appdaemon_setup_nvmon(char* gpuStr, char* eventStr)
{
    int ret = 0;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Nvmon GPU string: %s", gpuStr);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Nvmon Event string: %s", eventStr);

    // Parse gpu string
    ret = parse_gpustr(gpuStr, &nvmon_numgpus, &nvmon_gpulist);
    if (ret < 0)
    {
        ERROR_PRINT("Failed to get nvmon gpulist from '%s'", gpuStr);
        goto appdaemon_setup_nvmon_cleanup;
    }

    // Parse event string
    bstring bev = bfromcstr(eventStr);
    struct bstrList* nvmon_eventlist = bsplit(bev, '|');
    bdestroy(bev);
    nvmon_gids = malloc(nvmon_eventlist->qty * sizeof(int));
    if (!nvmon_gids)
    {
        ERROR_PRINT("Failed to allocate space for nvmon group IDs");
        goto appdaemon_setup_nvmon_cleanup;
    }

    // Init nvmon
    ret = nvmon_init(nvmon_numgpus, nvmon_gpulist);
    if (ret < 0)
    {
        ERROR_PRINT("Failed to initialize nvmon");
        goto appdaemon_setup_nvmon_cleanup;
    }
    nvmon_initialized = 1;

    // Add event sets
    for (int i = 0; i < nvmon_eventlist->qty; i++)
    {
        ret = nvmon_addEventSet(bdata(nvmon_eventlist->entry[i]));
        if (ret < 0)
        {
            ERROR_PRINT("Failed to add nvmon group: %s", bdata(nvmon_eventlist->entry[i]));
            continue;
        }
        nvmon_gids[nvmon_numgids++] = ret;
    }
    if (nvmon_numgids == 0)
    {
        ERROR_PRINT("Failed to add any events to nvmon");
        goto appdaemon_setup_nvmon_cleanup;
    }

    // Setup counters
    ret = nvmon_setupCounters(nvmon_gids[0]);
    if (ret < 0)
    {
        ERROR_PRINT("Failed to setup nvmon");
        goto appdaemon_setup_nvmon_cleanup;
    }

    // Start counters
    ret = nvmon_startCounters();
    if (ret < 0)
    {
        ERROR_PRINT("Failed to start nvmon");
        goto appdaemon_setup_nvmon_cleanup;
    }
    return 0;
appdaemon_setup_nvmon_cleanup:
    if (nvmon_initialized)
    {
        nvmon_finalize();
        nvmon_initialized = 0;
    }
    if (nvmon_gids)
    {
        free(nvmon_gids);
        nvmon_gids = NULL;
        nvmon_numgids = 0;
    }
    if (nvmon_eventlist)
    {
        bstrListDestroy(nvmon_eventlist);
        nvmon_eventlist = NULL;
    }
    if (nvmon_gpulist)
    {
        free(nvmon_gpulist);
        nvmon_gpulist = NULL;
        nvmon_numgpus = 0;
    }
    return ret;
}

static void appdaemon_close_nvmon(void)
{
    // Stop counters
    int ret = nvmon_stopCounters();
    if (ret < 0)
    {
        ERROR_PRINT("Failed to stop nvmon");
    }

    // Print results
    if (getenv("LIKWID_NVMON_MARKER_FORMAT") == NULL)
    {
        for (int g = 0; g < nvmon_numgids; g++)
        {
            int gid = nvmon_gids[g];
            for (int i = 0; i < nvmon_getNumberOfEvents(gid); i++)
            {
                for (int j = 0; j < nvmon_numgpus; j++)
                {
                    fprintf(output_file, "Nvmon, %d, %f, %s, %f, %f\n", nvmon_gpulist[j], nvmon_getTimeOfGroup(gid), nvmon_getEventName(gid, i), nvmon_getResult(gid, i, j), nvmon_getLastResult(gid, i, j));
                }
            }
        }
        fflush(output_file);
    } else {
        appdaemon_output_data data = {
            .numDevices = nvmon_numgpus,
            .devices = nvmon_gpulist,
            .numGroups = nvmon_numgids,
            .groups = nvmon_gids,
            .getTime = nvmon_getTimeOfGroup,
            .getResult = nvmon_getResult,
            .numEvents = nvmon_getNumberOfEvents,
        };
        ret = appdaemon_write_output_file(getenv("LIKWID_NVMON_OUTPUTFILE"), &data);
        if (ret < 0) {
            ERROR_PRINT("Failed to write appdaemon data to %s", getenv("LIKWID_NVMON_OUTPUTFILE"));
        }
    }

    // Cleanup
    if (nvmon_initialized)
    {
        nvmon_finalize();
        nvmon_initialized = 0;
    }
    if (nvmon_gids)
    {
        free(nvmon_gids);
        nvmon_gids = NULL;
        nvmon_numgids = 0;
    }
    if (nvmon_gpulist)
    {
        free(nvmon_gpulist);
        nvmon_gpulist = NULL;
        nvmon_numgpus = 0;
    }
}

static void appdaemon_read_nvmon(void)
{
    // Read counters
    int ret = nvmon_readCounters();
    if (ret < 0)
    {
        fprintf(stderr, "Failed to read Nvmon counters\n");
        return;
    }

    // Print results
    for (int g = 0; g < nvmon_numgids; g++)
    {
        int gid = nvmon_gids[g];
        for (int i = 0; i < nvmon_getNumberOfEvents(gid); i++)
        {
            for (int j = 0; j < nvmon_numgpus; j++)
            {
                fprintf(output_file, "Nvmon, %d, %f, %s, %f, %f\n", nvmon_gpulist[j], nvmon_getTimeToLastReadOfGroup(gid), nvmon_getEventName(gid, i), nvmon_getResult(gid, i, j), nvmon_getLastResult(gid, i, j));
            }
        }
    }
}
#endif

/*
Rocmon
*/
#ifdef LIKWID_ROCMON
static bool rocmon_initialized = false;
static int *rocmon_gpuIds;
static size_t rocmon_numGpuIds;
static int rocmon_groupId = -1;
static char **rocmon_eventNames;
static size_t rocmon_numEventNames;
static char **rocmon_metricNames;
static char **rocmon_metricFormulas;
static size_t rocmon_numMetrics;

static int appdaemon_setup_rocmon(char* gpuStr, char* eventStr)
{
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "Rocmon GPU string: %s\n", gpuStr);
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "Rocmon Event string: %s\n", eventStr);

    int err = rocmon_markerInit();
    if (err < 0) {
        ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "rocmon_markerInit failed: %d", err);
        return err;
    }

    err = rocmon_markerGetGpuIds(&rocmon_gpuIds, &rocmon_numGpuIds);
    if (err < 0) {
        ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "rocmon_markerGetGpuIds failed: %d", err);
        rocmon_markerClose();
        return err;
    }

    int *groupIds = NULL;
    size_t numGroupIds = 0;
    err = rocmon_markerGetGroupIds(&groupIds, &numGroupIds);
    if (err < 0) {
        ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "rocmon_markerGetGroupIds failed: %d", err);
        rocmon_markerClose();
        return err;
    }

    if (numGroupIds != 1) {
        ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Must need exactly one performance group, got: %zu", numGroupIds);
        rocmon_markerClose();
        return -EINVAL;
    }

    rocmon_groupId = groupIds[0];
    free(groupIds);

    err = rocmon_markerGetGroupInfo(
            rocmon_groupId,
            &rocmon_eventNames,
            &rocmon_numEventNames,
            &rocmon_metricNames,
            &rocmon_metricFormulas,
            &rocmon_numMetrics
    );
    if (err < 0) {
        ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "rocmon_markerGetGroupInfo failed: %s", strerror(-err));
        rocmon_markerClose();
        return -EINVAL;
    }

    rocmon_initialized = true;

    rocmon_markerStartRegion("main");
    return 0;
}

static void appdaemon_print_results_rocmon(void)
{
    for (size_t i = 0; i < rocmon_numGpuIds; i++) {
        // Using group = 0 here is kind of cheating, but we know it'll be always a single group 0.
        const int group = 0;

        double *counters = NULL;
        size_t numCounters = 0;
        double *metrics = NULL;
        size_t numMetrics = 0;

        int err = rocmon_markerGetRegionCounters("main", group, rocmon_gpuIds[i], &numCounters, &counters, &numMetrics, &metrics);
        if (err < 0) {
            ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "rocmon_markerGetRegionCounters failed: %d", err);
            continue;
        }

        size_t execCount = 0;
        double execTime = 0.0;
        err = rocmon_markerGetRegionStats("main", group, &execCount, &execTime);
        if (err < 0) {
            ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "rocmon_markerGetRegionStats failed: %d", err);
            goto cleanup_continue;
        }

        assert(numCounters == rocmon_numEventNames);

        for (size_t c = 0; c < numCounters; c++) {
            fprintf(output_file,
                    "ROCMON (counter), %d %f %s %f\n",
                    rocmon_gpuIds[i],
                    execTime,
                    rocmon_eventNames[c],
                    counters[c]
                    );
        }

        assert(numMetrics == rocmon_numMetrics);

        for (size_t m = 0; m < numMetrics; m++) {
            fprintf(output_file,
                    "ROCMON (metric), %d %f %s %f\n",
                    rocmon_gpuIds[i],
                    execTime,
                    rocmon_metricNames[m],
                    metrics[m]
                    );
        }

cleanup_continue:
        free(counters);
        free(metrics);
    }
}

static void appdaemon_close_rocmon(void)
{
    rocmon_markerStopRegion("main");

    if (getenv("LIKWID_ROCMON_MARKER_FORMAT") == NULL) {
        appdaemon_print_results_rocmon();
    }

    // Cleanup
    if (rocmon_initialized)
    {
        rocmon_markerClose();

        free(rocmon_gpuIds);
        rocmon_gpuIds = NULL;
        rocmon_numGpuIds = 0;

        for (size_t i = 0; i < rocmon_numEventNames; i++) {
            free(rocmon_eventNames[i]);
        }

        free(rocmon_eventNames);
        rocmon_eventNames = NULL;
        rocmon_numEventNames = 0;

        for (size_t i = 0; i < rocmon_numMetrics; i++) {
            free(rocmon_metricNames[i]);
            free(rocmon_metricFormulas[i]);
        }

        free(rocmon_metricNames);
        free(rocmon_metricFormulas);
        rocmon_metricNames = NULL;
        rocmon_metricFormulas = NULL;
        rocmon_numMetrics = 0;

        rocmon_initialized = 0;
    }
}

static void appdaemon_read_rocmon(void)
{
    int err = rocmon_markerStopRegion("main");
    if (err < 0) {
        ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "rocmon_markerStopRegion failed: %d", err);
        return;
    }

    appdaemon_print_results_rocmon();

    err = rocmon_markerStartRegion("main");
    if (err < 0)
        ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "rocmon_markerStartRegion failed: %d", err);
}
#endif



/*
Timeline mode
*/
static void* appdaemon_timeline_main(void* arg)
{
    int target_delay_ms = *((int*)arg);

#ifdef LIKWID_NVMON
    char* nvEventStr = getenv("LIKWID_NVMON_EVENTS");
    char* nvGpuStr = getenv("LIKWID_NVMON_GPUS");
    if (nvEventStr && nvGpuStr)
    {
        char *nvVerbosity = getenv("LIKWID_NVMON_VERBOSITY");
        if (nvVerbosity) {
            int likwid_nvmon_verbosity = atoi(nvVerbosity);
            nvmon_setVerbosity(likwid_nvmon_verbosity);
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Setting verbosity to %d", likwid_nvmon_verbosity);
        }
        int ret = appdaemon_setup_nvmon(nvGpuStr, nvEventStr);
        if (ret < 0)
        {
            fprintf(stderr, "Failed to setup NVMON: %d\n", ret);
        }
        else
        {
            appdaemon_register_exit(appdaemon_close_nvmon);
        }
    }
    printf("NVMON initialized\n");
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "NVMON initialized");
#endif

#ifdef LIKWID_ROCMON
    char* rocmonEventStr = getenv("LIKWID_ROCMON_EVENTS");
    char* rocmonGpuStr = getenv("LIKWID_ROCMON_GPUS");
    if (rocmonEventStr && rocmonGpuStr)
    {
        int ret = appdaemon_setup_rocmon(rocmonGpuStr, rocmonEventStr);
        if (ret < 0)
        {
            fprintf(stderr, "Failed to setup ROCMON: %d\n", ret);
        }
        else
        {
            appdaemon_register_exit(appdaemon_close_rocmon);
        }
    }
#endif

    while (true)
    {
        printf("Thread sleeps for %d ms\n", target_delay_ms);
        usleep(target_delay_ms / 1000);

        if (__atomic_load_n(&timelineStop, __ATOMIC_ACQUIRE))
            break;

        printf("Thread Reads\n");
#ifdef LIKWID_NVMON
        appdaemon_read_nvmon();
#endif
#ifdef LIKWID_ROCMON
        appdaemon_read_rocmon();
#endif
    }

#ifdef LIKWID_NVMON
    appdaemon_close_nvmon();
#endif
#ifdef LIKWID_ROCMON
    appdaemon_close_rocmon();
#endif
}


/*
Main
*/
static void main_wrapper_prolog()
{
    // Why do we lock and unlock the memory?
    mlockall(MCL_CURRENT);
    munlockall();

    prepare_ldpreload();

    // Get timeline mode info
    char *timelineStr = getenv("LIKWID_INTERVAL");
    int timelineInterval = -1; // in ms
    int gotTimelineInterval = 0;
    if (timelineStr != NULL)
    {
        timelineInterval = atoi(timelineStr);
        gotTimelineInterval = 1;
    }
    if (gotTimelineInterval && timelineInterval <= 0)
    {
        fprintf(stderr, "Timeline interval (LIKWID_INTERVAL) must be non-zero\n");
        return;
    }

    // Open output file
    char *outputFilename = getenv("LIKWID_OUTPUTFILE");
    if (!outputFilename)
        output_file = stderr;
    else
        output_file = fopen(outputFilename,"w");

    if (output_file == NULL)
    {
        fprintf(stderr, "Cannot open file (LIWKID_OUTPUTFILE) %s\n", outputFilename);
        perror("fopen");
        return;
    }
    if ((getenv("LIKWID_NVMON_MARKER_FORMAT") == NULL) && (getenv("LIKWID_ROCMON_MARKER_FORMAT") == NULL))
    {
        fprintf(output_file, "Backend, GPU, Time, Event, Full Value, Last Value\n");
    }

    // Start timeline thread
    if (timelineInterval > 0)
    {
        //printf("Start thread with interval %d\n", timelineInterval);
        int ret = pthread_create(&timelineTid, NULL, &appdaemon_timeline_main, &timelineInterval);
        if (ret != 0)
            fprintf(stderr, "Failed to create timeline thread: %s\n", strerror(ret));
        else
            timelineRunning = true;
    } else {
#ifdef LIKWID_NVMON
        char* nvEventStr = getenv("LIKWID_NVMON_EVENTS");
        char* nvGpuStr = getenv("LIKWID_NVMON_GPUS");
        if (nvEventStr && nvGpuStr)
        {
            char *nvVerbosity = getenv("LIKWID_NVMON_VERBOSITY");
            if (nvVerbosity)
            {
                int likwid_nvmon_verbosity = atoi(nvVerbosity);
                nvmon_setVerbosity(likwid_nvmon_verbosity);
                GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Setting verbosity to %d", likwid_nvmon_verbosity);
            }
            int ret = appdaemon_setup_nvmon(nvGpuStr, nvEventStr);
            if (ret == 0)
            {
                appdaemon_register_exit(appdaemon_close_nvmon);
            }
        }
#endif

#ifdef LIKWID_ROCMON
        char* rocmonEventStr = getenv("LIKWID_ROCMON_EVENTS");
        char* rocmonGpuStr = getenv("LIKWID_ROCMON_GPUS");
        if (rocmonEventStr && rocmonGpuStr)
        {
            char *rocmomVerbosity = getenv("LIKWID_ROCMON_VERBOSITY");
            if (rocmomVerbosity)
            {
                rocmon_setVerbosity(atoi(rocmomVerbosity));
            }
            int ret = appdaemon_setup_rocmon(rocmonGpuStr, rocmonEventStr);
            if (ret == 0)
            {
                appdaemon_register_exit(appdaemon_close_rocmon);
            }
        }
#endif
    }
}

static void main_wrapper_epilog()
{
    // Stop timeline thread (if running)
    if (timelineRunning)
    {
        __atomic_store_n(&timelineStop, true, __ATOMIC_RELEASE);
        pthread_join(timelineTid, NULL);
    }

    for (int i = 0; i < appdaemon_num_exit_funcs; i++)
        appdaemon_exit_funcs[i]();

    if (output_file && output_file != stderr)
        fclose(output_file);
}

static int (*main_original)(int, char **, char **);

static int main_wrapper(int argc, char **argv, char **envp)
{
    main_wrapper_prolog();

    if (!main_original)
    {
        fprintf(stderr, "Unable to find original main function");
        abort();
    }

    const int main_return = main_original(argc, argv, envp);

    main_wrapper_epilog();

    return main_return;
}

__attribute__((visibility("default")))
int __libc_start_main(int (*main) (int,char **,char **),
              int argc,char **ubp_av,
              void (*init) (void),
              void (*fini)(void),
              void (*rtld_fini)(void),
              void (*stack_end))
{
    int (*original__libc_start_main)(int (*main) (int,char **,char **),
                    int argc,char **ubp_av,
                    void (*init) (void),
                    void (*fini)(void),
                    void (*rtld_fini)(void),
                    void (*stack_end));

    // Save program's original main function, so we can call it later.
    main_original = main;

    original__libc_start_main = dlsym(RTLD_NEXT, "__libc_start_main");

    // Execute original __libc_start_main with our main_wrapper instead.
    return original__libc_start_main(main_wrapper, argc, ubp_av,
                     init, fini, rtld_fini, stack_end);
}

