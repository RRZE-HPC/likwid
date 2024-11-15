/*
 * =======================================================================================
 *
 *      Filename:  appDaemon.c
 *
 *      Description:  Implementation a interface library to hook into applications
 *                    using the GOTCHA library
 *
 *      Version:   5.4.0
 *      Released:  15.11.2024
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2024 RRZE, University Erlangen-Nuremberg
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


#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

#include <likwid.h>
#include <error.h>

typedef void(*appdaemon_exit_func)(void);
#define APPDAEMON_MAX_EXIT_FUNCS 2
static appdaemon_exit_func appdaemon_exit_funcs[APPDAEMON_MAX_EXIT_FUNCS];
static int appdaemon_num_exit_funcs = 0;

static struct tagbstring daemon_name = bsStatic("likwid-appDaemon.so");
static FILE* output_file = NULL;

// Timeline mode
static int stopIssued = 0;
static pthread_mutex_t stopMutex;

static int appdaemon_register_exit(appdaemon_exit_func f)
{
    if (appdaemon_num_exit_funcs < APPDAEMON_MAX_EXIT_FUNCS)
    {
        appdaemon_exit_funcs[appdaemon_num_exit_funcs] = f;
        appdaemon_num_exit_funcs++;
    }
}

static void after_main()
{
    // Stop timeline thread (if running)
    pthread_mutex_lock(&stopMutex);
    stopIssued = 1;
    pthread_mutex_unlock(&stopMutex);

    for (int i = 0; i < appdaemon_num_exit_funcs; i++)
    {
        appdaemon_exit_funcs[i]();
    }

    if (output_file)
    {
        fclose(output_file);
    }
}

static void prepare_ldpreload()
{
    int (*mysetenv)(const char *name, const char *value, int overwrite) = setenv;
    char* ldpreload = getenv("LD_PRELOAD");
    if (ldpreload)
    {
        printf("Old LD_PRELOAD=%s\n", ldpreload);
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
        printf("New LD_PRELOAD=%s\n", bdata(new_bldpre));
        mysetenv("LD_PRELOAD", bdata(new_bldpre), 1);
        bstrListDestroy(liblist);
        bdestroy(new_bldpre);
        bdestroy(bldpre);
    }
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
    printf("Nvmon GPU string: %s\n", gpuStr);
    printf("Nvmon Event string: %s\n", eventStr);

    // Parse gpu string
    ret = parse_gpustr(gpuStr, &nvmon_numgpus, &nvmon_gpulist);
    if (ret < 0)
    {
        ERROR_PRINT(Failed to get nvmon gpulist from '%s', gpuStr);
        goto appdaemon_setup_nvmon_cleanup;
    }

    // Parse event string
    bstring bev = bfromcstr(eventStr);
    struct bstrList* nvmon_eventlist = bsplit(bev, '|');
    bdestroy(bev);
    nvmon_gids = malloc(nvmon_eventlist->qty * sizeof(int));
    if (!nvmon_gids)
    {
        ERROR_PRINT(Failed to allocate space for nvmon group IDs);
        goto appdaemon_setup_nvmon_cleanup;
    }

    // Init nvmon
    ret = nvmon_init(nvmon_numgpus, nvmon_gpulist);
    if (ret < 0)
    {
        ERROR_PRINT(Failed to initialize nvmon);
        goto appdaemon_setup_nvmon_cleanup;
    }
    nvmon_initialized = 1;

    // Add event sets
    for (int i = 0; i < nvmon_eventlist->qty; i++)
    {
        ret = nvmon_addEventSet(bdata(nvmon_eventlist->entry[i]));
        if (ret < 0)
        {
            ERROR_PRINT(Failed to add nvmon group: %s, bdata(nvmon_eventlist->entry[i]));
            continue;
        }
        nvmon_gids[nvmon_numgids++] = ret;
    }
    if (nvmon_numgids == 0)
    {
        ERROR_PRINT(Failed to add any events to nvmon);
        goto appdaemon_setup_nvmon_cleanup;
    }

    // Setup counters
    ret = nvmon_setupCounters(nvmon_gids[0]);
    if (ret < 0)
    {
        ERROR_PRINT(Failed to setup nvmon);
        goto appdaemon_setup_nvmon_cleanup;
    }

    // Start counters
    ret = nvmon_startCounters();
    if (ret < 0)
    {
        ERROR_PRINT(Failed to start nvmon);
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
        ERROR_PRINT(Failed to stop nvmon);
    }

    // Print results
    for (int g = 0; g < nvmon_numgids; g++)
    {
        int gid = nvmon_gids[g];
        for (int i = 0; i < nvmon_getNumberOfEvents(gid); i++)
        {
            for (int j = 0; j < nvmon_numgpus; j++)
            {
                fprintf(output_file, "Nvmon, %d, %f, %s, %f, %f\n", nvmon_gpulist[j], nvmon_getTimeOfGroup(nvmon_gpulist[j]), nvmon_getEventName(gid, i), nvmon_getResult(gid, i, j), nvmon_getLastResult(gid, i, j));
            }
        }
    }
    fflush(output_file);

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
                fprintf(output_file, "Nvmon, %d, %f, %s, %f, %f\n", nvmon_gpulist[j], nvmon_getTimeToLastReadOfGroup(nvmon_gpulist[j]), nvmon_getEventName(gid, i), nvmon_getResult(gid, i, j), nvmon_getLastResult(gid, i, j));
            }
        }
    }
}
#endif

/*
Rocmon
*/
#ifdef LIKWID_ROCMON
static int  rocmon_initialized = 0;
static int* rocmon_gpulist = NULL;
static int  rocmon_numgpus = 0;
static int* rocmon_gids = NULL;
static int  rocmon_numgids = 0;

static int appdaemon_setup_rocmon(char* gpuStr, char* eventStr)
{
    int ret = 0;
    printf("Rocmon GPU string: %s\n", gpuStr);
    printf("Rocmon Event string: %s\n", eventStr);

    // Parse gpu string
    ret = parse_gpustr(gpuStr, &rocmon_numgpus, &rocmon_gpulist);
    if (ret < 0)
    {
        ERROR_PRINT(Failed to get rocmon gpulist from '%s', gpuStr);
        goto appdaemon_setup_rocmon_cleanup;
    }

    // Parse event string
    bstring bev = bfromcstr(eventStr);
    struct bstrList* rocmon_eventlist = bsplit(bev, '|'); // TODO: multiple event sets not supported
    bdestroy(bev);
    rocmon_gids = malloc(rocmon_eventlist->qty * sizeof(int));
    if (!rocmon_gids)
    {
        ERROR_PRINT(Failed to allocate space for rocmon group IDs);
        goto appdaemon_setup_rocmon_cleanup;
    }

    // Init rocmon
    ret = rocmon_init(rocmon_numgpus, rocmon_gpulist);
    if (ret < 0)
    {
        ERROR_PRINT(Failed to initialize rocmon);
        goto appdaemon_setup_rocmon_cleanup;
    }
    rocmon_initialized = 1;

    // Add event sets
    for (int i = 0; i < rocmon_eventlist->qty; i++)
    {
        ret = rocmon_addEventSet(bdata(rocmon_eventlist->entry[i]), &rocmon_gids[rocmon_numgids++]);
        if (ret < 0)
        {
            ERROR_PRINT(Failed to add rocmon group: %s, bdata(rocmon_eventlist->entry[i]));
        }
    }
    if (rocmon_numgids == 0)
    {
        ERROR_PRINT(Failed to add any events to rocmon);
        goto appdaemon_setup_rocmon_cleanup;
    }

    // Setup counters
    ret = rocmon_setupCounters(rocmon_gids[0]);
    if (ret < 0)
    {
        ERROR_PRINT(Failed to setup rocmon);
        goto appdaemon_setup_rocmon_cleanup;
    }

    // Start counters
    ret = rocmon_startCounters();
    if (ret < 0)
    {
        ERROR_PRINT(Failed to start rocmon);
        goto appdaemon_setup_rocmon_cleanup;
    }
    return 0;
appdaemon_setup_rocmon_cleanup:
    if (rocmon_initialized)
    {
        rocmon_finalize();
        rocmon_initialized = 0;
    }
    if (rocmon_gids)
    {
        free(rocmon_gids);
        rocmon_gids = NULL;
        rocmon_numgids = 0;
    }
    if (rocmon_eventlist)
    {
        bstrListDestroy(rocmon_eventlist);
        rocmon_eventlist = NULL;
    }
    if (rocmon_gpulist)
    {
        free(rocmon_gpulist);
        rocmon_gpulist = NULL;
        rocmon_numgpus = 0;
    }
    return ret;
}

static void appdaemon_close_rocmon(void)
{
    // Stop counters
    int ret = rocmon_stopCounters();
    if (ret < 0)
    {
        ERROR_PRINT(Failed to stop rocmon);
    }

    // Print results
    for (int g = 0; g < rocmon_numgids; g++)
    {
        int gid = rocmon_gids[g];
        for (int i = 0; i < rocmon_getNumberOfEvents(gid); i++)
        {
            for (int j = 0; j < rocmon_numgpus; j++)
            {
                fprintf(output_file, "Rocmon, %d, %f, %s, %f, %f\n", rocmon_gpulist[j], rocmon_getTimeOfGroup(rocmon_gpulist[j]), rocmon_getEventName(gid, i), rocmon_getResult(j, gid, i), rocmon_getLastResult(j, gid, i));
            }
        }
    }

    // Cleanup
    if (rocmon_initialized)
    {
        rocmon_finalize();
        rocmon_initialized = 0;
    }
    if (rocmon_gids)
    {
        free(rocmon_gids);
        rocmon_gids = NULL;
        rocmon_numgids = 0;
    }
    if (rocmon_gpulist)
    {
        free(rocmon_gpulist);
        rocmon_gpulist = NULL;
        rocmon_numgpus = 0;
    }
}

static void appdaemon_read_rocmon(void)
{
    // Read counters
    int ret = rocmon_readCounters();
    if (ret < 0)
    {
        fprintf(stderr, "Failed to read Rocmon counters\n");
        return;
    }

    // Print results
    for (int g = 0; g < rocmon_numgids; g++)
    {
        int gid = rocmon_gids[g];
        for (int i = 0; i < rocmon_getNumberOfEvents(gid); i++)
        {
            for (int j = 0; j < rocmon_numgpus; j++)
            {
                fprintf(output_file, "Rocmon, %d, %f, %s, %f, %f\n", rocmon_gpulist[j], rocmon_getTimeToLastReadOfGroup(rocmon_gpulist[j]), rocmon_getEventName(gid, i), rocmon_getResult(j, gid, i), rocmon_getLastResult(j, gid, i));
            }
        }
    }
}
#endif


/*
Timeline mode
*/
static void* appdaemon_timeline_main(void* arg)
{
    int stop = 0;
    int target_delay_ms = *((int*)arg);
    ;

    while (1)
    {
        usleep(target_delay_ms * 1E3);

        // Check stop status
        pthread_mutex_lock(&stopMutex);
        stop = stopIssued;
        pthread_mutex_unlock(&stopMutex);
        if (stop > 0) break;
        
#ifdef LIKWID_NVMON
        appdaemon_read_nvmon();
#endif
#ifdef LIKWID_ROCMON
        appdaemon_read_rocmon();
#endif
    }
}


/*
Main
*/
__attribute__((visibility("default")))
int __libc_start_main(int (*main) (int,char **,char **),
              int argc,char **ubp_av,
              void (*init) (void),
              void (*fini)(void),
              void (*rtld_fini)(void),
              void (*stack_end)) {
    int ret = 0;
    int (*original__libc_start_main)(int (*main) (int,char **,char **),
                    int argc,char **ubp_av,
                    void (*init) (void),
                    void (*fini)(void),
                    void (*rtld_fini)(void),
                    void (*stack_end));

    mlockall(MCL_CURRENT);
    munlockall();
    atexit(after_main);


    original__libc_start_main = dlsym(RTLD_NEXT, "__libc_start_main");

    prepare_ldpreload();

    // Get timeline mode info
    char* timelineStr = getenv("LIKWID_INTERVAL");
    int timelineInterval = -1; // in ms
    if (timelineStr != NULL)
    {
        timelineInterval = atoi(timelineStr);
    }
    if (timelineInterval == 0)
    {
        fprintf(stderr, "Invalid timeline interval\n");
        return -1;
    }

    // Open output file
    char* outputFilename = getenv("LIKWID_OUTPUTFILE");
    if (outputFilename == NULL)
    {
        output_file = stderr;
    } else {
        output_file = fopen(outputFilename,"w");
    }

    if (output_file == NULL)
    {
        fprintf(stderr, "Cannot open file %s\n", outputFilename);
        fprintf(stderr, "%s", strerror(errno));
        return -1;
    }
    fprintf(output_file, "Backend, GPU, Time, Event, Full Value, Last Value\n");

#ifdef LIKWID_NVMON
    char* nvEventStr = getenv("LIKWID_NVMON_EVENTS");
    char* nvGpuStr = getenv("LIKWID_NVMON_GPUS");
    if (nvEventStr && nvGpuStr)
    {
        ret = appdaemon_setup_nvmon(nvGpuStr, nvEventStr);
        if (!ret)
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
        ret = appdaemon_setup_rocmon(rocmonGpuStr, rocmonEventStr);
        if (!ret)
        {
            appdaemon_register_exit(appdaemon_close_rocmon);
        }
    }
#endif

    // Start timeline thread
    if (timelineInterval >= 0)
    {
        pthread_t tid;
        ret = pthread_create(&tid, NULL, &appdaemon_timeline_main, &timelineInterval);
        if (ret < 0)
        {
            fprintf(stderr, "Failed to create timeline thread\n");
            return -1;
        }
    }

    return original__libc_start_main(main,argc,ubp_av,
                     init,fini,rtld_fini,stack_end);
}

