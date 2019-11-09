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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <syscall.h>

#include <likwid.h>
#include <lock.h>
#include <bstrlib.h>
#include <error.h>
#include <map.h>
#include <libnvctr_types.h>

#define gettid() syscall(SYS_gettid)

static int likwid_gpu_init = 0;
static int* gpu_groups = NULL;
static int activeGpuGroup = -1;
static int numberOfGpuGroups = 0;
static int* id2Gpu;
static int num_gpus = 0;
static pid_t main_tid = -1;
static Map_t* gpu_maps = NULL;
/*static int use_cpu = -1;*/



void
likwid_gpuMarkerInit(void)
{
    int i = 0;
    int setgpuinit = 0;
    int gpuverbosity = 0;
    char* eventStr = getenv("LIKWID_GEVENTS");
    char* gpuStr = getenv("LIKWID_GPUS");
    char* gpuFileStr = getenv("LIKWID_GPUFILEPATH");
/*    char* cpu4gpuStr = getenv("LIKWID_CPU4GPUS");*/
    bstring bGpuStr;
    bstring bGeventStr;
    int (*ownatoi)(const char*);
    ownatoi = &atoi;

    if ((eventStr != NULL) && (gpuStr != NULL) && (gpuFileStr != NULL) && likwid_gpu_init == 0)
    {
        setgpuinit = 1;
    }
    else if (likwid_gpu_init == 0)
    {
        fprintf(stderr, "Running without GPU Marker API. Activate GPU Marker API with -m, -G and -W on commandline.\n");
        return;
    }
    else
    {
        return;
    }

    if (!lock_check())
    {
        fprintf(stderr,"Access to GPU performance counters is locked.\n");
        exit(EXIT_FAILURE);
    }

    timer_init();
    topology_gpu_init();
    if (getenv("LIKWID_DEBUG") != NULL)
    {
        nvmon_setVerbosity(ownatoi(getenv("LIKWID_DEBUG")));
        gpuverbosity = perfmon_verbosity;
    }
/*    if (cpu4gpuStr != NULL)*/
/*    {*/
/*        use_cpu = ownatoi(getenv("LIKWID_CPU4GPUS"))*/
/*    }*/

    main_tid = gettid();

    bGpuStr = bfromcstr(gpuStr);
    struct bstrList* gpuTokens = bsplit(bGpuStr,',');
    num_gpus = gpuTokens->qty;
    id2Gpu = malloc(num_gpus * sizeof(int));
    if (!id2Gpu)
    {
        fprintf(stderr,"Cannot allocate space for GPU list.\n");
        bdestroy(bGpuStr);
        bstrListDestroy(gpuTokens);
        exit(EXIT_FAILURE);
    }
    gpu_maps = malloc(num_gpus * sizeof(Map_t));
    if (!gpu_maps)
    {
        fprintf(stderr,"Cannot allocate space for results.\n");
        free(id2Gpu);
        bdestroy(bGpuStr);
        bstrListDestroy(gpuTokens);
        exit(EXIT_FAILURE);
    }
    for (i=0; i<num_gpus; i++)
    {
        id2Gpu[i] = ownatoi(bdata(gpuTokens->entry[i]));
    }
    bdestroy(bGpuStr);
    bstrListDestroy(gpuTokens);

    bGeventStr = bfromcstr(eventStr);
    struct bstrList* gEventStrings = bsplit(bGeventStr,'|');
    numberOfGpuGroups = gEventStrings->qty;
    gpu_groups = malloc(numberOfGpuGroups * sizeof(int));
    if (!gpu_groups)
    {
        fprintf(stderr,"Cannot allocate space for group handling.\n");
        bstrListDestroy(gEventStrings);
        free(id2Gpu);
        free(gpu_maps);
        bdestroy(bGeventStr);
        exit(EXIT_FAILURE);
    }

    i = nvmon_init(num_gpus, id2Gpu);
    if (i < 0)
    {
        fprintf(stderr,"Error init GPU Marker API.\n");
        free(id2Gpu);
        free(gpu_maps);
        free(gpu_groups);
        bstrListDestroy(gEventStrings);
        bdestroy(bGeventStr);
        exit(EXIT_FAILURE);
    }

    for (i=0; i<gEventStrings->qty; i++)
    {
        gpu_groups[i] = nvmon_addEventSet(bdata(gEventStrings->entry[i]));
    }
    bstrListDestroy(gEventStrings);
    bdestroy(bGeventStr);

    for (i=0; i<num_gpus; i++)
    {
        init_smap(&gpu_maps[i]);
    }
    activeGpuGroup = 0;

    i = nvmon_setupCounters(gpu_groups[activeGpuGroup]);
    if (i)
    {
        fprintf(stderr,"Error setting up GPU Marker API.\n");
        free(gpu_groups);
        gpu_groups = NULL;
        numberOfGpuGroups = 0;
        setgpuinit = 0;
    }
    i = nvmon_startCounters();
    if (i)
    {
        fprintf(stderr,"Error starting up GPU Marker API.\n");
        free(gpu_groups);
        gpu_groups = NULL;
        numberOfGpuGroups = 0;
        setgpuinit = 0;
    }
    if (setgpuinit)
    {
        likwid_gpu_init = 1;
    }
    else
    {
        nvmon_finalize();
    }
}

/* File format
 * 1 numberOfGPUs numberOfRegions numberOfGpuGroups
 * 2 regionID:regionTag0
 * 3 regionID:regionTag1
 * 4 regionID gpuID countersvalues(space separated)
 * 5 regionID gpuID countersvalues
 */
void
likwid_gpuMarkerClose(void)
{
    FILE *file = NULL;
    char* markerfile = NULL;
    int numberOfGPUs = 0;
    int numberOfRegions = 0;
    if (!likwid_gpu_init)
    {
        return;
    }
    if (gettid() != main_tid)
    {
        return;
    }
    nvmon_stopCounters();
    markerfile = getenv("LIKWID_GPUFILEPATH");
    if (markerfile == NULL)
    {
        fprintf(stderr,
                "Is the application executed with LIKWID wrapper? No file path for the GPU Marker API output defined.\n");
        return;
    }
    numberOfRegions = get_map_size(gpu_maps[0]);
    numberOfGPUs = nvmon_getNumberOfGPUs();
    if ((numberOfGPUs == 0)||(numberOfRegions == 0))
    {
        fprintf(stderr, "No GPUs or regions defined in hash table\n");
        return;
    }

    file = fopen(markerfile,"w");
    if (file != NULL)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP,Creating GPU Marker file %s with %d regions %d groups and %d GPUs, markerfile, numberOfRegions, numberOfGpuGroups, numberOfGPUs);
        bstring thread_regs_grps = bformat("%d %d %d", numberOfGPUs, numberOfRegions, numberOfGpuGroups);
        fprintf(file,"%s\n", bdata(thread_regs_grps));
        DEBUG_PRINT(DEBUGLEV_DEVELOP, %s, bdata(thread_regs_grps));
        bdestroy(thread_regs_grps);

        for (int j = 0; j < numberOfRegions; j++)
        {
            LikwidGpuResults *results = NULL;
            int ret = get_smap_by_idx(gpu_maps[0], j, (void**)&results);
            if (ret == 0)
            {
                bstring tmp = bformat("%d:%s", j, bdata(results->label));
                fprintf(file,"%s\n", bdata(tmp));
                DEBUG_PRINT(DEBUGLEV_DEVELOP, %s, bdata(tmp));
                bdestroy(tmp);
            }
        }

        for (int j = 0; j < numberOfRegions; j++)
        {

            for (int i = 0; i < numberOfGPUs; i++)
            {
                LikwidGpuResults *results = NULL;
                int ret = get_smap_by_idx(gpu_maps[i], j, (void**)&results);
                if (!ret)
                {
                    bstring l = bformat("%d %d %d %u %e %d ", j,
                                                              results->groupID,
                                                              id2Gpu[results->gpuID],
                                                              results->count,
                                                              results->time,
                                                              nvmon_getNumberOfEvents(results->groupID));
                    for (int k = 0; k < nvmon_getNumberOfEvents(results->groupID); k++)
                    {
                        bstring tmp = bformat("%e ", results->PMcounters[k]);
                        bconcat(l, tmp);
                        bdestroy(tmp);
                    }
                    fprintf(file,"%s\n", bdata(l));
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, %s, bdata(l));
                    bdestroy(l);
                }
                free(results);
            }
        }
        for (int i = 0; i < nvmon_getNumberOfGPUs(); i++)
        {
            destroy_smap(gpu_maps[i]);
        }
    }
    else
    {
        fprintf(stderr, "Cannot open file %s\n", markerfile);
        fprintf(stderr, "%s", strerror(errno));
    }

    //nvmon_finalize();
}


int
likwid_gpuMarkerRegisterRegion(const char* regionTag)
{
    if (!likwid_gpu_init)
    {
        return -EFAULT;
    }
    if (gettid() != main_tid)
    {
        return 0;
    }
    for (int i = 0; i < nvmon_getNumberOfGPUs(); i++)
    {
        LikwidGpuResults* res = malloc(sizeof(LikwidGpuResults));
        if (!res)
        {
            fprintf(stderr, "Failed to register region %s\n", regionTag);
        }
        res->time = 0;
        res->count = 0;
        res->gpuID = i;
        res->state = GPUMARKER_STATE_NEW;
        res->groupID = activeGpuGroup;
        res->label = bformat("%s-%d", regionTag, activeGpuGroup);
        res->nevents = nvmon_getNumberOfEvents(activeGpuGroup);
        add_smap(gpu_maps[i], bdata(res->label), res);
    }
}


int
likwid_gpuMarkerStartRegion(const char* regionTag)
{
    bstring tag;
    if (!likwid_gpu_init)
    {
        return -EFAULT;
    }
    if (activeGpuGroup < 0)
    {
        return -EFAULT;
    }
    if (gettid() != main_tid)
    {
        return 0;
    }

    nvmon_readCounters();
    tag = bformat("%s-%d", regionTag, activeGpuGroup);

    for (int i = 0; i < nvmon_getNumberOfGPUs(); i++)
    {
        LikwidGpuResults *results = NULL;
        int ret = get_smap_by_key(gpu_maps[i], bdata(tag), (void**)&results);
        if (ret < 0)
        {
            results = malloc(sizeof(LikwidGpuResults));
            if (!results)
            {
                fprintf(stderr, "Failed to register region %s\n", regionTag);
                return -EFAULT;
            }
            results->time = 0;
            results->count = 0;
            results->gpuID = i;
            results->state = GPUMARKER_STATE_NEW;
            results->groupID = activeGpuGroup;
            results->label = bstrcpy(tag);
            results->nevents = nvmon_getNumberOfEvents(activeGpuGroup);
            add_smap(gpu_maps[i], bdata(results->label), results);
            ret = 0;
        }
        if (ret == 0 && results->state == GPUMARKER_STATE_START)
        {
            fprintf(stderr, "WARN: Starting an already-started region %s\n", regionTag);
            return -EFAULT;
        }
        for (int j = 0; j < nvmon_getNumberOfEvents(activeGpuGroup); j++)
        {
            results->StartPMcounters[j] = nvmon_getLastResult(activeGpuGroup, j, i);
        }
        results->state = GPUMARKER_STATE_START;
        timer_start(&(results->startTime));
    }
    bdestroy(tag);
    return 0;
}

int
likwid_gpuMarkerStopRegion(const char* regionTag)
{
    bstring tag;
    if (!likwid_gpu_init)
    {
        return -EFAULT;
    }
    if (activeGpuGroup < 0)
    {
        return -EFAULT;
    }
    if (gettid() != main_tid)
    {
        return 0;
    }
    TimerData timestamp;
    timer_stop(&timestamp);

    nvmon_readCounters();
    tag = bformat("%s-%d", regionTag, activeGpuGroup);
    for (int i = 0; i < nvmon_getNumberOfGPUs(); i++)
    {
        LikwidGpuResults *results = NULL;
        int ret = get_smap_by_key(gpu_maps[i], bdata(tag), (void**)&results);
        if ((ret < 0) || (results->state != GPUMARKER_STATE_START))
        {
            fprintf(stderr, "WARN: Stopping an unknown/not-started region %s\n", regionTag);
            return -EFAULT;
        }
        results->startTime.stop.int64 = timestamp.stop.int64;
        results->time += timer_print(&(results->startTime));
        results->count++;
        for (int j = 0; j < nvmon_getNumberOfEvents(activeGpuGroup); j++)
        {
            double end = nvmon_getLastResult(activeGpuGroup, j, i);
            results->PMcounters[j] += end - results->StartPMcounters[j];
        }
        results->state = GPUMARKER_STATE_STOP;
    }
    bdestroy(tag);
    return 0;
}

void
likwid_gpuMarkerGetRegion(
        const char* regionTag,
        int* nr_gpus,
        int* nr_events,
        double** events,
        double** time,
        int **count)
{
    if (!likwid_gpu_init)
    {
        *nr_gpus = 0;
        *nr_events = 0;
        *time = NULL;
        *events = NULL;
        *count = NULL;
        return;
    }
    if (gettid() != main_tid)
    {
        return;
    }
}


int
likwid_gpuMarkerResetRegion(const char* regionTag)
{
    if (!likwid_gpu_init)
    {
        return -EFAULT;
    }
    if (gettid() != main_tid)
    {
        return 0;
    }
    bstring tag = bformat("%s-%d", regionTag, activeGpuGroup);
    for (int i = 0; i < nvmon_getNumberOfGPUs(); i++)
    {
        LikwidGpuResults *results = NULL;
        int ret = get_smap_by_key(gpu_maps[i], bdata(tag), (void**)&results);
        if ((ret < 0) || (results->state != GPUMARKER_STATE_START))
        {
            fprintf(stderr, "ERROR: Can only reset known/stopped regions\n");
            return -EFAULT;
        }
        memset(results->PMcounters, 0, nvmon_getNumberOfEvents(activeGpuGroup)*sizeof(double));
        results->count = 0;
        results->time = 0;
        timer_reset(&results->startTime);
    }
}

void
likwid_gpuMarkerNextGroup(void)
{
    if (!likwid_gpu_init)
    {
        return;
    }
    if (gettid() != main_tid)
    {
        return;
    }
    int next_group = (activeGpuGroup + 1) % numberOfGpuGroups;
    if (next_group != activeGpuGroup)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Switch from GPU group %d to group %d, activeGpuGroup, next_group);
        nvmon_switchActiveGroup(next_group);
    }
}
