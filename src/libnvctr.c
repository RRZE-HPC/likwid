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
 *      This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 *      This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 *      You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <syscall.h>

#include <bstrlib.h>
#include <error.h>
#include <libnvctr_types.h>
#include <likwid.h>
#include <lock.h>
#include <map.h>
#include <nvmon_nvml.h>
#include <nvmon_types.h>

#define gettid() syscall(SYS_gettid)

static int likwid_cuda_init = 0;
static int *cuda_groups = NULL;
static int activeCudaGroup = -1;
static int numberOfCudaGroups = 0;
static int *id2Cuda;
static int num_cuda_gpus = 0;
static pid_t main_tid = -1;
static Map_t *cuda_maps = NULL;
/*static int use_cpu = -1;*/

void nvmon_markerInit(void) {
  int i = 0;
  int setgpuinit = 0;
  int gpuverbosity = 0;
  char *eventStr = getenv("LIKWID_GEVENTS");
  char *gpuStr = getenv("LIKWID_GPUS");
  char *gpuFileStr = getenv("LIKWID_GPUFILEPATH");
  /*    char* cpu4gpuStr = getenv("LIKWID_CPU4GPUS");*/
  bstring bGpuStr;
  bstring bGeventStr;
  int (*ownatoi)(const char *);
  ownatoi = &atoi;

  if ((eventStr != NULL) && (gpuStr != NULL) && (gpuFileStr != NULL) &&
      likwid_cuda_init == 0) {
    setgpuinit = 1;
  } else if (likwid_cuda_init == 0) {
    fprintf(stderr, "Running without GPU Marker API. Activate GPU Marker API "
                    "with -m, -G and -W on commandline.\n");
    return;
  } else {
    return;
  }

  // if (!lock_check())
  // {
  //     fprintf(stderr,"Access to GPU performance counters is locked.\n");
  //     exit(EXIT_FAILURE);
  // }

  timer_init();
  topology_cuda_init();
  if (getenv("LIKWID_DEBUG") != NULL) {
    nvmon_setVerbosity(ownatoi(getenv("LIKWID_DEBUG")));
    gpuverbosity = perfmon_verbosity;
  }
  /*    if (cpu4gpuStr != NULL)*/
  /*    {*/
  /*        use_cpu = ownatoi(getenv("LIKWID_CPU4GPUS"))*/
  /*    }*/

  main_tid = gettid();

  bGpuStr = bfromcstr(gpuStr);
  struct bstrList *gpuTokens = bsplit(bGpuStr, ',');
  num_cuda_gpus = gpuTokens->qty;
  id2Cuda = malloc(num_cuda_gpus * sizeof(int));
  if (!id2Cuda) {
    fprintf(stderr, "Cannot allocate space for GPU list.\n");
    bdestroy(bGpuStr);
    bstrListDestroy(gpuTokens);
    return;
  }
  cuda_maps = malloc(num_cuda_gpus * sizeof(Map_t));
  if (!cuda_maps) {
    fprintf(stderr, "Cannot allocate space for results.\n");
    free(id2Cuda);
    bdestroy(bGpuStr);
    bstrListDestroy(gpuTokens);
    return;
  }
  for (i = 0; i < num_cuda_gpus; i++) {
    id2Cuda[i] = ownatoi(bdata(gpuTokens->entry[i]));
  }
  bdestroy(bGpuStr);
  bstrListDestroy(gpuTokens);

  bGeventStr = bfromcstr(eventStr);
  struct bstrList *gEventStrings = bsplit(bGeventStr, '|');
  numberOfCudaGroups = gEventStrings->qty;
  cuda_groups = malloc(numberOfCudaGroups * sizeof(int));
  if (!cuda_groups) {
    fprintf(stderr, "Cannot allocate space for group handling.\n");
    bstrListDestroy(gEventStrings);
    free(id2Cuda);
    free(cuda_maps);
    bdestroy(bGeventStr);
    return;
  }

  i = nvmon_init(num_cuda_gpus, id2Cuda);
  if (i < 0) {
    fprintf(stderr, "Error init GPU Marker API.\n");
    free(id2Cuda);
    free(cuda_maps);
    free(cuda_groups);
    bstrListDestroy(gEventStrings);
    bdestroy(bGeventStr);
    return;
  }

  for (i = 0; i < gEventStrings->qty; i++) {
    cuda_groups[i] = nvmon_addEventSet(bdata(gEventStrings->entry[i]));
  }
  bstrListDestroy(gEventStrings);
  bdestroy(bGeventStr);

  for (i = 0; i < num_cuda_gpus; i++) {
    init_smap(&cuda_maps[i]);
  }
  activeCudaGroup = 0;

  i = nvmon_setupCounters(cuda_groups[activeCudaGroup]);
  if (i) {
    fprintf(stderr, "Error setting up GPU Marker API.\n");
    free(cuda_groups);
    cuda_groups = NULL;
    numberOfCudaGroups = 0;
    setgpuinit = 0;
  }
  i = nvmon_startCounters();
  if (i) {
    fprintf(stderr, "Error starting up GPU Marker API.\n");
    free(cuda_groups);
    cuda_groups = NULL;
    numberOfCudaGroups = 0;
    setgpuinit = 0;
  }
  if (setgpuinit) {
    likwid_cuda_init = 1;
  } else {
    nvmon_finalize();
  }
}

/* File format
 * 1 numberOfGPUs numberOfRegions numberOfCudaGroups
 * 2 regionID:regionTag0
 * 3 regionID:regionTag1
 * 4 regionID gpuID countersvalues(space separated)
 * 5 regionID gpuID countersvalues
 */
void nvmon_markerClose(void) {
  FILE *file = NULL;
  char *markerfile = NULL;
  int numberOfGPUs = 0;
  int numberOfRegions = 0;
  if (!likwid_cuda_init) {
    return;
  }
  if (gettid() != main_tid) {
    return;
  }
  nvmon_stopCounters();
  markerfile = getenv("LIKWID_GPUFILEPATH");
  if (markerfile == NULL) {
    fprintf(stderr, "Is the application executed with LIKWID wrapper? No file "
                    "path for the GPU Marker API output defined.\n");
    return;
  }
  numberOfRegions = get_map_size(cuda_maps[0]);
  numberOfGPUs = nvmon_getNumberOfGPUs();
  if ((numberOfGPUs == 0) || (numberOfRegions == 0)) {
    fprintf(stderr, "No GPUs or regions defined in hash table\n");
    return;
  }

  file = fopen(markerfile, "w");
  if (file != NULL) {
    DEBUG_PRINT(DEBUGLEV_DEVELOP,
                Creating GPU Marker file % s with % d regions % d groups and
                    % d GPUs,
                markerfile, numberOfRegions, numberOfCudaGroups, numberOfGPUs);
    bstring thread_regs_grps =
        bformat("%d %d %d", numberOfGPUs, numberOfRegions, numberOfCudaGroups);
    fprintf(file, "%s\n", bdata(thread_regs_grps));
    DEBUG_PRINT(DEBUGLEV_DEVELOP, % s, bdata(thread_regs_grps));
    bdestroy(thread_regs_grps);

    for (int j = 0; j < numberOfRegions; j++) {
      LikwidGpuResults *results = NULL;
      int ret = get_smap_by_idx(cuda_maps[0], j, (void **)&results);
      if (ret == 0) {
        bstring tmp = bformat("%d:%s", j, bdata(results->label));
        fprintf(file, "%s\n", bdata(tmp));
        DEBUG_PRINT(DEBUGLEV_DEVELOP, % s, bdata(tmp));
        bdestroy(tmp);
      }
    }

    for (int j = 0; j < numberOfRegions; j++) {

      for (int i = 0; i < numberOfGPUs; i++) {
        LikwidGpuResults *results = NULL;
        int ret = get_smap_by_idx(cuda_maps[i], j, (void **)&results);
        if (!ret) {
          bstring l =
              bformat("%d %d %d %u %e %d ", j, results->groupID,
                      id2Cuda[results->gpuID], results->count, results->time,
                      nvmon_getNumberOfEvents(results->groupID));
          for (int k = 0; k < nvmon_getNumberOfEvents(results->groupID); k++) {
            bstring tmp = bformat("%e ", results->PMcounters[k]);
            bconcat(l, tmp);
            bdestroy(tmp);
          }
          fprintf(file, "%s\n", bdata(l));
          DEBUG_PRINT(DEBUGLEV_DEVELOP, % s, bdata(l));
          bdestroy(l);
        }
        free(results);
      }
    }
    for (int i = 0; i < nvmon_getNumberOfGPUs(); i++) {
      destroy_smap(cuda_maps[i]);
    }
  } else {
    fprintf(stderr, "Cannot open file %s\n", markerfile);
    fprintf(stderr, "%s", strerror(errno));
  }

  // nvmon_finalize();
}

int nvmon_markerWriteFile(const char* markerfile)
{
  FILE *file = NULL;
  int numberOfGPUs = 0;
  int numberOfRegions = 0;
  if (markerfile == NULL) {
    fprintf(stderr, "File can not be NULL.\n");
    return -EFAULT;
  }
  numberOfRegions = get_map_size(cuda_maps[0]);
  numberOfGPUs = nvmon_getNumberOfGPUs();
  if ((numberOfGPUs == 0) || (numberOfRegions == 0)) {
    fprintf(stderr, "No GPUs or regions defined in hash table\n");
    return -EFAULT;
  }

  file = fopen(markerfile, "w");
  if (file != NULL) {
    DEBUG_PRINT(DEBUGLEV_DEVELOP,
                Creating GPU Marker file % s with % d regions % d groups and
                    % d GPUs,
                markerfile, numberOfRegions, numberOfCudaGroups, numberOfGPUs);
    bstring thread_regs_grps =
        bformat("%d %d %d", numberOfGPUs, numberOfRegions, numberOfCudaGroups);
    fprintf(file, "%s\n", bdata(thread_regs_grps));
    DEBUG_PRINT(DEBUGLEV_DEVELOP, % s, bdata(thread_regs_grps));
    bdestroy(thread_regs_grps);

    for (int j = 0; j < numberOfRegions; j++) {
      LikwidGpuResults *results = NULL;
      int ret = get_smap_by_idx(cuda_maps[0], j, (void **)&results);
      if (ret == 0) {
        bstring tmp = bformat("%d:%s", j, bdata(results->label));
        fprintf(file, "%s\n", bdata(tmp));
        DEBUG_PRINT(DEBUGLEV_DEVELOP, % s, bdata(tmp));
        bdestroy(tmp);
      }
    }

    for (int j = 0; j < numberOfRegions; j++) {

      for (int i = 0; i < numberOfGPUs; i++) {
        LikwidGpuResults *results = NULL;
        int ret = get_smap_by_idx(cuda_maps[i], j, (void **)&results);
        if (!ret) {
          bstring l =
              bformat("%d %d %d %u %e %d ", j, results->groupID,
                      id2Cuda[results->gpuID], results->count, results->time,
                      nvmon_getNumberOfEvents(results->groupID));
          for (int k = 0; k < nvmon_getNumberOfEvents(results->groupID); k++) {
            bstring tmp = bformat("%e ", results->PMcounters[k]);
            bconcat(l, tmp);
            bdestroy(tmp);
          }
          fprintf(file, "%s\n", bdata(l));
          DEBUG_PRINT(DEBUGLEV_DEVELOP, % s, bdata(l));
          bdestroy(l);
        }
        free(results);
      }
    }
  } else {
    int err = errno;
    fprintf(stderr, "Cannot open file %s\n", markerfile);
    fprintf(stderr, "%s", strerror(err));
    return -err;
  }
  return 0;
}

int nvmon_markerRegisterRegion(const char *regionTag) {
  if (!likwid_cuda_init) {
    return -EFAULT;
  }
  if (gettid() != main_tid) {
    return 0;
  }
  for (int i = 0; i < nvmon_getNumberOfGPUs(); i++) {
    LikwidGpuResults *res = malloc(sizeof(LikwidGpuResults));
    if (!res) {
      fprintf(stderr, "Failed to register region %s\n", regionTag);
    }
    res->time = 0;
    res->count = 0;
    res->gpuID = i;
    res->state = GPUMARKER_STATE_NEW;
    res->groupID = activeCudaGroup;
    res->label = bformat("%s-%d", regionTag, activeCudaGroup);
    res->nevents = nvmon_getNumberOfEvents(activeCudaGroup);
    for (int j = 0; j < res->nevents; j++) {
      res->StartPMcounters[j] = 0.0;
      res->PMcounters[j] = 0.0;
    }
    add_smap(cuda_maps[i], bdata(res->label), res);
  }
}

int nvmon_markerStartRegion(const char *regionTag) {
  bstring tag;
  if (!likwid_cuda_init) {
    return -EFAULT;
  }
  if (activeCudaGroup < 0) {
    return -EFAULT;
  }
  if (gettid() != main_tid) {
    return 0;
  }

  nvmon_readCounters();
  tag = bformat("%s-%d", regionTag, activeCudaGroup);

  for (int i = 0; i < nvmon_getNumberOfGPUs(); i++) {
    LikwidGpuResults *results = NULL;
    int ret = get_smap_by_key(cuda_maps[i], bdata(tag), (void **)&results);
    if (ret < 0) {
      results = malloc(sizeof(LikwidGpuResults));
      if (!results) {
        fprintf(stderr, "Failed to register region %s\n", regionTag);
        return -EFAULT;
      }
      memset(results, 0, sizeof(LikwidGpuResults));
      results->time = 0;
      results->count = 0;
      results->gpuID = i;
      results->state = GPUMARKER_STATE_NEW;
      results->groupID = activeCudaGroup;
      results->label = bstrcpy(tag);
      results->nevents = nvmon_getNumberOfEvents(results->groupID);
      for (int j = 0; j < results->nevents; j++) {
        results->StartPMcounters[j] = 0.0;
        results->PMcounters[j] = 0.0;
      }
      add_smap(cuda_maps[i], bdata(results->label), results);
      ret = 0;
    }
    if (ret == 0 && results->state == GPUMARKER_STATE_START) {
      fprintf(stderr, "WARN: Starting an already-started region %s\n",
              regionTag);
      return -EFAULT;
    }
    for (int j = 0; j < results->nevents; j++) {
      NvmonDevice_t device = &nvGroupSet->gpus[i];
      if (device->backend == LIKWID_NVMON_CUPTI_BACKEND)
        results->StartPMcounters[j] =
            nvmon_getLastResult(results->groupID, j, i);
      else if (device->backend == LIKWID_NVMON_PERFWORKS_BACKEND)
        results->StartPMcounters[j] = nvmon_getResult(results->groupID, j, i);
      GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, START Device % d Event % d
                     : % f, i, j, results->StartPMcounters[j]);
    }
    results->state = GPUMARKER_STATE_START;
    timer_start(&(results->startTime));
  }
  bdestroy(tag);
  return 0;
}

int nvmon_markerStopRegion(const char *regionTag) {
  bstring tag;
  if (!likwid_cuda_init) {
    return -EFAULT;
  }
  if (activeCudaGroup < 0) {
    return -EFAULT;
  }
  if (gettid() != main_tid) {
    return 0;
  }
  TimerData timestamp;
  timer_stop(&timestamp);

  nvmon_readCounters();
  tag = bformat("%s-%d", regionTag, activeCudaGroup);
  for (int i = 0; i < nvmon_getNumberOfGPUs(); i++) {
    LikwidGpuResults *results = NULL;
    int ret = get_smap_by_key(cuda_maps[i], bdata(tag), (void **)&results);
    if ((ret < 0) || (results->state != GPUMARKER_STATE_START)) {
      fprintf(stderr, "WARN: Stopping an unknown/not-started region %s\n",
              regionTag);
      return -EFAULT;
    }

    results->startTime.stop.int64 = timestamp.stop.int64;
    results->time += timer_print(&(results->startTime));
    results->count++;
    for (int j = 0; j < results->nevents; j++) {
      double end = nvmon_getResult(results->groupID, j, i);
      NvmonDevice_t device = &nvGroupSet->gpus[i];
      /*            if (device->backend == LIKWID_NVMON_CUPTI_BACKEND)*/
      /*                results->PMcounters[j] += end -
       * results->StartPMcounters[j];*/
      /*            else if (device->backend ==
       * LIKWID_NVMON_PERFWORKS_BACKEND)*/
      /*            {*/
      /*                */
      /*            }*/
      results->PMcounters[j] += end - results->StartPMcounters[j];
      GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, STOP Device % d Event % d
                     : % f - % f, i, j, end, results->StartPMcounters[j]);
    }
    results->state = GPUMARKER_STATE_STOP;
  }
  bdestroy(tag);
  return 0;
}

void nvmon_markerGetRegion(const char *regionTag, int *nr_gpus,
                               int *nr_events, double **events, double *time,
                               int *count) {
  if (!likwid_cuda_init) {
    *nr_gpus = 0;
    *nr_events = 0;
    return;
  }
  if (gettid() != main_tid) {
    *nr_gpus = 0;
    *nr_events = 0;
    return;
  }
  bstring tag = bformat("%s-%d", regionTag, activeCudaGroup);
  if (count != NULL) {
    for (int i = 0; i < MIN(nvmon_getNumberOfGPUs(), *nr_gpus); i++) {
      LikwidGpuResults *results = NULL;
      int ret = get_smap_by_key(cuda_maps[i], bdata(tag), (void **)&results);
      if (ret == 0) {
        count[i] = results->count;
      }
    }
  }
  if (time != NULL) {
    for (int i = 0; i < MIN(nvmon_getNumberOfGPUs(), *nr_gpus); i++) {
      LikwidGpuResults *results = NULL;
      int ret = get_smap_by_key(cuda_maps[i], bdata(tag), (void **)&results);
      if (ret == 0) {
        time[i] = results->time;
      }
    }
  }
  if (nr_events != NULL && events != NULL && *nr_events > 0) {
    for (int i = 0; i < MIN(nvmon_getNumberOfGPUs(), *nr_gpus); i++) {
      LikwidGpuResults *results = NULL;
      int ret = get_smap_by_key(cuda_maps[i], bdata(tag), (void **)&results);
      if (ret == 0) {
        for (int j = 0;
             j < MIN(nvmon_getNumberOfEvents(activeCudaGroup), *nr_events);
             j++) {
          events[i][j] = results->PMcounters[j];
        }
      }
    }
    *nr_events = MIN(nvmon_getNumberOfEvents(activeCudaGroup), *nr_events);
  }
  *nr_gpus = MIN(nvmon_getNumberOfGPUs(), *nr_gpus);
  bdestroy(tag);
  return;
}

int nvmon_markerResetRegion(const char *regionTag) {
  if (!likwid_cuda_init) {
    return -EFAULT;
  }
  if (gettid() != main_tid) {
    return 0;
  }
  bstring tag = bformat("%s-%d", regionTag, activeCudaGroup);
  for (int i = 0; i < nvmon_getNumberOfGPUs(); i++) {
    LikwidGpuResults *results = NULL;
    int ret = get_smap_by_key(cuda_maps[i], bdata(tag), (void **)&results);
    if ((ret < 0) || (results->state != GPUMARKER_STATE_STOP)) {
      fprintf(stderr, "ERROR: Can only reset known/stopped regions\n");
      return -EFAULT;
    }
    memset(results->PMcounters, 0,
           nvmon_getNumberOfEvents(activeCudaGroup) * sizeof(double));
    results->count = 0;
    results->time = 0;
    timer_reset(&results->startTime);
  }
}

void nvmon_markerNextGroup(void) {
  if (!likwid_cuda_init) {
    return;
  }
  if (gettid() != main_tid) {
    return;
  }
  int next_group = (activeCudaGroup + 1) % numberOfCudaGroups;
  if (next_group != activeCudaGroup) {
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Switch from GPU group % d to group % d,
                activeCudaGroup, next_group);
    nvmon_switchActiveGroup(next_group);
  }
}
