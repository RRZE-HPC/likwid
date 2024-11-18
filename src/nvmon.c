/*
 * =======================================================================================
 *
 *      Filename:  nvmon.c
 *
 *      Description:  Main implementation of the performance monitoring module
 *                    for NVIDIA GPUs
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tg), thomas.roehl@googlemail.com
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

/*
 * Currently this code assumes a single context per device. In most cases there is only
 * a single context but since other interfaces allow multiple contexts it might be
 * necessary. On the other hand, the other interfaces assume that the same events
 * are measured on all devices while this interface allows different set of events
 * per device
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include <sys/types.h>

#include <dlfcn.h>
#include <cupti.h>
#include <cuda.h>


#include <types.h>
#include <likwid.h>
#include <bitUtil.h>
#include <timer.h>
#include <lock.h>
#include <ghash.h>
#include <error.h>
#include <bstrlib.h>

#include <nvmon_types.h>
#include <libnvctr_types.h>

#include <likwid.h>



static int nvmon_initialized = 0;
NvmonGroupSet* nvGroupSet = NULL;
int likwid_nvmon_verbosity = DEBUGLEV_ONLY_ERROR;

#include <nvmon_cupti.h>
#include <nvmon_nvml.h>
#include <nvmon_perfworks.h>

LikwidNvResults* gMarkerResults = NULL;
int gMarkerRegions = 0;




int
nvmon_init(int nrGpus, const int* gpuIds)
{
    int idx = 0;
    int ret = 0;
    CudaTopology_t gtopo = NULL;
    if (nvmon_initialized == 1)
    {
        return 0;
    }

    if (nrGpus <= 0)
    {
        ERROR_PRINT(Number of gpus must be greater than 0 but only %d given, nrGpus);
        return -EINVAL;
    }

    // if (!lock_check())
    // {
    //     ERROR_PLAIN_PRINT(Access to performance monitoring locked);
    //     return -EINVAL;
    // }

    if (nvGroupSet != NULL)
    {
        return -EEXIST;
    }

    ret = topology_cuda_init();
    if (ret != EXIT_SUCCESS)
    {
        return -ENODEV;
    }
    gtopo = get_cudaTopology();

    init_configuration();

    nvGroupSet = calloc(1, sizeof(NvmonGroupSet));
    if (nvGroupSet == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate group descriptor);
        return -ENOMEM;
    }
    nvGroupSet->gpus = (NvmonDevice*) malloc(nrGpus * sizeof(NvmonDevice));
    if (nvGroupSet->gpus == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate set of GPUs);
        free(nvGroupSet);
        nvGroupSet = NULL;
        return -ENOMEM;
    }
    nvGroupSet->numberOfGPUs = nrGpus;
    nvGroupSet->numberOfGroups = 0;
    nvGroupSet->numberOfActiveGroups = 0;
    nvGroupSet->groups = NULL;
    nvGroupSet->activeGroup = -1;
    nvGroupSet->numberOfBackends = 0;
    nvGroupSet->numGroupSources = 0;
    nvGroupSet->groupSources = NULL;

#ifdef LIKWID_NVMON_CUPTI_H
    nvGroupSet->backends[LIKWID_NVMON_CUPTI_BACKEND] = &nvmon_cupti_functions;
    nvGroupSet->numberOfBackends++;
#endif
#ifdef LIKWID_NVMON_PERFWORKS_H
    nvGroupSet->backends[LIKWID_NVMON_PERFWORKS_BACKEND] = &nvmon_perfworks_functions;
    nvGroupSet->numberOfBackends++;
#endif

    int cputicount = 0;
    int perfworkscount = 0;
    for (int i = 0; i < nrGpus; i++)
    {
        if (gpuIds[i] < 0) continue;
        int available = 0;
        for (int j = 0; j < gtopo->numDevices; j++)
        {
            if (gtopo->devices[j].devid == gpuIds[i])
            {
                available = 1;
                break;
            }
        }
        if (!available)
        {
            ERROR_PRINT(No device with ID %d, gpuIds[i]);
            free(nvGroupSet->gpus);
            free(nvGroupSet);
            nvGroupSet = NULL;
            return -ENOMEM;
        }
        NvmonDevice_t device = &nvGroupSet->gpus[i];
        if (gtopo->devices[gpuIds[i]].ccapMajor < 7)
        {
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Device %d runs with CUPTI Event API backend, gpuIds[i]);
            device->backend = LIKWID_NVMON_CUPTI_BACKEND;
            cputicount++;
        }
        else
        {
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Device %d runs with CUPTI Profiling API backend, gpuIds[i]);
            device->backend = LIKWID_NVMON_PERFWORKS_BACKEND;
            perfworkscount++;
        }
        device->deviceId = gpuIds[i];
        device->timeStart = 0;
        device->timeStop = 0;
        device->timeRead = 0;
        if (cputicount > 0 && perfworkscount > 0)
        {
            ERROR_PRINT(Cannot use GPUs with different backends in a session, gpuIds[i]);
            free(nvGroupSet->gpus);
            free(nvGroupSet);
            nvGroupSet = NULL;
            return -ENOMEM;
        }

        NvmonFunctions* funcs = nvGroupSet->backends[device->backend];
        if (funcs && funcs->createDevice)
        {
            ret = funcs->createDevice(device->deviceId, device);
            if (ret < 0)
            {
                ERROR_PRINT(Cannot create device %d (error: %d), device->deviceId, ret);
                free(nvGroupSet->gpus);
                free(nvGroupSet);
                nvGroupSet = NULL;
                return -ENOMEM;
            }
        }
    }

    ret = nvml_init();
    if (ret < 0)
    {
        free(nvGroupSet->gpus);
        free(nvGroupSet);
        nvGroupSet = NULL;
        return ret;
    }

    nvmon_initialized = 1;
    return 0;
}



void
nvmon_finalize(void)
{
    if (nvmon_initialized && nvGroupSet)
    {
        nvml_finalize();
        for (int i = 0; i < nvGroupSet->numberOfGPUs; i++)
        {
            NvmonDevice_t device = &nvGroupSet->gpus[i];
            NvmonFunctions* funcs = nvGroupSet->backends[device->backend];
            if (funcs && funcs->freeDevice)
                funcs->freeDevice(&nvGroupSet->gpus[i]);
        }
        for (int i = 0; i < nvGroupSet->numGroupSources; i++)
        {
            free(nvGroupSet->groupSources[i].sourceTypes);
            free(nvGroupSet->groupSources[i].sourceIds);
        }
        free(nvGroupSet->groupSources);
        free(nvGroupSet->gpus);
        free(nvGroupSet);
        nvGroupSet = NULL;
        nvmon_initialized = 0;
    }
}

static int concatNvmonEventLists(NvmonEventList_t base, NvmonEventList_t new)
{
    if ((!new) || (!base))
    {
        return -EINVAL;
    }

    int totalevents = base->numEvents + new->numEvents;

    NvmonEventListEntry* tmp = realloc(base->events, totalevents*sizeof(NvmonEventListEntry));
    if (!tmp)
    {
        return -ENOMEM;
    }
    base->events = tmp;

    for (int i = 0; i < new->numEvents; i++)
    {
        int bidx = base->numEvents + i;
        int len = strlen(new->events[i].name);
        base->events[bidx].name = malloc((len+1)*sizeof(char));
        if (base->events[bidx].name)
        {
            strncpy(base->events[bidx].name, new->events[i].name, len);
            base->events[bidx].name[len] = '\0';
        }
        len = strlen(new->events[i].desc);
        base->events[bidx].desc = malloc((len+1)*sizeof(char));
        if (base->events[bidx].desc)
        {
            strncpy(base->events[bidx].desc, new->events[i].desc, len);
            base->events[bidx].desc[len] = '\0';
        }
        len = strlen(new->events[i].limit);
        base->events[bidx].limit = malloc((len+1)*sizeof(char));
        if (base->events[bidx].limit)
        {
            strncpy(base->events[bidx].limit, new->events[i].limit, len);
            base->events[bidx].limit[len] = '\0';
        }
    }

    base->numEvents = totalevents;
    return 0;
}



int
nvmon_getEventsOfGpu(int gpuId, NvmonEventList_t* list)
{
    int err = 0;
    int ret = topology_cuda_init();
    if (ret != EXIT_SUCCESS)
    {
        return -ENODEV;
    }
    CudaTopology_t gtopo = get_cudaTopology();
    int available = -1;
    for (int i = 0; i < gtopo->numDevices; i++)
    {
        if (gtopo->devices[i].devid == gpuId)
        {
            available = i;
            break;
        }
    }
    if (available < 0)
    {
        return -EINVAL;
    }

    if (gtopo->devices[available].ccapMajor < 7)
    {
        if (nvmon_cupti_functions.getEventList)
        {
            err = nvmon_cupti_functions.getEventList(available, list);
        }
    }
    else
    {
        if (nvmon_perfworks_functions.getEventList)
        {
            err = nvmon_perfworks_functions.getEventList(available, list);
        }
    }

    // Get nvml events and merge lists
    NvmonEventList_t nvmlList = NULL;
    err = nvml_getEventsOfGpu(gpuId, &nvmlList);
    if (err < 0)
    {
        nvmon_returnEventsOfGpu(*list);
        *list = NULL;
        return err;
    }
    err = concatNvmonEventLists(*list, nvmlList);
    if (err < 0)
    {
        ERROR_PLAIN_PRINT(Failed to concatenate event lists);
        nvmon_returnEventsOfGpu(*list);
        nvml_returnEventsOfGpu(nvmlList);
        *list = NULL;
        return err;
    }
    nvml_returnEventsOfGpu(nvmlList);

    return 0;
}

void nvmon_returnEventsOfGpu(NvmonEventList_t list)
{
    if (list)
    {
        if (list->numEvents > 0 && list->events)
        {
            for (int i = 0; i < list->numEvents; i++)
            {
                NvmonEventListEntry* out = &list->events[i];
                if (out->name)
                    free(out->name);
                if (out->desc)
                    free(out->desc);
                if (out->limit)
                    free(out->limit);
            }
            free(list->events);
            list->events = NULL;
            list->numEvents = 0;
        }
    }
}


static int
nvmon_splitEventSet(GroupInfo* backendEvents, GroupInfo* nvmlEvents, int gid)
{
    int ret;

    // Initialize groups
    perfgroup_new(backendEvents);
    perfgroup_new(nvmlEvents);

    // Sort events (shallow copy)
    GroupInfo* info = &nvGroupSet->groups[gid];
    for (int i = 0; i < info->nevents; i++)
    {
        if (nvGroupSet->groupSources[gid].sourceTypes[i] == NVMON_SOURCE_NVML)
        {
            ret = perfgroup_addEvent(nvmlEvents, info->counters[i], info->events[i]);
            if (ret < 0)
            {
                ERROR_PRINT(Failed to add event while splitting);
                return ret;
            }
        }
        else
        {
            ret = perfgroup_addEvent(backendEvents, info->counters[i], info->events[i]);
            if (ret < 0)
            {
                ERROR_PRINT(Failed to add event while splitting);
                return ret;
            }
        }
    }

    return 0;
}


static void
nvmon_returnSplitEventSet(GroupInfo* backendEvents, GroupInfo* nvmlEvents)
{
    if (backendEvents != NULL)
    {
        perfgroup_returnGroup(backendEvents);
    }
    if (nvmlEvents != NULL)
    {
        perfgroup_returnGroup(nvmlEvents);
    }
}


static int
nvmon_initEventSourceLookupMaps(int gid, int gpuId)
{
    int ret;
    GroupInfo* group = &nvGroupSet->groups[gid];

    // Allocate memory
    if (gid >= nvGroupSet->numGroupSources)
    {
        int* tmpSourceTypes = (int*) malloc(group->nevents * sizeof(int));
        if (tmpSourceTypes == NULL)
        {
            ERROR_PLAIN_PRINT(Failed to allocate source type map);
            return -ENOMEM;
        }
        int* tmpSourceIds = (int*) malloc(group->nevents * sizeof(int));
        if (tmpSourceIds == NULL)
        {
            ERROR_PLAIN_PRINT(Failed to allocate source id map);
            free(tmpSourceTypes);
            return -ENOMEM;
        }
        NvmonGroupSourceInfo* tmpInfo = (NvmonGroupSourceInfo*) realloc(nvGroupSet->groupSources, (gid+1) * sizeof(NvmonGroupSourceInfo));
        if (tmpInfo == NULL)
        {
            ERROR_PLAIN_PRINT(Failed to allocate source infos);
            free(tmpSourceTypes);
            free(tmpSourceIds);
            return -ENOMEM;
        }

        nvGroupSet->groupSources = tmpInfo;
        nvGroupSet->groupSources[gid].numEvents = group->nevents;
        nvGroupSet->groupSources[gid].sourceTypes = tmpSourceTypes;
        nvGroupSet->groupSources[gid].sourceIds = tmpSourceIds;
        nvGroupSet->numGroupSources++;
    }

    // Get list of nvml events for sorting
    NvmonEventList_t nvmlList;
    ret = nvml_getEventsOfGpu(gpuId, &nvmlList);
    if (ret < 0) return ret;

    // Sort events
    int nvmlId = 0;
    int backendId = 0;
    for (int i = 0; i < group->nevents; i++)
    {
        // Check if event is in nvml list
        int isNvmlEvent = 0;
        for (int j = 0; j < nvmlList->numEvents; j++)
        {
            if (strcmp(group->events[i], nvmlList->events[j].name) == 0)
            {
                isNvmlEvent = 1;
                break;
            }
        }
        if (isNvmlEvent)
        {
            nvGroupSet->groupSources[gid].sourceTypes[i] = NVMON_SOURCE_NVML;
            nvGroupSet->groupSources[gid].sourceIds[i] = nvmlId++;
        }
        else
        {
            nvGroupSet->groupSources[gid].sourceTypes[i] = NVMON_SOURCE_BACKEND;
            nvGroupSet->groupSources[gid].sourceIds[i] = backendId++;
        }
    }

    nvml_returnEventsOfGpu(nvmlList);
    return 0;
}


int
nvmon_addEventSet(const char* eventCString)
{
    int i = 0, j = 0, k = 0, l = 0, m = 0, err = 0;

    int devId = 0;
    int configuredEvents = 0;

    NvmonDevice_t device = NULL;
    Configuration_t config = NULL;
    GroupInfo ginfo;


    if (!eventCString)
    {
        return -EINVAL;
    }
    if (!nvmon_initialized)
    {
        return -EFAULT;
    }
    // init_configuration();
    config = get_configuration();

    if (nvGroupSet->numberOfActiveGroups == nvGroupSet->numberOfGroups)
    {
        GroupInfo* tmpInfo = (GroupInfo*)realloc(nvGroupSet->groups, (nvGroupSet->numberOfGroups+1)*sizeof(GroupInfo));
        if (tmpInfo == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate additional group);
            return -ENOMEM;
        }
        nvGroupSet->groups = tmpInfo;
        nvGroupSet->numberOfGroups++;
        GPUDEBUG_PRINT(DEBUGLEV_INFO, Allocating new group structure for group.);
    }
    GPUDEBUG_PRINT(DEBUGLEV_INFO, NVMON: Currently %d groups of %d active,
                                        nvGroupSet->numberOfActiveGroups+1,
                                        nvGroupSet->numberOfGroups+1);

    bstring eventBString = bfromcstr(eventCString);
    if (bstrchrp(eventBString, ':', 0) != BSTR_ERR)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Custom eventset);
        err = perfgroup_customGroup(eventCString, &nvGroupSet->groups[nvGroupSet->numberOfGroups-1]);
        if (err)
        {
            ERROR_PRINT(Cannot transform %s to performance group, eventCString);
            return err;
        }
    }
    else
    {
        int cputicount = 0;
        int perfworkscount = 0;
        for (devId = 0; devId < nvGroupSet->numberOfGPUs; devId++)
        {
            device = &nvGroupSet->gpus[devId];
            if (device->backend == LIKWID_NVMON_CUPTI_BACKEND)
            {
                cputicount++;
            }
            else if (device->backend == LIKWID_NVMON_PERFWORKS_BACKEND)
            {
                perfworkscount++;
            }
        }
        if (cputicount > 0 && perfworkscount > 0)
        {
            ERROR_PRINT(GPUs with compute capability <7 and >= 7 are not allowed);
            return -ENODEV;
        }
        if (cputicount > 0)
        {
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Performance group for CUPTI backend);
            err = perfgroup_readGroup(config->groupPath, "nvidia_gpu_cc_lt_7", eventCString,
                                      &nvGroupSet->groups[nvGroupSet->numberOfGroups-1]);
        }
        else if (perfworkscount > 0)
        {
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Performance group for PerfWorks backend);
            err = perfgroup_readGroup(config->groupPath, "nvidia_gpu_cc_ge_7", eventCString,
                                      &nvGroupSet->groups[nvGroupSet->numberOfGroups-1]);
        }
        if (err == -EACCES)
        {
            ERROR_PRINT(Access to performance group %s not allowed, eventCString);
            return err;
        }
        else if (err == -ENODEV)
        {
            ERROR_PRINT(Performance group %s only available with deactivated HyperThreading, eventCString);
            return err;
        }
        else if (err < 0)
        {
            ERROR_PRINT(Cannot read performance group %s, eventCString);
            return err;
        }
    }

    // Build source lookup
    err = nvmon_initEventSourceLookupMaps(nvGroupSet->numberOfGroups - 1, nvGroupSet->gpus[0].deviceId); // it is assumed that all gpus split the same
    if (err < 0)
    {
        ERROR_PRINT(Failed to init source lookup for group %d, nvGroupSet->numberOfGroups - 1);
        return err;
    }

    // Split events into nvml and backend events
    GroupInfo backendEvents;
    GroupInfo nvmlEvents;
    err = nvmon_splitEventSet(&backendEvents, &nvmlEvents, nvGroupSet->numberOfGroups-1);
    if (err < 0)
    {
        ERROR_PRINT(Failed to split events);
        return -1;
    }

    // Add nvml events
    err = nvml_addEventSet(nvmlEvents.events, nvmlEvents.nevents);
    if (err < 0)
    {
        ERROR_PRINT(Failed to add nvml events);
        return -1;
    }

    bdestroy(eventBString);
    char * evstr = perfgroup_getEventStr(&backendEvents);
/*    eventBString = bfromcstr(evstr);*/
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, EventStr %s, evstr);
/*    eventtokens = bsplit(eventBString, ',');*/
/*    bdestroy(eventBString);*/

    // Event string is null when there are no events for backend
    for (devId = 0; devId < nvGroupSet->numberOfGPUs; devId++)
    {
        device = &nvGroupSet->gpus[devId];
        if (evstr != NULL)
        {
            int err = 0;
            NvmonFunctions* funcs = nvGroupSet->backends[device->backend];
            if (!funcs)
            {
                ERROR_PRINT(Backend functions undefined?);
            }
            if (funcs->addEvents)
            {
                GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Calling addevents);
                err = funcs->addEvents(device, evstr);
                if (err < 0)
                {
                    errno = -err;
                    ERROR_PRINT(Failed to add event set for GPU %d, devId);
                    return err;
                }
            }
        }   
        else
        {
            // Add empty event set 
            NvmonEventSet* tmpEventSet = realloc(device->nvEventSets, (device->numNvEventSets+1)*sizeof(NvmonEventSet));
            if (!tmpEventSet)
            {
                ERROR_PRINT(Cannot enlarge GPU %d eventSet list, device->deviceId);
                return -ENOMEM;
            }
            device->nvEventSets = tmpEventSet;
            NvmonEventSet* newEventSet = &device->nvEventSets[device->numNvEventSets];
            memset(newEventSet, 0, sizeof(NvmonEventSet));
        }
    }

    // Cleanup
    nvmon_returnSplitEventSet(&backendEvents, &nvmlEvents);

    // Check whether group has any event in any device
    nvGroupSet->numberOfActiveGroups++;
    return (nvGroupSet->numberOfActiveGroups-1);
}


int nvmon_setupCounters(int gid)
{
    int i = 0;
    int err = 0;
    int oldDevId;
    CUcontext curContext;

    if ((!nvGroupSet) || (!nvmon_initialized) || (gid < 0))
    {
        return -EFAULT;
    }

    if (gid >= 0 && gid < nvGroupSet->numberOfGroups)
    {
        for (i = 0; i < nvGroupSet->numberOfGPUs; i++)
        {
            NvmonDevice_t device = &nvGroupSet->gpus[i];

            NvmonEventSet* devEventSet = &device->nvEventSets[gid];

            NvmonFunctions* funcs = nvGroupSet->backends[device->backend];
            if (devEventSet->numberOfEvents > 0 && funcs->setupCounters)
            {
                err = funcs->setupCounters(device, devEventSet);
            }
        }
        nvml_setupCounters(gid);
    }
    nvGroupSet->activeGroup = gid;

    return err;
}


int
nvmon_startCounters(void)
{
    int i = 0;
    int err = 0;
    if ((!nvGroupSet) || (!nvmon_initialized) || (nvGroupSet->activeGroup < 0))
    {
        return -EFAULT;
    }

    for (i = 0; i < nvGroupSet->numberOfGPUs; i++)
    {
        NvmonDevice_t device = &nvGroupSet->gpus[i];
        NvmonEventSet* devEventSet = &device->nvEventSets[nvGroupSet->activeGroup];
        NvmonFunctions* funcs = nvGroupSet->backends[device->backend];
        if (devEventSet->numberOfEvents > 0 && funcs->startCounters)
        {
            err = funcs->startCounters(device);
            if (err < 0) return err;
        }
    }

    // Start nvml counters
    err = nvml_startCounters();
    if (err < 0) return err;

    return 0;
}

int
nvmon_stopCounters(void)
{
    int i = 0;
    int err = 0;
    if ((!nvGroupSet) || (!nvmon_initialized) || (nvGroupSet->activeGroup < 0))
    {
        return -EFAULT;
    }

    for (i = 0; i < nvGroupSet->numberOfGPUs; i++)
    {
        NvmonDevice_t device = &nvGroupSet->gpus[i];
        NvmonEventSet* devEventSet = &device->nvEventSets[nvGroupSet->activeGroup];
        NvmonFunctions* funcs = nvGroupSet->backends[device->backend];
        if (devEventSet->numberOfEvents > 0 && funcs->stopCounters)
        {
            err = funcs->stopCounters(device);
            if (err < 0) return err;
        }
    }

    // Stop nvml counters
    err = nvml_stopCounters();
    if (err < 0) return err;

    return 0;
}

int nvmon_readCounters(void)
{

    int i = 0;
    int err = 0;
    if ((!nvGroupSet) || (!nvmon_initialized) || (nvGroupSet->activeGroup < 0))
    {
        return -EFAULT;
    }

    for (i = 0; i < nvGroupSet->numberOfGPUs; i++)
    {
        NvmonDevice_t device = &nvGroupSet->gpus[i];
        NvmonEventSet* devEventSet = &device->nvEventSets[nvGroupSet->activeGroup];
        NvmonFunctions* funcs = nvGroupSet->backends[device->backend];
        if (devEventSet->numberOfEvents > 0 && funcs->readCounters)
        {
            err = funcs->readCounters(device);
            if (err < 0) return err;
        }
    }
    
    // Read nvml counters
    err = nvml_readCounters();
    if (err < 0) return err;

    return 0;
}

double nvmon_getResult(int groupId, int eventId, int gpuId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (nvGroupSet->activeGroup < 0))
    {
        return -EFAULT;
    }
    if (gpuId < 0 || gpuId >= nvGroupSet->numberOfGPUs)
    {
        return -EFAULT;
    }
    if (groupId < 0 || groupId >= nvGroupSet->numGroupSources)
    {
        return -EFAULT;
    }
    NvmonGroupSourceInfo* info = &nvGroupSet->groupSources[groupId];
    if (eventId < 0 || eventId >= info->numEvents)
    {
        return -EFAULT;
    }

    // Get value from respective source
    if (info->sourceTypes[eventId] == NVMON_SOURCE_NVML)
    {
        return nvml_getResult(gpuId, groupId, info->sourceIds[eventId]);
    }
    else
    {
        NvmonDevice *device = &nvGroupSet->gpus[gpuId];
        if (groupId < 0 || groupId >= device->numNvEventSets)
        {
            return -EFAULT;
        }
        NvmonEventSet* evset = &device->nvEventSets[groupId];
        return evset->results[info->sourceIds[eventId]].fullValue;
    }
}


int
nvmon_switchActiveGroup(int new_group)
{
    nvmon_stopCounters();
    nvmon_setupCounters(new_group);
    nvmon_startCounters();
    return 0;
}

double nvmon_getLastResult(int groupId, int eventId, int gpuId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (nvGroupSet->activeGroup < 0))
    {
        return -EFAULT;
    }
    if (gpuId < 0 || gpuId >= nvGroupSet->numberOfGPUs)
    {
        return -EFAULT;
    }
    if (groupId < 0 || groupId >= nvGroupSet->numGroupSources)
    {
        return -EFAULT;
    }
    NvmonGroupSourceInfo* info = &nvGroupSet->groupSources[groupId];
    if (eventId < 0 || eventId >= info->numEvents)
    {
        return -EFAULT;
    }

    // Get value from respective source
    if (info->sourceTypes[eventId] == NVMON_SOURCE_NVML)
    {
        return nvml_getLastResult(gpuId, groupId, info->sourceIds[eventId]);
    }
    else
    {
        NvmonDevice *device = &nvGroupSet->gpus[gpuId];
        if (groupId < 0 || groupId >= device->numNvEventSets)
        {
            return -EFAULT;
        }
        NvmonEventSet* evset = &device->nvEventSets[groupId];
        return evset->results[info->sourceIds[eventId]].lastValue;
    }
}





int nvmon_getNumberOfGroups(void)
{
    if ((!nvGroupSet) || (!nvmon_initialized))
    {
        return -EFAULT;
    }
    return nvGroupSet->numberOfActiveGroups;
}

int nvmon_getIdOfActiveGroup(void)
{
    if ((!nvGroupSet) || (!nvmon_initialized))
    {
        return -EFAULT;
    }
    return nvGroupSet->activeGroup;
}

int nvmon_getNumberOfGPUs(void)
{
    if ((!nvGroupSet) || (!nvmon_initialized))
    {
        return -EFAULT;
    }
    return nvGroupSet->numberOfGPUs;
}

int nvmon_getNumberOfEvents(int groupId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return -EFAULT;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    return ginfo->nevents;
}

double nvmon_getTimeOfGroup(int groupId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return -EFAULT;
    }

    // Get largest time from backend measurings
    double backendTime = 0;
    for (int i = 0; i < nvGroupSet->numberOfGPUs; i++)
    {
        // timeStart and timeStop are zero if no backend event is registered
        backendTime = MAX(backendTime, (double)(nvGroupSet->gpus[i].timeStop - nvGroupSet->gpus[i].timeStart));
    }
    backendTime *= 1E-9; // ns to seconds

    // Get largest time of nvml events
    double nvmlTime = 0;
    nvmlTime = nvml_getTimeOfGroup(groupId);
    if (nvmlTime < 0)
    {
        return nvmlTime;
    }

    // Return largest time
    return MAX(backendTime, nvmlTime);
}


double nvmon_getLastTimeOfGroup(int groupId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return -EFAULT;
    }

    // Get largest time from backend measurings
    double backendTime = 0;
    for (int i = 0; i < nvGroupSet->numberOfGPUs; i++)
    {
        // timeStart and timeStop are zero if no backend event is registered
        backendTime = MAX(backendTime, (double)(nvGroupSet->gpus[i].timeStop - nvGroupSet->gpus[i].timeRead));
    }
    backendTime *= 1E-9; // ns to seconds

    // Get largest time of nvml events
    double nvmlTime = 0;
    nvmlTime = nvml_getLastTimeOfGroup(groupId);
    if (nvmlTime < 0)
    {
        return nvmlTime;
    }

    // Return largest time
    return MAX(backendTime, nvmlTime);
}


double nvmon_getTimeToLastReadOfGroup(int groupId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return -EFAULT;
    }

    // Get largest time from backend measurings
    double backendTime = 0;
    for (int i = 0; i < nvGroupSet->numberOfGPUs; i++)
    {
        // timeStart and timeStop are zero if no backend event is registered
        backendTime = MAX(backendTime, (double)(nvGroupSet->gpus[i].timeRead - nvGroupSet->gpus[i].timeStart));
    }
    backendTime *= 1E-9; // ns to seconds

    // Get largest time of nvml events
    double nvmlTime = 0;
    nvmlTime = nvml_getTimeToLastReadOfGroup(groupId);
    if (nvmlTime < 0)
    {
        return nvmlTime;
    }

    // Return largest time
    return MAX(backendTime, nvmlTime);
}


char* nvmon_getEventName(int groupId, int eventId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    if ((eventId < 0) || (eventId >= ginfo->nevents))
    {
        return NULL;
    }
    return ginfo->events[eventId];
}

char* nvmon_getCounterName(int groupId, int eventId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    if ((eventId < 0) || (eventId >= ginfo->nevents))
    {
        return NULL;
    }
    return ginfo->counters[eventId];
}

char* nvmon_getMetricName(int groupId, int metricId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    if ((metricId < 0) || (metricId >= ginfo->nmetrics))
    {
        return NULL;
    }
    return ginfo->metricnames[metricId];
}

char* nvmon_getGroupName(int groupId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    return ginfo->groupname;
}

char* nvmon_getGroupInfoShort(int groupId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    return ginfo->shortinfo;
}

char* nvmon_getGroupInfoLong(int groupId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    return ginfo->longinfo;
}

int nvmon_getGroups(int gpuId, char*** groups, char*** shortinfos, char*** longinfos)
{
    int ret = 0;

    init_configuration();
    int ccapMajor = 0;
    Configuration_t config = get_configuration();
    ret = topology_cuda_init();
    if (ret != EXIT_SUCCESS)
    {
        return -ENODEV;
    }
    CudaTopology_t gtopo = get_cudaTopology();
    if (gpuId < 0 || gpuId >= gtopo->numDevices)
    {
        return -ENODEV;
    }
    if (gtopo->devices[gpuId].ccapMajor >= 7)
    {
        ret = perfgroup_getGroups(config->groupPath, "nvidia_gpu_cc_ge_7", groups, shortinfos, longinfos);
    }
    else
    {
        ret = perfgroup_getGroups(config->groupPath, "nvidia_gpu_cc_lt_7", groups, shortinfos, longinfos);
    }
    return ret;
}

int nvmon_returnGroups(int nrgroups, char** groups, char** shortinfos, char** longinfos)
{
    perfgroup_returnGroups(nrgroups, groups, shortinfos, longinfos);
    return 0;
}

int nvmon_getNumberOfMetrics(int groupId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return -EFAULT;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    return ginfo->nmetrics;
}

void nvmon_setVerbosity(int level)
{
    if (level >= DEBUGLEV_ONLY_ERROR && level <= DEBUGLEV_DEVELOP)
    {
        likwid_nvmon_verbosity = level;
    }
}


double nvmon_getMetric(int groupId, int metricId, int gpuId)
{
    int e = 0;
    double result = 0;
    CounterList clist;
    if (unlikely(nvGroupSet == NULL))
    {
        return NAN;
    }
    if (nvmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return NAN;
    }
    if (nvGroupSet->numberOfActiveGroups == 0)
    {
        return NAN;
    }
    if ((groupId < 0) && (nvGroupSet->activeGroup >= 0))
    {
        groupId = groupSet->activeGroup;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    if (ginfo->nmetrics == 0)
    {
        return NAN;
    }
    if ((gpuId < 0) || (gpuId >= nvGroupSet->numberOfGPUs))
    {
        return NAN;
    }
    if ((metricId < 0 || metricId >= ginfo->nmetrics))
    {
        return NAN;
    }
    
    NvmonDevice_t device = &nvGroupSet->gpus[gpuId];
    if (groupId < 0 || groupId >= device->numNvEventSets)
    {
        return -EFAULT;
    }
    NvmonEventSet* evset = &device->nvEventSets[groupId];
    timer_init();
    init_clist(&clist);
    for (e=0;e<evset->numberOfEvents;e++)
    {
        add_to_clist(&clist,ginfo->counters[e], nvmon_getResult(groupId, e, gpuId));
    }
    add_to_clist(&clist, "time", nvmon_getTimeOfGroup(groupId));
    add_to_clist(&clist, "inverseClock", 1.0/timer_getCycleClock());
    add_to_clist(&clist, "true", 1);
    add_to_clist(&clist, "false", 0);

    double result2;
    e = calc_metric(ginfo->metricformulas[metricId], &clist, &result);
    e = calc_metric_new(ginfo->metrictrees[metricId], &clist, &result2);
    if(fabs(result - result2) > 1.0E-6) {
        fprintf(stderr "Error: results don't match");
        exit(EXIT_FAILURE);
    }
    
    if (e < 0)
    {
        result = NAN;
    }
    destroy_clist(&clist);
    return result;
}

double nvmon_getLastMetric(int groupId, int metricId, int gpuId)
{
    int e = 0;
    double result = 0;
    CounterList clist;
    if (unlikely(nvGroupSet == NULL))
    {
        return NAN;
    }
    if (nvmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return NAN;
    }
    if (nvGroupSet->numberOfActiveGroups == 0)
    {
        return NAN;
    }
    if ((groupId < 0) && (nvGroupSet->activeGroup >= 0))
    {
        groupId = groupSet->activeGroup;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    if (ginfo->nmetrics == 0)
    {
        return NAN;
    }
    if ((gpuId < 0) || (gpuId >= nvGroupSet->numberOfGPUs))
    {
        return NAN;
    }
    if ((metricId < 0 || metricId >= ginfo->nmetrics))
    {
        return NAN;
    }
    
    NvmonDevice_t device = &nvGroupSet->gpus[gpuId];
    if (groupId < 0 || groupId >= device->numNvEventSets)
    {
        return -EFAULT;
    }
    NvmonEventSet* evset = &device->nvEventSets[groupId];
    timer_init();
    init_clist(&clist);
    for (e=0;e<evset->numberOfEvents;e++)
    {
        add_to_clist(&clist,ginfo->counters[e], nvmon_getLastResult(groupId, e, gpuId));
    }
    add_to_clist(&clist, "time", nvmon_getTimeOfGroup(groupId));
    add_to_clist(&clist, "inverseClock", 1.0/timer_getCycleClock());
    add_to_clist(&clist, "true", 1);
    add_to_clist(&clist, "false", 0);
    double result2;
    e = calc_metric(ginfo->metricformulas[metricId], &clist, &result);
    e = calc_metric_new(ginfo->metrictrees[metricId], &clist, &result2);
    if(fabs(result - result2) > 1.0E-6) {
        fprintf(stderr "Error: results don't match");
        exit(EXIT_FAILURE);
    }
    
    if (e < 0)
    {
        result = NAN;
    }
    destroy_clist(&clist);
    return result;
}




void
nvmon_printMarkerResults()
{
    int i = 0, j = 0, k = 0;
    for (i=0; i<gMarkerRegions; i++)
    {
        printf("Region %d : %s\n", i, bdata(gMarkerResults[i].tag));
        printf("Group %d\n", gMarkerResults[i].groupID);
        for (j=0;j<gMarkerResults[i].gpuCount; j++)
        {
            printf("GPU %d:\n", j, gMarkerResults[i].gpulist[j]);
            printf("\t Measurement time %f sec\n", gMarkerResults[i].time[j]);
            printf("\t Call count %d\n", gMarkerResults[i].count[j]);
            for(k=0;k<gMarkerResults[i].eventCount;k++)
            {
                printf("\t Event %d : %f\n", k, gMarkerResults[i].counters[j][k]);
            }
        }
    }
}

int
nvmon_readMarkerFile(const char* filename)
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
    ret = sscanf(buf, "%d %d %d", &gpus, &regions, &groups);
    if (ret != 3)
    {
        fprintf(stderr, "GPUMarker file missformatted.\n");
        return -EINVAL;
    }
    gMarkerResults = realloc(gMarkerResults, regions * sizeof(LikwidNvResults));
    if (gMarkerResults == NULL)
    {
        fprintf(stderr, "Failed to allocate %lu bytes for the marker results storage\n", regions * sizeof(LikwidNvResults));
        return -ENOMEM;
    }
    int* regionGPUs = (int*)malloc(regions * sizeof(int));
    if (regionGPUs == NULL)
    {
        fprintf(stderr, "Failed to allocate %lu bytes for temporal gpu count storage\n", regions * sizeof(int));
        return -ENOMEM;
    }
    gMarkerRegions = regions;
    for (int i = 0; i < regions; i++)
    {
        regionGPUs[i] = 0;
        gMarkerResults[i].gpuCount = gpus;
        gMarkerResults[i].time = (double*) malloc(gpus * sizeof(double));
        if (!gMarkerResults[i].time)
        {
            fprintf(stderr, "Failed to allocate %lu bytes for the time storage\n", gpus * sizeof(double));
            break;
        }
        gMarkerResults[i].count = (uint32_t*) malloc(gpus * sizeof(uint32_t));
        if (!gMarkerResults[i].count)
        {
            fprintf(stderr, "Failed to allocate %lu bytes for the count storage\n", gpus * sizeof(uint32_t));
            break;
        }
        gMarkerResults[i].gpulist = (int*) malloc(gpus * sizeof(int));
        if (!gMarkerResults[i].gpulist)
        {
            fprintf(stderr, "Failed to allocate %lu bytes for the gpulist storage\n", gpus * sizeof(int));
            break;
        }
        gMarkerResults[i].counters = (double**) malloc(gpus * sizeof(double*));
        if (!gMarkerResults[i].counters)
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
            // zero out ALL of regiontag due to replacing %s with %Nc
            memset(regiontag, 0, sizeof(regiontag) * sizeof(char));
            char fmt[64];
            // using %d:%s for sscanf doesn't support spaces so replace %s with %Nc where N is one minus
            // the size of regiontag, thus to avoid hardcoding N, compose fmt from the size of regiontag, e.g.:
            //      regiontag[50]  --> %d:%49c
            //      regiontag[100] --> %d:%99c
            snprintf(fmt, 60, "%s:%s%ic", "%d", "%", (int) (sizeof(regiontag) - 1));
            // use fmt (%d:%Nc) in lieu of %d:%s to support spaces
            ret = sscanf(buf, fmt, &regionid, regiontag);

            ptr = strrchr(regiontag,'-');
            colonptr = strchr(buf,':');
            if (ret != 2 || ptr == NULL || colonptr == NULL)
            {
                fprintf(stderr, "Line %s not a valid region description: %s\n", buf, regiontag);
                continue;
            }
            groupid = atoi(ptr+1);
            snprintf(regiontag, strlen(regiontag)-strlen(ptr)+1, "%s", &(buf[colonptr-buf+1]));
            gMarkerResults[regionid].groupID = groupid;
            gMarkerResults[regionid].tag = bfromcstr(regiontag);
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
                gMarkerResults[regionid].gpulist[gpuidx] = gpu;
                gMarkerResults[regionid].eventCount = nevents;
                gMarkerResults[regionid].time[gpuidx] = time;
                gMarkerResults[regionid].count[gpuidx] = count;
                gMarkerResults[regionid].counters[gpuidx] = malloc(nevents * sizeof(double));

                eventidx = 0;
                ptr = strtok(remain, " ");
                while (ptr != NULL && eventidx < nevents)
                {
                    sscanf(ptr, "%lf", &(gMarkerResults[regionid].counters[gpuidx][eventidx]));
                    ptr = strtok(NULL, " ");
                    eventidx++;
                }
                regionGPUs[regionid]++;
            }
        }
    }
    for (int i = 0; i < regions; i++)
    {
        gMarkerResults[i].gpuCount = regionGPUs[i];
    }
    free(regionGPUs);
    fclose(fp);
    return nr_regions;
}

void
nvmon_destroyMarkerResults()
{
    int i = 0, j = 0;
    if (gMarkerResults != NULL)
    {
        for (i = 0; i < gMarkerRegions; i++)
        {
            free(gMarkerResults[i].time);
            free(gMarkerResults[i].count);
            free(gMarkerResults[i].gpulist);
            for (j = 0; j < gMarkerResults[i].gpuCount; j++)
            {
                free(gMarkerResults[i].counters[j]);
            }
            free(gMarkerResults[i].counters);
            bdestroy(gMarkerResults[i].tag);
        }
        free(gMarkerResults);
        gMarkerResults = NULL;
        gMarkerRegions = 0;
    }
}


int
nvmon_getCountOfRegion(int region, int gpu)
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return -EINVAL;
    }
    if (gpu < 0 || gpu >= gMarkerResults[region].gpuCount)
    {
        return -EINVAL;
    }
    if (gMarkerResults[region].count == NULL)
    {
        return 0;
    }
    return gMarkerResults[region].count[gpu];
}

double
nvmon_getTimeOfRegion(int region, int gpu)
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return -EINVAL;
    }
    if (gpu < 0 || gpu >= gMarkerResults[region].gpuCount)
    {
        return -EINVAL;
    }
    if (gMarkerResults[region].time == NULL)
    {
        return 0.0;
    }
    return gMarkerResults[region].time[gpu];
}

int
nvmon_getGpulistOfRegion(int region, int count, int* gpulist)
{
    int i;
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return -EINVAL;
    }
    if (gpulist == NULL)
    {
        return -EINVAL;
    }
    for (i=0; i< MIN(count, gMarkerResults[region].gpuCount); i++)
    {
        gpulist[i] = gMarkerResults[region].gpulist[i];
    }
    return MIN(count, gMarkerResults[region].gpuCount);
}

int
nvmon_getGpusOfRegion(int region)
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return -EINVAL;
    }
    return gMarkerResults[region].gpuCount;
}

int
nvmon_getMetricsOfRegion(int region)
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return -EINVAL;
    }
    return nvmon_getNumberOfMetrics(gMarkerResults[region].groupID);
}

int
nvmon_getNumberOfRegions()
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return -EINVAL;
    }
    return gMarkerRegions;
}

int
nvmon_getGroupOfRegion(int region)
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return -EINVAL;
    }
    return gMarkerResults[region].groupID;
}

char*
nvmon_getTagOfRegion(int region)
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return NULL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return NULL;
    }
    return bdata(gMarkerResults[region].tag);
}

int
nvmon_getEventsOfRegion(int region)
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return -EINVAL;
    }
    return gMarkerResults[region].eventCount;
}

double nvmon_getResultOfRegionGpu(int region, int eventId, int gpuId)
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return -EINVAL;
    }
    if (gpuId < 0 || gpuId >= gMarkerResults[region].gpuCount)
    {
        return -EINVAL;
    }
    if (eventId < 0 || eventId >= gMarkerResults[region].eventCount)
    {
        return -EINVAL;
    }
    if (gMarkerResults[region].counters[gpuId] == NULL)
    {
        return 0.0;
    }
    return gMarkerResults[region].counters[gpuId][eventId];
}

double nvmon_getMetricOfRegionGpu(int region, int metricId, int gpuId)
{
    int e = 0, err = 0;
    double result = 0.0;
    CounterList clist;
    if (nvmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return NAN;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return NAN;
    }
    if (gMarkerResults == NULL)
    {
        return NAN;
    }
    if (gpuId < 0 || gpuId >= gMarkerResults[region].gpuCount)
    {
        return NAN;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[gMarkerResults[region].groupID];
    if (metricId < 0 || metricId >= ginfo->nmetrics)
    {
        return NAN;
    }
    
    timer_init();
    init_clist(&clist);
    for (e = 0; e < gMarkerResults[region].eventCount; e++)
    {
        double res = nvmon_getResultOfRegionGpu(region, e, gpuId);
        char* ctr = ginfo->counters[e];
        add_to_clist(&clist, ctr, res);
    }
    add_to_clist(&clist, "time", nvmon_getTimeOfRegion(gMarkerResults[region].groupID, gpuId));
    add_to_clist(&clist, "inverseClock", 1.0/timer_getCycleClock());
    add_to_clist(&clist, "true", 1);
    add_to_clist(&clist, "false", 0);

    double result2;
    e = calc_metric(ginfo->metricformulas[metricId], &clist, &result);
    e = calc_metric_new(ginfo->metrictrees[metricId], &clist, &result2);
    if(fabs(result - result2) > 1.0E-6) {
        fprintf(stderr "Error: results don't match");
        exit(EXIT_FAILURE);
    }
    
    if (err < 0)
    {
        ERROR_PRINT(Cannot calculate formula %s, f);
        return NAN;
    }
    destroy_clist(&clist);
    return result;
}
