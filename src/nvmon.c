/*
 * =======================================================================================
 *
 *      Filename:  nvmon.c
 *
 *      Description:  Main implementation of the performance monitoring module
 *                    for NVIDIA GPUs
 *
 *      Version:   5.0.2
 *      Released:  06.10.2020
 *
 *      Author:   Thomas Gruber (tg), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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
//#include <cupti.h>
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
    GpuTopology_t gtopo = NULL;
    if (nvmon_initialized == 1)
    {
        return 0;
    }

    if (nrGpus <= 0)
    {
        ERROR_PRINT(Number of gpus must be greater than 0 but only %d given, nrGpus);
        return -EINVAL;
    }

    if (!lock_check())
    {
        ERROR_PLAIN_PRINT(Access to performance monitoring locked);
        return -EINVAL;
    }

    if (nvGroupSet != NULL)
    {
        return -EEXIST;
    }

    ret = topology_gpu_init();
    if (ret != EXIT_SUCCESS)
    {
        return -ENODEV;
    }
    gtopo = get_gpuTopology();

    nvGroupSet = (NvmonGroupSet*) malloc(sizeof(NvmonGroupSet));
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

#ifdef LIKWID_NVMON_CUPTI_H
    nvGroupSet->backends[LIKWID_NVMON_CUPTI_BACKEND] = &nvmon_cupti_functions;
    nvGroupSet->numberOfBackends++;
#endif
#ifdef LIKWID_NVMON_PERFWORKS_H
    nvGroupSet->backends[LIKWID_NVMON_PERFWORKS_BACKEND] = &nvmon_perfworks_functions;
    nvGroupSet->numberOfBackends++;
#endif

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
        NvmonDevice_t device = &nvGroupSet->gpus[idx];
        if (gtopo->devices[gpuIds[i]].ccapMajor < 7)
        {
            device->backend = LIKWID_NVMON_CUPTI_BACKEND;
        }
        else
        {
            fprintf(stderr, "NVIDIA PerfWorks API current not supported. Trying CUPTI\n");
/*            free(nvGroupSet->gpus);*/
/*            free(nvGroupSet);*/
/*            nvGroupSet = NULL;*/
/*            return -ENOMEM;*/
            device->backend = LIKWID_NVMON_CUPTI_BACKEND;
        }

        NvmonFunctions* funcs = nvGroupSet->backends[device->backend];
        if (funcs)
        {
            if (funcs->createDevice)
                ret = funcs->createDevice(gpuIds[i], device);
        }
        if (ret < 0)
        {
            ERROR_PRINT(Cannot create device %d, gpuIds[i]);
            free(nvGroupSet->gpus);
            free(nvGroupSet);
            nvGroupSet = NULL;
            return -ENOMEM;
        }
    }

    nvmon_initialized = 1;
    return 0;
}



void
nvmon_finalize(void)
{
    if (nvmon_initialized && nvGroupSet)
    {
        for (int i = 0; i < nvGroupSet->numberOfGPUs; i++)
        {
            NvmonDevice_t device = &nvGroupSet->gpus[i];
            NvmonFunctions* funcs = nvGroupSet->backends[device->backend];
            if (funcs)
                funcs->freeDevice(&nvGroupSet->gpus[i]);
        }
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
        base->events[bidx].name = malloc((strlen(new->events[i].name)+1)*sizeof(char));
        if (base->events[bidx].name)
        {
            strncpy(base->events[bidx].name, new->events[i].name, strlen(new->events[i].name));
        }
        base->events[bidx].desc = malloc((strlen(new->events[i].desc)+1)*sizeof(char));
        if (base->events[bidx].desc)
        {
            strncpy(base->events[bidx].desc, new->events[i].desc, strlen(new->events[i].desc));
        }
        base->events[bidx].limit = malloc((strlen(new->events[i].limit)+1)*sizeof(char));
        if (base->events[bidx].limit)
        {
            strncpy(base->events[bidx].limit, new->events[i].limit, strlen(new->events[i].limit));
        }
    }
    return 0;
}



int
nvmon_getEventsOfGpu(int gpuId, NvmonEventList_t* list)
{
    int err = 0;
    int ret = topology_gpu_init();
    if (ret != EXIT_SUCCESS)
    {
        return -ENODEV;
    }
    GpuTopology_t gtopo = get_gpuTopology();
    int available = -1;
    for (int i = 0; i < gtopo->numDevices; i++)
    {
        if (gtopo->devices[i].devid == gpuId)
        {
            available = i;
            break;
        }
    }
    if (available >= 0)
    {
        err = nvmon_cupti_functions.getEventList(available, list);
/*        if (gtopo->devices[available].ccapMajor < 7)*/
/*        {*/
/*            err = nvmon_cupti_functions.getEventList(available, list);*/
/*        }*/
/*        else*/
/*        {*/
/*            ERROR_PRINT(NVIDIA PerfWorks API current not supported);*/
/*            return -ENODEV;*/
/*            //err = nvmon_perfworks_functions.getEventList(available, list);*/
/*        }*/
    }
    return err;

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
                free(out->name);
                free(out->desc);
                free(out->limit);
            }
            free(list->events);
            list->events = NULL;
            list->numEvents = 0;
        }
    }
}



int
nvmon_addEventSet(const char* eventCString)
{
    int i = 0, j = 0, k = 0, l = 0, m = 0, err = 0;
    int isPerfGroup = 0;

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
    init_configuration();
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
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Performance group);
        err = perfgroup_readGroup(config->groupPath, "nvidiagpu",
                         eventCString,
                         &nvGroupSet->groups[nvGroupSet->numberOfGroups-1]);
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
        isPerfGroup = 1;
    }
    bdestroy(eventBString);
    char * evstr = perfgroup_getEventStr(&nvGroupSet->groups[nvGroupSet->numberOfGroups-1]);
/*    eventBString = bfromcstr(evstr);*/
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, EventStr %s, evstr);
/*    eventtokens = bsplit(eventBString, ',');*/
/*    bdestroy(eventBString);*/

    for (devId = 0; devId < nvGroupSet->numberOfGPUs; devId++)
    {
        int err = 0;
        device = &nvGroupSet->gpus[devId];
        NvmonFunctions* funcs = nvGroupSet->backends[device->backend];
        if (funcs->addEvents)
        {
            err = funcs->addEvents(device, evstr);
        }
    }

    // Check whether group has any event in any device
    nvGroupSet->numberOfActiveGroups++;
    return (nvGroupSet->numberOfActiveGroups-1);
}


int nvmon_setupCounters(int gid)
{
    int i = 0;
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

            for (int j = 0; j < nvGroupSet->numberOfBackends; j++)
            {
                int err = 0;
                NvmonFunctions* funcs = nvGroupSet->backends[j];
                if (funcs->setupCounters)
                {
                    err = funcs->setupCounters(device, devEventSet);
                }
            }
        }
    }
    nvGroupSet->activeGroup = gid;

    return 0;
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
        for (int j = 0; j < nvGroupSet->numberOfBackends; j++)
        {
            NvmonFunctions* funcs = nvGroupSet->backends[j];
            if (funcs->startCounters)
            {
                err = funcs->startCounters(device);
            }
        }
    }

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
        for (int j = 0; j < nvGroupSet->numberOfBackends; j++)
        {
            NvmonFunctions* funcs = nvGroupSet->backends[j];
            if (funcs->stopCounters)
            {
                err = funcs->stopCounters(device);
            }
        }
    }

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
        for (int j = 0; j < nvGroupSet->numberOfBackends; j++)
        {
            NvmonFunctions* funcs = nvGroupSet->backends[j];
            if (funcs->readCounters)
            {
                err = funcs->readCounters(device);
            }
        }

    }
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
    NvmonDevice *device = &nvGroupSet->gpus[gpuId];
    if (groupId < 0 || groupId >= device->numNvEventSets)
    {
        return -EFAULT;
    }
    NvmonEventSet* evset = &device->nvEventSets[nvGroupSet->activeGroup];
    if (eventId < 0 || eventId >= evset->numberOfEvents)
    {
        return -EFAULT;
    }
    return evset->results[eventId].fullValue;
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
    NvmonDevice *device = &nvGroupSet->gpus[gpuId];
    if (groupId < 0 || groupId >= device->numNvEventSets)
    {
        return -EFAULT;
    }
    NvmonEventSet* evset = &device->nvEventSets[nvGroupSet->activeGroup];
    if (eventId < 0 || eventId >= evset->numberOfEvents)
    {
        return -EFAULT;
    }
    return evset->results[eventId].lastValue;
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
    int i = 0;
    double t = 0;
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return -EFAULT;
    }
    for (i = 0; i < nvGroupSet->numberOfGPUs; i++)
    {
        t = MAX(t, (double)(nvGroupSet->gpus[i].timeStop - nvGroupSet->gpus[i].timeStart));
    }
    return t*1E-9;
}


double nvmon_getLastTimeOfGroup(int groupId)
{
    int i = 0;
    double t = 0;
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return -EFAULT;
    }
    for (i = 0; i < nvGroupSet->numberOfGPUs; i++)
    {
        t = MAX(t, (double)(nvGroupSet->gpus[i].timeStop - nvGroupSet->gpus[i].timeRead));
    }
    return t*1E-9;
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

int nvmon_getGroups(char*** groups, char*** shortinfos, char*** longinfos)
{
    int ret = 0;
    init_configuration();
    Configuration_t config = get_configuration();
    ret = perfgroup_getGroups(config->groupPath, "nvidiagpu", groups, shortinfos, longinfos);
    return ret;
}

int nvmon_returnGroups(int nrgroups, char** groups, char** shortinfos, char** longinfos)
{
    perfgroup_returnGroups(nrgroups, groups, shortinfos, longinfos);
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
    
    char* f = ginfo->metricformulas[metricId];
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

    e = calc_metric(f, &clist, &result);
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
    char* f = ginfo->metricformulas[metricId];
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


    e = calc_metric(f, &clist, &result);
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
    for ( uint32_t i=0; i < regions; i++ )
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
    for ( uint32_t i=0; i < regions; i++ )
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
    char *f = ginfo->metricformulas[metricId];
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

    err = calc_metric(f, &clist, &result);
    if (err < 0)
    {
        ERROR_PRINT(Cannot calculate formula %s, f);
        return NAN;
    }
    destroy_clist(&clist);
    return result;
}
