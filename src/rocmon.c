 /* =======================================================================================
 *
 *      Filename:  rocmon.c
 *
 *      Description:  Main implementation of the performance monitoring module
 *                    for AMD GPUs
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
#ifdef LIKWID_WITH_ROCMON

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include <types.h>
#include <sys/types.h>
#include <inttypes.h>

#include <likwid.h>
#include <bstrlib.h>
#include <error.h>
#include <dlfcn.h>

#include <types.h>
#ifdef HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE
#undef HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE
#endif

#include <rocmon_common_types.h>
int likwid_rocmon_verbosity = DEBUGLEV_ONLY_ERROR;
static int rocmon_initialized = 0;
static RocmonContext* rocmon_context = NULL;

// Include backends
#include <rocmon_smi_types.h>
#include <rocmon_smi.h>
#ifdef LIKWID_ROCPROF_SDK
#include <rocmon_sdk_types.h>
#include <rocmon_sdk.h>
#endif
#include <rocmon_v1_types.h>
#include <rocmon_v1.h>

//#include <amd_smi/amdsmi.h>

const char* rocprofiler_group_arch = "amd_gpu";

void
rocmon_finalize(void)
{
    if ((!rocmon_initialized) || (rocmon_context == NULL))
    {
        rocmon_context = NULL;
        rocmon_initialized = 0;
        return;
    }
    if (rocmon_context->use_rocprofiler_v1)
    {
        rocmon_v1_finalize(rocmon_context);
    }
#ifdef LIKWID_ROCPROF_SDK
    else
    {
        rocmon_sdk_finalize(rocmon_context);
    }
#endif

    rocmon_smi_finalize(rocmon_context);

    if (rocmon_context->devices)
    {
        for (int i = 0; i < rocmon_context->numDevices; i++)
        {
            RocmonDevice* dev = &rocmon_context->devices[i];
            if (dev->groupResults)
            {
                for (int j = 0; j < dev->numGroupResults; j++)
                {
                    RocmonEventResultList* l = &dev->groupResults[j];
                    if (l->results)
                    {
                        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Destroy group result %d for device %d, j, dev->deviceId);
                        free(l->results);
                        l->results = NULL;
                        l->numResults = 0;
                    }
                }
                ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Destroy group results for device %d, dev->deviceId);
                free(dev->groupResults);
                dev->groupResults = NULL;
            }
        }
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Destroy devices);
        free(rocmon_context->devices);
        rocmon_context->devices = NULL;
        rocmon_context->numDevices = 0;
    }
    if (rocmon_context->groups)
    {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Destroy groups);
        free(rocmon_context->groups);
        rocmon_context->groups = NULL;
        rocmon_context->numGroups = 0;
        rocmon_context->numActiveGroups = 0;
        rocmon_context->activeGroup = -1;
    }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Destroy context);
    free(rocmon_context);
    rocmon_context = NULL;

    rocmon_initialized = FALSE;
    return;
}

int
rocmon_init(int numGpus, const int* gpuIds)
{
    int err = 0;

    // check if already initialized
    if (rocmon_initialized)
    {
        return 0;
    }
    // Validate arguments
    if (numGpus <= 0)
    {
        ERROR_PRINT(Number of gpus must be greater than 0 but only %d given, numGpus);
        return -EINVAL;
    }
    if (!gpuIds)
    {
        ERROR_PRINT(Invalid GPU list);
        return -EINVAL;
    }

    // Initialize other parts
    init_configuration();

    // Allocate memory for context
    rocmon_context = (RocmonContext*) malloc(sizeof(RocmonContext));
    if (rocmon_context == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate Rocmon context);
        return -ENOMEM;
    }
    memset(rocmon_context, 0, sizeof(RocmonContext));
    rocmon_context->groups = NULL;
    rocmon_context->devices = NULL;

#ifdef LIKWID_ROCPROF_SDK
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Initializing RocProfiler SDK);
    err = rocmon_sdk_init(rocmon_context, numGpus, gpuIds);
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Initializing RocProfiler SDK returned %d, err);
#else
    err = -1;
#endif
    if (err != 0)
    {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Initializing RocProfiler V1);
        err = rocmon_v1_init(rocmon_context, numGpus, gpuIds);
        if (err == 0)
        {
            rocmon_context->use_rocprofiler_v1 = 1;
        }
        else
        {
            ERROR_PRINT(Failed to initialize Rocprofiler v1 and SDK);
            free(rocmon_context);
            rocmon_context = NULL;
            return err;
        }
    }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Initializing ROCm SMI);
    err = rocmon_smi_init(rocmon_context, numGpus, gpuIds);
    if (err != 0)
    {
        // Only fail if there are no devices -> neither v1 nor sdk added them
        if (rocmon_context->devices == NULL)
        {
            ERROR_PRINT(Failed to initialize Rocprofiler SMI);
            free(rocmon_context);
            rocmon_context = NULL;
            return err;
        }
    }
    rocmon_context->state = ROCMON_STATE_INITIALIZED;
    rocmon_initialized = TRUE;
    return err;
}

int find_colon(const char* str)
{
    for (int i = 0; i < strlen(str); i++)
    {
        if (str[i] == ':')
        {
            return 1;
        }
    }
    return 0;
}

static int
_rocmon_parse_eventstring(const char* eventString, const char* arch, GroupInfo* group)
{
    int err = 0;
    const char colon = ':';
    Configuration_t config = get_configuration();

    if ((strstr(eventString, &colon) != NULL) || (find_colon(eventString)))
    {
        // If custom group -> perfgroup_customGroup
        err = perfgroup_customGroup(eventString, group);
        if (err < 0)
        {
            ERROR_PRINT(Cannot transform %s to performance group, eventString);
            return err;
        }
    }
    else
    {
        // If performance group -> perfgroup_readGroup
        err = perfgroup_readGroup(config->groupPath, arch, eventString, group);
        if (err == -EACCES)
        {
            ERROR_PRINT(Access to performance group %s not allowed, eventString);
            return err;
        }
        else if (err == -ENODEV)
        {
            ERROR_PRINT(Performance group %s only available with deactivated HyperThreading, eventString);
            return err;
        }
        if (err < 0)
        {
            ERROR_PRINT(Cannot read performance group %s for %s, eventString, arch);
            return err;
        }
    }

    return 0;
}

int
rocmon_addEventSet(const char* eventString, int* gid)
{
    int ret = 0;
    GroupInfo group = {};
    // Check arguments
    if ((!gid) || (!eventString))
    {
        return -EINVAL;
    }

    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        ERROR_PRINT(ROCMON not initialized);
        return -EFAULT;
    }

    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Adding Eventstring %s, eventString);
    ret = _rocmon_parse_eventstring(eventString, rocprofiler_group_arch, &group);
    if (ret < 0)
    {
        return ret;
    }

    // Allocate memory for event group if necessary
    if (rocmon_context->numActiveGroups == rocmon_context->numGroups)
    {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Increasing group space to %d, rocmon_context->numGroups+1);
        GroupInfo* tmpInfo = (GroupInfo*) realloc(rocmon_context->groups, (rocmon_context->numGroups+1) * sizeof(GroupInfo));
        if (tmpInfo == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate additional group);
            return -ENOMEM;
        }
        rocmon_context->groups = tmpInfo;
        rocmon_context->numGroups++;
    }

    // Allocate memory for event results
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Allocate result space);
    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];

        // Allocate memory for event results
        int numEvents = group.nevents;
        RocmonEventResult* tmpResults = (RocmonEventResult*) malloc(numEvents * sizeof(RocmonEventResult));
        if (tmpResults == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate event results);
            return -ENOMEM;
        }
        memset(tmpResults, 0, numEvents * sizeof(RocmonEventResult));

        // Allocate memory for new event result list entry
        RocmonEventResultList* tmpGroupResults = (RocmonEventResultList*) realloc(device->groupResults, (device->numGroupResults+1) * sizeof(RocmonEventResultList));
        if (tmpGroupResults == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate new event group result list);
            return -ENOMEM;
        }
        device->groupResults = tmpGroupResults;
        device->groupResults[device->numGroupResults].results = tmpResults;
        device->groupResults[device->numGroupResults].numResults = numEvents;
        device->numGroupResults++;
    }

    rocmon_context->groups[rocmon_context->numActiveGroups] = group;
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Eventstring %s got GID %d, eventString, rocmon_context->numActiveGroups);
    *gid = rocmon_context->numActiveGroups;
    rocmon_context->numActiveGroups++;
    return 0;
}



int
rocmon_setupCounters(int gid)
{
    int ret;

    // Check arguments
    if (gid < 0 || gid >= rocmon_context->numActiveGroups)
    {
        ERROR_PRINT(Invalid eventset ID %d, gid);
        return -EINVAL;
    }
    
    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        ERROR_PRINT(Rocmon not initialized);
        return -EFAULT;
    }
    if ((rocmon_context->state != ROCMON_STATE_STOPPED) && (rocmon_context->state != ROCMON_STATE_INITIALIZED))
    {
        ERROR_PRINT(Rocmon not in a valid state to setup -> %d, rocmon_context->state);
        return -EFAULT;
    }

    // Get group info
    GroupInfo* group = &rocmon_context->groups[gid];

    //
    // Separate rocprofiler and SMI events
    //
    int numSmiEvents = 0, numRocEvents = 0;


    // Go through each event and sort it
    for (int i = 0; i < group->nevents; i++)
    {
        const char* name = group->events[i];
        if (strncmp(name, "RSMI_", 5) == 0)
        {
            // RSMI event
            numSmiEvents++;
        }
        else if (strncmp(name, "ROCP_", 5) == 0)
        {
            // Rocprofiler event
            numRocEvents++;
        }
        else
        {
            // Unknown event
            ERROR_PRINT(Event '%s' has no prefix ('ROCP_' or 'RSMI_'), name);
            return -EINVAL;
        }
    }

    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];
        device->numActiveSmiEvents = 0;
        device->numActiveRocEvents = 0;
    }

    // Add rocprofiler events
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, SETUP ROCPROFILER WITH %d events, numRocEvents);
    if (rocmon_context->use_rocprofiler_v1)
    {
        ret = rocmon_v1_setupCounters(rocmon_context, gid);
    }
#ifdef LIKWID_ROCPROF_SDK
    else
    {
        ret = rocmon_sdk_setupCounters(rocmon_context, gid);
    }
#endif
    if (ret < 0)
    {
        ERROR_PRINT(Setting up rocprofiler counters failed);
/*        free(smiEvents);*/
/*        free(rocEvents);*/
        return ret;
    }

    // Add SMI events
    if (numSmiEvents > 0)
    {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, SETUP ROCM SMI WITH %d events, numSmiEvents);
        ret = rocmon_smi_setupCounters(rocmon_context, gid);
        if (ret < 0)
        {
            ERROR_PRINT(Setting up SMI counters failed);
/*            free(smiEvents);*/
/*            free(rocEvents);*/
            return ret;
        }
    }
    else
    {
        for (int i = 0; i < rocmon_context->numDevices; i++)
        {
            RocmonDevice* device = &rocmon_context->devices[i];
            device->numActiveSmiEvents = 0;
        }
    }

    // Add events to each device
    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];
        device->activeGroup = gid;
    }
    rocmon_context->activeGroup = gid;
    rocmon_context->state = ROCMON_STATE_SETUP;
/*    // Cleanup*/
/*    free(smiEvents);*/
/*    free(rocEvents);*/

    return 0;
}



int
rocmon_startCounters(void)
{
    int ret = 0;

    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        ERROR_PRINT(ROCMON not initialized);
        return -EFAULT;
    }
    if ((rocmon_context->activeGroup < 0) || (rocmon_context->state != ROCMON_STATE_SETUP))
    {
        ERROR_PRINT(No eventset configured for ROCMON);
        return -EFAULT;
    }
    if (rocmon_context->use_rocprofiler_v1)
    {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Starting ROCMON rocprofiler_v1 counters);
        ret = rocmon_v1_startCounters(rocmon_context);
    }
#ifdef LIKWID_ROCPROF_SDK
    else
    {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Starting ROCMON rocprofiler_sdk counters);
        ret = rocmon_sdk_startCounters(rocmon_context);
    }
#endif
    if (ret < 0)
    {
        return ret;
    }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Starting ROCMON SMI counters);
    ret = rocmon_smi_startCounters(rocmon_context);
    if (ret < 0)
    {
        return ret;
    }
    rocmon_context->state = ROCMON_STATE_RUNNING;
    return 0;
}


int
rocmon_stopCounters(void)
{
    int ret = 0;

    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }
    if ((rocmon_context->activeGroup < 0) || (rocmon_context->state != ROCMON_STATE_RUNNING))
    {
        return -EFAULT;
    }
    if (rocmon_context->use_rocprofiler_v1)
    {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Stopping ROCMON rocprofiler_v1 counters);
        ret = rocmon_v1_stopCounters(rocmon_context);
    }
#ifdef LIKWID_ROCPROF_SDK
    else
    {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Stopping ROCMON rocprofiler_sdk counters);
        ret = rocmon_sdk_stopCounters(rocmon_context);
    }
#endif
    if (ret < 0)
    {
        return ret;
    }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Stopping ROCMON SMI counters);
    ret = rocmon_smi_stopCounters(rocmon_context);
    if (ret < 0)
    {
        return ret;
    }
    rocmon_context->state = ROCMON_STATE_STOPPED;
    return 0;
}

int
rocmon_readCounters(void)
{
    int ret = 0;

    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }
    if ((rocmon_context->activeGroup < 0) || (rocmon_context->state != ROCMON_STATE_RUNNING))
    {
        return -EFAULT;
    }
    if (rocmon_context->use_rocprofiler_v1)
    {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Reading ROCMON rocprofiler_v1 counters);
        ret = rocmon_v1_readCounters(rocmon_context);
    }
#ifdef LIKWID_ROCPROF_SDK
    else
    {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Reading ROCMON rocprofiler_sdk counters);
        ret = rocmon_sdk_readCounters(rocmon_context);
    }
#endif
    if (ret < 0)
    {
        ERROR_PRINT(Failed to read ROCMON rocprofiler counters);
        return ret;
    }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Reading ROCMON SMI counters);
    ret = rocmon_smi_readCounters(rocmon_context);
    if (ret < 0)
    {
        ERROR_PRINT(Failed to read ROCMON SMI counters);
        return ret;
    }
    return 0;
}


int
rocmon_getEventsOfGpu(int gpuIdx, EventList_rocm_t* list)
{
    int ret = 0;
    EventList_rocm_t l = malloc(sizeof(EventList_rocm));
    if (!l)
    {
        return -ENOMEM;
    }
    memset(l, 0, sizeof(EventList_rocm));
    if (rocmon_context->use_rocprofiler_v1)
    {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Adding RocProfiler V1 events);
        ret = rocmon_v1_getEventsOfGpu(rocmon_context, gpuIdx, &l);
    }
#ifdef LIKWID_ROCPROF_SDK
    else
    {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Adding RocProfiler SDK events);
        ret = rocmon_sdk_getEventsOfGpu(rocmon_context, gpuIdx, &l);
    }
#endif
    if (ret < 0)
    {
        rocmon_freeEventsOfGpu(l);
        return ret;
    }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Adding ROCm SMI events);
    ret = rocmon_smi_getEventsOfGpu(rocmon_context, gpuIdx, &l);
    if (ret < 0)
    {
        rocmon_freeEventsOfGpu(l);
        return ret;
    }
    *list = l;
    return 0;
}

void
rocmon_freeEventsOfGpu(EventList_rocm_t list)
{
    if (!list)
    {
        return;
    }
    if (list->events != NULL)
    {
        for (int i = 0; i < list->numEvents; i++)
        {
            Event_rocm_t* event = &list->events[i];
            if (event->name) {
                free(event->name);
                event->name = NULL;
            }
            if (event->description) {
                free(event->description);
                event->description = NULL;
            }
        }
        free(list->events);
        list->events = NULL;
    }
    free(list);
    return;
}


int
rocmon_switchActiveGroup(int newGroupId)
{
    int ret = 0;
    if (rocmon_context->use_rocprofiler_v1)
    {
        ret = rocmon_v1_switchActiveGroup(rocmon_context, newGroupId);
    }
#ifdef LIKWID_ROCPROF_SDK
    else
    {
        ret = rocmon_sdk_switchActiveGroup(rocmon_context, newGroupId);
    }
#endif
    if (ret < 0)
    {
        return ret;
    }
    ret = rocmon_smi_switchActiveGroup(rocmon_context, newGroupId);
    if (ret < 0)
    {
        return ret;
    }
    return 0;
}



void rocmon_setVerbosity(int level)
{
    if (level >= DEBUGLEV_ONLY_ERROR && level <= DEBUGLEV_DEVELOP)
    {
        likwid_rocmon_verbosity = level;
    }
}



double
rocmon_getResult(int gpuIdx, int groupId, int eventId)
{
    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }

    // Validate gpuIdx
    if (gpuIdx < 0 || gpuIdx >= rocmon_context->numDevices)
    {
        return -EFAULT;
    }

    // Validate groupId
    RocmonDevice* device = &rocmon_context->devices[gpuIdx];
    if (groupId < 0 || groupId >= device->numGroupResults)
    {
        return -EFAULT;
    }

    // Validate eventId
    RocmonEventResultList* groupResult = &device->groupResults[groupId];
    if (eventId < 0 || eventId >= groupResult->numResults)
    {
        return -EFAULT;
    }

    // Return result
    return groupResult->results[eventId].fullValue;
}


// TODO: multiple groups
double
rocmon_getLastResult(int gpuIdx, int groupId, int eventId)
{
    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }

    // Validate gpuIdx
    if (gpuIdx < 0 || gpuIdx >= rocmon_context->numDevices)
    {
        return -EFAULT;
    }

    // Validate groupId
    RocmonDevice* device = &rocmon_context->devices[gpuIdx];
    if (groupId < 0 || groupId >= device->numGroupResults)
    {
        return -EFAULT;
    }

    // Validate eventId
    RocmonEventResultList* groupResult = &device->groupResults[groupId];
    if (eventId < 0 || eventId >= groupResult->numResults)
    {
        return -EFAULT;
    }

    // Return result
    return groupResult->results[eventId].lastValue;
}


int
rocmon_getNumberOfGroups(void)
{
    if (!rocmon_context || !rocmon_initialized)
    {
        return -EFAULT;
    }
    return rocmon_context->numActiveGroups;
}


int
rocmon_getIdOfActiveGroup(void)
{
    if (!rocmon_context || !rocmon_initialized)
    {
        return -EFAULT;
    }
    return rocmon_context->activeGroup;
}


int
rocmon_getNumberOfGPUs(void)
{
    if (!rocmon_context || !rocmon_initialized)
    {
        return -EFAULT;
    }
    return rocmon_context->numDevices;
}


int
rocmon_getNumberOfEvents(int groupId)
{
    if (!rocmon_context || !rocmon_initialized || (groupId < 0) || groupId >= rocmon_context->numActiveGroups)
    {
        return -EFAULT;
    }
    GroupInfo* ginfo = &rocmon_context->groups[groupId];
    return ginfo->nevents;
}


int
rocmon_getNumberOfMetrics(int groupId)
{
    if (!rocmon_context || !rocmon_initialized || (groupId < 0) || groupId > rocmon_context->numActiveGroups)
    {
        return -EFAULT;
    }
    GroupInfo* ginfo = &rocmon_context->groups[groupId];
    return ginfo->nmetrics;
}


double
rocmon_getTimeOfGroup(int groupId)
{
    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }
    return 0;
}


double
rocmon_getLastTimeOfGroup(int groupId)
{
    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }
    return 0;
}


double
rocmon_getTimeToLastReadOfGroup(int groupId)
{
    return 0;
}


char*
rocmon_getEventName(int groupId, int eventId)
{
    if (!rocmon_context || !rocmon_initialized || (groupId < 0) || groupId >= rocmon_context->numActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &rocmon_context->groups[groupId];
    if ((eventId < 0) || (eventId >= ginfo->nevents))
    {
        return NULL;
    }
    return ginfo->events[eventId];
}


char*
rocmon_getCounterName(int groupId, int eventId)
{
    if (!rocmon_context || !rocmon_initialized || (groupId < 0) || groupId >= rocmon_context->numActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &rocmon_context->groups[groupId];
    if ((eventId < 0) || (eventId >= ginfo->nevents))
    {
        return NULL;
    }
    return ginfo->counters[eventId];
}


char*
rocmon_getMetricName(int groupId, int metricId)
{
    if (!rocmon_context || !rocmon_initialized || (groupId < 0) || groupId >= rocmon_context->numActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &rocmon_context->groups[groupId];
    if ((metricId < 0) || (metricId >= ginfo->nmetrics))
    {
        return NULL;
    }
    return ginfo->metricnames[metricId];
}


char* 
rocmon_getGroupName(int groupId)
{
    if (!rocmon_context || !rocmon_initialized || (groupId < 0) || groupId >= rocmon_context->numActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &rocmon_context->groups[groupId];
    return ginfo->groupname;
}


char*
rocmon_getGroupInfoShort(int groupId)
{
    if (!rocmon_context || !rocmon_initialized || (groupId < 0) || groupId >= rocmon_context->numActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &rocmon_context->groups[groupId];
    return ginfo->shortinfo;
}


char*
rocmon_getGroupInfoLong(int groupId)
{
    if (!rocmon_context || !rocmon_initialized || (groupId < 0) || groupId >= rocmon_context->numActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &rocmon_context->groups[groupId];
    return ginfo->longinfo;
}

int
rocmon_getGroups(char*** groups, char*** shortinfos, char*** longinfos)
{
    init_configuration();
    Configuration_t config = get_configuration();


    return perfgroup_getGroups(config->groupPath, rocprofiler_group_arch, groups, shortinfos, longinfos);
}


int
rocmon_returnGroups(int nrgroups, char** groups, char** shortinfos, char** longinfos)
{
    perfgroup_returnGroups(nrgroups, groups, shortinfos, longinfos);
}



// only used internally by the ROCMON MarkerAPI
GroupInfo* rocmon_get_group(int gid)
{
    if ((gid >= 0) && (gid < rocmon_context->numActiveGroups))
    {
        return &rocmon_context->groups[gid];
    }
    return NULL;
}


#endif /* LIKWID_WITH_ROCMON */
