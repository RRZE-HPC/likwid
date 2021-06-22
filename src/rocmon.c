/*
 * =======================================================================================
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

#include <likwid.h>
#include <bstrlib.h>
#include <error.h>
#include <dlfcn.h>

#include <likwid.h>
#include <rocmon_types.h>

// #include <hsa.h>
// #include <rocprofiler.h>
// #include <hsa/hsa_ext_amd.h>

// Variables
static void *dl_hsa_lib = NULL;
static void *dl_profiler_lib = NULL;

RocmonContext *rocmon_context = NULL;
static bool rocmon_initialized = FALSE;

// Macros
#define ROCM_CALL( call, args, handleerror )                                  \
    do {                                                                \
        hsa_status_t _status = (*call##_ptr)args;                                  \
        if (_status != HSA_STATUS_SUCCESS && _status != HSA_STATUS_INFO_BREAK) {           \
            fprintf(stderr, "Error: function %s failed with error %d.\n", #call, _status); \
            handleerror;                                                \
        }                                                               \
    } while (0)

// ROCm function declarations
#define ROCMWEAK __attribute__(( weak ))
#define DECLAREROCMFUNC(funcname, funcsig) hsa_status_t ROCMWEAK funcname funcsig;  hsa_status_t ( *funcname##_ptr ) funcsig;

DECLAREROCMFUNC(hsa_init, ());
DECLAREROCMFUNC(hsa_shut_down, ());
DECLAREROCMFUNC(hsa_iterate_agents, (hsa_status_t (*callback)(hsa_agent_t agent, void* data), void* data));
DECLAREROCMFUNC(hsa_agent_get_info, (hsa_agent_t agent, hsa_agent_info_t attribute, void* value));
DECLAREROCMFUNC(hsa_system_get_info, (hsa_system_info_t attribute, void *value));
DECLAREROCMFUNC(rocprofiler_iterate_info, (const hsa_agent_t* agent, rocprofiler_info_kind_t kind, hsa_status_t (*callback)(const rocprofiler_info_data_t, void* data), void* data));
DECLAREROCMFUNC(rocprofiler_close, (rocprofiler_t* context));
DECLAREROCMFUNC(rocprofiler_open, (hsa_agent_t agent, rocprofiler_feature_t* features, uint32_t feature_count, rocprofiler_t** context, uint32_t mode, rocprofiler_properties_t* properties));
DECLAREROCMFUNC(rocprofiler_error_string, ());
DECLAREROCMFUNC(rocprofiler_start, (rocprofiler_t* context, uint32_t group_index));
DECLAREROCMFUNC(rocprofiler_stop, (rocprofiler_t* context, uint32_t group_index));
DECLAREROCMFUNC(rocprofiler_read, (rocprofiler_t* context, uint32_t group_index));
DECLAREROCMFUNC(rocprofiler_get_data, (rocprofiler_t* context, uint32_t group_index));
DECLAREROCMFUNC(rocprofiler_get_metrics, (const rocprofiler_t* context));

static int
_rocmon_link_libraries()
{
    #define DLSYM_AND_CHECK( dllib, name ) name##_ptr = dlsym( dllib, #name ); if ( dlerror() != NULL ) { return -1; }

    // Need to link in the ROCm HSA libraries
    dl_hsa_lib = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_hsa_lib)
    {
        fprintf(stderr, "ROCm HSA library libhsa-runtime64.so not found.\n");
        return -1;
    }

    // Need to link in the Rocprofiler libraries
    dl_profiler_lib = dlopen("librocprofiler64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_profiler_lib)
    {
        fprintf(stderr, "Rocprofiler library libhsa-runtime64.so not found.\n");
        return -1;
    }

    // Link HSA functions
    DLSYM_AND_CHECK(dl_hsa_lib, hsa_init);
    DLSYM_AND_CHECK(dl_hsa_lib, hsa_shut_down);
    DLSYM_AND_CHECK(dl_hsa_lib, hsa_iterate_agents);
    DLSYM_AND_CHECK(dl_hsa_lib, hsa_agent_get_info);
    DLSYM_AND_CHECK(dl_hsa_lib, hsa_system_get_info);

    // Link Rocprofiler functions
    DLSYM_AND_CHECK(dl_profiler_lib, rocprofiler_iterate_info);
    DLSYM_AND_CHECK(dl_profiler_lib, rocprofiler_close);
    DLSYM_AND_CHECK(dl_profiler_lib, rocprofiler_open);
    DLSYM_AND_CHECK(dl_profiler_lib, rocprofiler_error_string);
    DLSYM_AND_CHECK(dl_profiler_lib, rocprofiler_start);
    DLSYM_AND_CHECK(dl_profiler_lib, rocprofiler_stop);
    DLSYM_AND_CHECK(dl_profiler_lib, rocprofiler_read);
    DLSYM_AND_CHECK(dl_profiler_lib, rocprofiler_get_data);
    DLSYM_AND_CHECK(dl_profiler_lib, rocprofiler_get_metrics);

    return 0;
}

typedef struct {
    RocmonContext* context;
    int numGpus;
    const int* gpuIds;
} iterate_agents_cb_arg;

typedef struct {
    RocmonDevice* device;
    int currIndex;
} iterate_info_cb_arg;


static hsa_status_t 
_rocmon_iterate_info_callback_count(const rocprofiler_info_data_t info, void* data)
{
    RocmonDevice* device = (RocmonDevice*) data;
    device->numAllMetrics++;
    return HSA_STATUS_SUCCESS;
}


static hsa_status_t 
_rocmon_iterate_info_callback_add(const rocprofiler_info_data_t info, void* data)
{
    iterate_info_cb_arg* arg = (iterate_info_cb_arg*) data;

    // Check info kind
    if (info.kind != ROCPROFILER_INFO_KIND_METRIC)
    {
        ERROR_PRINT(Wrong info kind %u, info.kind);
        return HSA_STATUS_ERROR;
    }

    // Check index
    if (arg->currIndex >= arg->device->numAllMetrics)
    {
        ERROR_PRINT(Metric index out of bounds: %d, arg->currIndex);
        return HSA_STATUS_ERROR;
    }

    // Copy info data
    rocprofiler_info_data_t* target_info = &arg->device->allMetrics[arg->currIndex];
    memcpy(target_info, &info, sizeof(rocprofiler_info_data_t));
    arg->currIndex++;

    return HSA_STATUS_SUCCESS;
}


static hsa_status_t
_rocmon_iterate_agents_callback(hsa_agent_t agent, void* argv)
{
    // Count number of callback invocations as the devices id
    static int nextDeviceId = 0;
    int deviceId = nextDeviceId;

    iterate_agents_cb_arg *arg = (iterate_agents_cb_arg*) argv;

    // Check if device is a GPU
    hsa_device_type_t type;
    ROCM_CALL(hsa_agent_get_info, (agent, HSA_AGENT_INFO_DEVICE, &type), return -1);
    if (type != HSA_DEVICE_TYPE_GPU)
    {
        return HSA_STATUS_SUCCESS;
    }
    nextDeviceId++;

    // Check if device is includes in arg->gpuIds
    int gpuIndex = -1;
    for (int i = 0; i < arg->numGpus; i++)
    {
        if (deviceId == arg->gpuIds[i])
        {
            gpuIndex = i;
            break;
        }
    }
    if (gpuIndex < 0)
    {
        return HSA_STATUS_SUCCESS;
    }

    // Add agent to context
    RocmonDevice *device = &arg->context->devices[gpuIndex];
    device->deviceId = deviceId;
    device->hsa_agent = agent;
    device->context = NULL;
    device->numActiveEvents = 0;
    device->activeEvents = NULL;
    device->numGroupResults = 0;
    device->groupResults = NULL;

    // Get number of available metrics
    device->numAllMetrics = 0;
    ROCM_CALL(rocprofiler_iterate_info, (&agent, ROCPROFILER_INFO_KIND_METRIC, _rocmon_iterate_info_callback_count, device), return HSA_STATUS_ERROR);

    // Allocate memory for metrics
    device->allMetrics = (rocprofiler_info_data_t*) malloc(device->numAllMetrics * sizeof(rocprofiler_info_data_t));
    if (device->allMetrics == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate set of allMetrics);
        return HSA_STATUS_ERROR;
    }

    // Fetch metric informatino
    iterate_info_cb_arg info_arg = {
        .device = device,
        .currIndex = 0,
    };
    ROCM_CALL(rocprofiler_iterate_info, (&agent, ROCPROFILER_INFO_KIND_METRIC, _rocmon_iterate_info_callback_add, &info_arg), return HSA_STATUS_ERROR);

    return HSA_STATUS_SUCCESS;
}


static int
_rocmon_parse_eventstring(const char* eventString, GroupInfo* group)
{
    int err = 0;
    Configuration_t config = get_configuration();
    bstring eventBString = bfromcstr(eventString);

    if (bstrchrp(eventBString, ':', 0) != BSTR_ERR)
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
        err = perfgroup_readGroup(config->groupPath, "amd_gpu", eventString, group);
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
            ERROR_PRINT(Cannot read performance group %s, eventString);
            return err;
        }
    }

    return 0;
}


int
_rocmon_get_timestamp(uint64_t* timestamp_ns)
{
    uint64_t timestamp;

    // Get timestamp from system
    ROCM_CALL(hsa_system_get_info, (HSA_SYSTEM_INFO_TIMESTAMP, &timestamp), return -1);
    // Convert to nanoseconds
    *timestamp_ns = (uint64_t)((long double)timestamp * rocmon_context->hsa_timestamp_factor);

    return 0;
}


int
_rocmon_getLastResult(int gpuId, int eventId, double* value)
{
    RocmonDevice* device = &rocmon_context->devices[gpuId];
    rocprofiler_data_t* data = &device->activeEvents[eventId].data;

    switch (data->kind)
    {
	case ROCPROFILER_DATA_KIND_INT32:
        *value = (double) data->result_int32;
        break;
	case ROCPROFILER_DATA_KIND_INT64:
        *value = (double) data->result_int64;
        break;
	case ROCPROFILER_DATA_KIND_FLOAT:
        *value = (double) data->result_float;
        break;
	case ROCPROFILER_DATA_KIND_DOUBLE:
        *value = data->result_double;
        break;
        
	case ROCPROFILER_DATA_KIND_BYTES:
    case ROCPROFILER_DATA_KIND_UNINIT:
    default:
        return -1;
    }

    return 0;
}


int
rocmon_init(int numGpus, const int* gpuIds)
{
    hsa_status_t status;

    // check if already initialized
    if (rocmon_initialized)
    {
        return 0;
    }
    if (rocmon_context != NULL)
    {
        return -EEXIST;
    }

    // Validate arguments
    if (numGpus <= 0)
    {
        ERROR_PRINT(Number of gpus must be greater than 0 but only %d given, numGpus);
        return -EINVAL;
    }
    
    // Initialize other parts
    init_configuration();

    // initialize libraries
    int ret = _rocmon_link_libraries();
    if (ret < 0)
    {
        return ret;
    }

    // Allocate memory for context
    rocmon_context = (RocmonContext*) malloc(sizeof(RocmonContext));
    if (rocmon_context == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate Rocmon context);
        return -ENOMEM;
    }
    rocmon_context->groups = NULL;
    rocmon_context->numGroups = 0;
    rocmon_context->numActiveGroups = 0;

    rocmon_context->devices = (RocmonDevice*) malloc(numGpus * sizeof(RocmonDevice));
    rocmon_context->numDevices = numGpus;
    if (rocmon_context->devices == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate set of GPUs);
        free(rocmon_context);
        rocmon_context = NULL;
        return -ENOMEM;
    }

    // init hsa library
    ROCM_CALL(hsa_init, (),
    {
        free(rocmon_context->devices);
        free(rocmon_context);
        rocmon_context = NULL;
        return -1;
    });

    // Get hsa timestamp factor
    uint64_t frequency_hz;
    ROCM_CALL(hsa_system_get_info, (HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency_hz),
    {
        free(rocmon_context->devices);
        free(rocmon_context);
        rocmon_context = NULL;
        return -1;
    });
    rocmon_context->hsa_timestamp_factor = (long double)1000000000 / (long double)frequency_hz;

    // initialize structures for specified devices (fetch ROCm specific info)
    iterate_agents_cb_arg arg = {
        .context = rocmon_context,
        .numGpus = numGpus,
        .gpuIds = gpuIds,
    };
    ROCM_CALL(hsa_iterate_agents, (_rocmon_iterate_agents_callback, &arg),
    {
        free(rocmon_context->devices);
        free(rocmon_context);
        rocmon_context = NULL;
        return -1;
    });

    rocmon_initialized = TRUE;
    return 0;
}


void
rocmon_finalize(void)
{
#define FREE_IF_NOT_NULL(var) if ( var ) { free( var ); var = NULL; }

    RocmonContext* context = rocmon_context;

    if (!rocmon_initialized)
    {
        return;
    }

    if (context)
    {
        if (context->devices)
        {
            // Free each devices fields
            for (int i = 0; i < context->numDevices; i++)
            {
                RocmonDevice* device = &context->devices[i];
                FREE_IF_NOT_NULL(device->allMetrics);
                FREE_IF_NOT_NULL(device->activeEvents);
                if (device->groupResults)
                {
                    // Free events of event result lists
                    for (int j = 0; j < device->numGroupResults; j++)
                    {
                        FREE_IF_NOT_NULL(device->groupResults[i].results);
                    }
                    // Free list
                    free(device->groupResults);
                }
                if (device->context)
                {
                    ROCM_CALL(rocprofiler_close, (device->context),);
                }
            }

            free(context->devices);
            context->devices = NULL;
        }

        FREE_IF_NOT_NULL(context->groups);

        free(context);
        context = NULL;
    }

    (*hsa_shut_down_ptr)();
}


int
rocmon_addEventSet(const char* eventString, int* gid)
{
    // Check arguments
    if (!eventString)
    {
        return -EINVAL;
    }
    
    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }

    // Allocate memory for event group if necessary
    if (rocmon_context->numActiveGroups == rocmon_context->numGroups)
    {
        GroupInfo* tmpInfo = (GroupInfo*) realloc(rocmon_context->groups, (rocmon_context->numGroups+1) * sizeof(GroupInfo));
        if (tmpInfo == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate additional group);
            return -ENOMEM;
        }
        rocmon_context->groups = tmpInfo;
        rocmon_context->numGroups++;
    }

    // Parse event string
    int err = _rocmon_parse_eventstring(eventString, &rocmon_context->groups[rocmon_context->numActiveGroups]);
    if (err < 0)
    {
        return err;
    }

    // Allocate memory for event results
    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];

        // Allocate memory for event results
        int numEvents = rocmon_context->groups[rocmon_context->numActiveGroups].nevents;
        RocmonEventResult* tmpResults = (RocmonEventResult*) malloc(numEvents * sizeof(RocmonEventResult));
        if (tmpResults == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate event results);
            return -ENOMEM;
        }

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

    *gid = rocmon_context->numActiveGroups;
    rocmon_context->numActiveGroups++;
    return 0;
}


int
rocmon_setupCounters(int gid)
{
    // Check arguments
    if (gid < 0 || gid >= rocmon_context->numActiveGroups)
    {
        return -EINVAL;
    }
    
    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }

    // Add events to each device
    GroupInfo* group = &rocmon_context->groups[gid];
    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];

        // Close previous rocprofiler context
        if (device->context)
        {
            ROCM_CALL(rocprofiler_close, (device->context), return 1);
        }

        // Create feature array to monitor
        rocprofiler_feature_t* features = (rocprofiler_feature_t*) malloc(group->nevents * sizeof(rocprofiler_feature_t));
        if (features == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate feature list);
            return -ENOMEM;
        }
        for (int j = 0; j < group->nevents; j++)
        {
            features[j].kind = ROCPROFILER_FEATURE_KIND_METRIC;
            features[j].name = group->events[j];
        }

        device->numActiveEvents = group->nevents;
        device->activeEvents = features;

        // Open context
        rocprofiler_properties_t properties = {};
        properties.queue_depth = 128;
        uint32_t mode = ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_CREATEQUEUE | ROCPROFILER_MODE_SINGLEGROUP;

        // Important: only a single profiling group is supported at this time which limits the number of events that can be monitored at a time.
        ROCM_CALL(rocprofiler_open, (device->hsa_agent, device->activeEvents, device->numActiveEvents, &device->context, mode, &properties), return -1);
    }
    rocmon_context->activeGroup = gid;
    
    return 0;
}


int
rocmon_startCounters(void)
{
    int ret;

    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }

    // Get timestamp
    uint64_t timestamp;
    if (ret = _rocmon_get_timestamp(&timestamp))
    {
        return ret;
    }

    // Start counters on each device
    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];
        device->time.start = timestamp;
        device->time.read = timestamp;

        // Reset results
        RocmonEventResultList* groupResult = &device->groupResults[rocmon_context->activeGroup];
        for (int j = 0; j < groupResult->numResults; j++)
        {
            RocmonEventResult* result = &groupResult->results[j];
            result->lastValue = 0;
            result->fullValue = 0;
        }

        if (device->context)
        {
            ROCM_CALL(rocprofiler_start, (device->context, 0), return -1);
        }
    }

    return 0;
}


int
rocmon_stopCounters(void)
{
    int ret;

    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }

    // Get timestamp
    uint64_t timestamp;
    if (ret = _rocmon_get_timestamp(&timestamp))
    {
        return ret;
    }

    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];
        device->time.stop = timestamp;

        if (!device->context)
        {
            continue;
        }

        // Read values and close context
        ROCM_CALL(rocprofiler_read, (device->context, 0), return -1);
        ROCM_CALL(rocprofiler_get_data, (device->context, 0), return -1);
        ROCM_CALL(rocprofiler_get_metrics, (device->context), return -1);
        ROCM_CALL(rocprofiler_stop, (device->context, 0), return -1);

        // Update results
        RocmonEventResultList* groupResult = &device->groupResults[rocmon_context->activeGroup];
        for (int j = 0; j < groupResult->numResults; j++)
        {
            RocmonEventResult* result = &groupResult->results[j];
            
            // Read value
            ret = _rocmon_getLastResult(i, j, &result->fullValue);
            if (ret < 0)
            {
                return -1;
            }

            // Calculate delta since last read
            result->lastValue = result->fullValue - result->lastValue;
        }
    }

    return 0;
}


int
rocmon_readCounters(void)
{
    int ret;

    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }

    // Get timestamp
    uint64_t timestamp;
    if (ret = _rocmon_get_timestamp(&timestamp))
    {
        return ret;
    }

    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];
        device->time.read = timestamp;

        if (!device->context)
        {
            continue;
        }

        ROCM_CALL(rocprofiler_read, (device->context, 0), return -1);
        ROCM_CALL(rocprofiler_get_data, (device->context, 0), return -1);
        ROCM_CALL(rocprofiler_get_metrics, (device->context), return -1);

        // Update results
        RocmonEventResultList* groupResult = &device->groupResults[rocmon_context->activeGroup];
        for (int j = 0; j < groupResult->numResults; j++)
        {
            RocmonEventResult* result = &groupResult->results[j];
            
            // Read value
            ret = _rocmon_getLastResult(i, j, &result->fullValue);
            if (ret < 0)
            {
                return -1;
            }

            // Calculate delta since last read
            result->lastValue = result->fullValue - result->lastValue;
        }
    }

    return 0;
}


double
rocmon_getResult(int gpuId, int groupId, int eventId)
{
    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }

    // Validate gpuId
    if (gpuId < 0 || gpuId >= rocmon_context->numDevices)
    {
        return -EFAULT;
    }

    // Validate groupId
    RocmonDevice* device = &rocmon_context->devices[gpuId];
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
rocmon_getLastResult(int gpuId, int groupId, int eventId)
{
    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }

    // Validate gpuId
    if (gpuId < 0 || gpuId >= rocmon_context->numDevices)
    {
        return -EFAULT;
    }

    // Validate groupId
    RocmonDevice* device = &rocmon_context->devices[gpuId];
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
rocmon_getEventsOfGpu(int gpuId, EventList_rocm_t** list)
{
    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }

    // Validate args
    if (gpuId < 0 || gpuId > rocmon_context->numDevices)
    {
        return -EINVAL;
    }
    if (list == NULL)
    {
        return -EINVAL;
    }

    RocmonDevice* device = &rocmon_context->devices[gpuId];

    // Allocate list structure
    EventList_rocm_t* tmpList = (EventList_rocm_t*) malloc(sizeof(EventList_rocm_t));
    if (list == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate event list);
        return -ENOMEM;
    }
    
    // Get number of events
    tmpList->numEvents = device->numAllMetrics;
    if (tmpList->numEvents == 0)
    {
        // No events -> return empty list
        tmpList->events = NULL;
        *list = tmpList;
        return 0;
    }

    // Allocate event array
    tmpList->events = (Event_rocm_t*) malloc(tmpList->numEvents * sizeof(Event_rocm_t));
    if (tmpList->events == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate events for event list);
        free(tmpList);
        return -ENOMEM;
    }

    // Copy event information
    for (int i = 0; i < tmpList->numEvents; i++)
    {
        rocprofiler_info_data_t* event = &device->allMetrics[i];
        Event_rocm_t* out = &tmpList->events[i];

        // Copy name
        out->name = (char*) malloc(strlen(event->metric.name) + 2 /* NULL byte */);
        if (out->name)
        {
            int ret = snprintf(out->name, strlen(event->metric.name)+1, "%s", event->metric.name);
            if (ret > 0)
            {
                out->name[ret] = '\0';
            }
        }

        // Copy description
        out->description = (char*) malloc(strlen(event->metric.description) + 2 /* NULL byte */);
        if (out->description)
        {
            int ret = snprintf(out->description, strlen(event->metric.description)+1, "%s", event->metric.description);
            if (ret > 0)
            {
                out->description[ret] = '\0';
            }
        }

        // Copy instances
        out->instances = event->metric.instances;
    }

    *list = tmpList;
    return 0;
}

void
rocmon_freeEventsOfGpu(EventList_rocm_t* list)
{
#define FREE_IF_NOT_NULL(var) if ( var ) { free( var ); var = NULL; }

    // Check pointer
    if (list == NULL)
    {
        return;
    }

    if (list->events != NULL)
    {
        for (int i = 0; i < list->numEvents; i++)
        {
            Event_rocm_t* event = &list->events[i];
            FREE_IF_NOT_NULL(event->name);
            FREE_IF_NOT_NULL(event->description);
        }
        free(list->events);
    }
    free(list);
}


int
rocmon_switchActiveGroup(int newGroupId)
{
    int ret;

    ret = rocmon_stopCounters();
    if (ret < 0)
    {
        return ret;
    }

    ret = rocmon_setupCounters(newGroupId);
    if (ret < 0)
    {
        return ret;
    }

    ret = rocmon_startCounters();
    if (ret < 0)
    {
        return ret;
    }

    return 0;
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
    if (!rocmon_context || !rocmon_initialized || (groupId < 0) || groupId >= rocmon_context->numActiveGroups)
    {
        return -EFAULT;
    }
    GroupInfo* ginfo = &rocmon_context->groups[groupId];
    return ginfo->nmetrics;
}


double
rocmon_getTimeOfGroup(int groupId)
{
    int i = 0;
    double t = 0;
    if (!rocmon_context || !rocmon_initialized || (groupId < 0) || groupId >= rocmon_context->numActiveGroups)
    {
        return -EFAULT;
    }
    for (i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];
        t = MAX(t, (double)(device->time.stop - device->time.start));
    }
    return t*1E-9;
}


double
rocmon_getLastTimeOfGroup(int groupId)
{
    int i = 0;
    double t = 0;
    if (!rocmon_context || !rocmon_initialized || (groupId < 0) || groupId >= rocmon_context->numActiveGroups)
    {
        return -EFAULT;
    }
    for (i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];
        t = MAX(t, (double)(device->time.stop - device->time.read));
    }
    return t*1E-9;
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

    return perfgroup_getGroups(config->groupPath, "amd_gpu", groups, shortinfos, longinfos);
}


int
rocmon_returnGroups(int nrgroups, char** groups, char** shortinfos, char** longinfos)
{
    perfgroup_returnGroups(nrgroups, groups, shortinfos, longinfos);
}

#endif /* LIKWID_WITH_ROCMON */
