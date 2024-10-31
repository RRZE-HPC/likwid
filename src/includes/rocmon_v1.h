/*
 * =======================================================================================
 *
 *      Filename:  rocmon_v1.h
 *
 *      Description:  Header File of rocmon module for ROCm < 6.2.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tg), thomas.gruber@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2019 RRZE, University Erlangen-Nuremberg
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
#ifndef LIKWID_ROCMON_V1_H
#define LIKWID_ROCMON_V1_H

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

#include <likwid.h>

#include <rocmon_common_types.h>
#include <rocmon_v1_types.h>
#include <rocmon_smi_types.h>



// #include <hsa.h>
// #include <rocprofiler.h>
// #include <hsa/hsa_ext_amd.h>

// Variables
static void *rocmon_v1_dl_hsa_lib = NULL;
static void *rocmon_v1_dl_profiler_lib = NULL;


static bool rocmon_v1_initialized = FALSE;

// Macros
#ifndef FREE_IF_NOT_NULL
#define FREE_IF_NOT_NULL(var) if ( var ) { free( var ); var = NULL; }
#endif

#ifndef ROCM_CALL
#define ROCM_CALL( call, args, handleerror )                                  \
    do {                                                                \
        hsa_status_t _status = (*call##_ptr)args;                                  \
        if (_status != HSA_STATUS_SUCCESS && _status != HSA_STATUS_INFO_BREAK) {           \
            fprintf(stderr, "Error: function %s failed with error %d\n", #call, _status); \
            const char* err = NULL; \
            rocprofiler_error_string(&err); \
            if (err) fprintf(stderr, "Error: %s\n", err); \
            handleerror;                                                \
        }                                                               \
    } while (0)
#endif


// ROCm function declarations
#ifndef DECLAREFUNC_HSA
#define DECLAREFUNC_HSA(funcname, funcsig) hsa_status_t ROCMWEAK funcname funcsig;  hsa_status_t ( *funcname##_ptr ) funcsig;
#endif

DECLAREFUNC_HSA(hsa_init, ());
DECLAREFUNC_HSA(hsa_shut_down, ());
DECLAREFUNC_HSA(hsa_iterate_agents, (hsa_status_t (*callback)(hsa_agent_t agent, void* data), void* data));
DECLAREFUNC_HSA(hsa_agent_get_info, (hsa_agent_t agent, hsa_agent_info_t attribute, void* value));
DECLAREFUNC_HSA(hsa_system_get_info, (hsa_system_info_t attribute, void *value));

DECLAREFUNC_HSA(rocprofiler_iterate_info, (const hsa_agent_t* agent, rocprofiler_info_kind_t kind, hsa_status_t (*callback)(const rocprofiler_info_data_t, void* data), void* data));
DECLAREFUNC_HSA(rocprofiler_close, (rocprofiler_t* context));
DECLAREFUNC_HSA(rocprofiler_open, (hsa_agent_t agent, rocprofiler_feature_t* features, uint32_t feature_count, rocprofiler_t** context, uint32_t mode, rocprofiler_properties_t* properties));
DECLAREFUNC_HSA(rocprofiler_error_string, ());
DECLAREFUNC_HSA(rocprofiler_start, (rocprofiler_t* context, uint32_t group_index));
DECLAREFUNC_HSA(rocprofiler_stop, (rocprofiler_t* context, uint32_t group_index));
DECLAREFUNC_HSA(rocprofiler_read, (rocprofiler_t* context, uint32_t group_index));
DECLAREFUNC_HSA(rocprofiler_get_data, (rocprofiler_t* context, uint32_t group_index));
DECLAREFUNC_HSA(rocprofiler_get_metrics, (const rocprofiler_t* context));



// ----------------------------------------------------
//   Rocmon helper functions
// ----------------------------------------------------

static int
_rocmon_v1_link_libraries()
{
    #define DLSYM_AND_CHECK( dllib, name ) name##_ptr = dlsym( dllib, #name ); if ( dlerror() != NULL ) { ERROR_PRINT(Failed to link  #name); return -1; }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Linking AMD ROCMm V1 libraries);
  
    // Need to link in the ROCm HSA libraries
    rocmon_v1_dl_hsa_lib = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!rocmon_v1_dl_hsa_lib)
    {
        ERROR_PRINT(ROCm HSA library libhsa-runtime64.so not found: %s, dlerror());
        return -1;
    }

    // Need to link in the Rocprofiler libraries
    rocmon_v1_dl_profiler_lib = dlopen("librocprofiler64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!rocmon_v1_dl_profiler_lib)
    {
        rocmon_v1_dl_profiler_lib = dlopen("librocprofiler64.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!rocmon_v1_dl_profiler_lib)
        {
            ERROR_PRINT(Rocprofiler library librocprofiler64.so not found: %s, dlerror());
            return -1;
        }
    }

    // Link HSA functions
    DLSYM_AND_CHECK(rocmon_v1_dl_hsa_lib, hsa_init);
    DLSYM_AND_CHECK(rocmon_v1_dl_hsa_lib, hsa_shut_down);
    DLSYM_AND_CHECK(rocmon_v1_dl_hsa_lib, hsa_iterate_agents);
    DLSYM_AND_CHECK(rocmon_v1_dl_hsa_lib, hsa_agent_get_info);
    DLSYM_AND_CHECK(rocmon_v1_dl_hsa_lib, hsa_system_get_info);

    // Link Rocprofiler functions
    DLSYM_AND_CHECK(rocmon_v1_dl_profiler_lib, rocprofiler_iterate_info);
    DLSYM_AND_CHECK(rocmon_v1_dl_profiler_lib, rocprofiler_close);
    DLSYM_AND_CHECK(rocmon_v1_dl_profiler_lib, rocprofiler_open);
    DLSYM_AND_CHECK(rocmon_v1_dl_profiler_lib, rocprofiler_error_string);
    DLSYM_AND_CHECK(rocmon_v1_dl_profiler_lib, rocprofiler_start);
    DLSYM_AND_CHECK(rocmon_v1_dl_profiler_lib, rocprofiler_stop);
    DLSYM_AND_CHECK(rocmon_v1_dl_profiler_lib, rocprofiler_read);
    DLSYM_AND_CHECK(rocmon_v1_dl_profiler_lib, rocprofiler_get_data);
    DLSYM_AND_CHECK(rocmon_v1_dl_profiler_lib, rocprofiler_get_metrics);

    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Linking AMD ROCMm V1 libraries done);
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
_rocmon_v1_iterate_info_callback_count(const rocprofiler_info_data_t info, void* data)
{
    RocmonDevice* device = (RocmonDevice*) data;
    if (device) {
        device->numRocMetrics++;
    }
    return HSA_STATUS_SUCCESS;
}

static void
_rocmon_v1_print_rocprofiler_info_data(const rocprofiler_info_data_t info)
{
    if (info.kind != ROCPROFILER_INFO_KIND_METRIC)
    {
        return;
    }
    printf("Name '%s':\n", info.metric.name);
    printf("\tKind: '%s'\n", (info.kind == ROCPROFILER_INFO_KIND_METRIC ? "Metric" : "Trace"));
    printf("\tInstances: %d\n", info.metric.instances);
    printf("\tDescription: '%s'\n", info.metric.description);
    printf("\tExpression: '%s'\n", info.metric.expr);
    printf("\tBlockName: '%s'\n", info.metric.block_name);
    printf("\tBlockCounters: %d\n", info.metric.block_counters);
}

static hsa_status_t 
_rocmon_v1_iterate_info_callback_add(const rocprofiler_info_data_t info, void* data)
{
    iterate_info_cb_arg* arg = (iterate_info_cb_arg*) data;

    //ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, _rocmon_iterate_info_callback_add);
/*    if (likwid_rocmon_verbosity == DEBUGLEV_DEVELOP)*/
/*    {*/
/*        _rocmon_v1_print_rocprofiler_info_data(info);*/
/*    }*/
    // Check info kind
    if (info.kind != ROCPROFILER_INFO_KIND_METRIC)
    {
        ERROR_PRINT(Wrong info kind %u, info.kind);
        return HSA_STATUS_ERROR;
    }

    // Check index
    if (arg->currIndex >= arg->device->numRocMetrics)
    {
        ERROR_PRINT(Metric index out of bounds: %d, arg->currIndex);
        return HSA_STATUS_ERROR;
    }

    // Copy info data
    rocprofiler_info_data_t* target_info = &arg->device->v1_rocMetrics[arg->currIndex];
    memcpy(target_info, &info, sizeof(rocprofiler_info_data_t));
    arg->currIndex++;

    return HSA_STATUS_SUCCESS;
}


static hsa_status_t
_rocmon_v1_iterate_agents_callback(hsa_agent_t agent, void* argv)
{
    // Count number of callback invocations as the devices id
    static int nextDeviceId = 0;
    int deviceId = nextDeviceId;
    bool noAgent = false;

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
    //ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Initializing agent %d, gpuIndex);

    // Add agent to context
    RocmonDevice *device = &arg->context->devices[gpuIndex];
    device->deviceId = deviceId;
    device->hsa_agent = agent;
    device->v1_context = NULL;
    device->numActiveRocEvents = 0;
    device->v1_activeRocEvents = NULL;
    device->numGroupResults = 0;
    device->groupResults = NULL;

    // Get number of available metrics
    device->numRocMetrics = 0;
    ROCM_CALL(rocprofiler_iterate_info, (&agent, ROCPROFILER_INFO_KIND_METRIC, _rocmon_v1_iterate_info_callback_count, device), return HSA_STATUS_ERROR);
    //ROCMON_DEBUG_PRINT(DEBUGLEV_INFO, RocProfiler provides %d events, device->numRocMetrics);

    // workaround for bug in ROCm 5.4.0
    if(device->numRocMetrics == 0) {
        ROCM_CALL(rocprofiler_iterate_info, (NULL, ROCPROFILER_INFO_KIND_METRIC, _rocmon_v1_iterate_info_callback_count, device), return HSA_STATUS_ERROR);
        noAgent = true;
    }

    // Allocate memory for metrics
    device->v1_rocMetrics = (rocprofiler_info_data_t*) malloc(device->numRocMetrics * sizeof(rocprofiler_info_data_t));
    if (device->v1_rocMetrics == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate set of v1_rocMetrics);
        return HSA_STATUS_ERROR;
    }

    // Fetch metric informatino
    iterate_info_cb_arg info_arg = {
        .device = device,
        .currIndex = 0,
    };
    //ROCMON_DEBUG_PRINT(DEBUGLEV_INFO, Read %d RocProfiler events for device %d, device->numRocMetrics, device->deviceId);

    // If the call fails with agent, call rocprofiler_iterate_info without agent
    if(noAgent)
    {
        ROCM_CALL(rocprofiler_iterate_info, (NULL, ROCPROFILER_INFO_KIND_METRIC, _rocmon_v1_iterate_info_callback_add, &info_arg), return HSA_STATUS_ERROR);
    } else {
        ROCM_CALL(rocprofiler_iterate_info, (&agent, ROCPROFILER_INFO_KIND_METRIC, _rocmon_v1_iterate_info_callback_add, &info_arg), return HSA_STATUS_ERROR);
    }

    return HSA_STATUS_SUCCESS;
}





static int
_rocmon_v1_get_timestamp(uint64_t* timestamp_ns)
{
    uint64_t timestamp;

    // Get timestamp from system
    ROCM_CALL(hsa_system_get_info, (HSA_SYSTEM_INFO_TIMESTAMP, &timestamp), return -1);
    // Convert to nanoseconds
    *timestamp_ns = (uint64_t)((long double)timestamp * rocmon_context->hsa_timestamp_factor);

    return 0;
}


static int
_rocmon_v1_getLastResult(RocmonDevice* device, int eventId, double* value)
{
    rocprofiler_data_t* data = &device->v1_activeRocEvents[eventId].data;

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


static int
_rocmon_readCounters_rocprofiler_v1(RocmonDevice* device)
{
    int ret;

    // Check if there are any counters to start
    if (device->numActiveRocEvents <= 0)
    {
        return 0;
    }

    if (!device->v1_context)
    {
        return 0;
    }

    ROCM_CALL(rocprofiler_read, (device->v1_context, 0), return -1);
    ROCM_CALL(rocprofiler_get_data, (device->v1_context, 0), return -1);
    ROCM_CALL(rocprofiler_get_metrics, (device->v1_context), return -1);

    // Update results
    RocmonEventResultList* groupResult = &device->groupResults[rocmon_context->activeGroup];
    for (int i = 0; i < device->numActiveRocEvents; i++)
    {
        RocmonEventResult* result = &groupResult->results[i];
        
        // Read value
        ret = _rocmon_v1_getLastResult(device, i, &result->fullValue);
        if (ret < 0)
        {
            return -1;
        }

        // Calculate delta since last read
        result->lastValue = result->fullValue - result->lastValue;
    }

    return 0;
}



int
_rocmon_v1_readCounters(RocmonContext* context, uint64_t* (*getDestTimestampFunc)(RocmonDevice* device))
{
    int ret;

    // Get timestamp
    uint64_t timestamp;
    if (ret = _rocmon_v1_get_timestamp(&timestamp))
    {
        return ret;
    }

    for (int i = 0; i < context->numDevices; i++)
    {
        RocmonDevice* device = &context->devices[i];
        if (!device->rocprof_v1) continue;

        // Save timestamp
        if (getDestTimestampFunc)
        {
            uint64_t* timestampDest = getDestTimestampFunc(device);
            if (timestampDest)
            {
                *timestampDest = timestamp;
            }
        }

        // Read rocprofiler counters
        ret = _rocmon_readCounters_rocprofiler_v1(device);
        if (ret < 0) return ret;
    }

    return 0;
}


static uint64_t*
_rocmon_v1_get_read_time(RocmonDevice* device)
{
    return &device->time.read;
}


static uint64_t*
_rocmon_v1_get_stop_time(RocmonDevice* device)
{
    return &device->time.stop;
}


int
rocmon_v1_init(RocmonContext* context, int numGpus, const int* gpuIds)
{
    hsa_status_t status = 0;
    RocmonDevice* devices = NULL;
    int num_devices = 0;

    // check if already initialized
    if (rocmon_v1_initialized)
    {
        return 0;
    }
    if (context == NULL)
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
    int ret = _rocmon_v1_link_libraries();
    if (ret < 0)
    {
        ERROR_PLAIN_PRINT(Failed to initialize libraries);
        return ret;
    }

    // init hsa library
    //ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Initializing HSA);
    ROCM_CALL(hsa_init, (),
    {
        ERROR_PLAIN_PRINT(Failed to init hsa library);
        goto rocmon_init_hsa_failed;
    });

    if (!context->devices)
    {
        context->devices = (RocmonDevice*) malloc(numGpus * sizeof(RocmonDevice));
        if (!context->devices)
        {
            ERROR_PLAIN_PRINT(Cannot allocate set of GPUs);
            free(devices);
            return -ENOMEM;
        }
        context->numDevices = numGpus;
    }
    // Get hsa timestamp factor
    uint64_t frequency_hz;
    //ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Getting HSA timestamp factor);
    ROCM_CALL(hsa_system_get_info, (HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency_hz),
    {
        ERROR_PLAIN_PRINT(Failed to get HSA timestamp factor);
        goto rocmon_init_info_agents_failed;
    });
    context->hsa_timestamp_factor = (long double)1000000000 / (long double)frequency_hz;

    // initialize structures for specified devices (fetch ROCm specific info)
    iterate_agents_cb_arg arg = {
        .context = context,
        .numGpus = numGpus,
        .gpuIds = gpuIds,
    };
    //ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Iterating through %d available agents, numGpus);
    ROCM_CALL(hsa_iterate_agents, (_rocmon_v1_iterate_agents_callback, &arg),
    {
        ERROR_PRINT(Error while iterating through available agents);
        goto rocmon_init_info_agents_failed;
    });

    rocmon_v1_initialized = TRUE;
    return 0;
rocmon_init_info_agents_failed:
    ROCM_CALL(hsa_shut_down, (), {
        // fall through
    });
rocmon_init_hsa_failed:
    free(context->devices);
    context->devices = NULL;
    context->numDevices = 0;
    return -1;
}


void
rocmon_v1_finalize(RocmonContext* context)
{

    if (!rocmon_v1_initialized)
    {
        return;
    }
    //ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Finalize LIKWID ROCMON);

    if (context)
    {
        if (context->devices)
        {
            // Free each devices fields
            for (int i = 0; i < context->numDevices; i++)
            {
                RocmonDevice* device = &context->devices[i];
                if (device->rocprof_v1)
                {
                    FREE_IF_NOT_NULL(device->v1_rocMetrics);
                    FREE_IF_NOT_NULL(device->v1_activeRocEvents);
                }
                if (device->v1_context)
                {
                    ROCM_CALL(rocprofiler_close, (device->v1_context),);
                }
            }
        }
    }

    ROCM_CALL(hsa_shut_down, (), {
        ERROR_PRINT(DEBUGLEV_DEVELOP, Shutdown HSA failed);
        // fall through
    });
}


/*int*/
/*rocmon_v1_addEventSet(const char* eventString, int* gid)*/
/*{*/
/*    // Check arguments*/
/*    if (!eventString)*/
/*    {*/
/*        return -EINVAL;*/
/*    }*/
/*    */
/*    // Ensure rocmon is initialized*/
/*    if (!rocmon_v1_initialized)*/
/*    {*/
/*        return -EFAULT;*/
/*    }*/

/*    // Allocate memory for event group if necessary*/
/*    if (rocmon_context->numActiveGroups == rocmon_context->numGroups)*/
/*    {*/
/*        GroupInfo* tmpInfo = (GroupInfo*) realloc(rocmon_context->groups, (rocmon_context->numGroups+1) * sizeof(GroupInfo));*/
/*        if (tmpInfo == NULL)*/
/*        {*/
/*            ERROR_PLAIN_PRINT(Cannot allocate additional group);*/
/*            return -ENOMEM;*/
/*        }*/
/*        rocmon_context->groups = tmpInfo;*/
/*        rocmon_context->numGroups++;*/
/*    }*/

/*    // Parse event string*/
/*    int err = _rocmon_v1_parse_eventstring(eventString, &rocmon_context->groups[rocmon_context->numActiveGroups]);*/
/*    if (err < 0)*/
/*    {*/
/*        return err;*/
/*    }*/

/*    */

/*    *gid = rocmon_context->numActiveGroups;*/
/*    rocmon_context->numActiveGroups++;*/
/*    return 0;*/
/*}*/


int
_rocmon_setupCounters_rocprofiler_v1(RocmonDevice* device, const char** events, int numEvents)
{
    // Close previous rocprofiler context
    if (device->v1_context)
    {
        //ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Closing previous rocprofiler context);
        ROCM_CALL(rocprofiler_close, (device->v1_context), return -1);
    }

    // Look if the are any events
    if (numEvents <= 0)
    {
        return 0;
    }

    // Create feature array to monitor
    rocprofiler_feature_t* features = (rocprofiler_feature_t*) malloc(numEvents * sizeof(rocprofiler_feature_t));
    if (features == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate feature list);
        return -ENOMEM;
    }
    for (int i = 0; i < numEvents; i++)
    {
        features[i].kind = ROCPROFILER_FEATURE_KIND_METRIC;
        features[i].name = events[i];
        //ROCMON_DEBUG_PRINT(DEBUGLEV_DEBUG, Setup ROCMON rocprofiler_v1 counter %d %s, i, events[i]);
    }

    // Free previous feature array if present
    FREE_IF_NOT_NULL(device->v1_activeRocEvents);

    device->numActiveRocEvents = numEvents;
    device->v1_activeRocEvents = features;

    // Open context
    rocprofiler_properties_t properties = {};
    properties.queue_depth = 128;
    uint32_t mode = ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_CREATEQUEUE | ROCPROFILER_MODE_SINGLEGROUP;

    // Important: only a single profiling group is supported at this time which limits the number of events that can be monitored at a time.
    ROCM_CALL(rocprofiler_open, (device->hsa_agent, device->v1_activeRocEvents, device->numActiveRocEvents, &device->v1_context, mode, &properties), return -1);

    return 0;
}


int
rocmon_v1_setupCounters(RocmonContext* context, int gid)
{
    int ret;

    // Check arguments
    if (gid < 0 || gid >= context->numActiveGroups)
    {
        return -EINVAL;
    }
    
    // Ensure rocmon is initialized
    if (!rocmon_v1_initialized)
    {
        return -EFAULT;
    }

    // Get group info
    GroupInfo* group = &context->groups[gid];

    //
    // Separate rocprofiler and SMI events
    //
    const char **rocEvents = NULL;
    int numRocEvents = 0;

    // Allocate memory for string arrays
    rocEvents = (const char**) malloc(group->nevents * sizeof(const char*));
    if (rocEvents == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate rocEvent name array);
        return -ENOMEM;
    }

    // Go through each event and sort it
    for (int i = 0; i < group->nevents; i++)
    {
        const char* name = group->events[i];
        if (strncmp(name, "ROCP_", 5) == 0)
        {
            // Rocprofiler event
            rocEvents[numRocEvents] = name + 5; // +5 removes 'ROCP_' prefix
            numRocEvents++;
        }
    }

    // Add events to each device
    for (int i = 0; i < context->numDevices; i++)
    {
        RocmonDevice* device = &context->devices[i];

        // Add rocprofiler events
        //ROCMON_DEBUG_PRINT(DEBUGLEV_INFO, SETUP ROCPROFILER WITH %d events, numRocEvents);
        ret = _rocmon_setupCounters_rocprofiler_v1(device, rocEvents, numRocEvents);
        if (ret < 0)
        {
            free(rocEvents);
            return ret;
        }
    }
    // Cleanup
    free(rocEvents);

    return 0;
}


static int
_rocmon_startCounters_rocprofiler_v1(RocmonDevice* device)
{
    // Check if there are any counters to start
    if (device->numActiveRocEvents <= 0)
    {
        return 0;
    }

    // Reset results
    RocmonEventResultList* groupResult = &device->groupResults[rocmon_context->activeGroup];
    for (int i = 0; i < device->numActiveRocEvents; i++)
    {
        RocmonEventResult* result = &groupResult->results[i];
        result->lastValue = 0;
        result->fullValue = 0;
    }

    if (device->v1_context)
    {
        ROCM_CALL(rocprofiler_start, (device->v1_context, 0), return -1);
    }

    return 0;
}



int
rocmon_v1_startCounters(RocmonContext* context)
{
    int ret;

    // Ensure rocmon is initialized
    if (!rocmon_v1_initialized)
    {
        return -EFAULT;
    }

    // Get timestamp
    uint64_t timestamp;
    if (ret = _rocmon_v1_get_timestamp(&timestamp))
    {
        return ret;
    }

    // Start counters on each device
    for (int i = 0; i < context->numDevices; i++)
    {
        RocmonDevice* device = &context->devices[i];
        device->time.start = timestamp;
        device->time.read = timestamp;

        // Start rocprofiler events
        ret = _rocmon_startCounters_rocprofiler_v1(device);
        if (ret < 0) return ret;

        // Start SMI events
/*        _rocmon_startCounters_smi(device);*/
/*        if (ret < 0) return ret;*/
    }

    return 0;
}


static int
_rocmon_stopCounters_rocprofiler_v1(RocmonDevice* device)
{
    if (device->v1_context)
    {
        // Close context
        ROCM_CALL(rocprofiler_stop, (device->v1_context, 0), return -1);
    }

    return 0;
}


int
rocmon_v1_stopCounters(RocmonContext* context)
{
    int ret;

    // Ensure rocmon is initialized
    if (!rocmon_v1_initialized)
    {
        return -EFAULT;
    }

    // Read counters
    ret = _rocmon_v1_readCounters(context, &_rocmon_v1_get_stop_time);
    if (ret < 0) return ret;

    for (int i = 0; i < context->numDevices; i++)
    {
        RocmonDevice* device = &context->devices[i];

        // Stop rocprofiler events
        ret = _rocmon_stopCounters_rocprofiler_v1(device);
        if (ret < 0) return ret;

        // Nothing to stop for SMI events
    }

    return 0;
}


int
rocmon_v1_readCounters(RocmonContext* context)
{
    int ret;

    // Ensure rocmon is initialized
    if (!rocmon_v1_initialized)
    {
        return -EFAULT;
    }

    // Read counters
    ret = _rocmon_v1_readCounters(context, &_rocmon_v1_get_read_time);
    if (ret < 0) return ret;

    return 0;
}


int
rocmon_v1_getEventsOfGpu(RocmonContext* context, int gpuIdx, EventList_rocm_t* list)
{
    EventList_rocm_t tmpList = NULL;
    Event_rocm_t* tmpEventList = NULL;
    // Ensure rocmon is initialized
    if (!rocmon_v1_initialized)
    {
        return -EFAULT;
    }
    // Validate args
    if ((gpuIdx < 0) || (gpuIdx > context->numDevices) || (!list))
    {
        return -EINVAL;
    }

    RocmonDevice* device = &context->devices[gpuIdx];

    if (*list)
    {
        tmpList = *list;
    }
    else
    {
        // Allocate list structure
        EventList_rocm_t tmpList = (EventList_rocm_t) malloc(sizeof(EventList_rocm));
        if (tmpList == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate event list);
            return -ENOMEM;
        }
        memset(tmpList, 0, sizeof(EventList_rocm));
    }

    // Get number of events
    printf("Number of events %d\n", device->numRocMetrics);
    
    if (device->numRocMetrics == 0)
    {
        // No events -> return list
        *list = tmpList;
        return 0;
    }
    // (Re-)Allocate event array
    tmpEventList = realloc(tmpList->events, (tmpList->numEvents + device->numRocMetrics) * sizeof(Event_rocm_t));
    if (!tmpEventList)
    {
        if (!*list) free(tmpList);
        ERROR_PLAIN_PRINT(Cannot allocate events for event list);
        return -ENOMEM;
    }
    tmpList->events = tmpEventList;
    int startindex = tmpList->numEvents;

    // Copy rocprofiler event information
    for (int i = 0; i < device->numRocMetrics; i++)
    {
        rocprofiler_info_data_t* event = &device->v1_rocMetrics[i];
        Event_rocm_t* out = &tmpList->events[startindex + i];
        int len;

        // Copy name
        printf("Name %s\n", event->metric.name);
        len = strlen(event->metric.name) + 5 /* Prefix */ + 1 /* NULL byte */;
        out->name = (char*) malloc(len);
        if (out->name)
        {
            snprintf(out->name, len, "ROCP_%s", event->metric.name);
        }

        // Copy description
        len = strlen(event->metric.description) + 1 /* NULL byte */;
        out->description = (char*) malloc(len);
        if (out->description)
        {
            snprintf(out->description, len, "%s", event->metric.description);
        }
        tmpList->numEvents++;
    }
    *list = tmpList;
    return 0;
}


int
rocmon_v1_switchActiveGroup(RocmonContext* context, int newGroupId)
{
    int ret;

    ret = rocmon_v1_stopCounters(context);
    if (ret < 0)
    {
        return ret;
    }

    ret = rocmon_v1_setupCounters(context, newGroupId);
    if (ret < 0)
    {
        return ret;
    }

    ret = rocmon_v1_startCounters(context);
    if (ret < 0)
    {
        return ret;
    }

    return 0;
}



#endif /* LIKWID_ROCMON_V1_H */

