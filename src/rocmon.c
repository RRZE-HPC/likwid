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
#include <rocmon_types.h>
#include <dlfcn.h>

#include <hsa/hsa.h>
#include <rocprofiler.h>
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
rocmon_addEventSet(const char* eventString)
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

    // Add events to each device
    GroupInfo* group = &rocmon_context->groups[rocmon_context->numActiveGroups];
    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];
        
        // Create feature array to monitor
        int numFeatures = group->nevents;
        rocprofiler_feature_t* features = (rocprofiler_feature_t*) malloc(numFeatures * sizeof(rocprofiler_feature_t));
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

        // (Re-)create rocprofiler context
        if (device->context)
        {
            ROCM_CALL(rocprofiler_close, (device->context),
            {
                free(features);
                return 1;
            });
        }

        rocprofiler_properties_t properties = {};
        properties.queue_depth = 128;
        uint32_t mode = ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_CREATEQUEUE | ROCPROFILER_MODE_SINGLEGROUP;

        ROCM_CALL(rocprofiler_open, (device->hsa_agent, features, numFeatures, &device->context, mode, &properties), return -1);

        device->numActiveEvents = numFeatures;
        device->activeEvents = features;
    }

    rocmon_context->numActiveGroups++;
    return 0;
}


int
rocmon_startCounters(void)
{
    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }

    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];
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
    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }

    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];
        if (device->context)
        {
            ROCM_CALL(rocprofiler_stop, (device->context, 0), return -1);
        }
    }

    return 0;
}


int
rocmon_readCounters(void)
{
    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }

    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];
        if (!device->context)
        {
            continue;
        }

        ROCM_CALL(rocprofiler_read, (device->context, 0), return -1);
        ROCM_CALL(rocprofiler_get_data, (device->context, 0), return -1);
        ROCM_CALL(rocprofiler_get_metrics, (device->context), return -1);
    }

    return 0;
}

// TODO: multiple groups
double
rocmon_getLastResult(int gpuId, int eventId)
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

    RocmonDevice* device = &rocmon_context->devices[gpuId];

    // Validate eventId
    if (eventId < 0 || eventId >= device->numActiveEvents)
    {
        return -EFAULT;
    }

    rocprofiler_data_t* data = &device->activeEvents[eventId].data;
    printf("Data kind: %u\n", data->kind);
    switch (data->kind)
    {
	case ROCPROFILER_DATA_KIND_INT32:
        return (double) data->result_int32;
	case ROCPROFILER_DATA_KIND_INT64:
        return (double) data->result_int64;
	case ROCPROFILER_DATA_KIND_FLOAT:
        return (double) data->result_float;
	case ROCPROFILER_DATA_KIND_DOUBLE:
        return data->result_double;
        
	case ROCPROFILER_DATA_KIND_BYTES:
    case ROCPROFILER_DATA_KIND_UNINIT:
    default:
        return -1;
    }
}

#endif /* LIKWID_WITH_ROCMON */
