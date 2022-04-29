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
#include <inttypes.h>

#include <likwid.h>
#include <bstrlib.h>
#include <error.h>
#include <dlfcn.h>

#include <likwid.h>
#include <rocmon_types.h>
#include <rocm_smi/rocm_smi.h>

// #include <hsa.h>
// #include <rocprofiler.h>
// #include <hsa/hsa_ext_amd.h>

// Variables
static void *dl_hsa_lib = NULL;
static void *dl_profiler_lib = NULL;
static void *dl_rsmi_lib = NULL;

RocmonContext *rocmon_context = NULL;
static bool rocmon_initialized = FALSE;
int likwid_rocmon_verbosity = DEBUGLEV_ONLY_ERROR;

// Macros
#define membersize(type, member) sizeof(((type *) NULL)->member)
#define FREE_IF_NOT_NULL(var) if ( var ) { free( var ); var = NULL; }
#define ROCM_CALL( call, args, handleerror )                                  \
    do {                                                                \
        hsa_status_t _status = (*call##_ptr)args;                                  \
        if (_status != HSA_STATUS_SUCCESS && _status != HSA_STATUS_INFO_BREAK) {           \
            const char* err = NULL; \
            fprintf(stderr, "Error: function %s failed with error %d\n", #call, _status); \
            rocprofiler_error_string(&err); \
            fprintf(stderr, "Error: %s\n", err); \
            handleerror;                                                \
        }                                                               \
    } while (0)

#define RSMI_CALL( call, args, handleerror )                                  \
    do {                                                                \
        rsmi_status_t _status = (*call##_ptr)args;                                  \
        if (_status != RSMI_STATUS_SUCCESS) {           \
            fprintf(stderr, "Error: function %s failed with error %d.\n", #call, _status); \
            handleerror;                                                \
        }                                                               \
    } while (0)

// ROCm function declarations
#define ROCMWEAK __attribute__(( weak ))
#define DECLAREFUNC_HSA(funcname, funcsig) hsa_status_t ROCMWEAK funcname funcsig;  hsa_status_t ( *funcname##_ptr ) funcsig;
#define DECLAREFUNC_SMI(funcname, funcsig) rsmi_status_t ROCMWEAK funcname funcsig; rsmi_status_t ( *funcname##_ptr ) funcsig;

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

DECLAREFUNC_SMI(rsmi_init, (uint64_t flags));
DECLAREFUNC_SMI(rsmi_shut_down, ());
DECLAREFUNC_SMI(rsmi_dev_supported_func_iterator_open, (uint32_t dv_ind, rsmi_func_id_iter_handle_t* handle));
DECLAREFUNC_SMI(rsmi_dev_supported_variant_iterator_open, (rsmi_func_id_iter_handle_t obj_h, rsmi_func_id_iter_handle_t* var_iter));
DECLAREFUNC_SMI(rsmi_func_iter_value_get, (rsmi_func_id_iter_handle_t handle, rsmi_func_id_value_t* value ));
DECLAREFUNC_SMI(rsmi_func_iter_next, (rsmi_func_id_iter_handle_t handle));
DECLAREFUNC_SMI(rsmi_dev_supported_func_iterator_close, (rsmi_func_id_iter_handle_t* handle));
DECLAREFUNC_SMI(rsmi_dev_power_ave_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t* power));
DECLAREFUNC_SMI(rsmi_dev_pci_throughput_get, (uint32_t dv_ind, uint64_t* sent, uint64_t* received, uint64_t* max_pkt_sz));
DECLAREFUNC_SMI(rsmi_dev_pci_replay_counter_get, (uint32_t dv_ind, uint64_t* counter));
DECLAREFUNC_SMI(rsmi_dev_memory_total_get, (uint32_t dv_ind, rsmi_memory_type_t mem_type, uint64_t* total));
DECLAREFUNC_SMI(rsmi_dev_memory_usage_get, (uint32_t dv_ind, rsmi_memory_type_t mem_type, uint64_t* used ));
DECLAREFUNC_SMI(rsmi_dev_memory_busy_percent_get, (uint32_t dv_ind, uint32_t* busy_percent));
DECLAREFUNC_SMI(rsmi_dev_memory_reserved_pages_get, (uint32_t dv_ind, uint32_t* num_pages, rsmi_retired_page_record_t* records));
DECLAREFUNC_SMI(rsmi_dev_fan_rpms_get, (uint32_t dv_ind, uint32_t sensor_ind, int64_t* speed));
DECLAREFUNC_SMI(rsmi_dev_fan_speed_get, (uint32_t dv_ind, uint32_t sensor_ind, int64_t* speed));
DECLAREFUNC_SMI(rsmi_dev_fan_speed_max_get, (uint32_t dv_ind, uint32_t sensor_ind, uint64_t* max_speed));
DECLAREFUNC_SMI(rsmi_dev_temp_metric_get, (uint32_t dv_ind, uint32_t sensor_type, rsmi_temperature_metric_t metric, int64_t* temperature));
DECLAREFUNC_SMI(rsmi_dev_volt_metric_get, (uint32_t dv_ind, rsmi_voltage_type_t sensor_type, rsmi_voltage_metric_t metric, int64_t* voltage));
DECLAREFUNC_SMI(rsmi_dev_overdrive_level_get, (uint32_t dv_ind, uint32_t* od));
DECLAREFUNC_SMI(rsmi_dev_ecc_count_get, (uint32_t dv_ind, rsmi_gpu_block_t block, rsmi_error_count_t* ec));
DECLAREFUNC_SMI(rsmi_compute_process_info_get, (rsmi_process_info_t* procs, uint32_t* num_items));


// ----------------------------------------------------
//   SMI event wrapper
// ----------------------------------------------------

static int
_smi_wrapper_pci_throughput_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t value;
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, _smi_wrapper_pci_throughput_get(%d, %d), deviceId, event->extra);
    // Internal variant: 0 for sent, 1 for received bytes and 2 for max packet size
    if (event->extra == 0)       RSMI_CALL(rsmi_dev_pci_throughput_get, (deviceId, &value, NULL, NULL), return -1);
    else if (event->extra == 1)  RSMI_CALL(rsmi_dev_pci_throughput_get, (deviceId, NULL, &value, NULL), return -1);
    else if (event->extra == 2)  RSMI_CALL(rsmi_dev_pci_throughput_get, (deviceId, NULL, NULL, &value), return -1);
    else return -1;

    result->fullValue += value;
    result->lastValue = value;

    return 0;
}


static int
_smi_wrapper_pci_replay_counter_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t counter;
    RSMI_CALL(rsmi_dev_pci_replay_counter_get, (deviceId, &counter), return -1);
    result->fullValue += counter;
    result->lastValue = counter;

    return 0;
}


static int
_smi_wrapper_power_ave_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t power;
    RSMI_CALL(rsmi_dev_power_ave_get, (deviceId, event->subvariant, &power), return -1);
    result->fullValue += power;
    result->lastValue = power;

    return 0;
}


static int
_smi_wrapper_memory_total_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t total;
    RSMI_CALL(rsmi_dev_memory_total_get, (deviceId, event->variant, &total), return -1);
    result->fullValue += total;
    result->lastValue = total;

    return 0;
}


static int
_smi_wrapper_memory_usage_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t used;
    RSMI_CALL(rsmi_dev_memory_usage_get, (deviceId, event->variant, &used), return -1);
    result->fullValue += used;
    result->lastValue = used;

    return 0;
}


static int
_smi_wrapper_memory_busy_percent_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint32_t percent;
    RSMI_CALL(rsmi_dev_memory_busy_percent_get, (deviceId, &percent), return -1);
    result->fullValue += percent;
    result->lastValue = percent;

    return 0;
}


static int
_smi_wrapper_memory_reserved_pages_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint32_t num_pages;
    RSMI_CALL(rsmi_dev_memory_reserved_pages_get, (deviceId, &num_pages, NULL), return -1);
    result->fullValue += num_pages;
    result->lastValue = num_pages;

    return 0;
}


static int
_smi_wrapper_fan_rpms_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t speed;
    RSMI_CALL(rsmi_dev_fan_rpms_get, (deviceId, event->subvariant, &speed), return -1);
    result->fullValue += speed;
    result->lastValue = speed;

    return 0;
}


static int
_smi_wrapper_fan_speed_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t speed;
    RSMI_CALL(rsmi_dev_fan_speed_get, (deviceId, event->subvariant, &speed), return -1);
    result->fullValue += speed;
    result->lastValue = speed;

    return 0;
}


static int
_smi_wrapper_fan_speed_max_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t max_speed;
    RSMI_CALL(rsmi_dev_fan_speed_max_get, (deviceId, event->subvariant, &max_speed), return -1);
    result->fullValue += max_speed;
    result->lastValue = max_speed;

    return 0;
}


static int
_smi_wrapper_temp_metric_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t temperature;
    RSMI_CALL(rsmi_dev_temp_metric_get, (deviceId, event->subvariant, event->variant, &temperature), return -1);
    result->fullValue += temperature;
    result->lastValue = temperature;

    return 0;
}


static int
_smi_wrapper_volt_metric_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t voltage;
    RSMI_CALL(rsmi_dev_volt_metric_get, (deviceId, event->subvariant, event->variant, &voltage), return -1);
    result->fullValue += voltage;
    result->lastValue = voltage;

    return 0;
}


static int
_smi_wrapper_overdrive_level_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint32_t overdrive;
    RSMI_CALL(rsmi_dev_overdrive_level_get, (deviceId, &overdrive), return -1);
    result->fullValue += overdrive;
    result->lastValue = overdrive;

    return 0;
}


static int
_smi_wrapper_ecc_count_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    rsmi_error_count_t error_count;
    RSMI_CALL(rsmi_dev_ecc_count_get, (deviceId, event->variant, &error_count), return -1);

    if (event->extra == 0)
    {
        result->lastValue = error_count.correctable_err - result->fullValue;
        result->fullValue = error_count.correctable_err;
    }
    else if (event->extra == 1)
    {
        result->lastValue = error_count.uncorrectable_err - result->fullValue;
        result->fullValue = error_count.uncorrectable_err;
    }
    else
    {
        return -1;
    }

    return 0;
}


static int
_smi_wrapper_compute_process_info_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint32_t num_items;
    RSMI_CALL(rsmi_compute_process_info_get, (NULL, &num_items), return -1);
    result->fullValue += num_items;
    result->lastValue = num_items;

    return 0;
}


// ----------------------------------------------------
//   Rocmon helper functions
// ----------------------------------------------------

static int
_rocmon_link_libraries()
{
    #define DLSYM_AND_CHECK( dllib, name ) name##_ptr = dlsym( dllib, #name ); if ( dlerror() != NULL ) { ERROR_PRINT(Failed to link  #name); return -1; }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Linking AMD ROCMm libraries);
    // Need to link in the ROCm HSA libraries
    dl_hsa_lib = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_hsa_lib)
    {
        ERROR_PRINT(ROCm HSA library libhsa-runtime64.so not found);
        return -1;
    }

    // Need to link in the Rocprofiler libraries
    dl_profiler_lib = dlopen("librocprofiler64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_profiler_lib)
    {
        ERROR_PRINT(Rocprofiler library librocprofiler64.so not found);
        return -1;
    }

    // Need to link in the Rocprofiler libraries
    dl_rsmi_lib = dlopen("librocm_smi64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_rsmi_lib)
    {
        ERROR_PRINT(ROCm SMI library librocm_smi64.so not found);
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

    // Link SMI functions
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_init);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_shut_down);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_supported_func_iterator_open);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_supported_variant_iterator_open);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_func_iter_value_get);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_func_iter_next);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_supported_func_iterator_close);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_power_ave_get);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_pci_throughput_get);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_pci_replay_counter_get);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_memory_total_get);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_memory_usage_get);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_memory_busy_percent_get);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_memory_reserved_pages_get);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_fan_rpms_get);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_fan_speed_get);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_fan_speed_max_get);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_temp_metric_get);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_volt_metric_get);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_overdrive_level_get);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_dev_ecc_count_get);
    DLSYM_AND_CHECK(dl_rsmi_lib, rsmi_compute_process_info_get);
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Linking AMD ROCMm libraries done);
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
    if (device) {
        device->numRocMetrics++;
    }
    return HSA_STATUS_SUCCESS;
}

static void
_rocmon_print_rocprofiler_info_data(const rocprofiler_info_data_t info)
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
_rocmon_iterate_info_callback_add(const rocprofiler_info_data_t info, void* data)
{
    iterate_info_cb_arg* arg = (iterate_info_cb_arg*) data;

    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, _rocmon_iterate_info_callback_add);
    if (likwid_rocmon_verbosity == DEBUGLEV_DEVELOP)
    {
        _rocmon_print_rocprofiler_info_data(info);
    }
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
    rocprofiler_info_data_t* target_info = &arg->device->rocMetrics[arg->currIndex];
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
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Initializing agent %d, gpuIndex);

    // Add agent to context
    RocmonDevice *device = &arg->context->devices[gpuIndex];
    device->deviceId = deviceId;
    device->hsa_agent = agent;
    device->context = NULL;
    device->numActiveRocEvents = 0;
    device->activeRocEvents = NULL;
    device->numGroupResults = 0;
    device->groupResults = NULL;

    // Get number of available metrics
    device->numRocMetrics = 0;
    ROCM_CALL(rocprofiler_iterate_info, (&agent, ROCPROFILER_INFO_KIND_METRIC, _rocmon_iterate_info_callback_count, device), return HSA_STATUS_ERROR);
    ROCMON_DEBUG_PRINT(DEBUGLEV_INFO, RocProfiler provides %d events, device->numRocMetrics);

    // Allocate memory for metrics
    device->rocMetrics = (rocprofiler_info_data_t*) malloc(device->numRocMetrics * sizeof(rocprofiler_info_data_t));
    if (device->rocMetrics == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate set of rocMetrics);
        return HSA_STATUS_ERROR;
    }

    // Initialize SMI events map
    if (init_map(&device->smiMetrics, MAP_KEY_TYPE_STR, 0, &free) < 0)
    {
        ERROR_PLAIN_PRINT(Cannot init smiMetrics map);
        return HSA_STATUS_ERROR;
    }

    // Fetch metric informatino
    iterate_info_cb_arg info_arg = {
        .device = device,
        .currIndex = 0,
    };
    ROCMON_DEBUG_PRINT(DEBUGLEV_INFO, Read %d RocProfiler events for device %d, device->numRocMetrics, device->deviceId);
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


static int
_rocmon_get_timestamp(uint64_t* timestamp_ns)
{
    uint64_t timestamp;

    // Get timestamp from system
    ROCM_CALL(hsa_system_get_info, (HSA_SYSTEM_INFO_TIMESTAMP, &timestamp), return -1);
    // Convert to nanoseconds
    *timestamp_ns = (uint64_t)((long double)timestamp * rocmon_context->hsa_timestamp_factor);

    return 0;
}


static int
_rocmon_getLastResult(RocmonDevice* device, int eventId, double* value)
{
    rocprofiler_data_t* data = &device->activeRocEvents[eventId].data;

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
_rocmon_readCounters_rocprofiler(RocmonDevice* device)
{
    int ret;

    // Check if there are any counters to start
    if (device->numActiveRocEvents <= 0)
    {
        return 0;
    }

    if (!device->context)
    {
        return 0;
    }

    ROCM_CALL(rocprofiler_read, (device->context, 0), return -1);
    ROCM_CALL(rocprofiler_get_data, (device->context, 0), return -1);
    ROCM_CALL(rocprofiler_get_metrics, (device->context), return -1);

    // Update results
    RocmonEventResultList* groupResult = &device->groupResults[rocmon_context->activeGroup];
    for (int i = 0; i < device->numActiveRocEvents; i++)
    {
        RocmonEventResult* result = &groupResult->results[i];
        
        // Read value
        ret = _rocmon_getLastResult(device, i, &result->fullValue);
        if (ret < 0)
        {
            return -1;
        }

        // Calculate delta since last read
        result->lastValue = result->fullValue - result->lastValue;
    }

    return 0;
}


static int
_rocmon_readCounters_smi(RocmonDevice* device)
{
    // Check if there are any counters to start
    if (device->numActiveSmiEvents <= 0)
    {
        return 0;
    }

    // Save baseline values
    RocmonEventResultList* groupResult = &device->groupResults[rocmon_context->activeGroup];
    for (int i = 0; i < device->numActiveSmiEvents; i++)
    {
        double value = 0;
        RocmonSmiEvent* event = &device->activeSmiEvents[i];
        RocmonEventResult* result = &groupResult->results[device->numActiveRocEvents+i];

        // Measure counter
        if (event->measureFunc)
        {
            event->measureFunc(device->deviceId, event, result);
        }
    }

    return 0;
}


static int
_rocmon_readCounters(uint64_t* (*getDestTimestampFunc)(RocmonDevice* device))
{
    int ret;

    // Get timestamp
    uint64_t timestamp;
    if (ret = _rocmon_get_timestamp(&timestamp))
    {
        return ret;
    }

    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];

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
        ret = _rocmon_readCounters_rocprofiler(device);
        if (ret < 0) return ret;

        // Read SMI counters
        ret = _rocmon_readCounters_smi(device);
        if (ret < 0) return ret;
    }

    return 0;
}


static uint64_t*
_rocmon_get_read_time(RocmonDevice* device)
{
    return &device->time.read;
}


static uint64_t*
_rocmon_get_stop_time(RocmonDevice* device)
{
    return &device->time.stop;
}


// ----------------------------------------------------
//   Rocmon SMI helper functions
// ----------------------------------------------------

static bstring
_rocmon_smi_build_label(RocmonSmiEventType type, const char* funcname, uint64_t variant, uint64_t subvariant)
{
    switch (type)
    {
    case ROCMON_SMI_EVENT_TYPE_NORMAL:
        return bfromcstr(funcname);
    case ROCMON_SMI_EVENT_TYPE_VARIANT:
        return bformat("%s|%" PRIu64, funcname, variant);
    case ROCMON_SMI_EVENT_TYPE_SUBVARIANT:
        return bformat("%s|%" PRIu64 "|%" PRIu64, funcname, variant, subvariant);
    case ROCMON_SMI_EVENT_TYPE_INSTANCES:
        return bfromcstr(funcname);
    }
}


static int
_rocmon_smi_add_event_to_device(RocmonDevice* device, const char* funcname, RocmonSmiEventType type, int64_t variant, uint64_t subvariant)
{
    int ret;
    
    // Get event by label
    RocmonSmiEventList* list = NULL;
    bstring label = _rocmon_smi_build_label(type, funcname, variant, subvariant);
    ret = get_smap_by_key(rocmon_context->smiEvents, bdata(label), (void**)&list);
    bdestroy(label);
    if (ret < 0)
    {
        // Event not registered -> ignore
        return 0;
    }

    // For events with multiple sensor, only make one entry -> find if one exists
    if (type == ROCMON_SMI_EVENT_TYPE_INSTANCES && subvariant > 0)
    {
        // Get list from map
        for (int i = 0; i < list->numEntries; i++)
        {
            RocmonSmiEvent* event = &list->entries[i];
            RocmonSmiEvent* existingEvent = NULL;
            ret = get_smap_by_key(device->smiMetrics, event->name, (void**)&existingEvent);
            if (ret < 0)
            {
                ERROR_PRINT(Failed to find previous instance for event %s, event->name);
                return -1;
            }

            // Update instance information
            existingEvent->instances++;
        }
        return 0;
    }

    for (int i = 0; i < list->numEntries; i++)
    {
        RocmonSmiEvent* event = &list->entries[i];

        // Allocate memory for device event description
        RocmonSmiEvent* tmpEvent = (RocmonSmiEvent*) malloc(sizeof(RocmonSmiEvent));
        if (tmpEvent == NULL)
        {
            ERROR_PRINT(Failed to allocate memory for SMI event in device list %s, event->name);
            return -ENOMEM;
        }

        // Copy information from global description
        memcpy(tmpEvent, event, sizeof(RocmonSmiEvent));
        tmpEvent->variant = variant;
        tmpEvent->subvariant = subvariant;
        tmpEvent->instances = 1;

        // Save event info to device event map
        add_smap(device->smiMetrics, tmpEvent->name, tmpEvent);
    }

    return 0;
}


static int
_rocmon_smi_get_function_subvariants(RocmonDevice* device, const char* funcname, uint64_t variant, rsmi_func_id_iter_handle_t var_iter)
{
    rsmi_func_id_iter_handle_t sub_var_iter;
    rsmi_func_id_value_t value;
    rsmi_status_t status;
    int ret;

    // Get open subvariants iterator
    status = (*rsmi_dev_supported_variant_iterator_open_ptr)(var_iter, &sub_var_iter);
    if (status == RSMI_STATUS_NO_DATA)
    {
        // No subvariants
        ret = _rocmon_smi_add_event_to_device(device, funcname, ROCMON_SMI_EVENT_TYPE_VARIANT, variant, 0);
        if (ret < 0) return -1;
        return 0;
    }
    
    // Subvariants available -> iterate them
    do {
        // Get subvariant information
        (*rsmi_func_iter_value_get_ptr)(sub_var_iter, &value);

        // Process info
        if (variant == RSMI_DEFAULT_VARIANT)
            ret = _rocmon_smi_add_event_to_device(device, funcname, ROCMON_SMI_EVENT_TYPE_INSTANCES, variant, value.id);
        else
            ret = _rocmon_smi_add_event_to_device(device, funcname, ROCMON_SMI_EVENT_TYPE_SUBVARIANT, variant, value.id);
        if (ret < 0) return ret;

        // Advance iterator
        status = (*rsmi_func_iter_next_ptr)(sub_var_iter);
    } while (status != RSMI_STATUS_NO_DATA);

    // Close iterator
    (*rsmi_dev_supported_func_iterator_close_ptr)(&sub_var_iter);

    return 0;
}


static int
_rocmon_smi_get_function_variants(RocmonDevice* device, const char* funcname, rsmi_func_id_iter_handle_t iter_handle)
{
    rsmi_func_id_iter_handle_t var_iter;
    rsmi_func_id_value_t value;
    rsmi_status_t status;
    int ret;

    // Get open variants iterator
    status = (*rsmi_dev_supported_variant_iterator_open_ptr)(iter_handle, &var_iter);
    if (status == RSMI_STATUS_NO_DATA)
    {
        // No variants
        ret = _rocmon_smi_add_event_to_device(device, funcname, ROCMON_SMI_EVENT_TYPE_NORMAL, 0, 0);
        if (ret < 0) return -1;
        return 0;
    }
    
    // Variants available -> iterate them
    do {
        // Get variant information
        (*rsmi_func_iter_value_get_ptr)(var_iter, &value);

        // Get function subvariants
        ret = _rocmon_smi_get_function_subvariants(device, funcname, value.id, var_iter);
        if (ret < 0) return -1;

        // Advance iterator
        status = (*rsmi_func_iter_next_ptr)(var_iter);
    } while (status != RSMI_STATUS_NO_DATA);

    // Close iterator
    (*rsmi_dev_supported_func_iterator_close_ptr)(&var_iter);

    return 0;
}


static int
_rocmon_smi_get_functions(RocmonDevice* device)
{
    rsmi_func_id_iter_handle_t iter_handle;
    rsmi_func_id_value_t value;
    rsmi_status_t status;
    int ret;

    // Open iterator
    //(*rsmi_dev_supported_func_iterator_open_ptr)(device->deviceId, &iter_handle);
    RSMI_CALL(rsmi_dev_supported_func_iterator_open, (device->deviceId, &iter_handle), {
        return -1;
    });

    do
    {
        // Get function information
        //(*rsmi_func_iter_value_get_ptr)(iter_handle, &value);
        RSMI_CALL(rsmi_func_iter_value_get, (iter_handle, &value), {
            ERROR_PRINT(Failed to get smi function value for device %d, device->deviceId);
            RSMI_CALL(rsmi_dev_supported_func_iterator_close, (&iter_handle), );
            return -1;
        });

        // Get function variants
        ret = _rocmon_smi_get_function_variants(device, value.name, iter_handle);
        if (ret < 0)
        {
            ERROR_PRINT(Failed to get smi function variants for device %d, device->deviceId);
            RSMI_CALL(rsmi_dev_supported_func_iterator_close, (&iter_handle), );
            return -1;
        }

        // Advance iterator (cannot use RSMI_CALL macro here because we have an assignment,
        // so we check that the function pointer exists to avoid segfaults.)
        if (rsmi_func_iter_next_ptr) {
            status = (*rsmi_func_iter_next_ptr)(iter_handle);
        }
    } while (status != RSMI_STATUS_NO_DATA);

    // Close iterator
    //(*rsmi_dev_supported_func_iterator_close_ptr)(&iter_handle);
    RSMI_CALL(rsmi_dev_supported_func_iterator_close, (&iter_handle), );

    // Add device independent functions
    ret = _rocmon_smi_add_event_to_device(device, "rsmi_compute_process_info_get", ROCMON_SMI_EVENT_TYPE_NORMAL, 0, 0);
    if (ret < 0) return -1;

    return 0;
}

#define ADD_SMI_EVENT(name, type, smifunc, variant, subvariant, extra, measurefunc) if (_rocmon_smi_add_event_to_map(name, type, smifunc, variant, subvariant, extra, measurefunc) < 0) { return -1; }
#define ADD_SMI_EVENT_N(name, smifunc, extra, measurefunc) ADD_SMI_EVENT(name, ROCMON_SMI_EVENT_TYPE_NORMAL, smifunc, 0, 0, extra, measurefunc)
#define ADD_SMI_EVENT_V(name, smifunc, variant, extra, measurefunc) ADD_SMI_EVENT(name, ROCMON_SMI_EVENT_TYPE_VARIANT, smifunc, variant, 0, extra, measurefunc)
#define ADD_SMI_EVENT_S(name, smifunc, variant, subvariant, extra, measurefunc) ADD_SMI_EVENT(name, ROCMON_SMI_EVENT_TYPE_SUBVARIANT, smifunc, variant, subvariant, extra, measurefunc)
#define ADD_SMI_EVENT_I(name, smifunc, extra, measurefunc) ADD_SMI_EVENT(name, ROCMON_SMI_EVENT_TYPE_INSTANCES, smifunc, 0, 0, extra, measurefunc)

static int
_rocmon_smi_add_event_to_map(char* name, RocmonSmiEventType type, char* smifunc, uint64_t variant, uint64_t subvariant, uint64_t extra, RocmonSmiMeasureFunc measureFunc)
{
    // Add new event list to map (if not already present)
    bstring label = _rocmon_smi_build_label(type, smifunc, variant, subvariant);
    RocmonSmiEventList* list;
    if (get_smap_by_key(rocmon_context->smiEvents, bdata(label), (void**)&list) < 0)
    {
        // Allocate memory for event list
        list = (RocmonSmiEventList*) malloc(sizeof(RocmonSmiEventList));
        if (list == NULL)
        {
            ERROR_PRINT(Failed to allocate memory for SMI event list %s, name);
            return -ENOMEM;
        }
        list->entries = NULL;
        list->numEntries = 0;

        add_smap(rocmon_context->smiEvents, bdata(label), list);
    }
    bdestroy(label);

    // Allocate memory for another event in list
    list->numEntries++;
    list->entries = (RocmonSmiEvent*) realloc(list->entries, list->numEntries * sizeof(RocmonSmiEvent));
    if (list->entries == NULL)
    {
        ERROR_PRINT(Failed to allocate memory for SMI event %s, name);
        return -ENOMEM;
    }

    // Set event properties
    RocmonSmiEvent* event = &list->entries[list->numEntries-1];
    strncpy(event->name, name, sizeof(event->name));
    event->name[sizeof(event->name)] = '\0';
    event->type = type;
    event->variant = variant;
    event->subvariant = subvariant;
    event->extra = extra;
    event->instances = 0; // gets set when scanning supported device functions
    event->measureFunc = measureFunc;

    return 0;
}


static void
_rcomon_smi_free_event_list(void* vlist)
{
    RocmonSmiEventList* list = (RocmonSmiEventList*)vlist;
    if (list)
    {
        FREE_IF_NOT_NULL(list->entries);
        free(list);
    }
}


static int
_rocmon_smi_init_events()
{
    int ret;

    // Init map
    ret = init_map(&rocmon_context->smiEvents, MAP_KEY_TYPE_STR, 0, &_rcomon_smi_free_event_list);
    if (ret < 0)
    {
        ERROR_PRINT(Failed to create map for ROCm SMI events);
        return -1;
    }

    // Add events
    ADD_SMI_EVENT_N("PCI_THROUGHPUT_SENT",                  "rsmi_dev_pci_throughput_get", 0,                                           &_smi_wrapper_pci_throughput_get        );
    ADD_SMI_EVENT_N("PCI_THROUGHPUT_RECEIVED",              "rsmi_dev_pci_throughput_get", 1,                                           &_smi_wrapper_pci_throughput_get        );
    ADD_SMI_EVENT_N("PCI_THROUGHPUT_MAX_PKT_SZ",            "rsmi_dev_pci_throughput_get", 2,                                           &_smi_wrapper_pci_throughput_get        );
    ADD_SMI_EVENT_N("PCI_REPLAY_COUNTER",                   "rsmi_dev_pci_replay_counter_get", 0,                                       &_smi_wrapper_pci_replay_counter_get    );
    ADD_SMI_EVENT_I("POWER_AVE",                            "rsmi_dev_power_ave_get", 0,                                                &_smi_wrapper_power_ave_get             );
    ADD_SMI_EVENT_V("MEMORY_TOTAL_VRAM",                    "rsmi_dev_memory_total_get", RSMI_MEM_TYPE_VRAM, 0,                         &_smi_wrapper_memory_total_get          );
    ADD_SMI_EVENT_V("MEMORY_TOTAL_VIS_VRAM",                "rsmi_dev_memory_total_get", RSMI_MEM_TYPE_VIS_VRAM, 0,                     &_smi_wrapper_memory_total_get          );
    ADD_SMI_EVENT_V("MEMORY_TOTAL_GTT",                     "rsmi_dev_memory_total_get", RSMI_MEM_TYPE_GTT, 0,                          &_smi_wrapper_memory_total_get          );
    ADD_SMI_EVENT_V("MEMORY_USAGE_VRAM",                    "rsmi_dev_memory_usage_get", RSMI_MEM_TYPE_VRAM, 0,                         &_smi_wrapper_memory_usage_get          );
    ADD_SMI_EVENT_V("MEMORY_USAGE_VIS_VRAM",                "rsmi_dev_memory_usage_get", RSMI_MEM_TYPE_VIS_VRAM, 0,                     &_smi_wrapper_memory_usage_get          );
    ADD_SMI_EVENT_V("MEMORY_USAGE_GTT",                     "rsmi_dev_memory_usage_get", RSMI_MEM_TYPE_GTT, 0,                          &_smi_wrapper_memory_usage_get          );
    ADD_SMI_EVENT_N("MEMORY_BUSY_PERCENT",                  "rsmi_dev_memory_busy_percent_get", 0,                                      &_smi_wrapper_memory_busy_percent_get   );
    ADD_SMI_EVENT_N("MEMORY_NUM_RESERVED_PAGES",            "rsmi_dev_memory_reserved_pages_get", 0,                                    &_smi_wrapper_memory_reserved_pages_get );
    ADD_SMI_EVENT_I("FAN_RPMS",                             "rsmi_dev_fan_rpms_get", 0,                                                 &_smi_wrapper_fan_rpms_get              );
    ADD_SMI_EVENT_I("FAN_SPEED",                            "rsmi_dev_fan_speed_get", 0,                                                &_smi_wrapper_fan_speed_get             );
    ADD_SMI_EVENT_I("FAN_SPEED_MAX",                        "rsmi_dev_fan_speed_max_get", 0,                                            &_smi_wrapper_fan_speed_max_get         );
    ADD_SMI_EVENT_S("TEMP_EDGE",                            "rsmi_dev_temp_metric_get", RSMI_TEMP_CURRENT, RSMI_TEMP_TYPE_EDGE, 0,      &_smi_wrapper_temp_metric_get           );
    ADD_SMI_EVENT_S("TEMP_JUNCTION",                        "rsmi_dev_temp_metric_get", RSMI_TEMP_CURRENT, RSMI_TEMP_TYPE_JUNCTION, 0,  &_smi_wrapper_temp_metric_get           );
    ADD_SMI_EVENT_S("TEMP_MEMORY",                          "rsmi_dev_temp_metric_get", RSMI_TEMP_CURRENT, RSMI_TEMP_TYPE_MEMORY, 0,    &_smi_wrapper_temp_metric_get           );
    ADD_SMI_EVENT_S("VOLT_VDDGFX",                          "rsmi_dev_volt_metric_get", RSMI_VOLT_CURRENT, RSMI_VOLT_TYPE_VDDGFX, 0,    &_smi_wrapper_volt_metric_get           );
    ADD_SMI_EVENT_N("OVERDRIVE_LEVEL",                      "rsmi_dev_overdrive_level_get", 0,                                          &_smi_wrapper_overdrive_level_get       );
    ADD_SMI_EVENT_V("ECC_COUNT_UMC_CORRECTABLE",            "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_UMC, 0,                            &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_UMC_UNCORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_UMC, 1,                            &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_SDMA_CORRECTABLE",           "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_SDMA, 0,                           &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_SDMA_UNCORRECTABLE",         "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_SDMA, 1,                           &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_GFX_CORRECTABLE",            "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_GFX, 0,                            &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_GFX_UNCORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_GFX, 1,                            &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_MMHUB_CORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_MMHUB, 0,                          &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_MMHUB_UNCORRECTABLE",        "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_MMHUB, 1,                          &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_ATHUB_CORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_ATHUB, 0,                          &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_ATHUB_UNCORRECTABLE",        "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_ATHUB, 1,                          &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_PCIE_BIF_CORRECTABLE",       "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_PCIE_BIF, 0,                       &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_PCIE_BIF_UNCORRECTABLE",     "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_PCIE_BIF, 1,                       &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_HDP_CORRECTABLE",            "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_HDP, 0,                            &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_HDP_UNCORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_HDP, 1,                            &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_XGMI_WAFL_CORRECTABLE",      "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_XGMI_WAFL, 0,                      &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_XGMI_WAFL_UNCORRECTABLE",    "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_XGMI_WAFL, 1,                      &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_DF_CORRECTABLE",             "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_DF, 0,                             &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_DF_UNCORRECTABLE",           "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_DF, 1,                             &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_SMN_CORRECTABLE",            "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_SMN, 0,                            &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_SMN_UNCORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_SMN, 1,                            &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_SEM_CORRECTABLE",            "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_SEM, 0,                            &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_SEM_UNCORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_SEM, 1,                            &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_MP0_CORRECTABLE",            "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_MP0, 0,                            &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_MP0_UNCORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_MP0, 1,                            &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_MP1_CORRECTABLE",            "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_MP1, 0,                            &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_MP1_UNCORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_MP1, 1,                            &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_FUSE_CORRECTABLE",           "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_FUSE, 0,                           &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_FUSE_UNCORRECTABLE",         "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_FUSE, 1,                           &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_LAST_CORRECTABLE",           "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_LAST, 0,                           &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_LAST_UNCORRECTABLE",         "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_LAST, 1,                           &_smi_wrapper_ecc_count_get             );
    ADD_SMI_EVENT_N("PROCS_USING_GPU",                      "rsmi_compute_process_info_get", 0,                                         &_smi_wrapper_compute_process_info_get  );

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
	ERROR_PLAIN_PRINT(Failed to initialize libraries);
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
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Initializing HSA);
    ROCM_CALL(hsa_init, (),
    {
        ERROR_PLAIN_PRINT(Failed to init hsa library);
        goto rocmon_init_hsa_failed;
    });

    // init rocm smi library
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Initializing RSMI);
    RSMI_CALL(rsmi_init, (0),
    {
        ERROR_PLAIN_PRINT(Failed to init rocm_smi);
        goto rocmon_init_rsmi_failed;
    });

    // Get hsa timestamp factor
    uint64_t frequency_hz;
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Getting HSA timestamp factor);
    ROCM_CALL(hsa_system_get_info, (HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency_hz),
    {
        ERROR_PLAIN_PRINT(Failed to get HSA timestamp factor);
        goto rocmon_init_info_agents_failed;
    });
    rocmon_context->hsa_timestamp_factor = (long double)1000000000 / (long double)frequency_hz;

    // initialize structures for specified devices (fetch ROCm specific info)
    iterate_agents_cb_arg arg = {
        .context = rocmon_context,
        .numGpus = numGpus,
        .gpuIds = gpuIds,
    };
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Iterating through %d available agents, numGpus);
    ROCM_CALL(hsa_iterate_agents, (_rocmon_iterate_agents_callback, &arg),
    {
        ERROR_PRINT(Error while iterating through available agents);
        goto rocmon_init_info_agents_failed;
    });

    // Get available SMI events for devices
    _rocmon_smi_init_events();
    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        if (_rocmon_smi_get_functions(&rocmon_context->devices[i]) < 0)
        {
            ERROR_PRINT(Failed to get SMI functions for device %d, rocmon_context->devices[i].deviceId);
            goto rocmon_init_info_agents_failed;
        }
    }

    rocmon_initialized = TRUE;
    return 0;
rocmon_init_info_agents_failed:
    RSMI_CALL(rsmi_shut_down, (), {
        // fall through
    });
rocmon_init_rsmi_failed:
    ROCM_CALL(hsa_shut_down, (), {
        // fall through
    });
rocmon_init_hsa_failed:
    free(rocmon_context->devices);
    free(rocmon_context);
    rocmon_context = NULL;
    return -1;
}


void
rocmon_finalize(void)
{
    RocmonContext* context = rocmon_context;

    if (!rocmon_initialized)
    {
        return;
    }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Finalize LIKWID ROCMON);

    if (context)
    {
        if (context->devices)
        {
            // Free each devices fields
            for (int i = 0; i < context->numDevices; i++)
            {
                RocmonDevice* device = &context->devices[i];
                FREE_IF_NOT_NULL(device->rocMetrics);
                FREE_IF_NOT_NULL(device->activeRocEvents);
                FREE_IF_NOT_NULL(device->activeSmiEvents);
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
                destroy_smap(device->smiMetrics);
            }

            free(context->devices);
            context->devices = NULL;
        }

        FREE_IF_NOT_NULL(context->groups);
        destroy_smap(context->smiEvents);

        free(context);
        context = NULL;
    }

    RSMI_CALL(rsmi_shut_down, (), {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Shutdown SMI);
        // fall through
    });
    ROCM_CALL(hsa_shut_down, (), {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Shutdown HSA);
        // fall through
    });
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


static int
_rocmon_setupCounters_rocprofiler(RocmonDevice* device, const char** events, int numEvents)
{
    // Close previous rocprofiler context
    if (device->context)
    {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Closing previous rocprofiler context);
        ROCM_CALL(rocprofiler_close, (device->context), return -1);
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
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, SETUP EVENT %d %s, i, events[i]);
    }

    // Free previous feature array if present
    FREE_IF_NOT_NULL(device->activeRocEvents);

    device->numActiveRocEvents = numEvents;
    device->activeRocEvents = features;

    // Open context
    rocprofiler_properties_t properties = {};
    properties.queue_depth = 128;
    uint32_t mode = ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_CREATEQUEUE | ROCPROFILER_MODE_SINGLEGROUP;

    // Important: only a single profiling group is supported at this time which limits the number of events that can be monitored at a time.
    ROCM_CALL(rocprofiler_open, (device->hsa_agent, device->activeRocEvents, device->numActiveRocEvents, &device->context, mode, &properties), return -1);

    return 0;
}


static int
_rocmon_setupCounters_smi(RocmonDevice* device, const char** events, int numEvents)
{
    int ret;
    const int instanceNumLen = 5;

    // Look if the are any events
    if (numEvents <= 0)
    {
        return 0;
    }

    // Create event array
    RocmonSmiEvent* activeEvents = (RocmonSmiEvent*) malloc(numEvents * sizeof(RocmonSmiEvent));
    if (activeEvents == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate active event list);
        return -ENOMEM;
    }

    for (int i = 0; i < numEvents; i++)
    {
        char eventName[membersize(RocmonSmiEvent, name)];
        int instance = -1;

        // Parse event name -> normal event vs one with multiple instances (EVENT[0])
        const char* event = events[i];
        char* instancePart = strrchr(event, '[');
        if (instancePart != NULL)
        {
            char withoutBrackets[instanceNumLen+1]; // +1 is '\0'
            int partlen = strlen(instancePart);

            // Check if number fit in 'withoutBrackets'
            if (partlen - 2 > instanceNumLen)
            {
                ERROR_PRINT(Instance number in '%s' is too large, event);
                free(activeEvents);
                return -EINVAL;
            }

            // Copy instance number without brackets
            strncpy(withoutBrackets, instancePart+1, partlen-2);
            withoutBrackets[instanceNumLen] = '\0';

            // Parse instance as number
            char* endParsed;
            instance = strtol(withoutBrackets, &endParsed, 10);

            // Check if parsing was successful
            char* endOfString = &withoutBrackets[partlen-2];
            if (endParsed != endOfString)
            {
                ERROR_PRINT(Failed to parse instance number in '%s', event);
                free(activeEvents);
                return -EINVAL;
            }

            // Copy event name without instance
            int eventNameLen = instancePart - event;
            strncpy(eventName, event, eventNameLen);
            eventName[eventNameLen] = '\0';
        }
        else
        {
            // Copy entire event name
            strncpy(eventName, event, membersize(RocmonSmiEvent, name));
        }

        // Lookup event in available events
        RocmonSmiEvent* metric = NULL;
        ret = get_smap_by_key(device->smiMetrics, eventName, (void**)&metric);
        if (ret < 0)
        {
            ERROR_PRINT(RSMI event '%s' not found for device %d, eventName, device->deviceId);
            free(activeEvents);
            return -EINVAL;
        }

        // Copy event
        RocmonSmiEvent* tmpEvent = &activeEvents[i];
        memcpy(tmpEvent, metric, sizeof(RocmonSmiEvent));

        // Check if event supports instances
        if (instance >= 0 && tmpEvent->type != ROCMON_SMI_EVENT_TYPE_INSTANCES)
        {
            ERROR_PRINT(Instance number given but event '%s' does not support one, eventName);
            free(activeEvents);
            return -EINVAL;
        }

        // Check if event requires instances
        if (instance < 0 && tmpEvent->type == ROCMON_SMI_EVENT_TYPE_INSTANCES)
        {
            ERROR_PRINT(No instance number given but event '%s' requires one, eventName);
            free(activeEvents);
            return -EINVAL;
        }

        // Check if event has enough instances
        if (instance >= 0 && instance >= metric->instances)
        {
            ERROR_PRINT(Instance %d seleced but event '%s' has only %d, instance, eventName, metric->instances);
            free(activeEvents);
            return -EINVAL;
        }

        // Set instance number
        if (instance >= 0)
        {
            tmpEvent->subvariant = instance;
        }
    }

    device->activeSmiEvents = activeEvents;
    device->numActiveSmiEvents = numEvents;

    return 0;
}


int
rocmon_setupCounters(int gid)
{
    int ret;

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

    // Get group info
    GroupInfo* group = &rocmon_context->groups[gid];

    //
    // Separate rocprofiler and SMI events
    //
    const char **smiEvents = NULL, **rocEvents = NULL;
    int numSmiEvents = 0, numRocEvents = 0;

    // Allocate memory for string arrays
    smiEvents = (const char**) malloc(group->nevents * sizeof(const char*));
    if (smiEvents == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate smiEvent name array);
        return -ENOMEM;
    }
    rocEvents = (const char**) malloc(group->nevents * sizeof(const char*));
    if (rocEvents == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate rocEvent name array);
        free(smiEvents);
        return -ENOMEM;
    }

    // Go through each event and sort it
    for (int i = 0; i < group->nevents; i++)
    {
        const char* name = group->events[i];
        if (strncmp(name, "RSMI_", 5) == 0)
        {
            // RSMI event
            smiEvents[numSmiEvents] = name + 5; // +5 removes 'RSMI_' prefix
            numSmiEvents++;
        }
        else if (strncmp(name, "ROCP_", 5) == 0)
        {
            // Rocprofiler event
            rocEvents[numRocEvents] = name + 5; // +5 removes 'ROCP_' prefix
            numRocEvents++;
        }
        else
        {
            // Unknown event
            ERROR_PRINT(Event '%s' has no prefix ('ROCP_' or 'RSMI_'), name);
            return -EINVAL;
        }
    }

    // Add events to each device
    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];

        // Add rocprofiler events
        ROCMON_DEBUG_PRINT(DEBUGLEV_INFO, SETUP ROCPROFILER WITH %d events, numRocEvents);
        ret = _rocmon_setupCounters_rocprofiler(device, rocEvents, numRocEvents);
        if (ret < 0)
        {
            free(smiEvents);
            free(rocEvents);
            return ret;
        }

        // Add SMI events
        ROCMON_DEBUG_PRINT(DEBUGLEV_INFO, SETUP ROCM SMI WITH %d events, numSmiEvents);
        ret = _rocmon_setupCounters_smi(device, smiEvents, numSmiEvents);
        if (ret < 0)
        {
            free(smiEvents);
            free(rocEvents);
            return ret;
        }
    }
    rocmon_context->activeGroup = gid;

    // Cleanup
    free(smiEvents);
    free(rocEvents);

    return 0;
}


static int
_rocmon_startCounters_rocprofiler(RocmonDevice* device)
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

    if (device->context)
    {
        ROCM_CALL(rocprofiler_start, (device->context, 0), return -1);
    }

    return 0;
}


static int
_rocmon_startCounters_smi(RocmonDevice* device)
{
    // Check if there are any counters to start
    if (device->numActiveSmiEvents <= 0)
    {
        return 0;
    }

    // Save baseline values
    RocmonEventResultList* groupResult = &device->groupResults[rocmon_context->activeGroup];
    for (int i = 0; i < device->numActiveSmiEvents; i++)
    {
        double value = 0;
        RocmonSmiEvent* event = &device->activeSmiEvents[i];
        RocmonEventResult* result = &groupResult->results[device->numActiveRocEvents+i];

        // Measure counter
        if (event->measureFunc)
        {
            event->measureFunc(device->deviceId, event, result);
        }

        // Save value
        result->fullValue = 0;
    }

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

        // Start rocprofiler events
        ret = _rocmon_startCounters_rocprofiler(device);
        if (ret < 0) return ret;

        // Start SMI events
        _rocmon_startCounters_smi(device);
        if (ret < 0) return ret;
    }

    return 0;
}


static int
_rocmon_stopCounters_rocprofiler(RocmonDevice* device)
{
    if (device->context)
    {
        // Close context
        ROCM_CALL(rocprofiler_stop, (device->context, 0), return -1);
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

    // Read counters
    ret = _rocmon_readCounters(&_rocmon_get_stop_time);
    if (ret < 0) return ret;

    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];

        // Stop rocprofiler events
        ret = _rocmon_stopCounters_rocprofiler(device);
        if (ret < 0) return ret;

        // Nothing to stop for SMI events
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

    // Read counters
    ret = _rocmon_readCounters(&_rocmon_get_read_time);
    if (ret < 0) return ret;

    return 0;
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
rocmon_getEventsOfGpu(int gpuIdx, EventList_rocm_t* list)
{
    // Ensure rocmon is initialized
    if (!rocmon_initialized)
    {
        return -EFAULT;
    }

    // Validate args
    if (gpuIdx < 0 || gpuIdx > rocmon_context->numDevices)
    {
        return -EINVAL;
    }
    if (list == NULL)
    {
        return -EINVAL;
    }

    RocmonDevice* device = &rocmon_context->devices[gpuIdx];

    // Allocate list structure
    EventList_rocm_t tmpList = (EventList_rocm_t) malloc(sizeof(EventList_rocm));
    if (tmpList == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate event list);
        return -ENOMEM;
    }
    
    // Get number of events
    printf("NUmber of events %d + %d\n", device->numRocMetrics , get_map_size(device->smiMetrics));
    tmpList->numEvents = device->numRocMetrics + get_map_size(device->smiMetrics);
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

    // Copy rocprofiler event information
    for (int i = 0; i < device->numRocMetrics; i++)
    {
        rocprofiler_info_data_t* event = &device->rocMetrics[i];
        Event_rocm_t* out = &tmpList->events[i];
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

        // Copy instances
        out->instances = event->metric.instances;
    }

    // Copy ROCm SMI metric information
    for (int i = 0; i < get_map_size(device->smiMetrics); i++)
    {
        RocmonSmiEvent* event = NULL;
        Event_rocm_t* out = &tmpList->events[device->numRocMetrics + i];
        int len;

        // Get event
        if (get_smap_by_idx(device->smiMetrics, i, (void**)&event) < 0)
        {
            continue;
        }

        // Copy name
        len = strlen(event->name) + 5 /* Prefix */ + 1 /* NULL byte */;
        out->name = (char*) malloc(len);
        if (out->name)
        {
            snprintf(out->name, len, "RSMI_%s", event->name);
        }

        // Copy description
        char* description = "SMI Event"; // TODO: use real descriptions
        len = strlen(description) + 1 /* NULL byte */;
        out->description = (char*) malloc(len);
        if (out->description)
        {
            snprintf(out->description, len, "%s", description);
        }

        // Copy instances
        out->instances = event->instances;
    }

    *list = tmpList;
    return 0;
}

void
rocmon_freeEventsOfGpu(EventList_rocm_t list)
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


double
rocmon_getTimeToLastReadOfGroup(int groupId)
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
        t = MAX(t, (double)(device->time.read - device->time.start));
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

void rocmon_setVerbosity(int level)
{
    if (level >= DEBUGLEV_ONLY_ERROR && level <= DEBUGLEV_DEVELOP)
    {
        likwid_rocmon_verbosity = level;
    }
}


#endif /* LIKWID_WITH_ROCMON */
