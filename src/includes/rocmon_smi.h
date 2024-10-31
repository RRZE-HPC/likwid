/*
 * =======================================================================================
 *
 *      Filename:  rocmon_smi.h
 *
 *      Description:  Header File of rocmon module for ROCm SMI.
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
#ifndef LIKWID_ROCMON_SMI_H
#define LIKWID_ROCMON_SMI_H

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
#include <map.h>

#include <rocmon_common_types.h>
#include <rocmon_smi_types.h>

static void *rocmon_dl_rsmi_lib = NULL;

static int rocmon_smi_initialized = 0;

#ifndef RSMI_CALL
#define RSMI_CALL( call, args, handleerror )                                  \
    do {                                                                \
        rsmi_status_t _status = (*call##_ptr)args;                                  \
        if (_status != RSMI_STATUS_SUCCESS) {           \
            fprintf(stderr, "Error: function %s failed with error %d.\n", #call, _status); \
            handleerror;                                                \
        }                                                               \
    } while (0)
#endif

#ifndef DECLAREFUNC_SMI
#define DECLAREFUNC_SMI(funcname, funcsig) rsmi_status_t ROCMWEAK funcname funcsig; rsmi_status_t ( *funcname##_ptr ) funcsig;
#endif

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


static int
_rocmon_smi_link_libraries()
{
    #define DLSYM_AND_CHECK( dllib, name ) name##_ptr = dlsym( dllib, #name ); if ( dlerror() != NULL ) { ERROR_PRINT(Failed to link  #name); return -1; }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Linking AMD SMI libraries);

    // Need to link in the Rocprofiler libraries
    rocmon_dl_rsmi_lib = dlopen("librocm_smi64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!rocmon_dl_rsmi_lib)
    {
        ERROR_PRINT(ROCm SMI library librocm_smi64.so not found: %s, dlerror());
        return -1;
    }

    // Link SMI functions
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_init);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_shut_down);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_supported_func_iterator_open);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_supported_variant_iterator_open);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_func_iter_value_get);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_func_iter_next);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_supported_func_iterator_close);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_power_ave_get);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_pci_throughput_get);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_pci_replay_counter_get);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_memory_total_get);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_memory_usage_get);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_memory_busy_percent_get);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_memory_reserved_pages_get);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_fan_rpms_get);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_fan_speed_get);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_fan_speed_max_get);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_temp_metric_get);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_volt_metric_get);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_overdrive_level_get);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_dev_ecc_count_get);
    DLSYM_AND_CHECK(rocmon_dl_rsmi_lib, rsmi_compute_process_info_get);
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Linking AMD ROCMm libraries done);
    return 0;
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

#define ADD_SMI_EVENT(name, type, smifunc, variant, subvariant, extra, measurefunc) if (_rocmon_smi_add_event_to_map(name, type, smifunc, variant, subvariant, extra, measurefunc) < 0) { return -1; }
#define ADD_SMI_EVENT_N(name, smifunc, extra, measurefunc) ADD_SMI_EVENT(name, ROCMON_SMI_EVENT_TYPE_NORMAL, smifunc, 0, 0, extra, measurefunc)
#define ADD_SMI_EVENT_V(name, smifunc, variant, extra, measurefunc) ADD_SMI_EVENT(name, ROCMON_SMI_EVENT_TYPE_VARIANT, smifunc, variant, 0, extra, measurefunc)
#define ADD_SMI_EVENT_S(name, smifunc, variant, subvariant, extra, measurefunc) ADD_SMI_EVENT(name, ROCMON_SMI_EVENT_TYPE_SUBVARIANT, smifunc, variant, subvariant, extra, measurefunc)
#define ADD_SMI_EVENT_I(name, smifunc, extra, measurefunc) ADD_SMI_EVENT(name, ROCMON_SMI_EVENT_TYPE_INSTANCES, smifunc, 0, 0, extra, measurefunc)


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
_rocmon_smi_init_events(RocmonContext* context)
{
    int ret;

    // Init map
    ret = init_map(&context->smiEvents, MAP_KEY_TYPE_STR, 0, &_rcomon_smi_free_event_list);
    if (ret < 0)
    {
        ERROR_PRINT(Failed to create map for ROCm SMI events);
        return ret;
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

static int
_rocmon_setupCounters_smi(RocmonDevice* device, const char** events, int numEvents)
{
    int ret;
    const int instanceNumLen = 5;

    // Delete previous events
    if (device->activeSmiEvents)
    {
        free(device->activeSmiEvents);
        device->activeSmiEvents = NULL;
        device->numActiveSmiEvents = 0;
    }

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
        char eventName[MAX_ROCMON_SMI_EVENT_NAME];
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
            strncpy(eventName, event, MAX_ROCMON_SMI_EVENT_NAME);
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
rocmon_smi_setupCounters(RocmonContext* context, int gid)
{
    int ret = 0;
    int numSmiEvents = 0;
    const char **smiEvents = NULL;
    // Check arguments
    if (gid < 0 || gid >= context->numActiveGroups)
    {
        return -EINVAL;
    }
    
    // Ensure rocmon is initialized
    if (!rocmon_smi_initialized)
    {
        return -EFAULT;
    }

    // Get group info
    GroupInfo* group = &context->groups[gid];

    // Allocate memory for string arrays
    smiEvents = (const char**) malloc(group->nevents * sizeof(const char*));
    if (smiEvents == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate smiEvents name array);
        return -ENOMEM;
    }

    // Go through each event and sort it
    for (int i = 0; i < group->nevents; i++)
    {
        const char* name = group->events[i];
        if (strncmp(name, "RSMI_", 5) == 0)
        {
            // Rocprofiler event
            smiEvents[numSmiEvents] = name + 5; // +5 removes 'ROCP_' prefix
            numSmiEvents++;
        }
    }
    if (numSmiEvents == 0)
    {
        free(smiEvents);
        return 0;
    }

    for (int i = 0; i < context->numDevices; i++)
    {
        RocmonDevice* device = &context->devices[i];
        ret = _rocmon_setupCounters_smi(device, smiEvents, numSmiEvents);
        if (ret < 0)
        {
            ERROR_PRINT(Failed to setup ROCMON SMI events for device %d, i);
        }
    }
    free(smiEvents);
    return 0;
}

int
rocmon_smi_readCounters(RocmonContext* context)
{
    // Ensure rocmon is initialized
    if (!rocmon_smi_initialized)
    {
        return -EFAULT;
    }
    if (context->activeGroup < 0)
    {
        return -EFAULT;
    }
    for (int i = 0; i < context->numDevices; i++)
    {
        RocmonDevice* device = &context->devices[i];
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
    }
    return 0;
}

int
rocmon_smi_startCounters(RocmonContext* context)
{
    // Ensure rocmon is initialized
    if (!rocmon_smi_initialized)
    {
        return -EFAULT;
    }
    if (context->activeGroup < 0)
    {
        return -EFAULT;
    }
    for (int i = 0; i < context->numDevices; i++)
    {
        RocmonDevice* device = &context->devices[i];
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
    }
    return 0;
}

int
rocmon_smi_stopCounters(RocmonContext* context)
{
    int ret;

    // Ensure rocmon is initialized
    if (!rocmon_smi_initialized)
    {
        return -EFAULT;
    }
    return 0;
}


static int
rocmon_smi_getEventsOfGpu(RocmonContext* context, int gpuIdx, EventList_rocm_t* list)
{
    EventList_rocm_t tmpList = NULL;
    Event_rocm_t* tmpEventList = NULL;
    // Ensure rocmon is initialized
    if (!rocmon_smi_initialized)
    {
        return -EFAULT;
    }
    // Validate args
    if ((gpuIdx < 0) || (gpuIdx > rocmon_context->numDevices) || (!list))
    {
        return -EINVAL;
    }

    RocmonDevice* device = &rocmon_context->devices[gpuIdx];

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
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Add %d ROCm SMI events, get_map_size(device->smiMetrics));
    if (get_map_size(device->smiMetrics) == 0)
    {
        // No events -> return list
        *list = tmpList;
        return 0;
    }
    // (Re-)Allocate event array
    tmpEventList = realloc(tmpList->events, (tmpList->numEvents + get_map_size(device->smiMetrics)) * sizeof(Event_rocm_t));
    if (!tmpEventList)
    {
        if (!*list) free(tmpList);
        ERROR_PLAIN_PRINT(Cannot allocate events for event list);
        return -ENOMEM;
    }
    tmpList->events = tmpEventList;
    int startindex = tmpList->numEvents;

    // Copy ROCm SMI metric information
    for (int i = 0; i < get_map_size(device->smiMetrics); i++)
    {
        RocmonSmiEvent* event = NULL;
        Event_rocm_t* out = &tmpList->events[startindex + i];
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
        tmpList->numEvents++;
    }

    *list = tmpList;
    return 0;
}


int rocmon_smi_init(RocmonContext* context, int numGpus, const int* gpuIds)
{
    int ret = 0;
    if ((!context) || (numGpus <= 0) || (!gpuIds))
    {
        return -EINVAL;
    }

    ret = _rocmon_smi_link_libraries();
    if (ret < 0)
    {
        return -EFAULT;
    }

    // init rocm smi library
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Initializing RSMI);
    RSMI_CALL(rsmi_init, (0),
    {
        ERROR_PLAIN_PRINT(Failed to init rocm_smi);
        goto rocmon_init_rsmi_failed;
    });

    // Get available SMI events for devices
    _rocmon_smi_init_events(context);
    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice *device = &context->devices[i];
        // Initialize SMI events map
        if (init_map(&device->smiMetrics, MAP_KEY_TYPE_STR, 0, &free) < 0)
        {
            ERROR_PLAIN_PRINT(Cannot init smiMetrics map);
            goto rocmon_init_rsmi_failed;
        }
        if (_rocmon_smi_get_functions(device) < 0)
        {
            ERROR_PRINT(Failed to get SMI functions for device %d, device->deviceId);
            goto rocmon_init_rsmi_failed;
        }
        device->activeSmiEvents = NULL;
        device->smiMetrics = NULL;
    }
    rocmon_smi_initialized = TRUE;
    return 0;
rocmon_init_rsmi_failed:
    RSMI_CALL(rsmi_shut_down, (), {
        // fall through
    });
    return 0;
}


void rocmon_smi_finalize(RocmonContext* context)
{
    if (!rocmon_smi_initialized)
    {
        return;
    }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Finalize LIKWID ROCMON SMI);
    if (context)
    {
        if (context->devices)
        {
            // Free each devices fields
            for (int i = 0; i < context->numDevices; i++)
            {
                RocmonDevice* device = &context->devices[i];
                if (device->activeSmiEvents)
                {
                    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Freeing active SMI events for device %d, device->deviceId);
                    free(device->activeSmiEvents);
                    device->activeSmiEvents = NULL;
                    device->numActiveSmiEvents = 0;
                }
                if (device->smiMetrics)
                {
                    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Freeing SMI event list for device %d, device->deviceId);
                    destroy_smap(device->smiMetrics);
                    device->smiMetrics = NULL;
                }
            }
        }
        if (context->smiEvents)
        {
            ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Freeing SMI event list);
            destroy_smap(context->smiEvents);
            context->smiEvents = NULL;
        }
    }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Shutdown RSMI);
    RSMI_CALL(rsmi_shut_down, (), {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Shutdown SMI);
        // fall through
    });
    rocmon_smi_initialized = FALSE;
}

int
rocmon_smi_switchActiveGroup(RocmonContext* context, int newGroupId)
{
    int ret;

    ret = rocmon_smi_stopCounters(context);
    if (ret < 0)
    {
        return ret;
    }

    ret = rocmon_smi_setupCounters(context, newGroupId);
    if (ret < 0)
    {
        return ret;
    }

    ret = rocmon_smi_startCounters(context);
    if (ret < 0)
    {
        return ret;
    }

    return 0;
}

#endif /* LIKWID_ROCMON_SMI_H */
