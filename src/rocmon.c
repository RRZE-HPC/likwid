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
#include <amd_smi/amdsmi.h>
#if AMDSMI_LIB_VERSION_YEAR == 23 && AMDSMI_LIB_VERSION_MAJOR == 4 && AMDSMI_LIB_VERSION_MINOR == 0 && AMDSMI_LIB_VERSION_RELEASE == 0
typedef struct metrics_table_header_t metrics_table_header_t;
#endif
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

__attribute__((visibility("default")))
int likwid_rocmon_verbosity = DEBUGLEV_ONLY_ERROR;

// Macros
#define membersize(type, member) sizeof(((type *) NULL)->member)
#define ROCM_CALL(handleerror, func, ...) \
    do { \
        hsa_status_t s = (*func##_ptr)(__VA_ARGS__);\
        if (s != HSA_STATUS_SUCCESS && s != HSA_STATUS_INFO_BREAK) {           \
            const char *errstr = NULL;\
            rocprofiler_error_string(&errstr); \
            ERROR_PRINT("Error: function %s failed with error: '%s' (hsa_status_t=%d).", #func, errstr, s);\
            handleerror;\
        }\
    } while (0)

#define RSMI_CALL(handleerror, func, ...)\
    do {\
        rsmi_status_t s = (*func##_ptr)(__VA_ARGS__);\
        if (s != RSMI_STATUS_SUCCESS) {\
            const char *errstr = NULL;\
            rsmi_status_string_ptr(s, &errstr);\
            ERROR_PRINT("Error: function %s failed with error: '%s' (rsmi_status_t=%d)", #func, errstr, s);\
            handleerror;\
        }\
    } while (0)

// ROCm function declarations
#define ROCMWEAK __attribute__(( weak ))
#define DECLAREFUNC_HSA(funcname, ...) hsa_status_t __attribute__((weak)) funcname(__VA_ARGS__);  static hsa_status_t (*funcname##_ptr)(__VA_ARGS__);
#define DECLAREFUNC_SMI(funcname, ...) rsmi_status_t __attribute__((weak)) funcname(__VA_ARGS__);  static rsmi_status_t (*funcname##_ptr)(__VA_ARGS__);

DECLAREFUNC_HSA(hsa_init);
DECLAREFUNC_HSA(hsa_shut_down);
DECLAREFUNC_HSA(hsa_iterate_agents, hsa_status_t (*callback)(hsa_agent_t agent, void* data), void* data);
DECLAREFUNC_HSA(hsa_agent_get_info, hsa_agent_t agent, hsa_agent_info_t attribute, void* value);
DECLAREFUNC_HSA(hsa_system_get_info, hsa_system_info_t attribute, void *value);

DECLAREFUNC_HSA(rocprofiler_iterate_info, const hsa_agent_t* agent, rocprofiler_info_kind_t kind, hsa_status_t (*callback)(const rocprofiler_info_data_t, void* data), void* data);
DECLAREFUNC_HSA(rocprofiler_close, rocprofiler_t* context);
DECLAREFUNC_HSA(rocprofiler_open, hsa_agent_t agent, rocprofiler_feature_t* features, uint32_t feature_count, rocprofiler_t** context, uint32_t mode, rocprofiler_properties_t* properties);
DECLAREFUNC_HSA(rocprofiler_error_string);
DECLAREFUNC_HSA(rocprofiler_start, rocprofiler_t* context, uint32_t group_index);
DECLAREFUNC_HSA(rocprofiler_stop, rocprofiler_t* context, uint32_t group_index);
DECLAREFUNC_HSA(rocprofiler_read, rocprofiler_t* context, uint32_t group_index);
DECLAREFUNC_HSA(rocprofiler_get_data, rocprofiler_t* context, uint32_t group_index);
DECLAREFUNC_HSA(rocprofiler_get_metrics, const rocprofiler_t* context);

DECLAREFUNC_SMI(rsmi_init, uint64_t flags);
DECLAREFUNC_SMI(rsmi_shut_down);
DECLAREFUNC_SMI(rsmi_dev_supported_func_iterator_open, uint32_t dv_ind, rsmi_func_id_iter_handle_t* handle);
DECLAREFUNC_SMI(rsmi_dev_supported_variant_iterator_open, rsmi_func_id_iter_handle_t obj_h, rsmi_func_id_iter_handle_t* var_iter);
DECLAREFUNC_SMI(rsmi_func_iter_value_get, rsmi_func_id_iter_handle_t handle, rsmi_func_id_value_t* value );
DECLAREFUNC_SMI(rsmi_func_iter_next, rsmi_func_id_iter_handle_t handle);
DECLAREFUNC_SMI(rsmi_dev_supported_func_iterator_close, rsmi_func_id_iter_handle_t* handle);
DECLAREFUNC_SMI(rsmi_dev_power_ave_get, uint32_t dv_ind, uint32_t sensor_ind, uint64_t* power);
DECLAREFUNC_SMI(rsmi_dev_pci_throughput_get, uint32_t dv_ind, uint64_t* sent, uint64_t* received, uint64_t* max_pkt_sz);
DECLAREFUNC_SMI(rsmi_dev_pci_replay_counter_get, uint32_t dv_ind, uint64_t* counter);
DECLAREFUNC_SMI(rsmi_dev_memory_total_get, uint32_t dv_ind, rsmi_memory_type_t mem_type, uint64_t* total);
DECLAREFUNC_SMI(rsmi_dev_memory_usage_get, uint32_t dv_ind, rsmi_memory_type_t mem_type, uint64_t* used );
DECLAREFUNC_SMI(rsmi_dev_memory_busy_percent_get, uint32_t dv_ind, uint32_t* busy_percent);
DECLAREFUNC_SMI(rsmi_dev_memory_reserved_pages_get, uint32_t dv_ind, uint32_t* num_pages, rsmi_retired_page_record_t* records);
DECLAREFUNC_SMI(rsmi_dev_fan_rpms_get, uint32_t dv_ind, uint32_t sensor_ind, int64_t* speed);
DECLAREFUNC_SMI(rsmi_dev_fan_speed_get, uint32_t dv_ind, uint32_t sensor_ind, int64_t* speed);
DECLAREFUNC_SMI(rsmi_dev_fan_speed_max_get, uint32_t dv_ind, uint32_t sensor_ind, uint64_t* max_speed);
DECLAREFUNC_SMI(rsmi_dev_temp_metric_get, uint32_t dv_ind, uint32_t sensor_type, rsmi_temperature_metric_t metric, int64_t* temperature);
DECLAREFUNC_SMI(rsmi_dev_volt_metric_get, uint32_t dv_ind, rsmi_voltage_type_t sensor_type, rsmi_voltage_metric_t metric, int64_t* voltage);
DECLAREFUNC_SMI(rsmi_dev_overdrive_level_get, uint32_t dv_ind, uint32_t* od);
DECLAREFUNC_SMI(rsmi_dev_ecc_count_get, uint32_t dv_ind, rsmi_gpu_block_t block, rsmi_error_count_t* ec);
DECLAREFUNC_SMI(rsmi_compute_process_info_get, rsmi_process_info_t* procs, uint32_t* num_items);
DECLAREFUNC_SMI(rsmi_status_string, rsmi_status_t status, const char **status_string);


// ----------------------------------------------------
//   SMI event wrapper
// ----------------------------------------------------

static int
_smi_wrapper_pci_throughput_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    assert(rocmon_ctx != NULL);

    RPR_CALL(return -EIO, rocprofiler_create_context, &rocmon_ctx->rocprofCtx);

    for (size_t i = 0; i < rocmon_ctx->numDevices; i++) {
        int err = rocmon_device_init(i);
        if (err < 0) {
            ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "rocmon device init failed");
            // The doc doesn't say anthing about what to return here. Let's just return a negative value?
            return -1;
        }
    }

    return 0;
}

static void tool_fini(void *) { }

static RocmonDevice *device_get(int hipDeviceId)
{
    assert(rocmon_ctx != NULL);

    for (size_t i = 0; i < rocmon_ctx->numDevices; i++) {
        RocmonDevice *deviceCandidate = &rocmon_ctx->devices[i];

        if (!deviceCandidate->enabled)
            continue;

        if (deviceCandidate->hipDeviceId == hipDeviceId)
            return deviceCandidate;
    }

    return NULL;
}

static void rocmon_smi_event_list_free(void *event_list_raw)
{
    RocmonSmiEventList *event_list = event_list_raw;

    if (!event_list)
        return;

    free(event_list->entries);
    free(event_list);
}

static rocprofiler_tool_configure_result_t *rocprofiler_configure_private(
    uint32_t version, const char *, uint32_t, rocprofiler_client_id_t *id)
{
    id->name = "LIKWID rocmon";

    const uint32_t major = version / 10000;

    assert(major == 1);

    static rocprofiler_tool_configure_result_t cfg = {
        sizeof(cfg),
        &tool_init,
        &tool_fini,
        NULL,
    };

    return &cfg;
}

static void *dlopen_any(const char *const *filenames, size_t num_filenames, int flags)
{
    // dlopen the first loadable candidate when multiple exist (e.g. ***.so, ***.so.1, etc.)
    for (size_t i = 0; i < num_filenames; i++) {
        const char *filename = filenames[i];

        void *retval = dlopen(filename, flags);
        if (retval)
            return retval;

        ROCMON_INFO_PRINT("Unable to load '%s': %s, trying next candidate...", filename, dlerror());
    }

    ROCMON_INFO_PRINT("Error, no loadable candidate found");
    return NULL;
}

static int rocmon_libraries_init(void)
{
    // helper macro
#define DLSYM_CHK2(dllib, name, symname)                                                           \
    do {                                                                                           \
        name##_ptr       = dlsym(dllib, #symname);                                                 \
        const char *err_ = dlerror();                                                              \
        if (err_) {                                                                                \
            ERROR_PRINT("Failed to link '%s': %s", #symname, err_);                                \
            err = -ENXIO;                                                                          \
            goto ret_err;                                                                          \
        }                                                                                          \
    } while (0)
#define DLSYM_CHK(dllib, name) DLSYM_CHK2(dllib, name, name)

    // Initialization must only occur a single time
    assert(lib_rocm_smi == NULL);
    assert(lib_rocprofiler_sdk == NULL);
    assert(lib_amdhip == NULL);

    // Load rocprofiler-sdk library
    static const char *const rocprofiler_names[] = {
        "librocprofiler-sdk.so",
        "librocprofiler-sdk.so.1",
        "librocprofiler-sdk.so.1.0.0",
    };

    int err = 0;
    lib_rocprofiler_sdk =
        dlopen_any(rocprofiler_names, ARRAY_COUNT(rocprofiler_names), RTLD_GLOBAL | RTLD_NOW);
    if (!lib_rocprofiler_sdk) {
        err = -ELIBACC;
        goto ret_err;
    }

    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_create_counter_config);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_destroy_counter_config);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_query_record_counter_id);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_query_counter_info);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_create_context);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_query_available_agents);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_iterate_agent_supported_counters);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_create_buffer);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_destroy_buffer);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_flush_buffer);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_create_callback_thread);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_assign_callback_thread);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_configure_buffer_dispatch_counting_service);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_configure_device_counting_service);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_sample_device_counting_service);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_start_context);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_stop_context);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_context_is_active);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_context_is_valid);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_force_configure);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_get_status_string);

    // Load rocm smi library
    lib_rocm_smi = dlopen("librocm_smi64.so", RTLD_GLOBAL | RTLD_NOW);
    if (!lib_rocm_smi) {
        err = -ELIBACC;
        goto ret_err;
    }

    DLSYM_CHK(lib_rocm_smi, rsmi_init);
    DLSYM_CHK(lib_rocm_smi, rsmi_shut_down);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_supported_func_iterator_open);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_supported_variant_iterator_open);
    DLSYM_CHK(lib_rocm_smi, rsmi_func_iter_value_get);
    DLSYM_CHK(lib_rocm_smi, rsmi_func_iter_next);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_supported_func_iterator_close);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_power_ave_get);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_pci_throughput_get);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_pci_replay_counter_get);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_memory_total_get);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_memory_usage_get);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_memory_busy_percent_get);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_memory_reserved_pages_get);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_fan_rpms_get);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_fan_speed_get);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_fan_speed_max_get);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_temp_metric_get);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_volt_metric_get);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_overdrive_level_get);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_ecc_count_get);
    DLSYM_CHK(lib_rocm_smi, rsmi_compute_process_info_get);
    DLSYM_CHK(lib_rocm_smi, rsmi_status_string);
    DLSYM_CHK(lib_rocm_smi, rsmi_num_monitor_devices);
    DLSYM_CHK(lib_rocm_smi, rsmi_dev_pci_id_get);

    lib_amdhip = dlopen("libamdhip64.so", RTLD_GLOBAL | RTLD_NOW);
    if (!lib_amdhip) {
        err = -ELIBACC;
        goto ret_err;
    }

    DLSYM_CHK(lib_amdhip, hipGetDeviceCount);
    DLSYM_CHK2(lib_amdhip, hipGetDeviceProperties, hipGetDevicePropertiesR0600);
    DLSYM_CHK(lib_amdhip, hipFree);
    DLSYM_CHK(lib_amdhip, hipGetErrorName);
    DLSYM_CHK(lib_amdhip, hipInit);

    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "Linking AMD ROCMm libraries done");

    return 0;

ret_err:
    if (lib_amdhip) {
        dlclose(lib_amdhip);
        lib_amdhip = NULL;
    }

    if (lib_rocm_smi) {
        dlclose(lib_rocm_smi);
        lib_rocm_smi = NULL;
    }

    if (lib_rocprofiler_sdk) {
        dlclose(lib_rocprofiler_sdk);
        lib_rocprofiler_sdk = NULL;
    }

    return err;

#undef DLSYM_CHK
}

static void rocmon_libraries_fini(void)
{
    dlclose(lib_rocprofiler_sdk);
    lib_rocprofiler_sdk = NULL;

    dlclose(lib_rocm_smi);
    lib_rocm_smi = NULL;
}

void rocmon_setVerbosity(int level)
{
    if (level < DEBUGLEV_ONLY_ERROR)
        level = DEBUGLEV_ONLY_ERROR;
    else if (level > DEBUGLEV_DEVELOP)
        level = DEBUGLEV_DEVELOP;

    likwid_rocmon_verbosity = level;
}

static void format_smi_event_label(char *buf, size_t size, RocmonSmiEventType type,
    const char *function, uint64_t variant, uint64_t subvariant)
{
    switch (type) {
    case ROCMON_SMI_EVENT_TYPE_NORMAL:
    case ROCMON_SMI_EVENT_TYPE_INSTANCES:
        snprintf(buf, size, "%s", function);
        break;
    case ROCMON_SMI_EVENT_TYPE_VARIANT:
        snprintf(buf, size, "%s|%" PRIu64, function, variant);
        break;
    case ROCMON_SMI_EVENT_TYPE_SUBVARIANT:
        snprintf(buf, size, "%s|%" PRIu64 "|%" PRIu64, function, variant, subvariant);
        break;
    default:
        ERROR_PRINT("Internal LIKWID bug: Invalid rocmon smi event type: %d", type);
        abort();
    }
}

static int smi_event_add_impl(const char *name, RocmonSmiEventType type, const char *function,
    uint64_t variant, uint64_t subvariant, uint64_t extra, RocmonSmiMeasureFunc measureFunc)
{
    /* In this function we add events, which are supported by LIKWID.
     * This does not guarantee that they are actually available on the hardware.
     * Therefore this is added to rocmon_ctx instead of per device.
     * Instead of a simple mapping of label -> event, we use a mapping of label -> event_list.
     * That is because in a few cases we implement multiple events with a single RSMI function. */
    char label[256];
    format_smi_event_label(label, sizeof(label), type, function, variant, subvariant);

    /* Insert new Event List for given label, if it doesn't already exists. */
    RocmonSmiEventList *list = NULL;
    if (get_smap_by_key(rocmon_ctx->implementedSmiEvents, label, (void **)&list) < 0) {
        list = calloc(1, sizeof(*list));
        if (!list)
            return -errno;

        int err = add_smap(rocmon_ctx->implementedSmiEvents, label, list);
        if (err < 0) {
            free(list);
            return err;
        }
    }

    /* Add event to list */
    const size_t newNumEntries = list->numEntries + 1;
    RocmonSmiEvent *newEntries = realloc(list->entries, newNumEntries * sizeof(*newEntries));
    if (!newEntries)
        return -errno;

    RocmonSmiEvent *newEvent = &newEntries[list->numEntries];

    list->numEntries = newNumEntries;
    list->entries    = newEntries;

    snprintf(newEvent->name, sizeof(newEvent->name)-1, "%s", name);
    newEvent->type       = type;
    newEvent->variant    = variant;
    newEvent->subvariant = subvariant;
    newEvent->extra =
        extra; // 'extra' is used to differentiate for multiple events, which use the same RSMI function.
    newEvent->measureFunc = measureFunc;

    return 0;
}

static int smi_events_add_avail(RocmonDevice *device, RocmonSmiEventType type, const char *function,
    uint64_t variant, uint64_t subvariant)
{
    char label[256];
    format_smi_event_label(label, sizeof(label), type, function, variant, subvariant);

    RocmonSmiEventList *list = NULL;
    int err = get_smap_by_key(rocmon_ctx->implementedSmiEvents, label, (void **)&list);
    if (err < 0) {
        ROCMON_DEBUG_PRINT(
            DEBUGLEV_DEVELOP, "ROCM-SMI supports event '%s', but we don't implement it", label);
        return 0;
    }

    for (size_t i = 0; i < list->numEntries; i++) {
        RocmonSmiEvent *implEvent = &list->entries[i];

        RocmonSmiEvent *availEvent = malloc(sizeof(*availEvent));
        if (!availEvent)
            return -errno;

        assert(type == implEvent->type);
        assert(variant == implEvent->variant);

        if (type == ROCMON_SMI_EVENT_TYPE_INSTANCES) {
            // For instanced events (like sensor lists), create a list of events
            int len = snprintf(availEvent->name,
                               sizeof(availEvent->name),
                               "%s[%zu]",
                               implEvent->name, subvariant);
            if (len < 0) {
                ERROR_PRINT("Failed to add subvariant %zu to event %s\n", subvariant, implEvent->name);
                continue;
            };
            availEvent->subvariant = subvariant;
        } else {
            assert(subvariant == implEvent->subvariant);
            snprintf(availEvent->name, sizeof(availEvent->name), "%s", implEvent->name);
            availEvent->subvariant = implEvent->subvariant;
        }

        availEvent->type        = implEvent->type;
        availEvent->variant     = implEvent->variant;
        availEvent->extra       = implEvent->extra;
        availEvent->measureFunc = implEvent->measureFunc;

        err = add_smap(device->availableSmiEvents, availEvent->name, availEvent);
        if (err < 0)
            return err;
    }

    return 0;
}

static int smi_init_events_subvariant(RocmonDevice *device,
    rsmi_func_id_iter_handle_t variant_iter_handle, const char *function, uint64_t variant)
{
    // Iterate over all sub variants begin
    rsmi_func_id_iter_handle_t subvariant_iter_handle;
    rsmi_status_t rerr =
        rsmi_dev_supported_variant_iterator_open_ptr(variant_iter_handle, &subvariant_iter_handle);

    if (rerr == RSMI_STATUS_NO_DATA) {
        // No subvariants for given function
        return smi_events_add_avail(device, ROCMON_SMI_EVENT_TYPE_VARIANT, function, variant, 0);
    } else if (rerr != RSMI_STATUS_SUCCESS) {
        const char *errstr = NULL;
        rsmi_status_string_ptr(rerr, &errstr);
        ERROR_PRINT("rsmi_dev_supported_variant_iterator_open failed: %s", errstr);
        return -EIO;
    }

    // Iterate over all sub variants body
    int err = 0;
    while (true) {
        // Get sub variant value
        rsmi_func_id_value_t subvariant_value;
        RSMI_CALL(err = -EIO;
            break, rsmi_func_iter_value_get, subvariant_iter_handle, &subvariant_value);

        RocmonSmiEventType type = (variant == RSMI_DEFAULT_VARIANT)
                                      ? ROCMON_SMI_EVENT_TYPE_INSTANCES
                                      : ROCMON_SMI_EVENT_TYPE_SUBVARIANT;
        int err = smi_events_add_avail(device, type, function, variant, subvariant_value.id);
        if (err < 0)
            return err;

        if (rsmi_func_iter_next_ptr(subvariant_iter_handle) == RSMI_STATUS_NO_DATA)
            break;
    }

    // Iterate over all sub variants end
    RSMI_CALL(abort(), rsmi_dev_supported_func_iterator_close, &subvariant_iter_handle);
    return err;
}

static int smi_init_events_variant(
    RocmonDevice *device, rsmi_func_id_iter_handle_t function_iter_handle, const char *function)
{
    // Iterate over all variants begin
    rsmi_func_id_iter_handle_t variant_iter_handle;
    rsmi_status_t rerr =
        rsmi_dev_supported_variant_iterator_open_ptr(function_iter_handle, &variant_iter_handle);

    if (rerr == RSMI_STATUS_NO_DATA) {
        // No variants for given function
        return smi_events_add_avail(device, ROCMON_SMI_EVENT_TYPE_NORMAL, function, 0, 0);
    } else if (rerr != RSMI_STATUS_SUCCESS) {
        const char *errstr = NULL;
        rsmi_status_string_ptr(rerr, &errstr);
        ERROR_PRINT("rsmi_dev_supported_variant_iterator_open failed: %s", errstr);
        return -EIO;
    }

    RSMI_CALL(return -EIO,
        rsmi_dev_supported_variant_iterator_open,
        function_iter_handle,
        &variant_iter_handle);

    // Iterate over all variants body
    int err = 0;
    while (true) {
        // Get variant value
        rsmi_func_id_value_t variant_value;
        RSMI_CALL(err = -EIO; break, rsmi_func_iter_value_get, variant_iter_handle, &variant_value);

        err = smi_init_events_subvariant(device, variant_iter_handle, function, variant_value.id);
        if (err < 0)
            break;

        if (rsmi_func_iter_next_ptr(variant_iter_handle) == RSMI_STATUS_NO_DATA)
            break;
    }

    // Iterate over all variants end
    RSMI_CALL(abort(), rsmi_dev_supported_func_iterator_close, &variant_iter_handle);
    return err;
}

static int smi_init_events_normal(RocmonDevice *device)
{
    int err = init_map(&device->availableSmiEvents, MAP_KEY_TYPE_STR, 0, free);
    if (err < 0)
        return err;

    // For explanations what "normal", "variant", and "subvariants" are,
    // please consult documentation of rocm_smi.

    // Iterate over all functions begin
    rsmi_func_id_iter_handle_t function_iter_handle;
    RSMI_CALL(return -EIO,
        rsmi_dev_supported_func_iterator_open,
        device->rsmiDeviceId,
        &function_iter_handle);

    // Iterate over all functions body
    while (true) {
        // Get function value
        rsmi_func_id_value_t function_value;
        RSMI_CALL(err = -EIO;
            break, rsmi_func_iter_value_get, function_iter_handle, &function_value);

        err = smi_init_events_variant(device, function_iter_handle, function_value.name);
        if (err < 0)
            break;

        if (rsmi_func_iter_next_ptr(function_iter_handle) == RSMI_STATUS_NO_DATA)
            break;
    }

    // Iterate over all functions end
    RSMI_CALL(abort(), rsmi_dev_supported_func_iterator_close, &function_iter_handle);
    if (err < 0)
        return err;

    // Add additional device independent functions
    return smi_events_add_avail(
        device, ROCMON_SMI_EVENT_TYPE_NORMAL, "rsmi_compute_process_info_get", 0, 0);
}

static rocprofiler_status_t counter_iterate_cb(
    rocprofiler_agent_id_t, rocprofiler_counter_id_t *counters, size_t num_counters, void *userdata)
{
    RocmonDevice *device = userdata;

    for (size_t i = 0; i < num_counters; i++) {
        RocmonRprEvent *availEvent = calloc(1, sizeof(*availEvent));
        if (!availEvent)
            return -errno;

        RPR_CALL(return -EIO,
            rocprofiler_query_counter_info,
            counters[i],
            ROCPROFILER_COUNTER_INFO_VERSION_1,
            &availEvent->counterInfo);

        if (add_smap(device->availableRprEvents, availEvent->counterInfo.name, availEvent) < 0) {
            free(availEvent);
            return ROCPROFILER_STATUS_ERROR;
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

static int rpr_init_events(RocmonDevice *device)
{
    int err = init_map(&device->availableRprEvents, MAP_KEY_TYPE_STR, 0, free);
    if (err < 0)
        return err;

    // rocprofCtx must already be initialized from 'tool_init' at this point.
    assert(rocmon_ctx->rocprofCtx.handle != 0);

    RPR_CALL(return -EIO,
        rocprofiler_iterate_agent_supported_counters,
        device->rocprofAgent->id,
        counter_iterate_cb,
        device);
    return 0;
}

static rocprofiler_status_t find_agent_for_rocmon_device(rocprofiler_agent_version_t agents_ver,
    const void **agents_arr_raw, size_t num_agents, void *userdata)
{
    if (agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0) {
        ERROR_PRINT("Unknown rocprofiler_agent version: %d", agents_ver);
        return ROCPROFILER_STATUS_ERROR;
    }

    RocmonDevice *device = userdata;

    const rocprofiler_agent_v0_t **agents_arr = (const rocprofiler_agent_v0_t **)agents_arr_raw;

    for (size_t i = 0; i < num_agents; i++) {
        const rocprofiler_agent_v0_t *agent_candidate = agents_arr[i];
        // Only allow GPU agents. This array will also have e.g. CPUs,
        // which we don't care about.
        if (agent_candidate->type != ROCPROFILER_AGENT_TYPE_GPU)
            continue;

        if (agent_candidate->domain == device->pciDomain &&
            agent_candidate->location_id == device->pciLocation) {
            device->rocprofAgent = agent_candidate;
            break;
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

static void set_counter_callback(rocprofiler_context_id_t context_id,
    rocprofiler_agent_id_t agent_id, rocprofiler_device_counting_agent_cb_t set_config,
    void *userdata)
{
    const RocmonDevice *device = userdata;

    assert(context_id.handle == rocmon_ctx->rocprofCtx.handle);
    assert(agent_id.handle == device->rocprofAgent->id.handle);

    rocprofiler_counter_config_id_t counter_config;

    RPR_CALL(return,
        rocprofiler_create_counter_config,
        agent_id,
        device->activeRprEvents,
        device->numActiveRprEvents,
        &counter_config);

    rocprofiler_status_t status = set_config(context_id, counter_config);
    if (status != ROCPROFILER_STATUS_SUCCESS)
        ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR,
            "rocprofiler-sdk: set_config failed: %s",
            rocprofiler_get_status_string_ptr(status));

    //RPR_CALL(
    //        return,
    //        rocprofiler_destroy_counter_config,
    //        counter_config
    //);
}

static void buffered_callback(rocprofiler_context_id_t, rocprofiler_buffer_id_t,
    rocprofiler_record_header_t **headers, size_t num_headers, void *userdata,
    uint64_t /* drop_count */)
{
    RocmonDevice *device = userdata;

    pthread_mutex_lock(&device->callbackRprMutex);

    for (size_t i = 0; i < num_headers; i++) {
        rocprofiler_record_header_t *header = headers[i];

        if (header->category != ROCPROFILER_BUFFER_CATEGORY_COUNTERS ||
            header->kind != ROCPROFILER_COUNTER_RECORD_VALUE)
            continue;

        rocprofiler_counter_id_t cid;
        rocprofiler_counter_record_t *record = header->payload;
        RPR_CALL(continue, rocprofiler_query_record_counter_id, record->id, &cid);

        char key[32];
        snprintf(key, sizeof(key), "%" PRIu64, cid.handle);

        double *value = NULL;
        int err       = get_smap_by_key(device->callbackRprResults, key, (void **)&value);
        if (err == -ENOENT) {
            value = calloc(1, sizeof(*value));
            if (!value) {
                ERROR_PRINT("Unable to allocate memory to store rocprofiler result");
                continue;
            }

            err = add_smap(device->callbackRprResults, key, value);
            if (err < 0) {
                ERROR_PRINT("Unable to save rocprofiler result to map: %s", strerror(-err));
                free(value);
                continue;
            }
        } else if (err < 0) {
            ERROR_PRINT("Error while getting value from result map: %s", strerror(-err));
            continue;
        }

        *value += record->counter_value;
    }

    pthread_mutex_unlock(&device->callbackRprMutex);
}

static int rpr_device_init(RocmonDevice *device)
{
    /* First we have to find which rocprofiler agent belongs to which RocmonDevice.
     * We do this via the PCI location. */

    RPR_CALL(return -EIO,
        rocprofiler_query_available_agents,
        ROCPROFILER_AGENT_INFO_VERSION_0,
        find_agent_for_rocmon_device,
        sizeof(rocprofiler_agent_t),
        device);

    // If the callback didn't match any available agent to our hip device we fail.
    if (!device->rocprofAgent)
        return -ENODEV;

    RPR_CALL(return -EIO,
        rocprofiler_create_buffer,
        rocmon_ctx->rocprofCtx,
        4096, // TODO ??? how do we choose a proper value?
        2048, // TODO ???
        ROCPROFILER_BUFFER_POLICY_LOSSLESS,
        buffered_callback,
        device,
        &device->rocprofBuf);

    RPR_CALL(return -EIO, rocprofiler_create_callback_thread, &device->rocprofThrd);

    RPR_CALL(
        return -EIO, rocprofiler_assign_callback_thread, device->rocprofBuf, device->rocprofThrd);

    // The set_counter_callback is not called here. It will be called later
    // during rocprofiler_start_context.
    // product_name e.g.: 'AMD Instinct MI210'
    // name e.g.: 'gfx90a'
    ROCMON_DEBUG_PRINT(DEBUGLEV_INFO,
        "Using device: '%s' (%s, RSMI-ID=%u)\n",
        device->rocprofAgent->product_name,
        device->rocprofAgent->name,
        device->rsmiDeviceId);
    RPR_CALL(return -EIO,
        rocprofiler_configure_device_counting_service,
        rocmon_ctx->rocprofCtx,
        device->rocprofBuf,
        device->rocprofAgent->id,
        set_counter_callback,
        device);

    int err = init_map(&device->callbackRprResults, MAP_KEY_TYPE_STR, 0, free);
    if (err < 0)
        return err;

    pthread_mutex_init(&device->callbackRprMutex, NULL);

    return 0;
}

static int smi_device_init(RocmonDevice *device)
{
    uint64_t bdfid;
    RSMI_CALL(return -EIO, rsmi_dev_pci_id_get, device->rsmiDeviceId, &bdfid);

    /* For details about the format of bdfid, check rocm_smi.h. As far as I can tell
     * there are no helper macros available to do this more nicely. */
    device->pciDomain   = (uint32_t)(bdfid >> 32);
    device->pciLocation = (uint32_t)(bdfid >> 0);
    return 0;
}

static int rocmon_device_init(size_t ctxDeviceIdx)
{
    if (ctxDeviceIdx >= rocmon_ctx->numDevices)
        return -EINVAL;

    RocmonDevice *device = &rocmon_ctx->devices[ctxDeviceIdx];

    device->rsmiDeviceId = ctxDeviceIdx;

    int err = smi_device_init(device);
    if (err < 0)
        return err;

    err = rpr_device_init(device);
    if (err < 0)
        return err;

    // Init SMI events
    err = smi_init_events_normal(device);
    if (err < 0)
        return err;

    // Init rocprofiler-sdk events
    err = rpr_init_events(device);
    if (err < 0)
        return err;

    return 0;
}

static int parse_hex(char c)
{
    if (c >= '0' && c <= '9')
        return (uint8_t)(c - '0');
    if (c >= 'a' && c <= 'f')
        return (uint8_t)(c - 'a' + 10);
    if (c >= 'A' && c <= 'F')
        return (uint8_t)(c - 'A' + 10);
    return 0;
}

static bool hip_uuid_equal_rocprof_uuid(
    const hipUUID *hip_uuid, const rocprofiler_uuid_t *rocp_uuid)
{
    assert(sizeof(hip_uuid->bytes) == 16);
    assert(sizeof(rocp_uuid->bytes) == 16);

    // For some reason hipUUID is stored in ASCII, while rocprofiler_uuid_t is stored in binary.
    for (size_t h_i = 0, r_i = 7; h_i < sizeof(*hip_uuid); h_i += 2, r_i--) {
        const char h_a        = (char)hip_uuid->bytes[h_i];
        const char h_b        = (char)hip_uuid->bytes[h_i + 1];
        const int h_digit     = (parse_hex(h_a) << 4) | parse_hex(h_b);
        const uint8_t r_digit = rocp_uuid->bytes[r_i];

        if (h_digit != r_digit)
            return false;
    }
    return true;
}

static int rocmon_init_hip(size_t numGpuIds, const int *gpuIds)
{
    // This function is separated from rocmon_device_init, since we need HIP, which must
    // not be initialized before 'tool_init' finishes. So instead we do all HIP related
    // things here.
    HIP_CALL(return -EIO, hipInit, 0);

    // Get number of devices
    int availDeviceCount;
    HIP_CALL(return -EIO, hipGetDeviceCount, &availDeviceCount);

    if (gpuIds == NULL && numGpuIds == 0)
        numGpuIds = availDeviceCount;

    if (numGpuIds > (size_t)availDeviceCount)
        return -EINVAL;

    rocmon_ctx->hipDeviceIdxToRocmonDeviceIdx =
        calloc(numGpuIds, sizeof(*rocmon_ctx->hipDeviceIdxToRocmonDeviceIdx));
    if (!rocmon_ctx->hipDeviceIdxToRocmonDeviceIdx)
        return -errno;

    rocmon_ctx->numHipDeviceIdxToRocmonDeviceIdx = numGpuIds;

    // Find matching RocmonDevice via UUID
    for (size_t i = 0; i < numGpuIds; i++) {
        const int gpuId = gpuIds ? gpuIds[i] : (int)i;

        hipDeviceProp_t hipProps;
        HIP_CALL(return -EIO, hipGetDeviceProperties, &hipProps, gpuId);

        bool found = false;

        for (size_t j = 0; j < rocmon_ctx->numDevices; j++) {
            RocmonDevice *device = &rocmon_ctx->devices[j];
            if (hip_uuid_equal_rocprof_uuid(&hipProps.uuid, &device->rocprofAgent->uuid)) {
                rocmon_ctx->hipDeviceIdxToRocmonDeviceIdx[i] = j;
                device->hipDeviceId                          = gpuId;
                device->hipProps                             = hipProps;
                device->enabled                              = true;
                found                                        = true;
                break;
            }
        }

        if (!found) {
            ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR,
                "Unable to find ROCm SMI / rocprofiler-sdk device for HIP device: %d",
                gpuId);
            return -ENODEV;
        }
    }

    return 0;
}

static void rocmon_device_fini(RocmonDevice *device)
{
    RPR_CALL(abort(), rocprofiler_flush_buffer, device->rocprofBuf);
    RPR_CALL(abort(), rocprofiler_destroy_buffer, device->rocprofBuf);

    if (device->callbackRprResults) {
        destroy_smap(device->callbackRprResults);
        pthread_mutex_destroy(&device->callbackRprMutex);
        device->callbackRprResults = NULL;
    }

    if (device->groupResults) {
        for (size_t i = 0; i < device->numGroupResults; i++) {
            RocmonEventResultList *groupResult = &device->groupResults[i];

            free(groupResult->eventResults);
        }

        free(device->groupResults);
        device->groupResults = NULL;
    }

    free(device->activeSmiEvents);
    device->activeSmiEvents = NULL;

    free(device->activeRprEvents);
    device->activeRprEvents = NULL;

    destroy_smap(device->availableSmiEvents);
    device->availableSmiEvents = NULL;

    destroy_smap(device->availableRprEvents);
    device->availableRprEvents = NULL;
}

static void rocmon_groupinfo_fini(RocmonGroupInfo *group)
{
    perfgroup_returnGroup(&group->groupInfo);
}

static void rocmon_ctx_free(void)
{
    if (!rocmon_ctx)
        return;

    int isActive;
    RPR_CALL(abort(), rocprofiler_context_is_active, rocmon_ctx->rocprofCtx, &isActive);
    if (isActive)
        RPR_CALL(abort(), rocprofiler_stop_context, rocmon_ctx->rocprofCtx);

    free(rocmon_ctx->hipDeviceIdxToRocmonDeviceIdx);

    if (rocmon_ctx->devices) {
        for (size_t i = 0; i < rocmon_ctx->numDevices; i++)
            rocmon_device_fini(&rocmon_ctx->devices[i]);

        free(rocmon_ctx->devices);
    }

    if (rocmon_ctx->groups) {
        for (size_t i = 0; i < rocmon_ctx->numGroups; i++)
            rocmon_groupinfo_fini(&rocmon_ctx->groups[i]);

        free(rocmon_ctx->groups);
    }

    destroy_smap(rocmon_ctx->implementedSmiEvents);

    free(rocmon_ctx);
    rocmon_ctx = NULL;
}

static int rsmi_measurefunc_pci_throughput_get(
    uint32_t rsmiDevId, RocmonSmiEvent *event, RocmonEventResult *result)
{
    ROCMON_DEBUG_PRINT(
        DEBUGLEV_DEVELOP, "rsmi_measurefunc_pci_throughput_get(%d, %lu)", rsmiDevId, event->extra);

    uint64_t sent, received, max_pkt_sz;
    RSMI_CALL(return -EIO, rsmi_dev_pci_throughput_get, rsmiDevId, &sent, &received, &max_pkt_sz);

    uint64_t value;
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "_smi_wrapper_pci_throughput_get(%d, %lu)", deviceId, event->extra);
    // Internal variant: 0 for sent, 1 for received bytes and 2 for max packet size
    if (event->extra == 0)       RSMI_CALL(return -1, rsmi_dev_pci_throughput_get, deviceId, &value, NULL, NULL);
    else if (event->extra == 1)  RSMI_CALL(return -1, rsmi_dev_pci_throughput_get, deviceId, NULL, &value, NULL);
    else if (event->extra == 2)  RSMI_CALL(return -1, rsmi_dev_pci_throughput_get, deviceId, NULL, NULL, &value);
    else return -1;

    result->fullValue += value;
    result->lastValue = value;

    return 0;
}


static int
_smi_wrapper_pci_replay_counter_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    (void)event;

    uint64_t counter;
    RSMI_CALL(return -1, rsmi_dev_pci_replay_counter_get, deviceId, &counter);
    result->fullValue += counter;
    result->lastValue = counter;

    return 0;
}


static int
_smi_wrapper_power_ave_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t power;
    RSMI_CALL(return -1, rsmi_dev_power_ave_get, deviceId, event->subvariant, &power);
    result->fullValue += power;
    result->lastValue = power;

    return 0;
}


static int
_smi_wrapper_memory_total_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t total;
    RSMI_CALL(return -1, rsmi_dev_memory_total_get, deviceId, event->variant, &total);
    result->fullValue += total;
    result->lastValue = total;

    return 0;
}


static int
_smi_wrapper_memory_usage_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t used;
    RSMI_CALL(return -1, rsmi_dev_memory_usage_get, deviceId, event->variant, &used);
    result->fullValue += used;
    result->lastValue = used;

    return 0;
}


static int
_smi_wrapper_memory_busy_percent_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    (void)event;

    uint32_t percent;
    RSMI_CALL(return -1, rsmi_dev_memory_busy_percent_get, deviceId, &percent);
    result->fullValue += percent;
    result->lastValue = percent;

    return 0;
}


static int
_smi_wrapper_memory_reserved_pages_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    (void)event;

    uint32_t num_pages;
    RSMI_CALL(return -1, rsmi_dev_memory_reserved_pages_get, deviceId, &num_pages, NULL);
    result->fullValue += num_pages;
    result->lastValue = num_pages;

    return 0;
}


static int
_smi_wrapper_fan_rpms_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t speed;
    RSMI_CALL(return -1, rsmi_dev_fan_rpms_get, deviceId, event->subvariant, &speed);
    result->fullValue += speed;
    result->lastValue = speed;

    return 0;
}


static int
_smi_wrapper_fan_speed_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t speed;
    RSMI_CALL(return -1, rsmi_dev_fan_speed_get, deviceId, event->subvariant, &speed);
    result->fullValue += speed;
    result->lastValue = speed;

    return 0;
}


static int
_smi_wrapper_fan_speed_max_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t max_speed;
    RSMI_CALL(return -1, rsmi_dev_fan_speed_max_get, deviceId, event->subvariant, &max_speed);
    result->fullValue += max_speed;
    result->lastValue = max_speed;

    return 0;
}


static int
_smi_wrapper_temp_metric_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t temperature;
    RSMI_CALL(return -1, rsmi_dev_temp_metric_get, deviceId, event->subvariant, event->variant, &temperature);
    result->fullValue += temperature;
    result->lastValue = temperature;

    return 0;
}


static int
_smi_wrapper_volt_metric_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t voltage;
    RSMI_CALL(return -1, rsmi_dev_volt_metric_get, deviceId, event->subvariant, event->variant, &voltage);
    result->fullValue += voltage;
    result->lastValue = voltage;

    return 0;
}


static int
_smi_wrapper_overdrive_level_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    (void)event;

    uint32_t overdrive;
    RSMI_CALL(return -1, rsmi_dev_overdrive_level_get, deviceId, &overdrive);
    result->fullValue += overdrive;
    result->lastValue = overdrive;

    return 0;
}


static int
_smi_wrapper_ecc_count_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    rsmi_error_count_t error_count;
    RSMI_CALL(return -1, rsmi_dev_ecc_count_get, deviceId, event->variant, &error_count);

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
    (void)deviceId;
    (void)event;

    uint32_t num_items;
    RSMI_CALL(return -1, rsmi_compute_process_info_get, NULL, &num_items);
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
    #define DLSYM_AND_CHECK( dllib, name ) name##_ptr = dlsym( dllib, #name ); if ( dlerror() != NULL ) { ERROR_PRINT("Failed to link " #name); return -1; }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "Linking AMD ROCMm libraries");
  
    // Need to link in the ROCm HSA libraries
    dl_hsa_lib = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_hsa_lib)
    {
        ERROR_PRINT("ROCm HSA library libhsa-runtime64.so not found: %s", dlerror());
        return -1;
    }

    // Need to link in the Rocprofiler libraries
    dl_profiler_lib = dlopen("librocprofiler64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_profiler_lib)
    {
        dl_profiler_lib = dlopen("librocprofiler64.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!dl_profiler_lib)
        {
            ERROR_PRINT("Rocprofiler library librocprofiler64.so not found: %s", dlerror());
            return -1;
        }
    }

    // Need to link in the Rocprofiler libraries
    dl_rsmi_lib = dlopen("librocm_smi64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_rsmi_lib)
    {
        ERROR_PRINT("ROCm SMI library librocm_smi64.so not found: %s", dlerror());
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
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "Linking AMD ROCMm libraries done");
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
    (void)info;

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

    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "_rocmon_iterate_info_callback_add");
    if (likwid_rocmon_verbosity == DEBUGLEV_DEVELOP)
    {
        _rocmon_print_rocprofiler_info_data(info);
    }
    // Check info kind
    if (info.kind != ROCPROFILER_INFO_KIND_METRIC)
    {
        ERROR_PRINT("Wrong info kind %u", info.kind);
        return HSA_STATUS_ERROR;
    }

    // Check index
    if (arg->currIndex >= arg->device->numRocMetrics)
    {
        ERROR_PRINT("Metric index out of bounds: %d", arg->currIndex);
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
    bool noAgent = false;

    iterate_agents_cb_arg *arg = (iterate_agents_cb_arg*) argv;

    // Check if device is a GPU
    hsa_device_type_t type;
    ROCM_CALL(return -1, hsa_agent_get_info, agent, HSA_AGENT_INFO_DEVICE, &type);
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
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "Initializing agent %d", gpuIndex);

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
    ROCM_CALL(return HSA_STATUS_ERROR, rocprofiler_iterate_info, &agent, ROCPROFILER_INFO_KIND_METRIC, _rocmon_iterate_info_callback_count, device);
    ROCMON_DEBUG_PRINT(DEBUGLEV_INFO, "RocProfiler provides %d events", device->numRocMetrics);

    // workaround for bug in ROCm 5.4.0
    if(device->numRocMetrics == 0) {
        ROCM_CALL(return HSA_STATUS_ERROR, rocprofiler_iterate_info, NULL, ROCPROFILER_INFO_KIND_METRIC, _rocmon_iterate_info_callback_count, device);
        noAgent = true;
    }

    // Allocate memory for metrics
    device->rocMetrics = (rocprofiler_info_data_t*) malloc(device->numRocMetrics * sizeof(rocprofiler_info_data_t));
    if (device->rocMetrics == NULL)
    {
        ERROR_PRINT("Cannot allocate set of rocMetrics");
        return HSA_STATUS_ERROR;
    }

    // Initialize SMI events map
    if (init_map(&device->smiMetrics, MAP_KEY_TYPE_STR, 0, &free) < 0)
    {
        ERROR_PRINT("Cannot init smiMetrics map");
        return HSA_STATUS_ERROR;
    }

    // Fetch metric informatino
    iterate_info_cb_arg info_arg = {
        .device = device,
        .currIndex = 0,
    };
    ROCMON_DEBUG_PRINT(DEBUGLEV_INFO, "Read %d RocProfiler events for device %d", device->numRocMetrics, device->deviceId);

    // If the call fails with agent, call rocprofiler_iterate_info without agent
    if(noAgent)
    {
        ROCM_CALL(return HSA_STATUS_ERROR, rocprofiler_iterate_info, NULL, ROCPROFILER_INFO_KIND_METRIC, _rocmon_iterate_info_callback_add, &info_arg);
    } else {
        ROCM_CALL(return HSA_STATUS_ERROR, rocprofiler_iterate_info, &agent, ROCPROFILER_INFO_KIND_METRIC, _rocmon_iterate_info_callback_add, &info_arg);
    }

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
            ERROR_PRINT("Cannot transform %s to performance group", eventString);
            return err;
        }
    }
    else
    {
        // If performance group -> perfgroup_readGroup
        err = perfgroup_readGroup(config->groupPath, "amd_gpu", eventString, group);
        if (err == -EACCES)
        {
            ERROR_PRINT("Access to performance group %s not allowed", eventString);
            return err;
        }
        else if (err == -ENODEV)
        {
            ERROR_PRINT("Performance group %s only available with deactivated HyperThreading", eventString);
            return err;
        }
        if (err < 0)
        {
            ERROR_PRINT("Cannot read performance group %s", eventString);
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
    ROCM_CALL(return -1, hsa_system_get_info, HSA_SYSTEM_INFO_TIMESTAMP, &timestamp);
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

    ROCM_CALL(return -1, rocprofiler_read, device->context, 0);
    ROCM_CALL(return -1, rocprofiler_get_data, device->context, 0);
    ROCM_CALL(return -1, rocprofiler_get_metrics, device->context);

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
    if ((ret = _rocmon_get_timestamp(&timestamp)))
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
    return bfromcstr("ERROR");
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
                ERROR_PRINT("Failed to find previous instance for event %s", event->name);
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
            ERROR_PRINT("Failed to allocate memory for SMI event in device list %s", event->name);
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
    RSMI_CALL(return -1, rsmi_dev_supported_func_iterator_open, device->deviceId, &iter_handle);

    do
    {
        // Get function information
        //(*rsmi_func_iter_value_get_ptr)(iter_handle, &value);
        RSMI_CALL({
            ERROR_PRINT("Failed to get smi function value for device %d", device->deviceId);
            RSMI_CALL(, rsmi_dev_supported_func_iterator_close, &iter_handle);
            return -1;
        }, rsmi_func_iter_value_get, iter_handle, &value);

        // Get function variants
        ret = _rocmon_smi_get_function_variants(device, value.name, iter_handle);
        if (ret < 0)
        {
            ERROR_PRINT("Failed to get smi function variants for device %d", device->deviceId);
            RSMI_CALL(, rsmi_dev_supported_func_iterator_close, &iter_handle);
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
    RSMI_CALL(, rsmi_dev_supported_func_iterator_close, &iter_handle);

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
            ERROR_PRINT("Failed to allocate memory for SMI event list %s", name);
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
        ERROR_PRINT("Failed to allocate memory for SMI event %s", name);
        return -ENOMEM;
    }

    // Set event properties
    RocmonSmiEvent* event = &list->entries[list->numEntries-1];
    snprintf(event->name, sizeof(event->name), "%s", name);
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
        free(list->entries);
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
        ERROR_PRINT("Failed to create map for ROCm SMI events");
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
        ERROR_PRINT("Number of gpus must be greater than 0 but only %d given", numGpus);
        return -EINVAL;
    }
    
    // Initialize other parts
    init_configuration();

    // initialize libraries
    int ret = _rocmon_link_libraries();
    if (ret < 0)
    {
	ERROR_PRINT("Failed to initialize libraries");
        return ret;
    }

    // Allocate memory for context
    rocmon_context = (RocmonContext*) malloc(sizeof(RocmonContext));
    if (rocmon_context == NULL)
    {
        ERROR_PRINT("Cannot allocate Rocmon context");
        return -ENOMEM;
    }
    rocmon_context->groups = NULL;
    rocmon_context->numGroups = 0;
    rocmon_context->numActiveGroups = 0;

    rocmon_context->devices = (RocmonDevice*) malloc(numGpus * sizeof(RocmonDevice));
    rocmon_context->numDevices = numGpus;
    if (rocmon_context->devices == NULL)
    {
        ERROR_PRINT("Cannot allocate set of GPUs");
        free(rocmon_context);
        rocmon_context = NULL;
        return -ENOMEM;
    }

    // init hsa library
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "Initializing HSA");
    ROCM_CALL(goto rocmon_init_hsa_failed, hsa_init);

    // init rocm smi library
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "Initializing RSMI");
    RSMI_CALL(goto rocmon_init_rsmi_failed, rsmi_init, 0);

    // Get hsa timestamp factor
    uint64_t frequency_hz;
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "Getting HSA timestamp factor");
    ROCM_CALL(ERROR_PRINT("Failed to get HSA timestamp factor"); goto rocmon_init_info_agents_failed, hsa_system_get_info, HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency_hz);
    rocmon_context->hsa_timestamp_factor = (long double)1000000000 / (long double)frequency_hz;

    // initialize structures for specified devices (fetch ROCm specific info)
    iterate_agents_cb_arg arg = {
        .context = rocmon_context,
        .numGpus = numGpus,
        .gpuIds = gpuIds,
    };
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "Iterating through %d available agents", numGpus);
    ROCM_CALL(ERROR_PRINT("Error while iterating through available agents"); goto rocmon_init_info_agents_failed, hsa_iterate_agents, _rocmon_iterate_agents_callback, &arg);

    // Get available SMI events for devices
    _rocmon_smi_init_events();
    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        if (_rocmon_smi_get_functions(&rocmon_context->devices[i]) < 0)
        {
            ERROR_PRINT("Failed to get SMI functions for device %d", rocmon_context->devices[i].deviceId);
            goto rocmon_init_info_agents_failed;
        }
    }

    rocmon_initialized = TRUE;
    return 0;
rocmon_init_info_agents_failed:
    RSMI_CALL(, rsmi_shut_down);
rocmon_init_rsmi_failed:
    ROCM_CALL(, hsa_shut_down);
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
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "Finalize LIKWID ROCMON");

    if (context)
    {
        if (context->devices)
        {
            // Free each devices fields
            for (int i = 0; i < context->numDevices; i++)
            {
                RocmonDevice* device = &context->devices[i];
                free(device->rocMetrics);
                free(device->activeRocEvents);
                free(device->activeSmiEvents);
                if (device->groupResults)
                {
                    // Free events of event result lists
                    for (int j = 0; j < device->numGroupResults; j++)
                    {
                        free(device->groupResults[j].results);
                    }
                    // Free list
                    free(device->groupResults);
                }
                if (device->context)
                {
                    ROCM_CALL(, rocprofiler_close, device->context);
                }
                destroy_smap(device->smiMetrics);
            }

            free(context->devices);
            context->devices = NULL;
        }

        free(context->groups);
        destroy_smap(context->smiEvents);

        free(context);
        context = NULL;
    }

    RSMI_CALL(ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "Shutdown SMI"), rsmi_shut_down);
    ROCM_CALL(ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "Shutdown HSA"), hsa_shut_down);

    rocmon_initialized = FALSE;
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
            ERROR_PRINT("Cannot allocate additional group");
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
            ERROR_PRINT("Cannot allocate event results");
            return -ENOMEM;
        }

        // Allocate memory for new event result list entry
        RocmonEventResultList* tmpGroupResults = (RocmonEventResultList*) realloc(device->groupResults, (device->numGroupResults+1) * sizeof(RocmonEventResultList));
        if (tmpGroupResults == NULL)
        {
            ERROR_PRINT("Cannot allocate new event group result list");
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
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "Closing previous rocprofiler context");
        ROCM_CALL(return -1, rocprofiler_close, device->context);
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
        ERROR_PRINT("Cannot allocate feature list");
        return -ENOMEM;
    }
    for (int i = 0; i < numEvents; i++)
    {
        features[i].kind = ROCPROFILER_FEATURE_KIND_METRIC;
        features[i].name = events[i];
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "SETUP EVENT %d %s", i, events[i]);
    }

    // Free previous feature array if present
    free(device->activeRocEvents);

    device->numActiveRocEvents = numEvents;
    device->activeRocEvents = features;

    // Open context
    rocprofiler_properties_t properties = {};
    properties.queue_depth = 128;
    uint32_t mode = ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_CREATEQUEUE | ROCPROFILER_MODE_SINGLEGROUP;

    // Important: only a single profiling group is supported at this time which limits the number of events that can be monitored at a time.
    ROCM_CALL(return -1, rocprofiler_open, device->hsa_agent, device->activeRocEvents, device->numActiveRocEvents, &device->context, mode, &properties);

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
        ERROR_PRINT("Cannot allocate active event list");
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
                ERROR_PRINT("Instance number in '%s' is too large", event);
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
                ERROR_PRINT("Failed to parse instance number in '%s'", event);
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
            snprintf(eventName, sizeof(eventName), "%s", event);
        }

        // Lookup event in available events
        RocmonSmiEvent* metric = NULL;
        ret = get_smap_by_key(device->smiMetrics, eventName, (void**)&metric);
        if (ret < 0)
        {
            ERROR_PRINT("RSMI event '%s' not found for device %d", eventName, device->deviceId);
            free(activeEvents);
            return -EINVAL;
        }

        // Copy event
        RocmonSmiEvent* tmpEvent = &activeEvents[i];
        memcpy(tmpEvent, metric, sizeof(RocmonSmiEvent));

        // Check if event supports instances
        if (instance >= 0 && tmpEvent->type != ROCMON_SMI_EVENT_TYPE_INSTANCES)
        {
            ERROR_PRINT("Instance number given but event '%s' does not support one", eventName);
            free(activeEvents);
            return -EINVAL;
        }

        // Check if event requires instances
        if (instance < 0 && tmpEvent->type == ROCMON_SMI_EVENT_TYPE_INSTANCES)
        {
            ERROR_PRINT("No instance number given but event '%s' requires one", eventName);
            free(activeEvents);
            return -EINVAL;
        }

        // Check if event has enough instances
        if (instance >= 0 && instance >= metric->instances)
        {
            ERROR_PRINT("Instance %d seleced but event '%s' has only %d", instance, eventName, metric->instances);
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
        ERROR_PRINT("Cannot allocate smiEvent name array");
        return -ENOMEM;
    }
    rocEvents = (const char**) malloc(group->nevents * sizeof(const char*));
    if (rocEvents == NULL)
    {
        ERROR_PRINT("Cannot allocate rocEvent name array");
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
            ERROR_PRINT("Event '%s' has no prefix ('ROCP_' or 'RSMI_')", name);
            return -EINVAL;
        }
    }

    // Add events to each device
    for (int i = 0; i < rocmon_context->numDevices; i++)
    {
        RocmonDevice* device = &rocmon_context->devices[i];

        // Add rocprofiler events
        ROCMON_DEBUG_PRINT(DEBUGLEV_INFO, "SETUP ROCPROFILER WITH %d events", numRocEvents);
        ret = _rocmon_setupCounters_rocprofiler(device, rocEvents, numRocEvents);
        if (ret < 0)
        {
            free(smiEvents);
            free(rocEvents);
            return ret;
        }

        // Add SMI events
        ROCMON_DEBUG_PRINT(DEBUGLEV_INFO, "SETUP ROCM SMI WITH %d events", numSmiEvents);
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
        ROCM_CALL(return -1, rocprofiler_start, device->context, 0);
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
    if ((ret = _rocmon_get_timestamp(&timestamp)))
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
        ROCM_CALL(return -1, rocprofiler_stop, device->context, 0);
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
        ERROR_PRINT("Cannot allocate event list");
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
        ERROR_PRINT("Cannot allocate events for event list");
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
    int i = 0;
    double t = 0;
    if (!rocmon_context || !rocmon_initialized || (groupId < 0) || groupId > rocmon_context->numActiveGroups)
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
    if (!rocmon_context || !rocmon_initialized || (groupId < 0) || groupId > rocmon_context->numActiveGroups)
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
    if (!rocmon_context || !rocmon_initialized || (groupId < 0) || groupId > rocmon_context->numActiveGroups)
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
    return 0;
}

void rocmon_setVerbosity(int level)
{
    if (level >= DEBUGLEV_ONLY_ERROR && level <= DEBUGLEV_DEVELOP)
    {
        likwid_rocmon_verbosity = level;
    }
}


#endif /* LIKWID_WITH_ROCMON */
