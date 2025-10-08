#include <rocmon_types.h>

#include <stdbool.h>
#include <types.h>
#include <util.h>
#include <likwid.h>
#include <pthread.h>
#include <stdio.h>
#include <dlfcn.h>
#include <error.h>

#include <rocm_smi/rocm_smi.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/registration.h>
#include <hip/hip_runtime_api.h>

enum ROCMON_SMI_PCI_EXTRA_ {
    ROCMON_SMI_PCI_EXTRA_SENT,
    ROCMON_SMI_PCI_EXTRA_RECEIVED,
    ROCMON_SMI_PCI_EXTRA_MAXPKTSZ,
};

enum ROCMON_SMI_ECC_EXTRA_ {
    ROCMON_SMI_ECC_EXTRA_CORR,
    ROCMON_SMI_ECC_EXTRA_UNCORR,
};

static const char *ROCPROFILER_EVENT_PREFIX = "ROCP";
static const char *ROCM_SMI_EVENT_PREFIX = "RSMI";

// TODO perhaps rename event to metric?

// TODO clean this up and sort the variables
__attribute__((visibility("default")))
int likwid_rocmon_verbosity = DEBUGLEV_ONLY_ERROR;

DECLARE_STATIC_PTMUTEX(rocmon_init_mutex);
static RocmonContext *rocmon_ctx;
static size_t rocmon_ctx_ref_count = 0;

static void *lib_amdhip;
static void *lib_rocm_smi;
static void *lib_rocprofiler_sdk;

// ROCm function declarations
#define RPR_CALL(handleerror, func, ...) \
    do { \
        assert(func##_ptr != NULL); \
        rocprofiler_status_t s_ = func##_ptr(__VA_ARGS__);\
        if (s_ != ROCPROFILER_STATUS_SUCCESS) {           \
            const char *errstr_ = rocprofiler_get_status_string_ptr(s_); \
            ERROR_PRINT("Error: function %s failed: '%s' (rocprofiler_status_t=%d)", #func, errstr_, s_);\
            handleerror;\
        }\
    } while (0)

#define RSMI_CALL(handleerror, func, ...) \
    do {\
        assert(func##_ptr != NULL); \
        rsmi_status_t s_ = func##_ptr(__VA_ARGS__);\
        if (s_ != RSMI_STATUS_SUCCESS) {\
            const char *errstr_ = NULL;\
            rsmi_status_string_ptr(s_, &errstr_);\
            ERROR_PRINT("Error: function %s failed: '%s' (rsmi_status_t=%d)", #func, errstr_, s_);\
            handleerror;\
        }\
    } while (0)

#define HIP_CALL(handlerror, func, ...) \
    do {\
        assert(func##_ptr != NULL); \
        hipError_t s_ = func##_ptr(__VA_ARGS__);\
        if (s_ != hipSuccess) {\
            const char *errstr_ = hipGetErrorName_ptr(s_);\
            ERROR_PRINT("Error: function %s failed: '%s' (hipError_t=%d)", #func, errstr_, s_);\
            handlerror;\
        }\
    } while (0)

#define DECLAREFUNC_SMI(funcname, ...) static rsmi_status_t (*funcname##_ptr)(__VA_ARGS__)
#define DECLAREFUNC_RPR(funcname, ...) static rocprofiler_status_t (*funcname##_ptr)(__VA_ARGS__)
#define DECLAREFUNC_HIP(funcname, ...) static hipError_t (*funcname##_ptr)(__VA_ARGS__)

DECLAREFUNC_SMI(rsmi_init, uint64_t flags);
DECLAREFUNC_SMI(rsmi_shut_down, void);
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
DECLAREFUNC_SMI(rsmi_num_monitor_devices, uint32_t *num_devices);
DECLAREFUNC_SMI(rsmi_dev_pci_id_get, uint32_t dv_ind, uint64_t *bdfid);

DECLAREFUNC_RPR(rocprofiler_create_counter_config, rocprofiler_agent_id_t, rocprofiler_counter_id_t *, size_t, rocprofiler_counter_config_id_t *);
DECLAREFUNC_RPR(rocprofiler_query_record_counter_id, rocprofiler_counter_instance_id_t, rocprofiler_counter_id_t *);
DECLAREFUNC_RPR(rocprofiler_query_counter_info, rocprofiler_counter_id_t, rocprofiler_counter_info_version_id_t, void *);
DECLAREFUNC_RPR(rocprofiler_create_context, rocprofiler_context_id_t *);
DECLAREFUNC_RPR(rocprofiler_query_available_agents, rocprofiler_agent_version_t, rocprofiler_query_available_agents_cb_t, size_t, void *);
DECLAREFUNC_RPR(rocprofiler_iterate_agent_supported_counters, rocprofiler_agent_id_t, rocprofiler_available_counters_cb_t, void *);
DECLAREFUNC_RPR(rocprofiler_create_buffer, rocprofiler_context_id_t, size_t, size_t, rocprofiler_buffer_policy_t, rocprofiler_buffer_tracing_cb_t, void *, rocprofiler_buffer_id_t *);
DECLAREFUNC_RPR(rocprofiler_create_callback_thread, rocprofiler_callback_thread_t *);
DECLAREFUNC_RPR(rocprofiler_assign_callback_thread, rocprofiler_buffer_id_t, rocprofiler_callback_thread_t);
DECLAREFUNC_RPR(rocprofiler_configure_buffer_dispatch_counting_service, rocprofiler_context_id_t, rocprofiler_buffer_id_t, rocprofiler_dispatch_counting_service_cb_t, void *);
DECLAREFUNC_RPR(rocprofiler_configure_device_counting_service, rocprofiler_context_id_t, rocprofiler_buffer_id_t, rocprofiler_agent_id_t, rocprofiler_device_counting_service_cb_t, void *);
DECLAREFUNC_RPR(rocprofiler_start_context, rocprofiler_context_id_t);
DECLAREFUNC_RPR(rocprofiler_force_configure, rocprofiler_configure_func_t);
static const char *(*rocprofiler_get_status_string_ptr)(rocprofiler_status_t);

DECLAREFUNC_HIP(hipGetDeviceProperties, hipDeviceProp_t *, int);
DECLAREFUNC_HIP(hipGetDeviceCount, int *);
static const char *(*hipGetErrorName_ptr)(hipError_t);

static int tool_init(rocprofiler_client_finalize_t, void *) {
    assert(rocmon_ctx != NULL);

    RPR_CALL(return -EIO, rocprofiler_create_context, &rocmon_ctx->rocprof_ctx);

    return 0;
}

static void tool_fini(void *)
{
}

static RocmonDevice *rocmon_device_get(int gpuIdx) {
    assert(rocmon_ctx != NULL);

    for (size_t i = 0; i < rocmon_ctx->numDevices; i++) {
        RocmonDevice *deviceCandidate = &rocmon_ctx->devices[i];

        if (deviceCandidate->hipDeviceIdx == gpuIdx)
            return deviceCandidate;
    }

    return NULL;
}

static void rocmon_smi_event_list_free(void *event_list_raw) {
    RocmonSmiEventList *event_list = event_list_raw;

    if (!event_list)
        return;

    free(event_list->entries);
    free(event_list);
}

static rocprofiler_tool_configure_result_t *rocprofiler_configure_private(
        uint32_t version,
        const char *,
        uint32_t,
        rocprofiler_client_id_t *id) {
    id->name = "LIKWID rocmon";

    const uint32_t major = version / 10000;
    const uint32_t minor = (version % 10000) / 100;
    const uint32_t patch = version % 100;

    assert(major == 1);
    assert(minor == 0);
    assert(patch == 0);

    static rocprofiler_tool_configure_result_t cfg = {
        sizeof(cfg),
        &tool_init,
        &tool_fini,
        NULL,
    };

    return &cfg;
}

static void *dlopen_any(const char * const *filenames, size_t num_filenames, int flags) {
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

static int rocmon_libraries_init(void) {
    // helper macro
#define DLSYM_CHK2(dllib, name, symname) \
    do { \
        name##_ptr = dlsym(dllib, #symname); \
        const char *err_ = dlerror(); \
        if (err_) { \
            ERROR_PRINT("Failed to link '%s': %s", #symname, err_); \
            err = -ENXIO; \
            goto ret_err; \
        } \
    } while (0)
#define DLSYM_CHK(dllib, name) DLSYM_CHK2(dllib, name, name)

    // Initialization must only occur a single time
    assert(lib_rocm_smi == NULL);
    assert(lib_rocprofiler_sdk == NULL);
    assert(lib_amdhip == NULL);

    // Load rocprofiler-sdk library
    static const char * const rocprofiler_names[] = {
        "librocprofiler-sdk.so",
        "librocprofiler-sdk.so.1",
        "librocprofiler-sdk.so.1.0.0",
    };

    int err = 0;
    lib_rocprofiler_sdk = dlopen_any(rocprofiler_names, ARRAY_COUNT(rocprofiler_names), RTLD_GLOBAL | RTLD_NOW);
    if (!lib_rocprofiler_sdk) {
        err = -ELIBACC;
        goto ret_err;
    }

    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_create_counter_config);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_query_record_counter_id);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_query_counter_info);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_create_context);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_query_available_agents);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_iterate_agent_supported_counters);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_create_buffer);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_create_callback_thread);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_assign_callback_thread);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_configure_buffer_dispatch_counting_service);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_configure_device_counting_service);
    DLSYM_CHK(lib_rocprofiler_sdk, rocprofiler_start_context);
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

    DLSYM_CHK2(lib_amdhip, hipGetDeviceProperties, hipGetDevicePropertiesR0600);
    DLSYM_CHK(lib_amdhip, hipGetDeviceCount);
    DLSYM_CHK(lib_amdhip, hipGetErrorName);

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

static void rocmon_libraries_fini(void) {
    dlclose(lib_rocprofiler_sdk);
    lib_rocprofiler_sdk = NULL;

    dlclose(lib_rocm_smi);
    lib_rocm_smi = NULL;
}

void rocmon_setVerbosity(int level) {
    if (level < DEBUGLEV_ONLY_ERROR)
        level = DEBUGLEV_ONLY_ERROR;
    else if (level > DEBUGLEV_DEVELOP)
        level = DEBUGLEV_DEVELOP;

    likwid_rocmon_verbosity = level;
}

static void format_smi_event_label(char *buf, size_t size, RocmonSmiEventType type, const char *function, uint64_t variant, uint64_t subvariant) {
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

static int metrics_smi_event_add_impl(const char *name, RocmonSmiEventType type, const char *function, uint64_t variant, uint64_t subvariant, uint64_t extra, RocmonSmiMeasureFunc measureFunc) {
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
    list->entries = newEntries;

    snprintf(newEvent->name, sizeof(newEvent->name), "%s", name);
    newEvent->type = type;
    newEvent->variant = variant;
    newEvent->subvariant = subvariant;
    newEvent->extra = extra; // 'extra' is used to differentiate for multiple events, which use the same RSMI function.
    newEvent->measureFunc = measureFunc;

    return 0;
}

static int metrics_smi_event_add_avail(RocmonDevice *device, RocmonSmiEventType type, const char *function, uint64_t variant, uint64_t subvariant) {
    char label[256];
    format_smi_event_label(label, sizeof(label), type, function, variant, subvariant);

    RocmonSmiEventList *list = NULL;
    int err = get_smap_by_key(rocmon_ctx->implementedSmiEvents, label, (void **)&list);
    if (err < 0) {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "ROCM-SMI supports event '%s', but we don't implement it", label);
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
            snprintf(availEvent->name, sizeof(availEvent->name), "%s[%zu]", implEvent->name, subvariant);
            availEvent->subvariant = subvariant;
        } else {
            assert(subvariant == implEvent->subvariant);
            snprintf(availEvent->name, sizeof(availEvent->name), "%s", implEvent->name);
            availEvent->subvariant = implEvent->subvariant;
        }

        availEvent->type = implEvent->type;
        availEvent->variant = implEvent->variant;
        availEvent->extra = implEvent->extra;
        availEvent->measureFunc = implEvent->measureFunc;

        err = add_smap(device->availableSmiEvents, availEvent->name, availEvent);
        if (err < 0)
            return err;
    }

    return 0;
}

static int metrics_smi_init_subvariant(RocmonDevice *device, rsmi_func_id_iter_handle_t variant_iter_handle, const char *function, uint64_t variant) {
    // Iterate over all sub variants begin
    rsmi_func_id_iter_handle_t subvariant_iter_handle;
    rsmi_status_t rerr = rsmi_dev_supported_variant_iterator_open_ptr(variant_iter_handle, &subvariant_iter_handle);

    if (rerr == RSMI_STATUS_NO_DATA) {
        // No subvariants for given function
        return metrics_smi_event_add_avail(device, ROCMON_SMI_EVENT_TYPE_VARIANT, function, variant, 0);
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
        RSMI_CALL(
                err = -EIO; break,
                rsmi_func_iter_value_get,
                subvariant_iter_handle,
                &subvariant_value
        );

        RocmonSmiEventType type = (variant == RSMI_DEFAULT_VARIANT) ?
            ROCMON_SMI_EVENT_TYPE_INSTANCES : ROCMON_SMI_EVENT_TYPE_SUBVARIANT;
        int err = metrics_smi_event_add_avail(device, type, function, variant, subvariant_value.id);
        if (err < 0)
            return err;

        if (rsmi_func_iter_next_ptr(subvariant_iter_handle) == RSMI_STATUS_NO_DATA)
            break;
    }

    // Iterate over all sub variants end
    RSMI_CALL(abort(), rsmi_dev_supported_func_iterator_close, &subvariant_iter_handle);
    return err;
}

static int metrics_smi_init_variant(RocmonDevice *device, rsmi_func_id_iter_handle_t function_iter_handle, const char *function) {
    // Iterate over all variants begin
    rsmi_func_id_iter_handle_t variant_iter_handle;
    rsmi_status_t rerr = rsmi_dev_supported_variant_iterator_open_ptr(function_iter_handle, &variant_iter_handle);

    if (rerr == RSMI_STATUS_NO_DATA) {
        // No variants for given function
        return metrics_smi_event_add_avail(device, ROCMON_SMI_EVENT_TYPE_NORMAL, function, 0, 0);
    } else if (rerr != RSMI_STATUS_SUCCESS) {
        const char *errstr = NULL;
        rsmi_status_string_ptr(rerr, &errstr);
        ERROR_PRINT("rsmi_dev_supported_variant_iterator_open failed: %s", errstr);
        return -EIO;
    }

    RSMI_CALL(
            return -EIO,
            rsmi_dev_supported_variant_iterator_open,
            function_iter_handle,
            &variant_iter_handle
    );

    // Iterate over all variants body
    int err = 0;
    while (true) {
        // Get variant value
        rsmi_func_id_value_t variant_value;
        RSMI_CALL(
                err = -EIO; break,
                rsmi_func_iter_value_get,
                variant_iter_handle,
                &variant_value
        );

        err = metrics_smi_init_subvariant(device, variant_iter_handle, function, variant_value.id);
        if (err < 0)
            break;

        if (rsmi_func_iter_next_ptr(variant_iter_handle) == RSMI_STATUS_NO_DATA)
            break;
    }

    // Iterate over all variants end
    RSMI_CALL(abort(), rsmi_dev_supported_func_iterator_close, &variant_iter_handle);
    return err;
}

static int metrics_smi_init_normal(RocmonDevice *device) {
    int err = init_map(&device->availableSmiEvents, MAP_KEY_TYPE_STR, 0, free);
    if (err < 0)
        return err;

    // For explanations what "normal", "variant", and "subvariants" are,
    // please consult documentation of rocm_smi.

    // Iterate over all functions begin
    rsmi_func_id_iter_handle_t function_iter_handle;
    RSMI_CALL(return -EIO, rsmi_dev_supported_func_iterator_open, device->rsmiDeviceId, &function_iter_handle);

    // Iterate over all functions body
    while (true) {
        // Get function value
        rsmi_func_id_value_t function_value;
        RSMI_CALL(
                err = -EIO; break,
                rsmi_func_iter_value_get,
                function_iter_handle,
                &function_value
        );

        err = metrics_smi_init_variant(device, function_iter_handle, function_value.name);
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
    return metrics_smi_event_add_avail(device, ROCMON_SMI_EVENT_TYPE_NORMAL, "rsmi_compute_process_info_get", 0, 0);
}

static rocprofiler_status_t counter_iterate_cb(
        rocprofiler_agent_id_t,
        rocprofiler_counter_id_t *counters,
        size_t num_counters,
        void *userdata) {
    RocmonDevice *device = userdata;

    for (size_t i = 0; i < num_counters; i++) {
        RocmonRprEvent *availEvent = calloc(1, sizeof(*availEvent));
        if (!availEvent)
            return -errno;

        RPR_CALL(
                return -EIO,
                rocprofiler_query_counter_info,
                counters[i],
                ROCPROFILER_COUNTER_INFO_VERSION_1,
                &availEvent->counterInfo
        );

        if (add_smap(device->availableRocprofEvents, availEvent->counterInfo.name, availEvent) < 0)
            return ROCPROFILER_STATUS_ERROR;
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

static int metrics_rpr_init(RocmonDevice *device) {
    int err = init_map(&device->availableRocprofEvents, MAP_KEY_TYPE_STR, 0, free);
    if (err < 0)
        return err;

    // rocprof_ctx must already be initialized from 'tool_init' at this point.
    assert(rocmon_ctx->rocprof_ctx.handle != 0);

    RPR_CALL(
            return -EIO,
            rocprofiler_iterate_agent_supported_counters,
            device->rocprofAgent->id,
            counter_iterate_cb,
            device
    );
}

static int parse_hex(char c) {
    if (c >= '0' && c <= '9')
        return (uint8_t)(c - '0');
    if (c >= 'a' && c <= 'f')
        return (uint8_t)(c - 'a' + 10);
    if (c >= 'A' && c <= 'F')
        return (uint8_t)(c - 'A' + 10);
    return 0;
}

static bool hip_uuid_equal_rocprof_uuid(const hipUUID *hip_uuid, const rocprofiler_uuid_t *rocp_uuid) {
    assert(sizeof(hip_uuid->bytes) == 16);
    assert(sizeof(rocp_uuid->bytes) == 16);

    // For some reason hipUUID is stored in ASCII, while rocprofiler_uuid_t is stored in binary.
    for (size_t h_i = 0, r_i = 7; h_i < sizeof(*hip_uuid); h_i += 2, r_i--) {
        const char h_a = (char)hip_uuid->bytes[h_i];
        const char h_b = (char)hip_uuid->bytes[h_i+1];
        const int h_digit = 
            (parse_hex(h_a) << 4) |
            parse_hex(h_b);
        const uint8_t r_digit = rocp_uuid->bytes[r_i];

        if (h_digit != r_digit)
            return false;
    }
    return true;
}

static rocprofiler_status_t find_agent_for_hip_device(
        rocprofiler_agent_version_t agents_ver,
        const void **agents_arr_raw,
        size_t num_agents,
        void *userdata) {
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

        if (hip_uuid_equal_rocprof_uuid(&device->hipProps.uuid, &agent_candidate->uuid)) {
            device->rocprofAgent = agent_candidate;
            break;
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

int rpr_agent_init(RocmonDevice *device) {
    // First we have to find which rocprofiler agent belongs to which hip device ID.
    HIP_CALL(return -EIO, hipGetDeviceProperties, &device->hipProps, device->hipDeviceIdx);

    // TODO, this call is wrong either like this or that:
    // - We must call it in tool_init
    // - We must only call it once
    RPR_CALL(
        return -EIO,
        rocprofiler_query_available_agents,
        ROCPROFILER_AGENT_INFO_VERSION_0,
        find_agent_for_hip_device,
        sizeof(rocprofiler_agent_t),
        device
    );

    // If the callback didn't match any available agent to our hip device we fail.
    if (!device->rocprofAgent)
        return -ENODEV;

    return 0;
}

int smi_device_init(RocmonDevice *device) {
    /* I can't find specifications whether rocm_smi device IDs are the same as HIP device IDs.
     * So instead we look through all of them and find the one with matching PCI IDs.
     * Unfortunately rocm_smi doesn't expose the UUID, so we can't use that.
     * One thing I am not certin about is if AMD has something similar as Nvidia MIG devices,
     * which may or may not have the same PCI ID? Let's ignore this for now and hope it doesn't
     * cause issues. */
    uint32_t num_rsmi_devices;
    RSMI_CALL(return -EIO, rsmi_num_monitor_devices, &num_rsmi_devices);

    bool found = false;

    for (uint32_t i = 0; i < num_rsmi_devices; i++) {
        uint64_t candidate_bdfid;
        RSMI_CALL(return -EIO, rsmi_dev_pci_id_get, i, &candidate_bdfid);

        /* For details about the format of bdfid, check rocm_smi.h. As far as I can tell
         * there are no helper macros available to do this more nicely. */
        const uint32_t candidate_domain = (uint32_t)((candidate_bdfid >> 32) & 0xFFFFFFFF);
        const uint32_t candidate_bus = (uint32_t)((candidate_bdfid >> 8) & 0xFF);
        const uint32_t candidate_dev = (uint32_t)((candidate_bdfid >> 3) & 0x1F);

        /* Ignore partition ID and function ID, since they are not in the hipProps,
         * so we can't match them either way. */

        if ((uint32_t)device->hipProps.pciDomainID == candidate_domain &&
            (uint32_t)device->hipProps.pciBusID == candidate_bus &&
            (uint32_t)device->hipProps.pciDeviceID == candidate_dev)
        {
            if (found) {
                ERROR_PRINT("Internal bug: Found duplicate rocm_smi devices");
                break;
            }

            device->rsmiDeviceId = i;
            found = true;
        }
    }

    if (!found)
        return -ENODEV;
    return 0;
}

int rocmon_init_device(int ctx_dev_idx, int hip_dev_idx) {
    if (ctx_dev_idx < 0 || (size_t)ctx_dev_idx >= rocmon_ctx->numDevices)
        return -EINVAL;

    RocmonDevice *device = &rocmon_ctx->devices[ctx_dev_idx];

    device->hipDeviceIdx = hip_dev_idx;

    int err = rpr_agent_init(device);
    if (err < 0)
        return err;

    err = smi_device_init(device);
    if (err < 0)
        return err;

    // Init SMI metrics events
    err = metrics_smi_init_normal(device);
    if (err < 0)
        return err;

    // Init rocprofiler-sdk events
    err = metrics_rpr_init(device);
    if (err < 0)
        return err;

    // TODO do we need to do anything else?
    return 0;
}

static void rocmon_ctx_free(void) {
    if (!rocmon_ctx)
        return;

    if (rocmon_ctx->devices) {
        for (size_t i = 0; i < rocmon_ctx->numDevices; i++) {
            RocmonDevice *device = &rocmon_ctx->devices[i];

            destroy_smap(device->availableSmiEvents);
            destroy_smap(device->availableRocprofEvents);

            free(device->activeSmiEvents);

            if (device->groupResults) {
                free(device->groupResults->results);
                free(device->groupResults);
            }
        }

        free(rocmon_ctx->devices);
    }

    destroy_smap(rocmon_ctx->implementedSmiEvents);

    free(rocmon_ctx);
    rocmon_ctx = NULL;
}

static int rsmi_measurefunc_pci_throughput_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "rsmi_measurefunc_pci_throughput_get(%d, %lu)", deviceId, event->extra);

    uint64_t sent, received, max_pkt_sz;
    RSMI_CALL(return -EIO, rsmi_dev_pci_throughput_get, deviceId, &sent, &received, &max_pkt_sz);

    uint64_t value;
    if (event->extra == ROCMON_SMI_PCI_EXTRA_SENT)
        value = sent;
    else if (event->extra == ROCMON_SMI_PCI_EXTRA_RECEIVED)
        value = received;
    else if (event->extra == ROCMON_SMI_PCI_EXTRA_MAXPKTSZ)
        value = max_pkt_sz;
    else
        return -EINVAL;

    result->fullValue += value;
    result->lastValue = value;

    return 0;
}


static int rsmi_measurefunc_pci_replay_counter_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    (void)event;

    uint64_t counter;
    RSMI_CALL(return -EIO, rsmi_dev_pci_replay_counter_get, deviceId, &counter);
    result->fullValue += counter;
    result->lastValue = counter;

    return 0;
}


static int rsmi_measurefunc_power_ave_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t power;
    RSMI_CALL(return -EIO, rsmi_dev_power_ave_get, deviceId, event->subvariant, &power);
    result->fullValue += power;
    result->lastValue = power;

    return 0;
}


static int rsmi_measurefunc_memory_total_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t total;
    RSMI_CALL(return -EIO, rsmi_dev_memory_total_get, deviceId, event->variant, &total);
    result->fullValue += total;
    result->lastValue = total;

    return 0;
}


static int rsmi_measurefunc_memory_usage_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t used;
    RSMI_CALL(return -EIO, rsmi_dev_memory_usage_get, deviceId, event->variant, &used);
    result->fullValue += used;
    result->lastValue = used;

    return 0;
}


static int rsmi_measurefunc_memory_busy_percent_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    (void)event;

    uint32_t percent;
    RSMI_CALL(return -EIO, rsmi_dev_memory_busy_percent_get, deviceId, &percent);
    result->fullValue += percent;
    result->lastValue = percent;

    return 0;
}


static int rsmi_measurefunc_memory_reserved_pages_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    (void)event;

    uint32_t num_pages;
    RSMI_CALL(return -EIO, rsmi_dev_memory_reserved_pages_get, deviceId, &num_pages, NULL);
    result->fullValue += num_pages;
    result->lastValue = num_pages;

    return 0;
}


static int rsmi_measurefunc_fan_rpms_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t speed;
    RSMI_CALL(return -EIO, rsmi_dev_fan_rpms_get, deviceId, event->subvariant, &speed);
    result->fullValue += speed;
    result->lastValue = speed;

    return 0;
}


static int rsmi_measurefunc_fan_speed_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t speed;
    RSMI_CALL(return -EIO, rsmi_dev_fan_speed_get, deviceId, event->subvariant, &speed);
    result->fullValue += speed;
    result->lastValue = speed;

    return 0;
}

static int rsmi_measurefunc_fan_speed_max_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t max_speed;
    RSMI_CALL(return -EIO, rsmi_dev_fan_speed_max_get, deviceId, event->subvariant, &max_speed);
    result->fullValue += max_speed;
    result->lastValue = max_speed;

    return 0;
}

static int rsmi_measurefunc_temp_metric_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t temperature;
    RSMI_CALL(return -EIO, rsmi_dev_temp_metric_get, deviceId, event->subvariant, event->variant, &temperature);
    result->fullValue += temperature;
    result->lastValue = temperature;

    return 0;
}


static int rsmi_measurefunc_volt_metric_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t voltage;
    RSMI_CALL(return -EIO, rsmi_dev_volt_metric_get, deviceId, event->subvariant, event->variant, &voltage);
    result->fullValue += voltage;
    result->lastValue = voltage;

    return 0;
}

static int rsmi_measurefunc_overdrive_level_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    (void)event;

    uint32_t overdrive;
    RSMI_CALL(return -EIO, rsmi_dev_overdrive_level_get, deviceId, &overdrive);
    result->fullValue += overdrive;
    result->lastValue = overdrive;

    return 0;
}

static int rsmi_measurefunc_ecc_count_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    rsmi_error_count_t error_count;
    RSMI_CALL(return -EIO, rsmi_dev_ecc_count_get, deviceId, event->variant, &error_count);

    if (event->extra == ROCMON_SMI_ECC_EXTRA_CORR) {
        result->lastValue = error_count.correctable_err - result->fullValue;
        result->fullValue = error_count.correctable_err;
    } else if (event->extra == ROCMON_SMI_ECC_EXTRA_UNCORR) {
        result->lastValue = error_count.uncorrectable_err - result->fullValue;
        result->fullValue = error_count.uncorrectable_err;
    } else {
        return -EINVAL;
    }

    return 0;
}

static int rsmi_measurefunc_compute_process_info_get(int deviceId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    (void)deviceId;
    (void)event;

    uint32_t num_items;
    RSMI_CALL(return -1, rsmi_compute_process_info_get, NULL, &num_items);
    result->fullValue += num_items;
    result->lastValue = num_items;

    return 0;
}

#define ADD_SMI_EVENT(name, type, smifunc, variant, subvariant, extra, measurefunc)\
    do {\
        int err_ = metrics_smi_event_add_impl(name, type, smifunc, variant, subvariant, extra, measurefunc);\
        if (err_ < 0) \
            return err;\
    } while (0)

#define ADD_SMI_EVENT_N(name, smifunc, extra, measurefunc) \
    ADD_SMI_EVENT(name, ROCMON_SMI_EVENT_TYPE_NORMAL, smifunc, 0, 0, extra, measurefunc)
#define ADD_SMI_EVENT_V(name, smifunc, variant, extra, measurefunc) \
    ADD_SMI_EVENT(name, ROCMON_SMI_EVENT_TYPE_VARIANT, smifunc, variant, 0, extra, measurefunc)
#define ADD_SMI_EVENT_S(name, smifunc, variant, subvariant, extra, measurefunc) \
    ADD_SMI_EVENT(name, ROCMON_SMI_EVENT_TYPE_SUBVARIANT, smifunc, variant, subvariant, extra, measurefunc)
#define ADD_SMI_EVENT_I(name, smifunc, extra, measurefunc) \
    ADD_SMI_EVENT(name, ROCMON_SMI_EVENT_TYPE_INSTANCES, smifunc, RSMI_DEFAULT_VARIANT, 0, extra, measurefunc)

static int rocmon_smi_init_implemented(void) {
    int err = init_map(&rocmon_ctx->implementedSmiEvents, MAP_KEY_TYPE_STR, 0, rocmon_smi_event_list_free);
    if (err < 0)
        return 0;

    ADD_SMI_EVENT_N("PCI_THROUGHPUT_SENT",                  "rsmi_dev_pci_throughput_get", ROCMON_SMI_PCI_EXTRA_SENT,                           &rsmi_measurefunc_pci_throughput_get        );
    ADD_SMI_EVENT_N("PCI_THROUGHPUT_RECEIVED",              "rsmi_dev_pci_throughput_get", ROCMON_SMI_PCI_EXTRA_RECEIVED,                       &rsmi_measurefunc_pci_throughput_get        );
    ADD_SMI_EVENT_N("PCI_THROUGHPUT_MAX_PKT_SZ",            "rsmi_dev_pci_throughput_get", ROCMON_SMI_PCI_EXTRA_MAXPKTSZ,                       &rsmi_measurefunc_pci_throughput_get        );
    ADD_SMI_EVENT_N("PCI_REPLAY_COUNTER",                   "rsmi_dev_pci_replay_counter_get", 0,                                               &rsmi_measurefunc_pci_replay_counter_get    );
    ADD_SMI_EVENT_I("POWER_AVE",                            "rsmi_dev_power_ave_get", 0,                                                        &rsmi_measurefunc_power_ave_get             );
    ADD_SMI_EVENT_V("MEMORY_TOTAL_VRAM",                    "rsmi_dev_memory_total_get", RSMI_MEM_TYPE_VRAM, 0,                                 &rsmi_measurefunc_memory_total_get          );
    ADD_SMI_EVENT_V("MEMORY_TOTAL_VIS_VRAM",                "rsmi_dev_memory_total_get", RSMI_MEM_TYPE_VIS_VRAM, 0,                             &rsmi_measurefunc_memory_total_get          );
    ADD_SMI_EVENT_V("MEMORY_TOTAL_GTT",                     "rsmi_dev_memory_total_get", RSMI_MEM_TYPE_GTT, 0,                                  &rsmi_measurefunc_memory_total_get          );
    ADD_SMI_EVENT_V("MEMORY_USAGE_VRAM",                    "rsmi_dev_memory_usage_get", RSMI_MEM_TYPE_VRAM, 0,                                 &rsmi_measurefunc_memory_usage_get          );
    ADD_SMI_EVENT_V("MEMORY_USAGE_VIS_VRAM",                "rsmi_dev_memory_usage_get", RSMI_MEM_TYPE_VIS_VRAM, 0,                             &rsmi_measurefunc_memory_usage_get          );
    ADD_SMI_EVENT_V("MEMORY_USAGE_GTT",                     "rsmi_dev_memory_usage_get", RSMI_MEM_TYPE_GTT, 0,                                  &rsmi_measurefunc_memory_usage_get          );
    ADD_SMI_EVENT_N("MEMORY_BUSY_PERCENT",                  "rsmi_dev_memory_busy_percent_get", 0,                                              &rsmi_measurefunc_memory_busy_percent_get   );
    ADD_SMI_EVENT_N("MEMORY_NUM_RESERVED_PAGES",            "rsmi_dev_memory_reserved_pages_get", 0,                                            &rsmi_measurefunc_memory_reserved_pages_get );
    ADD_SMI_EVENT_I("FAN_RPMS",                             "rsmi_dev_fan_rpms_get", 0,                                                         &rsmi_measurefunc_fan_rpms_get              );
    ADD_SMI_EVENT_I("FAN_SPEED",                            "rsmi_dev_fan_speed_get", 0,                                                        &rsmi_measurefunc_fan_speed_get             );
    ADD_SMI_EVENT_I("FAN_SPEED_MAX",                        "rsmi_dev_fan_speed_max_get", 0,                                                    &rsmi_measurefunc_fan_speed_max_get         );
    ADD_SMI_EVENT_S("TEMP_EDGE",                            "rsmi_dev_temp_metric_get", RSMI_TEMP_CURRENT, RSMI_TEMP_TYPE_EDGE, 0,              &rsmi_measurefunc_temp_metric_get           );
    ADD_SMI_EVENT_S("TEMP_JUNCTION",                        "rsmi_dev_temp_metric_get", RSMI_TEMP_CURRENT, RSMI_TEMP_TYPE_JUNCTION, 0,          &rsmi_measurefunc_temp_metric_get           );
    ADD_SMI_EVENT_S("TEMP_MEMORY",                          "rsmi_dev_temp_metric_get", RSMI_TEMP_CURRENT, RSMI_TEMP_TYPE_MEMORY, 0,            &rsmi_measurefunc_temp_metric_get           );
    ADD_SMI_EVENT_S("VOLT_VDDGFX",                          "rsmi_dev_volt_metric_get", RSMI_VOLT_CURRENT, RSMI_VOLT_TYPE_VDDGFX, 0,            &rsmi_measurefunc_volt_metric_get           );
    ADD_SMI_EVENT_N("OVERDRIVE_LEVEL",                      "rsmi_dev_overdrive_level_get", 0,                                                  &rsmi_measurefunc_overdrive_level_get       );
    ADD_SMI_EVENT_V("ECC_COUNT_UMC_CORRECTABLE",            "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_UMC, ROCMON_SMI_ECC_EXTRA_CORR,            &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_UMC_UNCORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_UMC, ROCMON_SMI_ECC_EXTRA_UNCORR,          &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_SDMA_CORRECTABLE",           "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_SDMA, ROCMON_SMI_ECC_EXTRA_CORR,           &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_SDMA_UNCORRECTABLE",         "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_SDMA, ROCMON_SMI_ECC_EXTRA_UNCORR,         &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_GFX_CORRECTABLE",            "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_GFX, ROCMON_SMI_ECC_EXTRA_CORR,            &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_GFX_UNCORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_GFX, ROCMON_SMI_ECC_EXTRA_UNCORR,          &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_MMHUB_CORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_MMHUB, ROCMON_SMI_ECC_EXTRA_CORR,          &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_MMHUB_UNCORRECTABLE",        "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_MMHUB, ROCMON_SMI_ECC_EXTRA_UNCORR,        &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_ATHUB_CORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_ATHUB, ROCMON_SMI_ECC_EXTRA_CORR,          &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_ATHUB_UNCORRECTABLE",        "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_ATHUB, ROCMON_SMI_ECC_EXTRA_UNCORR,        &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_PCIE_BIF_CORRECTABLE",       "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_PCIE_BIF, ROCMON_SMI_ECC_EXTRA_CORR,       &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_PCIE_BIF_UNCORRECTABLE",     "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_PCIE_BIF, ROCMON_SMI_ECC_EXTRA_UNCORR,     &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_HDP_CORRECTABLE",            "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_HDP, ROCMON_SMI_ECC_EXTRA_CORR,            &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_HDP_UNCORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_HDP, ROCMON_SMI_ECC_EXTRA_UNCORR,          &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_XGMI_WAFL_CORRECTABLE",      "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_XGMI_WAFL, ROCMON_SMI_ECC_EXTRA_CORR,      &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_XGMI_WAFL_UNCORRECTABLE",    "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_XGMI_WAFL, ROCMON_SMI_ECC_EXTRA_UNCORR,    &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_DF_CORRECTABLE",             "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_DF, ROCMON_SMI_ECC_EXTRA_CORR,             &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_DF_UNCORRECTABLE",           "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_DF, ROCMON_SMI_ECC_EXTRA_UNCORR,           &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_SMN_CORRECTABLE",            "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_SMN, ROCMON_SMI_ECC_EXTRA_CORR,            &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_SMN_UNCORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_SMN, ROCMON_SMI_ECC_EXTRA_UNCORR,          &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_SEM_CORRECTABLE",            "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_SEM, ROCMON_SMI_ECC_EXTRA_CORR,            &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_SEM_UNCORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_SEM, ROCMON_SMI_ECC_EXTRA_UNCORR,          &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_MP0_CORRECTABLE",            "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_MP0, ROCMON_SMI_ECC_EXTRA_CORR,            &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_MP0_UNCORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_MP0, ROCMON_SMI_ECC_EXTRA_UNCORR,          &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_MP1_CORRECTABLE",            "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_MP1, ROCMON_SMI_ECC_EXTRA_CORR,            &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_MP1_UNCORRECTABLE",          "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_MP1, ROCMON_SMI_ECC_EXTRA_UNCORR,          &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_FUSE_CORRECTABLE",           "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_FUSE, ROCMON_SMI_ECC_EXTRA_CORR,           &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_FUSE_UNCORRECTABLE",         "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_FUSE, ROCMON_SMI_ECC_EXTRA_UNCORR,         &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_LAST_CORRECTABLE",           "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_LAST, ROCMON_SMI_ECC_EXTRA_CORR,           &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_V("ECC_COUNT_LAST_UNCORRECTABLE",         "rsmi_dev_ecc_count_get", RSMI_GPU_BLOCK_LAST, ROCMON_SMI_ECC_EXTRA_UNCORR,         &rsmi_measurefunc_ecc_count_get             );
    ADD_SMI_EVENT_N("PROCS_USING_GPU",                      "rsmi_compute_process_info_get", 0,                                                 &rsmi_measurefunc_compute_process_info_get  );
}

int rocmon_init(int numGpus, const int *gpuIds) {
    if (numGpus < 0 || !gpuIds)
        return -EINVAL;

    pthread_mutex_lock(&rocmon_init_mutex);

    if (rocmon_ctx_ref_count++ > 0)
        goto unlock_ok;

    bool rsmi_initialized = false;
    bool libs_initialized = false;

    int err = rocmon_libraries_init();
    if (err < 0)
        goto unlock_err;

    libs_initialized = true;

    rocmon_ctx = calloc(1, sizeof(*rocmon_ctx));
    if (!rocmon_ctx) {
        err = -errno;
        goto unlock_err;
    }

    rocmon_ctx->devices = calloc(numGpus, sizeof(*rocmon_ctx->devices));
    if (!rocmon_ctx->devices) {
        err = -errno;
        goto unlock_err;
    }

    rocmon_ctx->numDevices = numGpus;

    RSMI_CALL(err = -EIO; goto unlock_err, rsmi_init, 0);
    rsmi_initialized = true;

    RPR_CALL(err = -EIO; goto unlock_err, rocprofiler_force_configure, rocprofiler_configure_private);

    // I don't know exactly why, but some initialization of the HIP runtime is required
    // before we can do anything meaningful with rocprofiler-sdk. Otherwise we get
    // an error ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED. Frankly, hsa_init and hsa_shut_down
    // don't appear to do the trick. So it is important that this comes before any other
    // rocprofiler calls.
    int deviceCount;
    HIP_CALL(err = -EIO; goto unlock_err, hipGetDeviceCount, &deviceCount);

    if (deviceCount > numGpus) {
        err = -EINVAL;
        goto unlock_err;
    }

    err = rocmon_smi_init_implemented();
    if (err < 0)
        goto unlock_err;

    for (int i = 0; i < deviceCount; i++) {
        err = rocmon_init_device(i, gpuIds[i]);
        if (err < 0)
            goto unlock_err;
    }

    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "rocmon initalization done");

unlock_ok:
    pthread_mutex_unlock(&rocmon_init_mutex);
    return 0;

unlock_err:
    if (rsmi_initialized)
        (void)rsmi_shut_down_ptr();

    if (libs_initialized)
        rocmon_libraries_fini();

    rocmon_ctx_free();

    pthread_mutex_unlock(&rocmon_init_mutex);
    return err;
}

void rocmon_finalize(void) {
    pthread_mutex_lock(&rocmon_init_mutex);

    // init must occur before finalize
    assert(rocmon_ctx_ref_count != 0);

    if (--rocmon_ctx_ref_count > 0)
        goto unlock_ret;

    RSMI_CALL(abort(), rsmi_shut_down);
    rocmon_ctx_free();
    rocmon_libraries_fini();

unlock_ret:
    pthread_mutex_unlock(&rocmon_init_mutex);
}

int rocmon_addEventSet(const char *eventString, int *gid) {
}

int rocmon_switchActiveGroup(int newGroupId) {
}

int rocmon_setupCounters(int gid) {
}

int rocmon_startCounters(void) {
}

int rocmon_stopCounters(void) {
}

int rocmon_readCounters(void) {
}

double rocmon_getResult(int gpuIdx, int groupId, int eventId) {
}

double rocmon_getLastResult(int gpuIdx, int groupId, int eventId) {
}

int rocmon_getEventsOfGpu(int gpuIdx, RocmonEventList_t *list) {
    RocmonDevice *device = rocmon_device_get(gpuIdx);
    if (!device)
        return -EINVAL;

    // Allocate new event list
    RocmonEventList *newList = calloc(1, sizeof(*newList));
    if (!newList)
        return -errno;

    int numEventsSmi = get_map_size(device->availableSmiEvents);
    assert(numEventsSmi >= 0);

    int numEventsRocprof = get_map_size(device->availableRocprofEvents);
    assert(numEventsRocprof >= 0);

    int err = 0;

    const size_t newNumEvents = (size_t)numEventsSmi + (size_t)numEventsRocprof;
    newList->events = calloc(newNumEvents, sizeof(*newList->events));
    if (!newList->events) {
        err = -errno;
        goto reterr;
    }

    newList->numEvents = newNumEvents;

    RocmonEventListEntry *newEntry = &newList->events[0];

    // Create ROCM SMI event entries
    for (int i = 0; i < numEventsSmi; i++) {
        RocmonSmiEvent *event;
        err = get_smap_by_idx(device->availableSmiEvents, i, (void **)&event);
        if (err < 0)
            goto reterr;

        // e.g. RSMI + FAN_SPEED -> RSMI_FAN_SPEED
        newEntry->name = xasprintf("%s_%s", ROCM_SMI_EVENT_PREFIX, event->name);
        if (!newEntry->name) {
            err = -errno;
            goto reterr;
        }
        newEntry->desc = strdup("ROCM SMI event: <no description>");
        if (!newEntry->desc) {
            err = -errno;
            goto reterr;
        }

        newEntry++;
    }

    // Create ROCPROFILER-SDK event entries
    for (int i = 0; i < numEventsRocprof; i++) {
        RocmonRprEvent *event;
        err = get_smap_by_idx(device->availableRocprofEvents, i, (void **)&event);
        if (err < 0)
            goto reterr;

        // e.g. ROCP + GPU_UTIL -> ROCP_GPU_UTIL
        newEntry->name = xasprintf("%s_%s", ROCPROFILER_EVENT_PREFIX, event->counterInfo.name);
        if (!newEntry->name) {
            err = -errno;
            goto reterr;
        }
        newEntry->desc = xasprintf("ROCPROFILER-SDK event: %s", event->counterInfo.description);
        if (!newEntry->desc) {
            err = -errno;
            goto reterr;
        }

        newEntry++;
    }

    *list = newList;
    return 0;

reterr:
    rocmon_freeEventsOfGpu(newList);

    return err;
}

void rocmon_freeEventsOfGpu(RocmonEventList_t list) {
    if (!list)
        return;

    for (size_t i = 0; i < list->numEvents; i++) {
        free(list->events[i].name);
        free(list->events[i].desc);
    }

    free(list->events);
    free(list);
}

int rocmon_getNumberOfGroups(void) {
}

int rocmon_getIdOfActiveGroup(void) {
}

int rocmon_getNumberOfGPUs(void) {
}

int rocmon_getNumberOfEvents(int groupId) {
}

int rocmon_getNumberOfMetrics(int groupId) {
}

char *rocmon_getEventName(int groupId, int eventId) {
}

char *rocmon_getCounterName(int groupId, int eventId) {
}

char *rocmon_getMetricName(int groupId, int metricId) {
}

double rocmon_getTimeOfGroup(int groupId) {
}

double rocmon_getLastTimeOfGroup(int groupId) {
}

double rocmon_getTimeToLastReadOfGroup(int groupId) {
}

char *rocmon_getGroupName(int groupId) {
}

char *rocmon_getGroupInfoShort(int groupId) {
}

char *rocmon_getGroupInfoLong(int groupId) {
}

int rocmon_getGroups(char ***groups, char ***shortinfos, char ***longinfos) {
}

int rocmon_returnGroups(int nrgroups, char **groups, char **shortinfos, char **longinfos) {
}

void rocmon_markerInit(void) {
}

void rocmon_markerClose(void) {
}

int rocmon_markerRegisterRegion(const char *regionTag) {
}

int rocmon_markerStartRegion(const char *regionTag) {
}

int rocmon_markerStopRegion(const char *regionTag) {
}

int rocmon_markerResetRegion(const char *regionTag) {
}

int rocmon_markerWriteFile(const char *markerfile) {
}

void rocmon_markerNextGroup(void) {
}

int rocmon_readMarkerFile(const char *filename) {
}

void rocmon_destroyMarkerResults(void) {
}

int rocmon_getCountOfRegion(int region, int gpu) {
}

double rocmon_getTimeOfRegion(int region, int gpu) {
}

int rocmon_getGpulistOfRegion(int region, int count, int *gpulist) {
}

int rocmon_getGpusOfRegion(int region) {
}

int rocmon_getMetricsOfRegion(int region) {
}

int rocmon_getNumberOfRegions(void) {
}

int rocmon_getGroupOfRegion(int region) {
}

char *rocmon_getTagOfRegion(int region) {
}

int rocmon_getEventsOfRegion(int region) {
}

double rocmon_getResultOfRegionGpu(int region, int eventId, int gpuId) {
}

double rocmon_getMetricOfRegionGpu(int region, int metricId, int gpuId) {
}
