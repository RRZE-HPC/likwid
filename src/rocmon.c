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

static const char *RPR_EVENT_PREFIX = "ROCP_";
static const char *RSMI_EVENT_PREFIX = "RSMI_";

// unify naming of 'event', 'counter'

// TODO clang format

// TODO rename things to camelCase

// TODO clean this up and sort the variables
__attribute__((visibility("default"))) int likwid_rocmon_verbosity = DEBUGLEV_ONLY_ERROR;

static pthread_mutex_t rocmon_init_mutex = PTHREAD_MUTEX_INITIALIZER;
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
DECLAREFUNC_RPR(rocprofiler_destroy_counter_config, rocprofiler_counter_config_id_t);
DECLAREFUNC_RPR(rocprofiler_query_record_counter_id, rocprofiler_counter_instance_id_t, rocprofiler_counter_id_t *);
DECLAREFUNC_RPR(rocprofiler_query_counter_info, rocprofiler_counter_id_t, rocprofiler_counter_info_version_id_t, void *);
DECLAREFUNC_RPR(rocprofiler_create_context, rocprofiler_context_id_t *);
DECLAREFUNC_RPR(rocprofiler_query_available_agents, rocprofiler_agent_version_t, rocprofiler_query_available_agents_cb_t, size_t, void *);
DECLAREFUNC_RPR(rocprofiler_iterate_agent_supported_counters, rocprofiler_agent_id_t, rocprofiler_available_counters_cb_t, void *);
DECLAREFUNC_RPR(rocprofiler_create_buffer, rocprofiler_context_id_t, size_t, size_t, rocprofiler_buffer_policy_t, rocprofiler_buffer_tracing_cb_t, void *, rocprofiler_buffer_id_t *);
DECLAREFUNC_RPR(rocprofiler_destroy_buffer, rocprofiler_buffer_id_t);
DECLAREFUNC_RPR(rocprofiler_flush_buffer, rocprofiler_buffer_id_t);
DECLAREFUNC_RPR(rocprofiler_create_callback_thread, rocprofiler_callback_thread_t *);
DECLAREFUNC_RPR(rocprofiler_assign_callback_thread, rocprofiler_buffer_id_t, rocprofiler_callback_thread_t);
DECLAREFUNC_RPR(rocprofiler_configure_buffer_dispatch_counting_service, rocprofiler_context_id_t, rocprofiler_buffer_id_t, rocprofiler_dispatch_counting_service_cb_t, void *);
DECLAREFUNC_RPR(rocprofiler_configure_device_counting_service, rocprofiler_context_id_t, rocprofiler_buffer_id_t, rocprofiler_agent_id_t, rocprofiler_device_counting_service_cb_t, void *);
DECLAREFUNC_RPR(rocprofiler_sample_device_counting_service, rocprofiler_context_id_t, rocprofiler_user_data_t, rocprofiler_counter_flag_t, rocprofiler_counter_record_t *, size_t *);
DECLAREFUNC_RPR(rocprofiler_start_context, rocprofiler_context_id_t);
DECLAREFUNC_RPR(rocprofiler_stop_context, rocprofiler_context_id_t);
DECLAREFUNC_RPR(rocprofiler_context_is_active, rocprofiler_context_id_t, int *);
DECLAREFUNC_RPR(rocprofiler_context_is_valid, rocprofiler_context_id_t, int *);
DECLAREFUNC_RPR(rocprofiler_force_configure, rocprofiler_configure_func_t);
static const char *(*rocprofiler_get_status_string_ptr)(rocprofiler_status_t);

DECLAREFUNC_HIP(hipGetDeviceProperties, hipDeviceProp_t *, int);
DECLAREFUNC_HIP(hipGetDeviceCount, int *);
DECLAREFUNC_HIP(hipFree, void *);
DECLAREFUNC_HIP(hipInit, unsigned int);
static const char *(*hipGetErrorName_ptr)(hipError_t);

static int rocmon_device_init(size_t ctxDeviceIdx);

static int tool_init(rocprofiler_client_finalize_t, void *) {
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

static void tool_fini(void *)
{
}

static RocmonDevice *device_get(int hipDeviceId) {
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

static int smi_event_add_impl(const char *name, RocmonSmiEventType type, const char *function, uint64_t variant, uint64_t subvariant, uint64_t extra, RocmonSmiMeasureFunc measureFunc) {
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

static int smi_events_add_avail(RocmonDevice *device, RocmonSmiEventType type, const char *function, uint64_t variant, uint64_t subvariant) {
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

static int smi_init_events_subvariant(RocmonDevice *device, rsmi_func_id_iter_handle_t variant_iter_handle, const char *function, uint64_t variant) {
    // Iterate over all sub variants begin
    rsmi_func_id_iter_handle_t subvariant_iter_handle;
    rsmi_status_t rerr = rsmi_dev_supported_variant_iterator_open_ptr(variant_iter_handle, &subvariant_iter_handle);

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
        RSMI_CALL(
                err = -EIO; break,
                rsmi_func_iter_value_get,
                subvariant_iter_handle,
                &subvariant_value
        );

        RocmonSmiEventType type = (variant == RSMI_DEFAULT_VARIANT) ?
            ROCMON_SMI_EVENT_TYPE_INSTANCES : ROCMON_SMI_EVENT_TYPE_SUBVARIANT;
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

static int smi_init_events_variant(RocmonDevice *device, rsmi_func_id_iter_handle_t function_iter_handle, const char *function) {
    // Iterate over all variants begin
    rsmi_func_id_iter_handle_t variant_iter_handle;
    rsmi_status_t rerr = rsmi_dev_supported_variant_iterator_open_ptr(function_iter_handle, &variant_iter_handle);

    if (rerr == RSMI_STATUS_NO_DATA) {
        // No variants for given function
        return smi_events_add_avail(device, ROCMON_SMI_EVENT_TYPE_NORMAL, function, 0, 0);
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

static int smi_init_events_normal(RocmonDevice *device) {
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
    return smi_events_add_avail(device, ROCMON_SMI_EVENT_TYPE_NORMAL, "rsmi_compute_process_info_get", 0, 0);
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

        if (add_smap(device->availableRprEvents, availEvent->counterInfo.name, availEvent) < 0) {
            free(availEvent);
            return ROCPROFILER_STATUS_ERROR;
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

static int rpr_init_events(RocmonDevice *device) {
    int err = init_map(&device->availableRprEvents, MAP_KEY_TYPE_STR, 0, free);
    if (err < 0)
        return err;

    // rocprofCtx must already be initialized from 'tool_init' at this point.
    assert(rocmon_ctx->rocprofCtx.handle != 0);

    RPR_CALL(
            return -EIO,
            rocprofiler_iterate_agent_supported_counters,
            device->rocprofAgent->id,
            counter_iterate_cb,
            device
    );
    return 0;
}

static rocprofiler_status_t find_agent_for_rocmon_device(
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

        if (agent_candidate->domain == device->pciDomain && agent_candidate->location_id) {
            device->rocprofAgent = agent_candidate;
            break;
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

static void set_counter_callback(
        rocprofiler_context_id_t context_id,
        rocprofiler_agent_id_t agent_id,
        rocprofiler_device_counting_agent_cb_t set_config,
        void *userdata) {
    const RocmonDevice *device = userdata;

    assert(context_id.handle == rocmon_ctx->rocprofCtx.handle);
    assert(agent_id.handle == device->rocprofAgent->id.handle);

    rocprofiler_counter_config_id_t counter_config;

    RPR_CALL(
            return,
            rocprofiler_create_counter_config,
            agent_id,
            device->activeRprEvents,
            device->numActiveRprEvents,
            &counter_config
    );

    rocprofiler_status_t status = set_config(context_id, counter_config);
    if (status != ROCPROFILER_STATUS_SUCCESS)
        ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "rocprofiler-sdk: set_config failed: %s", rocprofiler_get_status_string_ptr(status));

    //RPR_CALL(
    //        return,
    //        rocprofiler_destroy_counter_config,
    //        counter_config
    //);
}

static void buffered_callback(
        rocprofiler_context_id_t,
        rocprofiler_buffer_id_t,
        rocprofiler_record_header_t **headers,
        size_t num_headers,
        void *userdata,
        uint64_t /* drop_count */) {
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
        int err = get_smap_by_key(device->callbackRprResults, key, (void **)&value);
        if (err == -ENOENT) {
            value = malloc(sizeof(*value));
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

        *value = record->counter_value;
    }

    pthread_mutex_unlock(&device->callbackRprMutex);
}

static int rpr_device_init(RocmonDevice *device) {
    /* First we have to find which rocprofiler agent belongs to which RocmonDevice.
     * We do this via the PCI location. */

    RPR_CALL(
        return -EIO,
        rocprofiler_query_available_agents,
        ROCPROFILER_AGENT_INFO_VERSION_0,
        find_agent_for_rocmon_device,
        sizeof(rocprofiler_agent_t),
        device
    );

    // If the callback didn't match any available agent to our hip device we fail.
    if (!device->rocprofAgent)
        return -ENODEV;

    RPR_CALL(
            return -EIO,
            rocprofiler_create_buffer,
            rocmon_ctx->rocprofCtx,
            4096, // TODO ??? how do we choose a proper value?
            2048, // TODO ???
            ROCPROFILER_BUFFER_POLICY_LOSSLESS,
            buffered_callback,
            device,
            &device->rocprofBuf
    );

    RPR_CALL(
            return -EIO,
            rocprofiler_create_callback_thread,
            &device->rocprofThrd
    );

    RPR_CALL(
            return -EIO,
            rocprofiler_assign_callback_thread,
            device->rocprofBuf,
            device->rocprofThrd
    );

    // The set_counter_callback is not called here. It will be called later
    // during rocprofiler_start_context.
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "Using device: %s\n", device->rocprofAgent->product_name);
    RPR_CALL(
            return -EIO,
            rocprofiler_configure_device_counting_service,
            rocmon_ctx->rocprofCtx,
            device->rocprofBuf,
            device->rocprofAgent->id,
            set_counter_callback,
            device
    );

    int err = init_map(&device->callbackRprResults, MAP_KEY_TYPE_STR, 0, free);
    if (err < 0)
        return err;

    pthread_mutex_init(&device->callbackRprMutex, NULL);

    return 0;
}

static int smi_device_init(RocmonDevice *device) {
    uint64_t bdfid;
    RSMI_CALL(return -EIO, rsmi_dev_pci_id_get, device->rsmiDeviceId, &bdfid);

    /* For details about the format of bdfid, check rocm_smi.h. As far as I can tell
     * there are no helper macros available to do this more nicely. */
    device->pciDomain = (uint32_t)(bdfid >> 32);
    device->pciLocation = (uint32_t)(bdfid >> 0);
    return 0;
}

static int rocmon_device_init(size_t ctxDeviceIdx) {
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

static int rocmon_init_hip(size_t numGpuIds, const int *gpuIds) {
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

    rocmon_ctx->hipDeviceIdxToRocmonDeviceIdx = calloc(numGpuIds, sizeof(*rocmon_ctx->hipDeviceIdxToRocmonDeviceIdx));
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
                device->hipDeviceId = gpuId;
                device->hipProps = hipProps;
                device->enabled = true;
                found = true;
                break;
            }
        }

        if (!found) {
            ROCMON_DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Unable to find ROCm SMI / rocprofiler-sdk device for HIP device: %d", gpuId);
            return -ENODEV;
        }
    }

    return 0;
}

static void rocmon_device_fini(RocmonDevice *device) {
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

static void rocmon_groupinfo_fini(RocmonGroupInfo *group) {
    perfgroup_returnGroup(&group->groupInfo);
}

static void rocmon_ctx_free(void) {
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

static int rsmi_measurefunc_pci_throughput_get(uint32_t rsmiDevId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "rsmi_measurefunc_pci_throughput_get(%d, %lu)", rsmiDevId, event->extra);

    uint64_t sent, received, max_pkt_sz;
    RSMI_CALL(return -EIO, rsmi_dev_pci_throughput_get, rsmiDevId, &sent, &received, &max_pkt_sz);

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


static int rsmi_measurefunc_pci_replay_counter_get(uint32_t rsmiDevId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    (void)event;

    uint64_t counter;
    RSMI_CALL(return -EIO, rsmi_dev_pci_replay_counter_get, rsmiDevId, &counter);
    result->fullValue += counter;
    result->lastValue = counter;

    return 0;
}


static int rsmi_measurefunc_power_ave_get(uint32_t rsmiDevId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t power;
    RSMI_CALL(return -EIO, rsmi_dev_power_ave_get, rsmiDevId, event->subvariant, &power);
    result->fullValue += power;
    result->lastValue = power;

    return 0;
}


static int rsmi_measurefunc_memory_total_get(uint32_t rsmiDevId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t total;
    RSMI_CALL(return -EIO, rsmi_dev_memory_total_get, rsmiDevId, event->variant, &total);
    result->fullValue += total;
    result->lastValue = total;

    return 0;
}


static int rsmi_measurefunc_memory_usage_get(uint32_t rsmiDevId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t used;
    RSMI_CALL(return -EIO, rsmi_dev_memory_usage_get, rsmiDevId, event->variant, &used);
    result->fullValue += used;
    result->lastValue = used;

    return 0;
}


static int rsmi_measurefunc_memory_busy_percent_get(uint32_t rsmiDevId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    (void)event;

    uint32_t percent;
    RSMI_CALL(return -EIO, rsmi_dev_memory_busy_percent_get, rsmiDevId, &percent);
    result->fullValue += percent;
    result->lastValue = percent;

    return 0;
}


static int rsmi_measurefunc_memory_reserved_pages_get(uint32_t rsmiDevId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    (void)event;

    uint32_t num_pages;
    RSMI_CALL(return -EIO, rsmi_dev_memory_reserved_pages_get, rsmiDevId, &num_pages, NULL);
    result->fullValue += num_pages;
    result->lastValue = num_pages;

    return 0;
}


static int rsmi_measurefunc_fan_rpms_get(uint32_t rsmiDevId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t speed;
    RSMI_CALL(return -EIO, rsmi_dev_fan_rpms_get, rsmiDevId, event->subvariant, &speed);
    result->fullValue += speed;
    result->lastValue = speed;

    return 0;
}


static int rsmi_measurefunc_fan_speed_get(uint32_t rsmiDevId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t speed;
    RSMI_CALL(return -EIO, rsmi_dev_fan_speed_get, rsmiDevId, event->subvariant, &speed);
    result->fullValue += speed;
    result->lastValue = speed;

    return 0;
}

static int rsmi_measurefunc_fan_speed_max_get(uint32_t rsmiDevId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    uint64_t max_speed;
    RSMI_CALL(return -EIO, rsmi_dev_fan_speed_max_get, rsmiDevId, event->subvariant, &max_speed);
    result->fullValue += max_speed;
    result->lastValue = max_speed;

    return 0;
}

static int rsmi_measurefunc_temp_metric_get(uint32_t rsmiDevId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t temperature;
    RSMI_CALL(return -EIO, rsmi_dev_temp_metric_get, rsmiDevId, event->subvariant, event->variant, &temperature);
    result->fullValue += temperature;
    result->lastValue = temperature;

    return 0;
}


static int rsmi_measurefunc_volt_metric_get(uint32_t rsmiDevId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    int64_t voltage;
    RSMI_CALL(return -EIO, rsmi_dev_volt_metric_get, rsmiDevId, event->subvariant, event->variant, &voltage);
    result->fullValue += voltage;
    result->lastValue = voltage;

    return 0;
}

static int rsmi_measurefunc_overdrive_level_get(uint32_t rsmiDevId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    (void)event;

    uint32_t overdrive;
    RSMI_CALL(return -EIO, rsmi_dev_overdrive_level_get, rsmiDevId, &overdrive);
    result->fullValue += overdrive;
    result->lastValue = overdrive;

    return 0;
}

static int rsmi_measurefunc_ecc_count_get(uint32_t rsmiDevId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    rsmi_error_count_t error_count;
    RSMI_CALL(return -EIO, rsmi_dev_ecc_count_get, rsmiDevId, event->variant, &error_count);

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

static int rsmi_measurefunc_compute_process_info_get(uint32_t rsmiDevId, RocmonSmiEvent* event, RocmonEventResult* result)
{
    (void)rsmiDevId;
    (void)event;

    uint32_t num_items;
    RSMI_CALL(return -1, rsmi_compute_process_info_get, NULL, &num_items);
    result->fullValue += num_items;
    result->lastValue = num_items;

    return 0;
}

#define ADD_SMI_EVENT(name, type, smifunc, variant, subvariant, extra, measurefunc)\
    do {\
        int err_ = smi_event_add_impl(name, type, smifunc, variant, subvariant, extra, measurefunc);\
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
    return 0;
}

int rocmon_init(size_t numGpuIds, const int *gpuIds) {
    pthread_mutex_lock(&rocmon_init_mutex);

    int err = 0;
    if (rocmon_ctx_ref_count++ > 0) {
        // If rocmon is already initialized, check if the gpu id lists match.
        // We cannot allow initialization with different GPU ids, as rocmon
        // can only operate with one at a time.
        if (gpuIds) {
            if (numGpuIds != rocmon_ctx->numHipDeviceIdxToRocmonDeviceIdx) {
                err = -EINVAL;
                goto unlock_err_already_initialized;
            }
            for (size_t i = 0; i < rocmon_ctx->numHipDeviceIdxToRocmonDeviceIdx; i++) {
                const size_t rocmonDeviceIdx = rocmon_ctx->hipDeviceIdxToRocmonDeviceIdx[i];

                if (gpuIds[i] != rocmon_ctx->devices[rocmonDeviceIdx].hipDeviceId) {
                    err = -EINVAL;
                    goto unlock_err_already_initialized;
                }
            }
        }
        goto unlock_ok;
    }

    bool rsmi_initialized = false;
    bool libs_initialized = false;

    err = init_configuration();
    if (err < 0)
        goto unlock_err;

    err = rocmon_libraries_init();
    if (err < 0)
        goto unlock_err;

    libs_initialized = true;

    rocmon_ctx = calloc(1, sizeof(*rocmon_ctx));
    if (!rocmon_ctx) {
        err = -errno;
        goto unlock_err;
    }

    RSMI_CALL(err = -EIO; goto unlock_err, rsmi_init, 0);
    rsmi_initialized = true;

    uint32_t numRsmiDevices;
    RSMI_CALL(return -EIO, rsmi_num_monitor_devices, &numRsmiDevices);

    rocmon_ctx->devices = calloc(numRsmiDevices, sizeof(*rocmon_ctx->devices));
    if (!rocmon_ctx->devices) {
        err = -errno;
        goto unlock_err;
    }

    rocmon_ctx->numDevices = numRsmiDevices;

    err = rocmon_smi_init_implemented();
    if (err < 0)
        goto unlock_err;

    // Init rocprofiler context and associated data structures
    RPR_CALL(err = -EIO; goto unlock_err, rocprofiler_force_configure, rocprofiler_configure_private);

    err = rocmon_init_hip(numGpuIds, gpuIds);
    if (err < 0)
        goto unlock_err;

    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "rocmon initalization done");

unlock_ok:
    pthread_mutex_unlock(&rocmon_init_mutex);
    return 0;

unlock_err_already_initialized:
    rocmon_ctx_ref_count--;
    pthread_mutex_unlock(&rocmon_init_mutex);
    return err;

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

int rocmon_addEventSet(const char *eventString) {
    if (!eventString)
        return -EINVAL;

    if (!rocmon_ctx)
        return -EFAULT;

    const size_t newNumGroups = rocmon_ctx->numGroups + 1;
    RocmonGroupInfo *newGroups = realloc(rocmon_ctx->groups, newNumGroups * sizeof(*rocmon_ctx->groups));
    if (!newGroups)
        return -errno;

    RocmonGroupInfo *newGroup = &newGroups[rocmon_ctx->numGroups];
    memset(newGroup, 0, sizeof(*newGroup));

    const int retval = (int)rocmon_ctx->numGroups;

    rocmon_ctx->groups = newGroups;
    rocmon_ctx->numGroups = newNumGroups;

    if (strchr(eventString, ':')) {
        // If this looks like a regular performance group, create it from the string
        int err = perfgroup_customGroup(eventString, &newGroup->groupInfo);
        if (err < 0)
            return err;
    } else {
        // Otherwise, load it from a file instead
        int err = perfgroup_readGroup(get_configuration()->groupPath, "amd_gpu", eventString, &newGroup->groupInfo);
        if (err < 0)
            return err;
    }

    for (size_t i = 0; i < rocmon_ctx->numDevices; i++) {
        RocmonDevice *device = &rocmon_ctx->devices[i];
        if (!device->enabled)
            continue;

        size_t newNumGroupResults = device->numGroupResults + 1;
        RocmonEventResultList *newGroupResults = realloc(device->groupResults, newNumGroupResults * sizeof(*newGroupResults));
        if (!newGroupResults)
            return -errno;

        RocmonEventResultList *newGroupResult = &newGroupResults[device->numGroupResults];
        memset(newGroupResult, 0, sizeof(*newGroupResult));

        device->groupResults = newGroupResults;
        device->numGroupResults = newNumGroupResults;

        newGroupResult->eventResults = calloc(newGroup->groupInfo.nevents, sizeof(*newGroupResults->eventResults));
        if (!newGroupResult->eventResults)
            return -errno;

        newGroupResult->numEventResults = newGroup->groupInfo.nevents;
    }

    return retval;
}

int rocmon_switchActiveGroup(int newGroupId) {
    int err = rocmon_stopCounters();
    if (err < 0)
        return err;

    err = rocmon_setupCounters(newGroupId);
    if (err < 0)
        return err;

    return rocmon_startCounters();
}

int setup_counters_rpr(RocmonDevice *device, const GroupInfo *group) {
    size_t num_matching_events = 0;

    for (int i = 0; i < group->nevents; i++) {
        if (strncmp(group->events[i], RPR_EVENT_PREFIX, strlen(RPR_EVENT_PREFIX)) == 0)
            num_matching_events++;
    }

    rocprofiler_counter_id_t *newActiveRprEvents = calloc(num_matching_events, sizeof(*newActiveRprEvents));
    if (!newActiveRprEvents)
        return -errno;

    int err = 0;
    rocprofiler_counter_id_t *newActiveRprEvent = newActiveRprEvents;

    for (int i = 0; i < group->nevents; i++) {
        char *event_name = group->events[i];
        if (strncmp(event_name, RPR_EVENT_PREFIX, strlen(RPR_EVENT_PREFIX)) != 0)
            continue;

        // skip 'ROCP_' prefix
        event_name += strlen(RPR_EVENT_PREFIX);

        RocmonRprEvent *event;
        err = get_smap_by_key(device->availableRprEvents, event_name, (void **)&event);
        if (err < 0)
            goto cleanup;

        *newActiveRprEvent++ = event->counterInfo.id;
    }

    free(device->activeRprEvents);
    device->activeRprEvents = newActiveRprEvents;
    device->numActiveRprEvents = num_matching_events;
    return 0;

cleanup:
    free(newActiveRprEvents);
    return err;
}

// TODO:
// This function is kind of expensive, so once we actually understood how that works, let's make it a bit more efficient.
// Primarily, we should avoid deep copying the events, which is kind of unnecessary.
int setup_counters_smi(RocmonDevice *device, const GroupInfo *group) {
    size_t num_matching_events = 0;

    for (int i = 0; i < group->nevents; i++) {
        if (strncmp(group->events[i], RSMI_EVENT_PREFIX, strlen(RSMI_EVENT_PREFIX)) == 0)
            num_matching_events++;
    }

    RocmonSmiEvent *newActiveSmiEvents = calloc(num_matching_events, sizeof(*newActiveSmiEvents));
    if (!newActiveSmiEvents)
        return -errno;

    int err = 0;
    RocmonSmiEvent *newActiveSmiEvent = newActiveSmiEvents;

    for (int i = 0; i < group->nevents; i++) {
        char *event_name = group->events[i];
        if (strncmp(event_name, RSMI_EVENT_PREFIX, strlen(RSMI_EVENT_PREFIX)) != 0)
            continue;

        // skip 'RSMI_' prefix
        event_name += strlen(RSMI_EVENT_PREFIX);

        RocmonSmiEvent *event;
        err = get_smap_by_key(device->availableSmiEvents, event_name, (void **)&event);
        if (err < 0)
            goto cleanup;

        memcpy(newActiveSmiEvent, event, sizeof(*newActiveSmiEvent));

        newActiveSmiEvent++;
    }

    free(device->activeSmiEvents);
    device->activeSmiEvents = newActiveSmiEvents;
    device->numActiveSmiEvents = num_matching_events;
    return 0;

cleanup:
    free(newActiveSmiEvents);
    return err;
}

int rocmon_setupCounters(int gid) {
    if (!rocmon_ctx)
        return -EFAULT;

    if (gid < 0 || (size_t)gid >= rocmon_ctx->numGroups)
        return -EINVAL;

    const RocmonGroupInfo *group = &rocmon_ctx->groups[gid];

    for (size_t i = 0; i < rocmon_ctx->numDevices; i++) {
        RocmonDevice *device = &rocmon_ctx->devices[i];
        if (!device->enabled)
            continue;

        // Setup rocprofiler counters.
        // Please keep in mind the actual counter configuration for rocprofiler
        // will be set when the context is started. It is not possible to do it
        // here in advance.
        int err = setup_counters_rpr(device, &group->groupInfo);
        if (err < 0)
            return err;

        err = setup_counters_smi(device, &group->groupInfo);
        if (err < 0)
            return err;
    }

    return 0;
}

static int time_ns_get(uint64_t *timestamp) {
    struct timespec ts;
    int err = clock_gettime(CLOCK_MONOTONIC, &ts);
    if (err < 0)
        return -errno;
    *timestamp = ts.tv_sec * 1000000000ull + ts.tv_nsec;
    return 0;
}

static int counters_smi_start(RocmonDevice *device) {
    RocmonEventResultList *groupResult = &device->groupResults[rocmon_ctx->activeGroupIdx];

    for (size_t i = 0; i < device->numActiveSmiEvents; i++) {
        RocmonSmiEvent *event = &device->activeSmiEvents[i];

        // I know, it's a bit ugly, but the results of rocprofiler and rocm_smi
        // are stored in the same array, so don't start at a zero, but start at
        // 'device->numActiveRocEvents' instead.
        RocmonEventResult* result = &groupResult->eventResults[device->numActiveRprEvents+i];

        if (event->measureFunc) {
            int err = event->measureFunc(device->rsmiDeviceId, event, result);
            if (err < 0)
                return err;
        }

        result->fullValue = 0.0;
    }

    return 0;
}

static int counters_rpr_start(RocmonDevice *device) {
    RocmonEventResultList *groupResult = &device->groupResults[rocmon_ctx->activeGroupIdx];

    for (size_t i = 0; i < device->numActiveRprEvents; i++) {
        RocmonEventResult *result = &groupResult->eventResults[i];
        result->lastValue = 0;
        result->fullValue = 0;
    }

    // The actual measurement is started once for all devices,
    // so nothing else is done here.
    return 0;
}

int rocmon_startCounters(void) {
    if (!rocmon_ctx)
        return -EFAULT;

    uint64_t timestamp = 0;
    int err = time_ns_get(&timestamp);
    if (err < 0)
        return err;

    rocmon_ctx->groups[rocmon_ctx->activeGroupIdx].time.start = timestamp;
    rocmon_ctx->groups[rocmon_ctx->activeGroupIdx].time.read = timestamp;

    for (size_t i = 0; i < rocmon_ctx->numDevices; i++) {
        RocmonDevice *device = &rocmon_ctx->devices[i];
        if (!device->enabled)
            continue;

        err = counters_smi_start(device);
        if (err < 0)
            return err;

        err = counters_rpr_start(device);
        if (err < 0)
            return err;
    }

    RPR_CALL(return -EIO, rocprofiler_start_context, rocmon_ctx->rocprofCtx);

    return 0;
}

static int readCounters_impl(bool stop);

int rocmon_stopCounters(void) {
    if (!rocmon_ctx)
        return -EFAULT;

    int err = readCounters_impl(true);
    if (err < 0)
        return err;

    RPR_CALL(return -EIO, rocprofiler_stop_context, rocmon_ctx->rocprofCtx);
    return 0;
}

int counters_read_smi(RocmonDevice *device) {
    RocmonEventResultList *groupResult = &device->groupResults[rocmon_ctx->activeGroupIdx];

    for (size_t i = 0; i < device->numActiveSmiEvents; i++) {
        RocmonSmiEvent *event = &device->activeSmiEvents[i];
        RocmonEventResult *result = &groupResult->eventResults[device->numActiveRprEvents+i];

        if (event->measureFunc) {
            int err = event->measureFunc(device->rsmiDeviceId, event, result);
            if (err < 0)
                return err;
        }
    }

    return 0;
}

int counters_read_rpr(RocmonDevice *device) {
    RocmonEventResultList *groupResult = &device->groupResults[rocmon_ctx->activeGroupIdx];

    rocprofiler_user_data_t dummy_userdata;
    RPR_CALL(return -EIO,
            rocprofiler_sample_device_counting_service,
            rocmon_ctx->rocprofCtx,
            dummy_userdata,
            ROCPROFILER_COUNTER_FLAG_NONE,
            NULL,
            NULL);
    RPR_CALL(return -EIO, rocprofiler_flush_buffer, device->rocprofBuf);

    pthread_mutex_lock(&device->callbackRprMutex);

    for (size_t j = 0; j < device->numActiveRprEvents; j++) {
        rocprofiler_counter_id_t cid = device->activeRprEvents[j];

        char key[32];
        snprintf(key, sizeof(key), "%" PRIu64, cid.handle);

        double *value = NULL;
        int err = get_smap_by_key(device->callbackRprResults, key, (void **)&value);
        if (err < 0)
            return err;

        groupResult->eventResults[j].fullValue += *value;
        groupResult->eventResults[j].lastValue = *value;
    }

    pthread_mutex_unlock(&device->callbackRprMutex);

    return 0;
}

static int readCounters_impl(bool stop) {
    uint64_t timestamp = 0;
    int err = time_ns_get(&timestamp);
    if (err < 0)
        return err;

    RocmonGroupInfo *info = &rocmon_ctx->groups[rocmon_ctx->activeGroupIdx];

    if (stop)
        info->time.stop = timestamp;
    else
        info->time.read = timestamp;

    for (size_t i = 0; i < rocmon_ctx->numDevices; i++) {
        RocmonDevice *device = &rocmon_ctx->devices[i];
        if (!device->enabled)
            continue;

        assert(device->numActiveSmiEvents + device->numActiveRprEvents ==
                device->groupResults[rocmon_ctx->activeGroupIdx].numEventResults);

        err = counters_read_smi(device);
        if (err < 0)
            return err;

        err = counters_read_rpr(device);
        if (err < 0)
            return err;
    }

    return 0;
}

int rocmon_readCounters(void) {
    if (!rocmon_ctx)
        return -EFAULT;
    
    return readCounters_impl(false);
}

static int getEventResult(int hipDeviceId, int groupId, int eventId, RocmonEventResult **result) {
    if (!rocmon_ctx)
        return -EFAULT;

    RocmonDevice *device = device_get(hipDeviceId);
    if (!device)
        return -EINVAL;

    if (groupId < 0 || (size_t)groupId >= device->numGroupResults)
        return -EINVAL;

    RocmonEventResultList *groupResult = &device->groupResults[groupId];
    if (eventId < 0 || (size_t)eventId >= groupResult->numEventResults)
        return -EINVAL;

    *result = &groupResult->eventResults[eventId];
    return 0;
}

double rocmon_getResult(int hipDeviceId, int groupId, int eventId) {
    RocmonEventResult *result;
    int err = getEventResult(hipDeviceId, groupId, eventId, &result);
    if (err < 0)
        return err;

    return result->fullValue;
}

double rocmon_getLastResult(int hipDeviceId, int groupId, int eventId) {
    RocmonEventResult *result;
    int err = getEventResult(hipDeviceId, groupId, eventId, &result);
    if (err < 0)
        return err;

    return result->lastValue;
}

int rocmon_getEventsOfGpu(int hipDeviceId, RocmonEventList_t *list) {
    RocmonDevice *device = device_get(hipDeviceId);
    if (!device)
        return -EINVAL;

    // Allocate new event list
    RocmonEventList *newList = calloc(1, sizeof(*newList));
    if (!newList)
        return -errno;

    int numEventsSmi = get_map_size(device->availableSmiEvents);
    assert(numEventsSmi >= 0);

    int numEventsRocprof = get_map_size(device->availableRprEvents);
    assert(numEventsRocprof >= 0);

    int err = 0;

    const size_t newNumEvents = (size_t)numEventsSmi + (size_t)numEventsRocprof;
    newList->events = calloc(newNumEvents, sizeof(*newList->events));
    if (!newList->events) {
        err = -errno;
        goto cleanup;
    }

    newList->numEvents = newNumEvents;

    RocmonEventListEntry *newEntry = &newList->events[0];

    // Create ROCM SMI event entries
    for (int i = 0; i < numEventsSmi; i++) {
        RocmonSmiEvent *event;
        err = get_smap_by_idx(device->availableSmiEvents, i, (void **)&event);
        if (err < 0)
            goto cleanup;

        // e.g. RSMI_ + FAN_SPEED -> RSMI_FAN_SPEED
        newEntry->name = xasprintf("%s%s", RSMI_EVENT_PREFIX, event->name);
        if (!newEntry->name) {
            err = -errno;
            goto cleanup;
        }
        newEntry->desc = strdup("ROCM SMI event: <no description>");
        if (!newEntry->desc) {
            err = -errno;
            goto cleanup;
        }

        newEntry++;
    }

    // Create ROCPROFILER-SDK event entries
    for (int i = 0; i < numEventsRocprof; i++) {
        RocmonRprEvent *event;
        err = get_smap_by_idx(device->availableRprEvents, i, (void **)&event);
        if (err < 0)
            goto cleanup;

        // e.g. ROCP_ + GPU_UTIL -> ROCP_GPU_UTIL
        newEntry->name = xasprintf("%s%s", RPR_EVENT_PREFIX, event->counterInfo.name);
        if (!newEntry->name) {
            err = -errno;
            goto cleanup;
        }
        newEntry->desc = xasprintf("ROCPROFILER-SDK event: %s", event->counterInfo.description);
        if (!newEntry->desc) {
            err = -errno;
            goto cleanup;
        }

        newEntry++;
    }

    *list = newList;
    return 0;

cleanup:
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
    if (!rocmon_ctx)
        return -EFAULT;

    return rocmon_ctx->numGroups;
}

int rocmon_getIdOfActiveGroup(void) {
    if (!rocmon_ctx)
        return -EFAULT;

    return rocmon_ctx->activeGroupIdx;
}

int rocmon_getNumberOfGPUs(void) {
    if (!rocmon_ctx)
        return -EFAULT;

    // Return the number of HIP devices.
    // Not all RocmonDevices may have a valid HIP device, since some of them can be
    // disabled via HIP_VISIBLE_DEVICES.
    return (int)rocmon_ctx->numHipDeviceIdxToRocmonDeviceIdx;
}

int rocmon_getIdOfGPU(size_t idx) {
    if (!rocmon_ctx)
        return -EFAULT;

    if (idx >= rocmon_ctx->numHipDeviceIdxToRocmonDeviceIdx)
        return -EINVAL;

    const size_t rocmonDeviceIdx = rocmon_ctx->hipDeviceIdxToRocmonDeviceIdx[idx];
    return rocmon_ctx->devices[rocmonDeviceIdx].hipDeviceId;
}

static int getGroupInfo(int groupId, RocmonGroupInfo **rocmonGroupInfo) {
    if (!rocmon_ctx)
        return -EFAULT;

    if (groupId < 0 || (size_t)groupId >= rocmon_ctx->numGroups)
        return -EINVAL;

    *rocmonGroupInfo = &rocmon_ctx->groups[groupId];
    return 0;
}

int rocmon_getNumberOfEvents(int groupId) {
    RocmonGroupInfo *info;

    int err = getGroupInfo(groupId, &info);
    if (err < 0)
        return err;

    return (int)info->groupInfo.nevents;
}

int rocmon_getNumberOfMetrics(int groupId) {
    RocmonGroupInfo *info;

    int err = getGroupInfo(groupId, &info);
    if (err < 0)
        return err;

    return (int)info->groupInfo.nmetrics;
}

int rocmon_getEventName(int groupId, int eventId, const char **eventName) {
    RocmonGroupInfo *info;

    int err = getGroupInfo(groupId, &info);
    if (err < 0)
        return err;

    if (eventId < 0 || eventId >= info->groupInfo.nevents)
        return -EINVAL;

    *eventName = info->groupInfo.events[eventId];
    return 0;
}

int rocmon_getCounterName(int groupId, int eventId, const char **counterName) {
    RocmonGroupInfo *info;

    int err = getGroupInfo(groupId, &info);
    if (err < 0)
        return err;

    if (eventId < 0 || eventId >= info->groupInfo.nevents)
        return -EINVAL;

    *counterName = info->groupInfo.counters[eventId];
    return 0;
}

int rocmon_getMetricName(int groupId, int metricId, const char **metricName) {
    RocmonGroupInfo *info;

    int err = getGroupInfo(groupId, &info);
    if (err < 0)
        return err;

    if (metricId < 0 || metricId >= info->groupInfo.nmetrics)
        return -EINVAL;

    *metricName = info->groupInfo.metricnames[metricId];
    return 0;
}

int rocmon_getMetricFormula(int groupId, int metricId, const char **formula) {
    RocmonGroupInfo *info;

    int err = getGroupInfo(groupId, &info);
    if (err < 0)
        return err;

    if (metricId < 0 || metricId >= info->groupInfo.nmetrics)
        return -EINVAL;

    *formula = info->groupInfo.metricformulas[metricId];
    return 0;
}

double rocmon_getTimeOfGroup(int groupId) {
    RocmonGroupInfo *info;

    int err = getGroupInfo(groupId, &info);
    if (err < 0)
        return err;

    return (double)(info->time.stop - info->time.start);
}

double rocmon_getLastTimeOfGroup(int groupId) {
    RocmonGroupInfo *info;

    int err = getGroupInfo(groupId, &info);
    if (err < 0)
        return err;

    return (double)(info->time.stop - info->time.read);
}

double rocmon_getTimeToLastReadOfGroup(int groupId) {
    RocmonGroupInfo *info;

    int err = getGroupInfo(groupId, &info);
    if (err < 0)
        return err;

    return (double)(info->time.read - info->time.start);
}

int rocmon_getTimestampOfLastReadOfGroup(int groupId, uint64_t *timestamp) {
    RocmonGroupInfo *info;

    int err = getGroupInfo(groupId, &info);
    if (err < 0)
        return err;

    *timestamp = info->time.read;
    return 0;
}

int rocmon_getGroupName(int groupId, const char **groupName) {
    RocmonGroupInfo *info;

    int err = getGroupInfo(groupId, &info);
    if (err < 0)
        return err;

    *groupName = info->groupInfo.groupname;
    return 0;
}

int rocmon_getGroupInfoShort(int groupId, const char **groupInfoShort) {
    RocmonGroupInfo *info;

    int err = getGroupInfo(groupId, &info);
    if (err < 0)
        return err;

    *groupInfoShort = info->groupInfo.shortinfo;
    return 0;
}

int rocmon_getGroupInfoLong(int groupId, const char **groupInfoLong) {
    RocmonGroupInfo *info;

    int err = getGroupInfo(groupId, &info);
    if (err < 0)
        return err;

    *groupInfoLong = info->groupInfo.longinfo;
    return 0;
}

int rocmon_getGroups(size_t *numGroups, char ***groups, char ***shortinfos, char ***longinfos) {
    int err = init_configuration();
    if (err < 0)
        return err;

    
    err = perfgroup_getGroups(get_configuration()->groupPath, "amd_gpu", groups, shortinfos, longinfos);
    if (err < 0)
        return err;

    *numGroups = (size_t)err;
    return 0;
}

void rocmon_returnGroups(size_t numGroups, char **groupNames, char **shortInfos, char **longInfos) {
    perfgroup_returnGroups((int)numGroups, groupNames, shortInfos, longInfos);
}
