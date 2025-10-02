#include <rocmon_types.h>

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

// TODO clean this up and sort the variables
__attribute__((visibility("default")))
int likwid_rocmon_verbosity = DEBUGLEV_ONLY_ERROR;

DECLARE_STATIC_PTMUTEX(rocmon_init_mutex);
static RocmonContext *rocmon_ctx;
static size_t rocmon_ctx_ref_count = 0;
static rocprofiler_context_id_t rocprof_ctx;

static void *lib_amdhip;
static void *lib_rocm_smi;
static void *lib_rocprofiler_sdk;

// ROCm function declarations
#define RPR_CALL(handleerror, func, ...) \
    do { \
        rocprofiler_status_t s_ = (*func##_ptr)(__VA_ARGS__);\
        if (s_ != ROCPROFILER_STATUS_SUCCESS) {           \
            const char *errstr_ = rocprofiler_get_status_string_ptr(s_); \
            ERROR_PRINT("Error: function %s failed with error: '%s' (hsa_status_t=%d).", #func, errstr_, s_);\
            handleerror;\
        }\
    } while (0)

#define RSMI_CALL(handleerror, func, ...)\
    do {\
        rsmi_status_t s_ = (*func##_ptr)(__VA_ARGS__);\
        if (s_ != RSMI_STATUS_SUCCESS) {\
            const char *errstr_ = NULL;\
            rsmi_status_string_ptr(s_, &errstr_);\
            ERROR_PRINT("Error: function %s failed with error: '%s' (rsmi_status_t=%d)", #func, errstr_, s_);\
            handleerror;\
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

DECLAREFUNC_HIP(hipGetDeviceCount, int *);
static const char *(*hipGetErrorName_ptr)(hipError_t);

static int tool_init(rocprofiler_client_finalize_t, void *) {
    rocprofiler_status_t s = rocprofiler_create_context_ptr(&rocprof_ctx);
    if (s != ROCPROFILER_STATUS_SUCCESS) {
        fprintf(stderr, "rocprofiler_create_context failed: %s\n", rocprofiler_get_status_string(s));
        return -1;
    }

    return 0;
}

static void tool_fini(void *) {
    // Caution, This function is called by the ROCm runtime prior to execution of main.
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
#define DLSYM_CHK(dllib, name) \
    do { \
        name##_ptr = dlsym(dllib, #name); \
        const char *err_ = dlerror(); \
        if (err_) { \
            ERROR_PRINT("Failed to link '%s': %s", #name, err_); \
            err = -ENXIO; \
            goto ret_err; \
        } \
    } while (0)

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

    lib_amdhip = dlopen("libamdhip64.so", RTLD_GLOBAL | RTLD_NOW);
    if (!lib_amdhip) {
        err = -ELIBACC;
        goto ret_err;
    }

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

    rocmon_ctx->devices = calloc(numGpus, sizeof(rocmon_ctx->devices));
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
    // don't appear to do the trick.
    int deviceCount;
    if (hipGetDeviceCount_ptr(&deviceCount) != hipSuccess) {
    }
    // TODO initialize more stuff

    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, "rocmon initalization done");

unlock_ok:
    pthread_mutex_unlock(&rocmon_init_mutex);
    return 0;

unlock_err:
    if (rsmi_initialized)
        (void)rsmi_shut_down_ptr();

    if (libs_initialized)
        rocmon_libraries_fini();

    if (rocmon_ctx)
        free(rocmon_ctx->devices);

    free(rocmon_ctx);
    rocmon_ctx = NULL;

    pthread_mutex_unlock(&rocmon_init_mutex);
    return err;
}

void rocmon_finalize(void) {
    pthread_mutex_lock(&rocmon_init_mutex);

    // init must occur before finalize
    assert(rocmon_ctx_ref_count != 0);

    if (--rocmon_ctx_ref_count > 0)
        goto unlock_ret;

    (void)rsmi_shut_down_ptr();

    free(rocmon_ctx);
    rocmon_ctx = NULL;

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

int rocmon_getEventsOfGpu(int gpuIdx, EventList_rocm_t *list) {
}

void rocmon_freeEventsOfGpu(EventList_rocm_t list) {
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
