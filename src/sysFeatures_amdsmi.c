/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_amdsmi.c
 *
 *      Description:  Interface to control various AMD-SMI based features
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Authors:  Thomas.Gruber, thomas.gruber@fau.de
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

#include <sysFeatures_amdsmi.h>

#include <assert.h>
#include <math.h>
#include <dlfcn.h>
#include <amd_smi/amdsmi.h>

#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <dlfcn.h>
#include <string.h>


#include <error.h>
#include <sysFeatures_common.h>
#include <types.h>
#include <lw_alloc.h>




#define DECLAREFUNC_ASMI(funcname, ...) static amdsmi_status_t (*funcname##_ptr)(__VA_ARGS__)

DECLAREFUNC_ASMI(amdsmi_init, uint64_t flags);
DECLAREFUNC_ASMI(amdsmi_shut_down, void);
DECLAREFUNC_ASMI(amdsmi_get_esmi_err_msg, amdsmi_status_t status, const char **status_string);

DECLAREFUNC_ASMI(amdsmi_get_socket_handles, uint32_t *socket_count, amdsmi_socket_handle *socket_handles);
DECLAREFUNC_ASMI(amdsmi_get_cpucore_handles, uint32_t *cores_count, amdsmi_processor_handle *processor_handles);
DECLAREFUNC_ASMI(amdsmi_get_processor_handles, amdsmi_socket_handle socket_handle, uint32_t *processor_count, amdsmi_processor_handle *processor_handles);
DECLAREFUNC_ASMI(amdsmi_get_processor_handle_from_bdf, amdsmi_bdf_t bdf, amdsmi_processor_handle *processor_handle);
DECLAREFUNC_ASMI(amdsmi_get_processor_handles_by_type, amdsmi_socket_handle socket_handle, processor_type_t processor_type, amdsmi_processor_handle *processor_handles, uint32_t *processor_count);
DECLAREFUNC_ASMI(amdsmi_get_processor_type, amdsmi_processor_handle processor_handle, processor_type_t *processor_type);
DECLAREFUNC_ASMI(amdsmi_get_gpu_board_info, amdsmi_processor_handle processor_handle, amdsmi_board_info_t *info);
DECLAREFUNC_ASMI(amdsmi_get_power_cap_info, amdsmi_processor_handle processor_handle, uint32_t sensor_ind, amdsmi_power_cap_info_t *info);
DECLAREFUNC_ASMI(amdsmi_set_power_cap, amdsmi_processor_handle processor_handle, uint32_t sensor_ind, uint64_t cap);

DECLAREFUNC_ASMI(amdsmi_get_pcie_info, amdsmi_processor_handle processor_handle, amdsmi_pcie_info_t *info);

DECLAREFUNC_ASMI(amdsmi_get_socket_info, amdsmi_socket_handle socket_handle, size_t len, char *name);
DECLAREFUNC_ASMI(amdsmi_get_processor_info, amdsmi_processor_handle processor_handle, size_t len, char *name);
DECLAREFUNC_ASMI(amdsmi_get_gpu_vram_info, amdsmi_processor_handle processor_handle, amdsmi_vram_info_t *info);
DECLAREFUNC_ASMI(amdsmi_get_gpu_kfd_info, amdsmi_processor_handle processor_handle, amdsmi_kfd_info_t *info);
DECLAREFUNC_ASMI(amdsmi_get_gpu_asic_info, amdsmi_processor_handle processor_handle, amdsmi_asic_info_t *info);
DECLAREFUNC_ASMI(amdsmi_get_gpu_device_bdf, amdsmi_processor_handle processor_handle, amdsmi_bdf_t *bdf);
DECLAREFUNC_ASMI(amdsmi_get_power_info, amdsmi_processor_handle processor_handle, amdsmi_power_info_t *info);
DECLAREFUNC_ASMI(amdsmi_get_clock_info, amdsmi_processor_handle processor_handle, amdsmi_clk_type_t clk_type, amdsmi_clk_info_t *info);
DECLAREFUNC_ASMI(amdsmi_get_temp_metric, amdsmi_processor_handle processor_handle, amdsmi_temperature_type_t sensor_type, amdsmi_temperature_metric_t metric, int64_t *temperature);

DECLAREFUNC_ASMI(amdsmi_get_clk_freq, amdsmi_processor_handle processor_handle, amdsmi_clk_type_t clk_type, amdsmi_frequencies_t *f);
DECLAREFUNC_ASMI(amdsmi_set_clk_freq, amdsmi_processor_handle processor_handle, amdsmi_clk_type_t clk_type, uint64_t freq_bitmask);
DECLAREFUNC_ASMI(amdsmi_set_gpu_clk_limit, amdsmi_processor_handle processor_handle, amdsmi_clk_type_t clk_type, amdsmi_clk_limit_type_t limit_type, uint64_t clk_value);
DECLAREFUNC_ASMI(amdsmi_get_cpu_socket_power, amdsmi_processor_handle processor_handle, uint32_t *ppower);
DECLAREFUNC_ASMI(amdsmi_get_cpu_socket_power_cap, amdsmi_processor_handle processor_handle, uint32_t *ppower);
DECLAREFUNC_ASMI(amdsmi_set_cpu_socket_power_cap, amdsmi_processor_handle processor_handle, uint32_t pcap);
DECLAREFUNC_ASMI(amdsmi_get_cpu_socket_power_cap_max, amdsmi_processor_handle processor_handle, uint32_t *pmax);

DECLAREFUNC_ASMI(amdsmi_get_soc_pstate, amdsmi_processor_handle processor_handle, amdsmi_dpm_policy_t *policy);
DECLAREFUNC_ASMI(amdsmi_set_soc_pstate, amdsmi_processor_handle processor_handle, uint32_t policy_id);

DECLAREFUNC_ASMI(amdsmi_get_gpu_perf_level, amdsmi_processor_handle processor_handle, amdsmi_dev_perf_level_t *perf);
DECLAREFUNC_ASMI(amdsmi_set_gpu_perf_level, amdsmi_processor_handle processor_handle, amdsmi_dev_perf_level_t perf_lvl);

DECLAREFUNC_ASMI(amdsmi_set_gpu_perf_determinism_mode, amdsmi_processor_handle processor_handle, uint64_t clkvalue);

DECLAREFUNC_ASMI(amdsmi_get_gpu_overdrive_level, amdsmi_processor_handle processor_handle, uint32_t *od);
DECLAREFUNC_ASMI(amdsmi_set_gpu_overdrive_level, amdsmi_processor_handle processor_handle, uint32_t od);
DECLAREFUNC_ASMI(amdsmi_get_gpu_mem_overdrive_level, amdsmi_processor_handle processor_handle, uint32_t *od);

DECLAREFUNC_ASMI(amdsmi_get_gpu_od_volt_info, amdsmi_processor_handle processor_handle, amdsmi_od_volt_freq_data_t *odv);
DECLAREFUNC_ASMI(amdsmi_set_gpu_od_volt_info, amdsmi_processor_handle processor_handle, uint32_t vpoint, uint64_t clkvalue, uint64_t voltvalue);

#if (AMDSMI_LIB_VERSION_MAJOR >= 26 && AMDSMI_LIB_VERSION_MINOR >= 2 && AMDSMI_LIB_VERSION_RELEASE >= 1)
DECLAREFUNC_ASMI(amdsmi_get_supported_power_cap, amdsmi_processor_handle processor_handle, uint32_t *sensor_count, uint32_t *sensor_inds, amdsmi_power_cap_type_t *sensor_types);
DECLAREFUNC_ASMI(amdsmi_get_node_handle, amdsmi_processor_handle processor_handle, amdsmi_node_handle *node_handle);
DECLAREFUNC_ASMI(amdsmi_get_npm_info, amdsmi_node_handle node_handle, amdsmi_npm_info_t *info);
#endif

DECLAREFUNC_ASMI(amdsmi_get_cpu_handles, uint32_t *cpu_count, amdsmi_processor_handle *processor_handles);
DECLAREFUNC_ASMI(amdsmi_clean_gpu_local_data, amdsmi_processor_handle processor_handle);

DECLAREFUNC_ASMI(amdsmi_set_gpu_fan_speed, amdsmi_processor_handle processor_handle, uint32_t sensor_ind, uint64_t speed);
DECLAREFUNC_ASMI(amdsmi_get_gpu_fan_rpms, amdsmi_processor_handle processor_handle, uint32_t sensor_ind, int64_t *speed);
DECLAREFUNC_ASMI(amdsmi_get_gpu_volt_metric, amdsmi_processor_handle processor_handle, amdsmi_voltage_type_t sensor_type, amdsmi_voltage_metric_t metric, int64_t *voltage);
DECLAREFUNC_ASMI(amdsmi_get_gpu_metrics_info, amdsmi_processor_handle processor_handle, amdsmi_gpu_metrics_t * pgpu_metrics);
DECLAREFUNC_ASMI(amdsmi_get_gpu_fan_speed_max, amdsmi_processor_handle processor_handle, uint32_t sensor_ind, uint64_t *max_speed);

DECLAREFUNC_ASMI(amdsmi_get_gpu_activity, amdsmi_processor_handle processor_handle, amdsmi_engine_usage_t *info);
DECLAREFUNC_ASMI(amdsmi_get_temp_metric, amdsmi_processor_handle processor_handle, amdsmi_temperature_type_t sensor_type, amdsmi_temperature_metric_t metric, int64_t *temperature);
DECLAREFUNC_ASMI(amdsmi_is_gpu_power_management_enabled, amdsmi_processor_handle processor_handle, bool *enabled);

#define ASMI_CALL(func, ...) \
    do {\
        assert(func##_ptr != NULL); \
        amdsmi_status_t s_ = func##_ptr(__VA_ARGS__);\
        if (s_ != AMDSMI_STATUS_SUCCESS) {\
            const char *errstr_ = NULL;\
            amdsmi_status_t es_ = amdsmi_get_esmi_err_msg_ptr(s_, &errstr_);\
            if (es_ != AMDSMI_STATUS_SUCCESS) { \
                ERROR_PRINT("Error: function %s failed but cannot resolve error string (amdsmi_status_t=%d, err_msg_status_t=%d)", #func, s_, es_);\
            } else { \
                ERROR_PRINT("Error: function %s failed: '%s' (amdsmi_status_t=%d)", #func, errstr_, s_);\
            } \
            return -EPERM; \
        }\
    } while (0)

static void *lib_amd_smi = NULL;
static bool amdsmi_initialized = false;
static const _SysFeatureList amd_smi_feature_list;

#define DLSYM_CHK(dllib, name) name##_ptr = dlsym(dllib, #name);  \
    do {                                                                \
        const char *err = dlerror();                                    \
        if (err) {                                                      \
            ERROR_PRINT("Error: dlsym on symbol '%s' failed with error: %s", #name, err); \
            return -EINVAL;                                             \
        }                                                               \
    } while (0)

static void cleanup_amdsmi(void)
{
    if (!amdsmi_initialized)
        return;

    amdsmi_shut_down_ptr();
    dlclose(lib_amd_smi);
}

/*static char* typeString(processor_type_t type) {*/
/*    switch (type) {*/
/*        case AMDSMI_PROCESSOR_TYPE_UNKNOWN:*/
/*            return "Unknown";*/
/*            break;*/
/*        case AMDSMI_PROCESSOR_TYPE_AMD_GPU:*/
/*            return "AMD GPU";*/
/*            break;*/
/*        case AMDSMI_PROCESSOR_TYPE_AMD_CPU:*/
/*            return "AMD CPU";*/
/*            break;*/
/*        case AMDSMI_PROCESSOR_TYPE_NON_AMD_GPU:*/
/*            return "Non AMD GPU";*/
/*            break;*/
/*        case AMDSMI_PROCESSOR_TYPE_NON_AMD_CPU:*/
/*            return "Non AMD CPU";*/
/*            break;*/
/*        case AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE:*/
/*            return "AMD CPU Core";*/
/*            break;*/
/*        case AMDSMI_PROCESSOR_TYPE_AMD_APU:*/
/*            return "AMD APU";*/
/*            break;*/
/*    }*/
/*    return "Invalid type";*/
/*}*/

int likwid_sysft_init_amdsmi(_SysFeatureList *list)
{
    if (amdsmi_initialized)
        return 0;

    lib_amd_smi = dlopen("libamd_smi.so", RTLD_GLOBAL | RTLD_NOW);
    if (!lib_amd_smi) {
        DEBUG_PRINT(DEBUGLEV_INFO, "dlopen(libamd_smi.so) failed: %s", dlerror());
        return -ELIBACC;
    }

    DLSYM_CHK(lib_amd_smi, amdsmi_init);
    DLSYM_CHK(lib_amd_smi, amdsmi_shut_down);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_esmi_err_msg);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_socket_handles);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_cpucore_handles);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_processor_handles);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_processor_handle_from_bdf);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_processor_handles_by_type);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_processor_type);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_board_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_power_cap_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_set_power_cap);

    DLSYM_CHK(lib_amd_smi, amdsmi_get_pcie_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_socket_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_processor_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_vram_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_kfd_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_asic_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_device_bdf);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_power_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_clock_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_temp_metric);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_clk_freq);
    DLSYM_CHK(lib_amd_smi, amdsmi_set_clk_freq);
    DLSYM_CHK(lib_amd_smi, amdsmi_set_gpu_clk_limit);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_cpu_socket_power);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_cpu_socket_power_cap);
    DLSYM_CHK(lib_amd_smi, amdsmi_set_cpu_socket_power_cap);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_cpu_socket_power_cap_max);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_soc_pstate);
    DLSYM_CHK(lib_amd_smi, amdsmi_set_soc_pstate);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_perf_level);
    DLSYM_CHK(lib_amd_smi, amdsmi_set_gpu_perf_level);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_overdrive_level);
    DLSYM_CHK(lib_amd_smi, amdsmi_set_gpu_overdrive_level);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_mem_overdrive_level);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_od_volt_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_set_gpu_od_volt_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_cpu_handles);
    DLSYM_CHK(lib_amd_smi, amdsmi_clean_gpu_local_data);
    DLSYM_CHK(lib_amd_smi, amdsmi_set_gpu_fan_speed);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_fan_rpms);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_volt_metric);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_metrics_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_fan_speed_max);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_activity);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_temp_metric);
    DLSYM_CHK(lib_amd_smi, amdsmi_is_gpu_power_management_enabled);

#if (AMDSMI_LIB_VERSION_MAJOR >= 26 && AMDSMI_LIB_VERSION_MINOR >= 2 && AMDSMI_LIB_VERSION_RELEASE >= 1)
    DLSYM_CHK(lib_amd_smi, amdsmi_get_supported_power_cap);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_node_handle);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_npm_info);
#endif

    amdsmi_status_t status = amdsmi_init_ptr(AMDSMI_INIT_AMD_GPUS | AMDSMI_INIT_AMD_CPUS);
    if (status != AMDSMI_STATUS_SUCCESS) {
        dlclose(lib_amd_smi);
        const char* errstr = NULL;
        amdsmi_get_esmi_err_msg_ptr(status, &errstr);
        ERROR_PRINT("amdsmi_init() failed: %s", errstr);
        return -EPERM;
    }

    atexit(cleanup_amdsmi);

    amdsmi_initialized = true;



    //TODO: Dynamically add sensors for amdsmi_get_supported_power_cap to sysfeatures list
    // amdsmi_get_supported_power_cap likely fails, so check against sysfs as well.
    // name power1_cap -> sensor 0


    return likwid_sysft_register_features(list, &amd_smi_feature_list);
}
#undef DLSYM_CHK

static int lw_device_to_amdsmi_gpu_handle(const LikwidDevice_t lwDevice, amdsmi_processor_handle *amdsmiHandle) {
    if (lwDevice->type != DEVICE_TYPE_AMD_GPU)
        return -EINVAL;
    amdsmi_bdf_t bdf;
    amdsmi_processor_handle out = NULL;

    bdf.domain_number = lwDevice->id.pci.pci_domain;
    bdf.bus_number = lwDevice->id.pci.pci_bus;
    bdf.device_number = lwDevice->id.pci.pci_dev;
    bdf.function_number = lwDevice->id.pci.pci_func;

    ASMI_CALL(amdsmi_get_processor_handle_from_bdf, bdf, &out);

    *amdsmiHandle = out;
    return 0;
}


static int amd_smi_device_count_getter(const LikwidDevice_t device, char **value)
{
    if (device->type != DEVICE_TYPE_NODE)
        return -EINVAL;

    uint64_t allDevices = 0;
    uint32_t socketCount = 0;
    ASMI_CALL(amdsmi_get_socket_handles, &socketCount, NULL);
    amdsmi_socket_handle* socketHandles = lw_malloc(socketCount * sizeof(amdsmi_socket_handle));

    ASMI_CALL(amdsmi_get_socket_handles, &socketCount, socketHandles);

    for (uint32_t i = 0; i < socketCount; i++) {
        uint32_t deviceCount = 0;
        ASMI_CALL(amdsmi_get_processor_handles, socketHandles[i], &deviceCount, NULL);
        
        allDevices += deviceCount;
    }
    free(socketHandles);

    return likwid_sysft_uint64_to_string(allDevices, value);
}


typedef enum {
    AMDSMI_CLOCK_FREQ_GETTER_FIELD_CURRENT = 0,
    AMDSMI_CLOCK_FREQ_GETTER_FIELD_MIN = AMDSMI_CLOCK_FREQ_GETTER_FIELD_CURRENT,
    AMDSMI_CLOCK_FREQ_GETTER_FIELD_DEEP_SLEEP,
    AMDSMI_CLOCK_FREQ_GETTER_FIELD_SUP_FREQS,
    AMDSMI_CLOCK_FREQ_GETTER_FIELD_MAX = AMDSMI_CLOCK_FREQ_GETTER_FIELD_SUP_FREQS,
} amd_smi_current_clock_freq_field;

static int amd_smi_generic_clk_freq_getter(const LikwidDevice_t device, char **value, amdsmi_clk_type_t clk_type, amd_smi_current_clock_freq_field field)
{
    if (device->type != DEVICE_TYPE_AMD_GPU)
        return -EINVAL;
    if (clk_type < AMDSMI_CLK_TYPE_FIRST || clk_type > AMDSMI_CLK_TYPE__MAX)
        return -EINVAL;
    if (field < AMDSMI_CLOCK_FREQ_GETTER_FIELD_MIN || field > AMDSMI_CLOCK_FREQ_GETTER_FIELD_MAX)
        return -EINVAL;

    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }
    amdsmi_frequencies_t freqs;
    bstring sup_freqs = NULL;

    ASMI_CALL(amdsmi_get_clk_freq, handle, clk_type, &freqs);
    switch (field) {
        case AMDSMI_CLOCK_FREQ_GETTER_FIELD_CURRENT:
            return likwid_sysft_uint64_to_string(freqs.current, value);
            break;
        case AMDSMI_CLOCK_FREQ_GETTER_FIELD_DEEP_SLEEP:
            return likwid_sysft_uint64_to_string((uint64_t)freqs.has_deep_sleep, value);
            break;
        case AMDSMI_CLOCK_FREQ_GETTER_FIELD_SUP_FREQS:
            if (freqs.num_supported > 0) {
                sup_freqs = bformat("%ld", freqs.frequency[0]);
            }
            for (uint32_t i = 1; i < freqs.num_supported; i++) {
                bconchar(sup_freqs, ' ');
                bstring tmp = bformat("%ld", freqs.frequency[i]);
                bconcat(sup_freqs, tmp);
                bdestroy(tmp);
            }
            int ret = likwid_sysft_copystr(bdata(sup_freqs), value);
            if (sup_freqs) bdestroy(sup_freqs);
            return ret;
            break;
    }
    return -EINVAL;
}

#define CLOCK_FREQ_GETTER_GEN(name, type, field) \
static int amd_smi_##name##_clock_freq_getter(const LikwidDevice_t device, char **value) { \
    return amd_smi_generic_clk_freq_getter(device, value, (type), (field)); \
}
#define CLOCK_FREQ_GETTER_GEN_TRIPLE(name, type) \
    CLOCK_FREQ_GETTER_GEN(name##_current, (type), AMDSMI_CLOCK_FREQ_GETTER_FIELD_CURRENT); \
    CLOCK_FREQ_GETTER_GEN(name##_hasdeepsleep, (type), AMDSMI_CLOCK_FREQ_GETTER_FIELD_DEEP_SLEEP); \
    CLOCK_FREQ_GETTER_GEN(name##_supported_freqs, (type), AMDSMI_CLOCK_FREQ_GETTER_FIELD_SUP_FREQS);

CLOCK_FREQ_GETTER_GEN_TRIPLE(sys, AMDSMI_CLK_TYPE_SYS);
CLOCK_FREQ_GETTER_GEN_TRIPLE(mem, AMDSMI_CLK_TYPE_MEM);
CLOCK_FREQ_GETTER_GEN_TRIPLE(gfx, AMDSMI_CLK_TYPE_GFX);
CLOCK_FREQ_GETTER_GEN_TRIPLE(df, AMDSMI_CLK_TYPE_DF);
CLOCK_FREQ_GETTER_GEN_TRIPLE(dcef, AMDSMI_CLK_TYPE_DCEF);
CLOCK_FREQ_GETTER_GEN_TRIPLE(soc, AMDSMI_CLK_TYPE_SOC);
CLOCK_FREQ_GETTER_GEN_TRIPLE(pcie, AMDSMI_CLK_TYPE_PCIE);
CLOCK_FREQ_GETTER_GEN_TRIPLE(video0, AMDSMI_CLK_TYPE_VCLK0);
CLOCK_FREQ_GETTER_GEN_TRIPLE(video1, AMDSMI_CLK_TYPE_VCLK1);
CLOCK_FREQ_GETTER_GEN_TRIPLE(display0, AMDSMI_CLK_TYPE_DCLK0);
CLOCK_FREQ_GETTER_GEN_TRIPLE(display1, AMDSMI_CLK_TYPE_DCLK1);

static int amd_smi_generic_clk_freq_setter(const LikwidDevice_t device, const char *value, amdsmi_clk_type_t clk_type, bool set_minimum) {
    if (device->type != DEVICE_TYPE_AMD_GPU) {
        return -EINVAL;
    }
    if (clk_type < AMDSMI_CLK_TYPE_FIRST || clk_type > AMDSMI_CLK_TYPE__MAX) {
        return -EINVAL;
    }

    uint64_t freq = 0;
    amdsmi_clk_limit_type_t set_type = (amdsmi_clk_limit_type_t)AMDSMI_FREQ_IND_MAX;
    if (set_minimum) {
        set_type = (amdsmi_clk_limit_type_t)AMDSMI_FREQ_IND_MIN;
    }

    int err = likwid_sysft_string_to_uint64(value, &freq);
    if (err < 0) {
        return err;
    }

    amdsmi_processor_handle handle;
    err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }

    ASMI_CALL(amdsmi_set_gpu_clk_limit, handle, clk_type, set_type, freq);
    return 0;
}

#define CLOCK_FREQ_SETTER_GEN(name, type) \
static int amd_smi_##name##_min_clk_freq_setter(const LikwidDevice_t device, const char *value) { \
    return amd_smi_generic_clk_freq_setter(device, value, (type), 0); \
}\
static int amd_smi_##name##_max_clk_freq_setter(const LikwidDevice_t device, const char *value) { \
    return amd_smi_generic_clk_freq_setter(device, value, (type), 1); \
}

CLOCK_FREQ_SETTER_GEN(sys, AMDSMI_CLK_TYPE_SYS);
CLOCK_FREQ_SETTER_GEN(mem, AMDSMI_CLK_TYPE_MEM);
CLOCK_FREQ_SETTER_GEN(gfx, AMDSMI_CLK_TYPE_GFX);
CLOCK_FREQ_SETTER_GEN(df, AMDSMI_CLK_TYPE_DF);
CLOCK_FREQ_SETTER_GEN(dcef, AMDSMI_CLK_TYPE_DCEF);
CLOCK_FREQ_SETTER_GEN(soc, AMDSMI_CLK_TYPE_SOC);
CLOCK_FREQ_SETTER_GEN(pcie, AMDSMI_CLK_TYPE_PCIE);
CLOCK_FREQ_SETTER_GEN(video0, AMDSMI_CLK_TYPE_VCLK0);
CLOCK_FREQ_SETTER_GEN(video1, AMDSMI_CLK_TYPE_VCLK1);
CLOCK_FREQ_SETTER_GEN(display0, AMDSMI_CLK_TYPE_DCLK0);
CLOCK_FREQ_SETTER_GEN(display1, AMDSMI_CLK_TYPE_DCLK1);


static int amd_smi_soc_pstate_current_getter(const LikwidDevice_t device, char **value) {
    if (device->type != DEVICE_TYPE_AMD_GPU) {
        return -EINVAL;
    }
    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }
    amdsmi_dpm_policy_t policy;
    ASMI_CALL(amdsmi_get_soc_pstate, handle, &policy);

    amdsmi_dpm_policy_entry_t *pol_entry = &policy.policies[policy.current];
    *value = lw_asprintf("%ld(%s)", pol_entry->policy_id, pol_entry->policy_description);
    return 0;
}

static int amd_smi_soc_pstate_avail_getter(const LikwidDevice_t device, char **value) {
    if (device->type != DEVICE_TYPE_AMD_GPU) {
        return -EINVAL;
    }
    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }
    amdsmi_dpm_policy_t policy;
    ASMI_CALL(amdsmi_get_soc_pstate, handle, &policy);

    bstring sup_pols = NULL;
    if (policy.num_supported > 0) {
        amdsmi_dpm_policy_entry_t *pol_entry = &policy.policies[0];
        sup_pols = bformat("%ld", pol_entry->policy_id);
    }
    for (uint32_t i = 1; i < policy.num_supported; i++) {
        amdsmi_dpm_policy_entry_t *pol_entry = &policy.policies[i];
        bconchar(sup_pols, ' ');
        bstring tmp = bformat("%ld(%s)", pol_entry->policy_id, pol_entry->policy_description);
        bconcat(sup_pols, tmp);
        bdestroy(tmp);
    }
    int ret = likwid_sysft_copystr(bdata(sup_pols), value);
    if (sup_pols) bdestroy(sup_pols);
    return ret;
}

static int amd_smi_soc_pstate_setter(const LikwidDevice_t device, const char *value) {
    if (device->type != DEVICE_TYPE_AMD_GPU) {
        return -EINVAL;
    }
    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }
    amdsmi_dpm_policy_t policy;
    ASMI_CALL(amdsmi_get_soc_pstate, handle, &policy);

    uint32_t inPstate;
    err = likwid_sysft_string_to_uint64(value, (uint64_t*)&inPstate);
    if (err) {
        return err;
    }

    int index = -1;
    for (uint32_t i = 0; i < policy.num_supported; i++) {
        if (inPstate == policy.policies[i].policy_id) {
            index = i;
            break;
        }
    }
    if (index >= 0) {
        ASMI_CALL(amdsmi_set_soc_pstate, handle, inPstate);
        return 0;
    }
    return -ENODEV;
}

typedef struct {
    amdsmi_dev_perf_level_t level;
    char* name;
} likwid_sysft_amdsmi_perf_level_s;

# define LIKWID_SYSFT_AMDSMI_PERF_LEVELS 9
static likwid_sysft_amdsmi_perf_level_s likwid_sysft_amdsmi_perf_level[LIKWID_SYSFT_AMDSMI_PERF_LEVELS] = {
    {AMDSMI_DEV_PERF_LEVEL_AUTO, "auto"},
    {AMDSMI_DEV_PERF_LEVEL_LOW , "low"},
    {AMDSMI_DEV_PERF_LEVEL_HIGH , "high"},
    {AMDSMI_DEV_PERF_LEVEL_MANUAL , "manual"},
    {AMDSMI_DEV_PERF_LEVEL_STABLE_STD , "stable_profiling"},
    {AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK , "stable_peak"},
    {AMDSMI_DEV_PERF_LEVEL_STABLE_MIN_MCLK , "stable_min_mclk"},
    {AMDSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK , "stable_min_sclk"},
    {AMDSMI_DEV_PERF_LEVEL_DETERMINISM, "determinism"}
};

static int amd_smi_perf_level_getter(const LikwidDevice_t device, char **value) {
    if (device->type != DEVICE_TYPE_AMD_GPU) {
        return -EINVAL;
    }
    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }

    amdsmi_dev_perf_level_t perf_level = AMDSMI_DEV_PERF_LEVEL_UNKNOWN;
    ASMI_CALL(amdsmi_get_gpu_perf_level, handle, &perf_level);

    for (int i = 0; i <= LIKWID_SYSFT_AMDSMI_PERF_LEVELS; i++) {
        if (perf_level == likwid_sysft_amdsmi_perf_level[i].level) {
            return likwid_sysft_copystr(likwid_sysft_amdsmi_perf_level[i].name, value);
        }
    }
    return likwid_sysft_copystr("unknown", value);
}

static int amd_smi_perf_level_setter(const LikwidDevice_t device, const char *value) {
    if (device->type != DEVICE_TYPE_AMD_GPU) {
        return -EINVAL;
    }
    amdsmi_dev_perf_level_t perf_level = AMDSMI_DEV_PERF_LEVEL_UNKNOWN;
    for (int i = 0; i <= LIKWID_SYSFT_AMDSMI_PERF_LEVELS; i++) {
        if (strncmp(value, likwid_sysft_amdsmi_perf_level[i].name, strlen(likwid_sysft_amdsmi_perf_level[i].name)) == 0) {
            perf_level = likwid_sysft_amdsmi_perf_level[i].level;
            break;
        }
    }

    if ((perf_level < AMDSMI_DEV_PERF_LEVEL_FIRST) || (perf_level > AMDSMI_DEV_PERF_LEVEL_LAST)) {
        return -EINVAL;
    }

    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }

    ASMI_CALL(amdsmi_set_gpu_perf_level, handle, perf_level);

    return 0;
}




static int amd_smi_gpu_override_level_getter(const LikwidDevice_t device, char **value) {
    if (device->type != DEVICE_TYPE_AMD_GPU) {
        return -EINVAL;
    }
    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }
    uint32_t level = 0;
    ASMI_CALL(amdsmi_get_gpu_overdrive_level, handle, &level);

    return likwid_sysft_uint64_to_string(level, value);
}

static int amd_smi_gpu_override_level_setter(const LikwidDevice_t device, const char *value) {
    if (device->type != DEVICE_TYPE_AMD_GPU) {
        return -EINVAL;
    }
    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }

    uint32_t level = 0;
    err = likwid_sysft_string_to_uint64(value, (uint64_t*)&level);
    if (err) {
        return err;
    }

    ASMI_CALL(amdsmi_set_gpu_overdrive_level, handle, level);

    return 0;
}


static int amd_smi_gpu_mem_override_level_getter(const LikwidDevice_t device, char **value) {
    if (device->type != DEVICE_TYPE_AMD_GPU) {
        return -EINVAL;
    }
    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }
    uint32_t level = 0;
    ASMI_CALL(amdsmi_get_gpu_mem_overdrive_level, handle, &level);

    return likwid_sysft_uint64_to_string(level, value);
}

typedef enum {
    LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_CURRENT = 0,
    LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_DEFAULT,
    LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_DPM,
    LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_MIN,
    LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_MAX,
} likwid_sysft_amd_smi_get_power_cap_field;

static int amd_smi_generic_power_cap_getter(const LikwidDevice_t device, char **value, uint32_t sensor_ind, likwid_sysft_amd_smi_get_power_cap_field field) {
    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        printf("Failed to get likwid device\n");
        return err;
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, "AMD_SMI amdsmi_get_power_cap_info failed, trying sysfs for %04x:%02x:%02x.%01x", device->id.pci.pci_domain, device->id.pci.pci_bus, device->id.pci.pci_dev, device->id.pci.pci_func);

    amdsmi_power_cap_info_t pcap_info;
    ASMI_CALL(amdsmi_get_power_cap_info, handle, sensor_ind, &pcap_info);
    switch (field) {
        case LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_CURRENT:
            return likwid_sysft_uint64_to_string(pcap_info.power_cap, value);
            break;
        case LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_DEFAULT:
            return likwid_sysft_uint64_to_string(pcap_info.default_power_cap, value);
            break;
        case LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_DPM:
            return likwid_sysft_uint64_to_string(pcap_info.dpm_cap, value);
            break;
        case LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_MIN:
            return likwid_sysft_uint64_to_string(pcap_info.min_power_cap, value);
            break;
        case LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_MAX:
            return likwid_sysft_uint64_to_string(pcap_info.max_power_cap, value);
            break;
        default:
    }
    return -EINVAL;
}

static int amd_smi_generic_power_cap_sysfs_getter(const LikwidDevice_t device, char **value, uint32_t sensor_ind, likwid_sysft_amd_smi_get_power_cap_field field) {

    DIR *dp = NULL;
    struct dirent *ep = NULL;
    dp = opendir("/sys/class/hwmon");
    char bdf[15];
    bstring hwmon_amdgpu = NULL;
    snprintf(bdf, 14, "%04x:%02x:%02x.%01x", device->id.pci.pci_domain, device->id.pci.pci_bus, device->id.pci.pci_dev, device->id.pci.pci_func);
    DEBUG_PRINT(DEBUGLEV_DEVELOP, "AMD_SMI amdsmi_get_power_cap_info failed, trying sysfs for %s", bdf);
    if (dp != NULL)
    {
        while ((ep = readdir(dp)))
        {
            bstring hwmon_device = bformat("/sys/class/hwmon/%s/device", ep->d_name);
            char device_link[2048];
            char* link_dev = bdata(hwmon_device);
            int link_len = readlink(link_dev, device_link, 2048);
            device_link[link_len] = '\0';
            bdestroy(hwmon_device);
            char* last_slash = strrchr(device_link, '/');
            if (last_slash) {
                if (strncmp(last_slash+1, bdf, 12) == 0) {
                    hwmon_amdgpu = bformat("/sys/class/hwmon/%s", ep->d_name);
                    break;
                }
            }
        }
        closedir(dp);
    }
    if (!hwmon_amdgpu) {
        return -EINVAL;
    }
    int ret = 0;
    bstring f;
    bstring data;
    switch (field) {
        case LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_CURRENT:
            f = bformat("%s/power%d_cap", bdata(hwmon_amdgpu), sensor_ind+1);
            data = read_file(bdata(f));
            if (data) {
                ret = likwid_sysft_copystr(bdata(data), value);
                bdestroy(data);
            }
            bdestroy(f);
            break;
        case LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_DEFAULT:
            f = bformat("%s/power%d_cap_default", bdata(hwmon_amdgpu), sensor_ind+1);
            data = read_file(bdata(f));
            if (data) {
                ret = likwid_sysft_copystr(bdata(data), value);
                bdestroy(data);
            }
            bdestroy(f);
            break;
        case LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_DPM:
            ret = -EINVAL;
            break;
        case LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_MIN:
            f = bformat("%s/power%d_cap_min", bdata(hwmon_amdgpu), sensor_ind+1);
            data = read_file(bdata(f));
            if (data) {
                ret = likwid_sysft_copystr(bdata(data), value);
                bdestroy(data);
            }
            bdestroy(f);
            break;

        case LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_MAX:
            f = bformat("%s/power%d_cap_max", bdata(hwmon_amdgpu), sensor_ind+1);
            data = read_file(bdata(f));
            if (data) {
                ret = likwid_sysft_copystr(bdata(data), value);
                bdestroy(data);
            }
            bdestroy(f);
            break;
        default:
    }
    bdestroy(hwmon_amdgpu);
    return ret;
}

#define LW_SYSFT_POWER_CAP_GET_GEN_SENSOR_FIELD(id, name, field) \
static int amd_smi_power_cap_sensor_##id##_##name##_getter(const LikwidDevice_t device, char **value) { \
    int ret = amd_smi_generic_power_cap_getter(device, value, (id), field); \
    if (ret != 0) { \
        return amd_smi_generic_power_cap_sysfs_getter(device, value, (id), field); \
    } \
    return 0; \
}

#define LW_SYSFT_POWER_CAP_GET_GEN_SENSOR(id) \
    LW_SYSFT_POWER_CAP_GET_GEN_SENSOR_FIELD(id, current, LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_CURRENT); \
    LW_SYSFT_POWER_CAP_GET_GEN_SENSOR_FIELD(id, default, LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_DEFAULT); \
    LW_SYSFT_POWER_CAP_GET_GEN_SENSOR_FIELD(id, dpm, LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_DPM); \
    LW_SYSFT_POWER_CAP_GET_GEN_SENSOR_FIELD(id, min, LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_MIN); \
    LW_SYSFT_POWER_CAP_GET_GEN_SENSOR_FIELD(id, max, LIKWID_SYSFT_AMDSMI_GET_POWER_CAP_MAX); \

LW_SYSFT_POWER_CAP_GET_GEN_SENSOR(0);
/*LW_SYSFT_POWER_CAP_GET_GEN_SENSOR(1);*/
/*LW_SYSFT_POWER_CAP_GET_GEN_SENSOR(2);*/
/*LW_SYSFT_POWER_CAP_GET_GEN_SENSOR(3);*/
/*LW_SYSFT_POWER_CAP_GET_GEN_SENSOR(4);*/
/*LW_SYSFT_POWER_CAP_GET_GEN_SENSOR(5);*/
/*LW_SYSFT_POWER_CAP_GET_GEN_SENSOR(6);*/
/*LW_SYSFT_POWER_CAP_GET_GEN_SENSOR(7);*/
/*LW_SYSFT_POWER_CAP_GET_GEN_SENSOR(8);*/
/*LW_SYSFT_POWER_CAP_GET_GEN_SENSOR(9);*/

static int amd_smi_generic_power_cap_setter(const LikwidDevice_t device, const char *value, uint32_t sensor_ind) {

    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }

    uint64_t pcap = 0;
    err = likwid_sysft_string_to_uint64(value, &pcap);
    if (err) {
        return err;
    }

    ASMI_CALL(amdsmi_set_power_cap, handle, sensor_ind, pcap);

    return 0;
}

#define LW_SYSFT_POWER_CAP_SET_GEN_SENSOR(id) \
static int amd_smi_power_cap_sensor_##id##_setter(const LikwidDevice_t device, const char *value) { \
    return amd_smi_generic_power_cap_setter(device, value, (id)); \
}

LW_SYSFT_POWER_CAP_SET_GEN_SENSOR(0);
/*LW_SYSFT_POWER_CAP_SET_GEN_SENSOR(1);*/
/*LW_SYSFT_POWER_CAP_SET_GEN_SENSOR(2);*/
/*LW_SYSFT_POWER_CAP_SET_GEN_SENSOR(3);*/
/*LW_SYSFT_POWER_CAP_SET_GEN_SENSOR(4);*/
/*LW_SYSFT_POWER_CAP_SET_GEN_SENSOR(5);*/
/*LW_SYSFT_POWER_CAP_SET_GEN_SENSOR(6);*/
/*LW_SYSFT_POWER_CAP_SET_GEN_SENSOR(7);*/
/*LW_SYSFT_POWER_CAP_SET_GEN_SENSOR(8);*/
/*LW_SYSFT_POWER_CAP_SET_GEN_SENSOR(9);*/

static int amd_smi_perf_determinism_mode_setter(const LikwidDevice_t device, const char *value) {

    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }

    uint64_t freq = 0;
    err = likwid_sysft_string_to_uint64(value, &freq);
    if (err) {
        return err;
    }

    ASMI_CALL(amdsmi_set_gpu_perf_determinism_mode, handle, freq);

    return 0;
}




static int amd_smi_socket_power_getter(const LikwidDevice_t device, char **value) {
    if (device->type != DEVICE_TYPE_AMD_GPU) {
        return -EINVAL;
    }

    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }
    uint32_t power = 0;
    ASMI_CALL(amdsmi_get_cpu_socket_power, handle, &power);
    return likwid_sysft_uint64_to_string(power, value);
}

static int amd_smi_socket_power_cap_getter(const LikwidDevice_t device, char **value) {
    if (device->type != DEVICE_TYPE_AMD_GPU) {
        return -EINVAL;
    }

    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }
    uint32_t power = 0;
    ASMI_CALL(amdsmi_get_cpu_socket_power_cap, handle, &power);
    return likwid_sysft_uint64_to_string(power, value);
}

static int amd_smi_socket_power_cap_setter(const LikwidDevice_t device, const char *value) {
    if (device->type != DEVICE_TYPE_AMD_GPU) {
        return -EINVAL;
    }

    uint64_t pcap64 = 0;
    uint32_t pcap = 0;
    int err = likwid_sysft_string_to_uint64(value, &pcap64);
    if (err < 0) {
        return err;
    }
    pcap = (uint32_t)pcap64;

    amdsmi_processor_handle handle;
    err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }

    ASMI_CALL(amdsmi_set_cpu_socket_power_cap, handle, pcap);
    return 0;
}

static int amd_smi_socket_power_cap_max_getter(const LikwidDevice_t device, char **value) {
    if (device->type != DEVICE_TYPE_AMD_GPU) {
        return -EINVAL;
    }

    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }

    uint32_t power = 0;
    ASMI_CALL(amdsmi_get_cpu_socket_power_cap_max, handle, &power);
    return likwid_sysft_uint64_to_string(power, value);
}


#define LW_SYSFT_FAN_SPEED_GET_GEN_RANGE(limit) \
static int amd_smi_gpu_voltage_##limit##_getter(const LikwidDevice_t device, char **value) { \
    if (device->type != DEVICE_TYPE_AMD_GPU) { \
        return -EINVAL; \
    } \
    amdsmi_processor_handle handle; \
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle); \
    if (err) { \
        return err; \
    } \
    amdsmi_od_volt_freq_data_t data; \
    ASMI_CALL(amdsmi_get_gpu_od_volt_info, handle, &data); \
    bstring tmpstr = bformat("%ld - %ld", data.limit.lower_bound, data.limit.upper_bound); \
    err = likwid_sysft_copystr(bdata(tmpstr), value); \
    bdestroy(tmpstr); \
    return err; \
}

LW_SYSFT_FAN_SPEED_GET_GEN_RANGE(curr_sclk_range);
LW_SYSFT_FAN_SPEED_GET_GEN_RANGE(curr_mclk_range);
LW_SYSFT_FAN_SPEED_GET_GEN_RANGE(sclk_freq_limits);
LW_SYSFT_FAN_SPEED_GET_GEN_RANGE(mclk_freq_limits);

static int amd_smi_gpu_voltage_volt_curve_getter(const LikwidDevice_t device, char **value) {
    if (device->type != DEVICE_TYPE_AMD_GPU) {
        return -EINVAL;
    }

    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }
    amdsmi_od_volt_freq_data_t data;
    ASMI_CALL(amdsmi_get_gpu_od_volt_info, handle, &data);

    int ret = 0;
    bstring tmpstr;
    struct bstrList *tmpList;
    struct tagbstring bcomma = bsStatic(",");

    amdsmi_od_volt_curve_t* curve = &data.curve;
    if (curve == NULL) return -EINVAL;
    tmpList = bstrListCreate();
    for (uint32_t i = 0; i < MIN(data.num_regions, AMDSMI_NUM_VOLTAGE_CURVE_POINTS); i++) {
        amdsmi_od_vddc_point_t c = curve->vc_points[i];
        tmpstr = bformat("%lu Hz: %lu mV", c.frequency, c.voltage);
        bstrListAdd(tmpList, tmpstr);
        bdestroy(tmpstr);
    }
    tmpstr = bjoin(tmpList, &bcomma);
    ret = likwid_sysft_copystr(bdata(tmpstr), value);
    bdestroy(tmpstr);
    bstrListDestroy(tmpList);
    return ret;
}


#define LW_SYSFT_POWER_INFO_GEN(field) \
static int amd_smi_power_info_##field##_getter(const LikwidDevice_t device, char **value) { \
    if (device->type != DEVICE_TYPE_AMD_GPU) { \
        return -EINVAL; \
    } \
    amdsmi_processor_handle handle; \
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle); \
    if (err) { \
        return err; \
    } \
    amdsmi_power_info_t data; \
    ASMI_CALL(amdsmi_get_power_info, handle, &data); \
    uint32_t not_supported_val = UINT16_MAX; \
    if (data.field == not_supported_val) { \
        return -ENOTSUP; \
    } \
    return likwid_sysft_uint64_to_string(data.field, value); \
}

LW_SYSFT_POWER_INFO_GEN(socket_power);
LW_SYSFT_POWER_INFO_GEN(current_socket_power);
LW_SYSFT_POWER_INFO_GEN(average_socket_power);
LW_SYSFT_POWER_INFO_GEN(soc_voltage);
LW_SYSFT_POWER_INFO_GEN(gfx_voltage);
LW_SYSFT_POWER_INFO_GEN(mem_voltage);
LW_SYSFT_POWER_INFO_GEN(power_limit);



static int amd_smi_gpu_clean_local_data_setter(const LikwidDevice_t device, const char *value) {
    if (device->type != DEVICE_TYPE_AMD_GPU) {
        return -EINVAL;
    }

    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        return err;
    }
    uint64_t flag = 0;
    err = likwid_sysft_string_to_uint64(value, &flag);
    if (err < 0) {
        return err;
    }
    if (flag) {
        ASMI_CALL(amdsmi_clean_gpu_local_data, handle);
    }
    return 0;
}


#define LW_SYSFT_FAN_SPEED_GET_GEN_SENSOR(id) \
static int amd_smi_fan_speed_##id##_getter(const LikwidDevice_t device, char **value) { \
    if (device->type != DEVICE_TYPE_AMD_GPU) { \
        return -EINVAL; \
    } \
    amdsmi_processor_handle handle; \
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle); \
    if (err) { \
        return err; \
    } \
    int64_t speed = 0; \
    ASMI_CALL(amdsmi_get_gpu_fan_rpms, handle, (id), &speed); \
    return likwid_sysft_uint64_to_string(speed, value); \
}


LW_SYSFT_FAN_SPEED_GET_GEN_SENSOR(0);

#define LW_SYSFT_FAN_SPEED_MAX_GET_GEN_SENSOR(id) \
static int amd_smi_fan_speed_max_##id##_getter(const LikwidDevice_t device, char **value) { \
    if (device->type != DEVICE_TYPE_AMD_GPU) { \
        return -EINVAL; \
    } \
    amdsmi_processor_handle handle; \
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle); \
    if (err) { \
        return err; \
    } \
    uint64_t speed = 0; \
    ASMI_CALL(amdsmi_get_gpu_fan_speed_max, handle, (id), &speed); \
    return likwid_sysft_uint64_to_string(speed, value); \
}


LW_SYSFT_FAN_SPEED_MAX_GET_GEN_SENSOR(0);



#define LW_SYSFT_FAN_SPEED_SET_GEN_SENSOR(id) \
static int amd_smi_fan_speed_##id##_setter(const LikwidDevice_t device, const char *value) { \
    if (device->type != DEVICE_TYPE_AMD_GPU) { \
        return -EINVAL; \
    } \
    amdsmi_processor_handle handle; \
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle); \
    if (err) { \
        return err; \
    } \
    int64_t speed = 0; \
    err = likwid_sysft_string_to_uint64(value, (uint64_t*)&speed); \
    if (err < 0) { \
        return err; \
    } \
    ASMI_CALL(amdsmi_set_gpu_fan_speed, handle, (id), speed); \
    return 0; \
}

LW_SYSFT_FAN_SPEED_SET_GEN_SENSOR(0);


#define LW_SYSFT_GPU_VOLT_METRIC_GEN(sensor, metric) \
static int amd_smi_generic_gpu_volt_metric_##sensor##_##metric##_getter(const LikwidDevice_t device, char **value) { \
    if (device->type != DEVICE_TYPE_AMD_GPU) { \
        return -EINVAL; \
    } \
    amdsmi_processor_handle handle; \
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle); \
    if (err) { \
        return err; \
    } \
    int64_t voltage = 0; \
    ASMI_CALL(amdsmi_get_gpu_volt_metric, handle, sensor, metric, &voltage); \
    return likwid_sysft_uint64_to_string(voltage, value); \
}

#define LW_SYSFT_GPU_VOLT_METRIC_SENSOR_GEN(sensor)\
    LW_SYSFT_GPU_VOLT_METRIC_GEN(sensor, AMDSMI_VOLT_CURRENT); \
    LW_SYSFT_GPU_VOLT_METRIC_GEN(sensor, AMDSMI_VOLT_MAX); \
    LW_SYSFT_GPU_VOLT_METRIC_GEN(sensor, AMDSMI_VOLT_MAX_CRIT); \
    LW_SYSFT_GPU_VOLT_METRIC_GEN(sensor, AMDSMI_VOLT_MIN); \
    LW_SYSFT_GPU_VOLT_METRIC_GEN(sensor, AMDSMI_VOLT_MIN_CRIT); \
    LW_SYSFT_GPU_VOLT_METRIC_GEN(sensor, AMDSMI_VOLT_AVERAGE); \
    LW_SYSFT_GPU_VOLT_METRIC_GEN(sensor, AMDSMI_VOLT_LOWEST); \
    LW_SYSFT_GPU_VOLT_METRIC_GEN(sensor, AMDSMI_VOLT_HIGHEST);

LW_SYSFT_GPU_VOLT_METRIC_SENSOR_GEN(AMDSMI_VOLT_TYPE_VDDGFX);
LW_SYSFT_GPU_VOLT_METRIC_SENSOR_GEN(AMDSMI_VOLT_TYPE_VDDBOARD);


#define LW_SYSFT_GPU_ACTIVITY_GEN(field) \
static int amd_smi_gpu_##field##_getter(const LikwidDevice_t device, char **value) { \
    if (device->type != DEVICE_TYPE_AMD_GPU) { \
        return -EINVAL; \
    } \
    amdsmi_processor_handle handle; \
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle); \
    if (err) { \
        return err; \
    } \
    amdsmi_engine_usage_t data; \
    ASMI_CALL(amdsmi_get_gpu_activity, handle, &data); \
    return likwid_sysft_uint64_to_string(data.field, value); \
}

LW_SYSFT_GPU_ACTIVITY_GEN(gfx_activity);
LW_SYSFT_GPU_ACTIVITY_GEN(umc_activity);
LW_SYSFT_GPU_ACTIVITY_GEN(mm_activity);

#define LW_SYSFT_TEMP_METRIC_GEN(sensor, field) \
static int amd_smi_temp_metric_##sensor##_##field##_getter(const LikwidDevice_t device, char **value) { \
    amdsmi_processor_handle handle; \
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle); \
    if (err) { \
        printf("Failed to get likwid device\n"); \
        return err; \
    } \
    amdsmi_status_t status = amdsmi_get_temp_metric_ptr(handle, sensor, field, NULL); \
    if (status == AMDSMI_STATUS_NOT_SUPPORTED) { \
        return -ENOTSUP; \
    } \
    int64_t data = 0; \
    ASMI_CALL(amdsmi_get_temp_metric, handle, sensor, field, &data); \
    return likwid_sysft_uint64_to_string(data, value); \
}



#define LW_SYSFT_TEMP_METRIC_GEN_TYPE(sensor) \
    LW_SYSFT_TEMP_METRIC_GEN(sensor, AMDSMI_TEMP_CURRENT); \
    LW_SYSFT_TEMP_METRIC_GEN(sensor, AMDSMI_TEMP_MAX); \
    LW_SYSFT_TEMP_METRIC_GEN(sensor, AMDSMI_TEMP_MIN); \
    LW_SYSFT_TEMP_METRIC_GEN(sensor, AMDSMI_TEMP_MAX_HYST); \
    LW_SYSFT_TEMP_METRIC_GEN(sensor, AMDSMI_TEMP_MIN_HYST); \
    LW_SYSFT_TEMP_METRIC_GEN(sensor, AMDSMI_TEMP_CRITICAL); \
    LW_SYSFT_TEMP_METRIC_GEN(sensor, AMDSMI_TEMP_CRITICAL_HYST); \
    LW_SYSFT_TEMP_METRIC_GEN(sensor, AMDSMI_TEMP_EMERGENCY); \
    LW_SYSFT_TEMP_METRIC_GEN(sensor, AMDSMI_TEMP_EMERGENCY_HYST); \
    LW_SYSFT_TEMP_METRIC_GEN(sensor, AMDSMI_TEMP_CRIT_MIN); \
    LW_SYSFT_TEMP_METRIC_GEN(sensor, AMDSMI_TEMP_CRIT_MIN_HYST); \
    LW_SYSFT_TEMP_METRIC_GEN(sensor, AMDSMI_TEMP_OFFSET); \
    LW_SYSFT_TEMP_METRIC_GEN(sensor, AMDSMI_TEMP_LOWEST); \
    LW_SYSFT_TEMP_METRIC_GEN(sensor, AMDSMI_TEMP_HIGHEST); \
    LW_SYSFT_TEMP_METRIC_GEN(sensor, AMDSMI_TEMP_SHUTDOWN);


LW_SYSFT_TEMP_METRIC_GEN_TYPE(AMDSMI_TEMPERATURE_TYPE_EDGE);
LW_SYSFT_TEMP_METRIC_GEN_TYPE(AMDSMI_TEMPERATURE_TYPE_HOTSPOT);
LW_SYSFT_TEMP_METRIC_GEN_TYPE(AMDSMI_TEMPERATURE_TYPE_VRAM);
LW_SYSFT_TEMP_METRIC_GEN_TYPE(AMDSMI_TEMPERATURE_TYPE_HBM_0);
LW_SYSFT_TEMP_METRIC_GEN_TYPE(AMDSMI_TEMPERATURE_TYPE_HBM_1);
LW_SYSFT_TEMP_METRIC_GEN_TYPE(AMDSMI_TEMPERATURE_TYPE_HBM_2);
LW_SYSFT_TEMP_METRIC_GEN_TYPE(AMDSMI_TEMPERATURE_TYPE_HBM_3);
LW_SYSFT_TEMP_METRIC_GEN_TYPE(AMDSMI_TEMPERATURE_TYPE_PLX);

static int amd_smi_gpu_pwr_manage_getter(const LikwidDevice_t device, char **value) {
    amdsmi_processor_handle handle;
    int err = lw_device_to_amdsmi_gpu_handle(device, &handle);
    if (err) {
        printf("Failed to get likwid device\n");
        return err;
    }
    bool data = false;
    ASMI_CALL(amdsmi_is_gpu_power_management_enabled, handle, &data);
    return likwid_sysft_copystr((data == true ? "true" : "false"), value);
}

/*#define LW_SYSFT_GPU_METRICS_GEN(field) \*/
/*static int amd_smi_gpu_metrics_##field##_getter(const LikwidDevice_t device, char **value) { \*/
/*    if (device->type != DEVICE_TYPE_AMD_GPU) { \*/
/*        return -EINVAL; \*/
/*    } \*/
/*    amdsmi_processor_handle handle; \*/
/*    int err = lw_device_to_amdsmi_gpu_handle(device, &handle); \*/
/*    if (err) { \*/
/*        return err; \*/
/*    } \*/
/*    amdsmi_gpu_metrics_t metrics; \*/
/*    ASMI_CALL(amdsmi_get_gpu_metrics_info, handle, &metrics); \*/
/*    return likwid_sysft_uint64_to_string(metrics.field, value); \*/
/*}*/

/*LW_SYSFT_GPU_METRICS_GEN(temperature_edge);*/
/*LW_SYSFT_GPU_METRICS_GEN(temperature_hotspot);*/
/*LW_SYSFT_GPU_METRICS_GEN(temperature_mem);*/
/*LW_SYSFT_GPU_METRICS_GEN(temperature_vrgfx);*/
/*LW_SYSFT_GPU_METRICS_GEN(temperature_vrsoc);*/
/*LW_SYSFT_GPU_METRICS_GEN(temperature_vrmem);*/
/*LW_SYSFT_GPU_METRICS_GEN(average_gfx_activity);*/
/*LW_SYSFT_GPU_METRICS_GEN(average_umc_activity);*/
/*LW_SYSFT_GPU_METRICS_GEN(average_mm_activity);*/
/*LW_SYSFT_GPU_METRICS_GEN(average_socket_power);*/
/*LW_SYSFT_GPU_METRICS_GEN(energy_accumulator);*/
/*LW_SYSFT_GPU_METRICS_GEN(average_gfxclk_frequency);*/
/*LW_SYSFT_GPU_METRICS_GEN(average_socclk_frequency);*/
/*LW_SYSFT_GPU_METRICS_GEN(average_uclk_frequency);*/
/*LW_SYSFT_GPU_METRICS_GEN(average_vclk0_frequency);*/
/*LW_SYSFT_GPU_METRICS_GEN(average_vclk1_frequency);*/
/*LW_SYSFT_GPU_METRICS_GEN(average_dclk0_frequency);*/
/*LW_SYSFT_GPU_METRICS_GEN(average_dclk1_frequency);*/
/*LW_SYSFT_GPU_METRICS_GEN(current_gfxclk);*/
/*LW_SYSFT_GPU_METRICS_GEN(current_socclk);*/
/*LW_SYSFT_GPU_METRICS_GEN(current_uclk);*/
/*LW_SYSFT_GPU_METRICS_GEN(current_vclk0);*/
/*LW_SYSFT_GPU_METRICS_GEN(current_vclk1);*/
/*LW_SYSFT_GPU_METRICS_GEN(current_dclk0);*/
/*LW_SYSFT_GPU_METRICS_GEN(current_dclk1);*/
/*LW_SYSFT_GPU_METRICS_GEN(throttle_status);*/
/*LW_SYSFT_GPU_METRICS_GEN(current_fan_speed);*/
/*LW_SYSFT_GPU_METRICS_GEN(pcie_link_width);*/
/*LW_SYSFT_GPU_METRICS_GEN(pcie_link_speed);*/
/*LW_SYSFT_GPU_METRICS_GEN(gfx_activity_acc);*/
/*LW_SYSFT_GPU_METRICS_GEN(mem_activity_acc);*/
/*LW_SYSFT_GPU_METRICS_GEN(voltage_soc);*/
/*LW_SYSFT_GPU_METRICS_GEN(voltage_gfx);*/
/*LW_SYSFT_GPU_METRICS_GEN(voltage_mem);*/
/*LW_SYSFT_GPU_METRICS_GEN(indep_throttle_status);*/
/*LW_SYSFT_GPU_METRICS_GEN(current_socket_power);*/
/*LW_SYSFT_GPU_METRICS_GEN(gfxclk_lock_status);*/
/*LW_SYSFT_GPU_METRICS_GEN(xgmi_link_width);*/
/*LW_SYSFT_GPU_METRICS_GEN(xgmi_link_speed);*/
/*LW_SYSFT_GPU_METRICS_GEN(pcie_bandwidth_acc);*/
/*LW_SYSFT_GPU_METRICS_GEN(pcie_bandwidth_inst);*/
/*LW_SYSFT_GPU_METRICS_GEN(pcie_l0_to_recov_count_acc);*/
/*LW_SYSFT_GPU_METRICS_GEN(pcie_replay_count_acc);*/
/*LW_SYSFT_GPU_METRICS_GEN(pcie_replay_rover_count_acc);*/
/*LW_SYSFT_GPU_METRICS_GEN(pcie_nak_sent_count_acc);*/
/*LW_SYSFT_GPU_METRICS_GEN(pcie_nak_rcvd_count_acc);*/
/*LW_SYSFT_GPU_METRICS_GEN(pcie_lc_perf_other_end_recovery);*/
/*LW_SYSFT_GPU_METRICS_GEN(accumulation_counter);*/
/*LW_SYSFT_GPU_METRICS_GEN(prochot_residency_acc);*/
/*LW_SYSFT_GPU_METRICS_GEN(ppt_residency_acc);*/
/*LW_SYSFT_GPU_METRICS_GEN(socket_thm_residency_acc);*/
/*LW_SYSFT_GPU_METRICS_GEN(vr_thm_residency_acc);*/
/*LW_SYSFT_GPU_METRICS_GEN(hbm_thm_residency_acc);*/
/*LW_SYSFT_GPU_METRICS_GEN(num_partition);*/
/*LW_SYSFT_GPU_METRICS_GEN(vram_max_bandwidth);*/


static _SysFeature amd_smi_features[] = {
    {"device_count", "amdsmi", "Number of GPUs on node. Not all GPUs may be accessible.", amd_smi_device_count_getter, NULL, DEVICE_TYPE_NODE, NULL, NULL},
    {"sys_cur_clk_freq", "amdsmi", "Current frequency for system clock.", amd_smi_sys_current_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"sys_has_deep_sleep", "amdsmi", "Deep sleep frequency support for system clock.", amd_smi_sys_hasdeepsleep_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"sys_avail_freqs", "amdsmi", "List of supported frequencies for system clock.", amd_smi_sys_supported_freqs_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"sys_min_clk_freq", "amdsmi", "Minimal frequency for system clock", NULL, amd_smi_sys_min_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"sys_max_clk_freq", "amdsmi", "Maximal frequency for system clock", NULL, amd_smi_sys_max_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"mem_cur_clk_freq", "amdsmi", "Current frequency for memory clock.", amd_smi_mem_current_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"mem_has_deep_sleep", "amdsmi", "Deep sleep frequency support for memory clock.", amd_smi_mem_hasdeepsleep_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"mem_avail_freqs", "amdsmi", "List of supported frequencies for memory clock.", amd_smi_mem_supported_freqs_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"mem_min_clk_freq", "amdsmi", "Minimal frequency for memory clock", NULL, amd_smi_mem_min_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"mem_max_clk_freq", "amdsmi", "Maximal frequency for memory clock", NULL, amd_smi_mem_max_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"gfx_cur_clk_freq", "amdsmi", "Current frequency for GFX clock.", amd_smi_gfx_current_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"gfx_has_deep_sleep", "amdsmi", "Deep sleep frequency support for GFX clock.", amd_smi_gfx_hasdeepsleep_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"gfx_avail_freqs", "amdsmi", "List of supported frequencies for GFX clock.", amd_smi_gfx_supported_freqs_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"gfx_min_clk_freq", "amdsmi", "Minimal frequency for GFX clock", NULL, amd_smi_gfx_min_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"gfx_max_clk_freq", "amdsmi", "Maximal frequency for GFX clock", NULL, amd_smi_gfx_max_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"df_cur_clk_freq", "amdsmi", "Current frequency for DataFabric clock.", amd_smi_df_current_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"df_has_deep_sleep", "amdsmi", "Deep sleep frequency support for DataFabric clock.", amd_smi_df_hasdeepsleep_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"df_avail_freqs", "amdsmi", "List of supported frequencies for DataFabric clock.", amd_smi_df_supported_freqs_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"df_min_clk_freq", "amdsmi", "Minimal frequency for DataFabric clock", NULL, amd_smi_df_min_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"df_max_clk_freq", "amdsmi", "Maximal frequency for DataFabric clock", NULL, amd_smi_df_max_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"dcef_cur_clk_freq", "amdsmi", "Current frequency for Display Controller Engine Front clock.", amd_smi_dcef_current_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"dcef_has_deep_sleep", "amdsmi", "Deep sleep frequency support for Display Controller Engine Front clock.", amd_smi_dcef_hasdeepsleep_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"dcef_avail_freqs", "amdsmi", "List of supported frequencies for Display Controller Engine Front clock.", amd_smi_dcef_supported_freqs_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"dcef_min_clk_freq", "amdsmi", "Minimal frequency for Display Controller Engine Front clock", NULL, amd_smi_dcef_min_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"dcef_max_clk_freq", "amdsmi", "Maximal frequency for Display Controller Engine Front clock", NULL, amd_smi_dcef_max_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"soc_cur_clk_freq", "amdsmi", "Current frequency for System-on-Chip clock.", amd_smi_soc_current_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"soc_has_deep_sleep", "amdsmi", "Deep sleep frequency support for System-on-Chip clock.", amd_smi_soc_hasdeepsleep_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"soc_avail_freqs", "amdsmi", "List of supported frequencies for System-on-Chip clock.", amd_smi_soc_supported_freqs_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"soc_min_clk_freq", "amdsmi", "Minimal frequency for System-on-Chip clock", NULL, amd_smi_soc_min_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"soc_max_clk_freq", "amdsmi", "Maximal frequency for System-on-Chip clock", NULL, amd_smi_soc_max_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"pcie_cur_clk_freq", "amdsmi", "Current frequency for PCI Express clock.", amd_smi_pcie_current_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"pcie_has_deep_sleep", "amdsmi", "Deep sleep frequency support for PCI Express clock.", amd_smi_pcie_hasdeepsleep_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"pcie_avail_freqs", "amdsmi", "List of supported frequencies for PCI Express clock.", amd_smi_pcie_supported_freqs_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"pcie_min_clk_freq", "amdsmi", "Minimal frequency for PCI Express clock", NULL, amd_smi_pcie_min_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"pcie_max_clk_freq", "amdsmi", "Maximal frequency for PCI Express clock", NULL, amd_smi_pcie_max_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"video0_cur_clk_freq", "amdsmi", "Current frequency for video processing units 0 clock.", amd_smi_video0_current_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"video0_has_deep_sleep", "amdsmi", "Deep sleep frequency support for video processing units 0 clock.", amd_smi_video0_hasdeepsleep_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"video0_avail_freqs", "amdsmi", "List of supported frequencies for video processing units 0 clock.", amd_smi_video0_supported_freqs_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"video0_min_clk_freq", "amdsmi", "Minimal frequency for video processing units 0 clock", NULL, amd_smi_video0_min_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"video0_max_clk_freq", "amdsmi", "Maximal frequency for video processing units 0 clock", NULL, amd_smi_video0_max_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"video1_cur_clk_freq", "amdsmi", "Current frequency for video processing units 1 clock.", amd_smi_video1_current_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"video1_has_deep_sleep", "amdsmi", "Deep sleep frequency support for video processing units 1 clock.", amd_smi_video1_hasdeepsleep_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"video1_avail_freqs", "amdsmi", "List of supported frequencies for video processing units 1 clock.", amd_smi_video1_supported_freqs_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"video1_min_clk_freq", "amdsmi", "Minimal frequency for video processing units 1 clock", NULL, amd_smi_video1_min_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"video1_max_clk_freq", "amdsmi", "Maximal frequency for video processing units 1 clock", NULL, amd_smi_video1_max_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"display0_cur_clk_freq", "amdsmi", "Current frequency for timing signals for display 0 clock.", amd_smi_display0_current_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"display0_has_deep_sleep", "amdsmi", "Deep sleep frequency support for timing signals for display 0 clock.", amd_smi_display0_hasdeepsleep_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"display0_avail_freqs", "amdsmi", "List of supported frequencies for timing signals for display 0 clock.", amd_smi_display0_supported_freqs_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"display0_min_clk_freq", "amdsmi", "Minimal frequency for timing signals for display 0 clock", NULL, amd_smi_display0_min_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"display0_max_clk_freq", "amdsmi", "Maximal frequency for timing signals for display 0 clock", NULL, amd_smi_display0_max_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"display1_cur_clk_freq", "amdsmi", "Current frequency for timing signals for display 1 clock.", amd_smi_display1_current_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"display1_has_deep_sleep", "amdsmi", "Deep sleep frequency support for timing signals for display 1 clock.", amd_smi_display1_hasdeepsleep_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"display1_avail_freqs", "amdsmi", "List of supported frequencies for timing signals for display 1 clock.", amd_smi_display1_supported_freqs_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"display1_min_clk_freq", "amdsmi", "Minimal frequency for timing signals for display 1 clock", NULL, amd_smi_display1_min_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"display1_max_clk_freq", "amdsmi", "Maximal frequency for timing signals for display 1 clock", NULL, amd_smi_display1_max_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"determinism_mode", "amdsmi", "Enforce given GFXCLK frequency SoftMax limit per GPU. Changes amdsmi.perf_level to 'determinism'", NULL, amd_smi_perf_determinism_mode_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"socket_power", "amdsmi", "Socket power", amd_smi_socket_power_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"socket_power_limit", "amdsmi", "Power cap for socket", amd_smi_socket_power_cap_getter, amd_smi_socket_power_cap_setter, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"socket_power_limit_max", "amdsmi", "Maximum power cap for socket", amd_smi_socket_power_cap_max_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"soc_pstate", "amdsmi", "SoC PState policy", amd_smi_soc_pstate_current_getter, amd_smi_soc_pstate_setter, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"soc_pstate_avail", "amdsmi", "List of available SoC PState policies", amd_smi_soc_pstate_avail_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"perf_level", "amdsmi", "PowerPlay performance level", amd_smi_perf_level_getter, amd_smi_perf_level_setter, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"gpu_override", "amdsmi", "Overdrive rate (WARNING)", amd_smi_gpu_override_level_getter, amd_smi_gpu_override_level_setter, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"gpu_mem_override", "amdsmi", "Memory overdrive rate", amd_smi_gpu_mem_override_level_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"cur_sclk_range", "amdsmi", "Current SCLK frequency range", amd_smi_gpu_voltage_curr_sclk_range_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"avail_sclk_range", "amdsmi", "Available SCLK frequency range", amd_smi_gpu_voltage_sclk_freq_limits_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"cur_mclk_range", "amdsmi", "Current MCLK frequency range", amd_smi_gpu_voltage_curr_mclk_range_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"avail_mclk_range", "amdsmi", "Available MCLK frequency range", amd_smi_gpu_voltage_mclk_freq_limits_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"voltage_curve", "amdsmi", "Voltage curve", amd_smi_gpu_voltage_volt_curve_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"socket_power2", "amdsmi", "Socket power", amd_smi_power_info_socket_power_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "W"},
    {"cur_socket_power2", "amdsmi", "Current socket power", amd_smi_power_info_current_socket_power_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "W"},
    {"avg_socket_power2", "amdsmi", "Average socket power", amd_smi_power_info_average_socket_power_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "W"},
    {"soc_voltage", "amdsmi", "SoC voltage", amd_smi_power_info_soc_voltage_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "mV"},
    {"gfx_voltage", "amdsmi", "GFX voltage", amd_smi_power_info_gfx_voltage_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "mV"},
    {"mem_voltage", "amdsmi", "Memory voltage", amd_smi_power_info_mem_voltage_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "mV"},
    {"power_limit2", "amdsmi", "Power limit", amd_smi_power_info_power_limit_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "W"},
    {"power_cap_0", "amdsmi", "Current power cap of sensor 0", amd_smi_power_cap_sensor_0_current_getter, amd_smi_power_cap_sensor_0_setter, DEVICE_TYPE_AMD_GPU, NULL, "W"},
    {"power_cap_0_min", "amdsmi", "Minimal power cap of sensor 0", amd_smi_power_cap_sensor_0_min_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "W"},
    {"power_cap_0_max", "amdsmi", "Maximal power cap of sensor 0", amd_smi_power_cap_sensor_0_max_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "W"},
    {"power_cap_0_default", "amdsmi", "Default power cap of sensor 0", amd_smi_power_cap_sensor_0_default_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "W"},
    {"power_cap_0_dpm", "amdsmi", "DPM power cap of sensor 0", amd_smi_power_cap_sensor_0_dpm_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "W"},
    {"clean_local_data", "amdsmi", "Run the cleaner shader to clean up data in LDS/GPRs", NULL, amd_smi_gpu_clean_local_data_setter, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"fan_speed_0", "amdsmi", "Current fan speed of fan 0", amd_smi_fan_speed_0_getter, amd_smi_fan_speed_0_setter, DEVICE_TYPE_AMD_GPU, NULL, "RPM"},
    {"fan_speed_max_0", "amdsmi", "Maximal fan speed of fan 0", amd_smi_fan_speed_max_0_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "RPM"},
    {"voltage_vddgfx_cur", "amdsmi", "Current voltage of VDDGFX sensor", amd_smi_generic_gpu_volt_metric_AMDSMI_VOLT_TYPE_VDDGFX_AMDSMI_VOLT_CURRENT_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "V"},
    {"voltage_vddgfx_min", "amdsmi", "Minimal voltage of VDDGFX sensor", amd_smi_generic_gpu_volt_metric_AMDSMI_VOLT_TYPE_VDDGFX_AMDSMI_VOLT_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "V"},
    {"voltage_vddgfx_min_crit", "amdsmi", "Minimal critical voltage of VDDGFX sensor", amd_smi_generic_gpu_volt_metric_AMDSMI_VOLT_TYPE_VDDGFX_AMDSMI_VOLT_MIN_CRIT_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "V"},
    {"voltage_vddgfx_max", "amdsmi", "Maximal voltage of VDDGFX sensor", amd_smi_generic_gpu_volt_metric_AMDSMI_VOLT_TYPE_VDDGFX_AMDSMI_VOLT_MAX_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "V"},
    {"voltage_vddgfx_max_crit", "amdsmi", "Maximal critical voltage of VDDGFX sensor", amd_smi_generic_gpu_volt_metric_AMDSMI_VOLT_TYPE_VDDGFX_AMDSMI_VOLT_MAX_CRIT_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "V"},
    {"voltage_vddgfx_lowest", "amdsmi", "Historical minimum voltage of VDDGFX sensor", amd_smi_generic_gpu_volt_metric_AMDSMI_VOLT_TYPE_VDDGFX_AMDSMI_VOLT_LOWEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "V"},
    {"voltage_vddgfx_highest", "amdsmi", "Historical maximum voltage of VDDGFX sensor", amd_smi_generic_gpu_volt_metric_AMDSMI_VOLT_TYPE_VDDGFX_AMDSMI_VOLT_HIGHEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "V"},
    {"voltage_vddgfx_avg", "amdsmi", "Average voltage of VDDGFX sensor", amd_smi_generic_gpu_volt_metric_AMDSMI_VOLT_TYPE_VDDGFX_AMDSMI_VOLT_AVERAGE_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "V"},
    {"voltage_vddboard_cur", "amdsmi", "Current voltage of VDDBOARD sensor", amd_smi_generic_gpu_volt_metric_AMDSMI_VOLT_TYPE_VDDBOARD_AMDSMI_VOLT_CURRENT_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "V"},
    {"voltage_vddboard_min", "amdsmi", "Minimal voltage of VDDBOARD sensor", amd_smi_generic_gpu_volt_metric_AMDSMI_VOLT_TYPE_VDDBOARD_AMDSMI_VOLT_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "V"},
    {"voltage_vddboard_min_crit", "amdsmi", "Minimal critical voltage of VDDBOARD sensor", amd_smi_generic_gpu_volt_metric_AMDSMI_VOLT_TYPE_VDDBOARD_AMDSMI_VOLT_MIN_CRIT_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "V"},
    {"voltage_vddboard_max", "amdsmi", "Maximal voltage of VDDBOARD sensor", amd_smi_generic_gpu_volt_metric_AMDSMI_VOLT_TYPE_VDDBOARD_AMDSMI_VOLT_MAX_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "V"},
    {"voltage_vddboard_max_crit", "amdsmi", "Maximal critical voltage of VDDBOARD sensor", amd_smi_generic_gpu_volt_metric_AMDSMI_VOLT_TYPE_VDDBOARD_AMDSMI_VOLT_MAX_CRIT_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "V"},
    {"voltage_vddboard_lowest", "amdsmi", "Historical minimum voltage of VDDBOARD sensor", amd_smi_generic_gpu_volt_metric_AMDSMI_VOLT_TYPE_VDDBOARD_AMDSMI_VOLT_LOWEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "V"},
    {"voltage_vddboard_highest", "amdsmi", "Historical maximum voltage of VDDBOARD sensor", amd_smi_generic_gpu_volt_metric_AMDSMI_VOLT_TYPE_VDDBOARD_AMDSMI_VOLT_HIGHEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "V"},
    {"voltage_vddboard_avg", "amdsmi", "Average voltage of VDDBOARD sensor", amd_smi_generic_gpu_volt_metric_AMDSMI_VOLT_TYPE_VDDBOARD_AMDSMI_VOLT_AVERAGE_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "V"},
    {"temp_edge_current", "amdsmi", "Current edge temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_EDGE_AMDSMI_TEMP_CURRENT_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_edge_min", "amdsmi", "Minimal edge temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_EDGE_AMDSMI_TEMP_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_edge_max", "amdsmi", "Maximal edge temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_EDGE_AMDSMI_TEMP_MAX_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_edge_min_hyst", "amdsmi", "Minimal limit hysteresis edge temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_EDGE_AMDSMI_TEMP_MIN_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_edge_max_hyst", "amdsmi", "Maximal limit hysteresis edge temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_EDGE_AMDSMI_TEMP_MAX_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_edge_critical_max", "amdsmi", "Critical maximal edge temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_EDGE_AMDSMI_TEMP_CRITICAL_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_edge_critical_hyst", "amdsmi", "Critical maximal limit hysteresis edge temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_EDGE_AMDSMI_TEMP_CRITICAL_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_edge_critical_min", "amdsmi", "Critical minimal edge temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_EDGE_AMDSMI_TEMP_CRIT_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_edge_critical_min_hyst", "amdsmi", "Critical minimal limit hysteresis edge temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_EDGE_AMDSMI_TEMP_CRIT_MIN_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_edge_emergency", "amdsmi", "Emergency maximal edge temperature, for chips supporting more than two upper temperature limits", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_EDGE_AMDSMI_TEMP_EMERGENCY_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_edge_emergency_hyst", "amdsmi", "Emergency limit hysteresis edge temperature, for chips supporting more than two upper temperature limits", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_EDGE_AMDSMI_TEMP_EMERGENCY_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_edge_lowest", "amdsmi", "Historical minimal edge temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_EDGE_AMDSMI_TEMP_LOWEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_edge_highest", "amdsmi", "Historical maximal edge temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_EDGE_AMDSMI_TEMP_HIGHEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_edge_shutdown", "amdsmi", "Shutdown edge temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_EDGE_AMDSMI_TEMP_SHUTDOWN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_edge_offset", "amdsmi", "Temperature offset for edge which is added to the temperature reading by the chip ", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_EDGE_AMDSMI_TEMP_OFFSET_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hotspot_current", "amdsmi", "Current hotspot temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HOTSPOT_AMDSMI_TEMP_CURRENT_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hotspot_min", "amdsmi", "Minimal hotspot temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HOTSPOT_AMDSMI_TEMP_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hotspot_max", "amdsmi", "Maximal hotspot temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HOTSPOT_AMDSMI_TEMP_MAX_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hotspot_min_hyst", "amdsmi", "Minimal limit hysteresis hotspot temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HOTSPOT_AMDSMI_TEMP_MIN_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hotspot_max_hyst", "amdsmi", "Maximal limit hysteresis hotspot temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HOTSPOT_AMDSMI_TEMP_MAX_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hotspot_critical_max", "amdsmi", "Critical maximal hotspot temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HOTSPOT_AMDSMI_TEMP_CRITICAL_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hotspot_critical_hyst", "amdsmi", "Critical maximal limit hysteresis hotspot temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HOTSPOT_AMDSMI_TEMP_CRITICAL_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hotspot_critical_min", "amdsmi", "Critical minimal hotspot temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HOTSPOT_AMDSMI_TEMP_CRIT_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hotspot_critical_min_hyst", "amdsmi", "Critical minimal limit hysteresis hotspot temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HOTSPOT_AMDSMI_TEMP_CRIT_MIN_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hotspot_emergency", "amdsmi", "Emergency maximal hotspot temperature, for chips supporting more than two upper temperature limits", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HOTSPOT_AMDSMI_TEMP_EMERGENCY_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hotspot_emergency_hyst", "amdsmi", "Emergency limit hysteresis hotspot temperature, for chips supporting more than two upper temperature limits", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HOTSPOT_AMDSMI_TEMP_EMERGENCY_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hotspot_lowest", "amdsmi", "Historical minimal hotspot temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HOTSPOT_AMDSMI_TEMP_LOWEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hotspot_highest", "amdsmi", "Historical maximal hotspot temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HOTSPOT_AMDSMI_TEMP_HIGHEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hotspot_shutdown", "amdsmi", "Shutdown hotspot temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HOTSPOT_AMDSMI_TEMP_SHUTDOWN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hotspot_offset", "amdsmi", "Temperature offset for hotspot which is added to the temperature reading by the chip ", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HOTSPOT_AMDSMI_TEMP_OFFSET_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_vram_current", "amdsmi", "Current vram temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_VRAM_AMDSMI_TEMP_CURRENT_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_vram_min", "amdsmi", "Minimal vram temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_VRAM_AMDSMI_TEMP_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_vram_max", "amdsmi", "Maximal vram temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_VRAM_AMDSMI_TEMP_MAX_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_vram_min_hyst", "amdsmi", "Minimal limit hysteresis vram temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_VRAM_AMDSMI_TEMP_MIN_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_vram_max_hyst", "amdsmi", "Maximal limit hysteresis vram temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_VRAM_AMDSMI_TEMP_MAX_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_vram_critical_max", "amdsmi", "Critical maximal vram temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_VRAM_AMDSMI_TEMP_CRITICAL_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_vram_critical_hyst", "amdsmi", "Critical maximal limit hysteresis vram temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_VRAM_AMDSMI_TEMP_CRITICAL_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_vram_critical_min", "amdsmi", "Critical minimal vram temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_VRAM_AMDSMI_TEMP_CRIT_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_vram_critical_min_hyst", "amdsmi", "Critical minimal limit hysteresis vram temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_VRAM_AMDSMI_TEMP_CRIT_MIN_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_vram_emergency", "amdsmi", "Emergency maximal vram temperature, for chips supporting more than two upper temperature limits", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_VRAM_AMDSMI_TEMP_EMERGENCY_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_vram_emergency_hyst", "amdsmi", "Emergency limit hysteresis vram temperature, for chips supporting more than two upper temperature limits", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_VRAM_AMDSMI_TEMP_EMERGENCY_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_vram_lowest", "amdsmi", "Historical minimal vram temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_VRAM_AMDSMI_TEMP_LOWEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_vram_highest", "amdsmi", "Historical maximal vram temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_VRAM_AMDSMI_TEMP_HIGHEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_vram_shutdown", "amdsmi", "Shutdown vram temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_VRAM_AMDSMI_TEMP_SHUTDOWN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_vram_offset", "amdsmi", "Temperature offset for vram which is added to the temperature reading by the chip ", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_VRAM_AMDSMI_TEMP_OFFSET_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm0_current", "amdsmi", "Current hbm0 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_0_AMDSMI_TEMP_CURRENT_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm0_min", "amdsmi", "Minimal hbm0 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_0_AMDSMI_TEMP_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm0_max", "amdsmi", "Maximal hbm0 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_0_AMDSMI_TEMP_MAX_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm0_min_hyst", "amdsmi", "Minimal limit hysteresis hbm0 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_0_AMDSMI_TEMP_MIN_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm0_max_hyst", "amdsmi", "Maximal limit hysteresis hbm0 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_0_AMDSMI_TEMP_MAX_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm0_critical_max", "amdsmi", "Critical maximal hbm0 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_0_AMDSMI_TEMP_CRITICAL_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm0_critical_hyst", "amdsmi", "Critical maximal limit hysteresis hbm0 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_0_AMDSMI_TEMP_CRITICAL_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm0_critical_min", "amdsmi", "Critical minimal hbm0 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_0_AMDSMI_TEMP_CRIT_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm0_critical_min_hyst", "amdsmi", "Critical minimal limit hysteresis hbm0 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_0_AMDSMI_TEMP_CRIT_MIN_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm0_emergency", "amdsmi", "Emergency maximal hbm0 temperature, for chips supporting more than two upper temperature limits", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_0_AMDSMI_TEMP_EMERGENCY_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm0_emergency_hyst", "amdsmi", "Emergency limit hysteresis hbm0 temperature, for chips supporting more than two upper temperature limits", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_0_AMDSMI_TEMP_EMERGENCY_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm0_lowest", "amdsmi", "Historical minimal hbm0 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_0_AMDSMI_TEMP_LOWEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm0_highest", "amdsmi", "Historical maximal hbm0 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_0_AMDSMI_TEMP_HIGHEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm0_shutdown", "amdsmi", "Shutdown hbm0 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_0_AMDSMI_TEMP_SHUTDOWN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm0_offset", "amdsmi", "Temperature offset for hbm0 which is added to the temperature reading by the chip ", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_0_AMDSMI_TEMP_OFFSET_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm1_current", "amdsmi", "Current hbm1 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_1_AMDSMI_TEMP_CURRENT_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm1_min", "amdsmi", "Minimal hbm1 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_1_AMDSMI_TEMP_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm1_max", "amdsmi", "Maximal hbm1 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_1_AMDSMI_TEMP_MAX_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm1_min_hyst", "amdsmi", "Minimal limit hysteresis hbm1 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_1_AMDSMI_TEMP_MIN_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm1_max_hyst", "amdsmi", "Maximal limit hysteresis hbm1 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_1_AMDSMI_TEMP_MAX_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm1_critical_max", "amdsmi", "Critical maximal hbm1 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_1_AMDSMI_TEMP_CRITICAL_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm1_critical_hyst", "amdsmi", "Critical maximal limit hysteresis hbm1 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_1_AMDSMI_TEMP_CRITICAL_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm1_critical_min", "amdsmi", "Critical minimal hbm1 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_1_AMDSMI_TEMP_CRIT_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm1_critical_min_hyst", "amdsmi", "Critical minimal limit hysteresis hbm1 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_1_AMDSMI_TEMP_CRIT_MIN_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm1_emergency", "amdsmi", "Emergency maximal hbm1 temperature, for chips supporting more than two upper temperature limits", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_1_AMDSMI_TEMP_EMERGENCY_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm1_emergency_hyst", "amdsmi", "Emergency limit hysteresis hbm1 temperature, for chips supporting more than two upper temperature limits", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_1_AMDSMI_TEMP_EMERGENCY_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm1_lowest", "amdsmi", "Historical minimal hbm1 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_1_AMDSMI_TEMP_LOWEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm1_highest", "amdsmi", "Historical maximal hbm1 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_1_AMDSMI_TEMP_HIGHEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm1_shutdown", "amdsmi", "Shutdown hbm1 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_1_AMDSMI_TEMP_SHUTDOWN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm1_offset", "amdsmi", "Temperature offset for hbm1 which is added to the temperature reading by the chip ", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_1_AMDSMI_TEMP_OFFSET_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm2_current", "amdsmi", "Current hbm2 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_2_AMDSMI_TEMP_CURRENT_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm2_min", "amdsmi", "Minimal hbm2 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_2_AMDSMI_TEMP_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm2_max", "amdsmi", "Maximal hbm2 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_2_AMDSMI_TEMP_MAX_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm2_min_hyst", "amdsmi", "Minimal limit hysteresis hbm2 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_2_AMDSMI_TEMP_MIN_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm2_max_hyst", "amdsmi", "Maximal limit hysteresis hbm2 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_2_AMDSMI_TEMP_MAX_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm2_critical_max", "amdsmi", "Critical maximal hbm2 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_2_AMDSMI_TEMP_CRITICAL_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm2_critical_hyst", "amdsmi", "Critical maximal limit hysteresis hbm2 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_2_AMDSMI_TEMP_CRITICAL_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm2_critical_min", "amdsmi", "Critical minimal hbm2 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_2_AMDSMI_TEMP_CRIT_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm2_critical_min_hyst", "amdsmi", "Critical minimal limit hysteresis hbm2 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_2_AMDSMI_TEMP_CRIT_MIN_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm2_emergency", "amdsmi", "Emergency maximal hbm2 temperature, for chips supporting more than two upper temperature limits", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_2_AMDSMI_TEMP_EMERGENCY_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm2_emergency_hyst", "amdsmi", "Emergency limit hysteresis hbm2 temperature, for chips supporting more than two upper temperature limits", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_2_AMDSMI_TEMP_EMERGENCY_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm2_lowest", "amdsmi", "Historical minimal hbm2 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_2_AMDSMI_TEMP_LOWEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm2_highest", "amdsmi", "Historical maximal hbm2 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_2_AMDSMI_TEMP_HIGHEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm2_shutdown", "amdsmi", "Shutdown hbm2 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_2_AMDSMI_TEMP_SHUTDOWN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm2_offset", "amdsmi", "Temperature offset for hbm2 which is added to the temperature reading by the chip ", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_2_AMDSMI_TEMP_OFFSET_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm3_current", "amdsmi", "Current hbm3 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_3_AMDSMI_TEMP_CURRENT_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm3_min", "amdsmi", "Minimal hbm3 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_3_AMDSMI_TEMP_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm3_max", "amdsmi", "Maximal hbm3 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_3_AMDSMI_TEMP_MAX_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm3_min_hyst", "amdsmi", "Minimal limit hysteresis hbm3 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_3_AMDSMI_TEMP_MIN_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm3_max_hyst", "amdsmi", "Maximal limit hysteresis hbm3 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_3_AMDSMI_TEMP_MAX_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm3_critical_max", "amdsmi", "Critical maximal hbm3 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_3_AMDSMI_TEMP_CRITICAL_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm3_critical_hyst", "amdsmi", "Critical maximal limit hysteresis hbm3 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_3_AMDSMI_TEMP_CRITICAL_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm3_critical_min", "amdsmi", "Critical minimal hbm3 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_3_AMDSMI_TEMP_CRIT_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm3_critical_min_hyst", "amdsmi", "Critical minimal limit hysteresis hbm3 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_3_AMDSMI_TEMP_CRIT_MIN_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm3_emergency", "amdsmi", "Emergency maximal hbm3 temperature, for chips supporting more than two upper temperature limits", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_3_AMDSMI_TEMP_EMERGENCY_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm3_emergency_hyst", "amdsmi", "Emergency limit hysteresis hbm3 temperature, for chips supporting more than two upper temperature limits", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_3_AMDSMI_TEMP_EMERGENCY_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm3_lowest", "amdsmi", "Historical minimal hbm3 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_3_AMDSMI_TEMP_LOWEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm3_highest", "amdsmi", "Historical maximal hbm3 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_3_AMDSMI_TEMP_HIGHEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm3_shutdown", "amdsmi", "Shutdown hbm3 temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_3_AMDSMI_TEMP_SHUTDOWN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_hbm3_offset", "amdsmi", "Temperature offset for hbm3 which is added to the temperature reading by the chip ", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_HBM_3_AMDSMI_TEMP_OFFSET_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_plx_current", "amdsmi", "Current plx temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_PLX_AMDSMI_TEMP_CURRENT_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_plx_min", "amdsmi", "Minimal plx temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_PLX_AMDSMI_TEMP_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_plx_max", "amdsmi", "Maximal plx temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_PLX_AMDSMI_TEMP_MAX_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_plx_min_hyst", "amdsmi", "Minimal limit hysteresis plx temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_PLX_AMDSMI_TEMP_MIN_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_plx_max_hyst", "amdsmi", "Maximal limit hysteresis plx temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_PLX_AMDSMI_TEMP_MAX_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_plx_critical_max", "amdsmi", "Critical maximal plx temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_PLX_AMDSMI_TEMP_CRITICAL_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_plx_critical_hyst", "amdsmi", "Critical maximal limit hysteresis plx temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_PLX_AMDSMI_TEMP_CRITICAL_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_plx_critical_min", "amdsmi", "Critical minimal plx temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_PLX_AMDSMI_TEMP_CRIT_MIN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_plx_critical_min_hyst", "amdsmi", "Critical minimal limit hysteresis plx temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_PLX_AMDSMI_TEMP_CRIT_MIN_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_plx_emergency", "amdsmi", "Emergency maximal plx temperature, for chips supporting more than two upper temperature limits", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_PLX_AMDSMI_TEMP_EMERGENCY_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_plx_emergency_hyst", "amdsmi", "Emergency limit hysteresis plx temperature, for chips supporting more than two upper temperature limits", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_PLX_AMDSMI_TEMP_EMERGENCY_HYST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_plx_lowest", "amdsmi", "Historical minimal plx temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_PLX_AMDSMI_TEMP_LOWEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_plx_highest", "amdsmi", "Historical maximal plx temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_PLX_AMDSMI_TEMP_HIGHEST_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_plx_shutdown", "amdsmi", "Shutdown plx temperature", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_PLX_AMDSMI_TEMP_SHUTDOWN_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"temp_plx_offset", "amdsmi", "Temperature offset for plx which is added to the temperature reading by the chip ", amd_smi_temp_metric_AMDSMI_TEMPERATURE_TYPE_PLX_AMDSMI_TEMP_OFFSET_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "°C"},
    {"gfx_active_avg", "amdsmi", "Average GFX activity", amd_smi_gpu_gfx_activity_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "%"},
    {"umc_active_avg", "amdsmi", "Average UMC activity", amd_smi_gpu_umc_activity_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "%"},
    {"mm_active_avg", "amdsmi", "Average MM activity", amd_smi_gpu_mm_activity_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "%"},
    {"pwr_managed", "amdsmi", "Power management activity", amd_smi_gpu_pwr_manage_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
};






static const _SysFeatureList amd_smi_feature_list = {
    .num_features = ARRAY_COUNT(amd_smi_features),
    .features = amd_smi_features,
};

