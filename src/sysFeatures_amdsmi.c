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

#include <error.h>
#include <sysFeatures_common.h>
#include <types.h>
#include <lw_alloc.h>

#define ASMI_CALL(func, ...) \
    do {\
        assert(func##_ptr != NULL); \
        amdsmi_status_t s_ = func##_ptr(__VA_ARGS__);\
        if (s_ != AMDSMI_STATUS_SUCCESS) {\
            const char *errstr_ = NULL;\
            amdsmi_status_t es = amdsmi_get_esmi_err_msg_ptr(s_, &errstr_);\
            if (es != AMDSMI_STATUS_SUCCESS) { \
                ERROR_PRINT("Error: function %s failed but cannot resolve error string (amdsmi_status_t=%d)", #func, s_);\
            } else { \
                ERROR_PRINT("Error: function %s failed: '%s' (amdsmi_status_t=%d)", #func, errstr_, s_);\
            } \
            return -EPERM; \
        }\
    } while (0)


#define DECLAREFUNC_ASMI(funcname, ...) static amdsmi_status_t (*funcname##_ptr)(__VA_ARGS__)

DECLAREFUNC_ASMI(amdsmi_init, uint64_t flags);
DECLAREFUNC_ASMI(amdsmi_shut_down, void);
DECLAREFUNC_ASMI(amdsmi_get_esmi_err_msg, amdsmi_status_t status, const char **status_string);

DECLAREFUNC_ASMI(amdsmi_get_socket_handles, uint32_t *socket_count, amdsmi_socket_handle *socket_handles);
DECLAREFUNC_ASMI(amdsmi_get_processor_handles, amdsmi_socket_handle socket_handle, uint32_t *processor_count, amdsmi_processor_handle *processor_handles);
DECLAREFUNC_ASMI(amdsmi_get_processor_handle_from_bdf, amdsmi_bdf_t bdf, amdsmi_processor_handle *processor_handle);
DECLAREFUNC_ASMI(amdsmi_get_processor_handles_by_type, amdsmi_socket_handle socket_handle, processor_type_t processor_type, amdsmi_processor_handle *processor_handles, uint32_t *processor_count);
DECLAREFUNC_ASMI(amdsmi_get_processor_type, amdsmi_processor_handle processor_handle, processor_type_t *processor_type);
DECLAREFUNC_ASMI(amdsmi_get_gpu_board_info, amdsmi_processor_handle processor_handle, amdsmi_board_info_t *info);
DECLAREFUNC_ASMI(amdsmi_get_power_cap_info, amdsmi_processor_handle processor_handle, uint32_t sensor_ind, amdsmi_power_cap_info_t *info);
DECLAREFUNC_ASMI(amdsmi_get_pcie_info, amdsmi_processor_handle processor_handle, amdsmi_pcie_info_t *info);

DECLAREFUNC_ASMI(amdsmi_get_gpu_vram_info, amdsmi_processor_handle processor_handle, amdsmi_vram_info_t *info);
DECLAREFUNC_ASMI(amdsmi_get_gpu_kfd_info, amdsmi_processor_handle processor_handle, amdsmi_kfd_info_t *info);
DECLAREFUNC_ASMI(amdsmi_get_gpu_asic_info, amdsmi_processor_handle processor_handle, amdsmi_asic_info_t *info);
DECLAREFUNC_ASMI(amdsmi_get_gpu_device_bdf, amdsmi_processor_handle processor_handle, amdsmi_bdf_t *bdf);
DECLAREFUNC_ASMI(amdsmi_get_power_info, amdsmi_processor_handle processor_handle, amdsmi_power_info_t *info);
DECLAREFUNC_ASMI(amdsmi_get_clock_info, amdsmi_processor_handle processor_handle, amdsmi_clk_type_t clk_type, amdsmi_clk_info_t *info);
DECLAREFUNC_ASMI(amdsmi_get_temp_metric, amdsmi_processor_handle processor_handle, amdsmi_temperature_type_t sensor_type, amdsmi_temperature_metric_t metric, int64_t *temperature);

DECLAREFUNC_ASMI(amdsmi_get_clk_freq, amdsmi_processor_handle processor_handle, amdsmi_clk_type_t clk_type, amdsmi_frequencies_t *f);
DECLAREFUNC_ASMI(amdsmi_set_gpu_clk_limit, amdsmi_processor_handle processor_handle, amdsmi_clk_type_t clk_type, amdsmi_clk_limit_type_t limit_type, uint64_t clk_value);
DECLAREFUNC_ASMI(amdsmi_get_cpu_socket_power, amdsmi_processor_handle processor_handle, uint32_t *ppower);
DECLAREFUNC_ASMI(amdsmi_get_cpu_socket_power_cap, amdsmi_processor_handle processor_handle, uint32_t *ppower);
DECLAREFUNC_ASMI(amdsmi_set_cpu_socket_power_cap, amdsmi_processor_handle processor_handle, uint32_t pcap);
DECLAREFUNC_ASMI(amdsmi_get_cpu_socket_power_cap_max, amdsmi_processor_handle processor_handle, uint32_t *pmax);
#if (AMDSMI_LIB_VERSION_MAJOR >= 26 && AMDSMI_LIB_VERSION_MINOR >= 2 && AMDSMI_LIB_VERSION_RELEASE >= 1)
DECLAREFUNC_ASMI(amdsmi_get_node_handle, amdsmi_processor_handle processor_handle, amdsmi_node_handle *node_handle);
DECLAREFUNC_ASMI(amdsmi_get_npm_info, amdsmi_node_handle node_handle, amdsmi_npm_info_t *info);
#endif


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
    DLSYM_CHK(lib_amd_smi, amdsmi_get_processor_handles);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_processor_handle_from_bdf);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_processor_handles_by_type);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_processor_type);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_board_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_power_cap_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_pcie_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_vram_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_kfd_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_asic_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_gpu_device_bdf);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_power_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_clock_info);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_temp_metric);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_clk_freq);
    DLSYM_CHK(lib_amd_smi, amdsmi_set_gpu_clk_limit);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_cpu_socket_power);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_cpu_socket_power_cap);
    DLSYM_CHK(lib_amd_smi, amdsmi_set_cpu_socket_power_cap);
    DLSYM_CHK(lib_amd_smi, amdsmi_get_cpu_socket_power_cap_max);
#if (AMDSMI_LIB_VERSION_MAJOR >= 26 && AMDSMI_LIB_VERSION_MINOR >= 2 && AMDSMI_LIB_VERSION_RELEASE >= 1)
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

    return likwid_sysft_register_features(list, &amd_smi_feature_list);
}
#undef DLSYM_CHK

static int lw_device_to_amdsmi_handle(const LikwidDevice_t lwDevice, amdsmi_processor_handle **amdsmiHandle) {
    if (lwDevice->type != DEVICE_TYPE_AMD_GPU)
        return -EINVAL;
    amdsmi_bdf_t bdf = {
        .domain_number = lwDevice->id.pci.pci_domain,
        .bus_number = lwDevice->id.pci.pci_bus,
        .device_number = lwDevice->id.pci.pci_dev,
        .function_number = lwDevice->id.pci.pci_func,
    };
    amdsmi_processor_handle handle = NULL;
    ASMI_CALL(amdsmi_get_processor_handle_from_bdf, bdf, &handle);

    *amdsmiHandle = handle;
    return 0;
}


static int amd_smi_device_count_getter(const LikwidDevice_t device, char **value)
{
    if (device->type != DEVICE_TYPE_NODE)
        return -EINVAL;

    uint64_t allDevices = 0;
    uint32_t socketCount = 0;
    DEBUG_PRINT(DEBUGLEV_DEVELOP, "Running amd_smi_device_count_getter");
    ASMI_CALL(amdsmi_get_socket_handles, &socketCount, NULL);
    amdsmi_socket_handle* socketHandles = lw_malloc(socketCount * sizeof(amdsmi_socket_handle));

    ASMI_CALL(amdsmi_get_socket_handles, &socketCount, socketHandles);

    for (uint32_t i = 0; i < socketCount; i++) {
        uint32_t deviceCount = 0;
        ASMI_CALL(amdsmi_get_processor_handles, socketHandles[i], &deviceCount, NULL);
        
        allDevices += deviceCount;
    }

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

    amdsmi_processor_handle *handle = NULL;
    int err = lw_device_to_amdsmi_handle(device, &handle);
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


static int amd_smi_sys_current_clock_freq_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_SYS, AMDSMI_CLOCK_FREQ_GETTER_FIELD_CURRENT);
}

static int amd_smi_gfx_current_clock_freq_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_GFX, AMDSMI_CLOCK_FREQ_GETTER_FIELD_CURRENT);
}

static int amd_smi_df_current_clock_freq_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_DF, AMDSMI_CLOCK_FREQ_GETTER_FIELD_CURRENT);
}

static int amd_smi_dcef_current_clock_freq_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_DCEF, AMDSMI_CLOCK_FREQ_GETTER_FIELD_CURRENT);
}

static int amd_smi_soc_current_clock_freq_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_SOC, AMDSMI_CLOCK_FREQ_GETTER_FIELD_CURRENT);
}

static int amd_smi_mem_current_clock_freq_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_MEM, AMDSMI_CLOCK_FREQ_GETTER_FIELD_CURRENT);
}

static int amd_smi_pcie_current_clock_freq_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_PCIE, AMDSMI_CLOCK_FREQ_GETTER_FIELD_CURRENT);
}

static int amd_smi_vclk0_current_clock_freq_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_VCLK0, AMDSMI_CLOCK_FREQ_GETTER_FIELD_CURRENT);
}

static int amd_smi_vclk1_current_clock_freq_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_VCLK1, AMDSMI_CLOCK_FREQ_GETTER_FIELD_CURRENT);
}

static int amd_smi_dclk0_current_clock_freq_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_DCLK0, AMDSMI_CLOCK_FREQ_GETTER_FIELD_CURRENT);
}

static int amd_smi_dclk1_current_clock_freq_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_DCLK1, AMDSMI_CLOCK_FREQ_GETTER_FIELD_CURRENT);
}


static int amd_smi_sys_hasdeepsleep_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_SYS, AMDSMI_CLOCK_FREQ_GETTER_FIELD_DEEP_SLEEP);
}

static int amd_smi_gfx_hasdeepsleep_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_GFX, AMDSMI_CLOCK_FREQ_GETTER_FIELD_DEEP_SLEEP);
}

static int amd_smi_df_hasdeepsleep_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_DF, AMDSMI_CLOCK_FREQ_GETTER_FIELD_DEEP_SLEEP);
}

static int amd_smi_dcef_hasdeepsleep_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_DCEF, AMDSMI_CLOCK_FREQ_GETTER_FIELD_DEEP_SLEEP);
}

static int amd_smi_soc_hasdeepsleep_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_SOC, AMDSMI_CLOCK_FREQ_GETTER_FIELD_DEEP_SLEEP);
}

static int amd_smi_mem_hasdeepsleep_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_MEM, AMDSMI_CLOCK_FREQ_GETTER_FIELD_DEEP_SLEEP);
}

static int amd_smi_pcie_hasdeepsleep_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_PCIE, AMDSMI_CLOCK_FREQ_GETTER_FIELD_DEEP_SLEEP);
}

static int amd_smi_vclk0_hasdeepsleep_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_VCLK0, AMDSMI_CLOCK_FREQ_GETTER_FIELD_DEEP_SLEEP);
}

static int amd_smi_vclk1_hasdeepsleep_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_VCLK1, AMDSMI_CLOCK_FREQ_GETTER_FIELD_DEEP_SLEEP);
}

static int amd_smi_dclk0_hasdeepsleep_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_DCLK0, AMDSMI_CLOCK_FREQ_GETTER_FIELD_DEEP_SLEEP);
}

static int amd_smi_dclk1_hasdeepsleep_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_DCLK1, AMDSMI_CLOCK_FREQ_GETTER_FIELD_DEEP_SLEEP);
}


static int amd_smi_sys_supported_freqs_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_SYS, AMDSMI_CLOCK_FREQ_GETTER_FIELD_SUP_FREQS);
}

static int amd_smi_gfx_supported_freqs_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_GFX, AMDSMI_CLOCK_FREQ_GETTER_FIELD_SUP_FREQS);
}

static int amd_smi_df_supported_freqs_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_DF, AMDSMI_CLOCK_FREQ_GETTER_FIELD_SUP_FREQS);
}

static int amd_smi_dcef_supported_freqs_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_DCEF, AMDSMI_CLOCK_FREQ_GETTER_FIELD_SUP_FREQS);
}

static int amd_smi_soc_supported_freqs_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_SOC, AMDSMI_CLOCK_FREQ_GETTER_FIELD_SUP_FREQS);
}

static int amd_smi_mem_supported_freqs_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_MEM, AMDSMI_CLOCK_FREQ_GETTER_FIELD_SUP_FREQS);
}

static int amd_smi_pcie_supported_freqs_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_PCIE, AMDSMI_CLOCK_FREQ_GETTER_FIELD_SUP_FREQS);
}

static int amd_smi_vclk0_supported_freqs_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_VCLK0, AMDSMI_CLOCK_FREQ_GETTER_FIELD_SUP_FREQS);
}

static int amd_smi_vclk1_supported_freqs_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_VCLK1, AMDSMI_CLOCK_FREQ_GETTER_FIELD_SUP_FREQS);
}

static int amd_smi_dclk0_supported_freqs_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_DCLK0, AMDSMI_CLOCK_FREQ_GETTER_FIELD_SUP_FREQS);
}

static int amd_smi_dclk1_supported_freqs_getter(const LikwidDevice_t device, char **value) {
    return amd_smi_generic_clk_freq_getter(device, value, AMDSMI_CLK_TYPE_DCLK1, AMDSMI_CLOCK_FREQ_GETTER_FIELD_SUP_FREQS);
}


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

    amdsmi_processor_handle *handle = NULL;
    err = lw_device_to_amdsmi_handle(device, &handle);
    if (err) {
        return err;
    }

    ASMI_CALL(amdsmi_set_gpu_clk_limit, handle, clk_type, set_type, freq);
    return 0;
}

static int amd_smi_sys_min_clk_freq_setter(const LikwidDevice_t device, const char *value) {
    return amd_smi_generic_clk_freq_setter(device, value, AMDSMI_CLK_TYPE_SYS, 1);
}

static int amd_smi_sys_max_clk_freq_setter(const LikwidDevice_t device, const char *value) {
    return amd_smi_generic_clk_freq_setter(device, value, AMDSMI_CLK_TYPE_SYS, 0);
}

static int amd_smi_socket_power_getter(const LikwidDevice_t device, char **value) {
    if (device->type != DEVICE_TYPE_AMD_GPU) {
        return -EINVAL;
    }

    amdsmi_processor_handle *handle = NULL;
    int err = lw_device_to_amdsmi_handle(device, &handle);
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

    amdsmi_processor_handle *handle = NULL;
    int err = lw_device_to_amdsmi_handle(device, &handle);
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

    amdsmi_processor_handle *handle = NULL;
    err = lw_device_to_amdsmi_handle(device, &handle);
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

    amdsmi_processor_handle *handle = NULL;
    int err = lw_device_to_amdsmi_handle(device, &handle);
    if (err) {
        return err;
    }

    uint32_t power = 0;
    ASMI_CALL(amdsmi_get_cpu_socket_power_cap_max, handle, &power);
    return likwid_sysft_uint64_to_string(power, value);
}




static _SysFeature amd_smi_features[] = {
    {"device_count", "amdsmi", "Number of GPUs on node. Not all GPUs may be accessible.", amd_smi_device_count_getter, NULL, DEVICE_TYPE_NODE, NULL, NULL},
    {"sys_cur_clk_freq", "amdsmi", "Current system clock.", amd_smi_sys_current_clock_freq_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"sys_has_deep_sleep", "amdsmi", "Deep sleep frequency support for system.", amd_smi_sys_hasdeepsleep_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"sys_avail_freqs", "amdsmi", "List of supported frequencies for system.", amd_smi_sys_supported_freqs_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"sys_min_clk_freq", "amdsmi", "Minimal frequency for system", NULL, amd_smi_sys_min_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"sys_max_clk_freq", "amdsmi", "Maximal frequency for system", NULL, amd_smi_sys_max_clk_freq_setter, DEVICE_TYPE_AMD_GPU, NULL, "MHz"},
    {"socket_power", "amdsmi", "Socket power", amd_smi_socket_power_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"socket_power_limit", "amdsmi", "Power cap for socket", amd_smi_socket_power_cap_getter, amd_smi_socket_power_cap_setter, DEVICE_TYPE_AMD_GPU, NULL, NULL},
    {"socket_power_limit_max", "amdsmi", "Maximum power cap for socket ", amd_smi_socket_power_cap_max_getter, NULL, DEVICE_TYPE_AMD_GPU, NULL, NULL},
};






static const _SysFeatureList amd_smi_feature_list = {
    .num_features = ARRAY_COUNT(amd_smi_features),
    .features = amd_smi_features,
};

