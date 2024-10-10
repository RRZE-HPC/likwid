#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <sysFeatures_intel.h>
#include <sysFeatures_common.h>
#include <sysFeatures_intel_rapl.h>
#include <access.h>
#include <registers.h>

#include <sysFeatures_common_rapl.h>

static RaplDomainInfo intel_rapl_pkg_info = {0, 0, 0};
static RaplDomainInfo intel_rapl_dram_info = {0, 0, 0};
static RaplDomainInfo intel_rapl_psys_info = {0, 0, 0};
static RaplDomainInfo intel_rapl_pp0_info = {0, 0, 0};
static RaplDomainInfo intel_rapl_pp1_info = {0, 0, 0};

static int sysFeatures_intel_rapl_energy_status_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_unusedinfo_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t energy;
    err = likwid_sysft_readmsr_field(device, reg, 0, 32, &energy);
    if (err)
        return err;
    return sysFeatures_double_to_string((double)energy * info->energyUnit, value);
}

static int sysFeatures_intel_rapl_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_unusedinfo_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t enable;
    err = likwid_sysft_readmsr_field(device, reg, 15, 1, &enable);
    if (err < 0)
        return err;
    return sysFeatures_uint64_to_string(enable, value);
}

static int sysFeatures_intel_rapl_energy_limit_1_enable_setter(const LikwidDevice_t device, const char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_unusedinfo_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t enable;
    err = sysFeatures_string_to_uint64(value, &enable);
    if (err < 0)
        return err;
    return likwid_sysft_writemsr_field(device, reg, 15, 1, enable);
}

static int sysFeatures_intel_rapl_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_unusedinfo_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t clamp;
    err = likwid_sysft_readmsr_field(device, reg, 16, 1, &clamp);
    if (err < 0)
        return err;
    return sysFeatures_uint64_to_string(clamp, value);
}

static int sysFeatures_intel_rapl_energy_limit_1_clamp_setter(const LikwidDevice_t device, const char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_unusedinfo_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t clamp;
    err = sysFeatures_string_to_uint64(value, &clamp);
    if (err < 0)
        return err;
    return likwid_sysft_writemsr_field(device, reg, 16, 1, clamp);
}

static int sysFeatures_intel_rapl_energy_limit_1_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_unusedinfo_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t powerUnits;
    err = likwid_sysft_readmsr_field(device, reg, 0, 15, &powerUnits);
    if (err < 0)
        return err;
    const double watts = (double)powerUnits * info->powerUnit;
    return sysFeatures_double_to_string(watts, value);
}

static int sysFeatures_intel_rapl_energy_limit_1_setter(const LikwidDevice_t device, const char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_info_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    double watts;
    err = sysFeatures_string_to_double(value, &watts);
    if (err < 0)
        return err;
    const uint64_t powerUnits = MIN((uint64_t)round(watts / info->powerUnit), 0x7FFF);
    return likwid_sysft_writemsr_field(device, reg, 0, 15, powerUnits);
}

static int sysFeatures_intel_rapl_energy_limit_1_time_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_unusedinfo_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t timeWindow;
    err = likwid_sysft_readmsr_field(device, reg, 17, 7, &timeWindow);
    if (err < 0)
        return err;
    const double seconds = timeWindow_to_seconds(info, timeWindow);
    return sysFeatures_double_to_string(seconds, value);
}

static int sysFeatures_intel_rapl_energy_limit_1_time_setter(const LikwidDevice_t device, const char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_unusedinfo_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    double seconds;
    err = sysFeatures_string_to_double(value, &seconds);
    if (err < 0)
        return err;
    const uint64_t timeWindow = seconds_to_timeWindow(info, seconds);
    return likwid_sysft_writemsr_field(device, reg, 17, 7, timeWindow);
}

static int sysFeatures_intel_rapl_energy_limit_2_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_unusedinfo_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t powerUnits;
    err = likwid_sysft_readmsr_field(device, reg, 32, 15, &powerUnits);
    if (err < 0)
        return err;
    const double watts = (double)powerUnits * info->powerUnit;
    return sysFeatures_double_to_string(watts, value);
}

static int sysFeatures_intel_rapl_energy_limit_2_setter(const LikwidDevice_t device, const char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_info_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    double watts;
    err = sysFeatures_string_to_double(value, &watts);
    if (err < 0)
        return err;
    const uint64_t powerUnits = MIN((uint64_t)round(watts / info->powerUnit), 0x7FFF);
    return likwid_sysft_writemsr_field(device, reg, 15, 15, powerUnits);
}

static int sysFeatures_intel_rapl_energy_limit_2_time_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_unusedinfo_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t timeWindow;
    err = likwid_sysft_readmsr_field(device, reg, 49, 7, &timeWindow);
    if (err < 0)
        return err;
    const double seconds = timeWindow_to_seconds(info, timeWindow);
    return sysFeatures_double_to_string(seconds, value);
}

static int sysFeatures_intel_rapl_energy_limit_2_time_setter(const LikwidDevice_t device, const char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_info_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    double seconds;
    err = sysFeatures_string_to_double(value, &seconds);
    if (err < 0)
        return err;
    const uint64_t timeWindow = seconds_to_timeWindow(info, seconds);
    return likwid_sysft_writemsr_field(device, reg, 49, 7, timeWindow);
}

static int sysFeatures_intel_rapl_energy_limit_2_enable_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_unusedinfo_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t enable;
    err = likwid_sysft_readmsr_field(device, reg, 47, 1, &enable);
    if (err < 0)
        return err;
    return sysFeatures_uint64_to_string(enable, value);
}

static int sysFeatures_intel_rapl_energy_limit_2_enable_setter(const LikwidDevice_t device, const char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_unusedinfo_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t enable;
    err = sysFeatures_string_to_uint64(value, &enable);
    if (err < 0)
        return err;
    return likwid_sysft_writemsr_field(device, reg, 47, 1, enable);
}

static int sysFeatures_intel_rapl_energy_limit_2_clamp_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_unusedinfo_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t clamp;
    err = likwid_sysft_readmsr_field(device, reg, 48, 1, &clamp);
    if (err < 0)
        return err;
    return sysFeatures_uint64_to_string(clamp, value);
}

static int sysFeatures_intel_rapl_energy_limit_2_clamp_setter(const LikwidDevice_t device, const char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = getset_unusedinfo_check(device, value, info, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t clamp;
    err = sysFeatures_string_to_uint64(value, &clamp);
    if (err < 0)
        return err;
    return likwid_sysft_writemsr_field(device, reg, 48, 1, clamp);
}

static int sysFeatures_intel_rapl_info_tdp(const LikwidDevice_t device, char** value, uint32_t reg)
{
    int err = getset_check(device, value, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t powerUnits;
    err = likwid_sysft_readmsr_field(device, reg, 0, 15, &powerUnits);
    if (err < 0)
        return err;
    return sysFeatures_double_to_string((double)powerUnits * intel_rapl_pkg_info.powerUnit, value);
}

static int sysFeatures_intel_rapl_info_min_power(const LikwidDevice_t device, char** value, uint32_t reg)
{
    int err = getset_check(device, value, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t powerUnits;
    err = likwid_sysft_readmsr_field(device, reg, 16, 15, &powerUnits);
    if (err < 0)
        return err;
    const double watts = (double)powerUnits * intel_rapl_pkg_info.powerUnit;
    return sysFeatures_double_to_string(watts, value);
}

static int sysFeatures_intel_rapl_info_max_power(const LikwidDevice_t device, char** value, uint32_t reg)
{
    int err = getset_check(device, value, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t powerUnits;
    err = likwid_sysft_readmsr_field(device, reg, 32, 15, &powerUnits);
    if (err < 0)
        return err;
    const double watts = (double)powerUnits * intel_rapl_pkg_info.powerUnit;
    return sysFeatures_double_to_string(watts, value);
}

static int sysFeatures_intel_rapl_info_max_time(const LikwidDevice_t device, char** value, uint32_t reg)
{
    int err = getset_check(device, value, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t timeUnits;
    err = likwid_sysft_readmsr_field(device, reg, 48, 7, &timeUnits);
    if (err < 0)
        return err;
    const double seconds = (double)timeUnits * intel_rapl_pkg_info.timeUnit;
    return sysFeatures_double_to_string(seconds, value);
}

static int sysFeatures_intel_rapl_policy_getter(const LikwidDevice_t device, char** value, uint32_t reg)
{
    int err = getset_check(device, value, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t policy;
    err = likwid_sysft_readmsr_field(device, reg, 0, 5, &policy);
    if (err < 0)
        return err;
    return sysFeatures_uint64_to_string(policy, value);
}

static int sysFeatures_intel_rapl_policy_setter(const LikwidDevice_t device, const char* value, uint32_t reg)
{
    int err = getset_check(device, value, DEVICE_TYPE_SOCKET);
    if (err < 0)
        return err;
    uint64_t policy;
    err = sysFeatures_string_to_uint64(value, &policy);
    if (err < 0)
        return err;
    return likwid_sysft_writemsr_field(device, reg, 0, 5, policy);
}

/*********************************************************************************************************************/
/*                          Intel RAPL (PKG domain)                                                                  */
/*********************************************************************************************************************/

static int pkg_test_testFunc(uint64_t msrData, void *)
{
    if (intel_rapl_pkg_info.powerUnit == 0 && intel_rapl_pkg_info.energyUnit == 0 && intel_rapl_pkg_info.timeUnit == 0)
    {
        intel_rapl_pkg_info.powerUnit = 1.0 / (1 << field64(msrData, 0, 4));
        intel_rapl_pkg_info.energyUnit = 1.0 / (1 << field64(msrData, 8, 5));
        intel_rapl_pkg_info.timeUnit = 1.0 / (1 << field64(msrData, 16, 4));
    }
    return 1;
}

int intel_rapl_pkg_test(void)
{
    return likwid_sysft_foreach_socket_testmsr_cb(MSR_RAPL_POWER_UNIT, pkg_test_testFunc, NULL);
}

static int pkg_limit_test_lock_testFunc(uint64_t msrData, void *)
{
    return field64(msrData, 63, 1);
}

int intel_rapl_pkg_limit_test_lock(void)
{
    return likwid_sysft_foreach_socket_testmsr_cb(MSR_PKG_RAPL_POWER_LIMIT, pkg_limit_test_lock_testFunc, NULL);
}

int sysFeatures_intel_pkg_energy_status_test(void)
{
    return likwid_sysft_foreach_socket_testmsr(MSR_PKG_ENERGY_STATUS);
}

int sysFeatures_intel_pkg_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_status_getter(device, value, MSR_PKG_ENERGY_STATUS, &intel_rapl_pkg_info);
}

int sysFeatures_intel_pkg_energy_limit_test(void)
{
    return likwid_sysft_foreach_socket_testmsr(MSR_PKG_RAPL_POWER_LIMIT);
}

int sysFeatures_intel_pkg_energy_limit_1_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info);
}
int sysFeatures_intel_pkg_energy_limit_1_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info);
}
int sysFeatures_intel_pkg_energy_limit_1_time_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_time_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info);
}
int sysFeatures_intel_pkg_energy_limit_1_time_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_time_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info);
}
int sysFeatures_intel_pkg_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_enable_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info);
}
int sysFeatures_intel_pkg_energy_limit_1_enable_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_enable_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info);
}
int sysFeatures_intel_pkg_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_clamp_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info);
}
int sysFeatures_intel_pkg_energy_limit_1_clamp_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_clamp_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info);
}



int sysFeatures_intel_pkg_energy_limit_2_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_2_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info);
}
int sysFeatures_intel_pkg_energy_limit_2_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_2_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info);
}
int sysFeatures_intel_pkg_energy_limit_2_time_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_2_time_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info);
}
int sysFeatures_intel_pkg_energy_limit_2_time_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_2_time_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info);
}
int sysFeatures_intel_pkg_energy_limit_2_enable_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_2_enable_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info);
}
int sysFeatures_intel_pkg_energy_limit_2_enable_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_2_enable_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info);
}
int sysFeatures_intel_pkg_energy_limit_2_clamp_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_2_clamp_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info);
}
int sysFeatures_intel_pkg_energy_limit_2_clamp_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_2_clamp_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info);
}

int sysFeatures_intel_pkg_info_test(void)
{
    return likwid_sysft_foreach_socket_testmsr(MSR_PKG_POWER_INFO);
}

int sysFeatures_intel_pkg_info_tdp(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_info_tdp(device, value, MSR_PKG_POWER_INFO);
}

int sysFeatures_intel_pkg_info_min_power(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_info_min_power(device, value, MSR_PKG_POWER_INFO);
}

int sysFeatures_intel_pkg_info_max_power(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_info_max_power(device, value, MSR_PKG_POWER_INFO);
}

int sysFeatures_intel_pkg_info_max_time(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_info_max_time(device, value, MSR_PKG_POWER_INFO);
}


/*********************************************************************************************************************/
/*                          Intel RAPL (DRAM domain)                                                                 */
/*********************************************************************************************************************/

static int dram_test_testFunc(uint64_t msrData, void *)
{
    if (intel_rapl_dram_info.powerUnit == 0 && intel_rapl_dram_info.energyUnit == 0 && intel_rapl_dram_info.timeUnit == 0)
    {
        CpuInfo_t info = get_cpuInfo();
        intel_rapl_dram_info.powerUnit = 1.0 / (1 << field64(msrData, 0, 4));
        intel_rapl_dram_info.energyUnit = 1.0 / (1 << field64(msrData, 8, 5));
        intel_rapl_dram_info.timeUnit = 1.0 / (1 << field64(msrData, 16, 4));
        if ((info->model == HASWELL_EP) ||
            (info->model == HASWELL_M1) ||
            (info->model == HASWELL_M2) ||
            (info->model == BROADWELL_D) ||
            (info->model == BROADWELL_E) ||
            (info->model == SKYLAKEX) ||
            (info->model == ICELAKEX1) ||
            (info->model == ICELAKEX2) ||
            (info->model == SAPPHIRERAPIDS) ||
            (info->model == XEON_PHI_KNL) ||
            (info->model == XEON_PHI_KML))
        {
            intel_rapl_dram_info.energyUnit = 15.3e-6;
        }
    }
    return 1;
}

int intel_rapl_dram_test(void)
{
    return likwid_sysft_foreach_socket_testmsr_cb(MSR_RAPL_POWER_UNIT, dram_test_testFunc, NULL);
}


int sysFeatures_intel_dram_energy_status_test(void)
{
    return likwid_sysft_foreach_socket_testmsr(MSR_DRAM_ENERGY_STATUS);
}

int sysFeatures_intel_dram_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_status_getter(device, value, MSR_DRAM_ENERGY_STATUS, &intel_rapl_dram_info);
}

int sysFeatures_intel_dram_energy_limit_test(void)
{
    return likwid_sysft_foreach_socket_testmsr(MSR_DRAM_RAPL_POWER_LIMIT);
}

static int dram_limit_test_lock_testFunc(uint64_t msrData, void *)
{
    return field64(msrData, 31, 1);
}
int intel_rapl_dram_limit_test_lock(void)
{
    return likwid_sysft_foreach_socket_testmsr_cb(MSR_DRAM_RAPL_POWER_LIMIT, dram_limit_test_lock_testFunc, NULL);
}

int sysFeatures_intel_dram_energy_limit_1_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_getter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &intel_rapl_dram_info);
}
int sysFeatures_intel_dram_energy_limit_1_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_setter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &intel_rapl_dram_info);
}
int sysFeatures_intel_dram_energy_limit_1_time_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_time_getter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &intel_rapl_dram_info);
}
int sysFeatures_intel_dram_energy_limit_1_time_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_time_setter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &intel_rapl_dram_info);
}
int sysFeatures_intel_dram_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_enable_getter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &intel_rapl_dram_info);
}
int sysFeatures_intel_dram_energy_limit_1_enable_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_enable_setter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &intel_rapl_dram_info);
}
int sysFeatures_intel_dram_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_clamp_getter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &intel_rapl_dram_info);
}
int sysFeatures_intel_dram_energy_limit_1_clamp_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_clamp_setter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &intel_rapl_dram_info);
}

int sysFeatures_intel_dram_info_test(void)
{
    return likwid_sysft_foreach_socket_testmsr(MSR_DRAM_POWER_INFO);
}

int sysFeatures_intel_dram_info_tdp(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_info_tdp(device, value, MSR_DRAM_POWER_INFO);
}

int sysFeatures_intel_dram_info_min_power(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_info_min_power(device, value, MSR_DRAM_POWER_INFO);
}

int sysFeatures_intel_dram_info_max_power(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_info_max_power(device, value, MSR_DRAM_POWER_INFO);
}

int sysFeatures_intel_dram_info_max_time(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_info_max_time(device, value, MSR_DRAM_POWER_INFO);
}


/*********************************************************************************************************************/
/*                          Intel RAPL (PSYS or PLATFORM domain)                                                     */
/*********************************************************************************************************************/

static int psys_test_testFunc(uint64_t msrData, void *)
{
    if (intel_rapl_psys_info.powerUnit == 0 && intel_rapl_psys_info.energyUnit == 0 && intel_rapl_psys_info.timeUnit == 0)
    {
        CpuInfo_t info = get_cpuInfo();
        intel_rapl_psys_info.powerUnit = 1.0 / (1 << field64(msrData, 0, 4));
        intel_rapl_psys_info.energyUnit = 1.0 / (1 << field64(msrData, 8, 5));
        if (info->model == SAPPHIRERAPIDS)
        {
            intel_rapl_psys_info.energyUnit = 1.0;
        }
        intel_rapl_psys_info.timeUnit = 1.0 / (1 << field64(msrData, 16, 4));
    }
    return 1;
}

int intel_rapl_psys_test(void)
{
    return likwid_sysft_foreach_socket_testmsr_cb(MSR_RAPL_POWER_UNIT, psys_test_testFunc, NULL);
}

int sysFeatures_intel_psys_energy_status_test(void)
{
    return likwid_sysft_foreach_socket_testmsr(MSR_PLATFORM_ENERGY_STATUS);
}

int sysFeatures_intel_psys_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_status_getter(device, value, MSR_PLATFORM_ENERGY_STATUS, &intel_rapl_psys_info);
}

int sysFeatures_intel_psys_energy_limit_test(void)
{
    return likwid_sysft_foreach_socket_testmsr(MSR_PLATFORM_POWER_LIMIT);
}

static int psys_limit_test_lock_testFunc(uint64_t msrData, void *)
{
    return field64(msrData, 63, 1);
}

int intel_rapl_psys_limit_test_lock(void)
{
    return likwid_sysft_foreach_socket_testmsr_cb(MSR_PLATFORM_POWER_LIMIT, psys_limit_test_lock_testFunc, NULL);
}

int sysFeatures_intel_psys_energy_limit_1_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info);
}
int sysFeatures_intel_psys_energy_limit_1_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info);
}
int sysFeatures_intel_psys_energy_limit_1_time_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_time_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info);
}
int sysFeatures_intel_psys_energy_limit_1_time_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_time_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info);
}
int sysFeatures_intel_psys_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_enable_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info);
}
int sysFeatures_intel_psys_energy_limit_1_enable_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_enable_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info);
}
int sysFeatures_intel_psys_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_clamp_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info);
}
int sysFeatures_intel_psys_energy_limit_1_clamp_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_clamp_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info);
}


int sysFeatures_intel_psys_energy_limit_2_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_2_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info);
}
int sysFeatures_intel_psys_energy_limit_2_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_2_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info);
}
int sysFeatures_intel_psys_energy_limit_2_time_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_2_time_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info);
}
int sysFeatures_intel_psys_energy_limit_2_time_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_2_time_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info);
}
int sysFeatures_intel_psys_energy_limit_2_enable_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_2_enable_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info);
}
int sysFeatures_intel_psys_energy_limit_2_enable_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_2_enable_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info);
}
int sysFeatures_intel_psys_energy_limit_2_clamp_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_2_clamp_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info);
}
int sysFeatures_intel_psys_energy_limit_2_clamp_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_2_clamp_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info);
}

/*********************************************************************************************************************/
/*                          Intel RAPL (PP0 domain)                                                                  */
/*********************************************************************************************************************/

static int pp0_test_testFunc(uint64_t msrData, void *)
{
    if (intel_rapl_pp0_info.powerUnit == 0 && intel_rapl_pp0_info.energyUnit == 0 && intel_rapl_pp0_info.timeUnit == 0)
    {
        intel_rapl_pp0_info.powerUnit = 1.0 / (1 << field64(msrData, 0, 4));
        intel_rapl_pp0_info.energyUnit = 1.0 / (1 << field64(msrData, 8, 5));
        intel_rapl_pp0_info.timeUnit = 1.0 / (1 << field64(msrData, 16, 4));
    }
    return 1;
}

int intel_rapl_pp0_test(void)
{
    return likwid_sysft_foreach_socket_testmsr_cb(MSR_RAPL_POWER_UNIT, pp0_test_testFunc, NULL);
}

int sysFeatures_intel_pp0_energy_status_test(void)
{
    return likwid_sysft_foreach_socket_testmsr(MSR_PP0_ENERGY_STATUS);
}

int sysFeatures_intel_pp0_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_status_getter(device, value, MSR_PP0_ENERGY_STATUS, &intel_rapl_pp0_info);
}

int sysFeatures_intel_pp0_energy_limit_test(void)
{
    return likwid_sysft_foreach_socket_testmsr(MSR_PP0_RAPL_POWER_LIMIT);
}

static int pp0_limit_test_lock_testFunc(uint64_t msrData, void *)
{
    return field64(msrData, 31, 1);
}

int intel_rapl_pp0_limit_test_lock(void)
{
    return likwid_sysft_foreach_socket_testmsr_cb(MSR_PP0_RAPL_POWER_LIMIT, pp0_limit_test_lock_testFunc, NULL);
}

int sysFeatures_intel_pp0_energy_limit_1_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_getter(device, value, MSR_PP0_RAPL_POWER_LIMIT, &intel_rapl_pp0_info);
}
int sysFeatures_intel_pp0_energy_limit_1_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_setter(device, value, MSR_PP0_RAPL_POWER_LIMIT, &intel_rapl_pp0_info);
}
int sysFeatures_intel_pp0_energy_limit_1_time_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_time_getter(device, value, MSR_PP0_RAPL_POWER_LIMIT, &intel_rapl_pp0_info);
}
int sysFeatures_intel_pp0_energy_limit_1_time_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_time_setter(device, value, MSR_PP0_RAPL_POWER_LIMIT, &intel_rapl_pp0_info);
}
int sysFeatures_intel_pp0_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_enable_getter(device, value, MSR_PP0_RAPL_POWER_LIMIT, &intel_rapl_pp0_info);
}
int sysFeatures_intel_pp0_energy_limit_1_enable_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_enable_setter(device, value, MSR_PP0_RAPL_POWER_LIMIT, &intel_rapl_pp0_info);
}
int sysFeatures_intel_pp0_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_clamp_getter(device, value, MSR_PP0_RAPL_POWER_LIMIT, &intel_rapl_pp0_info);
}
int sysFeatures_intel_pp0_energy_limit_1_clamp_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_clamp_setter(device, value, MSR_PP0_RAPL_POWER_LIMIT, &intel_rapl_pp0_info);
}
int sysFeatures_intel_pp0_policy_test(void)
{
    return likwid_sysft_foreach_socket_testmsr(MSR_PP0_ENERGY_POLICY);
}
int sysFeatures_intel_pp0_policy_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_policy_getter(device, value, MSR_PP0_ENERGY_POLICY);
}
int sysFeatures_intel_pp0_policy_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_policy_setter(device, value, MSR_PP0_ENERGY_POLICY);
}


/*********************************************************************************************************************/
/*                          Intel RAPL (PP1 domain)                                                                  */
/*********************************************************************************************************************/

static int pp1_test_testFunc(uint64_t msrData, void *)
{
    if (intel_rapl_pp1_info.powerUnit == 0 && intel_rapl_pp1_info.energyUnit == 0 && intel_rapl_pp1_info.timeUnit == 0)
    {
        intel_rapl_pp1_info.powerUnit = 1.0 / (1 << field64(msrData, 0, 4));
        intel_rapl_pp1_info.energyUnit = 1.0 / (1 << field64(msrData, 8, 5));
        intel_rapl_pp1_info.timeUnit = 1.0 / (1 << field64(msrData, 16, 4));
    }
    return 1;
}

int intel_rapl_pp1_test(void)
{
    return likwid_sysft_foreach_socket_testmsr_cb(MSR_RAPL_POWER_UNIT, pp1_test_testFunc, NULL);
}

int sysFeatures_intel_pp1_energy_status_test(void)
{
    return likwid_sysft_foreach_socket_testmsr(MSR_PP1_ENERGY_STATUS);
}

int sysFeatures_intel_pp1_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_status_getter(device, value, MSR_PP1_ENERGY_STATUS, &intel_rapl_pp1_info);
}

int sysFeatures_intel_pp1_energy_limit_test(void)
{
    return likwid_sysft_foreach_socket_testmsr(MSR_PP1_RAPL_POWER_LIMIT);
}

static int pp1_limit_test_lock(uint64_t msrData, void *)
{
    return field64(msrData, 31, 1);
}

int intel_rapl_pp1_limit_test_lock(void)
{
    return likwid_sysft_foreach_socket_testmsr_cb(MSR_PP1_RAPL_POWER_LIMIT, pp1_limit_test_lock, NULL);
}

int sysFeatures_intel_pp1_energy_limit_1_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_getter(device, value, MSR_PP1_RAPL_POWER_LIMIT, &intel_rapl_pp1_info);
}
int sysFeatures_intel_pp1_energy_limit_1_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_setter(device, value, MSR_PP1_RAPL_POWER_LIMIT, &intel_rapl_pp1_info);
}
int sysFeatures_intel_pp1_energy_limit_1_time_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_time_getter(device, value, MSR_PP1_RAPL_POWER_LIMIT, &intel_rapl_pp1_info);
}
int sysFeatures_intel_pp1_energy_limit_1_time_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_time_setter(device, value, MSR_PP1_RAPL_POWER_LIMIT, &intel_rapl_pp1_info);
}
int sysFeatures_intel_pp1_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_enable_getter(device, value, MSR_PP1_RAPL_POWER_LIMIT, &intel_rapl_pp1_info);
}
int sysFeatures_intel_pp1_energy_limit_1_enable_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_enable_setter(device, value, MSR_PP1_RAPL_POWER_LIMIT, &intel_rapl_pp1_info);
}
int sysFeatures_intel_pp1_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_energy_limit_1_clamp_getter(device, value, MSR_PP1_RAPL_POWER_LIMIT, &intel_rapl_pp1_info);
}
int sysFeatures_intel_pp1_energy_limit_1_clamp_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_energy_limit_1_clamp_setter(device, value, MSR_PP1_RAPL_POWER_LIMIT, &intel_rapl_pp1_info);
}


int sysFeatures_intel_pp1_policy_test(void)
{
    return likwid_sysft_foreach_socket_testmsr(MSR_PP1_ENERGY_POLICY);
}
int sysFeatures_intel_pp1_policy_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_intel_rapl_policy_getter(device, value, MSR_PP1_ENERGY_POLICY);
}
int sysFeatures_intel_pp1_policy_setter(const LikwidDevice_t device, const char* value)
{
    return sysFeatures_intel_rapl_policy_setter(device, value, MSR_PP1_ENERGY_POLICY);
}


/* Init function */

int sysFeatures_init_intel_rapl(_SysFeatureList* out)
{
    int err = 0;
    if (intel_rapl_pkg_test())
    {
        DEBUG_PRINT(DEBUGLEV_INFO, Register Intel RAPL PKG domain);
        if (intel_rapl_pkg_limit_test_lock() > 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, Intel RAPL PKG domain locked);
            for (int i = 0; i < intel_rapl_pkg_feature_list.num_features; i++)
            {
                intel_rapl_pkg_feature_list.features[i].setter = NULL;
            }
        }
        err = register_features(out, &intel_rapl_pkg_feature_list);
        if (err < 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, RAPL domain PKG not supported);
        }
    }
    else
    {
        DEBUG_PRINT(DEBUGLEV_INFO, RAPL domain PKG not supported);
    }
    if (intel_rapl_dram_test())
    {
        DEBUG_PRINT(DEBUGLEV_INFO, Register Intel RAPL DRAM domain);
        if (intel_rapl_dram_limit_test_lock() > 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, Intel RAPL DRAM domain locked);
            for (int i = 0; i < intel_rapl_dram_feature_list.num_features; i++)
            {
                intel_rapl_dram_feature_list.features[i].setter = NULL;
            }
        }
        err = register_features(out, &intel_rapl_dram_feature_list);
        if (err < 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, RAPL domain DRAM not supported);
        }
    }
    else
    {
        DEBUG_PRINT(DEBUGLEV_INFO, RAPL domain DRAM not supported);
    }
    if (intel_rapl_pp0_test())
    {
        DEBUG_PRINT(DEBUGLEV_INFO, Register Intel RAPL PP0 domain);
        if (intel_rapl_pp0_limit_test_lock() > 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, Intel RAPL PP0 domain locked);
            for (int i = 0; i < intel_rapl_pp0_feature_list.num_features; i++)
            {
                intel_rapl_pp0_feature_list.features[i].setter = NULL;
            }
        }
        err = register_features(out, &intel_rapl_pp0_feature_list);
        if (err < 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, RAPL domain PP0 not supported);
        }
    }
    else
    {
        DEBUG_PRINT(DEBUGLEV_INFO, RAPL domain PP0 not supported);
    }
    if (intel_rapl_pp1_test())
    {
        DEBUG_PRINT(DEBUGLEV_INFO, Register Intel RAPL PP1 domain);
        if (intel_rapl_pp1_limit_test_lock() > 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, Intel RAPL PP1 domain locked);
            for (int i = 0; i < intel_rapl_pp1_feature_list.num_features; i++)
            {
                intel_rapl_pp1_feature_list.features[i].setter = NULL;
            }
        }
        err = register_features(out, &intel_rapl_pp1_feature_list);
        if (err < 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, RAPL domain PP1 not supported);
        }
    }
    else
    {
        DEBUG_PRINT(DEBUGLEV_INFO, RAPL domain PP1 not supported);
    }
    if (intel_rapl_psys_test())
    {
        DEBUG_PRINT(DEBUGLEV_INFO, Register Intel RAPL PSYS domain);
        if (intel_rapl_psys_limit_test_lock() > 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, Intel RAPL PSYS domain locked);
            for (int i = 0; i < intel_rapl_psys_feature_list.num_features; i++)
            {
                intel_rapl_psys_feature_list.features[i].setter = NULL;
            }
        }
        err = register_features(out, &intel_rapl_psys_feature_list);
        if (err < 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, RAPL domain PSYS not supported);
        }
    }
    else
    {
        DEBUG_PRINT(DEBUGLEV_INFO, RAPL domain PSYS not supported);
    }
    return 0;
}
