/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_intel_rapl.c
 *
 *      Description:  Interface to control Intel RAPL for the sysFeatures component
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Authors:  Thomas Gruber (tg), thomas.roehl@googlemail.com
 *                Michael Panzlaff, michael.panzlaff@fau.de
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
#include <types.h>
#include <topology.h>

#include <sysFeatures_common_rapl.h>
#include "debug.h"

static RaplDomainInfo intel_rapl_pkg_info = {0, 0, 0};
static RaplDomainInfo intel_rapl_dram_info = {0, 0, 0};
static RaplDomainInfo intel_rapl_psys_info = {0, 0, 0};
static RaplDomainInfo intel_rapl_pp0_info = {0, 0, 0};
static RaplDomainInfo intel_rapl_pp1_info = {0, 0, 0};

static cerr_t intel_rapl_energy_status_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t energy;
    if (likwid_sysft_readmsr_field(device, reg, 0, 32, &energy))
        return ERROR_WRAP();
    return ERROR_WRAP_CALL(likwid_sysft_double_to_string((double)energy * info->energyUnit, value));
}

static cerr_t intel_rapl_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    (void)info;
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t enable;
    if (likwid_sysft_readmsr_field(device, reg, 15, 1, &enable))
        return ERROR_WRAP();
    return ERROR_WRAP_CALL(likwid_sysft_uint64_to_string(enable, value));
}

static cerr_t intel_rapl_energy_limit_1_enable_setter(const LikwidDevice_t device, const char* value, uint32_t reg, const RaplDomainInfo* info)
{
    (void)info;
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t enable;
    if (likwid_sysft_string_to_uint64(value, &enable))
        return ERROR_WRAP();
    return ERROR_WRAP_CALL(likwid_sysft_writemsr_field(device, reg, 15, 1, enable));
}

static cerr_t intel_rapl_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    (void)info;
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t clamp;
    if (likwid_sysft_readmsr_field(device, reg, 16, 1, &clamp))
        return ERROR_WRAP();
    return ERROR_WRAP_CALL(likwid_sysft_uint64_to_string(clamp, value));
}

static cerr_t intel_rapl_energy_limit_1_clamp_setter(const LikwidDevice_t device, const char* value, uint32_t reg, const RaplDomainInfo* info)
{
    (void)info;
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t clamp;
    if (likwid_sysft_string_to_uint64(value, &clamp))
        return ERROR_WRAP();
    return ERROR_WRAP_CALL(likwid_sysft_writemsr_field(device, reg, 16, 1, clamp));
}

static cerr_t intel_rapl_energy_limit_1_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t powerUnits;
    if (likwid_sysft_readmsr_field(device, reg, 0, 15, &powerUnits))
        return ERROR_WRAP();
    const double watts = (double)powerUnits * info->powerUnit;
    return ERROR_WRAP_CALL(likwid_sysft_double_to_string(watts, value));
}

static cerr_t intel_rapl_energy_limit_1_setter(const LikwidDevice_t device, const char* value, uint32_t reg, const RaplDomainInfo* info)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    double watts;
    if (likwid_sysft_string_to_double(value, &watts))
        return ERROR_WRAP();
    const uint64_t powerUnits = MIN((uint64_t)round(watts / info->powerUnit), 0x7FFF);
    return ERROR_WRAP_CALL(likwid_sysft_writemsr_field(device, reg, 0, 15, powerUnits));
}

static cerr_t intel_rapl_energy_limit_1_time_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t timeWindow;
    if (likwid_sysft_readmsr_field(device, reg, 17, 7, &timeWindow))
        return ERROR_WRAP();
    const double seconds = timeWindow_to_seconds(info, timeWindow);
    return ERROR_WRAP_CALL(likwid_sysft_double_to_string(seconds, value));
}

static cerr_t intel_rapl_energy_limit_1_time_setter(const LikwidDevice_t device, const char* value, uint32_t reg, const RaplDomainInfo* info)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    double seconds;
    if (likwid_sysft_string_to_double(value, &seconds))
        return ERROR_WRAP();
    const uint64_t timeWindow = seconds_to_timeWindow(info, seconds);
    return ERROR_WRAP_CALL(likwid_sysft_writemsr_field(device, reg, 17, 7, timeWindow));
}

static cerr_t intel_rapl_energy_limit_2_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t powerUnits;
    if (likwid_sysft_readmsr_field(device, reg, 32, 15, &powerUnits))
        return ERROR_WRAP();
    const double watts = (double)powerUnits * info->powerUnit;
    return ERROR_WRAP_CALL(likwid_sysft_double_to_string(watts, value));
}

static cerr_t intel_rapl_energy_limit_2_setter(const LikwidDevice_t device, const char* value, uint32_t reg, const RaplDomainInfo* info)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    double watts;
    if (likwid_sysft_string_to_double(value, &watts))
        return ERROR_WRAP();
    const uint64_t powerUnits = MIN((uint64_t)round(watts / info->powerUnit), 0x7FFF);
    return ERROR_WRAP_CALL(likwid_sysft_writemsr_field(device, reg, 32, 15, powerUnits));
}

static cerr_t intel_rapl_energy_limit_2_time_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t timeWindow;
    if (likwid_sysft_readmsr_field(device, reg, 49, 7, &timeWindow))
        return ERROR_WRAP();
    const double seconds = timeWindow_to_seconds(info, timeWindow);
    return ERROR_WRAP_CALL(likwid_sysft_double_to_string(seconds, value));
}

static cerr_t intel_rapl_energy_limit_2_time_setter(const LikwidDevice_t device, const char* value, uint32_t reg, const RaplDomainInfo* info)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    double seconds;
    if (likwid_sysft_string_to_double(value, &seconds))
        return ERROR_WRAP();
    const uint64_t timeWindow = seconds_to_timeWindow(info, seconds);
    return ERROR_WRAP_CALL(likwid_sysft_writemsr_field(device, reg, 49, 7, timeWindow));
}

static cerr_t intel_rapl_energy_limit_2_enable_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    (void)info;
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t enable;
    if (likwid_sysft_readmsr_field(device, reg, 47, 1, &enable))
        return ERROR_WRAP();
    return ERROR_WRAP_CALL(likwid_sysft_uint64_to_string(enable, value));
}

static cerr_t intel_rapl_energy_limit_2_enable_setter(const LikwidDevice_t device, const char* value, uint32_t reg, const RaplDomainInfo* info)
{
    (void)info;
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t enable;
    if (likwid_sysft_string_to_uint64(value, &enable))
        return ERROR_WRAP();
    return ERROR_WRAP_CALL(likwid_sysft_writemsr_field(device, reg, 47, 1, enable));
}

static cerr_t intel_rapl_energy_limit_2_clamp_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    (void)info;
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t clamp;
    if (likwid_sysft_readmsr_field(device, reg, 48, 1, &clamp))
        return ERROR_WRAP();
    return ERROR_WRAP_CALL(likwid_sysft_uint64_to_string(clamp, value));
}

static cerr_t intel_rapl_energy_limit_2_clamp_setter(const LikwidDevice_t device, const char* value, uint32_t reg, const RaplDomainInfo* info)
{
    (void)info;
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t clamp;
    if (likwid_sysft_string_to_uint64(value, &clamp))
        return ERROR_WRAP();
    return ERROR_WRAP_CALL(likwid_sysft_writemsr_field(device, reg, 48, 1, clamp));
}

static cerr_t intel_rapl_info_tdp(const LikwidDevice_t device, char** value, uint32_t reg)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t powerUnits;
    if (likwid_sysft_readmsr_field(device, reg, 0, 15, &powerUnits))
        return ERROR_WRAP();
    return ERROR_WRAP_CALL(likwid_sysft_double_to_string((double)powerUnits * intel_rapl_pkg_info.powerUnit, value));
}

static cerr_t intel_rapl_info_min_power(const LikwidDevice_t device, char** value, uint32_t reg)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t powerUnits;
    if (likwid_sysft_readmsr_field(device, reg, 16, 15, &powerUnits))
        return ERROR_WRAP();
    const double watts = (double)powerUnits * intel_rapl_pkg_info.powerUnit;
    return ERROR_WRAP_CALL(likwid_sysft_double_to_string(watts, value));
}

static cerr_t intel_rapl_info_max_power(const LikwidDevice_t device, char** value, uint32_t reg)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t powerUnits;
    if (likwid_sysft_readmsr_field(device, reg, 32, 15, &powerUnits))
        return ERROR_WRAP();
    const double watts = (double)powerUnits * intel_rapl_pkg_info.powerUnit;
    return ERROR_WRAP_CALL(likwid_sysft_double_to_string(watts, value));
}

static cerr_t intel_rapl_info_max_time(const LikwidDevice_t device, char** value, uint32_t reg)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t timeUnits;
    if (likwid_sysft_readmsr_field(device, reg, 48, 7, &timeUnits))
        return ERROR_WRAP();
    const double seconds = (double)timeUnits * intel_rapl_pkg_info.timeUnit;
    return ERROR_WRAP_CALL(likwid_sysft_double_to_string(seconds, value));
}

static cerr_t intel_rapl_policy_getter(const LikwidDevice_t device, char** value, uint32_t reg)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t policy;
    if (likwid_sysft_readmsr_field(device, reg, 0, 5, &policy))
        return ERROR_WRAP();
    return ERROR_WRAP_CALL(likwid_sysft_uint64_to_string(policy, value));
}

static cerr_t intel_rapl_policy_setter(const LikwidDevice_t device, const char* value, uint32_t reg)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t policy;
    if (likwid_sysft_string_to_uint64(value, &policy))
        return ERROR_WRAP();
    return ERROR_WRAP_CALL(likwid_sysft_writemsr_field(device, reg, 0, 5, policy));
}

/*********************************************************************************************************************/
/*                          Intel RAPL (PKG domain)                                                                  */
/*********************************************************************************************************************/

static cerr_t pkg_test_testFunc(bool *ok, uint64_t msrData, void * value)
{
    (void)value;
    if (intel_rapl_pkg_info.powerUnit == 0 && intel_rapl_pkg_info.energyUnit == 0 && intel_rapl_pkg_info.timeUnit == 0)
    {
        intel_rapl_pkg_info.powerUnit = 1.0 / (1 << field64(msrData, 0, 4));
        intel_rapl_pkg_info.energyUnit = 1.0 / (1 << field64(msrData, 8, 5));
        intel_rapl_pkg_info.timeUnit = 1.0 / (1 << field64(msrData, 16, 4));
    }
    *ok = true;
    return NULL;
}

static cerr_t intel_rapl_pkg_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr_cb(ok, MSR_RAPL_POWER_UNIT, pkg_test_testFunc, NULL));
}

static cerr_t pkg_limit_test_lock_testFunc(bool *ok, uint64_t msrData, void * value)
{
    (void)value;
    *ok = field64(msrData, 63, 1);
    return NULL;
}

static cerr_t intel_rapl_pkg_limit_test_lock(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr_cb(ok, MSR_PKG_RAPL_POWER_LIMIT, pkg_limit_test_lock_testFunc, NULL));
}

static cerr_t intel_pkg_energy_status_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr(ok, MSR_PKG_ENERGY_STATUS));
}

static cerr_t intel_pkg_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_status_getter(device, value, MSR_PKG_ENERGY_STATUS, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_energy_limit_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr(ok, MSR_PKG_RAPL_POWER_LIMIT));
}

static cerr_t intel_pkg_energy_limit_1_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_energy_limit_1_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_energy_limit_1_time_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_time_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_energy_limit_1_time_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_time_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_enable_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_energy_limit_1_enable_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_enable_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_clamp_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_energy_limit_1_clamp_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_clamp_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_energy_limit_2_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_2_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_energy_limit_2_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_2_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_energy_limit_2_time_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_2_time_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_energy_limit_2_time_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_2_time_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_energy_limit_2_enable_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_2_enable_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_energy_limit_2_enable_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_2_enable_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_energy_limit_2_clamp_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_2_clamp_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_energy_limit_2_clamp_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_2_clamp_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &intel_rapl_pkg_info));
}

static cerr_t intel_pkg_info_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr(ok, MSR_PKG_POWER_INFO));
}

static cerr_t intel_pkg_info_tdp(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_info_tdp(device, value, MSR_PKG_POWER_INFO));
}

static cerr_t intel_pkg_info_min_power(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_info_min_power(device, value, MSR_PKG_POWER_INFO));
}

static cerr_t intel_pkg_info_max_power(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_info_max_power(device, value, MSR_PKG_POWER_INFO));
}

static cerr_t intel_pkg_info_max_time(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_info_max_time(device, value, MSR_PKG_POWER_INFO));
}

static _SysFeature intel_rapl_pkg_features[] = {
    {"pkg_energy", "rapl", "Current energy consumtion (PKG domain)", intel_pkg_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, intel_pkg_energy_status_test, "J"},
    {"pkg_tdp", "rapl", "Thermal Spec Power", intel_pkg_info_tdp, NULL, DEVICE_TYPE_SOCKET, intel_pkg_info_test, "W"},
    {"pkg_min_limit", "rapl", "Minimum Power", intel_pkg_info_min_power, NULL, DEVICE_TYPE_SOCKET, intel_pkg_info_test, "W"},
    {"pkg_max_limit", "rapl", "Maximum Power", intel_pkg_info_max_power, NULL, DEVICE_TYPE_SOCKET, intel_pkg_info_test, "W"},
    {"pkg_max_time", "rapl", "Maximum Time", intel_pkg_info_max_time, NULL, DEVICE_TYPE_SOCKET, intel_pkg_info_test, "s"},
    {"pkg_limit_1", "rapl", "Long-term energy limit (PKG domain)", intel_pkg_energy_limit_1_getter, intel_pkg_energy_limit_1_setter, DEVICE_TYPE_SOCKET, intel_pkg_energy_limit_test, "W"},
    {"pkg_limit_1_time", "rapl", "Long-term time window (PKG domain)", intel_pkg_energy_limit_1_time_getter, intel_pkg_energy_limit_1_time_setter, DEVICE_TYPE_SOCKET, intel_pkg_energy_limit_test, "s"},
    {"pkg_limit_1_enable", "rapl", "Status of long-term energy limit (PKG domain)", intel_pkg_energy_limit_1_enable_getter, intel_pkg_energy_limit_1_enable_setter, DEVICE_TYPE_SOCKET, intel_pkg_energy_limit_test, "bool"},
    {"pkg_limit_1_clamp", "rapl", "Clamping status of long-term energy limit (PKG domain)", intel_pkg_energy_limit_1_clamp_getter, intel_pkg_energy_limit_1_clamp_setter, DEVICE_TYPE_SOCKET, intel_pkg_energy_limit_test, "bool"},
    {"pkg_limit_2", "rapl", "Short-term energy limit (PKG domain)", intel_pkg_energy_limit_2_getter, intel_pkg_energy_limit_2_setter, DEVICE_TYPE_SOCKET, intel_pkg_energy_limit_test, "W"},
    {"pkg_limit_2_time", "rapl", "Short-term time window (PKG domain)", intel_pkg_energy_limit_2_time_getter, intel_pkg_energy_limit_2_time_setter, DEVICE_TYPE_SOCKET, intel_pkg_energy_limit_test, "s"},
    {"pkg_limit_2_enable", "rapl", "Status of short-term energy limit (PKG domain)", intel_pkg_energy_limit_2_enable_getter, intel_pkg_energy_limit_2_enable_setter, DEVICE_TYPE_SOCKET, intel_pkg_energy_limit_test, "bool"},
    {"pkg_limit_2_clamp", "rapl", "Clamping status of short-term energy limit (PKG domain)", intel_pkg_energy_limit_2_clamp_getter, intel_pkg_energy_limit_2_clamp_setter, DEVICE_TYPE_SOCKET, intel_pkg_energy_limit_test, "bool"},
};

static const _SysFeatureList intel_rapl_pkg_feature_list = {
    .num_features = ARRAY_COUNT(intel_rapl_pkg_features),
    .tester = intel_rapl_pkg_test,
    .features = intel_rapl_pkg_features,
};

/*********************************************************************************************************************/
/*                          Intel RAPL (DRAM domain)                                                                 */
/*********************************************************************************************************************/

static cerr_t dram_test_testFunc(bool *ok, uint64_t msrData, void * value)
{
    (void)value;
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
            (info->model == XEON_PHI_KNL) ||
            (info->model == XEON_PHI_KML))
        {
            intel_rapl_dram_info.energyUnit = 15.3e-6;
        }
        else if (info->model == SAPPHIRERAPIDS || info->model == EMERALDRAPIDS)
        {
            intel_rapl_dram_info.energyUnit = 61e-6;
        }
    }

    *ok = true;
    return NULL;
}

static cerr_t intel_rapl_dram_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr_cb(ok, MSR_RAPL_POWER_UNIT, dram_test_testFunc, NULL));
}

static cerr_t intel_dram_energy_status_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr(ok, MSR_DRAM_ENERGY_STATUS));
}

static cerr_t intel_dram_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_status_getter(device, value, MSR_DRAM_ENERGY_STATUS, &intel_rapl_dram_info));
}

static cerr_t intel_dram_energy_limit_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr(ok, MSR_DRAM_RAPL_POWER_LIMIT));
}

static cerr_t dram_limit_test_lock_testFunc(bool *ok, uint64_t msrData, void * value)
{
    (void)value;
    *ok = field64(msrData, 31, 1);
    return NULL;
}
static cerr_t intel_rapl_dram_limit_test_lock(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr_cb(ok, MSR_DRAM_RAPL_POWER_LIMIT, dram_limit_test_lock_testFunc, NULL));
}

static cerr_t intel_dram_energy_limit_1_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_getter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &intel_rapl_dram_info));
}

static cerr_t intel_dram_energy_limit_1_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_setter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &intel_rapl_dram_info));
}

static cerr_t intel_dram_energy_limit_1_time_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_time_getter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &intel_rapl_dram_info));
}

static cerr_t intel_dram_energy_limit_1_time_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_time_setter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &intel_rapl_dram_info));
}

static cerr_t intel_dram_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_enable_getter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &intel_rapl_dram_info));
}

static cerr_t intel_dram_energy_limit_1_enable_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_enable_setter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &intel_rapl_dram_info));
}

static cerr_t intel_dram_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_clamp_getter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &intel_rapl_dram_info));
}

static cerr_t intel_dram_energy_limit_1_clamp_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_clamp_setter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &intel_rapl_dram_info));
}

static cerr_t intel_dram_info_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr(ok, MSR_DRAM_POWER_INFO));
}

static cerr_t intel_dram_info_tdp(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_info_tdp(device, value, MSR_DRAM_POWER_INFO));
}

static cerr_t intel_dram_info_min_power(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_info_min_power(device, value, MSR_DRAM_POWER_INFO));
}

static cerr_t intel_dram_info_max_power(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_info_max_power(device, value, MSR_DRAM_POWER_INFO));
}

static cerr_t intel_dram_info_max_time(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_info_max_time(device, value, MSR_DRAM_POWER_INFO));
}

static _SysFeature intel_rapl_dram_features[] = {
    {"dram_energy", "rapl", "Current energy consumtion (DRAM domain)", intel_dram_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, intel_dram_energy_status_test, "J"},
    {"dram_tdp", "rapl", "Thermal Spec Power", intel_dram_info_tdp, NULL, DEVICE_TYPE_SOCKET, intel_dram_info_test, "W"},
    {"dram_min_limit", "rapl", "Minimum Power", intel_dram_info_min_power, NULL, DEVICE_TYPE_SOCKET, intel_dram_info_test, "W"},
    {"dram_max_limit", "rapl", "Maximum Power", intel_dram_info_max_power, NULL, DEVICE_TYPE_SOCKET, intel_dram_info_test, "W"},
    {"dram_max_time", "rapl", "Maximum Time", intel_dram_info_max_time, NULL, DEVICE_TYPE_SOCKET, intel_dram_info_test, "s"},
    {"dram_limit", "rapl", "Long-term energy limit (DRAM domain)", intel_dram_energy_limit_1_getter, intel_dram_energy_limit_1_setter, DEVICE_TYPE_SOCKET, intel_dram_energy_limit_test, "W"},
    {"dram_limit_time", "rapl", "Long-term time window (DRAM domain)", intel_dram_energy_limit_1_time_getter, intel_dram_energy_limit_1_time_setter, DEVICE_TYPE_SOCKET, intel_dram_energy_limit_test, "s"},
    {"dram_limit_enable", "rapl", "Status of long-term energy limit (DRAM domain)", intel_dram_energy_limit_1_enable_getter, intel_dram_energy_limit_1_enable_setter, DEVICE_TYPE_SOCKET, intel_dram_energy_limit_test, "bool"},
    {"dram_limit_clamp", "rapl", "Clamping status of long-term energy limit (DRAM domain)", intel_dram_energy_limit_1_clamp_getter, intel_dram_energy_limit_1_clamp_setter, DEVICE_TYPE_SOCKET, intel_dram_energy_limit_test, "bool"},
};

static const _SysFeatureList intel_rapl_dram_feature_list = {
    .num_features = ARRAY_COUNT(intel_rapl_dram_features),
    .tester = intel_rapl_dram_test,
    .features = intel_rapl_dram_features,
};

/*********************************************************************************************************************/
/*                          Intel RAPL (PSYS or PLATFORM domain)                                                     */
/*********************************************************************************************************************/

static cerr_t psys_test_testFunc(bool *ok, uint64_t msrData, void * value)
{
    (void)value;
    if (intel_rapl_psys_info.powerUnit == 0 && intel_rapl_psys_info.energyUnit == 0 && intel_rapl_psys_info.timeUnit == 0)
    {
        CpuInfo_t info = get_cpuInfo();
        intel_rapl_psys_info.powerUnit = 1.0 / (1 << field64(msrData, 0, 4));
        intel_rapl_psys_info.energyUnit = 1.0 / (1 << field64(msrData, 8, 5));
        if (info->model == SAPPHIRERAPIDS || info->model == EMERALDRAPIDS)
        {
            intel_rapl_psys_info.energyUnit = 1.0;
        }
        intel_rapl_psys_info.timeUnit = 1.0 / (1 << field64(msrData, 16, 4));
    }
    *ok = true;
    return NULL;
}

static cerr_t intel_rapl_psys_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr_cb(ok, MSR_RAPL_POWER_UNIT, psys_test_testFunc, NULL));
}

static cerr_t intel_psys_energy_status_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr(ok, MSR_PLATFORM_ENERGY_STATUS));
}

static cerr_t intel_psys_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_status_getter(device, value, MSR_PLATFORM_ENERGY_STATUS, &intel_rapl_psys_info));
}

static cerr_t intel_psys_energy_limit_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr(ok, MSR_PLATFORM_POWER_LIMIT));
}

static cerr_t psys_limit_test_lock_testFunc(bool *ok, uint64_t msrData, void * value)
{
    (void)value;
    *ok = field64(msrData, 63, 1);
    return NULL;
}

static cerr_t intel_rapl_psys_limit_test_lock(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr_cb(ok, MSR_PLATFORM_POWER_LIMIT, psys_limit_test_lock_testFunc, NULL));
}

static cerr_t intel_psys_energy_limit_1_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info));
}

static cerr_t intel_psys_energy_limit_1_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info));
}

static cerr_t intel_psys_energy_limit_1_time_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_time_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info));
}

static cerr_t intel_psys_energy_limit_1_time_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_time_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info));
}

static cerr_t intel_psys_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_enable_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info));
}

static cerr_t intel_psys_energy_limit_1_enable_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_enable_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info));
}

static cerr_t intel_psys_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_clamp_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info));
}

static cerr_t intel_psys_energy_limit_1_clamp_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_clamp_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info));
}

static cerr_t intel_psys_energy_limit_2_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_2_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info));
}

static cerr_t intel_psys_energy_limit_2_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_2_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info));
}

static cerr_t intel_psys_energy_limit_2_time_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_2_time_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info));
}

static cerr_t intel_psys_energy_limit_2_time_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_2_time_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info));
}

static cerr_t intel_psys_energy_limit_2_enable_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_2_enable_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info));
}

static cerr_t intel_psys_energy_limit_2_enable_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_2_enable_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info));
}

static cerr_t intel_psys_energy_limit_2_clamp_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_2_clamp_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info));
}

static cerr_t intel_psys_energy_limit_2_clamp_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_2_clamp_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &intel_rapl_psys_info));
}

static _SysFeature intel_rapl_psys_features[] = {
    {"psys_energy", "rapl", "Current energy consumtion (PSYS domain)", intel_psys_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, intel_psys_energy_status_test, "J"},
    {"psys_limit_1", "rapl", "Long-term energy limit (PSYS domain)", intel_psys_energy_limit_1_getter, intel_psys_energy_limit_1_setter, DEVICE_TYPE_SOCKET, intel_psys_energy_limit_test, "W"},
    {"psys_limit_1_time", "rapl", "Long-term time window (PSYS domain)", intel_psys_energy_limit_1_time_getter, intel_psys_energy_limit_1_time_setter, DEVICE_TYPE_SOCKET, intel_psys_energy_limit_test, "s"},
    {"psys_limit_1_enable", "rapl", "Status of long-term energy limit (PSYS domain)", intel_psys_energy_limit_1_enable_getter, intel_psys_energy_limit_1_enable_setter, DEVICE_TYPE_SOCKET, intel_psys_energy_limit_test, "bool"},
    {"psys_limit_1_clamp", "rapl", "Clamping status of long-term energy limit (PSYS domain)", intel_psys_energy_limit_1_clamp_getter, intel_psys_energy_limit_1_clamp_setter, DEVICE_TYPE_SOCKET, intel_psys_energy_limit_test, "bool"},
    {"psys_limit_2", "rapl", "Short-term energy limit (PSYS domain)", intel_psys_energy_limit_2_getter, intel_psys_energy_limit_2_setter, DEVICE_TYPE_SOCKET, intel_psys_energy_limit_test, "W"},
    {"psys_limit_2_time", "rapl", "Short-term time window (PSYS domain)", intel_psys_energy_limit_2_time_getter, intel_psys_energy_limit_2_time_setter, DEVICE_TYPE_SOCKET, intel_psys_energy_limit_test, "s"},
    {"psys_limit_2_enable", "rapl", "Status of short-term energy limit (PSYS domain)", intel_psys_energy_limit_2_enable_getter, intel_psys_energy_limit_2_enable_setter, DEVICE_TYPE_SOCKET, intel_psys_energy_limit_test, "bool"},
    {"psys_limit_2_clamp", "rapl", "Clamping status of short-term energy limit (PSYS domain)", intel_psys_energy_limit_2_clamp_getter, intel_psys_energy_limit_2_clamp_setter, DEVICE_TYPE_SOCKET, intel_psys_energy_limit_test, "bool"},
};

static const _SysFeatureList intel_rapl_psys_feature_list = {
    .num_features = ARRAY_COUNT(intel_rapl_psys_features),
    .tester = intel_rapl_psys_test,
    .features = intel_rapl_psys_features,
};

/*********************************************************************************************************************/
/*                          Intel RAPL (PP0 domain)                                                                  */
/*********************************************************************************************************************/

static cerr_t pp0_test_testFunc(bool *ok, uint64_t msrData, void * value)
{
    (void)value;
    if (intel_rapl_pp0_info.powerUnit == 0 && intel_rapl_pp0_info.energyUnit == 0 && intel_rapl_pp0_info.timeUnit == 0)
    {
        intel_rapl_pp0_info.powerUnit = 1.0 / (1 << field64(msrData, 0, 4));
        intel_rapl_pp0_info.energyUnit = 1.0 / (1 << field64(msrData, 8, 5));
        intel_rapl_pp0_info.timeUnit = 1.0 / (1 << field64(msrData, 16, 4));
    }
    *ok = true;
    return NULL;
}

static cerr_t intel_rapl_pp0_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr_cb(ok, MSR_RAPL_POWER_UNIT, pp0_test_testFunc, NULL));
}

static cerr_t intel_pp0_energy_status_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr(ok, MSR_PP0_ENERGY_STATUS));
}

static cerr_t intel_pp0_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_status_getter(device, value, MSR_PP0_ENERGY_STATUS, &intel_rapl_pp0_info));
}

static cerr_t intel_pp0_energy_limit_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr(ok, MSR_PP0_RAPL_POWER_LIMIT));
}

static cerr_t pp0_limit_test_lock_testFunc(bool *ok, uint64_t msrData, void * value)
{
    (void)value;
    *ok = field64(msrData, 31, 1);
    return NULL;
}

static cerr_t intel_rapl_pp0_limit_test_lock(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr_cb(ok, MSR_PP0_RAPL_POWER_LIMIT, pp0_limit_test_lock_testFunc, NULL));
}

static cerr_t intel_pp0_energy_limit_1_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_getter(device, value, MSR_PP0_RAPL_POWER_LIMIT, &intel_rapl_pp0_info));
}

static cerr_t intel_pp0_energy_limit_1_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_setter(device, value, MSR_PP0_RAPL_POWER_LIMIT, &intel_rapl_pp0_info));
}

static cerr_t intel_pp0_energy_limit_1_time_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_time_getter(device, value, MSR_PP0_RAPL_POWER_LIMIT, &intel_rapl_pp0_info));
}

static cerr_t intel_pp0_energy_limit_1_time_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_time_setter(device, value, MSR_PP0_RAPL_POWER_LIMIT, &intel_rapl_pp0_info));
}

static cerr_t intel_pp0_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_enable_getter(device, value, MSR_PP0_RAPL_POWER_LIMIT, &intel_rapl_pp0_info));
}

static cerr_t intel_pp0_energy_limit_1_enable_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_enable_setter(device, value, MSR_PP0_RAPL_POWER_LIMIT, &intel_rapl_pp0_info));
}

static cerr_t intel_pp0_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_clamp_getter(device, value, MSR_PP0_RAPL_POWER_LIMIT, &intel_rapl_pp0_info));
}

static cerr_t intel_pp0_energy_limit_1_clamp_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_clamp_setter(device, value, MSR_PP0_RAPL_POWER_LIMIT, &intel_rapl_pp0_info));
}

static cerr_t intel_pp0_policy_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr(ok, MSR_PP0_ENERGY_POLICY));
}

static cerr_t intel_pp0_policy_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_policy_getter(device, value, MSR_PP0_ENERGY_POLICY));
}

static cerr_t intel_pp0_policy_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_policy_setter(device, value, MSR_PP0_ENERGY_POLICY));
}

static _SysFeature intel_rapl_pp0_features[] = {
    {"pp0_energy", "rapl", "Current energy consumtion (PP0 domain)", intel_pp0_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, intel_pp0_energy_status_test, "J"},
    {"pp0_limit", "rapl", "Long-term energy limit (PP0 domain)", intel_pp0_energy_limit_1_getter, intel_pp0_energy_limit_1_setter, DEVICE_TYPE_SOCKET, intel_pp0_energy_limit_test, "W"},
    {"pp0_limit_time", "rapl", "Long-term time window (PP0 domain)", intel_pp0_energy_limit_1_time_getter, intel_pp0_energy_limit_1_time_setter, DEVICE_TYPE_SOCKET, intel_pp0_energy_limit_test, "s"},
    {"pp0_limit_enable", "rapl", "Status of long-term energy limit (PP0 domain)", intel_pp0_energy_limit_1_enable_getter, intel_pp0_energy_limit_1_enable_setter, DEVICE_TYPE_SOCKET, intel_pp0_energy_limit_test, "bool"},
    {"pp0_limit_clamp", "rapl", "Clamping status of long-term energy limit (PP0 domain)", intel_pp0_energy_limit_1_clamp_getter, intel_pp0_energy_limit_1_clamp_setter, DEVICE_TYPE_SOCKET, intel_pp0_energy_limit_test, "bool"},
    {"pp0_policy", "rapl", "Balance Power Policy (PP0 domain)", intel_pp0_policy_getter, intel_pp0_policy_setter, DEVICE_TYPE_SOCKET, intel_pp0_policy_test, NULL},
};

static const _SysFeatureList intel_rapl_pp0_feature_list = {
    .num_features = ARRAY_COUNT(intel_rapl_pp0_features),
    .tester = intel_rapl_pp0_test,
    .features = intel_rapl_pp0_features,
};

/*********************************************************************************************************************/
/*                          Intel RAPL (PP1 domain)                                                                  */
/*********************************************************************************************************************/

static cerr_t pp1_test_testFunc(bool *ok, uint64_t msrData, void * value)
{
    (void)value;
    if (intel_rapl_pp1_info.powerUnit == 0 && intel_rapl_pp1_info.energyUnit == 0 && intel_rapl_pp1_info.timeUnit == 0)
    {
        intel_rapl_pp1_info.powerUnit = 1.0 / (1 << field64(msrData, 0, 4));
        intel_rapl_pp1_info.energyUnit = 1.0 / (1 << field64(msrData, 8, 5));
        intel_rapl_pp1_info.timeUnit = 1.0 / (1 << field64(msrData, 16, 4));
    }
    *ok = true;
    return NULL;
}

static cerr_t intel_rapl_pp1_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr_cb(ok, MSR_RAPL_POWER_UNIT, pp1_test_testFunc, NULL));
}

static cerr_t intel_pp1_energy_status_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr(ok, MSR_PP1_ENERGY_STATUS));
}

static cerr_t intel_pp1_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_status_getter(device, value, MSR_PP1_ENERGY_STATUS, &intel_rapl_pp1_info));
}

static cerr_t intel_pp1_energy_limit_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr(ok, MSR_PP1_RAPL_POWER_LIMIT));
}

static cerr_t pp1_limit_test_lock(bool *ok, uint64_t msrData, void * value)
{
    (void)value;
    *ok = field64(msrData, 31, 1);
    return NULL;
}

static cerr_t intel_rapl_pp1_limit_test_lock(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr_cb(ok, MSR_PP1_RAPL_POWER_LIMIT, pp1_limit_test_lock, NULL));
}

static cerr_t intel_pp1_energy_limit_1_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_getter(device, value, MSR_PP1_RAPL_POWER_LIMIT, &intel_rapl_pp1_info));
}

static cerr_t intel_pp1_energy_limit_1_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_setter(device, value, MSR_PP1_RAPL_POWER_LIMIT, &intel_rapl_pp1_info));
}

static cerr_t intel_pp1_energy_limit_1_time_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_time_getter(device, value, MSR_PP1_RAPL_POWER_LIMIT, &intel_rapl_pp1_info));
}

static cerr_t intel_pp1_energy_limit_1_time_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_time_setter(device, value, MSR_PP1_RAPL_POWER_LIMIT, &intel_rapl_pp1_info));
}

static cerr_t intel_pp1_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_enable_getter(device, value, MSR_PP1_RAPL_POWER_LIMIT, &intel_rapl_pp1_info));
}

static cerr_t intel_pp1_energy_limit_1_enable_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_enable_setter(device, value, MSR_PP1_RAPL_POWER_LIMIT, &intel_rapl_pp1_info));
}

static cerr_t intel_pp1_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_clamp_getter(device, value, MSR_PP1_RAPL_POWER_LIMIT, &intel_rapl_pp1_info));
}

static cerr_t intel_pp1_energy_limit_1_clamp_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_energy_limit_1_clamp_setter(device, value, MSR_PP1_RAPL_POWER_LIMIT, &intel_rapl_pp1_info));
}

static cerr_t intel_pp1_policy_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr(ok, MSR_PP1_ENERGY_POLICY));
}

static cerr_t intel_pp1_policy_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(intel_rapl_policy_getter(device, value, MSR_PP1_ENERGY_POLICY));
}

static cerr_t intel_pp1_policy_setter(const LikwidDevice_t device, const char* value)
{
    return ERROR_WRAP_CALL(intel_rapl_policy_setter(device, value, MSR_PP1_ENERGY_POLICY));
}

static _SysFeature intel_rapl_pp1_features[] = {
    {"pp1_energy", "rapl", "Current energy consumtion (PP1 domain)", intel_pp1_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, intel_pp1_energy_status_test, "uJ"},
    {"pp1_limit", "rapl", "Long-term energy limit (PP1 domain)", intel_pp1_energy_limit_1_getter, intel_pp1_energy_limit_1_setter, DEVICE_TYPE_SOCKET, intel_pp1_energy_limit_test, "mW"},
    {"pp1_limit_time", "rapl", "Long-term time window (PP1 domain)", intel_pp1_energy_limit_1_time_getter, intel_pp1_energy_limit_1_time_setter, DEVICE_TYPE_SOCKET, intel_pp1_energy_limit_test, "ms"},
    {"pp1_limit_enable", "rapl", "Status of long-term energy limit (PP1 domain)", intel_pp1_energy_limit_1_enable_getter, intel_pp1_energy_limit_1_enable_setter, DEVICE_TYPE_SOCKET, intel_pp1_energy_limit_test, "bool"},
    {"pp1_limit_clamp", "rapl", "Clamping status of long-term energy limit (PP1 domain)", intel_pp1_energy_limit_1_clamp_getter, intel_pp1_energy_limit_1_clamp_setter, DEVICE_TYPE_SOCKET, intel_pp1_energy_limit_test, "bool"},
    {"pp1_policy", "rapl", "Balance Power Policy (PP1 domain)", intel_pp1_policy_getter, intel_pp1_policy_setter, DEVICE_TYPE_SOCKET, intel_pp1_policy_test, "bool[4..0]"},
};

static const _SysFeatureList intel_rapl_pp1_feature_list = {
    .num_features = ARRAY_COUNT(intel_rapl_pp1_features),
    .tester = intel_rapl_pp1_test,
    .features = intel_rapl_pp1_features,
};

/* Init function */

cerr_t likwid_sysft_init_intel_rapl(_SysFeatureList* out)
{
    Configuration_t config = NULL;
    int err = init_configuration();
    if (err < 0)
        return ERROR_SET_LWERR(err, "Failed to initialize configuration");

    config = get_configuration();
    if (config->daemonMode == ACCESSMODE_PERF) {
        PRINT_INFO("No Intel RAPL support with accessmode=perf_event");
        return NULL;
    }

    bool pkg_avail;
    if (intel_rapl_pkg_test(&pkg_avail))
        return ERROR_WRAP();

    if (pkg_avail) {
        PRINT_INFO("Register Intel RAPL PKG domain");

        bool pkg_locked;
        if (intel_rapl_pkg_limit_test_lock(&pkg_locked))
            return ERROR_WRAP();

        if (pkg_locked) {
            PRINT_INFO("Intel RAPL PKG domain locked");
            for (int i = 0; i < intel_rapl_pkg_feature_list.num_features; i++)
                intel_rapl_pkg_feature_list.features[i].setter = NULL;
        }

        if (likwid_sysft_register_features(out, &intel_rapl_pkg_feature_list))
            return ERROR_WRAP();
    } else {
        PRINT_INFO("Intel RAPL domain PKG not supported");
    }

    bool dram_avail;
    if (intel_rapl_dram_test(&dram_avail))
        return ERROR_WRAP();

    if (dram_avail) {
        PRINT_INFO("Register Intel RAPL DRAM domain");

        bool dram_locked;
        if (intel_rapl_dram_limit_test_lock(&dram_locked))
            return ERROR_WRAP();

        if (dram_locked) {
            PRINT_INFO("Intel RAPL DRAM domain locked");
            for (int i = 0; i < intel_rapl_dram_feature_list.num_features; i++)
                intel_rapl_dram_feature_list.features[i].setter = NULL;
        }

        if (likwid_sysft_register_features(out, &intel_rapl_dram_feature_list))
            return ERROR_WRAP();
    } else {
        PRINT_INFO("Intel RAPL domain DRAM not supported");
    }

    bool pp0_avail;
    if (intel_rapl_pp0_test(&pp0_avail))
        return ERROR_WRAP();

    if (pp0_avail) {
        PRINT_INFO("Register Intel RAPL PP0 domain");

        bool pp0_locked;
        if (intel_rapl_pp0_limit_test_lock(&pp0_locked))
            return ERROR_WRAP();

        if (pp0_locked) {
            PRINT_INFO("Intel RAPL PP0 domain locked");
            for (int i = 0; i < intel_rapl_pp0_feature_list.num_features; i++)
                intel_rapl_pp0_feature_list.features[i].setter = NULL;
        }

        if (likwid_sysft_register_features(out, &intel_rapl_pp0_feature_list))
            return ERROR_WRAP();
    } else {
        PRINT_INFO("Intel RAPL domain PP0 not supported");
    }

    bool pp1_avail;
    if (intel_rapl_pp1_test(&pp1_avail))
        return ERROR_WRAP();

    if (pp1_avail) {
        PRINT_INFO("Register Intel RAPL PP1 domain");

        bool pp1_locked;
        if (intel_rapl_pp1_limit_test_lock(&pp1_locked)) {
            PRINT_INFO("Intel RAPL PP1 domain locked");
            for (int i = 0; i < intel_rapl_pp1_feature_list.num_features; i++)
                intel_rapl_pp1_feature_list.features[i].setter = NULL;
        }

        if (likwid_sysft_register_features(out, &intel_rapl_pp1_feature_list))
            return ERROR_WRAP();
    } else {
        PRINT_INFO("Intel RAPL domain PP1 not supported");
    }

    bool psys_avail;
    if (intel_rapl_psys_test(&psys_avail))
        return ERROR_WRAP();
    
    if (psys_avail) {
        PRINT_INFO("Register Intel RAPL PSYS domain");

        bool psys_locked;
        if (intel_rapl_psys_limit_test_lock(&psys_locked))
            return ERROR_WRAP();
        
        if (psys_locked) {
            PRINT_INFO("Intel RAPL PSYS domain locked");
            for (int i = 0; i < intel_rapl_psys_feature_list.num_features; i++)
                intel_rapl_psys_feature_list.features[i].setter = NULL;
        }

        if (likwid_sysft_register_features(out, &intel_rapl_psys_feature_list))
            return ERROR_WRAP();
    } else {
        PRINT_INFO("Intel RAPL domain PSYS not supported");
    }
    return NULL;
}
