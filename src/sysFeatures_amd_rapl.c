/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_amd_rapl.c
 *
 *      Description:  Interface to control AMD RAPL for the sysFeatures component
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

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include <bitUtil.h>
#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <sysFeatures_amd.h>
#include <sysFeatures_common.h>
#include <sysFeatures_amd_rapl.h>
#include <access.h>
#include <registers.h>
#include <topology.h>
#include <sysFeatures_common_rapl.h>

#include "debug.h"

static RaplDomainInfo amd_rapl_pkg_info = {0, 0, 0};
static RaplDomainInfo amd_rapl_core_info = {0, 0, 0};
static RaplDomainInfo amd_rapl_l3_info = {0, 0, 0};

static cerr_t amd_rapl_energy_status_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info, LikwidDeviceType type)
{
    uint64_t msrData = 0;
    int err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
        return ERROR_SET_LWERR(err, "HPMread failed");

    const uint64_t energy = field64(msrData, 0, 32);
    return ERROR_WRAP_CALL(likwid_sysft_double_to_string((double)energy * info->energyUnit, value));
}

/*********************************************************************************************************************/
/*                          AMD RAPL (PKG domain)                                                                  */
/*********************************************************************************************************************/

static cerr_t pkg_test_testFunc(bool *ok, uint64_t msrData, void * value)
{
    (void)value;
    if (amd_rapl_pkg_info.powerUnit == 0 && amd_rapl_pkg_info.energyUnit == 0 && amd_rapl_pkg_info.timeUnit == 0)
    {
        amd_rapl_pkg_info.powerUnit = 1.0 / (1 << field64(msrData, 0, 4));
        amd_rapl_pkg_info.energyUnit = 1.0 / (1 << field64(msrData, 8, 5));
        amd_rapl_pkg_info.timeUnit = 1.0 / (1 << field64(msrData, 16, 4));
    }
    *ok = true;
    return NULL;
}

static cerr_t amd_rapl_pkg_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr_cb(ok, MSR_AMD17_RAPL_POWER_UNIT, pkg_test_testFunc, NULL));
}

static cerr_t amd_pkg_energy_status_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr(ok, MSR_AMD17_RAPL_PKG_STATUS));
}

static cerr_t amd_pkg_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(amd_rapl_energy_status_getter(device, value, MSR_AMD17_RAPL_PKG_STATUS, &amd_rapl_pkg_info, DEVICE_TYPE_SOCKET));
}

static _SysFeature amd_rapl_pkg_features[] = {
    {"pkg_energy", "rapl", "Current energy consumtion (PKG domain)", amd_pkg_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, amd_pkg_energy_status_test, "J"},
};

static const _SysFeatureList amd_rapl_pkg_feature_list = {
    .num_features = ARRAY_COUNT(amd_rapl_pkg_features),
    .tester = amd_rapl_pkg_test,
    .features = amd_rapl_pkg_features,
};

/*********************************************************************************************************************/
/*                          AMD RAPL (CORE domain)                                                                 */
/*********************************************************************************************************************/

static cerr_t core_test_testFunc(bool *ok, uint64_t msrData, void * value)
{
    (void)value;
    if (amd_rapl_core_info.powerUnit == 0 && amd_rapl_core_info.energyUnit == 0 && amd_rapl_core_info.timeUnit == 0)
    {
        amd_rapl_core_info.powerUnit = 1.0 / (1 << field64(msrData, 0, 4));
        amd_rapl_core_info.energyUnit = 1.0 / (1 << field64(msrData, 8, 5));
        amd_rapl_core_info.timeUnit = 1.0 / (1 << field64(msrData, 16, 4));
    }
    *ok = true;
    return NULL;
}

static cerr_t amd_rapl_core_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_core_testmsr_cb(ok, MSR_AMD17_RAPL_POWER_UNIT, core_test_testFunc, NULL));
}

static cerr_t amd_core_energy_status_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_core_testmsr(ok, MSR_AMD17_RAPL_CORE_STATUS));
}

static cerr_t amd_core_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(amd_rapl_energy_status_getter(device, value, MSR_AMD17_RAPL_CORE_STATUS, &amd_rapl_core_info, DEVICE_TYPE_CORE));
}

static _SysFeature amd_rapl_core_features[] = {
    {"core_energy", "rapl", "Current energy consumtion (Core domain)", amd_core_energy_status_getter, NULL, DEVICE_TYPE_CORE, amd_core_energy_status_test, "J"},
};

static const _SysFeatureList amd_rapl_core_feature_list = {
    .num_features = ARRAY_COUNT(amd_rapl_core_features),
    .tester = amd_rapl_core_test,
    .features = amd_rapl_core_features,
};

/*********************************************************************************************************************/
/*                          AMD RAPL (L3 domain)                                                                     */
/*********************************************************************************************************************/

static cerr_t l3_test_testFunc(bool *ok, uint64_t msrData, void * value)
{
    (void)value;
    if (amd_rapl_l3_info.powerUnit == 0 && amd_rapl_l3_info.energyUnit == 0 && amd_rapl_l3_info.timeUnit == 0)
    {
        amd_rapl_l3_info.powerUnit = 1.0 / (1 << field64(msrData, 0, 4));
        amd_rapl_l3_info.energyUnit = 1.0 / (1 << field64(msrData, 8, 5));
        amd_rapl_l3_info.timeUnit = 1.0 / (1 << field64(msrData, 16, 4));
    }
    *ok = true;
    return NULL;
}

static cerr_t amd_rapl_l3_test(bool *ok)
{
    *ok = false;
    CpuInfo_t info = get_cpuInfo();
    if (info->family != ZEN3_FAMILY)
        return NULL;

    if (   info->model != ZEN4_RYZEN
        && info->model != ZEN4_RYZEN_PRO
        && info->model != ZEN4_EPYC
        && info->model != ZEN4_RYZEN2
        && info->model != ZEN4_RYZEN3 )
        return NULL;

    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr_cb(ok, MSR_AMD19_RAPL_L3_UNIT, l3_test_testFunc, NULL));
}

static cerr_t amd_l3_energy_status_test(bool *ok)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_core_testmsr(ok, MSR_AMD19_RAPL_L3_STATUS));
}

static cerr_t amd_l3_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(amd_rapl_energy_status_getter(device, value, MSR_AMD19_RAPL_L3_STATUS, &amd_rapl_l3_info, DEVICE_TYPE_SOCKET));
}

static _SysFeature amd_rapl_l3_features[] = {
    {"l3_energy", "rapl", "Current energy consumtion (L3 domain)", amd_l3_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, amd_l3_energy_status_test, "J"},
};

static const _SysFeatureList amd_rapl_l3_feature_list = {
    .num_features = ARRAY_COUNT(amd_rapl_l3_features),
    .tester = amd_rapl_l3_test,
    .features = amd_rapl_l3_features,
};

/* Init function */

cerr_t likwid_sysft_init_amd_rapl(_SysFeatureList* out)
{
    int err = init_configuration();
    if (err < 0)
        return ERROR_SET_LWERR(err, "init_configuration failed");

    Configuration_t config = get_configuration();
    if (config->daemonMode == ACCESSMODE_PERF) {
        PRINT_INFO("No AMD RAPL support with accessmode=perf_event");
        return NULL;
    }

    bool pkg_avail;
    if (amd_rapl_pkg_test(&pkg_avail))
        return ERROR_WRAP();

    if (pkg_avail) {
        PRINT_INFO("Register AMD RAPL PKG domain");
        if (likwid_sysft_register_features(out, &amd_rapl_pkg_feature_list))
            return ERROR_WRAP();
    } else {
        PRINT_INFO("AMD RAPL domain PKG not supported");
    }

    bool core_avail;
    if (amd_rapl_core_test(&core_avail))
        return ERROR_WRAP();

    if (core_avail) {
        PRINT_INFO("Register AMD RAPL CORE domain");
        if (likwid_sysft_register_features(out, &amd_rapl_core_feature_list))
            return ERROR_WRAP();
    } else {
        DEBUG_PRINT(DEBUGLEV_INFO, "AMD RAPL domain CORE not supported");
    }

    bool l3_avail;
    if (amd_rapl_l3_test(&l3_avail))
        return ERROR_WRAP();

    if (l3_avail) {
        PRINT_INFO("Register AMD RAPL L3 domain");
        if (likwid_sysft_register_features(out, &amd_rapl_l3_feature_list))
            return ERROR_WRAP();
    } else {
        DEBUG_PRINT(DEBUGLEV_INFO, "AMD RAPL domain L3 not supported");
    }

    return NULL;
}
