/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_intel_hwp.c
 *
 *      Description:  Interface to read Intel HWP (hardware p-states) for the
 *                    sysFeatures component
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
#include <cpuid.h>


static int intel_hwp_capabilities_test()
{
    int err = 0;
    int testhwthread = 0;
    uint64_t msrData;
    unsigned eax = 0x0, ebx = 0x0, ecx = 0x0, edx = 0x0;

    err = topology_init();
    if (err < 0)
        return 0;

    CpuTopology_t topo = get_cpuTopology();
    for (int i = 0; i < topo->numHWThreads; i++)
    {
        if (topo->threadPool[i].inCpuSet)
        {
            testhwthread = topo->threadPool[i].apicId;
            break;
        }
    }

    eax = 0x06;
    CPUID(eax, ebx, ecx, edx);
    if (!testBit(eax, 7))
    {
        return 0;
    }
    err = HPMread(testhwthread, MSR_DEV, MSR_HWP_CAPABILITIES, &msrData);
    if (err < 0)
    {
        return 0;
    }
    err = HPMread(testhwthread, MSR_DEV, MSR_HWP_ENABLE, &msrData);
    if (err < 0)
    {
        return 0;
    }
    err = HPMread(testhwthread, MSR_DEV, MSR_HWP_REQUEST_PKG, &msrData);
    if (err < 0)
    {
        return 0;
    }
    err = HPMread(testhwthread, MSR_DEV, MSR_HWP_REQUEST, &msrData);
    if (err < 0)
    {
        return 0;
    }
    err = HPMread(testhwthread, MSR_DEV, MSR_HWP_REQUEST_INFO, &msrData);
    if (err < 0)
    {
        return 0;
    }
    return 1;
}

static int intel_hwp_capabilities_high_getter(const LikwidDevice_t device, char** value)
{
    uint64_t val;
    int err = likwid_sysft_readmsr_field(device, MSR_HWP_CAPABILITIES, 0, 8, &val);
    if (err)
        return err;
    return likwid_sysft_uint64_to_string(val, value);
}

static int intel_hwp_capabilities_guarantee_getter(const LikwidDevice_t device, char** value)
{
    uint64_t val;
    int err = likwid_sysft_readmsr_field(device, MSR_HWP_CAPABILITIES, 8, 8, &val);
    if (err)
        return err;
    return likwid_sysft_uint64_to_string(val, value);
}

static int intel_hwp_capabilities_mostefficient_getter(const LikwidDevice_t device, char** value)
{
    uint64_t val;
    int err = likwid_sysft_readmsr_field(device, MSR_HWP_CAPABILITIES, 16, 8, &val);
    if (err)
        return err;
    return likwid_sysft_uint64_to_string(val, value);
}

static int intel_hwp_capabilities_lowest_getter(const LikwidDevice_t device, char** value)
{
    uint64_t val;
    int err = likwid_sysft_readmsr_field(device, MSR_HWP_CAPABILITIES, 24, 8, &val);
    if (err)
        return err;
    return likwid_sysft_uint64_to_string(val, value);
}

/*static int intel_hwp_capabilities_req_min_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    uint64_t val;*/
/*    int err = likwid_sysft_readmsr_field(device, MSR_HWP_REQUEST, 0, 8, &val);*/
/*    if (err)*/
/*        return err;*/
/*    return likwid_sysft_uint64_to_string(val, value);*/
/*}*/
/*static int intel_hwp_capabilities_req_max_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    uint64_t val;*/
/*    int err = likwid_sysft_readmsr_field(device, MSR_HWP_REQUEST, 8, 8, &val);*/
/*    if (err)*/
/*        return err;*/
/*    return likwid_sysft_uint64_to_string(val, value);*/
/*}*/
/*static int intel_hwp_capabilities_req_desire_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    uint64_t val;*/
/*    int err = likwid_sysft_readmsr_field(device, MSR_HWP_REQUEST, 16, 8, &val);*/
/*    if (err)*/
/*        return err;*/
/*    return likwid_sysft_uint64_to_string(val, value);*/
/*}*/
/*static int intel_hwp_capabilities_req_epp_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    uint64_t val;*/
/*    int err = likwid_sysft_readmsr_field(device, MSR_HWP_REQUEST, 24, 8, &val);*/
/*    if (err)*/
/*        return err;*/
/*    return likwid_sysft_uint64_to_string(val, value);*/
/*}*/
/*static int intel_hwp_capabilities_req_act_win_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    uint64_t val;*/
/*    int err = likwid_sysft_readmsr_field(device, MSR_HWP_REQUEST, 32, 10, &val);*/
/*    if (err)*/
/*        return err;*/
/*    return likwid_sysft_uint64_to_string(val, value);*/
/*}*/

/*static int intel_hwp_capabilities_req_pkg_control_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    uint64_t val;*/
/*    int err = likwid_sysft_readmsr_field(device, MSR_HWP_REQUEST, 42, 1, &val);*/
/*    if (err)*/
/*        return err;*/
/*    return likwid_sysft_uint64_to_string(val, value);*/
/*}*/

/*static int intel_hwp_capabilities_req_act_win_valid_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    uint64_t val;*/
/*    int err = likwid_sysft_readmsr_field(device, MSR_HWP_REQUEST, 59, 1, &val);*/
/*    if (err)*/
/*        return err;*/
/*    return likwid_sysft_uint64_to_string(val, value);*/
/*}*/

/*static int intel_hwp_capabilities_req_epp_valid_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    uint64_t val;*/
/*    int err = likwid_sysft_readmsr_field(device, MSR_HWP_REQUEST, 60, 1, &val);*/
/*    if (err)*/
/*        return err;*/
/*    return likwid_sysft_uint64_to_string(val, value);*/
/*}*/

/*static int intel_hwp_capabilities_req_desired_perf_valid_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    uint64_t val;*/
/*    int err = likwid_sysft_readmsr_field(device, MSR_HWP_REQUEST, 61, 1, &val);*/
/*    if (err)*/
/*        return err;*/
/*    return likwid_sysft_uint64_to_string(val, value);*/
/*}*/

/*static int intel_hwp_capabilities_req_max_perf_valid_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    uint64_t val;*/
/*    int err = likwid_sysft_readmsr_field(device, MSR_HWP_REQUEST, 62, 1, &val);*/
/*    if (err)*/
/*        return err;*/
/*    return likwid_sysft_uint64_to_string(val, value);*/
/*}*/

/*static int intel_hwp_capabilities_req_min_perf_valid_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    uint64_t val;*/
/*    int err = likwid_sysft_readmsr_field(device, MSR_HWP_REQUEST, 63, 1, &val);*/
/*    if (err)*/
/*        return err;*/
/*    return likwid_sysft_uint64_to_string(val, value);*/
/*}*/

static _SysFeature intel_hwp_features[] = {
    {"cap_highest_perf", "hwp", "Get highest performance HWP capability", intel_hwp_capabilities_high_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_hwp_capabilities_test},
    {"cap_guarantee_perf", "hwp", "Get guaranteed performance HWP capability", intel_hwp_capabilities_guarantee_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_hwp_capabilities_test},
    {"cap_most_efficient_perf", "hwp", "Get most efficient performance HWP capability", intel_hwp_capabilities_mostefficient_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_hwp_capabilities_test},
    {"cap_lowest_perf", "hwp", "Get lowest performance HWP capability", intel_hwp_capabilities_lowest_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_hwp_capabilities_test},
/*    {"req_min_perf", "hwp", "Get minimum performance HWP value requested by the OS", intel_hwp_capabilities_req_min_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_hwp_capabilities_test},*/
/*    {"req_max_perf", "hwp", "Get maximum performance HWP value requested by the OS", intel_hwp_capabilities_req_max_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_hwp_capabilities_test},*/
/*    {"req_desired_perf", "hwp", "Get desired performance HWP value requested by the OS", intel_hwp_capabilities_req_desire_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_hwp_capabilities_test},*/
/*    {"req_epp", "hwp", "Get HWP energy efficiency preference requested by the OS", intel_hwp_capabilities_req_epp_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_hwp_capabilities_test},*/
/*    {"req_activity_win", "hwp", "Get HWP activity window requested by the OS", intel_hwp_capabilities_req_act_win_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_hwp_capabilities_test},*/
/*    {"req_pkg_control", "hwp", "HWP uses thread-local or package-wide requests", intel_hwp_capabilities_req_pkg_control_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_hwp_capabilities_test},*/
/*    {"req_activity_win_valid", "hwp", "Is HWP activity window requested by the OS valid", intel_hwp_capabilities_req_act_win_valid_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_hwp_capabilities_test},*/
/*    {"req_epp_valid", "hwp", "Is HWP energy efficiency preference requested by the OS valid", intel_hwp_capabilities_req_epp_valid_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_hwp_capabilities_test},*/
/*    {"req_desired_perf_valid", "hwp", "Is desired performance HWP value requested by the OS valid", intel_hwp_capabilities_req_desired_perf_valid_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_hwp_capabilities_test},*/
/*    {"req_max_perf_valid", "hwp", "Is HWP maximum performance requested by the OS valid", intel_hwp_capabilities_req_max_perf_valid_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_hwp_capabilities_test},*/
/*    {"req_min_perf_valid", "hwp", "Is HWP maximum performance requested by the OS valid", intel_hwp_capabilities_req_min_perf_valid_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_hwp_capabilities_test},*/
};

const _SysFeatureList likwid_sysft_intel_hwp_feature_list = {
    .num_features = ARRAY_COUNT(intel_hwp_features),
    .tester = intel_hwp_capabilities_test,
    .features = intel_hwp_features,
};

/* Init function */

int likwid_sysft_init_intel_hwp(_SysFeatureList* out)
{
    int err = 0;
    Configuration_t config = NULL;
    err = init_configuration();
    if (err < 0)
    {
        errno = -err;
        ERROR_PRINT(Failed to initialize configuration);
        return err;
    }
    config = get_configuration();
    if (config->daemonMode == ACCESSMODE_PERF)
    {
        DEBUG_PRINT(DEBUGLEV_INFO, No Intel HWP support with accessmode=perf_event);
        return 0;
    }
    err = likwid_sysft_register_features(out, &likwid_sysft_intel_hwp_feature_list);
    if (err < 0)
    {
        DEBUG_PRINT(DEBUGLEV_INFO, Intel HWP not supported);
    }
    return 0;
}
