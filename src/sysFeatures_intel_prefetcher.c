/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_intel_prefetcher.c
 *
 *      Description:  Interface to control CPU prefetchers for the sysFeatures component
 *
 *      Version:   5.4.0
 *      Released:  15.11.2024
 *
 *      Authors:  Thomas Gruber (tg), thomas.roehl@googlemail.com
 *                Michael Panzlaff, michael.panzlaff@fau.de
 *      Project:  likwid
 *
 *      Copyright (C) 2024 RRZE, University Erlangen-Nuremberg
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

#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <sysFeatures_intel.h>
#include <access.h>
#include <sysFeatures_common.h>
#include <bitUtil.h>
#include <cpuid.h>
#include <registers.h>

/*********************************************************************************************************************/
/*                          Intel prefetchers                                                                        */
/*********************************************************************************************************************/

static int intel_cpu_l2_hwpf_register_test(void)
{
    return likwid_sysft_foreach_hwt_testmsr(MSR_PREFETCH_ENABLE);
}

static int intel_cpu_l2_hwpf_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_PREFETCH_ENABLE, 0, true, value);
}

static int intel_cpu_l2_hwpf_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_PREFETCH_ENABLE, 0, true, value);
}

static int intel_cpu_l2_adj_pf_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_PREFETCH_ENABLE, 1, true, value);
}

static int intel_cpu_l2_adj_pf_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_PREFETCH_ENABLE, 1, true, value);
}

static int intel_cpu_l1_dcu_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_PREFETCH_ENABLE, 2, true, value);
}

static int intel_cpu_l1_dcu_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_PREFETCH_ENABLE, 2, true, value);
}

static int intel_cpu_l1_dcu_ip_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_PREFETCH_ENABLE, 3, true, value);
}

static int intel_cpu_l1_dcu_ip_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_PREFETCH_ENABLE, 3, true, value);
}

static _SysFeature intel_cpu_prefetch_features[] = {
    {"l2_hwpf", "prefetch", "L2 Hardware Prefetcher", intel_cpu_l2_hwpf_getter, intel_cpu_l2_hwpf_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_l2_hwpf_register_test},
    {"l2_adj_pf", "prefetch", "L2 Adjacent Cache Line Prefetcher", intel_cpu_l2_adj_pf_getter, intel_cpu_l2_adj_pf_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_l2_hwpf_register_test},
    {"l1_dcu", "prefetch", "DCU Hardware Prefetcher", intel_cpu_l1_dcu_getter, intel_cpu_l1_dcu_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_l2_hwpf_register_test},
    {"l1_dcu_ip", "prefetch", "DCU IP Prefetcher", intel_cpu_l1_dcu_ip_getter, intel_cpu_l1_dcu_ip_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_l2_hwpf_register_test},
    //{"data_pf", "Data Dependent Prefetcher", DEVICE_TYPE_HWTHREAD},
};

const _SysFeatureList likwid_sysft_intel_cpu_prefetch_feature_list = {
    .num_features = ARRAY_COUNT(intel_cpu_prefetch_features),
    .features = intel_cpu_prefetch_features,
};

/*********************************************************************************************************************/
/*                          Intel 0x8F prefetchers                                                                   */
/*********************************************************************************************************************/

static int intel_cpu_l2_multipath_pf_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_PREFETCH_ENABLE, 6, true, value);
}

static int intel_cpu_l2_multipath_pf_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_PREFETCH_ENABLE, 6, true, value);
}

static _SysFeature intel_8f_cpu_features[] = {
    {"l2_multipath_pf", "prefetch", "L2 Adaptive Multipath Probability Prefetcher", intel_cpu_l2_multipath_pf_getter, intel_cpu_l2_multipath_pf_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_l2_hwpf_register_test}
};

const _SysFeatureList likwid_sysft_intel_8f_cpu_feature_list = {
    .num_features = ARRAY_COUNT(intel_8f_cpu_features),
    .features = intel_8f_cpu_features,
};

/*********************************************************************************************************************/
/*                          Intel Knights Landing prefetchers                                                        */
/*********************************************************************************************************************/
static int intel_knl_l1_dcu_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_PREFETCH_ENABLE, 0, true, value);
}

static int intel_knl_l1_dcu_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_PREFETCH_ENABLE, 0, true, value);
}

static int intel_knl_l2_hwpf_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_PREFETCH_ENABLE, 1, true, value);
}

static int intel_knl_l2_hwpf_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_PREFETCH_ENABLE, 1, true, value);
}

static _SysFeature intel_knl_cpu_prefetch_features[] = {
    {"l2_hwpf", "prefetch", "L2 Hardware Prefetcher", intel_knl_l2_hwpf_getter, intel_knl_l2_hwpf_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_l2_hwpf_register_test},
    {"l1_dcu", "prefetch", "DCU Hardware Prefetcher", intel_knl_l1_dcu_getter, intel_knl_l1_dcu_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_l2_hwpf_register_test},
};

const _SysFeatureList likwid_sysft_intel_knl_cpu_feature_list = {
    .num_features = ARRAY_COUNT(intel_knl_cpu_prefetch_features),
    .features = intel_knl_cpu_prefetch_features,
};

/*********************************************************************************************************************/
/*                          Intel Core2 prefetchers                                                                  */
/*********************************************************************************************************************/

static int intel_core2_l2_hwpf_register_test(void)
{
    return likwid_sysft_foreach_hwt_testmsr(MSR_IA32_MISC_ENABLE);
}

static int intel_core2_l2_hwpf_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_MISC_ENABLE, 9, true, value);
}

static int intel_core2_l2_hwpf_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_IA32_MISC_ENABLE, 9, true, value);
}

static int intel_core2_l2_adjpf_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_MISC_ENABLE, 19, true, value);
}

static int intel_core2_l2_adjpf_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_IA32_MISC_ENABLE, 19, true, value);
}

static int intel_core2_l1_dcu_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_MISC_ENABLE, 37, true, value);
}

static int intel_core2_l1_dcu_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_IA32_MISC_ENABLE, 37, true, value);
}

static int intel_core2_l1_dcu_ip_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_MISC_ENABLE, 39, true, value);
}

static int intel_core2_l1_dcu_ip_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_IA32_MISC_ENABLE, 39, true, value);
}

static _SysFeature intel_core2_cpu_prefetch_features[] = {
    {"hwpf", "prefetch", "Hardware prefetcher operation on streams of data", intel_core2_l2_hwpf_getter, intel_knl_l2_hwpf_setter, DEVICE_TYPE_HWTHREAD, intel_core2_l2_hwpf_register_test},
    {"adj_pf", "prefetch", "Adjacent Cache Line Prefetcher", intel_core2_l2_adjpf_getter, intel_core2_l2_adjpf_setter, DEVICE_TYPE_HWTHREAD, intel_core2_l2_hwpf_register_test},
    {"l1_dcu", "prefetch", "DCU L1 data cache prefetcher", intel_core2_l1_dcu_getter, intel_core2_l1_dcu_setter, DEVICE_TYPE_HWTHREAD, intel_core2_l2_hwpf_register_test},
    {"l1_dcu_ip", "prefetch", "DCU IP Prefetcher", intel_core2_l1_dcu_ip_getter, intel_core2_l1_dcu_ip_setter, DEVICE_TYPE_HWTHREAD, intel_core2_l2_hwpf_register_test},
};

const _SysFeatureList likwid_sysft_intel_core2_cpu_feature_list = {
    .num_features = ARRAY_COUNT(intel_core2_cpu_prefetch_features),
    .features = intel_core2_cpu_prefetch_features,
};

/*********************************************************************************************************************/
/*                          Intel Dynamic Acceleration                                                               */
/*********************************************************************************************************************/

static int intel_core2_ida_tester(void)
{
    // TODO Not sure if Dynamic Acceleration was defacto replaced with Turbo Boost?
    // Should we advertise this capability on newer processors?
    //CpuInfo_t cpuinfo = get_cpuInfo();
    //if (cpuinfo->family != CORE2_45 && cpuinfo->family != CORE2_65 && cpuinfo->family != CORE_DUO)
    //{
    //    return 0;
    //}
    unsigned eax = 0x06, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    if (field32(eax, 1, 1))
    {
        return likwid_sysft_foreach_hwt_testmsr(MSR_IA32_MISC_ENABLE);
    }
    return 0;
}

static int intel_core2_ida_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_MISC_ENABLE, 38, true, value);
}

static int intel_core2_ida_setter(const LikwidDevice_t device, const char* value) {
    return likwid_sysft_writemsr_bit_from_string(device, MSR_IA32_MISC_ENABLE, 38, true, value);
}

static _SysFeature intel_cpu_ida_features[] = {
    // TODO does this really belong into the "prefetcher" category?
    {"ida", "prefetch", "Intel Dynamic Acceleration", intel_core2_ida_getter, intel_core2_ida_setter, DEVICE_TYPE_HWTHREAD, intel_core2_ida_tester},
};

const _SysFeatureList likwid_sysft_intel_cpu_ida_feature_list = {
    .num_features = ARRAY_COUNT(intel_cpu_ida_features),
    .tester = intel_core2_ida_tester,
    .features = intel_cpu_ida_features,
};
