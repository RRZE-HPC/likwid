/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_amd.c
 *
 *      Description:  AMD interface of the sysFeatures component
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
#include <sysFeatures_amd.h>

#include <stdbool.h>

#include <access.h>
#include <bitUtil.h>
#include <sysFeatures_amd_hsmp.h>
#include <sysFeatures_amd_rapl.h>
#include <sysFeatures_common.h>
#include <topology.h>
#include <registers.h>

static const _HWArchFeatures amd_arch_features[];

int likwid_sysft_init_x86_amd(_SysFeatureList* out)
{
    int err = likwid_sysft_init_generic(amd_arch_features, out);
    if (err < 0)
    {
        ERROR_PRINT(Failed to init general x86 HWFetures);
        return err;
    }
    err = likwid_sysft_init_amd_rapl(out);
    if (err < 0)
    {
        ERROR_PRINT(Failed to init AMD RAPL HWFetures);
        return err;
    }
    err = likwid_sysft_init_amd_hsmp(out);
    if (err < 0)
    {
        ERROR_PRINT(Failed to init AMD HSMP HWFeaturse);
        return err;
    }
    return 0;
}

static int amd_cpu_l1_stream_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_AMD19_PREFETCH_CONTROL, 0, true, value);
}

static int amd_cpu_l1_stream_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_AMD19_PREFETCH_CONTROL, 0, true, value);
}

static int amd_cpu_l1_stride_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_AMD19_PREFETCH_CONTROL, 1, true, value);
}

static int amd_cpu_l1_stride_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_AMD19_PREFETCH_CONTROL, 1, true, value);
}

static int amd_cpu_l1_region_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_AMD19_PREFETCH_CONTROL, 2, true, value);
}

static int amd_cpu_l1_region_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_AMD19_PREFETCH_CONTROL, 2, true, value);
}

static int amd_cpu_l2_stream_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_AMD19_PREFETCH_CONTROL, 3, true, value);
}

static int amd_cpu_l2_stream_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_AMD19_PREFETCH_CONTROL, 3, true, value);
}

static int amd_cpu_up_down_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_AMD19_PREFETCH_CONTROL, 5, true, value);
}

static int amd_cpu_up_down_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_AMD19_PREFETCH_CONTROL, 5, true, value);
}

static _SysFeature amd_k19_cpu_prefetch_features[] = {
    {"l1_stream", "prefetch", "Stream prefetcher that uses history of memory access patterns to fetch additional sequential lines into L1 cache", amd_cpu_l1_stream_getter, amd_cpu_l1_stream_setter, DEVICE_TYPE_HWTHREAD},
    {"l1_stride", "prefetch", "Stride prefetcher that uses memory access history of individual instructions to fetch additional lines into L1 cache when each access is a constant distance from the previous", amd_cpu_l1_stride_getter, amd_cpu_l1_stride_setter, DEVICE_TYPE_HWTHREAD},
    {"l1_region", "prefetch", "Prefetcher that uses memory access history to fetch additional lines into L1 cache when the data access for a given instruction tends to be followed by a consistent pattern of other accesses within a localized region", amd_cpu_l1_region_getter, amd_cpu_l1_region_setter, DEVICE_TYPE_HWTHREAD},
    {"l2_stream", "prefetch", "Stream prefetcher that uses history of memory access patterns to fetch additional sequential lines into L2 cache", amd_cpu_l2_stream_getter, amd_cpu_l2_stream_setter, DEVICE_TYPE_HWTHREAD},
    {"up_down", "prefetch", "Prefetcher that uses memory access history to determine whether to fetch the next or previous line into L2 cache for all memory accesses", amd_cpu_up_down_getter, amd_cpu_up_down_setter, DEVICE_TYPE_HWTHREAD},
};

static const _SysFeatureList amd_k19_cpu_prefetch_feature_list = {
    .num_features = ARRAY_COUNT(amd_k19_cpu_prefetch_features),
    .features = amd_k19_cpu_prefetch_features,
};

static int amd_cpu_spec_ibrs_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_AMD19_SPEC_CTRL, 0, false, value);
}

static int amd_cpu_spec_ibrs_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_AMD19_SPEC_CTRL, 0, false, value);
}

static int amd_cpu_spec_stibp_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_AMD19_SPEC_CTRL, 1, false, value);
}

static int amd_cpu_spec_stibp_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_AMD19_SPEC_CTRL, 1, false, value);
}

static int amd_cpu_spec_ssbd_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_AMD19_SPEC_CTRL, 2, true, value);
}

static int amd_cpu_spec_ssbd_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_AMD19_SPEC_CTRL, 2, true, value);
}

static int amd_cpu_spec_pfsd_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_AMD19_SPEC_CTRL, 7, true, value);
}

static int amd_cpu_spec_pfsd_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_AMD19_SPEC_CTRL, 7, true, value);
}

static _SysFeature amd_k19_cpu_speculation_features[] = {
    {"ibrs", "spec_ctrl", "Indirect branch restriction speculation", amd_cpu_spec_ibrs_getter, amd_cpu_spec_ibrs_setter, DEVICE_TYPE_HWTHREAD},
    {"stibp", "spec_ctrl", "Single thread indirect branch predictor", amd_cpu_spec_stibp_getter, amd_cpu_spec_stibp_setter, DEVICE_TYPE_HWTHREAD},
    {"ssbd", "spec_ctrl", "Speculative Store Bypass", amd_cpu_spec_ssbd_getter, amd_cpu_spec_ssbd_setter, DEVICE_TYPE_HWTHREAD},
    {"psfd", "spec_ctrl", "Predictive Store Forwarding", amd_cpu_spec_pfsd_getter, amd_cpu_spec_pfsd_setter, DEVICE_TYPE_HWTHREAD},
};

static const _SysFeatureList amd_k19_cpu_speculation_feature_list = {
    .num_features = ARRAY_COUNT(amd_k19_cpu_speculation_features),
    .features = amd_k19_cpu_speculation_features,
};

static int amd_cpu_flush_l1(const LikwidDevice_t device, const char* value)
{
    uint64_t flush;
    int err = likwid_sysft_string_to_uint64(value, &flush);
    if (err < 0)
        return err;
    err = HPMinit();
    if (err < 0)
        return err;
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
        return err;
    return HPMwrite(device->id.simple.id, MSR_DEV, MSR_AMD19_L1D_FLUSH_REGISTER, flush & 0x1);
}

static _SysFeature amd_k19_cpu_l1dflush_features[] = {
    {"l1dflush", "cache", "Performs a write-back and invalidate of the L1 data cache", NULL, amd_cpu_flush_l1, DEVICE_TYPE_HWTHREAD},
};

static const _SysFeatureList amd_k19_cpu_l1dflush_feature_list = {
    .num_features = ARRAY_COUNT(amd_k19_cpu_l1dflush_features),
    .features = amd_k19_cpu_l1dflush_features,
};

static int amd_cpu_hwconfig_cpddis_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_AMD17_HW_CONFIG, 25, true, value);
}

static int amd_cpu_hwconfig_cpddis_setter(const LikwidDevice_t device, const char* value)
{
    return likwid_sysft_writemsr_bit_from_string(device, MSR_AMD17_HW_CONFIG, 25, true, value);
}

static _SysFeature amd_k19_cpu_hwconfig_features[] = {
    {"TurboMode", "cpufreq", "Specifies whether core performance boost is requested to be enabled or disabled", amd_cpu_hwconfig_cpddis_getter, amd_cpu_hwconfig_cpddis_setter, DEVICE_TYPE_HWTHREAD},
};

static const _SysFeatureList amd_k19_cpu_hwconfig_feature_list = {
    .num_features = ARRAY_COUNT(amd_k19_cpu_hwconfig_features),
    .features = amd_k19_cpu_hwconfig_features,
};

static const _SysFeatureList* amd_k19_cpu_feature_inputs[] = {
    &amd_k19_cpu_prefetch_feature_list,
    &amd_k19_cpu_speculation_feature_list,
    &amd_k19_cpu_l1dflush_feature_list,
    &amd_k19_cpu_hwconfig_feature_list,
    NULL,
};

static const _HWArchFeatures amd_arch_features[] = {
    {ZEN3_FAMILY, ZEN4_RYZEN, amd_k19_cpu_feature_inputs},
    {ZEN3_FAMILY, ZEN4_RYZEN_PRO, amd_k19_cpu_feature_inputs},
    {ZEN3_FAMILY, ZEN4_EPYC, amd_k19_cpu_feature_inputs},
    {ZEN3_FAMILY, ZEN4_EPYC_BERGAMO, amd_k19_cpu_feature_inputs},
    {-1, -1, NULL},
};
