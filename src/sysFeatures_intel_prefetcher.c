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

int intel_cpu_l2_hwpf_register_test(void)
{
    return likwid_sysft_foreach_hwt_testmsr(MSR_PREFETCH_ENABLE);
}

int intel_cpu_l2_hwpf_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 0, 1, true, value);
}

int intel_cpu_l2_hwpf_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 0, 1, true, value);
}

int intel_cpu_l2_adj_pf_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 1, 1, true, value);
}

int intel_cpu_l2_adj_pf_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 1, 1, true, value);
}

int intel_cpu_l1_dcu_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 2, 1, true, value);
}

int intel_cpu_l1_dcu_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 2, 1, true, value);
}

int intel_cpu_l1_dcu_ip_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 3, 1, true, value);
}

int intel_cpu_l1_dcu_ip_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 3, 1, true, value);
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

int intel_cpu_l2_multipath_pf_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 6, 1, true, value);
}

int intel_cpu_l2_multipath_pf_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 6, 1, true, value);
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
int intel_knl_l1_dcu_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 0, 1, true, value);
}

int intel_knl_l1_dcu_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 0, 1, true, value);
}

int intel_knl_l2_hwpf_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 1, 1, true, value);
}

int intel_knl_l2_hwpf_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 1, 1, true, value);
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

int intel_core2_l2_hwpf_register_test(void)
{
    return likwid_sysft_foreach_hwt_testmsr(MSR_IA32_MISC_ENABLE);
}

int intel_core2_l2_hwpf_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_IA32_MISC_ENABLE, 9, 1, true, value);
}

int intel_core2_l2_hwpf_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_IA32_MISC_ENABLE, 9, 1, true, value);
}

int intel_core2_l2_adjpf_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_IA32_MISC_ENABLE, 19, 1, true, value);
}

int intel_core2_l2_adjpf_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_IA32_MISC_ENABLE, 19, 1, true, value);
}

int intel_core2_l1_dcu_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_IA32_MISC_ENABLE, 37, 1, true, value);
}

int intel_core2_l1_dcu_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_IA32_MISC_ENABLE, 37, 1, true, value);
}

int intel_core2_l1_dcu_ip_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_IA32_MISC_ENABLE, 39, 1, true, value);
}

int intel_core2_l1_dcu_ip_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_IA32_MISC_ENABLE, 39, 1, true, value);
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

int intel_core2_ida_tester(void)
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

int intel_core2_ida_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_IA32_MISC_ENABLE, 38, 1, true, value);
}

int intel_core2_ida_setter(const LikwidDevice_t device, const char* value) {
    return intel_cpu_msr_register_setter(device, MSR_IA32_MISC_ENABLE, 38, 1, true, value);
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
