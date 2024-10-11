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
