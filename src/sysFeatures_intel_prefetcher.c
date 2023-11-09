#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <sysFeatures_intel.h>

/*********************************************************************************************************************/
/*                          Intel prefetchers                                                                        */
/*********************************************************************************************************************/
int intel_cpu_l2_hwpf_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 0x1, 0, 1, value);
}

int intel_cpu_l2_hwpf_setter(LikwidDevice_t device, char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 0x1, 0, 1, value);
}

int intel_cpu_l2_adj_pf_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 0x2, 1, 1, value);
}

int intel_cpu_l2_adj_pf_setter(LikwidDevice_t device, char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 0x2, 1, 1, value);
}

int intel_cpu_l1_dcu_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 0x4, 3, 1, value);
}

int intel_cpu_l1_dcu_setter(LikwidDevice_t device, char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 0x4, 3, 1, value);
}

int intel_cpu_l1_dcu_ip_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 0x8, 4, 1, value);
}

int intel_cpu_l1_dcu_ip_setter(LikwidDevice_t device, char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 0x8, 4, 1, value);
}


/*********************************************************************************************************************/
/*                          Intel 0x8F prefetchers                                                                   */
/*********************************************************************************************************************/

int intel_cpu_l2_multipath_pf_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 0x20, 6, 1, value);
}

int intel_cpu_l2_multipath_pf_setter(LikwidDevice_t device, char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 0x20, 6, 1, value);
}


/*********************************************************************************************************************/
/*                          Intel Knights Landing prefetchers                                                        */
/*********************************************************************************************************************/
int intel_knl_l1_dcu_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 0x1, 0, 1, value);
}

int intel_knl_l1_dcu_setter(LikwidDevice_t device, char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 0x1, 0, 1, value);
}

int intel_knl_l2_hwpf_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 0x2, 1, 1, value);
}

int intel_knl_l2_hwpf_setter(LikwidDevice_t device, char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 0x2, 1, 1, value);
}


/*********************************************************************************************************************/
/*                          Intel Core2 prefetchers                                                                  */
/*********************************************************************************************************************/

int intel_core2_l2_hwpf_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_IA32_MISC_ENABLE, (1ULL<<9), 9, 1, value);
}

int intel_core2_l2_hwpf_setter(LikwidDevice_t device, char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_IA32_MISC_ENABLE, (1ULL<<9), 9, 1, value);
}

int intel_core2_l2_adjpf_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_IA32_MISC_ENABLE, (1ULL<<19), 19, 1, value);
}

int intel_core2_l2_adjpf_setter(LikwidDevice_t device, char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_IA32_MISC_ENABLE, (1ULL<<19), 19, 1, value);
}

int intel_core2_l1_dcu_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, (1ULL<<37), 37, 1, value);
}

int intel_core2_l1_dcu_setter(LikwidDevice_t device, char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, (1ULL<<37), 37, 1, value);
}

int intel_core2_l1_dcu_ip_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, (1ULL<<39), 39, 1, value);
}

int intel_core2_l1_dcu_ip_setter(LikwidDevice_t device, char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, (1ULL<<39), 39, 1, value);
}

/*********************************************************************************************************************/
/*                          Intel Dynamic Acceleration                                                               */
/*********************************************************************************************************************/

int intel_core2_ida_tester()
{
    unsigned eax = 0x06, ebx, ecx, edx;
    CPUID(eax, ebx, ecx, edx);
    return ((eax >> 1) & 0x1);
}

int intel_core2_ida_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, (1ULL<<38), 38, 1, value);
}

int intel_core2_ida_setter(LikwidDevice_t device, char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, (1ULL<<38), 38, 1, value);
}

/* Init function */

int sysFeatures_init_intel_prefetchers(_SysFeatureList* out)
{
    return 0;
}
