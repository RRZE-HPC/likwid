#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <sysFeatures_intel.h>
#include <access.h>

/*********************************************************************************************************************/
/*                          Intel prefetchers                                                                        */
/*********************************************************************************************************************/

static int _intel_cpu_hwpf_register_test(uint64_t reg)
{
    int err = topology_init();
    if (err < 0)
    {
        return 0;
    }
    CpuTopology_t topo = get_cpuTopology();
    int valid = 0;
    for (unsigned j = 0; j < topo->numHWThreads; j++)
    {
        HWThread* t = &topo->threadPool[j];
        uint64_t msrData = 0;
        err = HPMread(t->apicId, MSR_DEV, reg, &msrData);
        if (err == 0) valid = 1;
        break;
    }
    return valid;
}

int intel_cpu_l2_hwpf_register_test(void)
{
    return _intel_cpu_hwpf_register_test(MSR_PREFETCH_ENABLE);
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
    return _intel_cpu_hwpf_register_test(MSR_IA32_MISC_ENABLE);
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
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 37, 1, true, value);
}

int intel_core2_l1_dcu_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 37, 1, true, value);
}

int intel_core2_l1_dcu_ip_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 39, 1, true, value);
}

int intel_core2_l1_dcu_ip_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 39, 1, true, value);
}

/*********************************************************************************************************************/
/*                          Intel Dynamic Acceleration                                                               */
/*********************************************************************************************************************/

int intel_core2_ida_tester(void)
{
    unsigned eax = 0x06, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    if (field32(eax, 1, 1))
    {
        return _intel_cpu_hwpf_register_test(MSR_PREFETCH_ENABLE);
    }
    return 0;
}

int intel_core2_ida_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 38, 1, true, value);
}

int intel_core2_ida_setter(const LikwidDevice_t device, const char* value) {
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 38, 1, true, value);
}
