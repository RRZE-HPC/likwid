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
    int err = 0;
    int valid = 0;
    CpuTopology_t topo = NULL;

    err = topology_init();
    if (err < 0)
    {
        return 0;
    }
    topo = get_cpuTopology();
    for (int j = 0; j < topo->numHWThreads; j++)
    {
        uint64_t data = 0;
        HWThread* t = &topo->threadPool[j];
        err = HPMread(t->apicId, MSR_DEV, MSR_PREFETCH_ENABLE, &data);
        if (err == 0) valid = 1;
        break;
    }
    return valid;
}

int intel_cpu_l2_hwpf_register_test()
{
    return _intel_cpu_hwpf_register_test(MSR_PREFETCH_ENABLE);
}

int intel_cpu_l2_hwpf_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 0x1, 0, 1, value);
}

int intel_cpu_l2_hwpf_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 0x1, 0, 1, value);
}

int intel_cpu_l2_adj_pf_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 0x2, 1, 1, value);
}

int intel_cpu_l2_adj_pf_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 0x2, 1, 1, value);
}

int intel_cpu_l1_dcu_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 0x4, 3, 1, value);
}

int intel_cpu_l1_dcu_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 0x4, 3, 1, value);
}

int intel_cpu_l1_dcu_ip_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 0x8, 4, 1, value);
}

int intel_cpu_l1_dcu_ip_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 0x8, 4, 1, value);
}


/*********************************************************************************************************************/
/*                          Intel 0x8F prefetchers                                                                   */
/*********************************************************************************************************************/

int intel_cpu_l2_multipath_pf_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 0x20, 6, 1, value);
}

int intel_cpu_l2_multipath_pf_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 0x20, 6, 1, value);
}


/*********************************************************************************************************************/
/*                          Intel Knights Landing prefetchers                                                        */
/*********************************************************************************************************************/
int intel_knl_l1_dcu_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 0x1, 0, 1, value);
}

int intel_knl_l1_dcu_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 0x1, 0, 1, value);
}

int intel_knl_l2_hwpf_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, 0x2, 1, 1, value);
}

int intel_knl_l2_hwpf_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, 0x2, 1, 1, value);
}


/*********************************************************************************************************************/
/*                          Intel Core2 prefetchers                                                                  */
/*********************************************************************************************************************/

int intel_core2_l2_hwpf_register_test()
{
    return _intel_cpu_hwpf_register_test(MSR_IA32_MISC_ENABLE);
}

int intel_core2_l2_hwpf_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_IA32_MISC_ENABLE, (1ULL<<9), 9, 1, value);
}

int intel_core2_l2_hwpf_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_IA32_MISC_ENABLE, (1ULL<<9), 9, 1, value);
}

int intel_core2_l2_adjpf_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_IA32_MISC_ENABLE, (1ULL<<19), 19, 1, value);
}

int intel_core2_l2_adjpf_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_IA32_MISC_ENABLE, (1ULL<<19), 19, 1, value);
}

int intel_core2_l1_dcu_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, (1ULL<<37), 37, 1, value);
}

int intel_core2_l1_dcu_setter(const LikwidDevice_t device, const char* value)
{
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, (1ULL<<37), 37, 1, value);
}

int intel_core2_l1_dcu_ip_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, (1ULL<<39), 39, 1, value);
}

int intel_core2_l1_dcu_ip_setter(const LikwidDevice_t device, const char* value)
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
    if ((eax >> 1) & 0x1)
    {
        return _intel_cpu_hwpf_register_test(MSR_PREFETCH_ENABLE);
    }
    return 0;
}

int intel_core2_ida_getter(const LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_PREFETCH_ENABLE, (1ULL<<38), 38, 1, value);
}

int intel_core2_ida_setter(const LikwidDevice_t device, const char* value) {
    return intel_cpu_msr_register_setter(device, MSR_PREFETCH_ENABLE, (1ULL<<38), 38, 1, value);
}

/* Init function */

int sysFeatures_init_intel_prefetchers(_SysFeatureList* out)
{
    return 0;
}
