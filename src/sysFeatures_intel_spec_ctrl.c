#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <sysFeatures_intel.h>
#include <cpuid.h>
#include <bitUtil.h>



int intel_cpu_spec_ibrs_tester()
{
    unsigned eax = 0x07, ebx = 0, ecx = 0, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    return testBit(edx, 26);
}

int intel_cpu_spec_ibrs_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_IA32_SPEC_CTRL, 0x1, 0, 0, value);
}


int intel_cpu_spec_stibp_tester()
{
    unsigned eax = 0x07, ebx = 0, ecx = 0, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    return testBit(edx, 27);
}

int intel_cpu_spec_stibp_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_IA32_SPEC_CTRL, 0x2, 1, 0, value);
}

int intel_cpu_spec_ssbd_tester()
{
    unsigned eax = 0x07, ebx = 0, ecx = 0, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    return testBit(edx, 31);
}

int intel_cpu_spec_ssbd_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_IA32_SPEC_CTRL, 0x4, 2, 1, value);
}


int intel_cpu_spec_ipred_dis_tester()
{
    unsigned eax = 0x07, ebx = 0, ecx = 0x02, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    return testBit(edx, 1);
}

int intel_cpu_spec_ipred_dis_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_IA32_SPEC_CTRL, 0x8, 3, 0, value);
}

int intel_cpu_spec_rrsba_dis_tester()
{
    unsigned eax = 0x07, ebx = 0, ecx = 0x02, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    return testBit(edx, 2);
}

int intel_cpu_spec_rrsba_dis_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_IA32_SPEC_CTRL, 0x20, 5, 1, value);
}

int intel_cpu_spec_psfd_tester()
{
    unsigned eax = 0x07, ebx = 0, edx = 0x02, ecx = 0;
    CPUID(eax, ebx, ecx, edx);
    return testBit(edx, 0);
}

int intel_cpu_spec_psfd_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_IA32_SPEC_CTRL, 0x80, 7, 1, value);
}

int intel_cpu_spec_ddpd_tester()
{
    unsigned eax = 0x07, ebx = 0, ecx = 0x02, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    return testBit(edx, 0);
}

int intel_cpu_spec_ddpd_getter(LikwidDevice_t device, char** value)
{
    return intel_cpu_msr_register_getter(device, MSR_IA32_SPEC_CTRL, 0x100, 8, 1, value);
}


int intel_cpu_spec_ctrl()
{
    int valid = 0;
    if (intel_cpu_spec_ibrs_tester()) valid++;
    if (intel_cpu_spec_stibp_tester()) valid++;
    if (intel_cpu_spec_ssbd_tester()) valid++;
    if (intel_cpu_spec_ipred_dis_tester()) valid++;
    if (intel_cpu_spec_rrsba_dis_tester()) valid++;
    if (intel_cpu_spec_psfd_tester()) valid++;
    if (intel_cpu_spec_ddpd_tester()) valid++;
    if (valid == 0)
    {
        printf("Speculation control not available\n");
    }
    return valid > 0;
}

/* Init function */

int sysFeatures_init_intel_spec_ctrl(_SysFeatureList* out)
{
    return 0;
}
