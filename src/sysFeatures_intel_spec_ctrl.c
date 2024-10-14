#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <sysFeatures_types.h>
#include <sysFeatures_common.h>
#include <likwid.h>
#include <error.h>
#include <cpuid.h>
#include <bitUtil.h>
#include <registers.h>

static int intel_cpu_spec_ibrs_tester(void)
{
    unsigned eax = 0x07, ebx = 0, ecx = 0, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    return testBit(edx, 26);
}

static int intel_cpu_spec_ibrs_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_SPEC_CTRL, 0, false, value);
}

static int intel_cpu_spec_stibp_tester(void)
{
    unsigned eax = 0x07, ebx = 0, ecx = 0, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    return testBit(edx, 27);
}

static int intel_cpu_spec_stibp_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_SPEC_CTRL, 1, false, value);
}

static int intel_cpu_spec_ssbd_tester(void)
{
    unsigned eax = 0x07, ebx = 0, ecx = 0, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    return testBit(edx, 31);
}

static int intel_cpu_spec_ssbd_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_SPEC_CTRL, 2, true, value);
}

static int intel_cpu_spec_ipred_dis_tester(void)
{
    unsigned eax = 0x07, ebx = 0, ecx = 0x02, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    return testBit(edx, 1);
}

static int intel_cpu_spec_ipred_dis_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_SPEC_CTRL, 3, false, value);
}

static int intel_cpu_spec_rrsba_dis_tester(void)
{
    unsigned eax = 0x07, ebx = 0, ecx = 0x02, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    return testBit(edx, 2);
}

static int intel_cpu_spec_rrsba_dis_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_SPEC_CTRL, 5, true, value);
}

static int intel_cpu_spec_psfd_tester(void)
{
    unsigned eax = 0x07, ebx = 0, edx = 0x02, ecx = 0;
    CPUID(eax, ebx, ecx, edx);
    return testBit(edx, 0);
}

static int intel_cpu_spec_psfd_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_SPEC_CTRL, 7, true, value);
}

static int intel_cpu_spec_ddpd_tester(void)
{
    unsigned eax = 0x07, ebx = 0, ecx = 0x02, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    return testBit(edx, 3);
}

static int intel_cpu_spec_ddpd_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_SPEC_CTRL, 8, true, value);
}

static int intel_cpu_spec_ctrl(void)
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

static _SysFeature intel_cpu_spec_ctrl_features[] = {
    {"ibrs", "spec_ctrl", "Indirect Branch Restricted Speculation", intel_cpu_spec_ibrs_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_ibrs_tester},
    {"stibp", "spec_ctrl", "Single Thread Indirect Branch Predictors", intel_cpu_spec_stibp_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_stibp_tester},
    {"ssbd", "spec_ctrl", "Speculative Store Bypass Disable", intel_cpu_spec_ssbd_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_ssbd_tester},
    {"ipred_dis", "spec_ctrl", "", intel_cpu_spec_ipred_dis_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_ipred_dis_tester},
    {"rrsba_dis", "spec_ctrl", "", intel_cpu_spec_rrsba_dis_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_rrsba_dis_tester},
    {"psfd", "spec_ctrl", "Fast Store Forwarding Predictor", intel_cpu_spec_psfd_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_psfd_tester},
    {"ddpd", "spec_ctrl", "Data Dependent Prefetcher", intel_cpu_spec_ddpd_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_ddpd_tester},
};

const _SysFeatureList likwid_sysft_intel_cpu_spec_ctrl_feature_list = {
    .num_features = ARRAY_COUNT(intel_cpu_spec_ctrl_features),
    .tester = intel_cpu_spec_ctrl,
    .features = intel_cpu_spec_ctrl_features,
};
