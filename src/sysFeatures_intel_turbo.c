#include <sysFeatures_intel_turbo.h>

#include <sysFeatures_intel.h>

#include <bitUtil.h>
#include <cpuid.h>
#include <error.h>
#include <registers.h>
#include <sysFeatures_common.h>

static int intel_cpu_turbo_test(void)
{
    uint32_t eax = 0x01, ebx, ecx = 0x0, edx;
    CPUID(eax, ebx, ecx, edx);
    if (field32(ecx, 7, 1) == 0)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Intel SpeedStep not supported by architecture);
        return 0;
    }

    return likwid_sysft_foreach_hwt_testmsr(MSR_IA32_MISC_ENABLE);
}

int intel_cpu_turbo_getter(const LikwidDevice_t device, char** value)
{
    if (intel_cpu_turbo_test())
    {
        return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_MISC_ENABLE, 36, true, value);
    }
    return -ENOTSUP;
}

int intel_cpu_turbo_setter(const LikwidDevice_t device, const char* value)
{
    if (intel_cpu_turbo_test())
    {
        return likwid_sysft_writemsr_bit_from_string(device, MSR_IA32_MISC_ENABLE, 36, true, value);
    }
    return -ENOTSUP;
}

static _SysFeature intel_cpu_turbo_features[] = {
    {"turbo", "cpu_freq", "Turbo mode", intel_cpu_turbo_getter, intel_cpu_turbo_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_turbo_test},
};

const _SysFeatureList likwid_sysft_intel_cpu_turbo_feature_list = {
    .num_features = ARRAY_COUNT(intel_cpu_turbo_features),
    .tester = intel_cpu_turbo_test,
    .features = intel_cpu_turbo_features,
};
