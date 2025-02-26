#include <sysFeatures_x86_tsc.h>

#include <bitUtil.h>
#include <cpuid.h>
#include <sysFeatures_common.h>
#include <types.h>

static int tsc_available(void)
{
    /* get maximum leaf for extended CPUID */
    uint32_t eax = 0, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    if (eax < 1)
        return 0;

    /* get cpu feature flags */
    eax = 1;
    CPUID(eax, ebx, ecx, edx);
    return field32(edx, 4, 1);
}

static int test_leaf_0x15(void)
{
    uint32_t eax = 0, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    const uint32_t leaf_max = eax;
    return leaf_max >= 0x15;
}

static int denominator_getter(LikwidDevice_t device, char **value)
{
    uint32_t eax = 0x15, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    return likwid_sysft_uint64_to_string(eax, value);
}

static int numerator_getter(LikwidDevice_t device, char **value)
{
    uint32_t eax = 0x15, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    return likwid_sysft_uint64_to_string(ebx, value);
}

static int ratio_tester(void)
{
    if (!tsc_available() || !test_leaf_0x15())
        return 0;

    uint32_t eax = 0x15, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    return ebx > 0;
}

static int crystal_freq_getter(LikwidDevice_t device, char **value)
{
    uint32_t eax = 0x15, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    return likwid_sysft_uint64_to_string(ecx, value);
}

static int crystal_freq_tester(void)
{
    if (!test_leaf_0x15())
        return 0;

    uint32_t eax = 0x15, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    return ecx > 0;
}

static int freq_getter(LikwidDevice_t device, char **value)
{
    uint32_t eax = 0x15, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);

    const double den = eax;
    const double num = ebx;
    const double crystal_clock = ecx;
    const double freq = crystal_clock * num / den;
    return likwid_sysft_double_to_string(freq, value);
}

static int freq_tester(void)
{
    return tsc_available() && ratio_tester() && crystal_freq_tester();
}

static int invariant_tester(void)
{
    if (!tsc_available())
        return 0;

    /* get maximum leaf for extended CPUID */
    uint32_t eax = 0x80000000, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);

    /* check if maximum leaf is min 0x7 for TSC invariant bit */
    return eax >= 7;
}

static int invariant_getter(LikwidDevice_t device, char **value)
{
    /* check if TSC is invariant */
    uint32_t eax = 0x80000007, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    return likwid_sysft_uint64_to_string(field32(edx, 8, 1), value);
}

static _SysFeature freq_features[] = {
    {"denominator", "tsc", "Denominator of Time Stamp Counter ratio", denominator_getter, NULL, DEVICE_TYPE_SOCKET, ratio_tester},
    {"numerator", "tsc", "Numerator of Time Stamp Counter ratio", numerator_getter, NULL, DEVICE_TYPE_SOCKET, ratio_tester},
    {"crystal_freq", "tsc", "Crystal frequency of Time Stamp Counter", crystal_freq_getter, NULL, DEVICE_TYPE_SOCKET, crystal_freq_tester, "Hz"},
    {"freq", "tsc", "Effective frequency of Time Stamp Counter", freq_getter, NULL, DEVICE_TYPE_SOCKET, freq_tester, "Hz"},
    {"invariant", "tsc", "Time Stamp Counter operates at a fixed frequency", invariant_getter, NULL, DEVICE_TYPE_SOCKET, invariant_tester},
};

const _SysFeatureList likwid_sysft_x86_cpu_freq_feature_list = {
    .num_features = ARRAY_COUNT(freq_features),
    .features = freq_features,
};
