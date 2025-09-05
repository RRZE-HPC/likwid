#include <sysFeatures_x86_tsc.h>

#include <bitUtil.h>
#include <cpuid.h>
#include <sysFeatures_common.h>
#include <types.h>

static bool tsc_available(void)
{
    /* get maximum leaf for extended CPUID */
    uint32_t eax = 0, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    if (eax < 1)
        return false;

    /* get cpu feature flags */
    eax = 1;
    CPUID(eax, ebx, ecx, edx);
    return field32(edx, 4, 1);
}

static bool test_leaf_0x15(void)
{
    uint32_t eax = 0, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    const uint32_t leaf_max = eax;
    return leaf_max >= 0x15;
}

static cerr_t denominator_getter(LikwidDevice_t device, char **value)
{
    (void)device;
    uint32_t eax = 0x15, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    return likwid_sysft_uint64_to_string(eax, value);
}

static cerr_t numerator_getter(LikwidDevice_t device, char **value)
{
    (void)device;
    uint32_t eax = 0x15, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    return likwid_sysft_uint64_to_string(ebx, value);
}

static cerr_t ratio_tester(bool *ok)
{
    *ok = tsc_available() && test_leaf_0x15();
    if (!*ok)
        return NULL;

    uint32_t eax = 0x15, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    *ok = ebx > 0;
    return NULL;
}

static cerr_t crystal_freq_getter(LikwidDevice_t device, char **value)
{
    (void)device;
    uint32_t eax = 0x15, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    return likwid_sysft_uint64_to_string(ecx, value);
}

static cerr_t crystal_freq_tester(bool *ok)
{
    *ok = test_leaf_0x15();
    if (!*ok)
        return NULL;

    uint32_t eax = 0x15, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    *ok = ecx > 0;
    return NULL;
}

static cerr_t freq_getter(LikwidDevice_t device, char **value)
{
    (void)device;
    uint32_t eax = 0x15, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);

    const double den = eax;
    const double num = ebx;
    const double crystal_clock = ecx;
    const double freq = crystal_clock * num / den;
    return likwid_sysft_double_to_string(freq, value);
}

static cerr_t freq_tester(bool *ok)
{
    *ok = tsc_available();
    if (!*ok)
        return NULL;
    
    ratio_tester(ok);
    if (!*ok)
        return NULL;

    return crystal_freq_tester(ok);
}

static cerr_t invariant_tester(bool *ok)
{
    *ok = tsc_available();
    if (!*ok)
        return NULL;

    /* get maximum leaf for extended CPUID */
    uint32_t eax = 0x80000000, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);

    /* check if maximum leaf is min 0x7 for TSC invariant bit */
    *ok = eax >= 7;
    return NULL;
}

static cerr_t invariant_getter(LikwidDevice_t device, char **value)
{
    (void)device;
    /* check if TSC is invariant */
    uint32_t eax = 0x80000007, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);
    return likwid_sysft_uint64_to_string(field32(edx, 8, 1), value);
}

static _SysFeature freq_features[] = {
    {"denominator", "tsc", "Denominator of Time Stamp Counter ratio", denominator_getter, NULL, DEVICE_TYPE_SOCKET, ratio_tester, NULL},
    {"numerator", "tsc", "Numerator of Time Stamp Counter ratio", numerator_getter, NULL, DEVICE_TYPE_SOCKET, ratio_tester, NULL},
    {"crystal_freq", "tsc", "Crystal frequency of Time Stamp Counter", crystal_freq_getter, NULL, DEVICE_TYPE_SOCKET, crystal_freq_tester, "Hz"},
    {"freq", "tsc", "Effective frequency of Time Stamp Counter", freq_getter, NULL, DEVICE_TYPE_SOCKET, freq_tester, "Hz"},
    {"invariant", "tsc", "Time Stamp Counter operates at a fixed frequency", invariant_getter, NULL, DEVICE_TYPE_SOCKET, invariant_tester, NULL},
};

const _SysFeatureList likwid_sysft_x86_cpu_freq_feature_list = {
    .num_features = ARRAY_COUNT(freq_features),
    .features = freq_features,
};
