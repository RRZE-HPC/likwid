#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <sysFeatures_intel.h>
#include <sysFeatures_intel_uncorefreq.h>
#include <access.h>
#include <sysFeatures_common.h>
#include <registers.h>
#include <bitUtil.h>

static int intel_uncorefreq_test(void)
{
    const int retval1 = likwid_sysft_foreach_socket_testmsr(MSR_UNCORE_FREQ);
    if (retval1 < 0)
    {
        return retval1;
    }
    const int retval2 = likwid_sysft_foreach_socket_testmsr(MSR_UNCORE_FREQ_READ);
    if (retval2 < 0)
    {
        return retval2;
    }
    return retval1 && retval2;
}

static int intel_uncore_cur_freq_getter(const LikwidDevice_t device, char** value)
{
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    uint64_t msrData;
    int err = likwid_sysft_readmsr(device, MSR_UNCORE_FREQ_READ, &msrData);
    if (err < 0)
    {
        return err;
    }
    return likwid_sysft_uint64_to_string(field64(msrData, 0, 8) * 100, value);
}

static int intel_uncore_min_freq_getter(const LikwidDevice_t device, char** value)
{
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    uint64_t msrData;
    int err = likwid_sysft_readmsr(device, MSR_UNCORE_FREQ, &msrData);
    if (err < 0)
    {
        return err;
    }
    return likwid_sysft_uint64_to_string(field64(msrData, 8, 8) * 100, value);
}

static int intel_uncore_max_freq_getter(const LikwidDevice_t device, char** value)
{
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    uint64_t msrData;
    int err = likwid_sysft_readmsr(device, MSR_UNCORE_FREQ, &msrData);
    if (err < 0)
    {
        return err;
    }
    return likwid_sysft_uint64_to_string(field64(msrData, 0, 8) * 100, value);
}

static _SysFeature intel_uncorefreq_features[] = {
    {"cur_uncore_freq", "uncore_freq", "Current Uncore frequency", intel_uncore_cur_freq_getter, NULL, DEVICE_TYPE_SOCKET, NULL, "MHz"},
    {"min_uncore_freq", "uncore_freq", "Minimum Uncore frequency", intel_uncore_min_freq_getter, NULL, DEVICE_TYPE_SOCKET, NULL, "MHz"},
    {"max_uncore_freq", "uncore_freq", "Maximal Uncore frequency", intel_uncore_max_freq_getter, NULL, DEVICE_TYPE_SOCKET, NULL, "MHz"},
};

const _SysFeatureList likwid_sysft_intel_uncorefreq_feature_list = {
    .num_features = ARRAY_COUNT(intel_uncorefreq_features),
    .tester = intel_uncorefreq_test,
    .features = intel_uncorefreq_features,
};
