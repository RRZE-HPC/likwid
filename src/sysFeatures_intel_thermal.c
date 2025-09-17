#include <sysFeatures_intel_thermal.h>

#include <string.h>
#include <stdbool.h>

#include <sysFeatures_common.h>
#include <topology.h>
#include <registers.h>

static cerr_t intel_thermal_temperature_getter(const LikwidDevice_t device, char **value)
{
    assert(device->type == DEVICE_TYPE_SOCKET || device->type == DEVICE_TYPE_CORE);

    const uint64_t reg = (device->type == DEVICE_TYPE_CORE) ? IA32_THERM_STATUS : IA32_PACKAGE_THERM_STATUS;

    uint64_t therm_status_raw;
    if (likwid_sysft_readmsr(device, reg, &therm_status_raw))
        return ERROR_WRAP();

    const int readout = (int)field64(therm_status_raw, 16, 7);

    uint64_t temp_target_raw;
    if (likwid_sysft_readmsr(device, MSR_TEMPERATURE_TARGET, &temp_target_raw))
        return ERROR_WRAP();

    const int temp_target = (int)field64(temp_target_raw, 16, 8);
    const int temp_offset = (int)field64(temp_target_raw, 24, 6);

    const int final_temp = temp_target - temp_offset - readout;

    return ERROR_WRAP_CALL(likwid_sysft_uint64_to_string(final_temp, value));
}

static cerr_t intel_thermal_tester(bool *ok)
{
    *ok = cpuid_hasFeature(TM2) != 0;
    return NULL;
}

static _SysFeature intel_thermal_features[] = {
    {"core_temp", "thermal", "Current CPU temperature (core)", intel_thermal_temperature_getter, NULL, DEVICE_TYPE_CORE, NULL, "degrees C"},
    {"pkg_temp", "thermal", "Current CPU temperature (package)", intel_thermal_temperature_getter, NULL, DEVICE_TYPE_SOCKET, NULL, "degrees C"},
};

const _SysFeatureList likwid_sysft_intel_cpu_thermal_feature_list = {
    .num_features = ARRAY_COUNT(intel_thermal_features),
    .tester = intel_thermal_tester,
    .features = intel_thermal_features,
};
