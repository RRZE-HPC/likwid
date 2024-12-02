#include <sysFeatures_intel_thermal.h>

#include <string.h>
#include <stdbool.h>

#include <sysFeatures_common.h>
#include <topology.h>
#include <registers.h>

static int intel_thermal_temperature_getter(const LikwidDevice_t device, bool core, char **value)
{
    if (!device || !value)
        return -EINVAL;

    int err;
    _LikwidDevice socketDev;
    if (core)
    {
        /* If we read temperature from the core, we need to determine its socket.
         * That is because the TEMPERATURE_TARGET is per socket only. */
        err = topology_init();
        if (err < 0)
            return err;

        bool found = false;
        CpuTopology_t topo = get_cpuTopology();
        for (unsigned i = 0; i < topo->numHWThreads; i++)
        {
            const HWThread *t = &topo->threadPool[i];
            if (t->packageId != i)
                continue;

            memset(&socketDev, 0, sizeof(socketDev));
            socketDev.type = DEVICE_TYPE_SOCKET;
            socketDev.id.simple.id = t->packageId;
            socketDev.internal_id = t->packageId;
            found = true;
            break;
        }

        if (!found)
            return -EINVAL;
    } else {
        socketDev = *device;
    }

    uint64_t therm_status_raw;
    err = likwid_sysft_readmsr(device, core ? IA32_THERM_STATUS : IA32_PACKAGE_THERM_STATUS, &therm_status_raw);
    if (err < 0)
        return err;

    const int readout = (int)field64(therm_status_raw, 16, 7);

    uint64_t temp_target_raw;
    err = likwid_sysft_readmsr(&socketDev, MSR_TEMPERATURE_TARGET, &temp_target_raw);
    if (err < 0)
        return err;

    const int temp_target = (int)field64(temp_target_raw, 16, 8);
    const int temp_offset = (int)field64(temp_target_raw, 24, 6);

    const int final_temp = temp_target - temp_offset - readout;

    return likwid_sysft_uint64_to_string(final_temp, value);
}

static int intel_thermal_temperature_core_getter(LikwidDevice_t dev, char **value)
{
    return intel_thermal_temperature_getter(dev, DEVICE_TYPE_CORE, value);
}

static int intel_thermal_temperature_socket_getter(LikwidDevice_t dev, char **value)
{
    return intel_thermal_temperature_getter(dev, DEVICE_TYPE_SOCKET, value);
}

static int intel_thermal_tester(void)
{
    return cpuid_hasFeature(TM2);
}

static _SysFeature intel_thermal_features[] = {
    {"core_temp", "thermal", "Current CPU temperature (core)", intel_thermal_temperature_core_getter, NULL, DEVICE_TYPE_CORE, NULL, "degrees C"},
    {"pkg_temp", "thermal", "Current CPU temperature (package)", intel_thermal_temperature_socket_getter, NULL, DEVICE_TYPE_SOCKET, NULL, "degrees C"},
};

const _SysFeatureList likwid_sysft_intel_cpu_thermal_feature_list = {
    .num_features = ARRAY_COUNT(intel_thermal_features),
    .tester = intel_thermal_tester,
    .features = intel_thermal_features,
};
