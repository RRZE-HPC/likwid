#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <sysFeatures_amd.h>
#include <sysFeatures_common.h>
#include <sysFeatures_amd_rapl.h>
#include <access.h>
#include <registers.h>

#include <sysFeatures_common_rapl.h>

static RaplDomainInfo amd_rapl_pkg_info = {0, 0, 0};
static RaplDomainInfo amd_rapl_core_info = {0, 0, 0};
static RaplDomainInfo amd_rapl_l3_info = {0, 0, 0};

static int amd_rapl_register_test(uint32_t reg)
{
    int err = 0;
    int valid = 0;
    CpuTopology_t topo = NULL;

    err = topology_init();
    if (err < 0)
    {
        return err;
    }
    topo = get_cpuTopology();
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    for (int i = 0; i < topo->numSockets; i++)
    {
        for (int j = 0; j < topo->numHWThreads; j++)
        {
            HWThread* t = &topo->threadPool[j];
            if (t->packageId == i)
            {
                uint64_t msrData = 0;
                err = HPMread(t->apicId, MSR_DEV, reg, &msrData);
                if (err == 0) valid++;
                break;
            }
        }
    }
    return valid == topo->numSockets;
}

static int amd_rapl_register_test_bit(uint32_t reg, int bitoffset)
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
    for (int i = 0; i < topo->numSockets; i++)
    {
        for (int j = 0; j < topo->numHWThreads; j++)
        {
            HWThread* t = &topo->threadPool[j];
            if (t->packageId == i)
            {
                uint64_t msrData = 0;
                err = HPMread(t->apicId, MSR_DEV, reg, &msrData);
                if (err == 0 && field64(msrData, bitoffset, 1)) valid++;
                break;
            }
        }
    }
    return valid == topo->numSockets;
}

static int sysFeatures_amd_rapl_energy_status_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t energy = field64(msrData, 0, 32);
    return sysFeatures_double_to_string((double)energy * info->energyUnit, value);
}

static int sysFeatures_amd_rapl_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t enable = field64(msrData, 15, 1);
    return _uint64_to_string(enable, value);
}

static int sysFeatures_amd_rapl_energy_limit_1_enable_setter(const LikwidDevice_t device, char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    uint64_t enable;
    err = _string_to_uint64(value, &enable);
    if (err < 0)
    {
        return err;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    field64set(&msrData, 15, 1, enable);
    return HPMwrite(device->id.simple.id, MSR_DEV, reg, msrData);
}

static int sysFeatures_amd_rapl_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t clamp = field64(msrData, 16, 1);
    return _uint64_to_string(clamp, value);
}

static int sysFeatures_amd_rapl_energy_limit_1_clamp_setter(const LikwidDevice_t device, char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    uint64_t clamp;
    err = _string_to_uint64(value, &clamp);
    if (err < 0)
    {
        return err;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    field64set(&msrData, 16, 1, clamp);
    return HPMwrite(device->id.simple.id, MSR_DEV, reg, msrData);
}

static int sysFeatures_amd_rapl_energy_limit_1_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t powerUnits = field64(msrData, 0, 15);
    const double watts = (double)powerUnits * info->powerUnit;
    return sysFeatures_double_to_string(watts, value);
}

static int sysFeatures_amd_rapl_energy_limit_1_setter(const LikwidDevice_t device, char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (reg == 0x0) || (!info) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    double watts = 0.0;
    err = sysFeatures_string_to_double(value, &watts);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t powerUnits = (uint64_t)round(watts / info->powerUnit);
    field64set(&msrData, 0, 15, MIN(powerUnits, 0x7FFF));
    return HPMwrite(device->id.simple.id, MSR_DEV, reg, msrData);
}

static int sysFeatures_amd_rapl_energy_limit_1_time_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t timeWindow = field64(msrData, 17, 7);
    const double seconds = timeWindow_to_seconds(info, timeWindow);
    return sysFeatures_double_to_string(seconds, value);
}

static int sysFeatures_amd_rapl_energy_limit_1_time_setter(const LikwidDevice_t device, char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    double seconds = 0.0;
    err = sysFeatures_string_to_double(value, &seconds);
    if (err < 0)
    {
        return err;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t timeWindow = seconds_to_timeWindow(info, seconds);
    field64set(&msrData, 17, 7, timeWindow);
    return HPMwrite(device->id.simple.id, MSR_DEV, reg, msrData);
}

static int sysFeatures_amd_rapl_energy_limit_2_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t powerUnits = field64(msrData, 32, 15);
    const double watts = (double)powerUnits * info->powerUnit;
    return sysFeatures_double_to_string(watts, value);
}

static int sysFeatures_amd_rapl_energy_limit_2_setter(const LikwidDevice_t device, char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (reg == 0x0) || (!info) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    double watts = 0.0;
    err = sysFeatures_string_to_double(value, &watts);
    if (err < 0)
    {
        return err;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t powerUnits = (uint64_t)round(watts / info->powerUnit);
    field64set(&msrData, 15, 15, MIN(powerUnits, 0x7FFF));
    return HPMwrite(device->id.simple.id, MSR_DEV, reg, msrData);
}

static int sysFeatures_amd_rapl_energy_limit_2_time_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t timeWindow = field64(msrData, 49, 7);
    const double seconds = timeWindow_to_seconds(info, timeWindow);
    return sysFeatures_double_to_string(seconds, value);
}

static int sysFeatures_amd_rapl_energy_limit_2_time_setter(const LikwidDevice_t device, char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    double seconds = 0.0;
    err = sysFeatures_string_to_double(value, &seconds);
    if (err < 0)
    {
        return err;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t timeWindow = seconds_to_timeWindow(info, seconds);
    field64set(&msrData, 49, 7, timeWindow);
    return HPMwrite(device->id.simple.id, MSR_DEV, reg, msrData);
}

static int sysFeatures_amd_rapl_energy_limit_2_enable_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t enable = field64(msrData, 47, 1);
    return _uint64_to_string(enable, value);
}

static int sysFeatures_amd_rapl_energy_limit_2_enable_setter(const LikwidDevice_t device, char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    uint64_t enable;
    err = _string_to_uint64(value, &enable);
    if (err < 0)
    {
        return err;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    field64set(&msrData, 47, 1, enable);
    return HPMwrite(device->id.simple.id, MSR_DEV, reg, msrData);
}

static int sysFeatures_amd_rapl_energy_limit_2_clamp_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t clamp = field64(msrData, 48, 1);
    return _uint64_to_string(clamp, value);
}

static int sysFeatures_amd_rapl_energy_limit_2_clamp_setter(const LikwidDevice_t device, char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    uint64_t clamp;
    err = _string_to_uint64(value, &clamp);
    if (err < 0)
    {
        return err;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    field64set(&msrData, 48, 1, clamp);
    return HPMwrite(device->id.simple.id, MSR_DEV, reg, msrData);
}

static int sysFeatures_amd_rapl_info_tdp(const LikwidDevice_t device, char** value, uint32_t reg)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t powerUnits = field64(msrData, 0, 15);
    return sysFeatures_double_to_string((double)powerUnits * amd_rapl_pkg_info.powerUnit, value);
}

static int sysFeatures_amd_rapl_info_min_power(const LikwidDevice_t device, char** value, uint32_t reg)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t powerUnits = field64(msrData, 16, 15);
    return sysFeatures_double_to_string((double)powerUnits * amd_rapl_pkg_info.powerUnit, value);
}

static int sysFeatures_amd_rapl_info_max_power(const LikwidDevice_t device, char** value, uint32_t reg)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t powerUnits = field64(msrData, 32, 15);
    return sysFeatures_double_to_string((double)powerUnits * amd_rapl_pkg_info.powerUnit, value);
}

static int sysFeatures_amd_rapl_info_max_time(const LikwidDevice_t device, char** value, uint32_t reg)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t timeUnits = field64(msrData, 48, 7);
    return sysFeatures_double_to_string((double)timeUnits * amd_rapl_pkg_info.timeUnit, value);
}

static int sysFeatures_amd_rapl_policy_getter(const LikwidDevice_t device, char** value, uint32_t reg)
{
    int err = 0;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    const uint64_t policy = field64(msrData, 0, 5);
    return _uint64_to_string(policy, value);
}

static int sysFeatures_amd_rapl_policy_setter(const LikwidDevice_t device, char* value, uint32_t reg)
{
    int err = 0;
    uint64_t policy;
    err = _string_to_uint64(value, &policy);
    if (err < 0)
    {
        return err;
    }
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    msrData &= ~(0x1F);
    msrData |= (policy & 0x1F);
    return HPMwrite(device->id.simple.id, MSR_DEV, reg, msrData);
}


/*********************************************************************************************************************/
/*                          Amd RAPL (PKG domain)                                                                  */
/*********************************************************************************************************************/

int amd_rapl_pkg_test()
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
    for (int i = 0; i < topo->numSockets; i++)
    {
        for (int j = 0; j < topo->numHWThreads; j++)
        {
            uint64_t data = 0;
            HWThread* t = &topo->threadPool[j];
            if (t->packageId == i)
            {
                err = HPMread(t->apicId, MSR_DEV, MSR_AMD17_RAPL_POWER_UNIT, &data);
                if (err == 0) valid++;
                if (amd_rapl_pkg_info.powerUnit == 0 && amd_rapl_pkg_info.energyUnit == 0 && amd_rapl_pkg_info.timeUnit == 0)
                {
                    amd_rapl_pkg_info.powerUnit = 1.0 / (1 << field64(data, 0, 4));
                    amd_rapl_pkg_info.energyUnit = 1.0 / (1 << field64(data, 8, 5));
                    amd_rapl_pkg_info.timeUnit = 1.0 / (1 << field64(data, 16, 4));
                }
                break;
            }
        }
    }
    return valid == topo->numSockets;
}

/*int amd_rapl_pkg_limit_test_lock()*/
/*{*/
/*    return amd_rapl_register_test_bit(MSR_PKG_RAPL_POWER_LIMIT, 63);*/
/*}*/


int sysFeatures_amd_pkg_energy_status_test()
{
    return amd_rapl_register_test(MSR_AMD17_RAPL_PKG_STATUS);
}

int sysFeatures_amd_pkg_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_amd_rapl_energy_status_getter(device, value, MSR_AMD17_RAPL_PKG_STATUS, &amd_rapl_pkg_info);
}

/*int sysFeatures_amd_pkg_energy_limit_test()*/
/*{*/
/*    return amd_rapl_register_test(MSR_PKG_RAPL_POWER_LIMIT);*/
/*}*/

/*int sysFeatures_amd_pkg_energy_limit_1_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &amd_rapl_pkg_info);*/
/*}*/
/*int sysFeatures_amd_pkg_energy_limit_1_setter(const LikwidDevice_t device, char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &amd_rapl_pkg_info);*/
/*}*/
/*int sysFeatures_amd_pkg_energy_limit_1_time_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_time_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &amd_rapl_pkg_info);*/
/*}*/
/*int sysFeatures_amd_pkg_energy_limit_1_time_setter(const LikwidDevice_t device, char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_time_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &amd_rapl_pkg_info);*/
/*}*/
/*int sysFeatures_amd_pkg_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_enable_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &amd_rapl_pkg_info);*/
/*}*/
/*int sysFeatures_amd_pkg_energy_limit_1_enable_setter(const LikwidDevice_t device, char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_enable_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &amd_rapl_pkg_info);*/
/*}*/
/*int sysFeatures_amd_pkg_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_clamp_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &amd_rapl_pkg_info);*/
/*}*/
/*int sysFeatures_amd_pkg_energy_limit_1_clamp_setter(const LikwidDevice_t device, char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_clamp_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &amd_rapl_pkg_info);*/
/*}*/



/*int sysFeatures_amd_pkg_energy_limit_2_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_2_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &amd_rapl_pkg_info);*/
/*}*/
/*int sysFeatures_amd_pkg_energy_limit_2_setter(const LikwidDevice_t device, const char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_2_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &amd_rapl_pkg_info);*/
/*}*/
/*int sysFeatures_amd_pkg_energy_limit_2_time_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_2_time_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &amd_rapl_pkg_info);*/
/*}*/
/*int sysFeatures_amd_pkg_energy_limit_2_time_setter(const LikwidDevice_t device, const char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_2_time_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &amd_rapl_pkg_info);*/
/*}*/
/*int sysFeatures_amd_pkg_energy_limit_2_enable_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_2_enable_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &amd_rapl_pkg_info);*/
/*}*/
/*int sysFeatures_amd_pkg_energy_limit_2_enable_setter(const LikwidDevice_t device, const char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_2_enable_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &amd_rapl_pkg_info);*/
/*}*/
/*int sysFeatures_amd_pkg_energy_limit_2_clamp_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_2_clamp_getter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &amd_rapl_pkg_info);*/
/*}*/
/*int sysFeatures_amd_pkg_energy_limit_2_clamp_setter(const LikwidDevice_t device, const char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_2_clamp_setter(device, value, MSR_PKG_RAPL_POWER_LIMIT, &amd_rapl_pkg_info);*/
/*}*/

/*int sysFeatures_amd_pkg_info_test()*/
/*{*/
/*    return amd_rapl_register_test(MSR_PKG_POWER_INFO);*/
/*}*/

/*int sysFeatures_amd_pkg_info_tdp(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_info_tdp(device, value, MSR_PKG_POWER_INFO);*/
/*}*/

/*int sysFeatures_amd_pkg_info_min_power(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_info_min_power(device, value, MSR_PKG_POWER_INFO);*/
/*}*/

/*int sysFeatures_amd_pkg_info_max_power(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_info_max_power(device, value, MSR_PKG_POWER_INFO);*/
/*}*/

/*int sysFeatures_amd_pkg_info_max_time(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_info_max_time(device, value, MSR_PKG_POWER_INFO);*/
/*}*/


/*********************************************************************************************************************/
/*                          AMD RAPL (CORE domain)                                                                 */
/*********************************************************************************************************************/

int amd_rapl_core_test()
{
    int err = 0;
    int valid = 0;
    CpuTopology_t topo = NULL;
    CpuInfo_t info = NULL;

    err = topology_init();
    if (err < 0)
    {
        return 0;
    }
    topo = get_cpuTopology();
    info = get_cpuInfo();
    for (int j = 0; j < topo->numHWThreads; j++)
    {
        uint64_t data = 0x0;
        HWThread* t = &topo->threadPool[j];
        err = HPMread(t->apicId, MSR_DEV, MSR_AMD17_RAPL_POWER_UNIT, &data);
        if (err == 0) valid++;
        if (amd_rapl_core_info.powerUnit == 0 && amd_rapl_core_info.energyUnit == 0 && amd_rapl_core_info.timeUnit == 0)
        {
            amd_rapl_core_info.powerUnit = 1.0 / (1 << field64(data, 0, 4));
            amd_rapl_core_info.energyUnit = 1.0 / (1 << field64(data, 8, 5));
            amd_rapl_core_info.timeUnit = 1.0 / (1 << field64(data, 16, 4));
        }
        break;
    }
    return valid == topo->numHWThreads;
}


int sysFeatures_amd_core_energy_status_test()
{
    return amd_rapl_register_test(MSR_AMD17_RAPL_CORE_STATUS);
}

int sysFeatures_amd_core_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_amd_rapl_energy_status_getter(device, value, MSR_AMD17_RAPL_CORE_STATUS, &amd_rapl_core_info);
}

/*int sysFeatures_amd_core_energy_limit_test()*/
/*{*/
/*    return amd_rapl_register_test(MSR_DRAM_RAPL_POWER_LIMIT);*/
/*}*/
/*int amd_rapl_core_limit_test_lock()*/
/*{*/
/*    return amd_rapl_register_test_bit(MSR_DRAM_RAPL_POWER_LIMIT, 31);*/
/*}*/


/*int sysFeatures_amd_core_energy_limit_1_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_getter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &amd_rapl_core_info);*/
/*}*/
/*int sysFeatures_amd_core_energy_limit_1_setter(const LikwidDevice_t device, const char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_setter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &amd_rapl_core_info);*/
/*}*/
/*int sysFeatures_amd_core_energy_limit_1_time_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_time_getter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &amd_rapl_core_info);*/
/*}*/
/*int sysFeatures_amd_core_energy_limit_1_time_setter(const LikwidDevice_t device, const char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_time_setter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &amd_rapl_core_info);*/
/*}*/
/*int sysFeatures_amd_core_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_enable_getter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &amd_rapl_core_info);*/
/*}*/
/*int sysFeatures_amd_core_energy_limit_1_enable_setter(const LikwidDevice_t device, const char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_enable_setter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &amd_rapl_core_info);*/
/*}*/
/*int sysFeatures_amd_core_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_clamp_getter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &amd_rapl_core_info);*/
/*}*/
/*int sysFeatures_amd_core_energy_limit_1_clamp_setter(const LikwidDevice_t device, const char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_clamp_setter(device, value, MSR_DRAM_RAPL_POWER_LIMIT, &amd_rapl_core_info);*/
/*}*/

/*int sysFeatures_amd_core_info_test()*/
/*{*/
/*    return amd_rapl_register_test(MSR_DRAM_POWER_INFO);*/
/*}*/

/*int sysFeatures_amd_core_info_tdp(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_info_tdp(device, value, MSR_DRAM_POWER_INFO);*/
/*}*/

/*int sysFeatures_amd_core_info_min_power(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_info_min_power(device, value, MSR_DRAM_POWER_INFO);*/
/*}*/

/*int sysFeatures_amd_core_info_max_power(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_info_max_power(device, value, MSR_DRAM_POWER_INFO);*/
/*}*/

/*int sysFeatures_amd_core_info_max_time(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_info_max_time(device, value, MSR_DRAM_POWER_INFO);*/
/*}*/


/*********************************************************************************************************************/
/*                          AMD RAPL (L3 domain)                                                                     */
/*********************************************************************************************************************/

int amd_rapl_l3_test()
{
    int err = 0;
    int valid = 0;
    CpuTopology_t topo = NULL;
    CpuInfo_t info = NULL;

    err = topology_init();
    if (err < 0)
    {
        return 0;
    }
    topo = get_cpuTopology();
    info = get_cpuInfo();
    if (info->family == ZEN3_FAMILY && (info->model == ZEN4_RYZEN || info->model == ZEN4_RYZEN_PRO || info->model == ZEN4_EPYC))
    {
        for (int i = 0; i < topo->numSockets; i++)
        {
            for (int j = 0; j < topo->numHWThreads; j++)
            {
                uint64_t data = 0;
                HWThread* t = &topo->threadPool[j];
                if (t->packageId == i)
                {
                    err = HPMread(t->apicId, MSR_DEV, MSR_AMD19_RAPL_L3_UNIT, &data);
                    if (err == 0) valid++;
                    if (amd_rapl_l3_info.powerUnit == 0 && amd_rapl_l3_info.energyUnit == 0 && amd_rapl_l3_info.timeUnit == 0)
                    {
                        amd_rapl_l3_info.powerUnit = 1.0 / (1 << field64(data, 0, 4));
                        amd_rapl_l3_info.energyUnit = 1.0 / (1 << field64(data, 8, 5));
                        amd_rapl_l3_info.timeUnit = 1.0 / (1 << field64(data, 16, 4));
                    }
                    break;
                }
            }
        }
    }
    return valid == topo->numSockets;
}

int sysFeatures_amd_l3_energy_status_test()
{
    return amd_rapl_register_test(MSR_AMD19_RAPL_L3_STATUS);
}

int sysFeatures_amd_l3_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_amd_rapl_energy_status_getter(device, value, MSR_AMD19_RAPL_L3_STATUS, &amd_rapl_l3_info);
}

/*int sysFeatures_amd_l3_energy_limit_test()*/
/*{*/
/*    return amd_rapl_register_test(MSR_PLATFORM_POWER_LIMIT);*/
/*}*/
/*int amd_rapl_l3_limit_test_lock()*/
/*{*/
/*    return amd_rapl_register_test_bit(MSR_PLATFORM_POWER_LIMIT, 63);*/
/*}*/

/*int sysFeatures_amd_l3_energy_limit_1_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &amd_rapl_l3_info);*/
/*}*/
/*int sysFeatures_amd_l3_energy_limit_1_setter(const LikwidDevice_t device, const char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &amd_rapl_l3_info);*/
/*}*/
/*int sysFeatures_amd_l3_energy_limit_1_time_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_time_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &amd_rapl_l3_info);*/
/*}*/
/*int sysFeatures_amd_l3_energy_limit_1_time_setter(const LikwidDevice_t device, const char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_time_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &amd_rapl_l3_info);*/
/*}*/
/*int sysFeatures_amd_l3_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_enable_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &amd_rapl_l3_info);*/
/*}*/
/*int sysFeatures_amd_l3_energy_limit_1_enable_setter(const LikwidDevice_t device, const char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_enable_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &amd_rapl_l3_info);*/
/*}*/
/*int sysFeatures_amd_l3_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_clamp_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &amd_rapl_l3_info);*/
/*}*/
/*int sysFeatures_amd_l3_energy_limit_1_clamp_setter(const LikwidDevice_t device, const char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_1_clamp_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &amd_rapl_l3_info);*/
/*}*/


/*int sysFeatures_amd_l3_energy_limit_2_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_2_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &amd_rapl_l3_info);*/
/*}*/
/*int sysFeatures_amd_l3_energy_limit_2_setter(const LikwidDevice_t device, const char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_2_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &amd_rapl_l3_info);*/
/*}*/
/*int sysFeatures_amd_l3_energy_limit_2_time_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_2_time_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &amd_rapl_l3_info);*/
/*}*/
/*int sysFeatures_amd_l3_energy_limit_2_time_setter(const LikwidDevice_t device, const char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_2_time_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &amd_rapl_l3_info);*/
/*}*/
/*int sysFeatures_amd_l3_energy_limit_2_enable_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_2_enable_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &amd_rapl_l3_info);*/
/*}*/
/*int sysFeatures_amd_l3_energy_limit_2_enable_setter(const LikwidDevice_t device, const char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_2_enable_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &amd_rapl_l3_info);*/
/*}*/
/*int sysFeatures_amd_l3_energy_limit_2_clamp_getter(const LikwidDevice_t device, char** value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_2_clamp_getter(device, value, MSR_PLATFORM_POWER_LIMIT, &amd_rapl_l3_info);*/
/*}*/
/*int sysFeatures_amd_l3_energy_limit_2_clamp_setter(const LikwidDevice_t device, const char* value)*/
/*{*/
/*    return sysFeatures_amd_rapl_energy_limit_2_clamp_setter(device, value, MSR_PLATFORM_POWER_LIMIT, &amd_rapl_l3_info);*/
/*}*/


/* Init function */

int sysFeatures_init_amd_rapl(_SysFeatureList* out)
{
    int err = 0;
    if (amd_rapl_pkg_test())
    {
        DEBUG_PRINT(DEBUGLEV_INFO, Register Amd RAPL PKG domain);
        err = register_features(out, &amd_rapl_pkg_feature_list);
        if (err < 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, RAPL domain PKG not supported);
        }
    }
    if (amd_rapl_core_test())
    {
        DEBUG_PRINT(DEBUGLEV_INFO, Register Amd RAPL CORE domain);
        err = register_features(out, &amd_rapl_core_feature_list);
        if (err < 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, RAPL domain CORE not supported);
        }
    }
    if (amd_rapl_l3_test())
    {
        DEBUG_PRINT(DEBUGLEV_INFO, Register Amd RAPL L3 domain);
        err = register_features(out, &amd_rapl_l3_feature_list);
        if (err < 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, RAPL domain L3 not supported);
        }
    }
    return 0;
}
