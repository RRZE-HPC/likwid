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
            uint64_t data = 0;
            HWThread* t = &topo->threadPool[j];
            if (t->packageId == i)
            {
                err = HPMread(t->apicId, MSR_DEV, reg, &data);
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
            uint64_t data = 0;
            HWThread* t = &topo->threadPool[j];
            if (t->packageId == i)
            {
                err = HPMread(t->apicId, MSR_DEV, reg, &data);
                if (err == 0 && (data & (1ULL<<bitoffset))) valid++;
                break;
            }
        }
    }
    return valid == topo->numSockets;
}

static int sysFeatures_amd_rapl_energy_status_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data = extractBitField(data, 32, 0);
    data = (uint64_t)(((double)data) * info->energyUnit);

    return _uint64_to_string(data, value);
}

static int sysFeatures_amd_rapl_energy_limit_1_enable_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    uint64_t newdata = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    newdata = (data >> 15) & 0x1;
    return _uint64_to_string(newdata, value);
}

static int sysFeatures_amd_rapl_energy_limit_1_enable_setter(const LikwidDevice_t device, char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    uint64_t limit = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = _string_to_uint64(value, &limit);
    if (err < 0)
    {
        return err;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data &= ~(1ULL << 15);
    data |= ((limit & 0x1ULL) << 15);
    return HPMwrite(device->id.simple.id, MSR_DEV, reg, data);
}

static int sysFeatures_amd_rapl_energy_limit_1_clamp_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data = (data >> 16) & 0x1ULL;
    return _uint64_to_string(data, value);
}

static int sysFeatures_amd_rapl_energy_limit_1_clamp_setter(const LikwidDevice_t device, char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    uint64_t limit = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = _string_to_uint64(value, &limit);
    if (err < 0)
    {
        return err;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data &= ~(1ULL << 16);
    data |= ((limit & 0x1ULL) << 16);
    
    return HPMwrite(device->id.simple.id, MSR_DEV, reg, data);
}

static int sysFeatures_amd_rapl_energy_limit_1_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data = extractBitField(data, 15, 0);
    data = (uint64_t)(((double)data) * info->powerUnit);

    return _uint64_to_string(data, value);
}

static int sysFeatures_amd_rapl_energy_limit_1_setter(const LikwidDevice_t device, char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    uint64_t limit = 0x0ULL;
    if ((!device) || (!value) || (reg == 0x0) || (!info) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = _string_to_uint64(value, &limit);
    if (err < 0)
    {
        return err;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data &= ~(0x7FFF);
    limit = (uint64_t)(((double)limit) / info->powerUnit);
    data |= (limit & 0x7FFF);
    err = HPMwrite(device->id.simple.id, MSR_DEV, reg, data);
    if (err < 0)
    {
        return err;
    }

    return 0;
}

static int sysFeatures_amd_rapl_energy_limit_1_time_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    uint64_t y = extractBitField(data, 5, 17);
    uint64_t z = extractBitField(data, 2, 22);
    data = 0x0ULL;
    data = (1 << y) * info->timeUnit;
    data *= (uint64_t)((1.0 + (((double)z) / 4.0)));
    //data = (uint64_t)(((double)data) * amd_rapl_pkg_info.timeUnit);

    return _uint64_to_string(data, value);
}

static int sysFeatures_amd_rapl_energy_limit_1_time_setter(const LikwidDevice_t device, char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    uint64_t time = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }

    err = _string_to_uint64(value, &time);
    if (err < 0)
    {
        return err;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    time = (uint64_t)(((double)time) / info->timeUnit);
    uint64_t y = (uint64_t)(log2(((double)time)));
    if (y > 0x1F)
        y = 0x7F;
    uint64_t o = (1 << y);
    uint64_t z = (4 * (time - o)) / o;
    time = (y & 0x1F) | ((z & 0x3) << 5);

    data &= ~(0x7F << 17);
    data |= (time << 17);

    return HPMwrite(device->id.simple.id, MSR_DEV, reg, data);
}

static int sysFeatures_amd_rapl_energy_limit_2_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data = extractBitField(data, 15, 32);
    data = (uint64_t)(((double)data) * info->powerUnit);

    return _uint64_to_string(data, value);
}

static int sysFeatures_amd_rapl_energy_limit_2_setter(const LikwidDevice_t device, char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    uint64_t limit = 0x0ULL;
    if ((!device) || (!value) || (reg == 0x0) || (!info) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = _string_to_uint64(value, &limit);
    if (err < 0)
    {
        return err;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data &= ~(0x7FFF << 15);
    limit = (uint64_t)(((double)limit) / info->powerUnit);
    data |= (limit << 15);

    err = HPMwrite(device->id.simple.id, MSR_DEV, reg, data);
    if (err < 0)
    {
        return err;
    }

    return 0;
}

static int sysFeatures_amd_rapl_energy_limit_2_time_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    uint64_t y = extractBitField(data, 5, 49);
    uint64_t z = extractBitField(data, 2, 54);
    data = 0x0ULL;
    data = (1 << y) * info->timeUnit;
    data *= (uint64_t)((1.0 + (((double)z) / 4.0)));
    //data = (uint64_t)(((double)data) * amd_rapl_pkg_info.timeUnit);

    return _uint64_to_string(data, value);
}

static int sysFeatures_amd_rapl_energy_limit_2_time_setter(const LikwidDevice_t device, char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    uint64_t time = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }

    err = _string_to_uint64(value, &time);
    if (err < 0)
    {
        return err;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    time = (uint64_t)(((double)time) / info->timeUnit);
    uint64_t y = (uint64_t)(log2(((double)time)));
    if (y > 0x1F)
        y = 0x7F;
    uint64_t o = (1 << y);
    uint64_t z = (4 * (time - o)) / o;
    time = (y & 0x1F) | ((z & 0x3) << 5);

    data &= ~(0x7FULL << 49);
    data |= (time << 49);

    return HPMwrite(device->id.simple.id, MSR_DEV, reg, data);
}

static int sysFeatures_amd_rapl_energy_limit_2_enable_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data = (data >> 47) & 0x1ULL;

    return _uint64_to_string(data, value);
}

static int sysFeatures_amd_rapl_energy_limit_2_enable_setter(const LikwidDevice_t device, char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    uint64_t limit = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = _string_to_uint64(value, &limit);
    if (err < 0)
    {
        return err;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data &= ~(1ULL << 47);
    data |= ((limit & 0x1ULL) << 47);

    return HPMwrite(device->id.simple.id, MSR_DEV, reg, data);
}

static int sysFeatures_amd_rapl_energy_limit_2_clamp_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data = (data >> 48) & 0x1;

    return _uint64_to_string(data, value);
}

static int sysFeatures_amd_rapl_energy_limit_2_clamp_setter(const LikwidDevice_t device, char* value, uint32_t reg, const RaplDomainInfo* info)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    uint64_t limit = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    err = _string_to_uint64(value, &limit);
    if (err < 0)
    {
        return err;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data &= ~(1ULL << 48);
    data |= ((limit & 0x1) << 48);

    return HPMwrite(device->id.simple.id, MSR_DEV, reg, data);
}

static int sysFeatures_amd_rapl_info_tdp(const LikwidDevice_t device, char** value, uint32_t reg)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data = extractBitField(data, 15, 0);
    data = (uint64_t)(((double)data) * amd_rapl_pkg_info.powerUnit);
    return _uint64_to_string(data, value);
}

static int sysFeatures_amd_rapl_info_min_power(const LikwidDevice_t device, char** value, uint32_t reg)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data = extractBitField(data, 15, 16);
    data = (uint64_t)(((double)data) * amd_rapl_pkg_info.powerUnit);
    return _uint64_to_string(data, value);
}

static int sysFeatures_amd_rapl_info_max_power(const LikwidDevice_t device, char** value, uint32_t reg)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data = extractBitField(data, 15, 32);
    data = (uint64_t)(((double)data) * 100000 * amd_rapl_pkg_info.powerUnit);
    return _uint64_to_string(data, value);
}

static int sysFeatures_amd_rapl_info_max_time(const LikwidDevice_t device, char** value, uint32_t reg)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data = extractBitField(data, 7, 48);
    data = (uint64_t)(((double)data) * amd_rapl_pkg_info.timeUnit);
    return _uint64_to_string(data, value);
}

static int sysFeatures_amd_rapl_policy_getter(const LikwidDevice_t device, char** value, uint32_t reg)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }

    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data = extractBitField(data, 5, 0);
    return _uint64_to_string(data, value);
}

static int sysFeatures_amd_rapl_policy_setter(const LikwidDevice_t device, char* value, uint32_t reg)
{
    int err = 0;
    uint64_t data = 0x0ULL;
    uint64_t policy = 0x0ULL;
    err = _string_to_uint64(value, &policy);
    if (err < 0)
    {
        return err;
    }
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err < 0)
    {
        return err;
    }
    data &= ~(0x1F);
    data |= policy;
    err = HPMwrite(device->id.simple.id, MSR_DEV, reg, data);
    if (err < 0)
    {
        return err;
    }
    return 0;
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
                    amd_rapl_pkg_info.powerUnit = 1000000 / (1 << (data & 0xF));
                    amd_rapl_pkg_info.energyUnit = 1.0 / (1 << ((data >> 8) & 0x1F));
                    amd_rapl_pkg_info.timeUnit = 1000000 / (1 << ((data >> 16) & 0xF));
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
            amd_rapl_core_info.powerUnit = 1000000 / (1 << (data & 0xF));
            amd_rapl_core_info.energyUnit = 1.0 / (1 << ((data >> 8) & 0x1F));
            amd_rapl_core_info.timeUnit = 1000000 / (1 << ((data >> 16) & 0xF));
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
                        amd_rapl_l3_info.powerUnit = 1000000 / (1 << (data & 0xF));
                        amd_rapl_l3_info.energyUnit = 1.0 / (1 << ((data >> 8) & 0x1F));
                        amd_rapl_l3_info.timeUnit = 1000000 / (1 << ((data >> 16) & 0xF));
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
