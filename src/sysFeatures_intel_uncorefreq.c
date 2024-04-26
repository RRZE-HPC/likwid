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

int intel_uncorefreq_test()
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
    for (int i = 0; i < topo->numHWThreads; i++)
    {
        for (int j = 0; j < topo->numSockets; j++)
        {
            HWThread* t = &topo->threadPool[i];
            if (t->inCpuSet)
            {
                uint64_t tmp = 0;
                if (t->packageId != j) continue;
                err = HPMaddThread(t->apicId);
                if (err < 0) continue;
                err = HPMread(t->apicId, MSR_DEV, MSR_UNCORE_FREQ, &tmp);
                if (err == 0)
                {
                    err = HPMread(t->apicId, MSR_DEV, MSR_UNCORE_FREQ_READ, &tmp);
                    if (err == 0)
                    {
                        valid++;
                    }
                }
                break;
            }
        }
    }
    DEBUG_PRINT(DEBUGLEV_INFO, Failed to access Uncore frequency registers);
    return valid == topo->numSockets;
}

int intel_uncore_cur_freq_getter(LikwidDevice_t device, char** value)
{
    int err = 0;
    CpuTopology_t topo = NULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
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
    for (int i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if (t->packageId == device->id.simple.id && t->inCpuSet)
        {
            uint64_t tmp = 0;
            err = HPMaddThread(t->apicId);
            if (err < 0) continue;
            err = HPMread(t->apicId, MSR_DEV, MSR_UNCORE_FREQ_READ, &tmp);
            if (err == 0)
            {
                tmp = (tmp & 0xFFULL) * 100;
                return _uint64_to_string(tmp, value);
            }
        }
    }
    return -ENODEV;
}

int intel_uncore_min_freq_getter(LikwidDevice_t device, char** value)
{
    int err = 0;
    CpuTopology_t topo = NULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
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
    for (int i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if (t->packageId == device->id.simple.id && t->inCpuSet)
        {
            uint64_t tmp = 0;
            err = HPMaddThread(t->apicId);
            if (err < 0) continue;
            err = HPMread(t->apicId, MSR_DEV, MSR_UNCORE_FREQ, &tmp);
            if (err == 0)
            {
                tmp = ((tmp>>8) & 0xFFULL) * 100;
                return _uint64_to_string(tmp, value);
            }
        }
    }
    return -ENODEV;
}

int intel_uncore_max_freq_getter(LikwidDevice_t device, char** value)
{
    int err = 0;
    CpuTopology_t topo = NULL;
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
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
    for (int i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if (t->packageId == device->id.simple.id && t->inCpuSet)
        {
            uint64_t tmp = 0;
            err = HPMaddThread(t->apicId);
            if (err < 0) continue;
            err = HPMread(t->apicId, MSR_DEV, MSR_UNCORE_FREQ, &tmp);
            if (err == 0)
            {
                tmp = (tmp & 0xFFULL) * 100;;
                return _uint64_to_string(tmp, value);
            }
        }
    }
    return -ENODEV;
}

