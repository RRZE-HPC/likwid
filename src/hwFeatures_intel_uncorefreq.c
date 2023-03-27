#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <hwFeatures_types.h>
#include <likwid_device_types.h>
#include <error.h>
#include <hwFeatures_intel.h>
#include <hwFeatures_intel_uncorefreq.h>
#include <access.h>
#include <hwFeatures_common.h>

int intel_uncorefreq_test()
{
    int err = 0;
    CpuTopology_t topo = NULL;
    
    err = topology_init();
    if (err < 0)
    {
        return err;
    }
    topo = get_cpuTopology();
    for (int i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if (t->inCpuSet)
        {
            uint64_t tmp = 0;
            HPMaddThread(t->apicId);
            err = HPMread(t->apicId, MSR_DEV, MSR_UNCORE_FREQ, &tmp);
            if (err == 0)
            {
                err = HPMread(t->apicId, MSR_DEV, MSR_UNCORE_FREQ_READ, &tmp);
                if (err == 0)
                {
                    return 1;
                }
                else
                {
                    ERROR_PRINT(Failed to access Uncore frequency current register);
                }
            }
            else
            {
                ERROR_PRINT(Failed to access Uncore frequency min/max register);
            }
            break;
        }
    }
    ERROR_PRINT(Failed to access Uncore frequency registers);
    return 0;
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
    for (int i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if (t->packageId == device->id.simple.id && t->inCpuSet)
        {
            uint64_t tmp = 0;
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
    for (int i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if (t->packageId == device->id.simple.id && t->inCpuSet)
        {
            uint64_t tmp = 0;
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
    for (int i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if (t->packageId == device->id.simple.id && t->inCpuSet)
        {
            uint64_t tmp = 0;
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

