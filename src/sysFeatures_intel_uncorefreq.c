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

int intel_uncorefreq_test(void)
{
    int err = topology_init();
    if (err < 0)
    {
        return err;
    }
    CpuTopology_t topo = get_cpuTopology();
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    unsigned valid = 0;
    for (unsigned j = 0; j < topo->numSockets; j++)
    {
        for (unsigned i = 0; i < topo->numHWThreads; i++)
        {
            HWThread* t = &topo->threadPool[i];
            if (t->inCpuSet)
            {
                if (t->packageId != j) continue;
                err = HPMaddThread(t->apicId);
                if (err < 0) continue;
                uint64_t msrData = 0;
                err = HPMread(t->apicId, MSR_DEV, MSR_UNCORE_FREQ, &msrData);
                if (err < 0) break;
                err = HPMread(t->apicId, MSR_DEV, MSR_UNCORE_FREQ_READ, &msrData);
                if (err == 0)
                {
                    valid++;
                }
                break;
            }
        }
    }
    if (valid != topo->numSockets)
    {
        DEBUG_PRINT(DEBUGLEV_INFO, Failed to access Uncore frequency registers);
        return 0;
    }
    return 1;
}

int intel_uncore_cur_freq_getter(const LikwidDevice_t device, char** value)
{
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    int err = topology_init();
    if (err < 0)
    {
        return err;
    }
    CpuTopology_t topo = get_cpuTopology();
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    for (unsigned i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if ((int)t->packageId == device->id.simple.id && t->inCpuSet)
        {
            err = HPMaddThread(t->apicId);
            if (err < 0) continue;
            uint64_t msrData = 0;
            err = HPMread(t->apicId, MSR_DEV, MSR_UNCORE_FREQ_READ, &msrData);
            if (err < 0) continue;
            // FIXME we stricly do not know the base frequency, but it is typically 100 Mhz
            return sysFeatures_uint64_to_string(field64(msrData, 0, 8) * 100, value);
        }
    }
    return -ENODEV;
}

int intel_uncore_min_freq_getter(const LikwidDevice_t device, char** value)
{
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    int err = topology_init();
    if (err < 0)
    {
        return err;
    }
    CpuTopology_t topo = get_cpuTopology();
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    for (unsigned i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if ((int)t->packageId == device->id.simple.id && t->inCpuSet)
        {
            err = HPMaddThread(t->apicId);
            if (err < 0) continue;
            uint64_t msrData = 0;
            err = HPMread(t->apicId, MSR_DEV, MSR_UNCORE_FREQ, &msrData);
            if (err < 0) continue;
            // FIXME we stricly do not know the base frequency, but it is typically 100 Mhz
            return sysFeatures_uint64_to_string(field64(msrData, 8, 8) * 100, value);
        }
    }
    return -ENODEV;
}

int intel_uncore_max_freq_getter(const LikwidDevice_t device, char** value)
{
    if ((!device) || (!value) || (device->type != DEVICE_TYPE_SOCKET))
    {
        return -EINVAL;
    }
    int err = topology_init();
    if (err < 0)
    {
        return err;
    }
    CpuTopology_t topo = get_cpuTopology();
    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    for (unsigned  i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if ((int)t->packageId == device->id.simple.id && t->inCpuSet)
        {
            err = HPMaddThread(t->apicId);
            if (err < 0) continue;
            uint64_t msrData = 0;
            err = HPMread(t->apicId, MSR_DEV, MSR_UNCORE_FREQ, &msrData);
            if (err < 0) continue;
            // FIXME we stricly do not know the base frequency, but it is typically 100 Mhz
            return sysFeatures_uint64_to_string(field64(msrData, 0, 8) * 100, value);
        }
    }
    return -ENODEV;
}

