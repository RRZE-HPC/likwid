#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <registers.h>
#include <cpuid.h>
#include <pci_types.h>
#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <sysFeatures_common.h>
#include <topology.h>
#include <sysFeatures_intel.h>
#include <access.h>

#include <sysFeatures_intel_rapl.h>


int sysFeatures_init_x86_intel(_SysFeatureList* out)
{
    int err = 0;
    err = sysFeatures_init_generic(intel_arch_features, out);
    if (err < 0)
    {
        ERROR_PRINT(Failed to init general Intel HWFetures);
        return err;
    }
    err = sysFeatures_init_intel_rapl(out);
    if (err < 0)
    {
        ERROR_PRINT(Failed to init Intel RAPL HWFetures);
        return err;
    }

    return 0;
}

int intel_cpu_msr_register_getter(LikwidDevice_t device, uint32_t reg, uint64_t mask, uint64_t shift, int invert, char** value)
{
    int err = 0;
    if (device->type != DEVICE_TYPE_HWTHREAD)
    {
        return -ENODEV;
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
    uint64_t data = 0x0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err == 0)
    {
        uint64_t _val = 0x0;
        if (!invert)
        {
            _val = (data & mask) >> shift;
        }
        else
        {
            _val = !((data & mask) >> shift);
        }
        return _uint64_to_string(_val, value);
    }
    return err;
}

int intel_cpu_msr_register_setter(LikwidDevice_t device, uint32_t reg, uint64_t mask, uint64_t shift, int invert, char* value)
{
    int err = 0;
    if (device->type != DEVICE_TYPE_HWTHREAD)
    {
        return -ENODEV;
    }
    uint64_t data = 0x0ULL;
    uint64_t _val = 0x0ULL;
    err = _string_to_uint64(value, &_val);
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
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &data);
    if (err == 0)
    {
        data &= ~(mask);
        if (invert)
        {
            data |= (((!_val) << shift) & mask);
        }
        else
        {
            data |= ((_val << shift) & mask);
        }
        err = HPMwrite(device->id.simple.id, MSR_DEV, reg, data);
    }
    return err;
}



/* Intel Turbo */
int intel_cpu_turbo_test()
{
    int err = 0;
    int valid = 0;
    CpuTopology_t topo = NULL;
    unsigned eax = 0x01, ebx, ecx, edx;
    CPUID(eax, ebx, ecx, edx);
    if (((ecx >> 7) & 0x1) == 0)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Intel SpeedStep not supported by architecture);
        return 0;
    }

    err = topology_init();
    if (err < 0)
    {
        return 0;
    }
    topo = get_cpuTopology();
    err = HPMinit();
    if (err < 0)
	{
		return err;
	}
    for (int j = 0; j < topo->numHWThreads; j++)
    {
        uint64_t data = 0;
        HWThread* t = &topo->threadPool[j];
		err = HPMaddThread(t->apicId);
		if (err < 0) continue;
        err = HPMread(t->apicId, MSR_DEV, MSR_IA32_MISC_ENABLE, &data);
        if (err == 0) valid++;
        break;
    }
    return valid = topo->numHWThreads;
}

int intel_cpu_turbo_getter(LikwidDevice_t device, char** value)
{
    if (intel_cpu_turbo_test())
    {
        return intel_cpu_msr_register_getter(device, MSR_IA32_MISC_ENABLE, (1ULL<<36), 36, 1, value);
    }
    return -ENOTSUP;
}

int intel_cpu_turbo_setter(LikwidDevice_t device, char* value)
{
    if (intel_cpu_turbo_test())
    {
        return intel_cpu_msr_register_setter(device, MSR_IA32_MISC_ENABLE, (1ULL<<36), 36, 1, value);
    }
    return -ENOTSUP;
}
