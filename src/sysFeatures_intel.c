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
    int err = sysFeatures_init_generic(intel_arch_features, out);
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

int intel_cpu_msr_register_getter(const LikwidDevice_t device, uint32_t reg, int bitoffset, int width, bool invert, char** value)
{
    if (device->type != DEVICE_TYPE_HWTHREAD)
    {
        return -ENODEV;
    }
    int err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    err = HPMaddThread(device->id.simple.id);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0x0;
    err = HPMread(device->id.simple.id, MSR_DEV, reg, &msrData);
    if (err < 0)
    {
        return err;
    }
    uint64_t result = field64(msrData, bitoffset, width);
    if (invert)
    {
        result = !result;
    }
    return sysFeatures_uint64_to_string(result, value);
}

int intel_cpu_msr_register_setter(const LikwidDevice_t device, uint32_t reg, int bitoffset, int width, bool invert, const char* value)
{
    if (device->type != DEVICE_TYPE_HWTHREAD)
    {
        return -ENODEV;
    }
    uint64_t intval;
    int err = sysFeatures_string_to_uint64(value, &intval);
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
    if (invert)
    {
        intval = !intval;
    }
    field64set(&msrData, bitoffset, width, intval);
    return HPMwrite(device->id.simple.id, MSR_DEV, reg, msrData);
}



/* Intel Turbo */
int intel_cpu_turbo_test(void)
{
    uint32_t eax = 0x01, ebx, ecx = 0x0, edx;
    CPUID(eax, ebx, ecx, edx);
    if (field32(ecx, 7, 1) == 0)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Intel SpeedStep not supported by architecture);
        return 0;
    }

    return likwid_sysft_foreach_hwt_testmsr(MSR_IA32_MISC_ENABLE);
}

int intel_cpu_turbo_getter(const LikwidDevice_t device, char** value)
{
    if (intel_cpu_turbo_test())
    {
        return intel_cpu_msr_register_getter(device, MSR_IA32_MISC_ENABLE, 36, 1, true, value);
    }
    return -ENOTSUP;
}

int intel_cpu_turbo_setter(const LikwidDevice_t device, const char* value)
{
    if (intel_cpu_turbo_test())
    {
        return intel_cpu_msr_register_setter(device, MSR_IA32_MISC_ENABLE, 36, 1, true, value);
    }
    return -ENOTSUP;
}
