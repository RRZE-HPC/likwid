#include <sysFeatures_amd.h>

#include <stdbool.h>

#include <access.h>
#include <sysFeatures_amd_rapl.h>

int sysFeatures_init_x86_amd(_SysFeatureList* out)
{
    int err = sysFeatures_init_generic(amd_arch_features, out);
    if (err < 0)
    {
        ERROR_PRINT(Failed to init general x86 HWFetures);
        return err;
    }
    err = sysFeatures_init_amd_rapl(out);
    if (err < 0)
    {
        ERROR_PRINT(Failed to init AMD RAPL HWFetures);
        return err;
    }
    return 0;
}

#define MSR_AMD19_PREFETCH_CONTROL 0xC0000108
static int amd_cpu_prefetch_control_getter(const LikwidDevice_t device, int bitoffset, int width, bool invert, char** value)
{
    uint64_t msrData = 0;
    int err = HPMread(device->id.simple.id, MSR_DEV, MSR_AMD19_PREFETCH_CONTROL, &msrData);
    if (err < 0)
    {
        return err;
    }
    uint64_t val = field64(msrData, bitoffset, width);
    if (invert)
    {
        val = !val;
    }
    return sysFeatures_uint64_to_string(val, value);
}

int amd_cpu_prefetch_control_setter(const LikwidDevice_t device, int bitoffset, int width, bool invert, const char* value)
{
    uint64_t control;
    int err = sysFeatures_string_to_uint64(value, &control);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, MSR_AMD19_PREFETCH_CONTROL, &msrData);
    if (err < 0)
    {
        return err;
    }
    if (invert)
    {
        control = !control;
    }
    field64set(&msrData, bitoffset, width, control);
    return HPMwrite(device->id.simple.id, MSR_DEV, MSR_AMD19_PREFETCH_CONTROL, msrData);
}

int amd_cpu_l1_stream_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_prefetch_control_getter(device, 0, 1, true, value);
}

int amd_cpu_l1_stream_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_prefetch_control_setter(device, 0, 1, true, value);
}

int amd_cpu_l1_stride_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_prefetch_control_getter(device, 1, 1, true, value);
}

int amd_cpu_l1_stride_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_prefetch_control_setter(device, 1, 1, true, value);
}

int amd_cpu_l1_region_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_prefetch_control_getter(device, 2, 1, true, value);
}

int amd_cpu_l1_region_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_prefetch_control_setter(device, 2, 1, true, value);
}

int amd_cpu_l2_stream_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_prefetch_control_getter(device, 3, 1, true, value);
}

int amd_cpu_l2_stream_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_prefetch_control_setter(device, 3, 1, true, value);
}

int amd_cpu_up_down_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_prefetch_control_getter(device, 5, 1, true, value);
}

int amd_cpu_up_down_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_prefetch_control_setter(device, 5, 1, true, value);
}

#define AMD_K19_SPEC_CONTROL 0x00000048
int amd_cpu_spec_control_getter(const LikwidDevice_t device, int bitoffset, int width, bool invert, char** value)
{
    uint64_t msrData = 0;
    int err = HPMread(device->id.simple.id, MSR_DEV, AMD_K19_SPEC_CONTROL, &msrData);
    if (err < 0)
    {
        return err;
    }
    uint64_t control = field64(msrData, bitoffset, width);
    if (invert)
    {
        control = !control;
    }
    return sysFeatures_uint64_to_string(control, value);
}

int amd_cpu_spec_control_setter(const LikwidDevice_t device, int bitoffset, int width, bool invert, const char* value)
{
    uint64_t control = 0x0;
    int err = sysFeatures_string_to_uint64(value, &control);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, AMD_K19_SPEC_CONTROL, &msrData);
    if (err < 0)
    {
        return err;
    }
    if (invert)
    {
        control = !control;
    }
    field64set(&msrData, bitoffset, width, control);
    return HPMwrite(device->id.simple.id, MSR_DEV, AMD_K19_SPEC_CONTROL, msrData);
}

int amd_cpu_spec_ibrs_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_spec_control_getter(device, 0, 1, false, value);
}

int amd_cpu_spec_ibrs_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_spec_control_setter(device, 0, 1, false, value);
}

int amd_cpu_spec_stibp_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_spec_control_getter(device, 1, 1, false, value);
}

int amd_cpu_spec_stibp_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_spec_control_setter(device, 1, 1, false, value);
}

int amd_cpu_spec_ssbd_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_spec_control_getter(device, 2, 1, true, value);
}

int amd_cpu_spec_ssbd_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_spec_control_setter(device, 2, 1, true, value);
}

int amd_cpu_spec_pfsd_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_spec_control_getter(device, 7, 1, true, value);
}

int amd_cpu_spec_pfsd_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_spec_control_setter(device, 7, 1, true, value);
}

#define AMD_K19_L1D_FLUSH_REGISTER 0x0000010B
int amd_cpu_flush_l1(const LikwidDevice_t device, const char* value)
{
    uint64_t flush;
    int err = sysFeatures_string_to_uint64(value, &flush);
    if (err < 0)
    {
        return err;
    }
    return HPMwrite(device->id.simple.id, MSR_DEV, AMD_K19_L1D_FLUSH_REGISTER, flush & 0x1);
}

#define AMD_K19_HWCONFIG_REGISTER 0xC0010015
int amd_cpu_hwconfig_getter(const LikwidDevice_t device, int bitoffset, int width, bool invert, char** value)
{
    uint64_t msrData = 0x0;
    int err = HPMread(device->id.simple.id, MSR_DEV, AMD_K19_HWCONFIG_REGISTER, &msrData);
    if (err < 0)
    {
        return err;
    }
    uint64_t hwconfig = field64(msrData, bitoffset, width);
    if (invert)
    {
        hwconfig = !hwconfig;
    }
    return sysFeatures_uint64_to_string(hwconfig, value);
}

int amd_cpu_hwconfig_setter(const LikwidDevice_t device, int bitoffset, int width, bool invert, const char* value)
{
    uint64_t hwconfig;
    int err = sysFeatures_string_to_uint64(value, &hwconfig);
    if (err < 0)
    {
        return err;
    }
    uint64_t msrData = 0;
    err = HPMread(device->id.simple.id, MSR_DEV, AMD_K19_HWCONFIG_REGISTER, &msrData);
    if (err < 0)
    {
        return err;
    }
    if (invert)
    {
        hwconfig = !hwconfig;
    }
    field64set(&msrData, bitoffset, width, hwconfig);
    return HPMwrite(device->id.simple.id, MSR_DEV, AMD_K19_HWCONFIG_REGISTER, msrData);
}

int amd_cpu_hwconfig_cpddis_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_hwconfig_getter(device, 25, 1, true, value);
}

int amd_cpu_hwconfig_cpddis_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_hwconfig_setter(device, 25, 1, true, value);
}
