#include <sysFeatures_amd.h>

#include <access.h>
#include <sysFeatures_amd_rapl.h>

int sysFeatures_init_x86_amd(_SysFeatureList* out)
{
    int err = 0;
    err = sysFeatures_init_generic(amd_arch_features, out);
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
int amd_cpu_prefetch_control_getter(const LikwidDevice_t device, uint64_t mask, uint64_t shift, int invert, char** value)
{
    int err = 0;
    /*if ((err = request_hw_access(dev)) != 0)
    {
        return err;
    }
    if (dev->scope != DEVICE_SCOPE_device)
    {
        return -ENODEV;
    }*/
    uint64_t data = 0x0;
    uint64_t _val = 0x0;
    err = HPMread(device->id.simple.id, MSR_DEV, MSR_AMD19_PREFETCH_CONTROL, &data);
    if (err == 0)
    {
        if (!invert)
        {
                //*value = (data >> shift) & mask;
            _val = (data & mask) >> shift;
        }
        else
        {
                //*value = !((data >> shift) & mask);
            _val = !((data & mask) >> shift);
        }
        return _uint64_to_string(_val, value);
    }
    return err;
}

int amd_cpu_prefetch_control_setter(const LikwidDevice_t device, uint64_t mask, uint64_t shift, int invert, const char* value)
{
    int err = 0;
    /*if ((err = request_hw_access(dev)) != 0)
    {
        return err;
    }
    if (dev->scope != DEVICE_SCOPE_device)
    {
        return -ENODEV;
    }*/
    uint64_t data = 0x0;
    uint64_t _val = 0x0ULL;
    err = _string_to_uint64(value, &_val);
    if (err < 0)
    {
        return err;
    }
    err = HPMread(device->id.simple.id, MSR_DEV, MSR_AMD19_PREFETCH_CONTROL, &data);
    if (err == 0)
    {
        data &= ~(mask);
        if (invert)
            _val = !_val;
        data |= ((_val << shift) & mask);
        err = HPMwrite(device->id.simple.id, MSR_DEV, MSR_AMD19_PREFETCH_CONTROL, data);
    }
    return err;
}

int amd_cpu_l1_stream_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_prefetch_control_getter(device, 0x1, 0, 1, value);
}

int amd_cpu_l1_stream_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_prefetch_control_setter(device, 0x1, 0, 1, value);
}

int amd_cpu_l1_stride_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_prefetch_control_getter(device, 0x2, 1, 1, value);
}

int amd_cpu_l1_stride_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_prefetch_control_setter(device, 0x2, 1, 1, value);
}

int amd_cpu_l1_region_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_prefetch_control_getter(device, 0x4, 2, 1, value);
}

int amd_cpu_l1_region_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_prefetch_control_setter(device, 0x4, 2, 1, value);
}

int amd_cpu_l2_stream_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_prefetch_control_getter(device, 0x8, 3, 1, value);
}

int amd_cpu_l2_stream_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_prefetch_control_setter(device, 0x8, 3, 1, value);
}

int amd_cpu_up_down_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_prefetch_control_getter(device, 0x20, 5, 1, value);
}

int amd_cpu_up_down_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_prefetch_control_setter(device, 0x20, 5, 1, value);
}

#define AMD_K19_SPEC_CONTROL 0x00000048
int amd_cpu_spec_control_getter(const LikwidDevice_t device, uint64_t mask, uint64_t shift, int invert, char** value)
{
    int err = 0;
    uint64_t data = 0x0;
    uint64_t _val = 0x0;
    err = HPMread(device->id.simple.id, MSR_DEV, AMD_K19_SPEC_CONTROL, &data);
    if (err == 0)
    {
        if (!invert)
        {
            _val = (data & mask) >> shift;
        }
        else
        {
            _val = !((data & mask) >> shift);
        }
    }
    return _uint64_to_string(_val, value);
}

int amd_cpu_spec_control_setter(const LikwidDevice_t device, uint64_t mask, uint64_t shift, int invert, const char* value)
{
    int err = 0;
    /*if ((err = request_hw_access(dev)) != 0)
    {
        return err;
    }
    if (dev->scope != DEVICE_SCOPE_device)
    {
        return -ENODEV;
    }*/
    uint64_t data = 0x0;
    uint64_t _val = 0x0;
    err = _string_to_uint64(value, &_val);
    if (err < 0)
    {
        return err;
    }
    err = HPMread(device->id.simple.id, MSR_DEV, AMD_K19_SPEC_CONTROL, &data);
    if (err == 0)
    {
        data &= ~(mask);
        if (invert)
            _val = !_val;
        data |= ((_val << shift) & mask);
        err = HPMwrite(device->id.simple.id, MSR_DEV, AMD_K19_SPEC_CONTROL, data);
    }
    return err;
}

int amd_cpu_spec_ibrs_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_spec_control_getter(device, 0x01, 0,  0, value);
}

int amd_cpu_spec_ibrs_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_spec_control_setter(device, 0x01, 0, 0, value);
}

int amd_cpu_spec_stibp_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_spec_control_getter(device, 0x02, 1,  0, value);
}

int amd_cpu_spec_stibp_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_spec_control_setter(device, 0x02, 1, 0, value);
}

int amd_cpu_spec_ssbd_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_spec_control_getter(device, 0x04, 2,  1, value);
}

int amd_cpu_spec_ssbd_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_spec_control_setter(device, 0x04, 2, 1, value);
}

int amd_cpu_spec_pfsd_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_spec_control_getter(device, 0x80, 7,  1, value);
}

int amd_cpu_spec_pfsd_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_spec_control_setter(device, 0x80, 7, 1, value);
}

#define AMD_K19_L1D_FLUSH_REGISTER 0x0000010B
int amd_cpu_flush_l1(const LikwidDevice_t device, const char* value)
{
    return HPMwrite(device->id.simple.id, MSR_DEV, AMD_K19_L1D_FLUSH_REGISTER, 0x01);
}

#define AMD_K19_HWCONFIG_REGISTER 0xC0010015
int amd_cpu_hwconfig_getter(const LikwidDevice_t device, uint64_t mask, uint64_t shift, int invert, char** value)
{
    int err = 0;
    /*if ((err = request_hw_access(dev)) != 0)
    {
        return err;
    }
    if (dev->scope != DEVICE_SCOPE_device)
    {
        return -ENODEV;
    }*/
    uint64_t data = 0x0;
    uint64_t _val = 0x0;
    err = HPMread(device->id.simple.id, MSR_DEV, AMD_K19_HWCONFIG_REGISTER, &data);
    if (err == 0)
    {
        if (!invert)
        {
            _val = (data & mask) >> shift;
        }
        else
        {
            _val = !((data & mask) >> shift);
        }
    }
    return _uint64_to_string(_val, value);
}

int amd_cpu_hwconfig_setter(const LikwidDevice_t device, uint64_t mask, uint64_t shift, int invert, const char* value)
{
    int err = 0;
    /*if ((err = request_hw_access(dev)) != 0)
    {
        return err;
    }
    if (dev->scope != DEVICE_SCOPE_device)
    {
        return -ENODEV;
    }*/
    uint64_t data = 0x0;
    uint64_t _val = 0x0;
    err = _string_to_uint64(value, &_val);
    if (err < 0)
    {
        return err;
    }
    err = HPMread(device->id.simple.id, MSR_DEV, AMD_K19_HWCONFIG_REGISTER, &data);
    if (err == 0)
    {
        data &= ~(mask);
        if (invert)
            _val = !_val;
        data |= ((_val << shift) & mask);
        err = HPMwrite(device->id.simple.id, MSR_DEV, AMD_K19_HWCONFIG_REGISTER, data);
    }
    return err;
}

int amd_cpu_hwconfig_cpddis_getter(const LikwidDevice_t device, char** value)
{
    return amd_cpu_hwconfig_getter(device, (0x1<<25), 25,  1, value);
}

int amd_cpu_hwconfig_cpddis_setter(const LikwidDevice_t device, const char* value)
{
    return amd_cpu_hwconfig_setter(device, (0x1<<25), 25, 1, value);
}
