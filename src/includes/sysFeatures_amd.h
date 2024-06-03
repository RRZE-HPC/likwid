#ifndef HWFEATURES_X86_AMD_H
#define HWFEATURES_X86_AMD_H

#include <registers.h>
#include <cpuid.h>
#include <topology.h>
#include <sysFeatures_types.h>
#include <likwid.h>
#include <access.h>
#include <sysFeatures_common.h>

#include <sysFeatures_amd_rapl.h>


#define MSR_AMD19_PREFETCH_CONTROL 0xC0000108

int amd_cpu_prefetch_control_getter(LikwidDevice_t device, uint64_t mask, uint64_t shift, int invert, char** value)
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

int amd_cpu_prefetch_control_setter(LikwidDevice_t device, uint64_t mask, uint64_t shift, int invert, char* value)
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


int amd_cpu_l1_stream_getter(LikwidDevice_t device, char** value)
{
    return amd_cpu_prefetch_control_getter(device, 0x1, 0, 1, value);
}

int amd_cpu_l1_stream_setter(LikwidDevice_t device, char* value)
{
    return amd_cpu_prefetch_control_setter(device, 0x1, 0, 1, value);
}

int amd_cpu_l1_stride_getter(LikwidDevice_t device, char** value)
{
    return amd_cpu_prefetch_control_getter(device, 0x2, 1, 1, value);
}

int amd_cpu_l1_stride_setter(LikwidDevice_t device, char* value)
{
    return amd_cpu_prefetch_control_setter(device, 0x2, 1, 1, value);
}

int amd_cpu_l1_region_getter(LikwidDevice_t device, char** value)
{
    return amd_cpu_prefetch_control_getter(device, 0x4, 2, 1, value);
}

int amd_cpu_l1_region_setter(LikwidDevice_t device, char* value)
{
    return amd_cpu_prefetch_control_setter(device, 0x4, 2, 1, value);
}

int amd_cpu_l2_stream_getter(LikwidDevice_t device, char** value)
{
    return amd_cpu_prefetch_control_getter(device, 0x8, 3, 1, value);
}

int amd_cpu_l2_stream_setter(LikwidDevice_t device, char* value)
{
    return amd_cpu_prefetch_control_setter(device, 0x8, 3, 1, value);
}

int amd_cpu_up_down_getter(LikwidDevice_t device, char** value)
{
    return amd_cpu_prefetch_control_getter(device, 0x20, 5, 1, value);
}

int amd_cpu_up_down_setter(LikwidDevice_t device, char* value)
{
    return amd_cpu_prefetch_control_setter(device, 0x20, 5, 1, value);
}

#define MAX_AMD_K19_CPU_PREFETCH_FEATURES 5
static _SysFeature amd_k19_cpu_prefetch_features[] = {
    {"l1_stream", "prefetch", "Stream prefetcher that uses history of memory access patterns to fetch additional sequential lines into L1 cache", amd_cpu_l1_stream_getter, amd_cpu_l1_stream_setter, DEVICE_TYPE_HWTHREAD},
    {"l1_stride", "prefetch", "Stride prefetcher that uses memory access history of individual instructions to fetch additional lines into L1 cache when each access is a constant distance from the previous", amd_cpu_l1_stride_getter, amd_cpu_l1_stride_setter, DEVICE_TYPE_HWTHREAD},
    {"l1_region", "prefetch", "Prefetcher that uses memory access history to fetch additional lines into L1 cache when the data access for a given instruction tends to be followed by a consistent pattern of other accesses within a localized region", amd_cpu_l1_region_getter, amd_cpu_l1_region_setter, DEVICE_TYPE_HWTHREAD},
    {"L2_stream", "prefetch", "Stream prefetcher that uses history of memory access patterns to fetch additional sequential lines into L2 cache", amd_cpu_l2_stream_getter, amd_cpu_l2_stream_setter, DEVICE_TYPE_HWTHREAD},
    {"up_down", "prefetch", "Prefetcher that uses memory access history to determine whether to fetch the next or previous line into L2 cache for all memory accesses", amd_cpu_up_down_getter, amd_cpu_up_down_setter, DEVICE_TYPE_HWTHREAD},
};

static _SysFeatureList amd_k19_cpu_prefetch_feature_list = {
    .num_features = MAX_AMD_K19_CPU_PREFETCH_FEATURES,
    .features = amd_k19_cpu_prefetch_features,
};

#define AMD_K19_SPEC_CONTROL 0x00000048
int amd_cpu_spec_control_getter(LikwidDevice_t device, uint64_t mask, uint64_t shift, int invert, char** value)
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
int amd_cpu_spec_control_setter(LikwidDevice_t device, uint64_t mask, uint64_t shift, int invert, char* value)
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


int amd_cpu_spec_ibrs_getter(LikwidDevice_t device, char** value)
{
    return amd_cpu_spec_control_getter(device, 0x01, 0,  0, value);
}

int amd_cpu_spec_ibrs_setter(LikwidDevice_t device, char* value)
{
    return amd_cpu_spec_control_setter(device, 0x01, 0, 0, value);
}

int amd_cpu_spec_stibp_getter(LikwidDevice_t device, char** value)
{
    return amd_cpu_spec_control_getter(device, 0x02, 1,  0, value);
}

int amd_cpu_spec_stibp_setter(LikwidDevice_t device, char* value)
{
    return amd_cpu_spec_control_setter(device, 0x02, 1, 0, value);
}

int amd_cpu_spec_ssbd_getter(LikwidDevice_t device, char** value)
{
    return amd_cpu_spec_control_getter(device, 0x04, 2,  1, value);
}

int amd_cpu_spec_ssbd_setter(LikwidDevice_t device, char* value)
{
    return amd_cpu_spec_control_setter(device, 0x04, 2, 1, value);
}

int amd_cpu_spec_pfsd_getter(LikwidDevice_t device, char** value)
{
    return amd_cpu_spec_control_getter(device, 0x80, 7,  1, value);
}

int amd_cpu_spec_pfsd_setter(LikwidDevice_t device, char* value)
{
    return amd_cpu_spec_control_setter(device, 0x80, 7, 1, value);
}

#define MAX_AMD_K19_CPU_SPECULATION_FEATURES 4
static _SysFeature amd_k19_cpu_speculation_features[] = {
    {"ibrs", "spec_ctrl", "Indirect branch restriction speculation", amd_cpu_spec_ibrs_getter, amd_cpu_spec_ibrs_setter, DEVICE_TYPE_HWTHREAD},
    {"stibp", "spec_ctrl", "Single thread indirect branch predictor", amd_cpu_spec_stibp_getter, amd_cpu_spec_stibp_setter, DEVICE_TYPE_HWTHREAD},
    {"ssbd", "spec_ctrl", "Speculative Store Bypass", amd_cpu_spec_ssbd_getter, amd_cpu_spec_ssbd_setter, DEVICE_TYPE_HWTHREAD},
    {"psfd", "spec_ctrl", "Predictive Store Forwarding", amd_cpu_spec_pfsd_getter, amd_cpu_spec_pfsd_setter, DEVICE_TYPE_HWTHREAD},
};
static _SysFeatureList amd_k19_cpu_speculation_feature_list = {
    .num_features = MAX_AMD_K19_CPU_SPECULATION_FEATURES,
    .features = amd_k19_cpu_speculation_features,
};


#define AMD_K19_L1D_FLUSH_REGISTER 0x0000010B
int amd_cpu_flush_l1(LikwidDevice_t device, char* value)
{
    return HPMwrite(device->id.simple.id, MSR_DEV, AMD_K19_L1D_FLUSH_REGISTER, 0x01);
}

#define MAX_AMD_K19_CPU_L1DFLUSH_FEATURES 1
static _SysFeature amd_k19_cpu_l1dflush_features[] = {
    {"l1dflush", "cache", "Performs a write-back and invalidate of the L1 data cache", NULL, amd_cpu_flush_l1, DEVICE_TYPE_HWTHREAD},
};
static _SysFeatureList amd_k19_cpu_l1dflush_feature_list = {
    .num_features = MAX_AMD_K19_CPU_L1DFLUSH_FEATURES,
    .features = amd_k19_cpu_l1dflush_features,
};

#define AMD_K19_HWCONFIG_REGISTER 0xC0010015
int amd_cpu_hwconfig_getter(LikwidDevice_t device, uint64_t mask, uint64_t shift, int invert, char** value)
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
int amd_cpu_hwconfig_setter(LikwidDevice_t device, uint64_t mask, uint64_t shift, int invert, char* value)
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

int amd_cpu_hwconfig_cpddis_getter(LikwidDevice_t device, char** value)
{
    return amd_cpu_hwconfig_getter(device, (0x1<<25), 25,  1, value);
}

int amd_cpu_hwconfig_cpddis_setter(LikwidDevice_t device, char* value)
{
    return amd_cpu_hwconfig_setter(device, (0x1<<25), 25, 1, value);
}


#define MAX_AMD_K19_CPU_HWCONFIG_FEATURES 1
static _SysFeature amd_k19_cpu_hwconfig_features[] = {
    {"TurboMode", "cpufreq", "Specifies whether core performance boost is requested to be enabled or disabled", amd_cpu_hwconfig_cpddis_getter, amd_cpu_hwconfig_cpddis_setter, DEVICE_TYPE_HWTHREAD},
};
static _SysFeatureList amd_k19_cpu_hwconfig_feature_list = {
    .num_features = MAX_AMD_K19_CPU_HWCONFIG_FEATURES,
    .features = amd_k19_cpu_hwconfig_features,
};

static _SysFeatureList* amd_k19_cpu_feature_inputs[] = {
    &amd_k19_cpu_prefetch_feature_list,
    &amd_k19_cpu_speculation_feature_list,
    &amd_k19_cpu_l1dflush_feature_list,
    &amd_k19_cpu_hwconfig_feature_list,
    &amd_rapl_core_feature_list,
    &amd_rapl_pkg_feature_list,
    &amd_rapl_l3_feature_list,
    NULL,
};

static _HWArchFeatures amd_arch_features[] = {
    {ZEN3_FAMILY, ZEN4_RYZEN, amd_k19_cpu_feature_inputs},
    {ZEN3_FAMILY, ZEN4_RYZEN_PRO, amd_k19_cpu_feature_inputs},
    {ZEN3_FAMILY, ZEN4_EPYC, amd_k19_cpu_feature_inputs},
    {-1, -1, NULL},
};



int sysFeatures_init_x86_amd(_SysFeatureList* out)
{
    return sysFeatures_init_generic(amd_arch_features, out);
}


#endif
