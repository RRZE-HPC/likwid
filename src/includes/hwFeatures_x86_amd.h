#ifndef HWFEATURES_X86_AMD_H
#define HWFEATURES_X86_AMD_H

#include <hwFeatures_types.h>
#include <errno.h>


#define MSR_AMD19_PREFETCH_CONTROL 0xC0000108

int amd_cpu_prefetch_control_getter(int hwthread, uint64_t mask, uint64_t shift, int invert, uint64_t* value)
{
    int err = 0;
    /*if ((err = request_hw_access(dev)) != 0)
    {
        return err;
    }
    if (dev->scope != DEVICE_SCOPE_HWTHREAD)
    {
        return -ENODEV;
    }*/
    uint64_t data = 0x0;
    err = HPMread(hwthread, MSR_DEV, MSR_AMD19_PREFETCH_CONTROL, &data);
    if (err == 0)
    {
        if (!invert)
    {
            //*value = (data >> shift) & mask;
        *value = (data & mask) >> shift;
    }
        else
    {
            //*value = !((data >> shift) & mask);
        *value = !((data & mask) >> shift);
    }
    }
    return err;
}

int amd_cpu_prefetch_control_setter(int hwthread, uint64_t mask, uint64_t shift, int invert, uint64_t value)
{
    int err = 0;
    /*if ((err = request_hw_access(dev)) != 0)
    {
        return err;
    }
    if (dev->scope != DEVICE_SCOPE_HWTHREAD)
    {
        return -ENODEV;
    }*/
    uint64_t data = 0x0;
    err = HPMread(hwthread, MSR_DEV, MSR_AMD19_PREFETCH_CONTROL, &data);
    if (err == 0)
    {
        data &= ~(mask);
        if (invert)
            value = !value;
        data |= ((value << shift) & mask);
        err = HPMwrite(hwthread, MSR_DEV, MSR_AMD19_PREFETCH_CONTROL, data);
    }
    return err;
}


int amd_cpu_l1_stream_getter(int hwthread, uint64_t* value)
{
    return amd_cpu_prefetch_control_getter(hwthread, 0x1, 0, 1, value);
}

int amd_cpu_l1_stream_setter(int hwthread, uint64_t value)
{
    return amd_cpu_prefetch_control_setter(hwthread, 0x1, 0, 1, value);
}

int amd_cpu_l1_stride_getter(int hwthread, uint64_t* value)
{
    return amd_cpu_prefetch_control_getter(hwthread, 0x2, 1, 1, value);
}

int amd_cpu_l1_stride_setter(int hwthread, uint64_t value)
{
    return amd_cpu_prefetch_control_setter(hwthread, 0x2, 1, 1, value);
}

int amd_cpu_l1_region_getter(int hwthread, uint64_t* value)
{
    return amd_cpu_prefetch_control_getter(hwthread, 0x4, 2, 1, value);
}

int amd_cpu_l1_region_setter(int hwthread, uint64_t value)
{
    return amd_cpu_prefetch_control_setter(hwthread, 0x4, 2, 1, value);
}

int amd_cpu_l2_stream_getter(int hwthread, uint64_t* value)
{
    return amd_cpu_prefetch_control_getter(hwthread, 0x8, 3, 1, value);
}

int amd_cpu_l2_stream_setter(int hwthread, uint64_t value)
{
    return amd_cpu_prefetch_control_setter(hwthread, 0x8, 3, 1, value);
}

int amd_cpu_up_down_getter(int hwthread, uint64_t* value)
{
    return amd_cpu_prefetch_control_getter(hwthread, 0x20, 5, 1, value);
}

int amd_cpu_up_down_setter(int hwthread, uint64_t value)
{
    return amd_cpu_prefetch_control_setter(hwthread, 0x20, 5, 1, value);
}

#define MAX_AMD_K19_CPU_PREFETCH_FEATURES 5
static _HWFeature amd_k19_cpu_prefetch_features[] = {
    {"L1Stream", "Stream prefetcher that uses history of memory access patterns to fetch additional sequential lines into L1 cache", amd_cpu_l1_stream_getter, amd_cpu_l1_stream_setter, HWFEATURE_SCOPE_HWTHREAD},
    {"L1Stride", "Stride prefetcher that uses memory access history of individual instructions to fetch additional lines into L1 cache when each access is a constant distance from the previous", amd_cpu_l1_stride_getter, amd_cpu_l1_stride_setter, HWFEATURE_SCOPE_HWTHREAD},
    {"L1Region", "Prefetcher that uses memory access history to fetch additional lines into L1 cache when the data access for a given instruction tends to be followed by a consistent pattern of other accesses within a localized region", amd_cpu_l1_region_getter, amd_cpu_l1_region_setter, HWFEATURE_SCOPE_HWTHREAD},
    {"L2Stream", "Stream prefetcher that uses history of memory access patterns to fetch additional sequential lines into L2 cache", amd_cpu_l2_stream_getter, amd_cpu_l2_stream_setter, HWFEATURE_SCOPE_HWTHREAD},
    {"UpDown", "Prefetcher that uses memory access history to determine whether to fetch the next or previous line into L2 cache for all memory accesses", amd_cpu_up_down_getter, amd_cpu_up_down_setter, HWFEATURE_SCOPE_HWTHREAD},
};

static _HWFeatureList amd_k19_cpu_prefetch_feature_list = {
    .num_features = MAX_AMD_K19_CPU_PREFETCH_FEATURES,
    .features = amd_k19_cpu_prefetch_features,
};

#define AMD_K19_SPEC_CONTROL 0x00000048
int amd_cpu_spec_control_getter(int hwthread, uint64_t mask, uint64_t shift, int invert, uint64_t* value)
{
    int err = 0;
    /*if ((err = request_hw_access(dev)) != 0)
    {
        return err;
    }
    if (dev->scope != DEVICE_SCOPE_HWTHREAD)
    {
        return -ENODEV;
    }*/
    uint64_t data = 0x0;
    err = HPMread(hwthread, MSR_DEV, AMD_K19_SPEC_CONTROL, &data);
    if (err == 0)
    {
        if (!invert)
        {
            //*value = (data >> shift) & mask;
            *value = (data & mask) >> shift;
        }
        else
        {
            //*value = !((data >> shift) & mask);
            *value = !((data & mask) >> shift);
        }
    }
    return err;
}
int amd_cpu_spec_control_setter(int hwthread, uint64_t mask, uint64_t shift, int invert, uint64_t value)
{
    int err = 0;
    /*if ((err = request_hw_access(dev)) != 0)
    {
        return err;
    }
    if (dev->scope != DEVICE_SCOPE_HWTHREAD)
    {
        return -ENODEV;
    }*/
    uint64_t data = 0x0;
    err = HPMread(hwthread, MSR_DEV, AMD_K19_SPEC_CONTROL, &data);
    if (err == 0)
    {
        data &= ~(mask);
        if (invert)
            value = !value;
        data |= ((value << shift) & mask);
        err = HPMwrite(hwthread, MSR_DEV, AMD_K19_SPEC_CONTROL, data);
    }
    return err;
}


int amd_cpu_spec_ibrs_getter(int hwthread, uint64_t* value)
{
    return amd_cpu_spec_control_getter(hwthread, 0x01, 0,  0, value);
}

int amd_cpu_spec_ibrs_setter(int hwthread, uint64_t value)
{
    return amd_cpu_spec_control_setter(hwthread, 0x01, 0, 0, value);
}

int amd_cpu_spec_stibp_getter(int hwthread, uint64_t* value)
{
    return amd_cpu_spec_control_getter(hwthread, 0x02, 1,  0, value);
}

int amd_cpu_spec_stibp_setter(int hwthread, uint64_t value)
{
    return amd_cpu_spec_control_setter(hwthread, 0x02, 1, 0, value);
}

int amd_cpu_spec_ssbd_getter(int hwthread, uint64_t* value)
{
    return amd_cpu_spec_control_getter(hwthread, 0x04, 2,  1, value);
}

int amd_cpu_spec_ssbd_setter(int hwthread, uint64_t value)
{
    return amd_cpu_spec_control_setter(hwthread, 0x04, 2, 1, value);
}

int amd_cpu_spec_pfsd_getter(int hwthread, uint64_t* value)
{
    return amd_cpu_spec_control_getter(hwthread, 0x80, 7,  1, value);
}

int amd_cpu_spec_pfsd_setter(int hwthread, uint64_t value)
{
    return amd_cpu_spec_control_setter(hwthread, 0x80, 7, 1, value);
}

#define MAX_AMD_K19_CPU_SPECULATION_FEATURES 4
static _HWFeature amd_k19_cpu_speculation_features[] = {
    {"IBRS", "Indirect branch restriction speculation", amd_cpu_spec_ibrs_getter, amd_cpu_spec_ibrs_setter, HWFEATURE_SCOPE_HWTHREAD},
    {"STIBP", "Single thread indirect branch predictor", amd_cpu_spec_stibp_getter, amd_cpu_spec_stibp_setter, HWFEATURE_SCOPE_HWTHREAD},
    {"SSBD", "Speculative Store Bypass", amd_cpu_spec_ssbd_getter, amd_cpu_spec_ssbd_setter, HWFEATURE_SCOPE_HWTHREAD},
    {"PSFD", "Predictive Store Forwarding", amd_cpu_spec_pfsd_getter, amd_cpu_spec_pfsd_setter, HWFEATURE_SCOPE_HWTHREAD},
};
static _HWFeatureList amd_k19_cpu_speculation_feature_list = {
    .num_features = MAX_AMD_K19_CPU_SPECULATION_FEATURES,
    .features = amd_k19_cpu_speculation_features,
};


#define AMD_K19_L1D_FLUSH_REGISTER 0x0000010B
int amd_cpu_flush_l1(int hwthread, uint64_t value)
{
    return HPMwrite(hwthread, MSR_DEV, AMD_K19_L1D_FLUSH_REGISTER, 0x01);
}

#define MAX_AMD_K19_CPU_L1DFLUSH_FEATURES 1
static _HWFeature amd_k19_cpu_l1dflush_features[] = {
    {"L1DFlush", "Performs a write-back and invalidate of the L1 data cache", NULL, amd_cpu_flush_l1, HWFEATURE_SCOPE_HWTHREAD},
};
static _HWFeatureList amd_k19_cpu_l1dflush_feature_list = {
    .num_features = MAX_AMD_K19_CPU_L1DFLUSH_FEATURES,
    .features = amd_k19_cpu_l1dflush_features,
};

#define AMD_K19_HWCONFIG_REGISTER 0xC0010015
int amd_cpu_hwconfig_getter(int hwthread, uint64_t mask, uint64_t shift, int invert, uint64_t* value)
{
    int err = 0;
    /*if ((err = request_hw_access(dev)) != 0)
    {
        return err;
    }
    if (dev->scope != DEVICE_SCOPE_HWTHREAD)
    {
        return -ENODEV;
    }*/
    uint64_t data = 0x0;
    err = HPMread(hwthread, MSR_DEV, AMD_K19_HWCONFIG_REGISTER, &data);
    if (err == 0)
    {
        if (!invert)
        {
            //*value = (data >> shift) & mask;
            *value = (data & mask) >> shift;
        }
        else
        {
            //*value = !((data >> shift) & mask);
            *value = !((data & mask) >> shift);
        }
    }
    return err;
}
int amd_cpu_hwconfig_setter(int hwthread, uint64_t mask, uint64_t shift, int invert, uint64_t value)
{
    int err = 0;
    /*if ((err = request_hw_access(dev)) != 0)
    {
        return err;
    }
    if (dev->scope != DEVICE_SCOPE_HWTHREAD)
    {
        return -ENODEV;
    }*/
    uint64_t data = 0x0;
    err = HPMread(hwthread, MSR_DEV, AMD_K19_HWCONFIG_REGISTER, &data);
    if (err == 0)
    {
        data &= ~(mask);
        if (invert)
            value = !value;
        data |= ((value << shift) & mask);
        err = HPMwrite(hwthread, MSR_DEV, AMD_K19_HWCONFIG_REGISTER, data);
    }
    return err;
}

int amd_cpu_hwconfig_cpddis_getter(int hwthread, uint64_t* value)
{
    return amd_cpu_hwconfig_getter(hwthread, (0x1<<25), 25,  1, value);
}

int amd_cpu_hwconfig_cpddis_setter(int hwthread, uint64_t value)
{
    return amd_cpu_hwconfig_setter(hwthread, (0x1<<25), 25, 1, value);
}


#define MAX_AMD_K19_CPU_HWCONFIG_FEATURES 1
static _HWFeature amd_k19_cpu_hwconfig_features[] = {
    {"TurboMode", "Specifies whether core performance boost is requested to be enabled or disabled", amd_cpu_hwconfig_cpddis_getter, amd_cpu_hwconfig_cpddis_setter, HWFEATURE_SCOPE_HWTHREAD},
};
static _HWFeatureList amd_k19_cpu_hwconfig_feature_list = {
    .num_features = MAX_AMD_K19_CPU_HWCONFIG_FEATURES,
    .features = amd_k19_cpu_hwconfig_features,
};

static _HWFeatureList* amd_k19_cpu_feature_inputs[] = {
    &amd_k19_cpu_prefetch_feature_list,
    &amd_k19_cpu_speculation_feature_list,
    &amd_k19_cpu_l1dflush_feature_list,
    &amd_k19_cpu_hwconfig_feature_list,
    NULL,
};

static _HWArchFeatures amd_arch_features[] = {
    {ZEN3_FAMILY, ZEN4_RYZEN, amd_k19_cpu_feature_inputs},
    {ZEN3_FAMILY, ZEN4_RYZEN2, amd_k19_cpu_feature_inputs},
    {ZEN3_FAMILY, ZEN4_EPYC, amd_k19_cpu_feature_inputs},
    {-1, -1, NULL},
};



int hwFeatures_init_x86_amd(int* num_features, _HWFeature **features)
{
    return hwFeatures_init_generic(amd_arch_features, num_features, features);
}


#endif
