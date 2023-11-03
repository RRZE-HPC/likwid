#ifndef HWFEATURES_X86_INTEL_H
#define HWFEATURES_X86_INTEL_H

#define MSR_PREFETCH_ENABLE    0xC0000108
int intel_cpu_prefetch_control_getter(int hwthread, uint64_t mask, uint64_t shift, int invert, uint64_t* value)
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
    err = HPMread(hwthread, MSR_DEV, MSR_PREFETCH_ENABLE, &data);
    if (err == 0)
    {
    if (!invert)
    {
        *value = (data & mask) >> shift;
    }
    else
    {
        *value = !((data & mask) >> shift);
    }
    }
    return err;
}

int intel_cpu_prefetch_control_setter(int hwthread, uint64_t mask, uint64_t shift, int invert, uint64_t value)
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
    err = HPMread(hwthread, MSR_DEV, MSR_PREFETCH_ENABLE, &data);
    if (err == 0)
    {
        data &= ~(mask);
        if (invert)
            data |= (((!value) << shift) & mask);
        else
            data |= ((value << shift) & mask);
        err = HPMwrite(hwthread, MSR_DEV, MSR_PREFETCH_ENABLE, data);
    }
    return err;
}

int intel_cpu_l2_hwpf_getter(int hwthread, uint64_t* value)
{
    return intel_cpu_prefetch_control_getter(hwthread, 0x1, 0, 1, value);
}

int intel_cpu_l2_hwpf_setter(int hwthread, uint64_t value)
{
    return intel_cpu_prefetch_control_setter(hwthread, 0x1, 0, 1, value);
}

int intel_cpu_l2_adj_pf_getter(int hwthread, uint64_t* value)
{
    return intel_cpu_prefetch_control_getter(hwthread, 0x2, 1, 1, value);
}

int intel_cpu_l2_adj_pf_setter(int hwthread, uint64_t value)
{
    return intel_cpu_prefetch_control_setter(hwthread, 0x2, 1, 1, value);
}

int intel_cpu_l1_dcu_getter(int hwthread, uint64_t* value)
{
    return intel_cpu_prefetch_control_getter(hwthread, 0x4, 3, 1, value);
}

int intel_cpu_l1_dcu_setter(int hwthread, uint64_t value)
{
    return intel_cpu_prefetch_control_setter(hwthread, 0x4, 3, 1, value);
}

int intel_cpu_l1_dcu_ip_getter(int hwthread, uint64_t* value)
{
    return intel_cpu_prefetch_control_getter(hwthread, 0x8, 4, 1, value);
}

int intel_cpu_l1_dcu_ip_setter(int hwthread, uint64_t value)
{
    return intel_cpu_prefetch_control_setter(hwthread, 0x8, 4, 1, value);
}

#define MAX_INTEL_CPU_PREFETCH_FEATURES 4
static _HWFeature intel_cpu_prefetch_features[] = {
    {"l2_hwpf", "L2 Hardware Prefetcher", intel_cpu_l2_hwpf_getter, intel_cpu_l2_hwpf_setter, HWFEATURE_SCOPE_HWTHREAD},
    {"l2_adj_pf", "L2 Adjacent Cache Line Prefetcher", intel_cpu_l2_adj_pf_getter, intel_cpu_l2_adj_pf_setter, HWFEATURE_SCOPE_HWTHREAD},
    {"l1_dcu", "DCU Hardware Prefetcher", intel_cpu_l1_dcu_getter, intel_cpu_l1_dcu_setter, HWFEATURE_SCOPE_HWTHREAD},
    {"l1_dcu_ip", "DCU IP Prefetcher", intel_cpu_l1_dcu_ip_getter, intel_cpu_l1_dcu_ip_setter, HWFEATURE_SCOPE_HWTHREAD},
    //{"data_pf", "Data Dependent Prefetcher", DEVICE_SCOPE_HWTHREAD},
};
static _HWFeatureList intel_cpu_prefetch_feature_list = {
    .num_features = MAX_INTEL_CPU_PREFETCH_FEATURES,
    .features = intel_cpu_prefetch_features,
};

int intel_cpu_l2_multipath_pf_getter(int hwthread, uint64_t* value)
{
    return intel_cpu_prefetch_control_getter(hwthread, 0x20, 6, 1, value);
}

int intel_cpu_l2_multipath_pf_setter(int hwthread, uint64_t value)
{
    return intel_cpu_prefetch_control_setter(hwthread, 0x20, 6, 1, value);
}

#define MAX_INTEL_8F_CPU_FEATURES 1
static _HWFeature intel_8f_cpu_features[] = {
    {"l2_multipath_pf", "L2 Adaptive Multipath Probability Prefetcher", intel_cpu_l2_multipath_pf_getter, intel_cpu_l2_multipath_pf_setter, HWFEATURE_SCOPE_HWTHREAD}
};
static _HWFeatureList intel_8f_cpu_feature_list = {
    .num_features = MAX_INTEL_8F_CPU_FEATURES,
    .features = intel_8f_cpu_features,
};

static _HWFeatureList* intel_arch_feature_inputs[] = {
    &intel_cpu_prefetch_feature_list,
    NULL,
};

static _HWFeatureList* intel_8f_arch_feature_inputs[] = {
    &intel_cpu_prefetch_feature_list,
    &intel_8f_cpu_feature_list,
    NULL,
};


static _HWArchFeatures intel_arch_features[] = {
    {P6_FAMILY, SANDYBRIDGE, intel_arch_feature_inputs},
    {P6_FAMILY, SANDYBRIDGE_EP, intel_arch_feature_inputs},
    {P6_FAMILY, IVYBRIDGE, intel_arch_feature_inputs},
    {P6_FAMILY, IVYBRIDGE_EP, intel_arch_feature_inputs},
    {P6_FAMILY, HASWELL, intel_arch_feature_inputs},
    {P6_FAMILY, HASWELL_EP, intel_arch_feature_inputs},
    {P6_FAMILY, HASWELL_M1, intel_arch_feature_inputs},
    {P6_FAMILY, HASWELL_M2, intel_arch_feature_inputs},
    {P6_FAMILY, BROADWELL, intel_arch_feature_inputs},
    {P6_FAMILY, BROADWELL_E, intel_arch_feature_inputs},
    {P6_FAMILY, BROADWELL_D, intel_arch_feature_inputs},
    {P6_FAMILY, BROADWELL_E3, intel_arch_feature_inputs},
    {P6_FAMILY, SKYLAKE1, intel_arch_feature_inputs},
    {P6_FAMILY, SKYLAKE2, intel_arch_feature_inputs},
    {P6_FAMILY, SKYLAKEX, intel_arch_feature_inputs},
    {P6_FAMILY, 0x8F, intel_8f_arch_feature_inputs},
    {P6_FAMILY, KABYLAKE1, intel_arch_feature_inputs},
    {P6_FAMILY, KABYLAKE2, intel_arch_feature_inputs},
    {P6_FAMILY, CANNONLAKE, intel_arch_feature_inputs},
    {P6_FAMILY, ICELAKE1, intel_arch_feature_inputs},
    {P6_FAMILY, ICELAKE2, intel_arch_feature_inputs},
    {P6_FAMILY, ROCKETLAKE, intel_arch_feature_inputs},
    {P6_FAMILY, COMETLAKE1, intel_arch_feature_inputs},
    {P6_FAMILY, COMETLAKE2, intel_arch_feature_inputs},
    {P6_FAMILY, ICELAKEX1, intel_arch_feature_inputs},
    {P6_FAMILY, ICELAKEX2, intel_arch_feature_inputs},
    {P6_FAMILY, SNOWRIDGEX, intel_arch_feature_inputs},
    {P6_FAMILY, TIGERLAKE1, intel_arch_feature_inputs},
    {P6_FAMILY, TIGERLAKE2, intel_arch_feature_inputs},
    {-1, -1, NULL},
};



int hwFeatures_init_x86_intel(int* num_features, _HWFeature **features)
{
    return hwFeatures_init_generic(intel_arch_features, num_features, features);
}

#endif
