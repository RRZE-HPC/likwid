#include <sysFeatures_amd_pstate.h>

#include <bitUtil.h>
#include <cpuid.h>
#include <error.h>
#include <registers.h>
#include <sysFeatures_common.h>
#include <topology.h>
#include <types.h>

static int amd_pstate_max_value_getter(LikwidDevice_t dev, char **value)
{
    return likwid_sysft_readmsr_field_to_string(dev, MSR_AMD_PSTATE_CUR_LIMIT, 4, 3, value);
}

static int amd_pstate_cur_limit_getter(LikwidDevice_t dev, char **value)
{
    return likwid_sysft_readmsr_field_to_string(dev, MSR_AMD_PSTATE_CUR_LIMIT, 0, 3, value);
}

static int amd_pstate_cmd_getter(LikwidDevice_t dev, char **value)
{
    return likwid_sysft_readmsr_field_to_string(dev, MSR_AMD_PSTATE_CTL, 0, 3, value);
}

static int amd_pstate_cmd_setter(LikwidDevice_t dev, const char *value)
{
    return likwid_sysft_writemsr_field_from_string(dev, MSR_AMD_PSTATE_CTL, 0, 3, value);
}

static int amd_pstate_cur_getter(LikwidDevice_t dev, char **value)
{
    return likwid_sysft_readmsr_field_to_string(dev, MSR_AMD_PSTATE_STAT, 0, 3, value);
}

static int amd_pstate_px_en_getter(LikwidDevice_t dev, uint8_t pstateId, char **value)
{
    return likwid_sysft_readmsr_field_to_string(dev, MSR_AMD_PSTATE_DEFx(pstateId), 63, 1, value);
}

static int amd_pstate_px_en_setter(LikwidDevice_t dev, uint8_t pstateId, const char *value)
{
    return likwid_sysft_writemsr_field_from_string(dev, MSR_AMD_PSTATE_DEFx(pstateId), 63, 1, value);
}

static int amd_pstate_cpu_vid_getter(LikwidDevice_t dev, uint8_t pstateId, char **value)
{
    uint64_t msrData;
    int err = likwid_sysft_readmsr(dev, MSR_AMD_PSTATE_DEFx(pstateId), &msrData);
    if (err < 0)
        return err;
    const uint64_t cpuVid = field64(msrData, 32, 1) | field64(msrData, 14, 8);
    return likwid_sysft_uint64_to_string(cpuVid, value);
}

static int amd_pstate_cpu_vid_setter(LikwidDevice_t dev, uint8_t pstateId, const char *value)
{
    uint64_t cpuVid;
    int err = likwid_sysft_string_to_uint64(value, &cpuVid);
    if (err < 0)
        return err;
    uint64_t msrData;
    err = likwid_sysft_readmsr(dev, MSR_AMD_PSTATE_DEFx(pstateId), &msrData);
    if (err < 0)
        return err;
    field64set(&msrData, 14, 8, cpuVid >> 0);
    field64set(&msrData, 32, 1, cpuVid >> 8);

    return likwid_sysft_writemsr_field(dev, MSR_AMD_PSTATE_DEFx(pstateId), 0, 64, msrData);
}


static int amd_pstate_idd_div_getter(LikwidDevice_t dev, uint8_t pstateId, char **value)
{
    return likwid_sysft_readmsr_field_to_string(dev, MSR_AMD_PSTATE_DEFx(pstateId), 30, 2, value);
}

static int amd_pstate_idd_div_setter(LikwidDevice_t dev, uint8_t pstateId, const char *value)
{
    return likwid_sysft_writemsr_field_from_string(dev, MSR_AMD_PSTATE_DEFx(pstateId), 30, 2, value);
}

static int amd_pstate_idd_val_getter(LikwidDevice_t dev, uint8_t pstateId, char **value)
{
    return likwid_sysft_readmsr_field_to_string(dev, MSR_AMD_PSTATE_DEFx(pstateId), 22, 8, value);
}

static int amd_pstate_idd_val_setter(LikwidDevice_t dev, uint8_t pstateId, const char *value)
{
    return likwid_sysft_writemsr_field_from_string(dev, MSR_AMD_PSTATE_DEFx(pstateId), 22, 8, value);
}

static int amd_pstate_idd_getter(LikwidDevice_t dev, uint8_t pstateId, char **value)
{
    uint64_t msrData;
    int err = likwid_sysft_readmsr(dev, MSR_AMD_PSTATE_DEFx(pstateId), &msrData);
    if (err < 0)
        return err;

    const uint64_t num = field64(msrData, 22, 8);
    const uint64_t denom = field64(msrData, 30, 2);

    if (denom == 0)
        return -EINVAL;

    const double current = (double)num / (double)denom;
    return likwid_sysft_double_to_string(current, value);
}

static int amd_pstate_cpu_dfs_id_getter_raw(LikwidDevice_t dev, uint8_t pstateId, char **value)
{
    return likwid_sysft_readmsr_field_to_string(dev, MSR_AMD_PSTATE_DEFx(pstateId), 8, 6, value);
}

static int amd_pstate_cpu_dfs_id_setter_raw(LikwidDevice_t dev, uint8_t pstateId, const char *value)
{
    return likwid_sysft_writemsr_field_from_string(dev, MSR_AMD_PSTATE_DEFx(pstateId), 8, 6, value);
}

static int dfs_get(uint8_t id, double *dfs) {
    if (id == 0x0)
        *dfs = 0.0;
    else if (id >= 0x1 && id <= 0x7)
        return -EINVAL;
    else if (id == 0x8)
        *dfs = 1.0;
    else if (id == 0x9)
        *dfs = 1.125;
    else if (id >= 0xa && id <= 0x1a)
        *dfs = (double)id / 8.0;
    else if (id >= 0x1c && id <= 0x2c && (id % 2) == 0)
        *dfs = (double)id / 8.0;
    else
        return -EINVAL;
    return 0;
}

static int amd_pstate_cpu_dfs_id_getter(LikwidDevice_t dev, uint8_t pstateId, char **value)
{
    uint64_t msrData;
    int err = likwid_sysft_readmsr(dev, MSR_AMD_PSTATE_DEFx(pstateId), &msrData);
    if (err < 0)
        return err;

    double dfs;
    err = dfs_get(field64(msrData, 8, 6), &dfs);
    if (err < 0)
        return err;

    return likwid_sysft_double_to_string(dfs, value);
}

static int amd_pstate_cpu_fid_getter_raw(LikwidDevice_t dev, uint8_t pstateId, char **value)
{
    return likwid_sysft_readmsr_field_to_string(dev, MSR_AMD_PSTATE_DEFx(pstateId), 0, 8, value);
}

static int amd_pstate_cpu_fid_setter_raw(LikwidDevice_t dev, uint8_t pstateId, const char *value)
{
    return likwid_sysft_writemsr_field_from_string(dev, MSR_AMD_PSTATE_DEFx(pstateId), 0, 8, value);
}

static int amd_pstate_cpu_fid_getter(LikwidDevice_t dev, uint8_t pstateId, char **value)
{
    uint64_t msrData;
    int err = likwid_sysft_readmsr(dev, MSR_AMD_PSTATE_DEFx(pstateId), &msrData);
    if (err < 0)
        return err;

    const double fid = field64(msrData, 0, 8);

    return likwid_sysft_double_to_string((double)fid * 25.0, value);
}

static int amd_pstate_cpu_clk_getter(LikwidDevice_t dev, uint8_t pstateId, char **value)
{
    uint64_t msrData;
    int err = likwid_sysft_readmsr(dev, MSR_AMD_PSTATE_DEFx(pstateId), &msrData);
    if (err < 0)
        return err;

    const double fid = field64(msrData, 0, 8);

    double dfs;
    err = dfs_get(field64(msrData, 8, 6), &dfs);
    if (err < 0)
        return err;

    if (dfs <= 0.0)
        return -EINVAL;

    return likwid_sysft_double_to_string(fid * 25.0 / dfs, value);
}

static bool has_leaf_80000007(void)
{
    /* get maximum leaf for extended CPUID */
    uint32_t eax = 0x80000000, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);

    /* If 0x80000007 isn't available, we don't have P-State */
    return eax >= 0x7;
}

static bool leaf_80000007_has_pstate(void)
{
    uint32_t eax = 0x80000007, ebx, ecx = 0, edx;
    CPUID(eax, ebx, ecx, edx);

    return field32(edx, 7, 1);
}

static int amd_pstate_test(void)
{
    CpuInfo_t info = get_cpuInfo();
    if (!info)
        return 0;

    /* Must have an AMD Zen CPU to support pstate.
     * Not sure if it's actually necessary to check this if we check for the CPUID leaf. */
    switch (info->family) {
    case ZEN_FAMILY:
    case ZEN3_FAMILY:
    case ZEN5_FAMILY:
        break;
    default:
        return 0;
    }

    if (!has_leaf_80000007())
        return 0;


    return leaf_80000007_has_pstate();
}

#define MAKE_PSTATE_FUNCS(id) \
    static int amd_pstate##id##_px_en_getter(LikwidDevice_t dev, char **value) \
    { \
        return amd_pstate_px_en_getter(dev, id, value); \
    } \
    static int amd_pstate##id##_px_en_setter(LikwidDevice_t dev, const char *value) \
    { \
        return amd_pstate_px_en_setter(dev, id, value); \
    } \
    static int amd_pstate##id##_cpu_vid_getter(LikwidDevice_t dev, char **value) \
    { \
        return amd_pstate_cpu_vid_getter(dev, id, value); \
    } \
    static int amd_pstate##id##_cpu_vid_setter(LikwidDevice_t dev, const char *value) \
    { \
        return amd_pstate_cpu_vid_setter(dev, id, value); \
    } \
    static int amd_pstate##id##_idd_div_getter(LikwidDevice_t dev, char **value) \
    { \
        return amd_pstate_idd_div_getter(dev, id, value); \
    } \
    static int amd_pstate##id##_idd_div_setter(LikwidDevice_t dev, const char *value) \
    { \
        return amd_pstate_idd_div_setter(dev, id, value); \
    } \
    static int amd_pstate##id##_idd_val_getter(LikwidDevice_t dev, char **value) \
    { \
        return amd_pstate_idd_val_getter(dev, id, value); \
    } \
    static int amd_pstate##id##_idd_val_setter(LikwidDevice_t dev, const char *value) \
    { \
        return amd_pstate_idd_val_setter(dev, id, value); \
    } \
    static int amd_pstate##id##_idd_getter(LikwidDevice_t dev, char **value) \
    { \
        return amd_pstate_idd_getter(dev, id, value); \
    } \
    static int amd_pstate##id##_cpu_dfs_id_getter_raw(LikwidDevice_t dev, char **value) \
    { \
        return amd_pstate_cpu_dfs_id_getter_raw(dev, id, value); \
    } \
    static int amd_pstate##id##_cpu_dfs_id_setter_raw(LikwidDevice_t dev, const char *value) \
    { \
        return amd_pstate_cpu_dfs_id_setter_raw(dev, id, value); \
    } \
    static int amd_pstate##id##_cpu_dfs_id_getter(LikwidDevice_t dev, char **value) \
    { \
        return amd_pstate_cpu_dfs_id_getter(dev, id, value); \
    } \
    static int amd_pstate##id##_cpu_fid_getter_raw(LikwidDevice_t dev, char **value) \
    { \
        return amd_pstate_cpu_fid_getter_raw(dev, id, value); \
    } \
    static int amd_pstate##id##_cpu_fid_setter_raw(LikwidDevice_t dev, const char *value) \
    { \
        return amd_pstate_cpu_fid_setter_raw(dev, id, value); \
    } \
    static int amd_pstate##id##_cpu_fid_getter(LikwidDevice_t dev, char **value) \
    { \
        return amd_pstate_cpu_fid_getter(dev, id, value); \
    } \
    static int amd_pstate##id##_cpu_clk_getter(LikwidDevice_t dev, char **value) \
    { \
        return amd_pstate_cpu_clk_getter(dev, id, value); \
    } \

MAKE_PSTATE_FUNCS(0)
MAKE_PSTATE_FUNCS(1)
MAKE_PSTATE_FUNCS(2)
MAKE_PSTATE_FUNCS(3)
MAKE_PSTATE_FUNCS(4)
MAKE_PSTATE_FUNCS(5)
MAKE_PSTATE_FUNCS(6)
MAKE_PSTATE_FUNCS(7)

#define MAKE_PSTATE_FEATURE(id) \
    {"p" #id "_en", "amd_pstate", "P" #id " enabled", amd_pstate##id##_px_en_getter, amd_pstate##id##_px_en_setter, DEVICE_TYPE_CORE, NULL, NULL},          \
    {"p" #id "_cpu_vid", "amd_pstate", "P" #id " Voltage ID of P-state (?)", amd_pstate##id##_cpu_vid_getter, amd_pstate##id##_cpu_vid_setter, DEVICE_TYPE_CORE, NULL, NULL},     \
    {"p" #id "_idd_div_raw", "amd_pstate", "P" #id " Maximum current draw (denom)", amd_pstate##id##_idd_div_getter, amd_pstate##id##_idd_div_setter, DEVICE_TYPE_CORE, NULL, NULL},     \
    {"p" #id "_idd_val_raw", "amd_pstate", "P" #id " Maximum current draw (num)", amd_pstate##id##_idd_val_getter, amd_pstate##id##_idd_val_setter, DEVICE_TYPE_CORE, NULL, "A"},     \
    {"p" #id "_idd", "amd_pstate", "P" #id " Maximum current draw (effective)", amd_pstate##id##_idd_getter, NULL, DEVICE_TYPE_CORE, NULL, "A"},     \
    {"p" #id "_cpu_dfs_id_raw", "amd_pstate", "P" #id " clock divisor ID (raw)", amd_pstate##id##_cpu_dfs_id_getter_raw, amd_pstate##id##_cpu_dfs_id_setter_raw, DEVICE_TYPE_CORE, NULL, NULL},  \
    {"p" #id "_cpu_dfs_id", "amd_pstate", "P" #id " clock divisor (effective)", amd_pstate##id##_cpu_dfs_id_getter, NULL, DEVICE_TYPE_CORE, NULL, NULL},  \
    {"p" #id "_cpu_fid_raw", "amd_pstate", "P" #id " clock numerator ID (raw)", amd_pstate##id##_cpu_fid_getter_raw, amd_pstate##id##_cpu_fid_setter_raw, DEVICE_TYPE_CORE, NULL, NULL}, \
    {"p" #id "_cpu_fid", "amd_pstate", "P" #id " clock numerator (effective)", amd_pstate##id##_cpu_fid_getter, NULL, DEVICE_TYPE_CORE, NULL, NULL}, \
    {"p" #id "_cpu_clk", "amd_pstate", "P" #id " clock frequency (final)", amd_pstate##id##_cpu_clk_getter, NULL, DEVICE_TYPE_CORE, NULL, NULL},

static _SysFeature amd_pstate_features[] = {
    {"max_value", "amd_pstate", "Lowest performance non-boosted P-state", amd_pstate_max_value_getter, NULL, DEVICE_TYPE_CORE, NULL, NULL},
    {"cur_limit", "amd_pstate", "Highest performance P-state", amd_pstate_cur_limit_getter, NULL, DEVICE_TYPE_CORE, NULL, NULL},
    {"cmd", "amd_pstate", "Last commanded non-boosted P-state", amd_pstate_cmd_getter, amd_pstate_cmd_setter, DEVICE_TYPE_CORE, NULL, NULL},
    {"cur", "amd_pstate", "Current non-boosted P-state", amd_pstate_cur_getter, NULL, DEVICE_TYPE_CORE, NULL, NULL},
    MAKE_PSTATE_FEATURE(0)
    MAKE_PSTATE_FEATURE(1)
    MAKE_PSTATE_FEATURE(2)
    MAKE_PSTATE_FEATURE(3)
    MAKE_PSTATE_FEATURE(4)
    MAKE_PSTATE_FEATURE(5)
    MAKE_PSTATE_FEATURE(6)
    MAKE_PSTATE_FEATURE(7)
};

static const _SysFeatureList amd_pstate_feature_list = {
    .num_features = ARRAY_COUNT(amd_pstate_features),
    .tester = amd_pstate_test,
    .features = amd_pstate_features,
};

int likwid_sysft_init_amd_pstate(_SysFeatureList* out)
{
    int err = init_configuration();
    if (err < 0)
    {
        errno = -err;
        ERROR_PRINT("Failed to initialize configuration");
        return err;
    }

    Configuration_t config = get_configuration();
    if (config->daemonMode == ACCESSMODE_PERF)
    {
        DEBUG_PRINT(DEBUGLEV_INFO, "No AMD P-state support with accessmode=perf_event");
        return 0;
    }

    if (amd_pstate_test()) {
        DEBUG_PRINT(DEBUGLEV_INFO, "Register AMD P-state");
        err = likwid_sysft_register_features(out, &amd_pstate_feature_list);
        if (err < 0)
            return err;
    } else {
        DEBUG_PRINT(DEBUGLEV_INFO, "AMD P-state not supported");
    }

    return 0;
}
