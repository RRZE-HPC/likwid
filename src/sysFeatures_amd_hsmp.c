#include <sysFeatures_amd_hsmp.h>

#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <stdio.h>

#include <amd_hsmp.h>
#include <cpuid.h>
#include <sysFeatures_common.h>
#include <sysFeatures_common_rapl.h>
#include <types.h>
#include <topology.h>

/* Useful links:
 * https://www.kernel.org/doc/html//latest/arch/x86/amd_hsmp.html
 * https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/programmer-references/55898_B1_pub_0_50.zip
 * (see pub2 in zip file above) */

static const _SysFeatureList amd_hsmp_featuer_list;
static RaplDomainInfo rapl_domain_info;

int likwid_sysft_init_amd_hsmp(_SysFeatureList *out)
{
    int err = likwid_sysft_register_features(out, &amd_hsmp_featuer_list);
    /* If HSMP is not available that is okay. Only raise an error if there was
     * another type of error during HSMP communication. */
    if (err == -ENOTSUP)
        return 0;
    return err;
}

static int hsmp_raw(uint32_t socket, uint32_t msg_id, const uint32_t *args, uint16_t argCount, uint32_t *result, uint16_t resultCount)
{
    /* Requires RW permissions! */
    const int fd = open("/dev/hsmp", O_RDWR);
    if (fd < 0)
        return -errno;

    assert(argCount <= HSMP_MAX_MSG_LEN);
    assert(resultCount <= HSMP_MAX_MSG_LEN);

    /* HSMP_TEST should increment the argument by one. */
    struct hsmp_message msg = {
        .msg_id = msg_id,
        .num_args = argCount,
        .response_sz = resultCount,
        .sock_ind = socket,
    };

    for (uint32_t i = 0; i < argCount; i++)
        msg.args[i] = args[i];

    int err = ioctl(fd, HSMP_IOCTL_CMD, &msg);
    if (err < 0)
    {
        close(fd);
        return -errno;
    }

    for (uint32_t i = 0; i < resultCount; i++)
        result[i] = msg.args[i];

    close(fd);
    return 0;
}

static int hsmp_arg0_res1_as_u32(LikwidDevice_t dev, uint32_t msg_id, char **value)
{
    uint32_t result;
    int err = hsmp_raw(dev->id.simple.id, msg_id, NULL, 0, &result, 1);
    if (err < 0)
        return -err;
    return likwid_sysft_uint64_to_string(result, value);
}

static int hsmp_arg1_res0_from_u32(LikwidDevice_t dev, int32_t msg_id, const char *value)
{
    uint64_t arg0;
    int err = likwid_sysft_string_to_uint64(value, &arg0);
    if (err < 0)
        return err;
    uint32_t arg0_32 = (uint32_t)arg0;
    return hsmp_raw(dev->id.simple.id, msg_id, &arg0_32, 1, NULL, 0);
}

static int hsmp_arg0_res1_as_double(LikwidDevice_t dev, uint32_t msg_id, double scale, char **value)
{
    uint32_t result;
    int err = hsmp_raw(dev->id.simple.id, msg_id, NULL, 0, &result, 1);
    if (err < 0)
        return -err;

    return likwid_sysft_double_to_string(result * scale, value);
}

static int hsmp_arg1_res0_from_double(LikwidDevice_t dev, uint32_t msg_id, double scale, const char *value)
{
    double arg;
    int err = likwid_sysft_string_to_double(value, &arg);
    if (err < 0)
        return err;

    uint32_t actualArg = (uint32_t)(arg * scale);
    return hsmp_raw(dev->id.simple.id, msg_id, &actualArg, 1, NULL, 0);
}

static int amd_hsmp_tester(void)
{
    const uint32_t TEST_NUM = 42;
    uint32_t response;
    int err = hsmp_raw(0, HSMP_TEST, &TEST_NUM, 1, &response, 1);
    if (err != 0)
        return 0;
    return 1;
}

static int amd_hsmp_test_ver(uint32_t test_ver)
{
    if (test_ver == 0)
        return 0;
    uint32_t proto_ver;
    int err = hsmp_raw(0, HSMP_GET_PROTO_VER, NULL, 0, &proto_ver, 1);
    if (err < 0)
        return 0;
    if (test_ver <= proto_ver)
        return 1;
    return 0;
}

static int amd_hsmp_test_ver1(void) { return amd_hsmp_test_ver(1); }
static int amd_hsmp_test_ver2(void) { return amd_hsmp_test_ver(2); }
static int amd_hsmp_test_ver3(void) { return amd_hsmp_test_ver(3); }
//static int amd_hsmp_test_ver4(void) { return amd_hsmp_test_ver(4); } // there is only one command in ver4, and it is reserved or undocumented
static int amd_hsmp_test_ver5(void) { return amd_hsmp_test_ver(5); }
static int amd_hsmp_test_fail(void) { return 0; } // HSMP commandss > 2Fh are not yet documented and do not seem to work by tody (Bergamo + Genoa)

static int amd_hsmp_smu_fw_getter(LikwidDevice_t dev, char **value)
{
    return hsmp_arg0_res1_as_u32(dev, HSMP_GET_SMU_VER, value);
}

static int amd_hsmp_proto_ver_getter(LikwidDevice_t dev, char **value)
{
    return hsmp_arg0_res1_as_u32(dev, HSMP_GET_PROTO_VER, value);
}

static int amd_hsmp_sock_power_getter(LikwidDevice_t dev, char **value)
{
    return hsmp_arg0_res1_as_double(dev, HSMP_GET_SOCKET_POWER, 1e-3, value);
}

static int amd_hsmp_sock_power_limit_cur_getter(LikwidDevice_t dev, char **value)
{
    return hsmp_arg0_res1_as_double(dev, HSMP_GET_SOCKET_POWER_LIMIT, 1e-3, value);
}

static int amd_hsmp_sock_power_limit_cur_setter(LikwidDevice_t dev, const char *value)
{
    return hsmp_arg1_res0_from_double(dev, HSMP_SET_SOCKET_POWER_LIMIT, 1e3, value);
}

static int amd_hsmp_sock_power_limit_max_getter(LikwidDevice_t dev, char **value)
{
    return hsmp_arg0_res1_as_double(dev, HSMP_GET_SOCKET_POWER_LIMIT_MAX, 1e-3, value);
}

static int get_real_apic_id(const HWThread *hwt)
{
    /* This function get's the "real" APIC ID, since hwloc and likwid
     * appear to use a wrong one in HWThread. */

    /* sched_getaffinity/setaffinity may cause a condition!
     * Though this is only a problem when the affinity is changed from outside during runtime. */
    cpu_set_t old_mask;
    int err = sched_getaffinity(0, sizeof(old_mask), &old_mask);
    if (err != 0)
        return -errno;

    /* Make sure we are running on the right core. */
    cpu_set_t new_mask;
    CPU_ZERO(&new_mask);
    CPU_SET(hwt->apicId, &new_mask);   // <-- we assume this "apicId" to be a Linux processor number to be scheduled onto.
    err = sched_setaffinity(0, sizeof(new_mask), &new_mask);
    if (err != 0)
        return -errno;

    /* First, query initial local APIC ID, then try to query x2APIC ID. */
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    const uint32_t leaf_max = eax;

    assert(leaf_max >= 1);

    /* Retrieve initial APIC ID */
    eax = 1;
    CPUID(eax, ebx, ecx, edx);
    uint32_t apic_id = field32(ebx, 24, 8);
    if (leaf_max < 0xB)
        goto cleanup;

    /* Try to read extended local APIC ID */
    eax = 0xB;
    ecx = 0x0;
    CPUID(eax, ebx, ecx, edx);
    apic_id = edx;

    /* Intel recommends to use 0x1F instead of 0xB, but AMD PPR does not mention
     * the existence of such a CPUID leaf. */
cleanup:
    err = sched_setaffinity(0, sizeof(old_mask), &old_mask);
    if (err != 0)
        return -errno;
    return (int)apic_id;
}

static HWThread *get_hwt_by_core(LikwidDevice_t dev)
{
    assert(dev->type == DEVICE_TYPE_CORE);
    CpuTopology_t topo = get_cpuTopology();
    for (uint32_t hwt = 0; hwt < topo->numHWThreads; hwt++)
    {
        HWThread *t = &topo->threadPool[hwt];
        if (t->coreId == (uint32_t)dev->id.simple.id)
            return t;
    }
    return NULL;
}

static int amd_hsmp_core_boost_limit_getter(LikwidDevice_t dev, char **value)
{
    HWThread *hwt = get_hwt_by_core(dev);
    if (!hwt)
        return -EINVAL;
    int err = get_real_apic_id(hwt);
    if (err < 0)
        return err;

    const uint32_t real_apic_id = (uint32_t)err;
    uint32_t boost_limit;
    err = hsmp_raw(hwt->packageId, HSMP_GET_BOOST_LIMIT, &real_apic_id, 1, &boost_limit, 1);
    if (err < 0)
        return err;
    return likwid_sysft_uint64_to_string(boost_limit, value);
}

static int amd_hsmp_core_boost_limit_setter(LikwidDevice_t dev, const char *value)
{
    HWThread *hwt = get_hwt_by_core(dev);
    if (!hwt)
        return -EINVAL;
    int err = get_real_apic_id(hwt);
    if (err < 0)
        return err;

    const uint32_t real_apic_id = (uint32_t)err;
    uint64_t boost_limit;
    err = likwid_sysft_string_to_uint64(value, &boost_limit);
    if (err < 0)
        return err;

    uint32_t arg0 = 0;
    field32set(&arg0, 0, 16, boost_limit);
    field32set(&arg0, 16, 16, real_apic_id);
    return hsmp_raw(hwt->packageId, HSMP_SET_BOOST_LIMIT, &arg0, 1, NULL, 0);
}

static int amd_hsmp_sock_boost_limit_setter(LikwidDevice_t dev, const char *value)
{
    return hsmp_arg1_res0_from_u32(dev, HSMP_SET_BOOST_LIMIT_SOCKET, value);
}

static int amd_hsmp_sock_proc_hot_getter(LikwidDevice_t dev, char **value)
{
    return hsmp_arg0_res1_as_u32(dev, HSMP_GET_PROC_HOT, value);
}

static int amd_hsmp_sock_fclk_mclk_getter(LikwidDevice_t dev, bool fclk, char **value)
{
    uint32_t results[2];
    int err = hsmp_raw(dev->id.simple.id, HSMP_GET_FCLK_MCLK, NULL, 0, results, ARRAY_COUNT(results));
    if (err < 0)
        return -err;

    if (fclk)
        return likwid_sysft_uint64_to_string(results[0], value);
    return likwid_sysft_uint64_to_string(results[1], value);
}

static int amd_hsmp_sock_fclk_getter(LikwidDevice_t dev, char **value)
{
    return amd_hsmp_sock_fclk_mclk_getter(dev, true, value);
}

static int amd_hsmp_sock_mclk_getter(LikwidDevice_t dev, char **value)
{
    return amd_hsmp_sock_fclk_mclk_getter(dev, false, value);
}

static int amd_hsmp_xgmi_link_width_setter(LikwidDevice_t dev, const char *value)
{
    return hsmp_arg1_res0_from_u32(dev, HSMP_SET_XGMI_LINK_WIDTH, value);
}

static int amd_hsmp_df_pstate_setter(LikwidDevice_t dev, const char *value)
{
    return hsmp_arg1_res0_from_u32(dev, HSMP_SET_DF_PSTATE, value);
}

static int amd_hsmp_auto_df_pstate_setter(LikwidDevice_t dev, const char *value)
{
    uint64_t intval;
    int err = likwid_sysft_string_to_uint64(value, &intval);
    if (err < 0)
        return err;
    if (intval != 1)
        return -EINVAL;
    return hsmp_raw(dev->id.simple.id, HSMP_SET_AUTO_DF_PSTATE, NULL, 0, NULL, 0);
}

static int amd_hsmp_sock_cclk_thrtl_limit_getter(LikwidDevice_t dev, char **value)
{
    return hsmp_arg0_res1_as_u32(dev, HSMP_GET_CCLK_THROTTLE_LIMIT, value);
}

static int amd_hsmp_sock_c0_percent_getter(LikwidDevice_t dev, char **value)
{
    return hsmp_arg0_res1_as_u32(dev, HSMP_GET_C0_PERCENT, value);
}

static int amd_hsmp_sock_lclk_dpm_setter(LikwidDevice_t dev, const char *value)
{
    return hsmp_arg1_res0_from_u32(dev, HSMP_SET_NBIO_DPM_LEVEL, value);
}

static int amd_hsmp_dram_bw_getter(LikwidDevice_t dev, bool max, bool percent, char **value)
{
    uint32_t result;
    int err = hsmp_raw(dev->id.simple.id, HSMP_GET_DDR_BANDWIDTH, NULL, 0, &result, 1);
    if (err < 0)
        return err;
    if (max)
        return likwid_sysft_double_to_string(field32(result, 20, 12) / 8.0, value);
    if (percent)
        return likwid_sysft_uint64_to_string(field32(result, 0, 8), value);
    return likwid_sysft_double_to_string(field32(result, 8, 12) / 8.0, value);
}

static int amd_hsmp_dram_bw_max_getter(LikwidDevice_t dev, char **value)
{
    return amd_hsmp_dram_bw_getter(dev, true, false, value);
}

static int amd_hsmp_dram_bw_avg_getter(LikwidDevice_t dev, char **value)
{
    return amd_hsmp_dram_bw_getter(dev, false, false, value);
}

static int amd_hsmp_dram_bw_perc_getter(LikwidDevice_t dev, char **value)
{
    return amd_hsmp_dram_bw_getter(dev, false, true, value);
}

static int amd_hsmp_temp_getter(LikwidDevice_t dev, char **value)
{
    uint32_t temp;
    int err = hsmp_raw(dev->id.simple.id, HSMP_GET_TEMP_MONITOR, NULL, 0, &temp, 1);
    if (err < 0)
        return err;
    return likwid_sysft_double_to_string(field32(temp, 5, 11) / 8.0, value);
}

static uint32_t make_dimm_addr(uint32_t channel, bool dimm_0_1, bool sensor_0_1)
{
    uint32_t dimm_addr = 0;
    field32set(&dimm_addr, 7, 1, 1);   // mode = 1
    field32set(&dimm_addr, 6, 1, sensor_0_1 ? 1 : 0);
    field32set(&dimm_addr, 4, 1, dimm_0_1 ? 1 : 0);
    field32set(&dimm_addr, 0, 4, channel);
    return dimm_addr;
}

static int amd_hsmp_dimm_temp_getter(LikwidDevice_t dev, uint32_t channel, bool dimm_0_1, bool sensor_0_1, bool rate, bool test, char **value)
{
    /* For reference, see AMD PPR 19h Model 11h B2 Vol 3 aka 55901_B2_pub_3.pdf Table 141  */
    assert(channel < 16); // HSMP protocol currently does not support more than 4 bit DDRPHY IDs

    const uint32_t dimm_addr = make_dimm_addr(channel, dimm_0_1, sensor_0_1);

    uint32_t range_raw;
    int err = hsmp_raw(dev->id.simple.id, HSMP_GET_DIMM_TEMP_RANGE, &dimm_addr, 1, &range_raw, 1);
    if (err < 0)
        return err;

    /* See same document as above, Table 154 function Id 16h */
    const double refresh_pre_scale = field32(range_raw, 3, 1) ? 2.0 : 1.0;
    if (field32(range_raw, 0, 3) != 0x1 && field32(range_raw, 0, 3) != 0x5)
    {
        ERROR_PRINT(AMD HSMP: received invalid or unknown temperature range: %x, field32(range_raw, 0, 3));
        return -EBADE;
    }
    const double temp_pre_scale = (field32(range_raw, 0, 3) == 0x1) ? 1.0 : 2.0;

    uint32_t temp_raw;
    err = hsmp_raw(dev->id.simple.id, HSMP_GET_DIMM_THERMAL, &dimm_addr, 1, &temp_raw, 1);
    if (err < 0)
        return err;

    if (test)
        return 0;

    if (rate)
    {
        const double last_update = refresh_pre_scale * field32(temp_raw, 8, 9) / 1000.0;
        return likwid_sysft_double_to_string(last_update, value);
    }
    else
    {
        /* Do not use field32(temp_raw, 21, 11) in order to get sign extension from the >> for free. */
        const double temp = temp_pre_scale * (temp_raw >> 21) * 0.25;
        return likwid_sysft_double_to_string(temp, value);
    }
}

static int amd_hsmp_dimm_power_getter(LikwidDevice_t dev, uint32_t channel, bool dimm_0_1, bool sensor_0_1, bool test, char **value)
{
    /* See notes from hsmp_amd_dimm_temp_getter */
    assert(channel < 16);
    const uint32_t dimm_addr = make_dimm_addr(channel, dimm_0_1, sensor_0_1);
    uint32_t power;
    int err = hsmp_raw(dev->id.simple.id, HSMP_GET_DIMM_THERMAL, &dimm_addr, 1, &power, 1);
    if (err < 0)
        return err;
    if (test)
        return 0;
    return likwid_sysft_double_to_string(power / 1000.0, value);
}

static int amd_hsmp_dimm_tester(uint32_t channel, bool dimm_0_1, bool sensor_0_1, bool temp)
{
    if (!amd_hsmp_test_ver5())
        return 0;
    CpuTopology_t topo = get_cpuTopology();
    bool dimm_found = false;
    for (uint32_t i = 0; i < topo->numSockets; i++)
    {
        LikwidDevice_t dev;
        int err = likwid_device_create(DEVICE_TYPE_SOCKET, (int)i, &dev);
        if (err < 0)
            return err;
        if (temp)
            err = amd_hsmp_dimm_temp_getter(dev, channel, dimm_0_1, sensor_0_1, false, true, NULL);
        else
            err = amd_hsmp_dimm_power_getter(dev, channel, dimm_0_1, sensor_0_1, true, NULL);
        likwid_device_destroy(dev);
        if (err == 0)
        {
            dimm_found = true;
            break;
        }
    }
    return dimm_found;
}

#define MAKE_DIMM_FUNC(channel, dimm, sensor)                               \
    static int amd_hsmp_dimm##channel##_##dimm##_ts##sensor##_temp_getter(LikwidDevice_t dev, char **value) \
    {                                                                       \
        return amd_hsmp_dimm_temp_getter(dev, channel, dimm, sensor, false, false, value); \
    }                                                                       \
    static int amd_hsmp_dimm##channel##_##dimm##_ts##sensor##_temp_rate_getter(LikwidDevice_t dev, char **value) \
    {                                                                       \
        return amd_hsmp_dimm_temp_getter(dev, channel, dimm, sensor, true, false, value);   \
    }                                                                       \
    static int amd_hsmp_dimm##channel##_##dimm##_ts##sensor##_temp_tester(void) \
    {                                                                       \
        return amd_hsmp_dimm_tester(channel, dimm, sensor, true);           \
    }                                                                       \
    static int amd_hsmp_dimm##channel##_##dimm##_ts##sensor##_power_getter(LikwidDevice_t dev, char **value) \
    {                                                                       \
        return amd_hsmp_dimm_power_getter(dev, channel, dimm, sensor, false, value); \
    }                                                                       \
    static int amd_hsmp_dimm##channel##_##dimm##_ts##sensor##_power_tester(void) \
    {                                                                       \
        return amd_hsmp_dimm_tester(channel, dimm, sensor, false);          \
    }
#define MAKE_DIMM_FUNC_SET(channel) \
    MAKE_DIMM_FUNC(channel, 0, 0)   \
    MAKE_DIMM_FUNC(channel, 0, 1)   \
    MAKE_DIMM_FUNC(channel, 1, 0)   \
    MAKE_DIMM_FUNC(channel, 1, 1)
MAKE_DIMM_FUNC_SET(0x0);
MAKE_DIMM_FUNC_SET(0x1);
MAKE_DIMM_FUNC_SET(0x2);
MAKE_DIMM_FUNC_SET(0x3);
MAKE_DIMM_FUNC_SET(0x4);
MAKE_DIMM_FUNC_SET(0x5);
MAKE_DIMM_FUNC_SET(0x6);
MAKE_DIMM_FUNC_SET(0x7);
MAKE_DIMM_FUNC_SET(0x8);
MAKE_DIMM_FUNC_SET(0x9);
MAKE_DIMM_FUNC_SET(0xA);
MAKE_DIMM_FUNC_SET(0xB);
MAKE_DIMM_FUNC_SET(0xC);
MAKE_DIMM_FUNC_SET(0xD);
MAKE_DIMM_FUNC_SET(0xE);
MAKE_DIMM_FUNC_SET(0xF);

#define MAKE_DIMM_FEATURE(channel, dimm, sensor)                    \
    {                                                               \
        "dimm" #channel "_" #dimm "_" #sensor "_temp", "hsmp", "DIMM temperature (channel " #channel ", module " #dimm ", sensor " #sensor ")",\
        amd_hsmp_dimm##channel##_##dimm##_ts##sensor##_temp_getter, \
        NULL,                                                       \
        DEVICE_TYPE_SOCKET,                                         \
        amd_hsmp_dimm##channel##_##dimm##_ts##sensor##_temp_tester, \
        "degrees C"                                                 \
    },                                                              \
    {                                                               \
        "dimm" #channel "_" #dimm "_" #sensor "_temp_upd", "hsmp", "DIMM temp last update (channel " #channel ", module " #dimm ", sensor " #sensor ")",\
        amd_hsmp_dimm##channel##_##dimm##_ts##sensor##_temp_rate_getter, \
        NULL,                                                       \
        DEVICE_TYPE_SOCKET,                                         \
        amd_hsmp_dimm##channel##_##dimm##_ts##sensor##_temp_tester, \
        "s"                                                         \
    },                                                              \
    {                                                               \
        "dimm" #channel "_" #dimm "_" #sensor "_power", "hsmp", "DIMM power usage (channel " #channel ", module " #dimm ", sensor " #sensor ")",\
        amd_hsmp_dimm##channel##_##dimm##_ts##sensor##_power_getter,\
        NULL,                                                       \
        DEVICE_TYPE_SOCKET,                                         \
        amd_hsmp_dimm##channel##_##dimm##_ts##sensor##_power_tester,\
        "W"                                                         \
    },
#define MAKE_DIMM_FEATURES(channel)     \
    MAKE_DIMM_FEATURE(channel, 0, 0)    \
    MAKE_DIMM_FEATURE(channel, 0, 1)    \
    MAKE_DIMM_FEATURE(channel, 1, 0)    \
    MAKE_DIMM_FEATURE(channel, 1, 1)

struct flag_freq_reason_mapping
{
    uint32_t flag;
    const char *reason;
};

static int amd_hsmp_sock_freq_limit_getter(LikwidDevice_t dev, bool show_reason, char **value)
{
    uint32_t freq;
    int err = hsmp_raw(dev->id.simple.id, HSMP_GET_SOCKET_FREQ_LIMIT, NULL, 0, &freq, 1);
    if (err < 0)
        return err;

    if (!show_reason)
        return likwid_sysft_uint64_to_string(field32(freq, 16, 16), value);

    static const struct flag_freq_reason_mapping map[] = {
        { 0x01, "cHTC-Active" },    // ???
        { 0x02, "PROCHOT" },        // ???
        { 0x04, "TDC" },            // Thermal Designed Current Limit
        { 0x08, "PPT" },            // Package Power Tracking Limit
        { 0x10, "OPN-Max" },        // ???
        { 0x20, "Reliability-Limit" }, // (Fused Max or Reliability Monitor Fmax@Vmax)
        { 0x40, "APML-Agent" },     // ???
        { 0x80, "HSMP-Agent" },     // ???
    };

    bstring reasons = bfromcstr("");
    if (!reasons)
        return -ENOMEM;

    bool first = true;
    uint32_t reason = field32(freq, 0, 16);
    for (size_t i = 0; i < ARRAY_COUNT(map); i++)
    {
        if (!(reason & map[i].flag))
            continue;
        reason &= ~map[i].flag;
        if (!first)
            bcatcstr(reasons, " ");
        first = false;
        bcatcstr(reasons, map[i].reason);
    }

    if (reason)
        ERROR_PRINT(Found unexpected bits in HSMP Freq Limit string: %04x, reason);

    err = likwid_sysft_copystr(bdata(reasons), value);
    bdestroy(reasons);
    return err;
}

static int amd_hsmp_sock_freq_limit_freq_getter(LikwidDevice_t dev, char **value)
{
    return amd_hsmp_sock_freq_limit_getter(dev, false, value);
}

static int amd_hsmp_sock_freq_limit_reason_getter(LikwidDevice_t dev, char **value)
{
    return amd_hsmp_sock_freq_limit_getter(dev, true, value);
}

static int amd_hsmp_core_cclk_limit_getter(LikwidDevice_t dev, char **value)
{
    HWThread *hwt = get_hwt_by_core(dev);
    if (!hwt)
        return -EINVAL;
    int err = get_real_apic_id(hwt);
    if (err < 0)
        return err;

    const uint32_t real_apic_id = (uint32_t)err;
    uint32_t freq;
    err = hsmp_raw(hwt->packageId, HSMP_GET_CCLK_CORE_LIMIT, &real_apic_id, 1, &freq, 1);
    if (err < 0)
        return err;
    return likwid_sysft_uint64_to_string(freq, value);
}

static int amd_hsmp_sock_rails_svi_getter(LikwidDevice_t dev, char **value)
{
    return hsmp_arg0_res1_as_double(dev, HSMP_GET_RAILS_SVI, 1e-3, value);
}

static int amd_hsmp_sock_fmax_fmin_getter(LikwidDevice_t dev, bool get_fmax, char **value)
{
    uint32_t result;
    int err = hsmp_raw(dev->id.simple.id, HSMP_GET_SOCKET_FMAX_FMIN, NULL, 0, &result, 1);
    if (err < 0)
        return err;

    if (get_fmax)
        return likwid_sysft_uint64_to_string(field32(result, 16, 16), value);
    return likwid_sysft_uint64_to_string(field32(result, 0, 16), value);
}

static int amd_hsmp_sock_fmax_getter(LikwidDevice_t dev, char **value)
{
    return amd_hsmp_sock_fmax_fmin_getter(dev, true, value);
}

static int amd_hsmp_sock_fmin_getter(LikwidDevice_t dev, char **value)
{
    return amd_hsmp_sock_fmax_fmin_getter(dev, false, value);
}

typedef enum {
    BW_aggr = 0x1,
    BW_read = 0x2,
    BW_write = 0x4,
} XGMIBw;

typedef enum {
    XGMI_p0 = 0x01,
    XGMI_p1 = 0x02,
    XGMI_p2 = 0x04,
    XGMI_p3 = 0x08,
    XGMI_g0 = 0x10,
    XGMI_g1 = 0x20,
    XGMI_g2 = 0x40,
    XGMI_g3 = 0x80,
} XGMILinkId;

static int amd_hsmp_sock_xgmi_bw_getter(LikwidDevice_t dev, XGMILinkId id, XGMIBw bw, char **value)
{
    uint32_t arg0 = 0;
    field32set(&arg0, 0, 3, bw);
    field32set(&arg0, 8, 8, id);

    uint32_t bw_result; // i guess even AMD makes typos in their constants (BANDWITH)
    int err = hsmp_raw(dev->id.simple.id, HSMP_GET_XGMI_BANDWITH, &arg0, 1, &bw_result, 1);
    if (err < 0)
        return err;

    return likwid_sysft_uint64_to_string(bw_result, value);
}

#define MAKE_XGMI_FUNC(id, bw) \
    static int amd_hsmp_sock_xgmi_bw_##id##_##bw##_getter(LikwidDevice_t dev, char **value)\
    {                                                                       \
        return amd_hsmp_sock_xgmi_bw_getter(dev, XGMI_##id, BW_##bw, value);\
    }
#define MAKE_XGMI_FUNCS(bw)     \
    MAKE_XGMI_FUNC(bw, aggr)    \
    MAKE_XGMI_FUNC(bw, read)    \
    MAKE_XGMI_FUNC(bw, write)
MAKE_XGMI_FUNCS(p0);
MAKE_XGMI_FUNCS(p1);
MAKE_XGMI_FUNCS(p2);
MAKE_XGMI_FUNCS(p3);
MAKE_XGMI_FUNCS(g0);
MAKE_XGMI_FUNCS(g1);
MAKE_XGMI_FUNCS(g2);
MAKE_XGMI_FUNCS(g3);

#define MAKE_XGMI_FEATURE(id)                       \
    {                                               \
        "pkg_xgmi_bw_" #id "_aggr", "hsmp", "Aggregated xGMI " #id " bandwidth",\
        amd_hsmp_sock_xgmi_bw_##id##_aggr_getter,   \
        NULL,                                       \
        DEVICE_TYPE_SOCKET,                         \
        amd_hsmp_test_ver5,                         \
        "GB/s"                                      \
    },                                              \
    {                                               \
        "pkg_xgmi_bw_" #id "_read", "hsmp", "xGMI " #id " read bandwidth",\
        amd_hsmp_sock_xgmi_bw_##id##_read_getter,   \
        NULL,                                       \
        DEVICE_TYPE_SOCKET,                         \
        amd_hsmp_test_ver5,                         \
        "GB/s"                                      \
    },                                              \
    {                                               \
        "pkg_xgmi_bw_" #id "_write", "hsmp", "xGMI " #id " write bandwidth",\
        amd_hsmp_sock_xgmi_bw_##id##_write_getter,  \
        NULL,                                       \
        DEVICE_TYPE_SOCKET,                         \
        amd_hsmp_test_ver5,                         \
        "GB/s"                                      \
    },

static int amd_hsmp_gmi3_width_setter(LikwidDevice_t dev, const char *value)
{
    return hsmp_arg1_res0_from_u32(dev, HSMP_SET_GMI3_WIDTH, value);
}

static int amd_hsmp_pci_gen_limit_setter(LikwidDevice_t dev, const char *value)
{
    return hsmp_arg1_res0_from_u32(dev, HSMP_SET_PCI_RATE, value);
}

struct power_mode_mapping
{
    uint32_t val;
    const char *name;
};

static const struct power_mode_mapping power_mode_map[] = {
    { 0, "high-perf" },
    { 1, "efficency" },
    { 2, "io-perf" },
    { 3, "balanced" },
};

static int amd_hsmp_power_mode_getter(LikwidDevice_t dev, char **value)
{
    const uint32_t arg0 = 0x80000000;
    uint32_t result;
    int err = hsmp_raw(dev->id.simple.id, HSMP_SET_POWER_MODE, &arg0, 1, &result, 1);
    if (err < 0)
        return err;
    for (size_t i = 0; i < ARRAY_COUNT(power_mode_map); i++)
    {
        if (field32(result, 0, 3) == power_mode_map[i].val)
            return likwid_sysft_copystr(power_mode_map[i].name, value);
    }
    char invalid_buf[64];
    snprintf(invalid_buf, sizeof(invalid_buf), "invalid (%u)", field32(result, 0, 3));
    return likwid_sysft_copystr(invalid_buf, value);
}

static int amd_hsmp_power_mode_setter(LikwidDevice_t dev, const char *value)
{
    bool found = false;
    size_t mapped_val;
    for (size_t i = 0; i < ARRAY_COUNT(power_mode_map); i++)
    {
        if (power_mode_map[i].name == value)
        {
            mapped_val = i;
            found = true;
        }
    }
    if (!found)
        return -EINVAL;
    uint32_t arg0 = 0;
    field32set(&arg0, 0, 3, mapped_val);
    return hsmp_raw(dev->id.simple.id, HSMP_SET_POWER_MODE, &arg0, 1, NULL, 0);
}

static int amd_hsmp_pstate_min_max_setter(LikwidDevice_t dev, const char *value)
{
    return hsmp_arg1_res0_from_u32(dev, HSMP_SET_PSTATE_MAX_MIN, value);
}

static int amd_hsmp_metric_table_ver_getter(LikwidDevice_t dev, char **value)
{
    return hsmp_arg0_res1_as_u32(dev, HSMP_GET_METRIC_TABLE_VER, value);
}

static int amd_hsmp_metric_table_addr_getter(LikwidDevice_t dev, char **value)
{
    uint32_t addr[2];
    int err = hsmp_raw(dev->id.simple.id, HSMP_GET_METRIC_TABLE_DRAM_ADDR, NULL, 0, addr, ARRAY_COUNT(addr));
    if (err < 0)
        return err;
    return likwid_sysft_uint64_to_string(addr[0] | ((uint64_t)addr[1] << 32), value);
}

static int amd_hsmp_xgmi_pstate_min_max_setter(LikwidDevice_t dev, const char *value)
{
    return hsmp_arg1_res0_from_u32(dev, HSMP_SET_XGMI_PSTATE_RANGE, value);
}

static int amd_hsmp_cpu_rail_iso_freq_policy_getter(LikwidDevice_t dev, char **value)
{
    const uint32_t arg0 = 0x80000000;
    uint32_t result;
    int err = hsmp_raw(dev->id.simple.id, HSMP_CPU_RAIL_ISO_FREQ_POLICY, &arg0, 1, &result, 1);
    if (err < 0)
        return err;
    if (field32(result, 0, 1))
        return likwid_sysft_copystr("true", value);
    return likwid_sysft_copystr("false", value);
}

static int amd_hsmp_cpu_rail_iso_freq_policy_setter(LikwidDevice_t dev, const char *value)
{
    return hsmp_arg1_res0_from_u32(dev, HSMP_CPU_RAIL_ISO_FREQ_POLICY, value);
}

static int amd_hsmp_dfc_enable_getter(LikwidDevice_t dev, char **value)
{
    const uint32_t arg0 = 0x80000000;
    uint32_t result;
    int err = hsmp_raw(dev->id.simple.id, HSMP_DFC_ENABLE_CTRL, &arg0, 1, &result, 1);
    if (err < 0)
        return err;
    if (field32(result, 0, 1))
        return likwid_sysft_copystr("true", value);
    return likwid_sysft_copystr("false", value);
}

static int amd_hsmp_dfc_enable_setter(LikwidDevice_t dev, const char *value)
{
    return hsmp_arg1_res0_from_u32(dev, HSMP_DFC_ENABLE_CTRL, value);
}

static int amd_hsmp_rapl_init(LikwidDevice_t dev)
{
    if (rapl_domain_info.energyUnit > 0.0 || rapl_domain_info.timeUnit > 0.0)
        return 0;
    uint32_t units;
    int err = hsmp_raw(dev->id.simple.id, HSMP_GET_RAPL_UNITS, NULL, 0, &units, 1);
    if (err < 0)
        return err;
    rapl_domain_info.energyUnit = 1.0 / (1 << field32(units, 8, 5));
    rapl_domain_info.timeUnit = 1.0 / (1 << field32(units, 16, 4));
    return 0;
}

static int amd_hsmp_core_energy_getter(LikwidDevice_t dev, char **value)
{
    int err = amd_hsmp_rapl_init(dev);
    if (err < 0)
        return err;
    HWThread *hwt = get_hwt_by_core(dev);
    if (!hwt)
        return -EINVAL;
    err = get_real_apic_id(hwt);
    if (err < 0)
        return err;

    const uint32_t real_apic_id = (uint32_t)err;
    uint32_t energy_raw[2];
    err = hsmp_raw(hwt->packageId, HSMP_GET_RAPL_CORE_COUNTER, &real_apic_id, 1, energy_raw, 2);
    if (err < 0)
        return err;
    const uint64_t energy_real = energy_raw[0] | ((uint64_t)energy_raw[1] << 32);
    return likwid_sysft_uint64_to_string((double)energy_real * rapl_domain_info.energyUnit, value);
}

static int amd_hsmp_sock_energy_getter(LikwidDevice_t dev, char **value)
{
    int err = amd_hsmp_rapl_init(dev);
    if (err < 0)
        return err;
    uint32_t energy_raw[2];
    err = hsmp_raw(dev->id.simple.id, HSMP_GET_RAPL_PACKAGE_COUNTER, NULL, 0, energy_raw, 2);
    if (err < 0)
        return err;
    const uint64_t energy_real = energy_raw[0] | ((uint64_t)energy_raw[1] << 32);
    return likwid_sysft_uint64_to_string((double)energy_real * rapl_domain_info.energyUnit, value);
}

static _SysFeature amd_hsmp_features[] = {
    {"smu_fw_ver", "hsmp", "SMU Firmware Version", amd_hsmp_smu_fw_getter, NULL, DEVICE_TYPE_SOCKET, NULL},
    {"proto_ver", "hsmp", "HSMP Protocol Version", amd_hsmp_proto_ver_getter, NULL, DEVICE_TYPE_SOCKET, NULL},
    {"pkg_power", "hsmp", "Current socket power consumption", amd_hsmp_sock_power_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver1, "W"},
    {"pkg_power_limit_cur", "hsmp", "Current socket power limit", amd_hsmp_sock_power_limit_cur_getter, amd_hsmp_sock_power_limit_cur_setter, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver1, "W"},
    {"pkg_power_limit_max", "hsmp", "Maximum socket power limit", amd_hsmp_sock_power_limit_max_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver1, "W"},
    {"core_boost_limit_cur", "hsmp", "Current core boost limit", amd_hsmp_core_boost_limit_getter, amd_hsmp_core_boost_limit_setter, DEVICE_TYPE_CORE, amd_hsmp_test_ver1, "MHz"},
    {"pkg_boost_limit_cur", "hsmp", "Current socket boost limit", NULL, amd_hsmp_sock_boost_limit_setter, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver1, "MHz"},
    {"pkg_prochot", "hsmp", "Processor hot (throttling?)", amd_hsmp_sock_proc_hot_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver1},
    {"pkg_xgmi_link_width", "hsmp", "xGMI Link width range ([15:8] = min, [7:0] = max, 0 = x4, 1 = x8, 2 = x16", NULL, amd_hsmp_xgmi_link_width_setter, DEVICE_TYPE_NODE, amd_hsmp_test_ver1},
    {"pkg_df_pstate", "hsmp", "Disable AMD Precision Boost (override P-State: 0 = high performance .. 2 = low performance)", NULL, amd_hsmp_df_pstate_setter, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver1},
    {"pkg_df_pstate_auto", "hsmp", "Enable AMD Precision Boost (auto manage P-State)", NULL, amd_hsmp_auto_df_pstate_setter, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver1},
    {"pkg_fclk", "hsmp", "Current Infinity Fabric clock speed", amd_hsmp_sock_fclk_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver1, "MHz"},
    {"pkg_mclk", "hsmp", "Current Memory Controller clock speed", amd_hsmp_sock_mclk_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver1, "MHz"},
    {"pkg_cclk_thrtl_limit", "hsmp", "Core Clock throttle limit", amd_hsmp_sock_cclk_thrtl_limit_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver1, "MHz"},
    {"pkg_c0", "hsmp", "Average C0 residency", amd_hsmp_sock_c0_percent_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver1},
    {"pkg_lclk_dpm_level_min_max", "hsmp", "Set LCLK DPM Level ([23:16] = NBIO ID (0..3), [15:8] = max DPM, [7:0] = min DPM, 0 = lowest DPM freq, 1..3 = highest DPM freq)", NULL, amd_hsmp_sock_lclk_dpm_setter, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver2},
    {"pkg_dram_bw_max", "hsmp", "Maximum possible DRAM bandwidth", amd_hsmp_dram_bw_max_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver3, "GB/s"},
    {"pkg_dram_bw_avg", "hsmp", "Average DRAM bandwidth", amd_hsmp_dram_bw_avg_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5, "GB/s"},
    {"pkg_dram_bw_perc", "hsmp", "DRAM bandwidth utilization", amd_hsmp_dram_bw_perc_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5, "%"},
    {"pkg_temp", "hsmp", "Current socket temperature", amd_hsmp_temp_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5, "degrees C"},
    /* Sorry, this below is insanely duplicated code, but there is currently no way
     * to avoid it without dynamically allocating the entire feature properties. */
    MAKE_DIMM_FEATURES(0x0)
    MAKE_DIMM_FEATURES(0x1)
    MAKE_DIMM_FEATURES(0x2)
    MAKE_DIMM_FEATURES(0x3)
    MAKE_DIMM_FEATURES(0x4)
    MAKE_DIMM_FEATURES(0x5)
    MAKE_DIMM_FEATURES(0x6)
    MAKE_DIMM_FEATURES(0x7)
    MAKE_DIMM_FEATURES(0x8)
    MAKE_DIMM_FEATURES(0x9)
    MAKE_DIMM_FEATURES(0xA)
    MAKE_DIMM_FEATURES(0xB)
    MAKE_DIMM_FEATURES(0xC)
    MAKE_DIMM_FEATURES(0xD)
    MAKE_DIMM_FEATURES(0xE)
    MAKE_DIMM_FEATURES(0xF)
    {"pkg_freq_limit", "hsmp", "Current socket frequency limit", amd_hsmp_sock_freq_limit_freq_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5, "MHz"},
    {"pkg_freq_limit_reason", "hsmp", "Current socket frequency limit reason", amd_hsmp_sock_freq_limit_reason_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5},
    {"core_cclk_limit", "hsmp", "Current Core clock limit", amd_hsmp_core_cclk_limit_getter, NULL, DEVICE_TYPE_CORE, amd_hsmp_test_ver5, "MHz"},
    {"pkg_rails_svi", "hsmp", "SVI based telemetry for all rails (?)", amd_hsmp_sock_rails_svi_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5, "W"},
    {"pkg_fmax", "hsmp", "Socket fmax", amd_hsmp_sock_fmax_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5, "Mhz"},
    {"pkg_fmin", "hsmp", "Socket fmin", amd_hsmp_sock_fmin_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5, "Mhz"},
    /* Same case here as above ... */
    MAKE_XGMI_FEATURE(p0)
    MAKE_XGMI_FEATURE(p1)
    MAKE_XGMI_FEATURE(p2)
    MAKE_XGMI_FEATURE(p3)
    MAKE_XGMI_FEATURE(g0)
    MAKE_XGMI_FEATURE(g1)
    MAKE_XGMI_FEATURE(g2)
    MAKE_XGMI_FEATURE(g3)
    {"pkg_gmi3_link_width", "hsmp", "Set minimum and maximum GMI3 link width (bitfield integer: min[15:8], max[7:0], 0 = quarter, 1 = half, 2 = full)", NULL, amd_hsmp_gmi3_width_setter, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5},
    {"pkg_pcie_gen_limit", "hsmp", "Set maximum PCIe gen (0 = auto, 1 = gen4, 2 = gen5)", NULL, amd_hsmp_pci_gen_limit_setter, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5},
    {"pkg_power_mode", "hsmp", "Current power mode", amd_hsmp_power_mode_getter, amd_hsmp_power_mode_setter, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5},
    {"pkg_pstate_min_max", "hsmp", "Set minimum and maximum Pstate ([15:8] = min, [7:0] = max, valid values: 0 = high performance .. 2 = low performance)", NULL, amd_hsmp_pstate_min_max_setter, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5},
    {"pkg_metric_table_ver", "hsmp", "Metric Table Version (?)", amd_hsmp_metric_table_ver_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5},
    {"pkg_metric_table_addr", "hsmp", "Metric Table DRAM address (?)", amd_hsmp_metric_table_addr_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5},
    {"pkg_xgmi_pstate_min_max", "hsmp", "Set minimum and maximum xGMI Pstate ([15:8] = min, [7:0] = max)", NULL, amd_hsmp_xgmi_pstate_min_max_setter, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5},
    {"pkg_rail_iso_freq_pol", "hsmp", "CPU Rail Iso Freq Policy (?)", amd_hsmp_cpu_rail_iso_freq_policy_getter, amd_hsmp_cpu_rail_iso_freq_policy_setter, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5},
    {"pkg_dfc_enable", "hsmp", "DFC Enable (?)", amd_hsmp_dfc_enable_getter, amd_hsmp_dfc_enable_setter, DEVICE_TYPE_SOCKET, amd_hsmp_test_ver5},
    {"pkg_energy", "hsmp", "Socket energy consumed", amd_hsmp_sock_energy_getter, NULL, DEVICE_TYPE_SOCKET, amd_hsmp_test_fail, "J"},
    {"core_energy", "hsmp", "Core energy consumed", amd_hsmp_core_energy_getter, NULL, DEVICE_TYPE_CORE, amd_hsmp_test_fail, "J"},
};

static const _SysFeatureList amd_hsmp_featuer_list = {
    .num_features = ARRAY_COUNT(amd_hsmp_features),
    .features = amd_hsmp_features,
    .tester = amd_hsmp_tester,
};
