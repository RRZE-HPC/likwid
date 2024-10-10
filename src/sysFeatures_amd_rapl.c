#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <sysFeatures_amd.h>
#include <sysFeatures_common.h>
#include <sysFeatures_amd_rapl.h>
#include <access.h>
#include <registers.h>

#include <sysFeatures_common_rapl.h>

static RaplDomainInfo amd_rapl_pkg_info = {0, 0, 0};
static RaplDomainInfo amd_rapl_core_info = {0, 0, 0};
static RaplDomainInfo amd_rapl_l3_info = {0, 0, 0};

static int sysFeatures_amd_rapl_energy_status_getter(const LikwidDevice_t device, char** value, uint32_t reg, const RaplDomainInfo* info, LikwidDeviceType type)
{
    int err = getset_info_check(device, value, info, type);
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
    const uint64_t energy = field64(msrData, 0, 32);
    return sysFeatures_double_to_string((double)energy * info->energyUnit, value);
}

/*********************************************************************************************************************/
/*                          Amd RAPL (PKG domain)                                                                  */
/*********************************************************************************************************************/

static int pkg_test_testFunc(uint64_t msrData, void *)
{
    if (amd_rapl_pkg_info.powerUnit == 0 && amd_rapl_pkg_info.energyUnit == 0 && amd_rapl_pkg_info.timeUnit == 0)
    {
        amd_rapl_pkg_info.powerUnit = 1.0 / (1 << field64(msrData, 0, 4));
        amd_rapl_pkg_info.energyUnit = 1.0 / (1 << field64(msrData, 8, 5));
        amd_rapl_pkg_info.timeUnit = 1.0 / (1 << field64(msrData, 16, 4));
    }
    return 1;
}

int amd_rapl_pkg_test(void)
{
    return likwid_sysft_foreach_socket_testmsr_cb(MSR_AMD17_RAPL_POWER_UNIT, pkg_test_testFunc, NULL);
}

int sysFeatures_amd_pkg_energy_status_test(void)
{
    return likwid_sysft_foreach_socket_testmsr(MSR_AMD17_RAPL_PKG_STATUS);
}

int sysFeatures_amd_pkg_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_amd_rapl_energy_status_getter(device, value, MSR_AMD17_RAPL_PKG_STATUS, &amd_rapl_pkg_info, DEVICE_TYPE_SOCKET);
}

/*********************************************************************************************************************/
/*                          AMD RAPL (CORE domain)                                                                 */
/*********************************************************************************************************************/

static int core_test_testFunc(uint64_t msrData, void *)
{
    if (amd_rapl_core_info.powerUnit == 0 && amd_rapl_core_info.energyUnit == 0 && amd_rapl_core_info.timeUnit == 0)
    {
        amd_rapl_core_info.powerUnit = 1.0 / (1 << field64(msrData, 0, 4));
        amd_rapl_core_info.energyUnit = 1.0 / (1 << field64(msrData, 8, 5));
        amd_rapl_core_info.timeUnit = 1.0 / (1 << field64(msrData, 16, 4));
    }
    return 1;
}

int amd_rapl_core_test(void)
{
    return likwid_sysft_foreach_core_testmsr_cb(MSR_AMD17_RAPL_POWER_UNIT, core_test_testFunc, NULL);
}

int sysFeatures_amd_core_energy_status_test(void)
{
    return likwid_sysft_foreach_core_testmsr(MSR_AMD17_RAPL_CORE_STATUS);
}

int sysFeatures_amd_core_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_amd_rapl_energy_status_getter(device, value, MSR_AMD17_RAPL_CORE_STATUS, &amd_rapl_core_info, DEVICE_TYPE_CORE);
}

/*********************************************************************************************************************/
/*                          AMD RAPL (L3 domain)                                                                     */
/*********************************************************************************************************************/

static int l3_test_testFunc(uint64_t msrData, void *)
{
    if (amd_rapl_l3_info.powerUnit == 0 && amd_rapl_l3_info.energyUnit == 0 && amd_rapl_l3_info.timeUnit == 0)
    {
        amd_rapl_l3_info.powerUnit = 1.0 / (1 << field64(msrData, 0, 4));
        amd_rapl_l3_info.energyUnit = 1.0 / (1 << field64(msrData, 8, 5));
        amd_rapl_l3_info.timeUnit = 1.0 / (1 << field64(msrData, 16, 4));
    }
    return 1;
}

int amd_rapl_l3_test(void)
{
    int err = topology_init();
    CpuInfo_t info = get_cpuInfo();
    if (info->family != ZEN3_FAMILY)
        return 0;
    if (info->model != ZEN4_RYZEN && info->model != ZEN4_RYZEN_PRO && info->model != ZEN4_EPYC)
        return 0;
    return likwid_sysft_foreach_socket_testmsr_cb(MSR_AMD19_RAPL_L3_UNIT, l3_test_testFunc, NULL);
}

int sysFeatures_amd_l3_energy_status_test(void)
{
    return likwid_sysft_foreach_core_testmsr(MSR_AMD19_RAPL_L3_STATUS);
}

int sysFeatures_amd_l3_energy_status_getter(const LikwidDevice_t device, char** value)
{
    return sysFeatures_amd_rapl_energy_status_getter(device, value, MSR_AMD19_RAPL_L3_STATUS, &amd_rapl_l3_info, DEVICE_TYPE_SOCKET);
}

/* Init function */

int sysFeatures_init_amd_rapl(_SysFeatureList* out)
{
    int err = 0;
    if (amd_rapl_pkg_test())
    {
        DEBUG_PRINT(DEBUGLEV_INFO, Register Amd RAPL PKG domain);
        err = register_features(out, &amd_rapl_pkg_feature_list);
        if (err < 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, RAPL domain PKG not supported);
        }
    }
    if (amd_rapl_core_test())
    {
        DEBUG_PRINT(DEBUGLEV_INFO, Register Amd RAPL CORE domain);
        err = register_features(out, &amd_rapl_core_feature_list);
        if (err < 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, RAPL domain CORE not supported);
        }
    }
    if (amd_rapl_l3_test())
    {
        DEBUG_PRINT(DEBUGLEV_INFO, Register Amd RAPL L3 domain);
        err = register_features(out, &amd_rapl_l3_feature_list);
        if (err < 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, RAPL domain L3 not supported);
        }
    }
    return 0;
}
