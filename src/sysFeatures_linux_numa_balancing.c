#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <sysFeatures_common.h>
#include <sysFeatures_linux_numa_balancing.h>
#include <bstrlib.h>
#include <bstrlib_helper.h>
#include <types.h>

static int numa_balancing_procfs_getter(const LikwidDevice_t device, char** value, const char* sysfsfile)
{
    int err = 0;
    if ((!device) || (!value) || (!sysfsfile) || (device->type != DEVICE_TYPE_NODE))
    {
        return -EINVAL;
    }
    bstring filename = bformat("/proc/sys/kernel/%s", sysfsfile);
#pragma GCC diagnostic ignored "-Wnonnull"
    if (!access(bdata(filename), R_OK))
    {
        bstring content = read_file(bdata(filename));
        btrimws(content);
        char *v = strdup(bdata(content));
        if (v)
        {
            free(*value);
            *value = v;
        }
        else
        {
            err = -ENOMEM;
        }
        bdestroy(content);
    }
    else
    {
        err = errno;
    }
    bdestroy(filename);
    return err;
}

static int numa_balancing_test(void)
{
    int err = access("/proc/sys/kernel/numa_balancing", F_OK);
    if (err < 0)
    {
        return -errno;
    }
    err = topology_init();
    if (err < 0)
    {
        return err;
    }
    err = numa_init();
    if (err < 0)
    {
        return err;
    }
    NumaTopology_t topo = get_numaTopology();
    if (topo->numberOfNodes > 1)
    {
        return 1;
    }
    DEBUG_PRINT(DEBUGLEV_INFO, NUMA balancing not available. System has only a single NUMA domain); return 0;
}

static int numa_balancing_state_getter(const LikwidDevice_t device, char** value)
{
    return numa_balancing_procfs_getter(device, value, "numa_balancing");
}

static int numa_balancing_scan_delay_getter(const LikwidDevice_t device, char** value)
{
    return numa_balancing_procfs_getter(device, value, "numa_balancing_scan_delay_ms");
}

static int numa_balancing_scan_period_min_getter(const LikwidDevice_t device, char** value)
{
    return numa_balancing_procfs_getter(device, value, "numa_balancing_scan_period_min_ms");
}

static int numa_balancing_scan_period_max_getter(const LikwidDevice_t device, char** value)
{
    return numa_balancing_procfs_getter(device, value, "numa_balancing_scan_period_max_ms");
}

static int numa_balancing_scan_size_getter(const LikwidDevice_t device, char** value)
{
    return numa_balancing_procfs_getter(device, value, "numa_balancing_scan_size_mb");
}

static _SysFeature numa_balancing_features[] = {
    {"numa_balancing", "os", "Current state of NUMA balancing", numa_balancing_state_getter, NULL, DEVICE_TYPE_NODE},
    {"numa_balancing_scan_delay_ms", "os", "Time between page scans", numa_balancing_scan_delay_getter, NULL, DEVICE_TYPE_NODE},
    {"numa_balancing_scan_period_min_ms", "os", "Minimal time for scan period", numa_balancing_scan_period_min_getter, NULL, DEVICE_TYPE_NODE},
    {"numa_balancing_scan_period_max_ms", "os", "Maximal time for scan period", numa_balancing_scan_period_max_getter, NULL, DEVICE_TYPE_NODE},
    {"numa_balancing_scan_size_mb", "os", "Scan size for NUMA balancing", numa_balancing_scan_size_getter, NULL, DEVICE_TYPE_NODE},
};

static const _SysFeatureList numa_balancing_feature_list = {
    .num_features = ARRAY_COUNT(numa_balancing_features),
    .tester = numa_balancing_test,
    .features = numa_balancing_features,
};

int likwid_sysft_init_linux_numa_balancing(_SysFeatureList* out)
{
    if (numa_balancing_test())
    {
        return likwid_sysft_register_features(out, &numa_balancing_feature_list);
    }
    return 0;
}
