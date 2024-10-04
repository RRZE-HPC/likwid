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



int numa_balancing_procfs_getter(const LikwidDevice_t device, char** value, const char* sysfsfile)
{
    int err = 0;
    if ((!device) || (!value) || (!sysfsfile) || (device->type != DEVICE_TYPE_NODE))
    {
        return -EINVAL;
    }
    bstring filename = bformat("/proc/sys/kernel/%s", sysfsfile);
    if (!access(bdata(filename), R_OK))
    {
        bstring content = read_file(bdata(filename));
        if (blength(content) > 0)
        {
            btrimws(content);
            char* v = malloc((blength(content)+1) * sizeof(char));
            if (v)
            {
                strncpy(v, bdata(content), blength(content));
                v[blength(content)] = '\0';
                *value = v;
            }
            else
            {
                err = -ENOMEM;
            }
            bdestroy(content);
        }
    }
    else
    {
        err = errno;
    }
    bdestroy(filename);
    return err;
}

int numa_balancing_test()
{
    int err = access("/proc/sys/kernel/numa_balancing", F_OK);
    if (err < 0)
    {
        return -errno;
    }
    if (err == 0)
    {
        topology_init();
        numa_init();
        NumaTopology_t topo = get_numaTopology();
        if (topo->numberOfNodes > 1)
        {
            return 1;
        }
        else
        {
            DEBUG_PRINT(DEBUGLEV_INFO, NUMA balancing not available. System has only a single NUMA domain);
        }
    }
    return 0;
}

int numa_balancing_state_getter(const LikwidDevice_t device, char** value)
{
    return numa_balancing_procfs_getter(device, value, "numa_balancing");
}
int numa_balancing_scan_delay_getter(const LikwidDevice_t device, char** value)
{
    return numa_balancing_procfs_getter(device, value, "numa_balancing_scan_delay_ms");
}
int numa_balancing_scan_period_min_getter(const LikwidDevice_t device, char** value)
{
    return numa_balancing_procfs_getter(device, value, "numa_balancing_scan_period_min_ms");
}
int numa_balancing_scan_period_max_getter(const LikwidDevice_t device, char** value)
{
    return numa_balancing_procfs_getter(device, value, "numa_balancing_scan_period_max_ms");
}
int numa_balancing_scan_size_getter(const LikwidDevice_t device, char** value)
{
    return numa_balancing_procfs_getter(device, value, "numa_balancing_scan_size_mb");
}


int sysFeatures_init_linux_numa_balancing(_SysFeatureList* out)
{
    if (numa_balancing_test())
    {
        return register_features(out, &numa_balancing_feature_list);
    }
    return 0;
}
