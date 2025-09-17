/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_linux_numa_balancing.c
 *
 *      Description:  Interface to control NUMA balancing settings for the sysFeatures
 *                    component
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Authors:  Thomas Gruber (tg), thomas.roehl@googlemail.com
 *                Michael Panzlaff, michael.panzlaff@fau.de
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */

#include <sysFeatures_linux_numa_balancing.h>

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <sysFeatures_common.h>
#include <bstrlib.h>
#include <bstrlib_helper.h>
#include <types.h>

#include "debug.h"

static cerr_t numa_balancing_procfs_getter(const LikwidDevice_t device, char** value, const char* sysfsfile)
{
    assert(device->type == DEVICE_TYPE_NODE);
    assert(sysfsfile);

    char filename[512];
    snprintf(filename, sizeof(filename), "/proc/sys/kernel/%s", sysfsfile);

    PRINT_DEBUG("Reading file %s", filename);
    if (access(filename, R_OK))
        return ERROR_SET_ERRNO("access(%s, R_OK) failed", filename);

    bstring content = read_file(filename);
    btrimws(content);
    cerr_t err = likwid_sysft_copystr(bdata(content), value);
    bdestroy(content);
    return err;
}

static cerr_t numa_balancing_procfs_tester(bool *ok, const char* sysfsfile)
{
    assert(sysfsfile);

    char filename[512];
    snprintf(filename, sizeof(filename), "/proc/sys/kernel/%s", sysfsfile);

    *ok = access(filename, R_OK) == 0;
    return NULL;
}

static cerr_t numa_balancing_test(bool *ok)
{
    if (access("/proc/sys/kernel/numa_balancing", F_OK) < 0)
        return ERROR_SET_ERRNO("access(.../numa_balancing) failed");

    int err = numa_init();
    if (err < 0)
        return ERROR_SET_LWERR(err, "numa_init failed");

    NumaTopology_t topo = get_numaTopology();
    *ok = topo->numberOfNodes > 1;
    if (!*ok)
        PRINT_INFO("NUMA balancing not available. System has only a single NUMA domain");

    return NULL;
}

static cerr_t numa_balancing_state_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(numa_balancing_procfs_getter(device, value, "numa_balancing"));
}

static cerr_t numa_balancing_scan_delay_test(bool *ok)
{
    return ERROR_WRAP_CALL(numa_balancing_procfs_tester(ok, "numa_balancing_scan_delay_ms"));
}

static cerr_t numa_balancing_scan_delay_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(numa_balancing_procfs_getter(device, value, "numa_balancing_scan_delay_ms"));
}

static cerr_t numa_balancing_scan_period_min_test(bool *ok)
{
    return ERROR_WRAP_CALL(numa_balancing_procfs_tester(ok, "numa_balancing_scan_period_min_ms"));
}

static cerr_t numa_balancing_scan_period_min_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(numa_balancing_procfs_getter(device, value, "numa_balancing_scan_period_min_ms"));
}

static cerr_t numa_balancing_scan_period_max_test(bool *ok)
{
    return ERROR_WRAP_CALL(numa_balancing_procfs_tester(ok, "numa_balancing_scan_period_max_ms"));
}

static cerr_t numa_balancing_scan_period_max_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(numa_balancing_procfs_getter(device, value, "numa_balancing_scan_period_max_ms"));
}

static cerr_t numa_balancing_scan_size_test(bool *ok)
{
    return ERROR_WRAP_CALL(numa_balancing_procfs_tester(ok, "numa_balancing_scan_size_mb"));
}

static cerr_t numa_balancing_scan_size_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(numa_balancing_procfs_getter(device, value, "numa_balancing_scan_size_mb"));
}

static cerr_t numa_balancing_rate_limit_test(bool *ok)
{
    return ERROR_WRAP_CALL(numa_balancing_procfs_tester(ok, "numa_balancing_promote_rate_limit_MBps"));
}

static cerr_t numa_balancing_rate_limit_getter(const LikwidDevice_t device, char** value)
{
    return ERROR_WRAP_CALL(numa_balancing_procfs_getter(device, value, "numa_balancing_promote_rate_limit_MBps"));
}

static _SysFeature numa_balancing_features[] = {
    {"numa_balancing", "os", "Current state of NUMA balancing", numa_balancing_state_getter, NULL, DEVICE_TYPE_NODE, NULL, NULL},
    {"numa_balancing_scan_delay", "os", "Time between page scans", numa_balancing_scan_delay_getter, NULL, DEVICE_TYPE_NODE, numa_balancing_scan_delay_test, "ms"},
    {"numa_balancing_scan_period_min", "os", "Minimal time for scan period", numa_balancing_scan_period_min_getter, NULL, DEVICE_TYPE_NODE, numa_balancing_scan_period_min_test, "ms"},
    {"numa_balancing_scan_period_max", "os", "Maximal time for scan period", numa_balancing_scan_period_max_getter, NULL, DEVICE_TYPE_NODE, numa_balancing_scan_period_max_test, "ms"},
    {"numa_balancing_scan_size", "os", "Scan size for NUMA balancing", numa_balancing_scan_size_getter, NULL, DEVICE_TYPE_NODE, numa_balancing_scan_size_test, "MB/s"},
    {"numa_balancing_promote_rate_limit", "os", "Rate limit for NUMA balancing", numa_balancing_rate_limit_getter, NULL, DEVICE_TYPE_NODE, numa_balancing_rate_limit_test, "MB/s"},
};

static const _SysFeatureList numa_balancing_feature_list = {
    .num_features = ARRAY_COUNT(numa_balancing_features),
    .tester = numa_balancing_test,
    .features = numa_balancing_features,
};

cerr_t likwid_sysft_init_linux_numa_balancing(_SysFeatureList* out)
{
    bool numa_avail;
    if (numa_balancing_test(&numa_avail))
        return ERROR_WRAP();

    if (!numa_avail)
        return NULL;

    PRINT_INFO("Register OS NUMA balancing");
    return ERROR_WRAP_CALL(likwid_sysft_register_features(out, &numa_balancing_feature_list));
}
