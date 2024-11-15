/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_cpufreq.c
 *
 *      Description:  Interface to control frequencies for the sysFeatures component
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

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <topology.h>

#include <bstrlib.h>
#include <bstrlib_helper.h>
#include <sysFeatures_cpufreq.h>
#include <sysFeatures_common.h>


static int cpufreq_sysfs_getter(const LikwidDevice_t device, char** value, const char* sysfs_filename)
{
    int err = 0;
    if ((!device) || (!value) || (!sysfs_filename) || (device->type != DEVICE_TYPE_HWTHREAD))
    {
        return -EINVAL;
    }
    bstring filename = bformat("/sys/devices/system/cpu/cpu%d/cpufreq/%s", device->id.simple.id, sysfs_filename);
    // bdata() conditionally returns NULL and gcc access doesn't like that with access()
#pragma GCC diagnostic ignored "-Wnonnull"
    if (!access(bdata(filename), R_OK))
    {
        bstring content = read_file(bdata(filename));
        if (blength(content) > 0)
        {
            btrimws(content);
            char* v = realloc(*value, (blength(content)+1) * sizeof(char));
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

static int cpufreq_sysfs_setter(const LikwidDevice_t device, const char* value, const char* sysfs_filename)
{
    int err = 0;
    if ((!device) || (!value) || (!sysfs_filename) || (device->type != DEVICE_TYPE_HWTHREAD))
    {
        return -EINVAL;
    }
    bstring filename = bformat("/sys/devices/system/cpu/cpu%d/cpufreq/%s", device->id.simple.id, sysfs_filename);
    errno = 0;
    if (!access(bdata(filename), W_OK))
    {
        FILE* fp = NULL;
        errno = 0;
        fp = fopen(bdata(filename), "w");
        if (fp == NULL) {
            err = -errno;
            ERROR_PRINT("Failed to open file '%s' for writing: %s", bdata(filename), strerror(errno))
        }
        else
        {
            const size_t vallen = strlen(value);
            const size_t ret = fwrite(value, sizeof(char), vallen, fp);
            if (ret != (sizeof(char) * vallen))
            {
                ERROR_PRINT("Failed to open file '%s' for writing: %s", bdata(filename), strerror(errno))
            }
            fclose(fp);
        }
    }
    else
    {
        err = -errno;
    }
    bdestroy(filename);
    return err;
}


static int cpufreq_driver_test(const char* testgovernor)
{
    int err = 0;
    bstring btest = bfromcstr(testgovernor);
    topology_init();
    CpuTopology_t topo = get_cpuTopology();

    for (unsigned i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if (t->inCpuSet)
        {
            bstring filename = bformat("/sys/devices/system/cpu/cpu%d/cpufreq/scaling_driver", t->apicId);
            if (!access(bdata(filename), R_OK))
            {
                bstring content = read_file(bdata(filename));
                btrimws(content);
                err = (bstrncmp(content, btest, blength(btest)) == BSTR_OK);
                bdestroy(content);
            }
            bdestroy(filename);
            break;
        }
    }
    bdestroy(btest);
    return err;
}

/* ACPI CPUfreq driver */

static int cpufreq_acpi_cur_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_cur_freq");
}

static int cpufreq_acpi_min_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_min_freq");
}

static int cpufreq_acpi_max_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_max_freq");
}

static int cpufreq_acpi_avail_cpu_freqs_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_available_frequencies");
}

static int cpufreq_acpi_governor_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_governor");
}

static int cpufreq_acpi_governor_setter(const LikwidDevice_t device, const char* value)
{
    return cpufreq_sysfs_setter(device, value, "scaling_governor");
}

static int cpufreq_acpi_avail_governors_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_available_governors");
}

static int cpufreq_acpi_test(void)
{
    int err = cpufreq_driver_test("acpi_cpufreq");
    if (err != 0)
        return err;
    /* On AMD Genoa the string appears to be with a dash '-'. */
    return cpufreq_driver_test("acpi-cpufreq");
}

static _SysFeature cpufreq_acpi_features[] = {
    {"cur_cpu_freq", "cpu_freq", "Current CPU frequency", cpufreq_acpi_cur_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"min_cpu_freq", "cpu_freq", "Minimal CPU frequency", cpufreq_acpi_min_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"max_cpu_freq", "cpu_freq", "Maximal CPU frequency", cpufreq_acpi_max_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"avail_freqs", "cpu_freq", "Available CPU frequencies", cpufreq_acpi_avail_cpu_freqs_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"governor", "cpu_freq", "CPU frequency governor", cpufreq_acpi_governor_getter, cpufreq_acpi_governor_setter, DEVICE_TYPE_HWTHREAD},
    {"avail_governors", "cpu_freq", "Available CPU frequency governor", cpufreq_acpi_avail_governors_getter, NULL, DEVICE_TYPE_HWTHREAD},
};

static const _SysFeatureList cpufreq_acpi_feature_list = {
    .num_features = ARRAY_COUNT(cpufreq_acpi_features),
    .tester = cpufreq_acpi_test,
    .features = cpufreq_acpi_features,
};

/* Intel Pstate driver */

static int cpufreq_intel_pstate_base_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    int err = cpufreq_sysfs_getter(device, value, "base_frequency");
    if (err == 0) return 0;
    return cpufreq_sysfs_getter(device, value, "bios_limit");
}

static int cpufreq_intel_pstate_cur_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_cur_freq");
}

static int cpufreq_intel_pstate_min_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_min_freq");
}

static int cpufreq_intel_pstate_min_cpu_freq_setter(const LikwidDevice_t device, const char* value)
{
    return cpufreq_sysfs_setter(device, value, "scaling_min_freq");
}

static int cpufreq_intel_pstate_max_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_max_freq");
}

static int cpufreq_intel_pstate_max_cpu_freq_setter(const LikwidDevice_t device, const char* value)
{
    return cpufreq_sysfs_setter(device, value, "scaling_max_freq");
}

static int cpufreq_intel_pstate_governor_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_governor");
}

static int cpufreq_intel_pstate_governor_setter(const LikwidDevice_t device, const char* value)
{
    return cpufreq_sysfs_setter(device, value, "scaling_governor");
}

static int cpufreq_intel_pstate_avail_governors_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_available_governors");
}

static int cpufreq_intel_pstate_test(void)
{
    return cpufreq_driver_test("intel_pstate");
}

static _SysFeature cpufreq_pstate_features[] = {
    {"base_freq", "cpu_freq", "Base CPU frequency", cpufreq_intel_pstate_base_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"cur_cpu_freq", "cpu_freq", "Current CPU frequency", cpufreq_intel_pstate_cur_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"min_cpu_freq", "cpu_freq", "Minimal CPU frequency", cpufreq_intel_pstate_min_cpu_freq_getter, cpufreq_intel_pstate_min_cpu_freq_setter, DEVICE_TYPE_HWTHREAD},
    {"max_cpu_freq", "cpu_freq", "Maximal CPU frequency", cpufreq_intel_pstate_max_cpu_freq_getter, cpufreq_intel_pstate_max_cpu_freq_setter, DEVICE_TYPE_HWTHREAD},
    {"governor", "cpu_freq", "CPU frequency governor", cpufreq_intel_pstate_governor_getter, cpufreq_intel_pstate_governor_setter, DEVICE_TYPE_HWTHREAD},
    {"avail_governors", "cpu_freq", "Available CPU frequencies", cpufreq_intel_pstate_avail_governors_getter, NULL, DEVICE_TYPE_HWTHREAD},
};

static const _SysFeatureList cpufreq_pstate_feature_list = {
    .num_features = ARRAY_COUNT(cpufreq_pstate_features), 
    .tester = cpufreq_intel_pstate_test,
    .features = cpufreq_pstate_features,
};

/* intel_cpufreq driver */

static int cpufreq_intel_cpufreq_test(void)
{
    return cpufreq_driver_test("intel_cpufreq");
}

/* INFO: Most sysfs entries are the same as for the intel_pstate driver,
 * so they share the same getters. */
static _SysFeature cpufreq_intel_cpufreq_features[] = {
    {"cur_cpu_freq", "cpu_freq", "Current CPU frequency", cpufreq_intel_pstate_cur_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"min_cpu_freq", "cpu_freq", "Minimal CPU frequency", cpufreq_intel_pstate_min_cpu_freq_getter, cpufreq_intel_pstate_min_cpu_freq_setter, DEVICE_TYPE_HWTHREAD},
    {"max_cpu_freq", "cpu_freq", "Maximal CPU frequency", cpufreq_intel_pstate_max_cpu_freq_getter, cpufreq_intel_pstate_max_cpu_freq_setter, DEVICE_TYPE_HWTHREAD},
    {"governor", "cpu_freq", "CPU frequency governor", cpufreq_intel_pstate_governor_getter, cpufreq_intel_pstate_governor_setter, DEVICE_TYPE_HWTHREAD},
    {"avail_governors", "cpu_freq", "Available CPU frequencies", cpufreq_intel_pstate_avail_governors_getter, NULL, DEVICE_TYPE_HWTHREAD},
};

static const _SysFeatureList cpufreq_intel_cpufreq_feature_list = {
    .num_features = ARRAY_COUNT(cpufreq_intel_cpufreq_features),
    .tester = cpufreq_intel_cpufreq_test,
    .features = cpufreq_intel_cpufreq_features,
};

/* Energy Performance Preference */

static int cpufreq_epp_test(void)
{
    if ((!access("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference", R_OK)) &&
        (!access("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_available_preferences", R_OK)))
    {
        return 1;
    }
    return 0;
}

static int cpufreq_intel_pstate_epp_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "energy_performance_preference");
}

static int cpufreq_intel_pstate_avail_epps_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "energy_performance_available_preferences");
}

static _SysFeature cpufreq_epp_features[] = {
    {"epp", "cpu_freq", "Current energy performance preference", cpufreq_intel_pstate_epp_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"avail_epps", "cpu_freq", "Available energy performance preferences", cpufreq_intel_pstate_epp_getter, NULL, DEVICE_TYPE_HWTHREAD},
};

static const _SysFeatureList cpufreq_epp_feature_list = {
    .num_features = ARRAY_COUNT(cpufreq_epp_features),
    .tester = cpufreq_epp_test,
    .features = cpufreq_epp_features,
};

/* Scaling Driver (common to all) */

static int cpufreq_scaling_driver_getter(const LikwidDevice_t device, char **value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_driver");
}

static int cpufreq_scaling_driver_test(void)
{
    return access("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver", R_OK) == 0;
}

static _SysFeature cpufreq_scaling_driver_features[] = {
    {"scaling_driver", "cpu_freq", "Kernel Scaling Driver", cpufreq_scaling_driver_getter, NULL, DEVICE_TYPE_HWTHREAD},
};

static const _SysFeatureList cpufreq_scaling_driver_feature_list = {
    .num_features = ARRAY_COUNT(cpufreq_scaling_driver_features),
    .tester = cpufreq_scaling_driver_test,
    .features = cpufreq_scaling_driver_features,
};

/* Init function */

int likwid_sysft_init_cpufreq(_SysFeatureList* out)
{
    int err = 0;
    if (cpufreq_intel_pstate_test())
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Registering Intel Pstate knobs for cpufreq)
        err = likwid_sysft_register_features(out, &cpufreq_pstate_feature_list);
        if (err < 0)
        {
            return err;
        }
    }
    else if (cpufreq_intel_cpufreq_test())
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Registering Intel Cpufreq knobs for cpufreq)
        err = likwid_sysft_register_features(out, &cpufreq_intel_cpufreq_feature_list);
        if (err < 0)
        {
            return err;
        }
    }
    else if (cpufreq_acpi_test())
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Registering ACPI cpufreq knobs for cpufreq)
        likwid_sysft_register_features(out, &cpufreq_acpi_feature_list);
        if (err < 0)
        {
            return err;
        }
    }

    if (cpufreq_epp_test())
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Registering Energy Performance Preference knobs for cpufreq)
        err = likwid_sysft_register_features(out, &cpufreq_epp_feature_list);
        if (err < 0)
        {
            return err;
        }
    }
    if (cpufreq_scaling_driver_test())
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Registering Scaling Driver knobs for cpufreq)
        err = likwid_sysft_register_features(out, &cpufreq_scaling_driver_feature_list);
        if (err < 0)
        {
            return err;
        }
    }
    return 0;
}
