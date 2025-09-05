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

#include "error_ng.h"
#include "debug.h"

static cerr_t cpufreq_sysfs_getter(const LikwidDevice_t device, char** value, const char* sysfs_filename)
{
    assert(device->type == DEVICE_TYPE_HWTHREAD);

    char filename[512];
    snprintf(filename, sizeof(filename), "/sys/devices/system/cpu/cpu%d/cpufreq/%s", device->id.simple.id, sysfs_filename);

    if (access(filename, R_OK) != 0)
        return ERROR_SET_ERRNO("access(%s, R_OK) failed", filename);

    bstring content = read_file(filename);
    btrimws(content);
    char *v = strdup(bdata(content));
    bdestroy(content);

    if (!v)
        return ERROR_SET_ERRNO("strdup failed");

    free(*value);
    *value = v;

    return NULL;
}

static cerr_t cpufreq_sysfs_setter(const LikwidDevice_t device, const char* value, const char* sysfs_filename)
{
    assert(device->type == DEVICE_TYPE_HWTHREAD);

    char filename[512];
    snprintf(filename, sizeof(filename), "/sys/devices/system/cpu/cpu%d/cpufreq/%s", device->id.simple.id, sysfs_filename);

    FILE *fp = fopen(filename, "w");
    if (!fp)
        return ERROR_SET_ERRNO("fopen(%s, w) failed", filename);

    const size_t vallen = strlen(value);
    const size_t ret = fwrite(value, sizeof(char), vallen, fp);
    fclose(fp);

    if (ret != sizeof(*value) * vallen)
        return ERROR_SET("fwrite failed to write all bytes");

    return NULL;
}

static cerr_t cpufreq_driver_test(bool *ok, const char* testgovernor)
{
    CpuTopology_t topo = get_cpuTopology();

    for (unsigned i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if (t->inCpuSet)
        {
            char filename[512];
            snprintf(filename, sizeof(filename), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_driver", t->apicId);

            if (access(filename, R_OK) != 0)
                return ERROR_SET_ERRNO("access(%s, R_OK) failed", filename);

            bstring content = read_file(filename);
            btrimws(content);
            *ok = strcmp(bdata(content), testgovernor) == 0;
            bdestroy(content);
            return NULL;
        }
    }

    *ok = false;
    return NULL;
}

/* ACPI CPUfreq driver */

static cerr_t cpufreq_acpi_cur_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_cur_freq");
}

static cerr_t cpufreq_acpi_min_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_min_freq");
}

static cerr_t cpufreq_acpi_max_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_max_freq");
}

static cerr_t cpufreq_acpi_avail_cpu_freqs_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_available_frequencies");
}

static cerr_t cpufreq_acpi_governor_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_governor");
}

// TODO I don't remember why we don't reference this anymore below.
// Was there a problem with it?
//static cerr_t cpufreq_acpi_governor_setter(const LikwidDevice_t device, const char* value)
//{
//    return cpufreq_sysfs_setter(device, value, "scaling_governor");
//}

static cerr_t cpufreq_acpi_avail_governors_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_available_governors");
}

static cerr_t cpufreq_acpi_test(bool *ok)
{
    if (cpufreq_driver_test(ok, "acpi_cpufreq"))
        return ERROR_APPEND("cpufreq_driver_test error");
    if (*ok)
        return NULL;

    /* On AMD Genoa the string appears to be with a dash '-'. */
    return cpufreq_driver_test(ok, "acpi-cpufreq");
}

static _SysFeature cpufreq_acpi_features[] = {
    {"cur_cpu_freq", "cpu_freq", "Current CPU frequency", cpufreq_acpi_cur_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, "kHz"},
    {"min_cpu_freq", "cpu_freq", "Minimal CPU frequency", cpufreq_acpi_min_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, "kHz"},
    {"max_cpu_freq", "cpu_freq", "Maximal CPU frequency", cpufreq_acpi_max_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, "kHz"},
    {"avail_freqs", "cpu_freq", "Available CPU frequencies", cpufreq_acpi_avail_cpu_freqs_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, NULL},
    {"governor", "cpu_freq", "CPU frequency governor", cpufreq_acpi_governor_getter, /*cpufreq_acpi_governor_setter*/ NULL, DEVICE_TYPE_HWTHREAD, NULL, NULL},
    {"avail_governors", "cpu_freq", "Available CPU frequency governors", cpufreq_acpi_avail_governors_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, NULL},
};

static const _SysFeatureList cpufreq_acpi_feature_list = {
    .num_features = ARRAY_COUNT(cpufreq_acpi_features),
    .tester = cpufreq_acpi_test,
    .features = cpufreq_acpi_features,
};

/* Intel Pstate driver */

static cerr_t cpufreq_intel_pstate_base_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    if (cpufreq_sysfs_getter(device, value, "base_frequency") == NULL)
        return NULL;
    return cpufreq_sysfs_getter(device, value, "bios_limit");
}

static cerr_t cpufreq_intel_pstate_cur_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_cur_freq");
}

static cerr_t cpufreq_intel_pstate_min_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_min_freq");
}

static cerr_t cpufreq_intel_pstate_min_cpu_freq_setter(const LikwidDevice_t device, const char* value)
{
    return cpufreq_sysfs_setter(device, value, "scaling_min_freq");
}

static cerr_t cpufreq_intel_pstate_max_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_max_freq");
}

static cerr_t cpufreq_intel_pstate_max_cpu_freq_setter(const LikwidDevice_t device, const char* value)
{
    return cpufreq_sysfs_setter(device, value, "scaling_max_freq");
}

static cerr_t cpufreq_intel_pstate_governor_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_governor");
}

static cerr_t cpufreq_intel_pstate_governor_setter(const LikwidDevice_t device, const char* value)
{
    return cpufreq_sysfs_setter(device, value, "scaling_governor");
}

static cerr_t cpufreq_intel_pstate_avail_governors_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_available_governors");
}

static cerr_t cpufreq_intel_pstate_test(bool *ok)
{
    return cpufreq_driver_test(ok, "intel_pstate");
}

static _SysFeature cpufreq_pstate_features[] = {
    {"base_freq", "cpu_freq", "Base CPU frequency", cpufreq_intel_pstate_base_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, "kHz"},
    {"cur_cpu_freq", "cpu_freq", "Current CPU frequency", cpufreq_intel_pstate_cur_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, "kHz"},
    {"min_cpu_freq", "cpu_freq", "Minimal CPU frequency", cpufreq_intel_pstate_min_cpu_freq_getter, cpufreq_intel_pstate_min_cpu_freq_setter, DEVICE_TYPE_HWTHREAD, NULL, "kHz"},
    {"max_cpu_freq", "cpu_freq", "Maximal CPU frequency", cpufreq_intel_pstate_max_cpu_freq_getter, cpufreq_intel_pstate_max_cpu_freq_setter, DEVICE_TYPE_HWTHREAD, NULL, "kHz"},
    {"governor", "cpu_freq", "CPU frequency governor", cpufreq_intel_pstate_governor_getter, cpufreq_intel_pstate_governor_setter, DEVICE_TYPE_HWTHREAD, NULL, NULL},
    {"avail_governors", "cpu_freq", "Available CPU frequency governors", cpufreq_intel_pstate_avail_governors_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, NULL},
};

static const _SysFeatureList cpufreq_pstate_feature_list = {
    .num_features = ARRAY_COUNT(cpufreq_pstate_features), 
    .tester = cpufreq_intel_pstate_test,
    .features = cpufreq_pstate_features,
};

/* intel_cpufreq driver */

static cerr_t cpufreq_intel_cpufreq_test(bool *ok)
{
    return cpufreq_driver_test(ok, "intel_cpufreq");
}

/* INFO: Most sysfs entries are the same as for the intel_pstate driver,
 * so they share the same getters. */
static _SysFeature cpufreq_intel_cpufreq_features[] = {
    {"cur_cpu_freq", "cpu_freq", "Current CPU frequency", cpufreq_intel_pstate_cur_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, "kHz"},
    {"min_cpu_freq", "cpu_freq", "Minimal CPU frequency", cpufreq_intel_pstate_min_cpu_freq_getter, cpufreq_intel_pstate_min_cpu_freq_setter, DEVICE_TYPE_HWTHREAD, NULL, "kHz"},
    {"max_cpu_freq", "cpu_freq", "Maximal CPU frequency", cpufreq_intel_pstate_max_cpu_freq_getter, cpufreq_intel_pstate_max_cpu_freq_setter, DEVICE_TYPE_HWTHREAD, NULL, "kHz"},
    {"governor", "cpu_freq", "CPU frequency governor", cpufreq_intel_pstate_governor_getter, cpufreq_intel_pstate_governor_setter, DEVICE_TYPE_HWTHREAD, NULL, NULL},
    {"avail_governors", "cpu_freq", "Available CPU frequency governors", cpufreq_intel_pstate_avail_governors_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, NULL},
};

static const _SysFeatureList cpufreq_intel_cpufreq_feature_list = {
    .num_features = ARRAY_COUNT(cpufreq_intel_cpufreq_features),
    .tester = cpufreq_intel_cpufreq_test,
    .features = cpufreq_intel_cpufreq_features,
};

/* cppc driver */

static cerr_t cpufreq_cppc_boost_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "boost");
}

static cerr_t cpufreq_cppc_boost_setter(const LikwidDevice_t device, const char* value)
{
    return cpufreq_sysfs_setter(device, value, "boost");
}

static cerr_t cpufreq_cppc_test(bool *ok)
{
    return cpufreq_driver_test(ok, "cppc_cpufreq");
}

static _SysFeature cpufreq_cppc_features[] = {
    {"cur_cpu_freq", "cpu_freq", "Current CPU frequency", cpufreq_intel_pstate_cur_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, "kHz"},
    {"min_cpu_freq", "cpu_freq", "Minimal CPU frequency", cpufreq_intel_pstate_min_cpu_freq_getter, cpufreq_intel_pstate_min_cpu_freq_setter, DEVICE_TYPE_HWTHREAD, NULL, "kHz"},
    {"max_cpu_freq", "cpu_freq", "Maximal CPU frequency", cpufreq_intel_pstate_max_cpu_freq_getter, cpufreq_intel_pstate_max_cpu_freq_setter, DEVICE_TYPE_HWTHREAD, NULL, "kHz"},
    {"boost", "cpu_freq", "Turbo boost", cpufreq_cppc_boost_getter, cpufreq_cppc_boost_setter, DEVICE_TYPE_HWTHREAD, NULL, NULL},
    {"governor", "cpu_freq", "CPU frequency governor", cpufreq_intel_pstate_governor_getter, cpufreq_intel_pstate_governor_setter, DEVICE_TYPE_HWTHREAD, NULL, NULL},
    {"avail_governors", "cpu_freq", "Available CPU frequency governors", cpufreq_intel_pstate_avail_governors_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, NULL},
};

static const _SysFeatureList cpufreq_cppc_feature_list = {
    .num_features = ARRAY_COUNT(cpufreq_cppc_features),
    .tester = cpufreq_cppc_test,
    .features = cpufreq_cppc_features,
};

/* apple-cpufreq driver */

static cerr_t cpufreq_apple_cpufreq_test(bool *ok)
{
    return cpufreq_driver_test(ok, "apple-cpufreq");
}

static _SysFeature cpufreq_apple_cpufreq_features[] = {
    {"cur_cpu_freq", "cpu_freq", "Current CPU frequency", cpufreq_intel_pstate_cur_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, "kHz"},
    {"min_cpu_freq", "cpu_freq", "Minimal CPU frequency", cpufreq_intel_pstate_min_cpu_freq_getter, cpufreq_intel_pstate_min_cpu_freq_setter, DEVICE_TYPE_HWTHREAD, NULL, "kHz"},
    {"max_cpu_freq", "cpu_freq", "Maximal CPU frequency", cpufreq_intel_pstate_max_cpu_freq_getter, cpufreq_intel_pstate_max_cpu_freq_setter, DEVICE_TYPE_HWTHREAD, NULL, "kHz"},
    {"avail_freqs", "cpu_freq", "Available CPU frequencies", cpufreq_acpi_avail_cpu_freqs_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, NULL},
    {"governor", "cpu_freq", "CPU frequency governor", cpufreq_intel_pstate_governor_getter, cpufreq_intel_pstate_governor_setter, DEVICE_TYPE_HWTHREAD, NULL, NULL},
    {"avail_governors", "cpu_freq", "Available CPU frequency governors", cpufreq_intel_pstate_avail_governors_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, NULL},
};

static const _SysFeatureList cpufreq_apple_cpufreq_feature_list = {
    .num_features = ARRAY_COUNT(cpufreq_apple_cpufreq_features),
    .tester = cpufreq_apple_cpufreq_test,
    .features = cpufreq_apple_cpufreq_features,
};


/* Energy Performance Preference */

static cerr_t cpufreq_epp_test(bool *ok)
{
    if ((!access("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference", R_OK)) &&
        (!access("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_available_preferences", R_OK)))
    {
        *ok = true;
    } else {
        *ok = false;
    }
    return NULL;
}

static cerr_t cpufreq_intel_pstate_epp_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "energy_performance_preference");
}

static cerr_t cpufreq_intel_pstate_avail_epps_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "energy_performance_available_preferences");
}

static _SysFeature cpufreq_epp_features[] = {
    {"epp", "cpu_freq", "Current energy performance preference", cpufreq_intel_pstate_epp_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, NULL},
    {"avail_epps", "cpu_freq", "Available energy performance preferences", cpufreq_intel_pstate_avail_epps_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, NULL},
};

static const _SysFeatureList cpufreq_epp_feature_list = {
    .num_features = ARRAY_COUNT(cpufreq_epp_features),
    .tester = cpufreq_epp_test,
    .features = cpufreq_epp_features,
};

/* Scaling Driver (common to all) */

static cerr_t cpufreq_scaling_driver_getter(const LikwidDevice_t device, char **value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_driver");
}

static cerr_t cpufreq_scaling_driver_test(bool *ok)
{
    *ok = access("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver", R_OK);
    return NULL;
}

static _SysFeature cpufreq_scaling_driver_features[] = {
    {"scaling_driver", "cpu_freq", "Kernel Scaling Driver", cpufreq_scaling_driver_getter, NULL, DEVICE_TYPE_HWTHREAD, NULL, NULL},
};

static const _SysFeatureList cpufreq_scaling_driver_feature_list = {
    .num_features = ARRAY_COUNT(cpufreq_scaling_driver_features),
    .tester = cpufreq_scaling_driver_test,
    .features = cpufreq_scaling_driver_features,
};

/* Init function */

cerr_t likwid_sysft_init_cpufreq(_SysFeatureList* out)
{
    bool intel_pstate_avail;
    bool intel_cpufreq_avail;
    bool acpi_avail;
    bool cppc_avail;
    bool apple_avail;

    if (cpufreq_intel_pstate_test(&intel_pstate_avail)) {
        return ERROR_APPEND("cpufreq_intel_pstate_test error");
    } else if (intel_pstate_avail) {
        PRINT_INFO("Registering Intel Pstate knobs for cpufreq");

        if (likwid_sysft_register_features(out, &cpufreq_pstate_feature_list))
            return ERROR_APPEND("likwid_sysft_register_features failed");
    } else if (cpufreq_intel_cpufreq_test(&intel_cpufreq_avail)) {
        return ERROR_APPEND("cpufreq_intel_cpufreq_test error");
    } else if (intel_cpufreq_avail) {
        PRINT_INFO("Registering Intel Cpufreq knobs for cpufreq");

        if (likwid_sysft_register_features(out, &cpufreq_intel_cpufreq_feature_list))
            return ERROR_APPEND("likwid_sysft_register_features failed");
    } else if (cpufreq_acpi_test(&acpi_avail)) {
        return ERROR_APPEND("cpufreq_acpi_test error");
    } else if (acpi_avail) {
        PRINT_INFO("Registering ACPI cpufreq knobs for cpufreq");

        if (likwid_sysft_register_features(out, &cpufreq_acpi_feature_list))
            return ERROR_APPEND("likwid_sysft_register_features failed");
    } else if (cpufreq_cppc_test(&cppc_avail)) {
        return ERROR_APPEND("cpufreq_acpi_test error");
    } else if (cppc_avail) {
        PRINT_INFO("Registering CPPC cpufreq knobs for cpufreq");

        if (likwid_sysft_register_features(out, &cpufreq_cppc_feature_list))
            return ERROR_APPEND("likwid_sysft_register_features failed");
    } else if (cpufreq_apple_cpufreq_test(&apple_avail)) {
        return ERROR_APPEND("cpufreq_apple_cpufreq_test error");
    } else if (apple_avail) {
        PRINT_DEBUG("Registering Apple cpufreq knobs for cpufreq");

        if (likwid_sysft_register_features(out, &cpufreq_apple_cpufreq_feature_list))
            return ERROR_APPEND("likwid_sysft_register_features failed");
    }

    bool epp_avail;
    if (cpufreq_epp_test(&epp_avail))
        return ERROR_APPEND("cpufreq_epp_test error");
    
    if (epp_avail) {
        PRINT_DEBUG("Registering Energy Performance Preference knobs for cpufreq");
        if (likwid_sysft_register_features(out, &cpufreq_epp_feature_list))
            return ERROR_APPEND("likwid_sysft_register_features failed");
    }

    bool scaling_driver_avail;
    if (cpufreq_scaling_driver_test(&scaling_driver_avail))
        return ERROR_APPEND("cpufreq_scaling_driver_test error");

    if (scaling_driver_avail) {
        PRINT_DEBUG("Registering Scaling Driver knobs for cpufreq");
        if (likwid_sysft_register_features(out, &cpufreq_scaling_driver_feature_list))
            return ERROR_APPEND("likwid_sysft_register_features failed");
    }

    return NULL;
}
