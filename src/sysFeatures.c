/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures.c
 *
 *      Description:  Main interface to the sysFeatures component to get and set
 *                    hardware and OS features
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

#include <stdio.h>

#include <access.h>
#include <error.h>
#include <likwid.h>

#include <sysFeatures_amd.h>
#include <sysFeatures_common.h>
#include <sysFeatures_cpufreq.h>
#include <sysFeatures_intel.h>
#include <sysFeatures_linux_numa_balancing.h>
#ifdef LIKWID_WITH_NVMON
#include <sysFeatures_nvml.h>
#endif
#include <sysFeatures_types.h>
#include <sysFeatures_x86_tsc.h>

static _SysFeatureList feature_list = { 0, NULL, NULL };

int likwid_sysft_init(void)
{
    int err = 0;

    if (feature_list.num_features > 0 || feature_list.features) {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "likwid_sysft_init: Already initialized");
        return 0;
    }

    topology_init();
    CpuInfo_t cpuinfo = get_cpuInfo();
#if defined(__x86_64) || defined(__i386__)
    if (!HPMinitialized()) {
        err = HPMinit();
        if (err < 0) {
            ERROR_PRINT("Failed to initialize access to hardware registers");
            return err;
        }
    }
    if (cpuinfo->isIntel) {
        err = likwid_sysft_init_x86_intel(&feature_list);
        if (err < 0) {
            ERROR_PRINT("Failed to initialize SysFeatures for Intel architecture");
            return err;
        }
    } else {
        err = likwid_sysft_init_x86_amd(&feature_list);
        if (err < 0) {
            ERROR_PRINT("Failed to initialize SysFeatures for AMD architecture");
            return err;
        }
    }

    err = likwid_sysft_register_features(&feature_list, &likwid_sysft_x86_tsc_feature_list);
    if (err < 0) {
        ERROR_PRINT("Failed to init SysFeatures x86 tsc module");
        return err;
    }
#endif
    err = likwid_sysft_init_cpufreq(&feature_list);
    if (err < 0) {
        ERROR_PRINT("Failed to initialize SysFeatures cpufreq module");
        return err;
    }

    err = likwid_sysft_init_linux_numa_balancing(&feature_list);
    if (err < 0) {
        ERROR_PRINT("Failed to initialize SysFeatures numa_balancing module");
        return err;
    }
#ifdef LIKWID_WITH_NVMON
    err = likwid_sysft_init_nvml(&feature_list);
    if (err < 0)
        DEBUG_PRINT(DEBUGLEV_INFO, "Failed to initialize SysFeatures nvml module");
#endif

    DEBUG_PRINT(DEBUGLEV_DEVELOP, "Initialized %d features", feature_list.num_features);
    return 0;
}

static int get_feature_index(const char *name)
{
    if (!name) {
        return -EINVAL;
    }
    if (!strchr(name, '.')) {
        int out = -1;
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "Features name has no dot -> compare with name");
        for (int i = 0; i < feature_list.num_features; i++) {
            if (strcmp(name, feature_list.features[i].name) == 0) {
                if (out < 0) {
                    out = i;
                } else {
                    ERROR_PRINT("Feature name '%s' matches multiple features", name);
                    return -EINVAL;
                }
            }
        }
        if (out >= 0) {
            return out;
        }
    } else {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "Features name contains dot -> compare with category.name");
        for (int i = 0; i < feature_list.num_features; i++) {
            size_t featlen = strlen(feature_list.features[i].name) +
                             strlen(feature_list.features[i].category) + 2;
            char combined_name[featlen];
            snprintf(combined_name,
                featlen,
                "%s.%s",
                feature_list.features[i].category,
                feature_list.features[i].name);
            if (strcmp(name, combined_name) == 0) {
                return i;
            }
        }
    }
    ERROR_PRINT("SysFeatures modules does not provide a feature called %s", name);
    return -ENOTSUP;
}

int likwid_sysft_getByName(const char *name, const LikwidDevice_t device, char **value)
{
    int err        = 0;
    _SysFeature *f = NULL;
    if (!name || !device || !value) {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "Invalid inputs to sysFeatures_getByName");
        return -EINVAL;
    }
    if (device->type == DEVICE_TYPE_INVALID) {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "Invalid device type");
        return -EINVAL;
    }
    int idx = get_feature_index(name);
    if (idx < 0) {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "Failed to get index for %s", name);
        return -EINVAL;
    }
    f = &feature_list.features[idx];
    if (!f || !f->getter) {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "No feature %s or no support to read current state", name);
        return -EINVAL;
    }
    if (f->type != device->type) {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "Feature %s has a different type than device", name);
        return -EINVAL;
    }
    err = f->getter(device, value);
    return err;
}

int likwid_sysft_get(const LikwidSysFeature *feature, const LikwidDevice_t device, char **value)
{
    size_t len = strlen(feature->name) + strlen(feature->category) + 2;
    char real[len];
    snprintf(real, len, "%s.%s", feature->category, feature->name);
    return likwid_sysft_getByName(real, device, value);
}

int likwid_sysft_modifyByName(const char *name, const LikwidDevice_t device, const char *value)
{
    _SysFeature *f = NULL;
    if (!name || !device || !value) {
        return -EINVAL;
    }
    if (device->type == DEVICE_TYPE_INVALID) {
        return -EINVAL;
    }
    int idx = get_feature_index(name);
    if (idx < 0) {
        return -EINVAL;
    }
    f = &feature_list.features[idx];
    if (!f || !f->setter) {
        return -EPERM;
    }
    if (f->type != device->type) {
        return -ENODEV;
    }
    return f->setter(device, value);
}

int likwid_sysft_modify(
    const LikwidSysFeature *feature, const LikwidDevice_t device, const char *value)
{
    size_t len = strlen(feature->name) + strlen(feature->category) + 2;
    char real[len];
    snprintf(real, len, "%s.%s", feature->category, feature->name);
    return likwid_sysft_modifyByName(feature->name, device, value);
}

void likwid_sysft_finalize(void)
{
    if (feature_list.num_features > 0) {
        _free_feature_list(&feature_list);
    }
}

int likwid_sysft_list(LikwidSysFeatureList *list)
{
    if (!list) {
        return -EINVAL;
    }
    return likwid_sysft_internal_to_external_feature_list(&feature_list, list);
}

void likwid_sysft_list_return(LikwidSysFeatureList *list)
{
    if (!list || !list->features) {
        return;
    }
    likwid_sysft_free_feature_list(list);
    list->features     = NULL;
    list->num_features = 0;
}
