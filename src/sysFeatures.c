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
#include "sysFeatures.h"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>

#include <topology.h>
#include <access.h>
#include <error.h>
#include <likwid.h>

#include <sysFeatures_types.h>
#include <sysFeatures_common.h>
#include <sysFeatures_amd.h>
#include <sysFeatures_intel.h>
#include <sysFeatures_cpufreq.h>
#include <sysFeatures_linux_numa_balancing.h>
#include <sysFeatures_nvml.h>

#include "error_ng.h"
#include "debug.h"

static _SysFeatureList _feature_list = {0, NULL, NULL};


static cerr_t get_device_access(LikwidDevice_t device)
{
    if (device->type == DEVICE_TYPE_INVALID)
        return ERROR_SET("Unable to init device access to invalid device type");

    /* This function is somewhat redundant to the initialization,
     * which is implicitly called in every getter/settter. */
    CpuTopology_t topo = get_cpuTopology();
    int hwt = -1;

    switch (device->type) {
        case DEVICE_TYPE_HWTHREAD:
            if (device->id.simple.id >= 0 && device->id.simple.id < (int)topo->numHWThreads)
            {
                hwt = device->id.simple.id;
            }
            break;
        case DEVICE_TYPE_CORE:
            for (unsigned i = 0; i < topo->numHWThreads; i++)
            {
                HWThread* t = &topo->threadPool[i];
                if (t->inCpuSet == 1 && device->id.simple.id == (int)t->coreId)
                {
                    hwt = t->apicId;
                    break;
                }
            }
            break;
        case DEVICE_TYPE_NODE:
            for (unsigned i = 0; i < topo->numHWThreads; i++)
            {
                HWThread* t = &topo->threadPool[i];
                if (t->inCpuSet == 1)
                {
                    hwt = t->apicId;
                    break;
                }
            }
            break;
        case DEVICE_TYPE_SOCKET:
            for (unsigned i = 0; i < topo->numHWThreads; i++)
            {
                HWThread* t = &topo->threadPool[i];
                if (t->inCpuSet == 1 && device->id.simple.id == (int)t->packageId)
                {
                    hwt = t->apicId;
                    break;
                }
            }
            break;
#ifdef LIKWID_WITH_NVMON
        case DEVICE_TYPE_NVIDIA_GPU:
            return NULL;
#endif
#ifdef LIKWID_WITH_ROCMON
        case DEVICE_TYPE_AMD_GPU:
            return NULL;
#endif
        default:
            return ERROR_SET("Unimplemented device type: %d\n", device->type);
    }

    if (hwt >= 0) {
#if defined(__x86_64) || defined(__i386__)
        int err = HPMaddThread(hwt);
        if (err < 0)
            return ERROR_SET_LWERR(err, "HPMaddThread failed");
#endif
        return NULL;
    }

    return ERROR_SET("Unable to get device access for hwt %d", hwt);
}


cerr_t likwid_sysft_init(void)
{
    if (_feature_list.num_features > 0 || _feature_list.features)
    {
        PRINT_DEBUG("likwid_sysft_init: Already initialized");
        return NULL;
    }

    int err = topology_init();
    if (err < 0)
        return ERROR_SET_LWERR(err, "topology_init failed");

    CpuInfo_t cpuinfo = get_cpuInfo();
#if defined(__x86_64) || defined(__i386__)
    if (!HPMinitialized()) {
        int err = HPMinit();
        if (err < 0)
            return ERROR_SET("Failed to initialize access to hardware registers");
    }

    if (cpuinfo->isIntel) {
        if (likwid_sysft_init_x86_intel(&_feature_list))
            return ERROR_APPEND("Failed to initialize SysFeatures for Intel architecture");
    } else {
        if (likwid_sysft_init_x86_amd(&_feature_list))
            return ERROR_APPEND("Failed to initialize SysFeatures for AMD architecture");
    }
#endif

    if (likwid_sysft_init_cpufreq(&_feature_list))
        return ERROR_APPEND("Failed to initialize SysFeatures cpufreq module");

    if (likwid_sysft_init_linux_numa_balancing(&_feature_list))
        return ERROR_APPEND("Failed to initialize SysFeatures numa_balancing module");

#ifdef LIKWID_WITH_NVMON
    if (likwid_sysft_init_nvml(&_feature_list))
        PRINT_INFO_ERR("Failed to initialize SysFeatures nvml module");
#endif
    
    PRINT_DEBUG("Initialized %d features", _feature_list.num_features);
    return NULL;
}

static cerr_t get_feature_index(size_t *index, const char* name)
{
    if (!strchr(name, '.'))
    {
        bool found = false;
        size_t found_index = 0;
        PRINT_DEBUG("Features name has no dot -> compare with name");

        for (size_t i = 0; i < (size_t)_feature_list.num_features; i++)
        {
            if (strcmp(name, _feature_list.features[i].name) == 0)
            {
                if (!found)
                    found_index = i;
                else
                    return ERROR_SET("Feature name '%s' matches multiple features", name);
            }
        }

        if (found) {
            *index = found_index;
            return NULL;
        }
    }
    else
    {
        PRINT_DEBUG("Features name contains dot -> compare with category.name");
        for (size_t i = 0; i < (size_t)_feature_list.num_features; i++)
        {
            const size_t featlen = strlen(_feature_list.features[i].name) + strlen(_feature_list.features[i].category) + 2;
            char combined_name[featlen];
            snprintf(combined_name, featlen, "%s.%s", _feature_list.features[i].category, _feature_list.features[i].name);
            if (strcmp(name, combined_name) == 0) {
                *index = i;
                return NULL;
            }
        }
    }

    return ERROR_SET("SysFeatures modules does not provide a feature called '%s'", name);
}

cerr_t likwid_sysft_getByName(const char* name, const LikwidDevice_t device, char** value)
{
    if (!name || !device || !value)
        return ERROR_SET("Invalid inputs to sysFeatures_getByName");

    if (device->type == DEVICE_TYPE_INVALID)
        return ERROR_SET("Invalid device type");

    size_t idx;
    if (get_feature_index(&idx, name))
        return ERROR_APPEND("Failed to get index for %s", name);

    _SysFeature *f = &_feature_list.features[idx];
    if (!f || !f->getter)
        return ERROR_SET("No feature %s or no support to read current state", name);

    if (f->type != device->type)
        return ERROR_SET("Feature %s has a different type than device", name);

    if (get_device_access(device))
        return ERROR_APPEND("Failed to get access to device");

    if (f->getter(device, value))
        return ERROR_APPEND("Failed to run sysfeature getter");
    return NULL;
}

cerr_t likwid_sysft_get(const LikwidSysFeature* feature, const LikwidDevice_t device, char** value)
{
    const size_t len = strlen(feature->name) + strlen(feature->category) + 2;
    char real[len];
    snprintf(real, len, "%s.%s", feature->category, feature->name);
    return likwid_sysft_getByName(real, device, value);
}

cerr_t likwid_sysft_modifyByName(const char* name, const LikwidDevice_t device, const char* value)
{
    if (!name || !device || !value)
        return ERROR_SET("name or device or value is NULL");

    if (device->type == DEVICE_TYPE_INVALID)
        return ERROR_SET("device type is invalid");

    size_t idx;
    if (get_feature_index(&idx, name))
        return ERROR_APPEND("get_feature_index failed");

    _SysFeature *f = &_feature_list.features[idx];
    if (!f->setter)
        return ERROR_SET("Cannot modify '%s', which has is read-only", name);

    if (f->type != device->type)
        return ERROR_SET("Cannot modify '%s' of type '%d', which is different to requested type '%d'", name, f->type, device->type);

    if (get_device_access(device))
        return ERROR_APPEND("get_device_access failed");
    if (f->setter(device, value))
        return ERROR_APPEND("setter for '%s' failed", name);
    return NULL;
}

cerr_t likwid_sysft_modify(const LikwidSysFeature* feature, const LikwidDevice_t device, const char* value)
{
    const size_t len = strlen(feature->name) + strlen(feature->category) + 2;
    char real[len];
    snprintf(real, len, "%s.%s", feature->category, feature->name);
    if (likwid_sysft_modifyByName(real, device, value))
        return ERROR_APPEND("likwid_sysft_modifyByName failed");
    return NULL;
}

void likwid_sysft_finalize(void)
{
    if (_feature_list.num_features > 0)
    {
        _free_feature_list(&_feature_list);
    }
}

cerr_t likwid_sysft_list(LikwidSysFeatureList* list)
{
    if (!list)
        return ERROR_SET("list is NULL");
    if (likwid_sysft_internal_to_external_feature_list(&_feature_list, list))
        return ERROR_SET("likwid_sysft_internal_to_external_feature_list failed");
    return NULL;
}

void likwid_sysft_list_return(LikwidSysFeatureList* list)
{
    if (!list || !list->features)
        return;

    likwid_sysft_free_feature_list(list);
    list->features = NULL;
    list->num_features = 0;
}
