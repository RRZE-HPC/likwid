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

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <topology.h>
#include <access.h>
#include <error.h>
#include <likwid.h>

#include <sysFeatures.h>
#include <sysFeatures_types.h>
#include <sysFeatures_common.h>
#include <sysFeatures_amd.h>
#include <sysFeatures_intel.h>
#include <sysFeatures_cpufreq.h>
#include <sysFeatures_linux_numa_balancing.h>
#include <sysFeatures_nvml.h>

static _SysFeature *local_features = NULL;
static int num_local_features = 0;

static _SysFeatureList _feature_list = {0, NULL, NULL};


static int get_device_access(LikwidDevice_t device)
{
    /* This function is somewhat redundant to the initialization,
     * which is implicitly called in every getter/settter. */
    int hwt = -1;
    int err = topology_init();
    if (err < 0)
    {
        return err;
    }
    CpuTopology_t topo = get_cpuTopology();

    if (device->type == DEVICE_TYPE_INVALID)
    {
        return -EINVAL;
    }
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
            return 0;
#endif
#ifdef LIKWID_WITH_ROCMON
        case DEVICE_TYPE_AMD_GPU:
            return 0;
#endif
        default:
            ERROR_PRINT(get_device_access: Unimplemented device type: %d\n, device->type);
            return -EPERM;
    }
    if (hwt >= 0)
    {
        return HPMaddThread(hwt);
    }
    return -EINVAL;
}


int likwid_sysft_init(void)
{
    int err = 0;

    topology_init();
    CpuInfo_t cpuinfo = get_cpuInfo();
    if (!HPMinitialized())
    {
        err = HPMinit();
        if (err < 0)
        {
            ERROR_PRINT(Failed to initialize access to hardware registers);
            return err;
        }
    }
    if (cpuinfo->isIntel)
    {
        err = likwid_sysft_init_x86_intel(&_feature_list);
        if (err < 0)
        {
            ERROR_PRINT(Failed to initialize SysFeatures for Intel architecture);
            return err;
        }
    }
    else
    {
        err = likwid_sysft_init_x86_amd(&_feature_list);
        if (err < 0)
        {
            ERROR_PRINT(Failed to initialize SysFeatures for AMD architecture);
            return err;
        }
    }
    
    err = likwid_sysft_init_cpufreq(&_feature_list);
    if (err < 0)
    {
        ERROR_PRINT(Failed to initialize SysFeatures cpufreq module);
        return err;
    }

    err = likwid_sysft_init_linux_numa_balancing(&_feature_list);
    if (err < 0)
    {
        ERROR_PRINT(Failed to initialize SysFeatures numa_balancing module);
        return err;
    }
#ifdef LIKWID_WITH_NVMON
    err = likwid_sysft_init_nvml(&_feature_list);
    if (err < 0)
    {
        ERROR_PRINT(Failed to initialize SysFeatures nvml module);
        return err;
    }
#endif
    
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Initialized %d features, _feature_list.num_features);
    return 0;
}

static int get_feature_index(const char* name)
{
    if (!name)
    {
        return -EINVAL;
    }
    if (!strchr(name, '.'))
    {
        int out = -1;
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Features name has no dot -> compare with name);
        for (int i = 0; i < _feature_list.num_features; i++)
        {
            if (strcmp(name, _feature_list.features[i].name) == 0)
            {
                if (out < 0)
                {
                    out = i;
                }
                else
                {
                    ERROR_PRINT(Feature name '%s' matches multiple features, name);
                    return -EINVAL;
                }
            }
        }
        if (out >= 0)
        {
            return out;
        }
    }
    else
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Features name contains dot -> compare with category.name);
        for (int i = 0; i < _feature_list.num_features; i++)
        {
            int featlen = strlen(_feature_list.features[i].name) + strlen(_feature_list.features[i].category) + 2;
            char combined_name[featlen];
            snprintf(combined_name, featlen, "%s.%s", _feature_list.features[i].category, _feature_list.features[i].name);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Comparing '%s' to '%s': %d, name, combined_name, strcmp(name, combined_name));
            if (strcmp(name, combined_name) == 0)
            {
                return i;
            }
        }
    }
    ERROR_PRINT(SysFeatures modules does not provide a feature called %s, name);
    return -ENOTSUP;
}



int likwid_sysft_getByName(const char* name, const LikwidDevice_t device, char** value)
{
    int err = 0;
    _SysFeature *f = NULL;
    if ((!name) || (!device) || (!value))
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Invalid inputs to sysFeatures_getByName);
        return -EINVAL;
    }
    if (device->type == DEVICE_TYPE_INVALID)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Invalid device type);
        return -EINVAL;
    }
    int idx = get_feature_index(name);
    if (idx < 0)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Failed to get index for %s, name);
        return -EINVAL;
    }
    f = &_feature_list.features[idx];
    if ((!f) || (!f->getter))
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, No feature %s or no support to read current state, name);
        return -EINVAL;
    }
    if (f->type != device->type)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Feature %s has a different type than device, name);
        return -EINVAL;
    }
    err = get_device_access(device);
    if (err < 0)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Failed to get access to device);
        return err;
    }
    err = f->getter(device, value);
    return err;
}

int likwid_sysft_get(const LikwidSysFeature* feature, const LikwidDevice_t device, char** value)
{
    int len = strlen(feature->name) + strlen(feature->category) + 2;
    char real[len];
    snprintf(real, len, "%s.%s", feature->category, feature->name);
    return likwid_sysft_getByName(real, device, value);
}

int likwid_sysft_modifyByName(const char* name, const LikwidDevice_t device, const char* value)
{
    int err = 0;
    int dev_id = -1;
    _SysFeature *f = NULL;
    if ((!name) || (!device) || (!value))
    {
        return -EINVAL;
    }
    if (device->type == DEVICE_TYPE_INVALID)
    {
        return -EINVAL;
    }
    int idx = get_feature_index(name);
    if (idx < 0)
    {
        return -EINVAL;
    }
    f = &_feature_list.features[idx];
    if ((!f) || (!f->setter))
    {
        return -EPERM;
    }
    if (f->type != device->type)
    {
        return -ENODEV;
    }
    err = get_device_access(device);
    if (err < 0)
    {
        return err;
    }
    return f->setter(device, value);
}

int likwid_sysft_modify(const LikwidSysFeature* feature, const LikwidDevice_t device, const char* value)
{
    int len = strlen(feature->name) + strlen(feature->category) + 2;
    char real[len];
    snprintf(real, len, "%s.%s", feature->category, feature->name);
    return likwid_sysft_modifyByName(feature->name, device, value);
}

void likwid_sysft_finalize(void)
{
    if (local_features != NULL)
    {
        free(local_features);
        local_features = NULL;
        num_local_features = 0;
    }
    if (_feature_list.num_features > 0)
    {
        _free_feature_list(&_feature_list);
    }
    
}

int likwid_sysft_list(LikwidSysFeatureList* list)
{
    if (!list)
    {
        return -EINVAL;
    }
    return likwid_sysft_internal_to_external_feature_list(&_feature_list, list);
}

void likwid_sysft_list_return(LikwidSysFeatureList* list)
{
    if (!list || !list->features)
    {
        return;
    }
    likwid_sysft_free_feature_list(list);
    list->features = NULL;
    list->num_features = 0;
}
