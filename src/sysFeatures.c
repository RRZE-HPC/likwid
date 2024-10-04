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
#include <sysFeatures_intel.h>
#include <sysFeatures_cpufreq.h>
#include <sysFeatures_linux_numa_balancing.h>

//#include <sysFeatures_x86_amd.h>

_SysFeature *local_features = NULL;
int num_local_features = 0;

_SysFeatureList _feature_list = {0, NULL, NULL};


static int get_device_access(LikwidDevice_t device)
{
    int hwt = -1;
    CpuTopology_t topo = NULL;
    int err = topology_init();
    if (err < 0)
    {
        return err;
    }
    topo = get_cpuTopology();

    if (device->type == DEVICE_TYPE_INVALID)
    {
        return -EINVAL;
    }
    switch (device->type) {
        case DEVICE_TYPE_HWTHREAD:
            if (device->id.simple.id >= 0 && device->id.simple.id < topo->numHWThreads)
            {
                hwt = device->id.simple.id;
            }
            break;
        case DEVICE_TYPE_NODE:
            for (int i = 0; i < topo->numHWThreads; i++)
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
            for (int i = 0; i < topo->numHWThreads; i++)
            {
                HWThread* t = &topo->threadPool[i];
                if (t->inCpuSet == 1 && device->id.simple.id == t->packageId)
                {
                    hwt = t->apicId;
                    break;
                }
            }
            break;
    }
    if (hwt >= 0)
    {
        return HPMaddThread(hwt);
    }
    return -EINVAL;
}


int sysFeatures_init()
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
        err = sysFeatures_init_x86_intel(&_feature_list);
        if (err < 0)
        {
            ERROR_PRINT(Failed to initialize SysFeatures for Intel architecture);
            return err;
        }
    }
    else
    {
        //err = sysFeatures_init_x86_amd(&num_features, &features);
        if (err < 0)
        {
            ERROR_PRINT(Failed to initialize SysFeatures for AMD architecture);
            return err;
        }
    }
    
    err = sysFeatures_init_cpufreq(&_feature_list);
    if (err < 0)
    {
        ERROR_PRINT(Failed to initialize SysFeatures cpufreq module);
        return err;
    }

    err = sysFeatures_init_linux_numa_balancing(&_feature_list);
    if (err < 0)
    {
        ERROR_PRINT(Failed to initialize SysFeatures numa_balancing module);
        return err;
    }
    
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Initialized %d features, _feature_list.num_features);
    return 0;
}

static int _sysFeatures_get_feature_index(const char* name)
{
    int dot = -1;
    if (!name)
    {
        return -EINVAL;
    }
    int namelen = strlen(name);
    for (int i = 0; i < namelen; i++)
    {
        if (name[i] == '.')
        {
            dot = i;
            break;
        }
    }
    if (dot < 0)
    {
        int out = -1;
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Features name has no dot -> compare with name, dot);
        for (int i = 0; i < _feature_list.num_features; i++)
        {
            int featlen = strlen(_feature_list.features[i].name);
            int checklen = (namelen < featlen ? featlen : namelen);
            if (strncmp(name, _feature_list.features[i].name, checklen) == 0)
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
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Features name contains dot at offset %d -> compare with category.name, dot);
        for (int i = 0; i < _feature_list.num_features; i++)
        {
            int featlen = strlen(_feature_list.features[i].name) + strlen(_feature_list.features[i].category) + 2;
            char real[featlen];
            int ret = snprintf(real, featlen, "%s.%s", _feature_list.features[i].category, _feature_list.features[i].name);
            if (ret > 0)
            {
                real[ret] = '\0';
                int checklen = (namelen < featlen ? featlen : namelen);
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Comparing '%s' to '%s': %d, name, real, strncmp(name, real, checklen));
                if (strncmp(name, real, checklen) == 0)
                {
                    return i;
                }
            }
        }
    }
    ERROR_PRINT(SysFeatures modules does not provide a feature called %s, name);
    return -ENOTSUP;
}



int sysFeatures_getByName(const char* name, const LikwidDevice_t device, char** value)
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
    int idx = _sysFeatures_get_feature_index(name);
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

int sysFeatures_get(const SysFeature* feature, const LikwidDevice_t device, char** value)
{
    int len = strlen(feature->name) + strlen(feature->category) + 2;
    char real[len];
    int ret = snprintf(real, len, "%s.%s", feature->category, feature->name);
    return sysFeatures_getByName(real, device, value);
}

int sysFeatures_modifyByName(const char* name, const LikwidDevice_t device, const char* value)
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
    int idx = _sysFeatures_get_feature_index(name);
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

int sysFeatures_modify(const SysFeature* feature, const LikwidDevice_t device, const char* value)
{
    int len = strlen(feature->name) + strlen(feature->category) + 2;
    char real[len];
    int ret = snprintf(real, len, "%s.%s", feature->category, feature->name);
    return sysFeatures_modifyByName(feature->name, device, value);
}

void sysFeatures_finalize()
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

int sysFeatures_list(SysFeatureList* list)
{
    if (!list)
    {
        return -EINVAL;
    }
    return internal_to_external_feature_list(&_feature_list, list);
}

void sysFeatures_list_return(SysFeatureList* list)
{
    if (!list || !list->features)
    {
        return;
    }
    free_feature_list(list);
/*    for (int i = 0; i < list->num_features; i++)*/
/*    {*/
/*        SysFeature* f = &(list->features[i]);*/
/*        if (f->name) free(f->name);*/
/*        if (f->category) free(f->category);*/
/*        if (f->description) free(f->description);*/
/*    }*/
/*    free(list->features);*/
    list->features = NULL;
    list->num_features = 0;
}
