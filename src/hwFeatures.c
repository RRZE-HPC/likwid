#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <topology.h>
#include <access.h>
#include <error.h>
#include <likwid_device.h>

#include <hwFeatures.h>
#include <hwFeatures_types.h>
#include <hwFeatures_common.h>
#include <hwFeatures_intel.h>
#include <hwFeatures_cpufreq.h>
#include <hwFeatures_linux_numa_balancing.h>

//#include <hwFeatures_x86_amd.h>

_HWFeature *local_features = NULL;
int num_local_features = 0;

_HWFeatureList _feature_list = {0, NULL, NULL};


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


int hwFeatures_init()
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
        err = hwFeatures_init_x86_intel(&_feature_list);
        if (err < 0)
        {
            ERROR_PRINT(Failed to initialize HWFeatures for Intel architecture);
            return err;
        }
    }
    else
    {
        //err = hwFeatures_init_x86_amd(&num_features, &features);
        if (err < 0)
        {
            ERROR_PRINT(Failed to initialize HWFeatures for AMD architecture);
            return err;
        }
    }
    
    err = hwFeatures_init_cpufreq(&_feature_list);
    if (err < 0)
    {
        ERROR_PRINT(Failed to initialize HWFeatures cpufreq module);
        return err;
    }

    err = hwFeatures_init_linux_numa_balancing(&_feature_list);
    if (err < 0)
    {
        ERROR_PRINT(Failed to initialize HWFeatures numa_balancing module);
        return err;
    }
    
    err = hwFeatures_init_linux_numa_balancing(&_feature_list);
    if (err < 0)
    {
        ERROR_PRINT(Failed to initialize HWFeatures numa_balancing module);
        return err;
    }

    DEBUG_PRINT(DEBUGLEV_DEVELOP, Initialized %d features, _feature_list.num_features);
    return 0;
}

static int _hwFeatures_get_feature_index(char* name)
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
                printf("'%s' vs. '%s' -> %d\n",name, _feature_list.features[i].name, strncmp(name, _feature_list.features[i].name, strlen(name)));
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
    ERROR_PRINT(HWFeatures modules does not provide a feature called %s, name);
    return -ENOTSUP;
}



int hwFeatures_getByName(char* name, LikwidDevice_t device, char** value)
{
    int err = 0;
    _HWFeature *f = NULL;
    if ((!name) || (!device) || (!value))
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Invalid inputs to hwFeatures_getByName);
        return -EINVAL;
    }
    if (device->type == DEVICE_TYPE_INVALID)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Invalid device type);
        return -EINVAL;
    }
    int idx = _hwFeatures_get_feature_index(name);
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

int hwFeatures_get(HWFeature* feature, LikwidDevice_t device, char** value)
{
    int len = strlen(feature->name) + strlen(feature->category) + 2;
    char real[len];
    int ret = snprintf(real, len, "%s.%s", feature->category, feature->name);
    return hwFeatures_getByName(real, device, value);
}

int hwFeatures_modifyByName(char* name, LikwidDevice_t device, char* value)
{
    int err = 0;
    int dev_id = -1;
    _HWFeature *f = NULL;
    if ((!name) || (!device) || (!value))
    {
        return -EINVAL;
    }
    if (device->type == DEVICE_TYPE_INVALID)
    {
        return -EINVAL;
    }
    int idx = _hwFeatures_get_feature_index(name);
    if (idx < 0)
    {
        return -EINVAL;
    }
    f = &_feature_list.features[idx];
    if ((!f) || (!f->setter))
    {
        return -EINVAL;
    }
    if (f->type != device->type)
    {
        return -EINVAL;
    }
    
    err = get_device_access(device);
    if (err < 0)
    {
        return err;
    }
    return f->setter(device, value);
}

int hwFeatures_modify(HWFeature* feature, LikwidDevice_t device, char* value)
{
    int len = strlen(feature->name) + strlen(feature->category) + 2;
    char real[len];
    int ret = snprintf(real, len, "%s.%s", feature->category, feature->name);
    return hwFeatures_modifyByName(feature->name, device, value);
}

void hwFeatures_finalize()
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

int hwFeatures_list(HWFeatureList* list)
{
    if (!list)
    {
        return -EINVAL;
    }
    return internal_to_external_feature_list(&_feature_list, list);
}

void hwFeatures_list_return(HWFeatureList* list)
{
    if (!list || !list->features)
    {
        return;
    }
    free_feature_list(list);
/*    for (int i = 0; i < list->num_features; i++)*/
/*    {*/
/*        HWFeature* f = &(list->features[i]);*/
/*        if (f->name) free(f->name);*/
/*        if (f->category) free(f->category);*/
/*        if (f->description) free(f->description);*/
/*    }*/
/*    free(list->features);*/
    list->features = NULL;
    list->num_features = 0;
}
