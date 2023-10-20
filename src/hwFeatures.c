#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <topology.h>
#include <access.h>
#include <error.h>

#include <hwFeatures.h>
#include <hwFeatures_types.h>
#include <hwFeatures_common.h>
#include <hwFeatures_x86_intel.h>
#include <hwFeatures_x86_amd.h>

_HWFeature *local_features = NULL;
int num_local_features = 0;

int hwFeatures_init()
{
    int err = 0;
    int num_features = 0;
    _HWFeature *features = NULL;
    topology_init();
    CpuInfo_t cpuinfo = get_cpuInfo();
    if (cpuinfo->isIntel)
    {
        err = hwFeatures_init_x86_intel(&num_features, &features);
    }
    else
    {
        err = hwFeatures_init_x86_amd(&num_features, &features);
    }
    if (err < 0)
    {
        return err;
    }

    err = HPMinit();
    if (err < 0)
    {
        return err;
    }
    num_local_features = num_features;
    local_features = features;
    return 0;
}

static int _hwFeatures_get_feature_index(char* name)
{
    for (int i = 0; i < num_local_features; i++)
    {
        if (strncmp(name, local_features[i].name, strlen(local_features[i].name)) == 0)
        {
            return i;
        }
    }
    return -1;
}



int hwFeatures_getByName(char* name, int hwthread, uint64_t* value)
{
    int err = 0;
    _HWFeature *f = NULL;
    if ((!name) || (hwthread < 0) || (!value))
    {
        return -EINVAL;
    }

    int idx = _hwFeatures_get_feature_index(name);
    if (idx < 0)
    {
        return -EINVAL;
    }
    f = &local_features[idx];
    if ((!f) || (!f->getter))
    {
        return -EINVAL;
    }
    err = HPMaddThread(hwthread);
    if (err < 0)
    {
        return err;
    }
    uint64_t v = 0;
    err = f->getter(hwthread, &v);
    if (err == 0)
    {
        *value = v;
    }
    return err;
}

int hwFeatures_get(HWFeature* feature, int hwthread, uint64_t* value)
{
    return hwFeatures_getByName(feature->name, hwthread, value);
}

int hwFeatures_modifyByName(char* name, int hwthread, uint64_t value)
{
    _HWFeature *f = NULL;
    if ((!name) || (hwthread < 0))
    {
        return -EINVAL;
    }
    int idx = _hwFeatures_get_feature_index(name);
    if (idx < 0)
    {
        return -EINVAL;
    }
    f = &local_features[idx];
    if ((!f) || (!f->setter))
    {
        return -EINVAL;
    }
    int err = HPMaddThread(hwthread);
    if (err < 0)
    {
        return err;
    }
    return f->setter(hwthread, value);
}

int hwFeatures_modify(HWFeature* feature, int hwthread, uint64_t value)
{
    return hwFeatures_modifyByName(feature->name, hwthread, value);
}

void hwFeatures_finalize()
{
    if (local_features != NULL)
    {
        free(local_features);
        local_features = NULL;
        num_local_features = 0;
    }
}

int hwFeatures_list(HWFeatureList* list)
{
    if (!list)
    {
        return -EINVAL;
    }
    HWFeature* features = malloc(num_local_features * sizeof(HWFeature));
    if (!features)
    {
        return -ENOMEM;
    }
    memset(features, 0, num_local_features * sizeof(HWFeature));

    for (int i = 0; i < num_local_features; i++)
    {
        features[i].name = malloc((strlen(local_features[i].name)+1) * sizeof(char));
        if (features[i].name)
        {
            features[i].description = malloc((strlen(local_features[i].description)+1) * sizeof(char));
            if (!features[i].description)
            {
                free(features[i].name);
                for (int j = 0; j < i; j++)
                {
                    if (features[j].name) free(features[j].name);
                    if (features[j].description) free(features[j].description);
                }
            }
        }
        else
        {
            for (int j = 0; j < i; j++)
            {
                if (features[j].name) free(features[j].name);
                if (features[j].description) free(features[j].description);
            }
        }
    }
    for (int i = 0; i < num_local_features; i++)
    {
        strcpy(features[i].name, local_features[i].name);
        strcpy(features[i].description, local_features[i].description);
        features[i].scope = local_features[i].scope;
        features[i].readonly = 0;
        if (local_features[i].getter != NULL && local_features[i].setter == NULL)
        {
            features[i].readonly = 1;
        }
    if (local_features[i].getter == NULL && local_features[i].setter != NULL)
    {
            features[i].writeonly = 1;
    }
    }
    list->num_features = num_local_features;
    list->features = features;
    return 0;
}

void hwFeatures_list_return(HWFeatureList* list)
{
    if (!list || !list->features)
    {
        return;
    }
    for (int i = 0; i < list->num_features; i++)
    {
        free(list->features[i].name);
        free(list->features[i].description);
    }
    free(list->features);
    list->features = NULL;
    list->num_features = 0;
}
