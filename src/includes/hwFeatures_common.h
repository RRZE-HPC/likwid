#ifndef HWFEATURES_COMMON_H
#define HWFEATURES_COMMON_H

int hwFeatures_init_generic(_HWArchFeatures* infeatures, int* num_features, _HWFeature **features)
{
    int i = 0;
    int j = 0;
    int c = 0;
    int err = 0;
    CpuInfo_t cpuinfo = NULL;
    _HWFeatureList** feature_list = NULL;
    _HWFeature* out = NULL;
    err = topology_init();
    if (err < 0)
    {
        ERROR_PRINT(Failed to initialize topology module);
        return err;
    }
    cpuinfo = get_cpuInfo();

    c = 0;
    while (infeatures[c].family >= 0 && infeatures[c].model >= 0)
    {
        if (infeatures[c].family == cpuinfo->family && infeatures[c].model == cpuinfo->model)
        {
            feature_list = infeatures[c].features;
            break;
        }
        c++;
    }
    if (!feature_list)
    {
        ERROR_PRINT(No feature list found for current architecture);
        *num_features = 0;
        *features = NULL;
        return 0;
    }

    j = 0;
    while (feature_list[j] != NULL && feature_list[j]->features != NULL)
    {
        c += feature_list[j]->num_features;
        j++;
    }

    out = malloc((c+1) * sizeof(_HWFeature));
    if (!out)
    {
        ERROR_PRINT(Failed to allocate space for HW feature list);
        return -ENOMEM;
    }
    memset(out, 0, (c+1) * sizeof(_HWFeature));

    j = 0;
    c = 0;
    while (feature_list[j] != NULL && feature_list[j]->features != NULL)
    {
        for (i = 0; i < feature_list[j]->num_features; i++)
        {
            //_COPY_HWFEATURE_ENTRY(out[c], feature_list[j][i]);
            out[c].name = feature_list[j]->features[i].name;
            out[c].description = feature_list[j]->features[i].description;
            out[c].setter = feature_list[j]->features[i].setter;
            out[c].getter = feature_list[j]->features[i].getter;
            out[c].scope = feature_list[j]->features[i].scope;
            c++;
        }
        j++;
    }
    *features = out;
    *num_features = c;
    return 0;
}


#endif
