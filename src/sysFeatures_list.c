#include <stdlib.h>
#include <stdio.h>

#include <sysFeatures.h>
#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>

int add_to_feature_list(SysFeatureList *list, SysFeature* feature)
{
    if ((!list) || (!feature))
    {
        ERROR_PRINT(Invalid arguments for add_to_feature_list);
        return -EINVAL;
    }

    SysFeature* flist = realloc(list->features, (list->num_features + 1) * sizeof(SysFeature));
    if (!flist)
    {
        ERROR_PRINT(Cannot allocate space for extended feature list);
        return -ENOMEM;
    }
    list->features = flist;

    SysFeature* iof = &(list->features[list->num_features]);
    iof->name = malloc((strlen(feature->name) + 1) * sizeof(char));
    if (!iof->name)
    {
        return -ENOMEM;
    }
    iof->category = malloc((strlen(feature->category) + 1) * sizeof(char));
    if (!iof->category)
    {
        free(iof->name);
        return -ENOMEM;
    }
    iof->description = malloc((strlen(feature->description) + 1) * sizeof(char));
    if (!iof->description)
    {
        free(iof->name);
        free(iof->category);
        return -ENOMEM;
    }
    int slen = HWFEATURES_MIN_STRLEN(strlen(feature->name) + 1, HWFEATURES_MAX_STR_LENGTH);
    strncpy(iof->name, feature->name, slen);
    slen = HWFEATURES_MIN_STRLEN(strlen(feature->category) + 1, HWFEATURES_MAX_STR_LENGTH);
    strncpy(iof->category, feature->category, slen);
    slen = HWFEATURES_MIN_STRLEN(strlen(feature->description) + 1, HWFEATURES_MAX_STR_LENGTH);
    strncpy(iof->description, feature->description, slen);

    iof->readonly = feature->readonly;
    iof->type = feature->type;
    iof->writeonly = feature->writeonly;

    list->num_features++;

    return 0;
}


int merge_feature_lists(SysFeatureList *inout, SysFeatureList *in)
{
    if ((!inout) || (!in))
    {
        ERROR_PRINT(Invalid arguments for merge_feature_lists);
        return -EINVAL;
    }

    SysFeature* flist = realloc(inout->features, (inout->num_features + in->num_features) * sizeof(SysFeature));
    if (!flist)
    {
        ERROR_PRINT(Cannot allocate space for extended feature list);
        return -ENOMEM;
    }
    inout->features = flist;

    for (int i = 0; i < in->num_features; i++)
    {
        SysFeature* ifeat = &(in->features[i]);
        add_to_feature_list(inout, ifeat);
    }


    return 0;
}

void free_feature_list(SysFeatureList *list)
{
    if (list)
    {
        for (int i = 0; i < list->num_features; i++)
        {
            SysFeature* f = &(list->features[i]);
            if (f->name) free(f->name);
            if (f->category) free(f->category);
            if (f->description) free(f->description);
        }
        memset(list->features, 0, list->num_features * sizeof(SysFeature));
        free(list->features);
        list->features = NULL;
        list->num_features = 0;
    }
}

int _add_to_feature_list(_SysFeatureList *list, _SysFeature* feature)
{
    if ((!list) || (!feature))
    {
        ERROR_PRINT(Invalid arguments for _add_to_feature_list);
        return -EINVAL;
    }

    _SysFeature* flist = realloc(list->features, (list->num_features + 1) * sizeof(_SysFeature));
    if (!flist)
    {
        ERROR_PRINT(Cannot allocate space for extended feature list);
        return -ENOMEM;
    }
    list->features = flist;

    _SysFeature* iof = &(list->features[list->num_features]);
    iof->name = feature->name;
    iof->category = feature->category;
    iof->description = feature->description;
    iof->type = feature->type;
    iof->getter = feature->getter;
    iof->setter = feature->setter;

    list->num_features++;

    return 0;
}

int _merge_feature_lists(_SysFeatureList *inout, _SysFeatureList *in)
{
    if ((!inout) || (!in))
    {
        ERROR_PRINT(Invalid arguments for _merge_feature_lists);
        return -EINVAL;
    }

    _SysFeature* flist = realloc(inout->features, (inout->num_features + in->num_features) * sizeof(_SysFeature));
    if (!flist)
    {
        ERROR_PRINT(Cannot allocate space for extended feature list);
        return -ENOMEM;
    }
    inout->features = flist;

    for (int i = 0; i < in->num_features; i++)
    {
        _SysFeature* ifeat = &(in->features[i]);
        _SysFeature* iof = &(inout->features[inout->num_features + i]);
        
        iof->name = ifeat->name;
        iof->category = ifeat->category;
        iof->description = ifeat->description;
        iof->type = ifeat->type;
        iof->getter = ifeat->getter;
        iof->setter = ifeat->setter;
        iof->tester = ifeat->tester;
    }
    inout->num_features += in->num_features;

    return 0;
}

void _free_feature_list(_SysFeatureList *list)
{
    if (list)
    {
        memset(list->features, 0, list->num_features * sizeof(_SysFeature));
        free(list->features);
        list->features = NULL;
        list->tester = NULL;
        list->num_features = 0;
    }
}


int internal_to_external_feature_list(_SysFeatureList *inlist, SysFeatureList* outlist)
{
    if ((!inlist) || (!outlist))
    {
        ERROR_PRINT(Invalid arguments for internal_to_external_feature_list);
        return -EINVAL;
    }
    outlist->num_features = 0;
    outlist->features = NULL;

    outlist->features = malloc(inlist->num_features * sizeof(SysFeature));
    if (!outlist->features)
    {
        return -ENOMEM;
    }

    for (int i = 0; i < inlist->num_features; i++)
    {
        SysFeature* out = &(outlist->features[i]);
        _SysFeature* in = &(inlist->features[i]);

        out->name = malloc((strlen(in->name)+1) * sizeof(char));
        if (!out->name)
        {
            for (int j = 0; j < i; j++)
            {
                SysFeature* c = &(outlist->features[j]);
                if (c->name) free(c->name);
                if (c->category) free(c->category);
                if (c->description) free(c->description);
            }
            free(outlist->features);
            outlist->features = NULL;
            return -ENOMEM;
        }
        out->category = malloc((strlen(in->category)+1) * sizeof(char));
        if (!out->category)
        {
            free(out->name);
            for (int j = 0; j < i; j++)
            {
                SysFeature* c = &(outlist->features[j]);
                if (c->name) free(c->name);
                if (c->category) free(c->category);
                if (c->description) free(c->description);
            }
            free(outlist->features);
            outlist->features = NULL;
            return -ENOMEM;
        }
        out->description = malloc((strlen(in->description)+1) * sizeof(char));
        if (!out->description)
        {
            free(out->name);
            free(out->category);
            for (int j = 0; j < i; j++)
            {
                SysFeature* c = &(outlist->features[j]);
                if (c->name) free(c->name);
                if (c->category) free(c->category);
                if (c->description) free(c->description);
            }
            free(outlist->features);
            outlist->features = NULL;
            return -ENOMEM;
        }
    }
    for (int i = 0; i < inlist->num_features; i++)
    {
        SysFeature* out = &(outlist->features[i]);
        _SysFeature* in = &(inlist->features[i]);
        int slen = HWFEATURES_MIN_STRLEN(strlen(in->name) + 1, HWFEATURES_MAX_STR_LENGTH);
        strncpy(out->name, in->name, slen);
        slen = HWFEATURES_MIN_STRLEN(strlen(in->category) + 1, HWFEATURES_MAX_STR_LENGTH);
        strncpy(out->category, in->category, slen);
        slen = HWFEATURES_MIN_STRLEN(strlen(in->description) + 1, HWFEATURES_MAX_STR_LENGTH);
        strncpy(out->description, in->description, slen);
        out->type = in->type;
        out->readonly = 0;
        out->writeonly = 0;
        if (in->getter != NULL && in->setter == NULL)
        {
            out->readonly = 1;
        }
        else if (in->getter == NULL && in->setter != NULL)
        {
            out->writeonly = 1;
        }
        outlist->num_features++;
    }
    return 0;
}
