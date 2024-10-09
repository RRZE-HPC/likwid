#include <stdlib.h>
#include <stdio.h>

#include <sysFeatures.h>
#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>

int add_to_feature_list(SysFeatureList *list, const SysFeature* feature)
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

    SysFeature* iof = &list->features[list->num_features];
    iof->name = strndup(feature->name, HWFEATURES_MAX_STR_LENGTH - 1);
    if (!iof->name)
    {
        return -ENOMEM;
    }
    iof->category = strndup(feature->category, HWFEATURES_MAX_STR_LENGTH - 1);
    if (!iof->category)
    {
        free(iof->name);
        return -ENOMEM;
    }
    iof->description = strndup(feature->category, HWFEATURES_MAX_STR_LENGTH - 1);
    if (!iof->description)
    {
        free(iof->name);
        free(iof->category);
        return -ENOMEM;
    }

    iof->readonly = feature->readonly;
    iof->type = feature->type;
    iof->writeonly = feature->writeonly;
    list->num_features++;

    return 0;
}


int merge_feature_lists(SysFeatureList *inout, const SysFeatureList *in)
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
        SysFeature* ifeat = &in->features[i];
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
            SysFeature* f = &list->features[i];
            if (f->name) free(f->name);
            if (f->category) free(f->category);
            if (f->description) free(f->description);
        }
        free(list->features);
        list->features = NULL;
        list->num_features = 0;
    }
}

int _add_to_feature_list(_SysFeatureList *list, const _SysFeature* feature)
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

    _SysFeature* iof = &list->features[list->num_features];
    iof->name = feature->name;
    iof->category = feature->category;
    iof->description = feature->description;
    iof->type = feature->type;
    iof->getter = feature->getter;
    iof->setter = feature->setter;
    iof->tester = feature->tester;
    iof->unit = feature->unit;

    list->num_features++;

    return 0;
}

int _merge_feature_lists(_SysFeatureList *inout, const _SysFeatureList *in)
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
        _SysFeature* ifeat = &in->features[i];
        _SysFeature* iof = &inout->features[inout->num_features + i];
        
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
        free(list->features);
        list->features = NULL;
        list->tester = NULL;
        list->num_features = 0;
    }
}


int internal_to_external_feature_list(const _SysFeatureList *inlist, SysFeatureList* outlist)
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

        out->name = strndup(in->name, HWFEATURES_MAX_STR_LENGTH - 1);
        if (!out->name)
        {
            for (int j = 0; j < i; j++)
            {
                SysFeature* c = &outlist->features[j];
                free(c->name);
                free(c->category);
                free(c->description);
            }
            free(outlist->features);
            outlist->features = NULL;
            return -ENOMEM;
        }
        out->category = strndup(in->category, HWFEATURES_MAX_STR_LENGTH - 1);
        if (!out->category)
        {
            free(out->name);
            for (int j = 0; j < i; j++)
            {
                SysFeature* c = &outlist->features[j];
                free(c->name);
                free(c->category);
                free(c->description);
            }
            free(outlist->features);
            outlist->features = NULL;
            return -ENOMEM;
        }
        out->description = strndup(in->description, HWFEATURES_MAX_STR_LENGTH - 1);
        if (!out->description)
        {
            free(out->name);
            free(out->category);
            for (int j = 0; j < i; j++)
            {
                SysFeature* c = &outlist->features[j];
                free(c->name);
                free(c->category);
                free(c->description);
            }
            free(outlist->features);
            outlist->features = NULL;
            return -ENOMEM;
        }
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

void sysFeatures_printlistint(const _SysFeatureList *list)
{
    if (list->num_features == 0)
    {
        printf("<blank feature list>\n");
    }
    for (int i = 0; i < list->num_features; i++)
    {
        const _SysFeature *f = &list->features[i];
        const char *cat = f->category ? f->category : "(null)";
        const char *desc = f->description ? f->description : "(null)";
        const char *unit = f->unit ? f->unit : "(null)";
        printf("[%03d] name=%s category=%s description=%s getter=%p setter=%p type=%d tester=%p unit=%s\n", i, f->name, cat, desc, f->getter, f->setter, f->type, f->tester, unit);
    }
}

void sysFeatures_printlistext(const SysFeatureList *list)
{
    if (list->num_features == 0)
    {
        printf("<blank feature list>\n");
    }
    for (int i = 0; i < list->num_features; i++)
    {
        const SysFeature *f = &list->features[i];
        const char *cat = f->category ? f->category : "(null)";
        const char *desc = f->description ? f->description : "(null)";
        printf("[%03d] name=%s category=%s description=%s type=%d readonly=%d writeonly=%d\n", i, f->name, cat, desc, f->type, f->readonly, f->writeonly);
    }
}
