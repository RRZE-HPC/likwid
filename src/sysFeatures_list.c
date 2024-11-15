/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_list.c
 *
 *      Description:  Management functions for list of sysFeatures
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

#include <sysFeatures.h>
#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>

int likwid_sysft_add_to_feature_list(LikwidSysFeatureList *list, const LikwidSysFeature* feature)
{
    if ((!list) || (!feature))
    {
        ERROR_PRINT(Invalid arguments for add_to_feature_list);
        return -EINVAL;
    }

    LikwidSysFeature* flist = realloc(list->features, (list->num_features + 1) * sizeof(LikwidSysFeature));
    if (!flist)
    {
        ERROR_PRINT(Cannot allocate space for extended feature list);
        return -ENOMEM;
    }
    list->features = flist;

    LikwidSysFeature* iof = &list->features[list->num_features];
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


int likwid_sysft_merge_feature_lists(LikwidSysFeatureList *inout, const LikwidSysFeatureList *in)
{
    if ((!inout) || (!in))
    {
        ERROR_PRINT(Invalid arguments for merge_feature_lists);
        return -EINVAL;
    }

    LikwidSysFeature *flist = realloc(inout->features, (inout->num_features + in->num_features) * sizeof(LikwidSysFeature));
    if (!flist)
    {
        ERROR_PRINT(Cannot allocate space for extended feature list);
        return -ENOMEM;
    }
    inout->features = flist;

    for (int i = 0; i < in->num_features; i++)
    {
        LikwidSysFeature* ifeat = &in->features[i];
        likwid_sysft_add_to_feature_list(inout, ifeat);
    }


    return 0;
}

void likwid_sysft_free_feature_list(LikwidSysFeatureList *list)
{
    if (list)
    {
        for (int i = 0; i < list->num_features; i++)
        {
            LikwidSysFeature* f = &list->features[i];
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

int likwid_sysft_internal_to_external_feature_list(const _SysFeatureList *inlist, LikwidSysFeatureList* outlist)
{
    if ((!inlist) || (!outlist))
    {
        ERROR_PRINT(Invalid arguments for internal_to_external_feature_list);
        return -EINVAL;
    }
    outlist->num_features = 0;
    outlist->features = NULL;

    outlist->features = malloc(inlist->num_features * sizeof(LikwidSysFeature));
    if (!outlist->features)
    {
        return -ENOMEM;
    }

    for (int i = 0; i < inlist->num_features; i++)
    {
        LikwidSysFeature* out = &outlist->features[i];
        const _SysFeature* in = &inlist->features[i];

        out->name = strndup(in->name, HWFEATURES_MAX_STR_LENGTH - 1);
        if (!out->name)
        {
            for (int j = 0; j < i; j++)
            {
                LikwidSysFeature* c = &outlist->features[j];
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
                LikwidSysFeature* c = &outlist->features[j];
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
                LikwidSysFeature* c = &outlist->features[j];
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

void likwid_sysft_printlistint(const _SysFeatureList *list)
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

void likwid_sysft_printlistext(const LikwidSysFeatureList *list)
{
    if (list->num_features == 0)
    {
        printf("<blank feature list>\n");
    }
    for (int i = 0; i < list->num_features; i++)
    {
        const LikwidSysFeature *f = &list->features[i];
        const char *cat = f->category ? f->category : "(null)";
        const char *desc = f->description ? f->description : "(null)";
        printf("[%03d] name=%s category=%s description=%s type=%d readonly=%d writeonly=%d\n", i, f->name, cat, desc, f->type, f->readonly, f->writeonly);
    }
}
