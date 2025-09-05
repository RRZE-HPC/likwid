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

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include <sysFeatures.h>
#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>

cerr_t likwid_sysft_add_to_feature_list(LikwidSysFeatureList *list, const LikwidSysFeature* feature)
{
    assert(list);
    assert(feature);

    LikwidSysFeature* flist = realloc(list->features, (list->num_features + 1) * sizeof(LikwidSysFeature));
    if (!flist)
        return ERROR_SET_ERRNO("realloc failed");

    list->features = flist;

    LikwidSysFeature *iof = &list->features[list->num_features];
    memset(iof, 0, sizeof(*iof));

    cerr_t err;

    iof->name = strndup(feature->name, HWFEATURES_MAX_STR_LENGTH - 1);
    if (!iof->name) {
        err = ERROR_SET_ERRNO("strndup failed");
        goto error;
    }

    iof->category = strndup(feature->category, HWFEATURES_MAX_STR_LENGTH - 1);
    if (!iof->category) {
        err = ERROR_SET_ERRNO("strndup failed");
        goto error;
    }

    iof->readonly = feature->readonly;
    iof->type = feature->type;
    iof->writeonly = feature->writeonly;
    list->num_features++;

    return NULL;

error:
    free(iof->name);
    free(iof->category);
    return err;
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

cerr_t _add_to_feature_list(_SysFeatureList *list, const _SysFeature* feature)
{
    assert(list);
    assert(feature);

    _SysFeature *flist = realloc(list->features, (list->num_features + 1) * sizeof(_SysFeature));
    if (!flist)
        return ERROR_SET_ERRNO("realloc failed");

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

    return NULL;
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

cerr_t likwid_sysft_internal_to_external_feature_list(const _SysFeatureList *inlist, LikwidSysFeatureList* outlist)
{
    assert(inlist);
    assert(outlist);

    outlist->features = calloc(inlist->num_features, sizeof(*outlist->features));
    if (!outlist->features)
        return ERROR_SET_ERRNO("malloc failed");
    outlist->num_features = inlist->num_features;

    cerr_t err;

    for (int i = 0; i < inlist->num_features; i++)
    {
        LikwidSysFeature* out = &outlist->features[i];
        const _SysFeature* in = &inlist->features[i];

        out->name = strndup(in->name, HWFEATURES_MAX_STR_LENGTH - 1);
        if (!out->name) {
            err = ERROR_SET_ERRNO("strdnup failed");
            goto error;
        }

        out->category = strndup(in->category, HWFEATURES_MAX_STR_LENGTH - 1);
        if (!out->category) {
            err = ERROR_SET_ERRNO("strdnup failed");
            goto error;
        }

        out->description = strndup(in->description, HWFEATURES_MAX_STR_LENGTH - 1);
        if (!out->description) {
            err = ERROR_SET_ERRNO("strdnup failed");
            goto error;
        }

        out->type = in->type;
        out->readonly = 0;
        out->writeonly = 0;

        if (in->getter != NULL && in->setter == NULL)
            out->readonly = 1;
        else if (in->getter == NULL && in->setter != NULL)
            out->writeonly = 1;
    }

    return NULL;

error:
    for (int i = 0; i < outlist->num_features; i++) {
        LikwidSysFeature *f = &outlist->features[i];
        free(f->name);
        free(f->category);
        free(f->description);
    }
    free(outlist->features);
    outlist->features = NULL;
    outlist->num_features = 0;
    return err;
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
