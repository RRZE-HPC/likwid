#ifndef HWFEATURES_LIST_H
#define HWFEATURES_LIST_H

#include <hwFeatures_types.h>
#include <likwid.h>

/* External lists with flags etc. */
int add_to_feature_list(HWFeatureList *list, HWFeature* feature);
int merge_feature_lists(HWFeatureList *inout, HWFeatureList *in);
void free_feature_list(HWFeatureList *list);

/* Internal lists with function pointers etc. */
int _add_to_feature_list(_HWFeatureList *list, _HWFeature* feature);
int _merge_feature_lists(_HWFeatureList *inout, _HWFeatureList *in);
void _free_feature_list(_HWFeatureList *list);

/* Get an external list from an internal one */
int internal_to_external_feature_list(_HWFeatureList *inlist, HWFeatureList* outlist);

#endif /* HWFEATURES_LIST_H */
