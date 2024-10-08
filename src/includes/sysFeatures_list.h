#ifndef HWFEATURES_LIST_H
#define HWFEATURES_LIST_H

#include <sysFeatures_types.h>
#include <likwid.h>

/* External lists with flags etc. */
int add_to_feature_list(SysFeatureList *list, const SysFeature* feature);
int merge_feature_lists(SysFeatureList *inout, const SysFeatureList *in);
void free_feature_list(SysFeatureList *list);

/* Internal lists with function pointers etc. */
int _add_to_feature_list(_SysFeatureList *list, const _SysFeature* feature);
int _merge_feature_lists(_SysFeatureList *inout, const _SysFeatureList *in);
void _free_feature_list(_SysFeatureList *list);

/* Get an external list from an internal one */
int internal_to_external_feature_list(const _SysFeatureList *inlist, SysFeatureList* outlist);

/* Print a list for debugging purposes */
void sysFeatures_printlistint(const _SysFeatureList *list);
void sysFeatures_printlistext(const SysFeatureList *list);

#endif /* HWFEATURES_LIST_H */
