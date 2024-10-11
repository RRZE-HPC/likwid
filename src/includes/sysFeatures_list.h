#ifndef HWFEATURES_LIST_H
#define HWFEATURES_LIST_H

#include <sysFeatures_types.h>
#include <likwid.h>

/* External lists with flags etc. */
int likwid_sysft_add_to_feature_list(LikwidSysFeatureList *list, const LikwidSysFeature* feature);
int likwid_sysft_merge_feature_lists(LikwidSysFeatureList *inout, const LikwidSysFeatureList *in);
void likwid_sysft_free_feature_list(LikwidSysFeatureList *list);

/* Internal lists with function pointers etc. */
int _add_to_feature_list(_SysFeatureList *list, const _SysFeature* feature);
int _merge_feature_lists(_SysFeatureList *inout, const _SysFeatureList *in);
void _free_feature_list(_SysFeatureList *list);

/* Get an external list from an internal one */
int likwid_sysft_internal_to_external_feature_list(const _SysFeatureList *inlist, LikwidSysFeatureList* outlist);

/* Print a list for debugging purposes */
void likwid_sysft_printlistint(const _SysFeatureList *list);
void likwid_sysft_printlistext(const LikwidSysFeatureList *list);

#endif /* HWFEATURES_LIST_H */
