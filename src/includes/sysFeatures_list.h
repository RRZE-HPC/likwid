/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_list.h
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
