/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_intel_prefetcher.h
 *
 *      Description:  Interface to control CPU prefetchers for the sysFeatures component
 *
 *      Version:   5.4.0
 *      Released:  15.11.2024
 *
 *      Authors:  Thomas Gruber (tg), thomas.roehl@googlemail.com
 *                Michael Panzlaff, michael.panzlaff@fau.de
 *      Project:  likwid
 *
 *      Copyright (C) 2024 RRZE, University Erlangen-Nuremberg
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

#ifndef HWFEATURES_X86_INTEL_PREFETCHER_H
#define HWFEATURES_X86_INTEL_PREFETCHER_H

#include <sysFeatures_types.h>

/*********************************************************************************************************************/
/*                          Intel prefetchers                                                                        */
/*********************************************************************************************************************/

extern const _SysFeatureList likwid_sysft_intel_cpu_prefetch_feature_list;

/*********************************************************************************************************************/
/*                          Intel 0x8F prefetchers                                                                   */
/*********************************************************************************************************************/

extern const _SysFeatureList likwid_sysft_intel_8f_cpu_feature_list;

/*********************************************************************************************************************/
/*                          Intel Knights Landing prefetchers                                                        */
/*********************************************************************************************************************/

extern const _SysFeatureList likwid_sysft_intel_knl_cpu_feature_list;

/*********************************************************************************************************************/
/*                          Intel Core2 prefetchers                                                                  */
/*********************************************************************************************************************/

extern const _SysFeatureList likwid_sysft_intel_core2_cpu_feature_list;

/*********************************************************************************************************************/
/*                          Intel Dynamic Acceleration                                                               */
/*********************************************************************************************************************/

extern const _SysFeatureList likwid_sysft_intel_cpu_ida_feature_list;

#endif /* HWFEATURES_X86_INTEL_PREFETCHER_H */
