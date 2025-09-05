/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_nvml.h
 *
 *      Description:  Interface to control various Nvidia NVML based features
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Authors:  Michael Panzlaff, michael.panzlaff@fau.de
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

#ifndef SYSFEATURES_NVML_H
#define SYSFEATURES_NVML_H

#include <sysFeatures_types.h>
#include "error_ng.h"

cerr_t likwid_sysft_init_nvml(_SysFeatureList *list);

#endif // SYSFEATURES_NVML_H
