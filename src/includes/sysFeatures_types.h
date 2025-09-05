/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_types.h
 *
 *      Description:  Internal types of the sysFeatures component
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


#ifndef HWFEATURES_TYPES_H
#define HWFEATURES_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <likwid.h>
#include <likwid_device.h>

#include "error_ng.h"

typedef enum {
    HWFEATURES_TYPE_UINT64 = 0,
    HWFEATURES_TYPE_DOUBLE,
    HWFEATURES_TYPE_STRING
} HWFEATURES_VALUE_TYPES;


typedef cerr_t (*hwfeature_getter_function)(const LikwidDevice_t device, char** value);
typedef cerr_t (*hwfeature_setter_function)(const LikwidDevice_t device, const char* value);
typedef cerr_t (*hwfeature_test_function)(bool *ok);

typedef struct {
    const char* name;
    const char* category;
    const char* description;
    hwfeature_getter_function getter;
    hwfeature_setter_function setter;
    LikwidDeviceType type;
    hwfeature_test_function tester;
    const char* unit;
} _SysFeature;

typedef struct {
    int num_features;
    _SysFeature* features;
    hwfeature_test_function tester;
} _SysFeatureList;

typedef struct {
    int family;
    int model;
    const _SysFeatureList** features;
    //int max_stepping;
} _HWArchFeatures;

#endif /* HWFEATURES_TYPES_H */
