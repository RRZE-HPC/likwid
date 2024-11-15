/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_types.h
 *
 *      Description:  Internal types of the sysFeatures component
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


#ifndef HWFEATURES_TYPES_H
#define HWFEATURES_TYPES_H

#include <stdint.h>
#include <likwid.h>
#include <likwid_device.h>

typedef enum {
    HWFEATURES_TYPE_UINT64 = 0,
    HWFEATURES_TYPE_DOUBLE,
    HWFEATURES_TYPE_STRING
} HWFEATURES_VALUE_TYPES;


typedef int (*hwfeature_getter_function)(const LikwidDevice_t device, char** value);
typedef int (*hwfeature_setter_function)(const LikwidDevice_t device, const char* value);
typedef int (*hwfeature_test_function)(void);

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
    int max_stepping;
} _HWArchFeatures;

#define IS_VALID_DEVICE_TYPE(scope) (((scope) >= MIN_DEVICE_TYPE) && ((scope) < MAX_DEVICE_TYPE))


#endif /* HWFEATURES_TYPES_H */
