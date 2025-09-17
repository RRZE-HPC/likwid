/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_intel_uncorefreq.c
 *
 *      Description:  Interface to control Intel Uncore frequency for the sysFeatures component
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

#include <sysFeatures_intel_uncorefreq.h>

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <sysFeatures_intel.h>
#include <access.h>
#include <sysFeatures_common.h>
#include <registers.h>
#include <bitUtil.h>

#include "error_ng.h"

static cerr_t intel_uncorefreq_test(bool *ok)
{
    if (likwid_sysft_foreach_socket_testmsr(ok, MSR_UNCORE_FREQ))
        return ERROR_WRAP();

    if (!*ok)
        return NULL;

    if (likwid_sysft_foreach_socket_testmsr(ok, MSR_UNCORE_FREQ_READ))
        return ERROR_WRAP();

    return NULL;
}

static cerr_t intel_uncore_cur_freq_getter(const LikwidDevice_t device, char** value)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t msrData;
    if (likwid_sysft_readmsr(device, MSR_UNCORE_FREQ_READ, &msrData))
        return ERROR_WRAP();
    return ERROR_WRAP_CALL(likwid_sysft_uint64_to_string(field64(msrData, 0, 8) * 100, value));
}

static cerr_t intel_uncore_min_freq_getter(const LikwidDevice_t device, char** value)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t msrData;
    if (likwid_sysft_readmsr(device, MSR_UNCORE_FREQ, &msrData))
        return ERROR_WRAP();
    return ERROR_WRAP_CALL(likwid_sysft_uint64_to_string(field64(msrData, 8, 8) * 100, value));
}

static cerr_t intel_uncore_max_freq_getter(const LikwidDevice_t device, char** value)
{
    assert(device->type == DEVICE_TYPE_SOCKET);

    uint64_t msrData;
    if (likwid_sysft_readmsr(device, MSR_UNCORE_FREQ, &msrData))
        return ERROR_WRAP();
    return ERROR_WRAP_CALL(likwid_sysft_uint64_to_string(field64(msrData, 0, 8) * 100, value));
}

static _SysFeature intel_uncorefreq_features[] = {
    {"cur_uncore_freq", "uncore_freq", "Current Uncore frequency", intel_uncore_cur_freq_getter, NULL, DEVICE_TYPE_SOCKET, NULL, "MHz"},
    {"min_uncore_freq", "uncore_freq", "Minimum Uncore frequency", intel_uncore_min_freq_getter, NULL, DEVICE_TYPE_SOCKET, NULL, "MHz"},
    {"max_uncore_freq", "uncore_freq", "Maximal Uncore frequency", intel_uncore_max_freq_getter, NULL, DEVICE_TYPE_SOCKET, NULL, "MHz"},
};

const _SysFeatureList likwid_sysft_intel_uncorefreq_feature_list = {
    .num_features = ARRAY_COUNT(intel_uncorefreq_features),
    .tester = intel_uncorefreq_test,
    .features = intel_uncorefreq_features,
};
