/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_common.c
 *
 *      Description:  Common functions used by the sysFeatures component
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

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include <bitUtil.h>
#include <access.h>
#include <registers.h>
#include <cpuid.h>
#include <pci_types.h>
#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <sysFeatures_common.h>
#include "debug.h"

cerr_t likwid_sysft_register_features(_SysFeatureList *features, const _SysFeatureList* in)
{
    if (in->tester) {
        bool avail;
        if (in->tester(&avail))
            return ERROR_WRAP_MSG("Sysfeaturelist %p raised an error while testing");
        if (!avail)
            return ERROR_SET("Sysfeaturelist %p is not available", in);
    }

    for (int i = 0; i < in->num_features; i++) {
        _SysFeature *f = &in->features[i];
        PRINT_DEBUG("Registering feature %s.%s", f->category, f->name);

        if (f->tester) {
            PRINT_DEBUG("Running test for feature %s.%s", f->category, f->name);

            bool avail;
            if (f->tester(&avail))
                return ERROR_WRAP_MSG("Test for feature %s.%s raised an error", f->category, f->name);

            if (avail) {
                PRINT_DEBUG("Test for feature %s.%s ok", f->category, f->name);

                if (_add_to_feature_list(features, f))
                    return ERROR_WRAP_MSG("Failed to add HW feature %s.%s to feature list", f->category, f->name);
            } else {
                PRINT_DEBUG("Test for feature %s.%s failed", f->category, f->name);
            }
        } else {
            PRINT_DEBUG("No test available for feature %s.%s", f->category, f->name);

            if (_add_to_feature_list(features, f))
                return ERROR_WRAP_MSG("Failed to add HW feature %s.%s to feature list", f->category, f->name);
        }
    }
    return NULL;
}

cerr_t likwid_sysft_init_generic(const _HWArchFeatures* infeatures, _SysFeatureList *list)
{
    CpuInfo_t cpuinfo = get_cpuInfo();

    const _SysFeatureList** feature_list = NULL;
    for (unsigned c = 0; infeatures[c].family >= 0 && infeatures[c].model >= 0; c++)
    {
        if ((unsigned)infeatures[c].family == cpuinfo->family && (unsigned)infeatures[c].model == cpuinfo->model)
        {
            PRINT_DEBUG("Using feature list for CPU family 0x%X and model 0x%X", cpuinfo->family, cpuinfo->model);
            feature_list = infeatures[c].features;
            break;
        }
    }

    if (!feature_list)
        return ERROR_SET("No architectural sysFeatures for family 0x%X and model 0x%X", cpuinfo->family, cpuinfo->model);

    for (unsigned j = 0; feature_list[j] != NULL; j++) {
        if (likwid_sysft_register_features(list, feature_list[j]))
            return ERROR_WRAP();
    }

    return NULL;
}

cerr_t likwid_sysft_uint64_to_string(uint64_t value, char** str)
{
    char s[HWFEATURES_MAX_STR_LENGTH];

    if (snprintf(s, sizeof(s), "%lu", value) < 0)
        return ERROR_SET_ERRNO("Conversion of uint64_t %lu failed");

    char *newstr = strdup(s);
    if (!newstr)
        return ERROR_SET_ERRNO("strdup failed");

    free(*str);
    *str = newstr;
    return NULL;
}

cerr_t likwid_sysft_string_to_uint64(const char* str, uint64_t* value)
{
    char *ptr = NULL;
    if (strncmp(str, "true", 4) == 0 || strncmp(str, "TRUE", 4) == 0) {
        *value = 1;
        return NULL;
    } else if (strncmp(str, "false", 5) == 0 || strncmp(str, "FALSE", 5) == 0) {
        *value = 0;
        return NULL;
    }

    errno = 0;
    uint64_t v = strtoull(str, &ptr, 0);
    if (v == 0 && errno != 0)
        return ERROR_SET_ERRNO("Conversion of string '%s' to uint64_t failed");

    *value = v;
    return NULL;
}

cerr_t likwid_sysft_double_to_string(double value, char **str)
{
    char s[HWFEATURES_MAX_STR_LENGTH];

    if (snprintf(s, sizeof(s), "%f", value) < 0)
        return ERROR_SET_ERRNO("Conversion of double %f failed", value);

    char *newstr = strdup(s);
    if (!newstr)
        return ERROR_SET_ERRNO("strdup failed");

    free(*str);
    *str = newstr;
    return NULL;
}

cerr_t likwid_sysft_string_to_double(const char* str, double *value)
{
    errno = 0;
    char *endptr = NULL;
    const double result = strtod(str, &endptr);
    if (!endptr)
        return ERROR_SET_ERRNO("Conversion of string '%s' to double failed", str);

    if (errno != 0)
        return ERROR_SET_ERRNO("Conversion of string '%s' to double failed", str);

    *value = result;
    return NULL;
}

cerr_t likwid_sysft_copystr(const char *str, char **value)
{
    if (!str)
        return ERROR_SET("cannot NULL string");

    char *newstr = strdup(str);
    if (!newstr)
        return ERROR_SET_ERRNO("strdup failed");

    free(*value);
    *value = newstr;
    return 0;
}

cerr_t likwid_sysft_foreach_core_testmsr(bool *ok, uint64_t reg)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_core_testmsr_cb(ok, reg, NULL, NULL));
}

cerr_t likwid_sysft_foreach_core_testmsr_cb(bool *ok, uint64_t reg, likwid_sysft_msr_test_func testFunc, void *cbData)
{
    int err = HPMinit();
    if (err < 0) {
        PRINT_WARN("HPMinit failed, MSRs (thus MSR %#lx) not available", reg);
        // If MSRs are generally not available, this particular MSR is also not available,
        // which we do not handle as error.
        *ok = false;
        return NULL;
    }

    CpuTopology_t topo = get_cpuTopology();
    const unsigned numCores = topo->numSockets * topo->numCoresPerSocket;

    unsigned valid = 0;
    for (unsigned i = 0; i < numCores; i++)
    {
        for (unsigned j = 0; j < topo->numHWThreads; j++)
        {
            HWThread* t = &topo->threadPool[j];
            if (t->coreId != i)
                continue;
            err = HPMaddThread(t->apicId);
            if (err < 0)
                continue;
            uint64_t msrData = 0;
            err = HPMread(t->apicId, MSR_DEV, reg, &msrData);
            if (err < 0)
                continue;
            if (testFunc)
            {
                bool msrOk;
                if (testFunc(&msrOk, msrData, cbData))
                    return ERROR_WRAP_MSG("testFunc error");
                if (msrOk)
                    valid += 1;
            }
            else
            {
                valid += 1;
            }
            break;
        }
    }

    *ok = (valid == numCores);
    return NULL;
}

cerr_t likwid_sysft_foreach_hwt_testmsr(bool *ok, uint64_t reg)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_hwt_testmsr_cb(ok, reg, NULL, NULL));
}

cerr_t likwid_sysft_foreach_hwt_testmsr_cb(bool *ok, uint64_t reg, likwid_sysft_msr_test_func testFunc, void *cbData)
{
    int err = HPMinit();
    if (err < 0) {
        PRINT_WARN("HPMinit failed, MSRs (thus MSR %#lx) not available", reg);
        *ok = false;
        return NULL;
    }

    CpuTopology_t topo = get_cpuTopology();

    unsigned valid = 0;
    for (unsigned j = 0; j < topo->numHWThreads; j++)
    {
        HWThread* t = &topo->threadPool[j];
        err = HPMaddThread(t->apicId);
        if (err < 0)
            continue;
        uint64_t msrData = 0;
        err = HPMread(t->apicId, MSR_DEV, reg, &msrData);
        if (err < 0)
            continue;
        if (testFunc)
        {
            bool msrOk;
            if (testFunc(&msrOk, msrData, cbData))
                return ERROR_WRAP_MSG("testFunc error");
            if (msrOk)
                valid += 1;
        }
        else
        {
            valid += 1;
        }
    }

    *ok = (valid == topo->numHWThreads);
    return NULL;
}

cerr_t likwid_sysft_foreach_socket_testmsr(bool *ok, uint64_t reg)
{
    return ERROR_WRAP_CALL(likwid_sysft_foreach_socket_testmsr_cb(ok, reg, NULL, NULL));
}

cerr_t likwid_sysft_foreach_socket_testmsr_cb(bool *ok, uint64_t reg, likwid_sysft_msr_test_func testFunc, void *cbData)
{
    int err = HPMinit();
    if (err < 0) {
        PRINT_WARN("HPMinit failed, MSRs (thus MSR %#lx) not available", reg);
        *ok = false;
        return NULL;
    }

    CpuTopology_t topo = get_cpuTopology();

    unsigned valid = 0;
    for (unsigned i = 0; i < topo->numSockets; i++)
    {
        for (unsigned j = 0; j < topo->numHWThreads; j++)
        {
            HWThread* t = &topo->threadPool[j];
            if (t->packageId != i)
                continue;
            err = HPMaddThread(t->apicId);
            if (err < 0)
                continue;
            uint64_t msrData = 0;
            err = HPMread(t->apicId, MSR_DEV, reg, &msrData);
            if (err < 0)
                continue;
            if (testFunc)
            {
                bool msrOk;
                if (testFunc(&msrOk, msrData, cbData))
                    return ERROR_WRAP_MSG("testFunc error");
                if (msrOk)
                    valid += 1;
            }
            else
            {
                valid += 1;
            }
            break;
        }
    }

    *ok = (valid == topo->numSockets);
    return NULL;
}

static cerr_t readmsr_socket(const LikwidDevice_t device, uint64_t reg, uint64_t *msrData)
{
    int err = 0;
    assert(device->type == DEVICE_TYPE_SOCKET);

    CpuTopology_t topo = get_cpuTopology();
    for (unsigned i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if ((int)t->packageId == device->id.simple.id && t->inCpuSet)
        {
            err = HPMaddThread(t->apicId);
            if (err < 0)
                continue;
            err = HPMread(t->apicId, MSR_DEV, reg, msrData);
            if (err < 0)
                continue;
            return NULL;
        }
    }

    if (err < 0)
        return ERROR_SET_LWERR(err, "HPMaddThread or HPMread failed");
    return ERROR_SET("Cannot read MSR for socket %d, for which no HWThread is available", device->id.simple.id);
}

static cerr_t writemsr_socket(const LikwidDevice_t device, uint64_t reg, uint64_t msrData)
{
    int err = 0;
    assert(device->type == DEVICE_TYPE_SOCKET);

    CpuTopology_t topo = get_cpuTopology();
    for (unsigned i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if ((int)t->packageId == device->id.simple.id && t->inCpuSet)
        {
            err = HPMaddThread(t->apicId);
            if (err < 0)
                continue;
            err = HPMwrite(t->apicId, MSR_DEV, reg, msrData);
            if (err < 0)
                continue;
            return 0;
        }
    }

    if (err < 0)
        return ERROR_SET_LWERR(err, "HPMaddThread or HPMwrite failed");
    return ERROR_SET("Cannot write MSR for socket %d, for which no HWThread is available", device->id.simple.id);
}

static cerr_t readmsr_core(const LikwidDevice_t device, uint64_t reg, uint64_t *msrData)
{
    int err = 0;
    assert(device->type == DEVICE_TYPE_CORE);

    CpuTopology_t topo = get_cpuTopology();
    for (unsigned i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if ((int)t->coreId == device->id.simple.id && t->inCpuSet)
        {
            err = HPMaddThread(t->apicId);
            if (err < 0)
                continue;
            err = HPMread(t->apicId, MSR_DEV, reg, msrData);
            if (err < 0)
                continue;
            return 0;
        }
    }

    if (err < 0)
        return ERROR_SET_LWERR(err, "HPMaddThread or HPMread failed");
    return ERROR_SET("Cannot read MSR for core %d, for which no HWThread is available", device->id.simple.id);
}

static cerr_t writemsr_core(const LikwidDevice_t device, uint64_t reg, uint64_t msrData)
{
    int err = 0;
    assert(device->type == DEVICE_TYPE_CORE);

    CpuTopology_t topo = get_cpuTopology();
    for (unsigned i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if ((int)t->coreId == device->id.simple.id && t->inCpuSet)
        {
            err = HPMaddThread(t->apicId);
            if (err < 0)
                continue;
            err = HPMwrite(t->apicId, MSR_DEV, reg, msrData);
            if (err < 0)
                continue;
            return 0;
        }
    }

    if (err < 0)
        return ERROR_SET_LWERR(err, "HPMaddThread or HPMwrite failed");
    return ERROR_SET("Cannot write MSR for core %d, for which no HWThread is available", device->id.simple.id);
}

static cerr_t readmsr_hwthread(const LikwidDevice_t device, uint64_t reg, uint64_t *msrData)
{
    assert(device->type == DEVICE_TYPE_HWTHREAD);
    int err = HPMaddThread(device->id.simple.id);
    if (err < 0)
        return ERROR_SET_LWERR(err, "HPMaddThread failed");
    err = HPMread(device->id.simple.id, MSR_DEV, reg, msrData);
    if (err < 0)
        return ERROR_SET_LWERR(err, "HPMread failed");
    return NULL;
}

static cerr_t writemsr_hwthread(const LikwidDevice_t device, uint64_t reg, uint64_t msrData)
{
    assert(device->type == DEVICE_TYPE_HWTHREAD);
    int err = HPMaddThread(device->id.simple.id);
    if (err < 0)
        return ERROR_SET_LWERR(err, "HPMaddThread failed");
    err = HPMwrite(device->id.simple.id, MSR_DEV, reg, msrData);
    if (err < 0)
        return ERROR_SET_LWERR(err, "HPMwrite failed");
    return NULL;
}

cerr_t likwid_sysft_readmsr(const LikwidDevice_t device, uint64_t reg, uint64_t *msrData)
{
    int err = HPMinit();
    if (err < 0)
        return ERROR_SET_LWERR(err, "HPMinit failed");

    switch (device->type)
    {
    case DEVICE_TYPE_SOCKET:
        if (readmsr_socket(device, reg, msrData))
            return ERROR_WRAP();
        break;
    case DEVICE_TYPE_CORE:
        if (readmsr_core(device, reg, msrData))
            return ERROR_WRAP();
        break;
    case DEVICE_TYPE_HWTHREAD:
        if (readmsr_hwthread(device, reg, msrData))
            return ERROR_WRAP();
        break;
    default:
        return ERROR_SET("Unimplemented device type: %d", device->type);
    }
    return NULL;
}

cerr_t likwid_sysft_readmsr_field(const LikwidDevice_t device, uint64_t reg, int bitoffset, int width, uint64_t *value)
{
    uint64_t msrData;
    if (likwid_sysft_readmsr(device, reg, &msrData))
        return ERROR_WRAP();
    *value = field64(msrData, bitoffset, width);
    return NULL;
}

cerr_t likwid_sysft_readmsr_bit_to_string(const LikwidDevice_t device, uint64_t reg, int bitoffset, bool invert, char **value)
{
    uint64_t field = false;
    if (likwid_sysft_readmsr_field(device, reg, bitoffset, 1, &field))
        return ERROR_WRAP();
    if (invert)
        field = !field;
    if (likwid_sysft_uint64_to_string(field, value))
        return ERROR_WRAP();
    return NULL;
}

cerr_t likwid_sysft_writemsr_field(const LikwidDevice_t device, uint64_t reg, int bitoffset, int width, uint64_t value)
{
    int err = HPMinit();
    if (err < 0)
        return ERROR_SET_LWERR(err, "HPMinit failed");

    uint64_t msrData;
    switch (device->type)
    {
    case DEVICE_TYPE_SOCKET:
        /* If we write the entire register, there is no need to fetch the old value first. */
        if (bitoffset != 0 || width != 64) {
            if (readmsr_socket(device, reg, &msrData))
                return ERROR_WRAP();
        } else {
            msrData = 0;
        }
        field64set(&msrData, bitoffset, width, value);
        if (writemsr_socket(device, reg, msrData))
            return ERROR_WRAP();
        break;
    case DEVICE_TYPE_CORE:
        if (bitoffset != 0 || width != 64) {
            if (readmsr_core(device, reg, &msrData))
                return ERROR_WRAP();
        } else {
            msrData = 0;
        }
        field64set(&msrData, bitoffset, width, value);
        if (writemsr_core(device, reg, msrData))
            return ERROR_WRAP();
        break;
    case DEVICE_TYPE_HWTHREAD:
        if (bitoffset != 0 || width != 64) {
            if (readmsr_hwthread(device, reg, &msrData))
                return ERROR_WRAP();
        } else {
            msrData = 0;
        }
        field64set(&msrData, bitoffset, width, value);
        if (writemsr_hwthread(device, reg, msrData))
            return ERROR_WRAP();
        break;
    default:
        return ERROR_SET("Unimplemented device type: %d", device->type);
    }

    return NULL;
}

cerr_t likwid_sysft_writemsr_bit_from_string(const LikwidDevice_t device, uint64_t reg, int bitoffset, bool invert, const char *value)
{
    uint64_t field;
    if (likwid_sysft_string_to_uint64(value, &field))
        return ERROR_WRAP();
    if (invert)
        field = !field;
    if (likwid_sysft_writemsr_field(device, reg, bitoffset, 1, field))
        return ERROR_WRAP();
    return NULL;
}
