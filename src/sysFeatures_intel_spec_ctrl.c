/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_intel_spec_ctrl.c
 *
 *      Description:  Interface to control CPU speculative execution behavior for the
 *                    sysFeatures component
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

#include <sysFeatures_types.h>
#include <sysFeatures_common.h>
#include <likwid.h>
#include <error.h>
#include <cpuid.h>
#include <bitUtil.h>
#include <registers.h>

#include "debug.h"

static cerr_t intel_cpu_spec_ibrs_tester(bool *ok)
{
    unsigned eax = 0x07, ebx = 0, ecx = 0, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    *ok = field32(edx, 26, 1);
    return NULL;
}

static cerr_t intel_cpu_spec_ibrs_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_SPEC_CTRL, 0, false, value);
}

static cerr_t intel_cpu_spec_stibp_tester(bool *ok)
{
    unsigned eax = 0x07, ebx = 0, ecx = 0, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    *ok = field32(edx, 27, 1);
    return NULL;
}

static cerr_t intel_cpu_spec_stibp_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_SPEC_CTRL, 1, false, value);
}

static cerr_t intel_cpu_spec_ssbd_tester(bool *ok)
{
    unsigned eax = 0x07, ebx = 0, ecx = 0, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    *ok = field32(edx, 31, 1);
    return NULL;
}

static cerr_t intel_cpu_spec_ssbd_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_SPEC_CTRL, 2, true, value);
}

static cerr_t intel_cpu_spec_ipred_dis_tester(bool *ok)
{
    unsigned eax = 0x07, ebx = 0, ecx = 0x02, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    *ok = field32(edx, 1, 1);
    return NULL;
}

static cerr_t intel_cpu_spec_ipred_dis_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_SPEC_CTRL, 3, false, value);
}

static cerr_t intel_cpu_spec_rrsba_dis_tester(bool *ok)
{
    unsigned eax = 0x07, ebx = 0, ecx = 0x02, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    *ok = field32(edx, 2, 1);
    return NULL;
}

static cerr_t intel_cpu_spec_rrsba_dis_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_SPEC_CTRL, 5, true, value);
}

static cerr_t intel_cpu_spec_psfd_tester(bool *ok)
{
    unsigned eax = 0x07, ebx = 0, edx = 0x02, ecx = 0;
    CPUID(eax, ebx, ecx, edx);
    *ok = field32(edx, 0, 1);
    return NULL;
}

static cerr_t intel_cpu_spec_psfd_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_SPEC_CTRL, 7, true, value);
}

static cerr_t intel_cpu_spec_ddpd_tester(bool *ok)
{
    unsigned eax = 0x07, ebx = 0, ecx = 0x02, edx = 0;
    CPUID(eax, ebx, ecx, edx);
    *ok = field32(edx, 3, 1);
    return NULL;
}

static cerr_t intel_cpu_spec_ddpd_getter(const LikwidDevice_t device, char** value)
{
    return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_SPEC_CTRL, 8, true, value);
}

static cerr_t intel_cpu_spec_ctrl(bool *ok)
{
    static const hwfeature_test_function funcs[] = {
        intel_cpu_spec_ibrs_tester,
        intel_cpu_spec_stibp_tester,
        intel_cpu_spec_ssbd_tester,
        intel_cpu_spec_ipred_dis_tester,
        intel_cpu_spec_rrsba_dis_tester,
        intel_cpu_spec_psfd_tester,
        intel_cpu_spec_ddpd_tester,
    };

    size_t valid = 0;
    for (size_t i = 0; i < ARRAY_COUNT(funcs); i++) {
        bool spec_ok;
        if (funcs[i](&spec_ok))
            return ERROR_APPEND("error in speculation tester function");
        if (spec_ok)
            valid++;
    }

    *ok = valid > 0;
    if (!*ok)
        PRINT_INFO("Intel speculation control not available");

    return NULL;
}

static _SysFeature intel_cpu_spec_ctrl_features[] = {
    {"ibrs", "spec_ctrl", "Indirect Branch Restricted Speculation", intel_cpu_spec_ibrs_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_ibrs_tester, NULL},
    {"stibp", "spec_ctrl", "Single Thread Indirect Branch Predictors", intel_cpu_spec_stibp_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_stibp_tester, NULL},
    {"ssbd", "spec_ctrl", "Speculative Store Bypass Disable", intel_cpu_spec_ssbd_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_ssbd_tester, NULL},
    {"ipred_dis", "spec_ctrl", "", intel_cpu_spec_ipred_dis_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_ipred_dis_tester, NULL},
    {"rrsba_dis", "spec_ctrl", "", intel_cpu_spec_rrsba_dis_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_rrsba_dis_tester, NULL},
    {"psfd", "spec_ctrl", "Fast Store Forwarding Predictor", intel_cpu_spec_psfd_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_psfd_tester, NULL},
    {"ddpd", "spec_ctrl", "Data Dependent Prefetcher", intel_cpu_spec_ddpd_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_ddpd_tester, NULL},
};

const _SysFeatureList likwid_sysft_intel_cpu_spec_ctrl_feature_list = {
    .num_features = ARRAY_COUNT(intel_cpu_spec_ctrl_features),
    .tester = intel_cpu_spec_ctrl,
    .features = intel_cpu_spec_ctrl_features,
};
