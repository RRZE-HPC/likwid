/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_intel_turbo.c
 *
 *      Description:  Interface to control Intel CPU turbo mode for the sysFeatures component
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

#include <sysFeatures_intel_turbo.h>

#include <sysFeatures_intel.h>

#include <bitUtil.h>
#include <cpuid.h>
#include <error.h>
#include <registers.h>
#include <sysFeatures_common.h>

static int intel_cpu_turbo_test(void)
{
    uint32_t eax = 0x01, ebx, ecx = 0x0, edx;
    CPUID(eax, ebx, ecx, edx);
    if (field32(ecx, 7, 1) == 0)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Intel SpeedStep not supported by architecture);
        return 0;
    }

    return likwid_sysft_foreach_hwt_testmsr(MSR_IA32_MISC_ENABLE);
}

static int intel_cpu_turbo_getter(const LikwidDevice_t device, char** value)
{
    if (intel_cpu_turbo_test())
    {
        return likwid_sysft_readmsr_bit_to_string(device, MSR_IA32_MISC_ENABLE, 36, true, value);
    }
    return -ENOTSUP;
}

static int intel_cpu_turbo_setter(const LikwidDevice_t device, const char* value)
{
    if (intel_cpu_turbo_test())
    {
        return likwid_sysft_writemsr_bit_from_string(device, MSR_IA32_MISC_ENABLE, 36, true, value);
    }
    return -ENOTSUP;
}

static _SysFeature intel_cpu_turbo_features[] = {
    {"turbo", "cpu_freq", "Turbo mode", intel_cpu_turbo_getter, intel_cpu_turbo_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_turbo_test},
};

const _SysFeatureList likwid_sysft_intel_cpu_turbo_feature_list = {
    .num_features = ARRAY_COUNT(intel_cpu_turbo_features),
    .tester = intel_cpu_turbo_test,
    .features = intel_cpu_turbo_features,
};
