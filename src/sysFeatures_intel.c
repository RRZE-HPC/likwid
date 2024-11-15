/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_intel.c
 *
 *      Description:  Intel interface of the sysFeatures component
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

#include <registers.h>
#include <cpuid.h>
#include <pci_types.h>
#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <sysFeatures_common.h>
#include <topology.h>
#include <sysFeatures_intel.h>
#include <access.h>
#include <sysFeatures_intel_rapl.h>
#include <sysFeatures_intel_prefetcher.h>
#include <sysFeatures_intel_turbo.h>
#include <sysFeatures_intel_uncorefreq.h>
#include <sysFeatures_intel_spec_ctrl.h>

static const _HWArchFeatures intel_arch_features[];

int likwid_sysft_init_x86_intel(_SysFeatureList* out)
{
    int err = likwid_sysft_init_generic(intel_arch_features, out);
    if (err < 0)
    {
        ERROR_PRINT(Failed to init general Intel HWFetures);
        return err;
    }
    err = likwid_sysft_init_intel_rapl(out);
    if (err < 0)
    {
        ERROR_PRINT(Failed to init Intel RAPL HWFetures);
        return err;
    }

    return 0;
}

static const _SysFeatureList* intel_arch_feature_inputs[] = {
    &likwid_sysft_intel_cpu_prefetch_feature_list,
    &likwid_sysft_intel_cpu_ida_feature_list,
    &likwid_sysft_intel_cpu_turbo_feature_list,
    &likwid_sysft_intel_uncorefreq_feature_list,
    &likwid_sysft_intel_cpu_spec_ctrl_feature_list,
    NULL,
};

static const _SysFeatureList* intel_8f_arch_feature_inputs[] = {
    &likwid_sysft_intel_cpu_prefetch_feature_list,
    &likwid_sysft_intel_8f_cpu_feature_list,
    &likwid_sysft_intel_cpu_ida_feature_list,
    &likwid_sysft_intel_cpu_turbo_feature_list,
    &likwid_sysft_intel_cpu_spec_ctrl_feature_list,
    NULL,
};

static const _SysFeatureList* intel_knl_arch_feature_inputs[] = {
    &likwid_sysft_intel_knl_cpu_feature_list,
    &likwid_sysft_intel_cpu_ida_feature_list,
    &likwid_sysft_intel_cpu_turbo_feature_list,
    &likwid_sysft_intel_cpu_spec_ctrl_feature_list,
    NULL,
};

static const _SysFeatureList* intel_core2_arch_feature_inputs[] = {
    &likwid_sysft_intel_core2_cpu_feature_list,
    &likwid_sysft_intel_cpu_ida_feature_list,
    &likwid_sysft_intel_cpu_turbo_feature_list,
    &likwid_sysft_intel_cpu_spec_ctrl_feature_list,
    NULL,
};

static const _HWArchFeatures intel_arch_features[] = {
    {P6_FAMILY, SANDYBRIDGE, intel_arch_feature_inputs},
    {P6_FAMILY, SANDYBRIDGE_EP, intel_arch_feature_inputs},
    {P6_FAMILY, IVYBRIDGE, intel_arch_feature_inputs},
    {P6_FAMILY, IVYBRIDGE_EP, intel_arch_feature_inputs},
    {P6_FAMILY, HASWELL, intel_arch_feature_inputs},
    {P6_FAMILY, HASWELL_EP, intel_arch_feature_inputs},
    {P6_FAMILY, HASWELL_M1, intel_arch_feature_inputs},
    {P6_FAMILY, HASWELL_M2, intel_arch_feature_inputs},
    {P6_FAMILY, BROADWELL, intel_arch_feature_inputs},
    {P6_FAMILY, BROADWELL_E, intel_arch_feature_inputs},
    {P6_FAMILY, BROADWELL_D, intel_arch_feature_inputs},
    {P6_FAMILY, BROADWELL_E3, intel_arch_feature_inputs},
    {P6_FAMILY, SKYLAKE1, intel_arch_feature_inputs},
    {P6_FAMILY, SKYLAKE2, intel_arch_feature_inputs},
    {P6_FAMILY, SKYLAKEX, intel_arch_feature_inputs},
    {P6_FAMILY, 0x8F, intel_8f_arch_feature_inputs},
    {P6_FAMILY, KABYLAKE1, intel_arch_feature_inputs},
    {P6_FAMILY, KABYLAKE2, intel_arch_feature_inputs},
    {P6_FAMILY, CANNONLAKE, intel_arch_feature_inputs},
    {P6_FAMILY, ICELAKE1, intel_arch_feature_inputs},
    {P6_FAMILY, ICELAKE2, intel_arch_feature_inputs},
    {P6_FAMILY, ROCKETLAKE, intel_arch_feature_inputs},
    {P6_FAMILY, COMETLAKE1, intel_arch_feature_inputs},
    {P6_FAMILY, COMETLAKE2, intel_arch_feature_inputs},
    {P6_FAMILY, ICELAKEX1, intel_arch_feature_inputs},
    {P6_FAMILY, ICELAKEX2, intel_arch_feature_inputs},
    {P6_FAMILY, SAPPHIRERAPIDS, intel_arch_feature_inputs},
    {P6_FAMILY, SNOWRIDGEX, intel_arch_feature_inputs},
    {P6_FAMILY, TIGERLAKE1, intel_arch_feature_inputs},
    {P6_FAMILY, TIGERLAKE2, intel_arch_feature_inputs},
    {P6_FAMILY, XEON_PHI_KNL, intel_knl_arch_feature_inputs},
    {P6_FAMILY, XEON_PHI_KML, intel_knl_arch_feature_inputs},
    {P6_FAMILY, CORE2_45, intel_core2_arch_feature_inputs},
    {P6_FAMILY, CORE2_65, intel_core2_arch_feature_inputs},
    {-1, -1, NULL},
};
