#ifndef HWFEATURES_X86_INTEL_H
#define HWFEATURES_X86_INTEL_H

#include <registers.h>
#include <cpuid.h>
#include <topology.h>
#include <sysFeatures_types.h>
#include <likwid_device_types.h>


int intel_cpu_msr_register_getter(LikwidDevice_t device, uint32_t reg, uint64_t mask, uint64_t shift, int invert, char** value);

int intel_cpu_msr_register_setter(LikwidDevice_t device, uint32_t reg, uint64_t mask, uint64_t shift, int invert, char* value);

#include <sysFeatures_intel_prefetcher.h>
#include <sysFeatures_intel_turbo.h>
#include <sysFeatures_intel_uncorefreq.h>
#include <sysFeatures_intel_spec_ctrl.h>
#include <sysFeatures_intel_rapl.h>

static _HWFeatureList* intel_arch_feature_inputs[] = {
    &intel_cpu_prefetch_feature_list,
    &intel_cpu_ida_feature_list,
    &intel_cpu_turbo_feature_list,
    &intel_uncorefreq_feature_list,
    &intel_cpu_spec_ctrl_feature_list,
    NULL,
};

static _HWFeatureList* intel_8f_arch_feature_inputs[] = {
    &intel_cpu_prefetch_feature_list,
    &intel_8f_cpu_feature_list,
    &intel_cpu_ida_feature_list,
    &intel_cpu_turbo_feature_list,
    &intel_cpu_spec_ctrl_feature_list,
    NULL,
};

static _HWFeatureList* intel_knl_arch_feature_inputs[] = {
    &intel_knl_cpu_feature_list,
    &intel_cpu_ida_feature_list,
    &intel_cpu_turbo_feature_list,
    &intel_cpu_spec_ctrl_feature_list,
    NULL,
};

static _HWFeatureList* intel_core2_arch_feature_inputs[] = {
    &intel_core2_cpu_feature_list,
    &intel_cpu_ida_feature_list,
    &intel_cpu_turbo_feature_list,
    &intel_cpu_spec_ctrl_feature_list,
    NULL,
};


static _HWArchFeatures intel_arch_features[] = {
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



int sysFeatures_init_x86_intel(_HWFeatureList* list);

#endif
