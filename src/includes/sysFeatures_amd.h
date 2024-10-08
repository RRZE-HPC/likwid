#ifndef HWFEATURES_X86_AMD_H
#define HWFEATURES_X86_AMD_H

#include <registers.h>
#include <cpuid.h>
#include <topology.h>
#include <sysFeatures_types.h>
#include <likwid.h>
#include <sysFeatures_common.h>
#include <sysFeatures_amd_rapl.h>

int sysFeatures_init_x86_amd(_SysFeatureList* out);

int amd_cpu_l1_stream_getter(const LikwidDevice_t device, char** value);
int amd_cpu_l1_stream_setter(const LikwidDevice_t device, const char* value);
int amd_cpu_l1_stride_getter(const LikwidDevice_t device, char** value);
int amd_cpu_l1_stride_setter(const LikwidDevice_t device, const char* value);
int amd_cpu_l1_region_getter(const LikwidDevice_t device, char** value);
int amd_cpu_l1_region_setter(const LikwidDevice_t device, const char* value);
int amd_cpu_l2_stream_getter(const LikwidDevice_t device, char** value);
int amd_cpu_l2_stream_setter(const LikwidDevice_t device, const char* value);
int amd_cpu_up_down_getter(const LikwidDevice_t device, char** value);
int amd_cpu_up_down_setter(const LikwidDevice_t device, const char* value);

#define MAX_AMD_K19_CPU_PREFETCH_FEATURES 5
static _SysFeature amd_k19_cpu_prefetch_features[] = {
    {"l1_stream", "prefetch", "Stream prefetcher that uses history of memory access patterns to fetch additional sequential lines into L1 cache", amd_cpu_l1_stream_getter, amd_cpu_l1_stream_setter, DEVICE_TYPE_HWTHREAD},
    {"l1_stride", "prefetch", "Stride prefetcher that uses memory access history of individual instructions to fetch additional lines into L1 cache when each access is a constant distance from the previous", amd_cpu_l1_stride_getter, amd_cpu_l1_stride_setter, DEVICE_TYPE_HWTHREAD},
    {"l1_region", "prefetch", "Prefetcher that uses memory access history to fetch additional lines into L1 cache when the data access for a given instruction tends to be followed by a consistent pattern of other accesses within a localized region", amd_cpu_l1_region_getter, amd_cpu_l1_region_setter, DEVICE_TYPE_HWTHREAD},
    {"L2_stream", "prefetch", "Stream prefetcher that uses history of memory access patterns to fetch additional sequential lines into L2 cache", amd_cpu_l2_stream_getter, amd_cpu_l2_stream_setter, DEVICE_TYPE_HWTHREAD},
    {"up_down", "prefetch", "Prefetcher that uses memory access history to determine whether to fetch the next or previous line into L2 cache for all memory accesses", amd_cpu_up_down_getter, amd_cpu_up_down_setter, DEVICE_TYPE_HWTHREAD},
};

static _SysFeatureList amd_k19_cpu_prefetch_feature_list = {
    .num_features = MAX_AMD_K19_CPU_PREFETCH_FEATURES,
    .features = amd_k19_cpu_prefetch_features,
};

int amd_cpu_spec_ibrs_getter(const LikwidDevice_t device, char** value);
int amd_cpu_spec_ibrs_setter(const LikwidDevice_t device, const char* value);
int amd_cpu_spec_stibp_getter(const LikwidDevice_t device, char** value);
int amd_cpu_spec_stibp_setter(const LikwidDevice_t device, const char* value);
int amd_cpu_spec_ssbd_getter(const LikwidDevice_t device, char** value);
int amd_cpu_spec_ssbd_setter(const LikwidDevice_t device, const char* value);
int amd_cpu_spec_pfsd_getter(const LikwidDevice_t device, char** value);
int amd_cpu_spec_pfsd_setter(const LikwidDevice_t device, const char* value);

#define MAX_AMD_K19_CPU_SPECULATION_FEATURES 4
static _SysFeature amd_k19_cpu_speculation_features[] = {
    {"ibrs", "spec_ctrl", "Indirect branch restriction speculation", amd_cpu_spec_ibrs_getter, amd_cpu_spec_ibrs_setter, DEVICE_TYPE_HWTHREAD},
    {"stibp", "spec_ctrl", "Single thread indirect branch predictor", amd_cpu_spec_stibp_getter, amd_cpu_spec_stibp_setter, DEVICE_TYPE_HWTHREAD},
    {"ssbd", "spec_ctrl", "Speculative Store Bypass", amd_cpu_spec_ssbd_getter, amd_cpu_spec_ssbd_setter, DEVICE_TYPE_HWTHREAD},
    {"psfd", "spec_ctrl", "Predictive Store Forwarding", amd_cpu_spec_pfsd_getter, amd_cpu_spec_pfsd_setter, DEVICE_TYPE_HWTHREAD},
};

static _SysFeatureList amd_k19_cpu_speculation_feature_list = {
    .num_features = MAX_AMD_K19_CPU_SPECULATION_FEATURES,
    .features = amd_k19_cpu_speculation_features,
};

int amd_cpu_flush_l1(const LikwidDevice_t device, const char* value);

#define MAX_AMD_K19_CPU_L1DFLUSH_FEATURES 1
static _SysFeature amd_k19_cpu_l1dflush_features[] = {
    {"l1dflush", "cache", "Performs a write-back and invalidate of the L1 data cache", NULL, amd_cpu_flush_l1, DEVICE_TYPE_HWTHREAD},
};

static _SysFeatureList amd_k19_cpu_l1dflush_feature_list = {
    .num_features = MAX_AMD_K19_CPU_L1DFLUSH_FEATURES,
    .features = amd_k19_cpu_l1dflush_features,
};

int amd_cpu_hwconfig_cpddis_getter(const LikwidDevice_t device, char** value);
int amd_cpu_hwconfig_cpddis_setter(const LikwidDevice_t device, const char* value);

#define MAX_AMD_K19_CPU_HWCONFIG_FEATURES 1
static _SysFeature amd_k19_cpu_hwconfig_features[] = {
    {"TurboMode", "cpufreq", "Specifies whether core performance boost is requested to be enabled or disabled", amd_cpu_hwconfig_cpddis_getter, amd_cpu_hwconfig_cpddis_setter, DEVICE_TYPE_HWTHREAD},
};

static _SysFeatureList amd_k19_cpu_hwconfig_feature_list = {
    .num_features = MAX_AMD_K19_CPU_HWCONFIG_FEATURES,
    .features = amd_k19_cpu_hwconfig_features,
};

static _SysFeatureList* amd_k19_cpu_feature_inputs[] = {
    &amd_k19_cpu_prefetch_feature_list,
    &amd_k19_cpu_speculation_feature_list,
    &amd_k19_cpu_l1dflush_feature_list,
    &amd_k19_cpu_hwconfig_feature_list,
    NULL,
};

static _HWArchFeatures amd_arch_features[] = {
    {ZEN3_FAMILY, ZEN4_RYZEN, amd_k19_cpu_feature_inputs},
    {ZEN3_FAMILY, ZEN4_RYZEN_PRO, amd_k19_cpu_feature_inputs},
    {ZEN3_FAMILY, ZEN4_EPYC, amd_k19_cpu_feature_inputs},
    {-1, -1, NULL},
};

#endif
