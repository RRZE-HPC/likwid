#ifndef HWFEATURES_X86_INTEL_PREFETCHER_H
#define HWFEATURES_X86_INTEL_PREFETCHER_H

/*********************************************************************************************************************/
/*                          Intel prefetchers                                                                        */
/*********************************************************************************************************************/
int intel_cpu_l2_hwpf_register_test(void);
int intel_cpu_l2_hwpf_getter(const LikwidDevice_t device, char** value);
int intel_cpu_l2_hwpf_setter(const LikwidDevice_t device, const char* value);
int intel_cpu_l2_adj_pf_getter(const LikwidDevice_t device, char** value);
int intel_cpu_l2_adj_pf_setter(const LikwidDevice_t device, const char* value);
int intel_cpu_l1_dcu_getter(const LikwidDevice_t device, char** value);
int intel_cpu_l1_dcu_setter(const LikwidDevice_t device, const char* value);
int intel_cpu_l1_dcu_ip_getter(const LikwidDevice_t device, char** value);
int intel_cpu_l1_dcu_ip_setter(const LikwidDevice_t device, const char* value);

#define MAX_INTEL_CPU_PREFETCH_FEATURES 4
static _SysFeature intel_cpu_prefetch_features[] = {
    {"l2_hwpf", "prefetch", "L2 Hardware Prefetcher", intel_cpu_l2_hwpf_getter, intel_cpu_l2_hwpf_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_l2_hwpf_register_test},
    {"l2_adj_pf", "prefetch", "L2 Adjacent Cache Line Prefetcher", intel_cpu_l2_adj_pf_getter, intel_cpu_l2_adj_pf_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_l2_hwpf_register_test},
    {"l1_dcu", "prefetch", "DCU Hardware Prefetcher", intel_cpu_l1_dcu_getter, intel_cpu_l1_dcu_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_l2_hwpf_register_test},
    {"l1_dcu_ip", "prefetch", "DCU IP Prefetcher", intel_cpu_l1_dcu_ip_getter, intel_cpu_l1_dcu_ip_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_l2_hwpf_register_test},
    //{"data_pf", "Data Dependent Prefetcher", DEVICE_TYPE_HWTHREAD},
};
static _SysFeatureList intel_cpu_prefetch_feature_list = {
    .num_features = MAX_INTEL_CPU_PREFETCH_FEATURES,
    .features = intel_cpu_prefetch_features,
};


/*********************************************************************************************************************/
/*                          Intel 0x8F prefetchers                                                                   */
/*********************************************************************************************************************/

int intel_cpu_l2_multipath_pf_getter(const LikwidDevice_t device, char** value);
int intel_cpu_l2_multipath_pf_setter(const LikwidDevice_t device, const char* value);

#define MAX_INTEL_8F_CPU_FEATURES 1
static _SysFeature intel_8f_cpu_features[] = {
    {"l2_multipath_pf", "prefetch", "L2 Adaptive Multipath Probability Prefetcher", intel_cpu_l2_multipath_pf_getter, intel_cpu_l2_multipath_pf_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_l2_hwpf_register_test}
};

static _SysFeatureList intel_8f_cpu_feature_list = {
    .num_features = MAX_INTEL_8F_CPU_FEATURES,
    .features = intel_8f_cpu_features,
};


/*********************************************************************************************************************/
/*                          Intel Knights Landing prefetchers                                                        */
/*********************************************************************************************************************/
int intel_knl_l1_dcu_getter(const LikwidDevice_t device, char** value);
int intel_knl_l1_dcu_setter(const LikwidDevice_t device, const char* value);
int intel_knl_l2_hwpf_getter(const LikwidDevice_t device, char** value);
int intel_knl_l2_hwpf_setter(const LikwidDevice_t device, const char* value);


#define MAX_INTEL_KNL_CPU_FEATURES 2
static _SysFeature intel_knl_cpu_prefetch_features[] = {
    {"l2_hwpf", "prefetch", "L2 Hardware Prefetcher", intel_knl_l2_hwpf_getter, intel_knl_l2_hwpf_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_l2_hwpf_register_test},
    {"l1_dcu", "prefetch", "DCU Hardware Prefetcher", intel_knl_l1_dcu_getter, intel_knl_l1_dcu_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_l2_hwpf_register_test},
};

static _SysFeatureList intel_knl_cpu_feature_list = {
    .num_features = MAX_INTEL_KNL_CPU_FEATURES,
    .features = intel_knl_cpu_prefetch_features,
};

/*********************************************************************************************************************/
/*                          Intel Core2 prefetchers                                                                  */
/*********************************************************************************************************************/
int intel_core2_l2_hwpf_register_test(void);
int intel_core2_l2_hwpf_getter(const LikwidDevice_t device, char** value);
int intel_core2_l2_hwpf_setter(const LikwidDevice_t device, const char* value);
int intel_core2_l2_adjpf_getter(const LikwidDevice_t device, char** value);
int intel_core2_l2_adjpf_setter(const LikwidDevice_t device, const char* value);
int intel_core2_l1_dcu_getter(const LikwidDevice_t device, char** value);
int intel_core2_l1_dcu_setter(const LikwidDevice_t device, const char* value);
int intel_core2_l1_dcu_ip_getter(const LikwidDevice_t device, char** value);
int intel_core2_l1_dcu_ip_setter(const LikwidDevice_t device, const char* value);

#define MAX_INTEL_CORE2_CPU_FEATURES 4
static _SysFeature intel_core2_cpu_prefetch_features[] = {
    {"hwpf", "prefetch", "Hardware prefetcher operation on streams of data", intel_core2_l2_hwpf_getter, intel_knl_l2_hwpf_setter, DEVICE_TYPE_HWTHREAD, intel_core2_l2_hwpf_register_test},
    {"adj_pf", "prefetch", "Adjacent Cache Line Prefetcher", intel_core2_l2_adjpf_getter, intel_core2_l2_adjpf_setter, DEVICE_TYPE_HWTHREAD, intel_core2_l2_hwpf_register_test},
    {"l1_dcu", "prefetch", "DCU L1 data cache prefetcher", intel_core2_l1_dcu_getter, intel_core2_l1_dcu_setter, DEVICE_TYPE_HWTHREAD, intel_core2_l2_hwpf_register_test},
    {"l1_dcu_ip", "prefetch", "DCU IP Prefetcher", intel_core2_l1_dcu_ip_getter, intel_core2_l1_dcu_ip_setter, DEVICE_TYPE_HWTHREAD, intel_core2_l2_hwpf_register_test},
};

static _SysFeatureList intel_core2_cpu_feature_list = {
    .num_features = MAX_INTEL_CORE2_CPU_FEATURES,
    .features = intel_core2_cpu_prefetch_features,
};


/*********************************************************************************************************************/
/*                          Intel Dynamic Acceleration                                                               */
/*********************************************************************************************************************/

int intel_core2_ida_tester(void);
int intel_core2_ida_getter(const LikwidDevice_t device, char** value);
int intel_core2_ida_setter(const LikwidDevice_t device, const char* value);

#define MAX_INTEL_CPU_IDA_FEATURES 1
static _SysFeature intel_cpu_ida_features[] = {
    {"ida", "prefetch", "Intel Dynamic Acceleration", intel_core2_ida_getter, intel_core2_ida_setter, DEVICE_TYPE_HWTHREAD, intel_core2_ida_tester},
};

static _SysFeatureList intel_cpu_ida_feature_list = {
    .num_features = MAX_INTEL_CPU_IDA_FEATURES,
    .tester = intel_core2_ida_tester,
    .features = intel_cpu_ida_features,
};

#endif /* HWFEATURES_X86_INTEL_PREFETCHER_H */
