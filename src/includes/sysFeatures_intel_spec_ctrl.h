#ifndef HWFEATURES_X86_INTEL_SPEC_CTRL_H
#define HWFEATURES_X86_INTEL_SPEC_CTRL_H

/*********************************************************************************************************************/
/*                          Intel speculation control                                                                */
/*********************************************************************************************************************/

int intel_cpu_spec_ibrs_tester();
int intel_cpu_spec_ibrs_getter(LikwidDevice_t device, char** value);
/*int intel_cpu_spec_ibrs_setter(LikwidDevice_t device, char** value);*/


int intel_cpu_spec_stibp_tester();
int intel_cpu_spec_stibp_getter(LikwidDevice_t device, char** value);
/*int intel_cpu_spec_stibp_setter(LikwidDevice_t device, char** value);*/

int intel_cpu_spec_ssbd_tester();
int intel_cpu_spec_ssbd_getter(LikwidDevice_t device, char** value);
/*int intel_cpu_spec_ssbd_setter(LikwidDevice_t device, char** value);*/

int intel_cpu_spec_ipred_dis_tester();
int intel_cpu_spec_ipred_dis_getter(LikwidDevice_t device, char** value);
/*int intel_cpu_spec_ipred_dis_setter(LikwidDevice_t device, char** value);*/


int intel_cpu_spec_rrsba_dis_tester();
int intel_cpu_spec_rrsba_dis_getter(LikwidDevice_t device, char** value);
/*int intel_cpu_spec_rrsba_dis_setter(LikwidDevice_t device, char** value);*/


int intel_cpu_spec_psfd_tester();
int intel_cpu_spec_psfd_getter(LikwidDevice_t device, char** value);
/*int intel_cpu_spec_psfd_setter(LikwidDevice_t device, char** value);*/


int intel_cpu_spec_ddpd_tester();
int intel_cpu_spec_ddpd_getter(LikwidDevice_t device, char** value);
/*int intel_cpu_spec_ddpd_setter(LikwidDevice_t device, char** value);*/


int intel_cpu_spec_ctrl();

#define MAX_INTEL_CPU_SPEC_CTRL_FEATURES 7
static _SysFeature intel_cpu_spec_ctrl_features[] = {
    {"ibrs", "spec_ctrl", "Indirect Branch Restricted Speculation", intel_cpu_spec_ibrs_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_ibrs_tester},
    {"stibp", "spec_ctrl", "Single Thread Indirect Branch Predictors", intel_cpu_spec_stibp_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_stibp_tester},
    {"ssbd", "spec_ctrl", "Speculative Store Bypass Disable", intel_cpu_spec_ssbd_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_ssbd_tester},
    {"ipred_dis", "spec_ctrl", "", intel_cpu_spec_ipred_dis_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_ipred_dis_tester},
    {"rrsba_dis", "spec_ctrl", "", intel_cpu_spec_rrsba_dis_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_rrsba_dis_tester},
    {"psfd", "spec_ctrl", "Fast Store Forwarding Predictor", intel_cpu_spec_psfd_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_psfd_tester},
    {"ddpd", "spec_ctrl", "Data Dependent Prefetcher", intel_cpu_spec_ddpd_getter, NULL, DEVICE_TYPE_HWTHREAD, intel_cpu_spec_ddpd_tester},
};
static _SysFeatureList intel_cpu_spec_ctrl_feature_list = {
    .num_features = MAX_INTEL_CPU_SPEC_CTRL_FEATURES,
    .tester = intel_cpu_spec_ctrl,
    .features = intel_cpu_spec_ctrl_features,
};


int sysFeatures_init_intel_spec_ctrl(_SysFeatureList* out);

#endif /* HWFEATURES_X86_INTEL_SPEC_CTRL_H */
