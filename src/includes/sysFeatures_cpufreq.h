#ifndef HWFEATURES_CPUFREQ_H
#define HWFEATURES_CPUFREQ_H

int sysFeatures_init_cpufreq(_HWFeatureList* out);


int cpufreq_acpi_test();
int cpufreq_acpi_cur_cpu_freq_getter(LikwidDevice_t device, char** value);
int cpufreq_acpi_min_cpu_freq_getter(LikwidDevice_t device, char** value);
int cpufreq_acpi_max_cpu_freq_getter(LikwidDevice_t device, char** value);
int cpufreq_acpi_avail_cpu_freqs_getter(LikwidDevice_t device, char** value);
int cpufreq_acpi_governor_getter(LikwidDevice_t device, char** value);
int cpufreq_acpi_governor_setter(LikwidDevice_t device, char* value);
int cpufreq_acpi_avail_governors_getter(LikwidDevice_t device, char** value);



#define MAX_CPUFREQ_ACPI_CPU_FEATURES 6
static _HWFeature cpufreq_acpi_features[] = {
    {"cur_cpu_freq", "cpu_freq", "Current CPU frequency", cpufreq_acpi_cur_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"min_cpu_freq", "cpu_freq", "Minimal CPU frequency", cpufreq_acpi_min_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"max_cpu_freq", "cpu_freq", "Maximal CPU frequency", cpufreq_acpi_max_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"avail_freqs", "cpu_freq", "Available CPU frequencies", cpufreq_acpi_cur_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"governor", "cpu_freq", "CPU frequency governor", cpufreq_acpi_governor_getter, cpufreq_acpi_governor_setter, DEVICE_TYPE_HWTHREAD},
    {"avail_governors", "cpu_freq", "Available CPU frequency governor", cpufreq_acpi_avail_governors_getter, NULL, DEVICE_TYPE_HWTHREAD},
};

static _HWFeatureList cpufreq_acpi_feature_list = {
    .num_features = MAX_CPUFREQ_ACPI_CPU_FEATURES,
    .tester = cpufreq_acpi_test,
    .features = cpufreq_acpi_features,
};

int cpufreq_intel_pstate_test();
int cpufreq_intel_pstate_base_cpu_freq_getter(LikwidDevice_t device, char** value);
int cpufreq_intel_pstate_cur_cpu_freq_getter(LikwidDevice_t device, char** value);
int cpufreq_intel_pstate_min_cpu_freq_getter(LikwidDevice_t device, char** value);
int cpufreq_intel_pstate_max_cpu_freq_getter(LikwidDevice_t device, char** value);
int cpufreq_intel_pstate_governor_getter(LikwidDevice_t device, char** value);
int cpufreq_intel_pstate_governor_setter(LikwidDevice_t device, char* value);
int cpufreq_intel_pstate_avail_governors_getter(LikwidDevice_t device, char** value);


#define MAX_CPUFREQ_PSTATE_CPU_FEATURES 6
static _HWFeature cpufreq_pstate_features[] = {
    {"base_freq", "cpu_freq", "Base CPU frequency", cpufreq_intel_pstate_base_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"cur_cpu_freq", "cpu_freq", "Current CPU frequency", cpufreq_intel_pstate_cur_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"min_cpu_freq", "cpu_freq", "Minimal CPU frequency", cpufreq_intel_pstate_min_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"max_cpu_freq", "cpu_freq", "Maximal CPU frequency", cpufreq_intel_pstate_max_cpu_freq_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"governor", "cpu_freq", "CPU frequency governor", cpufreq_intel_pstate_governor_getter, cpufreq_intel_pstate_governor_setter, DEVICE_TYPE_HWTHREAD},
    {"avail_freqs", "cpu_freq", "Available CPU frequencies", cpufreq_intel_pstate_avail_governors_getter, NULL, DEVICE_TYPE_HWTHREAD},
};

static _HWFeatureList cpufreq_pstate_feature_list = {
    .num_features = MAX_CPUFREQ_PSTATE_CPU_FEATURES,
    .tester = cpufreq_intel_pstate_test,
    .features = cpufreq_pstate_features,
};

int cpufreq_epp_test();
int cpufreq_intel_pstate_epp_getter(LikwidDevice_t device, char** value);
int cpufreq_intel_pstate_avail_epps_getter(LikwidDevice_t device, char** value);


#define MAX_CPUFREQ_EPP_FEATURES 2
static _HWFeature cpufreq_epp_features[] = {
    {"epp", "cpu_freq", "Current energy performance preference", cpufreq_intel_pstate_epp_getter, NULL, DEVICE_TYPE_HWTHREAD},
    {"avail_epps", "cpu_freq", "Available energy performance preferences", cpufreq_intel_pstate_epp_getter, NULL, DEVICE_TYPE_HWTHREAD},
};

static _HWFeatureList cpufreq_epp_feature_list = {
    .num_features = MAX_CPUFREQ_EPP_FEATURES,
    .tester = cpufreq_epp_test,
    .features = cpufreq_epp_features,
};


#endif /* HWFEATURES_CPUFREQ_H */

