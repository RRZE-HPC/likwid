#ifndef HWFEATURES_X86_INTEL_TURBO_H
#define HWFEATURES_X86_INTEL_TURBO_H


int intel_cpu_turbo_test(void);
int intel_cpu_turbo_getter(const LikwidDevice_t device, char** value);
int intel_cpu_turbo_setter(const LikwidDevice_t device, const char* value);

#define MAX_INTEL_TURBO_CPU_FEATURES 1
static _SysFeature intel_cpu_turbo_features[] = {
    {"turbo", "cpu_freq", "Turbo mode", intel_cpu_turbo_getter, intel_cpu_turbo_setter, DEVICE_TYPE_HWTHREAD, intel_cpu_turbo_test},
};

static _SysFeatureList intel_cpu_turbo_feature_list = {
    .num_features = MAX_INTEL_TURBO_CPU_FEATURES,
    .tester = intel_cpu_turbo_test,
    .features = intel_cpu_turbo_features,
};




#endif /* HWFEATURES_X86_INTEL_TURBO_H */
