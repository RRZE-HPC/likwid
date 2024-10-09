#ifndef HWFEATURES_X86_INTEL_UNCOREFREQ_H
#define HWFEATURES_X86_INTEL_UNCOREFREQ_H


int intel_uncorefreq_test(void);
int intel_uncore_cur_freq_getter(LikwidDevice_t device, char** value);
int intel_uncore_min_freq_getter(LikwidDevice_t device, char** value);
int intel_uncore_max_freq_getter(LikwidDevice_t device, char** value);


#define MAX_INTEL_UNCOREFREQ_FEATURES 3
static _SysFeature intel_uncorefreq_features[] = {
    {"cur_uncore_freq", "uncore_freq", "Current Uncore frequency", intel_uncore_cur_freq_getter, NULL, DEVICE_TYPE_SOCKET, NULL, "MHz"},
    {"min_uncore_freq", "uncore_freq", "Minimum Uncore frequency", intel_uncore_min_freq_getter, NULL, DEVICE_TYPE_SOCKET, NULL, "MHz"},
    {"max_uncore_freq", "uncore_freq", "Maximal Uncore frequency", intel_uncore_max_freq_getter, NULL, DEVICE_TYPE_SOCKET, NULL, "MHz"},
};

static _SysFeatureList intel_uncorefreq_feature_list = {
    .num_features = MAX_INTEL_UNCOREFREQ_FEATURES,
    .tester = intel_uncorefreq_test,
    .features = intel_uncorefreq_features,
};




#endif /* HWFEATURES_X86_INTEL_UNCOREFREQ_H */
