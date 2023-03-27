#ifndef HWFEATURES_X86_INTEL_RAPL_H
#define HWFEATURES_X86_INTEL_RAPL_H


int intel_rapl_pkg_test();

int hwFeatures_intel_pkg_energy_status_test();
int hwFeatures_intel_pkg_energy_status_getter(LikwidDevice_t device, char** value);

int hwFeatures_intel_pkg_energy_limit_test();

int hwFeatures_intel_pkg_energy_limit_1_getter(LikwidDevice_t device, char** value);
int hwFeatures_intel_pkg_energy_limit_1_setter(LikwidDevice_t device, char* value);

int hwFeatures_intel_pkg_energy_limit_1_time_getter(LikwidDevice_t device, char** value);
int hwFeatures_intel_pkg_energy_limit_1_time_setter(LikwidDevice_t device, char* value);

int hwFeatures_intel_pkg_energy_limit_2_getter(LikwidDevice_t device, char** value);
int hwFeatures_intel_pkg_energy_limit_2_setter(LikwidDevice_t device, char* value);

int hwFeatures_intel_pkg_energy_limit_2_time_getter(LikwidDevice_t device, char** value);
int hwFeatures_intel_pkg_energy_limit_2_time_setter(LikwidDevice_t device, char* value);

int hwFeatures_intel_pkg_info_test();
int hwFeatures_intel_pkg_info_tdp(LikwidDevice_t device, char** value);
int hwFeatures_intel_pkg_info_min_power(LikwidDevice_t device, char** value);
int hwFeatures_intel_pkg_info_max_power(LikwidDevice_t device, char** value);
int hwFeatures_intel_pkg_info_max_time(LikwidDevice_t device, char** value);


#define MAX_INTEL_RAPL_PKG_FEATURES 9
static _HWFeature intel_rapl_pkg_features[] = {
    {"pkg_energy", "rapl", "Current energy consumtion (PKG domain)", hwFeatures_intel_pkg_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_pkg_energy_status_test, "uJ"},
    {"pkg_tdp", "rapl", "Thermal Spec Power", hwFeatures_intel_pkg_info_tdp, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_pkg_info_test, "mW"},
    {"pkg_min_limit", "rapl", "Minimum Power", hwFeatures_intel_pkg_info_min_power, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_pkg_info_test, "mW"},
    {"pkg_max_limit", "rapl", "Maximum Power", hwFeatures_intel_pkg_info_max_power, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_pkg_info_test, "mW"},
    {"pkg_max_time", "rapl", "Maximum Time", hwFeatures_intel_pkg_info_max_time, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_pkg_info_test, "ms"},
    {"pkg_limit_1", "rapl", "Long-term energy limit (PKG domain)", hwFeatures_intel_pkg_energy_limit_1_getter, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_pkg_energy_limit_test, "mW"},
    {"pkg_time_1", "rapl", "Long-term time window (PKG domain)", hwFeatures_intel_pkg_energy_limit_1_time_getter, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_pkg_energy_limit_test, "ms"},
    {"pkg_limit_2", "rapl", "Short-term energy limit (PKG domain)", hwFeatures_intel_pkg_energy_limit_2_getter, hwFeatures_intel_pkg_energy_limit_2_setter, DEVICE_TYPE_SOCKET, hwFeatures_intel_pkg_energy_limit_test, "mW"},
    {"pkg_time_2", "rapl", "Short-term time window (PKG domain)", hwFeatures_intel_pkg_energy_limit_2_time_getter, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_pkg_energy_limit_test, "ms"},
};

static _HWFeatureList intel_rapl_pkg_feature_list = {
    .num_features = MAX_INTEL_RAPL_PKG_FEATURES,
    .tester = intel_rapl_pkg_test,
    .features = intel_rapl_pkg_features,
};

int intel_rapl_dram_test();

int hwFeatures_intel_dram_energy_status_test();
int hwFeatures_intel_dram_energy_status_getter(LikwidDevice_t device, char** value);

int hwFeatures_intel_dram_energy_limit_test();
int hwFeatures_intel_dram_energy_limit_1_getter(LikwidDevice_t device, char** value);
int hwFeatures_intel_dram_energy_limit_1_setter(LikwidDevice_t device, char* value);
int hwFeatures_intel_dram_energy_limit_1_time_getter(LikwidDevice_t device, char** value);
int hwFeatures_intel_dram_energy_limit_1_time_setter(LikwidDevice_t device, char* value);


int hwFeatures_intel_dram_info_test();
int hwFeatures_intel_dram_info_tdp(LikwidDevice_t device, char** value);
int hwFeatures_intel_dram_info_min_power(LikwidDevice_t device, char** value);
int hwFeatures_intel_dram_info_max_power(LikwidDevice_t device, char** value);
int hwFeatures_intel_dram_info_max_time(LikwidDevice_t device, char** value);


#define MAX_INTEL_RAPL_DRAM_FEATURES 7
static _HWFeature intel_rapl_dram_features[] = {
    {"dram_energy", "rapl", "Current energy consumtion (DRAM domain)", hwFeatures_intel_dram_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_dram_energy_status_test, "uJ"},
    {"dram_tdp", "rapl", "Thermal Spec Power", hwFeatures_intel_dram_info_tdp, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_dram_info_test, "mW"},
    {"dram_min_limit", "rapl", "Minimum Power", hwFeatures_intel_dram_info_min_power, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_dram_info_test, "mW"},
    {"dram_max_limit", "rapl", "Maximum Power", hwFeatures_intel_dram_info_max_power, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_dram_info_test, "mW"},
    {"dram_max_time", "rapl", "Maximum Time", hwFeatures_intel_dram_info_max_time, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_dram_info_test, "ms"},
    {"dram_limit", "rapl", "Long-term energy limit (DRAM domain)", hwFeatures_intel_dram_energy_limit_1_getter, hwFeatures_intel_dram_energy_limit_1_setter, DEVICE_TYPE_SOCKET, hwFeatures_intel_dram_energy_limit_test, "mW"},
    {"dram_time", "rapl", "Long-term time window (DRAM domain)", hwFeatures_intel_dram_energy_limit_1_time_getter, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_dram_energy_limit_test, "ms"},

};

static _HWFeatureList intel_rapl_dram_feature_list = {
    .num_features = MAX_INTEL_RAPL_DRAM_FEATURES,
    .tester = intel_rapl_dram_test,
    .features = intel_rapl_dram_features,
};


int intel_rapl_pp0_test();

int hwFeatures_intel_pp0_energy_status_test();
int hwFeatures_intel_pp0_energy_status_getter(LikwidDevice_t device, char** value);

int hwFeatures_intel_pp0_energy_limit_test();
int hwFeatures_intel_pp0_energy_limit_1_getter(LikwidDevice_t device, char** value);
int hwFeatures_intel_pp0_energy_limit_1_setter(LikwidDevice_t device, char* value);
int hwFeatures_intel_pp0_energy_limit_1_time_getter(LikwidDevice_t device, char** value);
int hwFeatures_intel_pp0_energy_limit_1_time_setter(LikwidDevice_t device, char* value);

int hwFeatures_intel_pp0_policy_test();
int hwFeatures_intel_pp0_policy_getter(LikwidDevice_t device, char** value);
int hwFeatures_intel_pp0_policy_setter(LikwidDevice_t device, char* value);


#define MAX_INTEL_RAPL_PP0_FEATURES 4
static _HWFeature intel_rapl_pp0_features[] = {
    {"pp0_energy", "rapl", "Current energy consumtion (PP0 domain)", hwFeatures_intel_pp0_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_pp0_energy_status_test, "uJ"},
    {"pp0_limit", "rapl", "Long-term energy limit (PP0 domain)", hwFeatures_intel_pp0_energy_limit_1_getter, hwFeatures_intel_pp0_energy_limit_1_setter, DEVICE_TYPE_SOCKET, hwFeatures_intel_pp0_energy_limit_test, "mW"},
    {"pp0_time", "rapl", "Long-term time window (PP0 domain)", hwFeatures_intel_pp0_energy_limit_1_time_getter, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_pp0_energy_limit_test, "ms"},
    {"pp0_policy", "rapl", "Balance Power Policy (PP0 domain)", hwFeatures_intel_pp0_policy_getter, hwFeatures_intel_pp0_policy_setter, DEVICE_TYPE_SOCKET, hwFeatures_intel_pp0_policy_test},
};

static _HWFeatureList intel_rapl_pp0_feature_list = {
    .num_features = MAX_INTEL_RAPL_PP0_FEATURES,
    .tester = intel_rapl_pp0_test,
    .features = intel_rapl_pp0_features,
};


int intel_rapl_pp1_test();

int hwFeatures_intel_pp1_energy_status_test();
int hwFeatures_intel_pp1_energy_status_getter(LikwidDevice_t device, char** value);

int hwFeatures_intel_pp1_energy_limit_test();
int hwFeatures_intel_pp1_energy_limit_1_getter(LikwidDevice_t device, char** value);
int hwFeatures_intel_pp1_energy_limit_1_setter(LikwidDevice_t device, char* value);
int hwFeatures_intel_pp1_energy_limit_1_time_getter(LikwidDevice_t device, char** value);
int hwFeatures_intel_pp1_energy_limit_1_time_setter(LikwidDevice_t device, char* value);

int hwFeatures_intel_pp1_policy_test();
int hwFeatures_intel_pp1_policy_getter(LikwidDevice_t device, char** value);
int hwFeatures_intel_pp1_policy_setter(LikwidDevice_t device, char* value);



#define MAX_INTEL_RAPL_PP1_FEATURES 4
static _HWFeature intel_rapl_pp1_features[] = {
    {"pp1_energy", "rapl", "Current energy consumtion (PP1 domain)", hwFeatures_intel_pp1_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_pp1_energy_status_test, "uJ"},
    {"pp1_limit", "rapl", "Long-term energy limit (PP1 domain)", hwFeatures_intel_pp1_energy_limit_1_getter, hwFeatures_intel_pp1_energy_limit_1_setter, DEVICE_TYPE_SOCKET, hwFeatures_intel_pp1_energy_limit_test, "mW"},
    {"pp1_time", "rapl", "Long-term time window (PP1 domain)", hwFeatures_intel_pp1_energy_limit_1_time_getter, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_pp1_energy_limit_test, "ms"},
    {"pp1_policy", "rapl", "Balance Power Policy (PP1 domain)", hwFeatures_intel_pp1_policy_getter, hwFeatures_intel_pp1_policy_setter, DEVICE_TYPE_SOCKET, hwFeatures_intel_pp1_policy_test},
};

static _HWFeatureList intel_rapl_pp1_feature_list = {
    .num_features = MAX_INTEL_RAPL_PP1_FEATURES,
    .tester = intel_rapl_pp1_test,
    .features = intel_rapl_pp1_features,
};


int intel_rapl_psys_test();

int hwFeatures_intel_psys_energy_status_test();
int hwFeatures_intel_psys_energy_status_getter(LikwidDevice_t device, char** value);

int hwFeatures_intel_psys_energy_limit_test();
int hwFeatures_intel_psys_energy_limit_1_getter(LikwidDevice_t device, char** value);
int hwFeatures_intel_psys_energy_limit_1_setter(LikwidDevice_t device, char* value);
int hwFeatures_intel_psys_energy_limit_1_time_getter(LikwidDevice_t device, char** value);
int hwFeatures_intel_psys_energy_limit_1_time_setter(LikwidDevice_t device, char* value);





#define MAX_INTEL_RAPL_PSYS_FEATURES 3
static _HWFeature intel_rapl_psys_features[] = {
    {"psys_energy", "rapl", "Current energy consumtion (PSYS domain)", hwFeatures_intel_psys_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_psys_energy_status_test, "uJ"},
    {"psys_limit", "rapl", "Long-term energy limit (PSYS domain)", hwFeatures_intel_psys_energy_limit_1_getter, hwFeatures_intel_psys_energy_limit_1_setter, DEVICE_TYPE_SOCKET, hwFeatures_intel_psys_energy_limit_test, "mW"},
    {"psys_time", "rapl", "Long-term time window (PSYS domain)", hwFeatures_intel_psys_energy_limit_1_time_getter, NULL, DEVICE_TYPE_SOCKET, hwFeatures_intel_psys_energy_limit_test, "ms"},
};


static _HWFeatureList intel_rapl_psys_feature_list = {
    .num_features = MAX_INTEL_RAPL_PSYS_FEATURES,
    .tester = intel_rapl_psys_test,
    .features = intel_rapl_psys_features,
};


int hwFeatures_init_intel_rapl(_HWFeatureList* out);


#endif /* HWFEATURES_X86_INTEL_UNCOREFREQ_H */
