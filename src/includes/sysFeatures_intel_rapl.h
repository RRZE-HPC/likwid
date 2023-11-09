#ifndef HWFEATURES_X86_INTEL_RAPL_H
#define HWFEATURES_X86_INTEL_RAPL_H


int intel_rapl_pkg_test();

int sysFeatures_intel_pkg_energy_status_test();
int sysFeatures_intel_pkg_energy_status_getter(LikwidDevice_t device, char** value);

int sysFeatures_intel_pkg_energy_limit_test();

int sysFeatures_intel_pkg_energy_limit_1_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pkg_energy_limit_1_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_pkg_energy_limit_1_time_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pkg_energy_limit_1_time_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_pkg_energy_limit_1_enable_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pkg_energy_limit_1_enable_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_pkg_energy_limit_1_clamp_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pkg_energy_limit_1_clamp_setter(LikwidDevice_t device, char* value);

int sysFeatures_intel_pkg_energy_limit_2_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pkg_energy_limit_2_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_pkg_energy_limit_2_time_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pkg_energy_limit_2_time_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_pkg_energy_limit_2_enable_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pkg_energy_limit_2_enable_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_pkg_energy_limit_2_clamp_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pkg_energy_limit_2_clamp_setter(LikwidDevice_t device, char* value);

int sysFeatures_intel_pkg_info_test();
int sysFeatures_intel_pkg_info_tdp(LikwidDevice_t device, char** value);
int sysFeatures_intel_pkg_info_min_power(LikwidDevice_t device, char** value);
int sysFeatures_intel_pkg_info_max_power(LikwidDevice_t device, char** value);
int sysFeatures_intel_pkg_info_max_time(LikwidDevice_t device, char** value);


#define MAX_INTEL_RAPL_PKG_FEATURES 13
static _HWFeature intel_rapl_pkg_features[] = {
    {"pkg_energy", "rapl", "Current energy consumtion (PKG domain)", sysFeatures_intel_pkg_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, sysFeatures_intel_pkg_energy_status_test, "uJ"},
    {"pkg_tdp", "rapl", "Thermal Spec Power", sysFeatures_intel_pkg_info_tdp, NULL, DEVICE_TYPE_SOCKET, sysFeatures_intel_pkg_info_test, "mW"},
    {"pkg_min_limit", "rapl", "Minimum Power", sysFeatures_intel_pkg_info_min_power, NULL, DEVICE_TYPE_SOCKET, sysFeatures_intel_pkg_info_test, "mW"},
    {"pkg_max_limit", "rapl", "Maximum Power", sysFeatures_intel_pkg_info_max_power, NULL, DEVICE_TYPE_SOCKET, sysFeatures_intel_pkg_info_test, "mW"},
    {"pkg_max_time", "rapl", "Maximum Time", sysFeatures_intel_pkg_info_max_time, NULL, DEVICE_TYPE_SOCKET, sysFeatures_intel_pkg_info_test, "ms"},
    {"pkg_limit_1", "rapl", "Long-term energy limit (PKG domain)", sysFeatures_intel_pkg_energy_limit_1_getter, sysFeatures_intel_pkg_energy_limit_1_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pkg_energy_limit_test, "mW"},
    {"pkg_limit_1_time", "rapl", "Long-term time window (PKG domain)", sysFeatures_intel_pkg_energy_limit_1_time_getter, sysFeatures_intel_pkg_energy_limit_1_time_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pkg_energy_limit_test, "ms"},
    {"pkg_limit_1_enable", "rapl", "Status of long-term energy limit (PKG domain)", sysFeatures_intel_pkg_energy_limit_1_enable_getter, sysFeatures_intel_pkg_energy_limit_1_enable_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pkg_energy_limit_test},
    {"pkg_limit_1_clamp", "rapl", "Clamping status of long-term energy limit (PKG domain)", sysFeatures_intel_pkg_energy_limit_1_clamp_getter, sysFeatures_intel_pkg_energy_limit_1_clamp_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pkg_energy_limit_test},
    {"pkg_limit_2", "rapl", "Short-term energy limit (PKG domain)", sysFeatures_intel_pkg_energy_limit_2_getter, sysFeatures_intel_pkg_energy_limit_2_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pkg_energy_limit_test, "mW"},
    {"pkg_limit_2_time", "rapl", "Short-term time window (PKG domain)", sysFeatures_intel_pkg_energy_limit_2_time_getter, sysFeatures_intel_pkg_energy_limit_2_time_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pkg_energy_limit_test, "ms"},
    {"pkg_limit_2_enable", "rapl", "Status of short-term energy limit (PKG domain)", sysFeatures_intel_pkg_energy_limit_2_enable_getter, sysFeatures_intel_pkg_energy_limit_2_enable_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pkg_energy_limit_test},
    {"pkg_limit_2_clamp", "rapl", "Clamping status of short-term energy limit (PKG domain)", sysFeatures_intel_pkg_energy_limit_2_clamp_getter, sysFeatures_intel_pkg_energy_limit_2_clamp_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pkg_energy_limit_test},
};

static _HWFeatureList intel_rapl_pkg_feature_list = {
    .num_features = MAX_INTEL_RAPL_PKG_FEATURES,
    .tester = intel_rapl_pkg_test,
    .features = intel_rapl_pkg_features,
};

int intel_rapl_dram_test();

int sysFeatures_intel_dram_energy_status_test();
int sysFeatures_intel_dram_energy_status_getter(LikwidDevice_t device, char** value);

int sysFeatures_intel_dram_energy_limit_test();
int sysFeatures_intel_dram_energy_limit_1_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_dram_energy_limit_1_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_dram_energy_limit_1_time_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_dram_energy_limit_1_time_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_dram_energy_limit_1_enable_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_dram_energy_limit_1_enable_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_dram_energy_limit_1_clamp_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_dram_energy_limit_1_clamp_setter(LikwidDevice_t device, char* value);

int sysFeatures_intel_dram_info_test();
int sysFeatures_intel_dram_info_tdp(LikwidDevice_t device, char** value);
int sysFeatures_intel_dram_info_min_power(LikwidDevice_t device, char** value);
int sysFeatures_intel_dram_info_max_power(LikwidDevice_t device, char** value);
int sysFeatures_intel_dram_info_max_time(LikwidDevice_t device, char** value);


#define MAX_INTEL_RAPL_DRAM_FEATURES 9
static _HWFeature intel_rapl_dram_features[] = {
    {"dram_energy", "rapl", "Current energy consumtion (DRAM domain)", sysFeatures_intel_dram_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, sysFeatures_intel_dram_energy_status_test, "uJ"},
    {"dram_tdp", "rapl", "Thermal Spec Power", sysFeatures_intel_dram_info_tdp, NULL, DEVICE_TYPE_SOCKET, sysFeatures_intel_dram_info_test, "mW"},
    {"dram_min_limit", "rapl", "Minimum Power", sysFeatures_intel_dram_info_min_power, NULL, DEVICE_TYPE_SOCKET, sysFeatures_intel_dram_info_test, "mW"},
    {"dram_max_limit", "rapl", "Maximum Power", sysFeatures_intel_dram_info_max_power, NULL, DEVICE_TYPE_SOCKET, sysFeatures_intel_dram_info_test, "mW"},
    {"dram_max_time", "rapl", "Maximum Time", sysFeatures_intel_dram_info_max_time, NULL, DEVICE_TYPE_SOCKET, sysFeatures_intel_dram_info_test, "ms"},
    {"dram_limit", "rapl", "Long-term energy limit (DRAM domain)", sysFeatures_intel_dram_energy_limit_1_getter, sysFeatures_intel_dram_energy_limit_1_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_dram_energy_limit_test, "mW"},
    {"dram_limit_time", "rapl", "Long-term time window (DRAM domain)", sysFeatures_intel_dram_energy_limit_1_time_getter, sysFeatures_intel_dram_energy_limit_1_time_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_dram_energy_limit_test, "ms"},
    {"dram_limit_enable", "rapl", "Status of long-term energy limit (DRAM domain)", sysFeatures_intel_dram_energy_limit_1_enable_getter, sysFeatures_intel_dram_energy_limit_1_enable_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_dram_energy_limit_test},
    {"dram_limit_clamp", "rapl", "Clamping status of long-term energy limit (DRAM domain)", sysFeatures_intel_dram_energy_limit_1_clamp_getter, sysFeatures_intel_dram_energy_limit_1_clamp_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_dram_energy_limit_test},
};

static _HWFeatureList intel_rapl_dram_feature_list = {
    .num_features = MAX_INTEL_RAPL_DRAM_FEATURES,
    .tester = intel_rapl_dram_test,
    .features = intel_rapl_dram_features,
};


int intel_rapl_pp0_test();

int sysFeatures_intel_pp0_energy_status_test();
int sysFeatures_intel_pp0_energy_status_getter(LikwidDevice_t device, char** value);

int sysFeatures_intel_pp0_energy_limit_test();
int sysFeatures_intel_pp0_energy_limit_1_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pp0_energy_limit_1_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_pp0_energy_limit_1_time_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pp0_energy_limit_1_time_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_pp0_energy_limit_1_enable_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pp0_energy_limit_1_enable_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_pp0_energy_limit_1_clamp_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pp0_energy_limit_1_clamp_setter(LikwidDevice_t device, char* value);

int sysFeatures_intel_pp0_policy_test();
int sysFeatures_intel_pp0_policy_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pp0_policy_setter(LikwidDevice_t device, char* value);


#define MAX_INTEL_RAPL_PP0_FEATURES 6
static _HWFeature intel_rapl_pp0_features[] = {
    {"pp0_energy", "rapl", "Current energy consumtion (PP0 domain)", sysFeatures_intel_pp0_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, sysFeatures_intel_pp0_energy_status_test, "uJ"},
    {"pp0_limit", "rapl", "Long-term energy limit (PP0 domain)", sysFeatures_intel_pp0_energy_limit_1_getter, sysFeatures_intel_pp0_energy_limit_1_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pp0_energy_limit_test, "mW"},
    {"pp0_limit_time", "rapl", "Long-term time window (PP0 domain)", sysFeatures_intel_pp0_energy_limit_1_time_getter, sysFeatures_intel_pp0_energy_limit_1_time_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pp0_energy_limit_test, "ms"},
    {"pp0_limit_enable", "rapl", "Status of long-term energy limit (PP0 domain)", sysFeatures_intel_pp0_energy_limit_1_enable_getter, sysFeatures_intel_pp0_energy_limit_1_enable_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pp0_energy_limit_test},
    {"pp0_limit_clamp", "rapl", "Clamping status of long-term energy limit (PP0 domain)", sysFeatures_intel_pp0_energy_limit_1_clamp_getter, sysFeatures_intel_pp0_energy_limit_1_clamp_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pp0_energy_limit_test},
    {"pp0_policy", "rapl", "Balance Power Policy (PP0 domain)", sysFeatures_intel_pp0_policy_getter, sysFeatures_intel_pp0_policy_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pp0_policy_test},
};

static _HWFeatureList intel_rapl_pp0_feature_list = {
    .num_features = MAX_INTEL_RAPL_PP0_FEATURES,
    .tester = intel_rapl_pp0_test,
    .features = intel_rapl_pp0_features,
};


int intel_rapl_pp1_test();

int sysFeatures_intel_pp1_energy_status_test();
int sysFeatures_intel_pp1_energy_status_getter(LikwidDevice_t device, char** value);

int sysFeatures_intel_pp1_energy_limit_test();
int sysFeatures_intel_pp1_energy_limit_1_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pp1_energy_limit_1_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_pp1_energy_limit_1_time_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pp1_energy_limit_1_time_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_pp1_energy_limit_1_enable_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pp1_energy_limit_1_enable_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_pp1_energy_limit_1_clamp_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pp1_energy_limit_1_clamp_setter(LikwidDevice_t device, char* value);

int sysFeatures_intel_pp1_policy_test();
int sysFeatures_intel_pp1_policy_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_pp1_policy_setter(LikwidDevice_t device, char* value);



#define MAX_INTEL_RAPL_PP1_FEATURES 6
static _HWFeature intel_rapl_pp1_features[] = {
    {"pp1_energy", "rapl", "Current energy consumtion (PP1 domain)", sysFeatures_intel_pp1_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, sysFeatures_intel_pp1_energy_status_test, "uJ"},
    {"pp1_limit", "rapl", "Long-term energy limit (PP1 domain)", sysFeatures_intel_pp1_energy_limit_1_getter, sysFeatures_intel_pp1_energy_limit_1_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pp1_energy_limit_test, "mW"},
    {"pp1_limit_time", "rapl", "Long-term time window (PP1 domain)", sysFeatures_intel_pp1_energy_limit_1_time_getter, sysFeatures_intel_pp1_energy_limit_1_time_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pp1_energy_limit_test, "ms"},
    {"pp1_limit_enable", "rapl", "Status of long-term energy limit (PP1 domain)", sysFeatures_intel_pp1_energy_limit_1_enable_getter, sysFeatures_intel_pp1_energy_limit_1_enable_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pp1_energy_limit_test},
    {"pp1_limit_clamp", "rapl", "Clamping status of long-term energy limit (PP1 domain)", sysFeatures_intel_pp1_energy_limit_1_clamp_getter, sysFeatures_intel_pp1_energy_limit_1_clamp_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pp1_energy_limit_test},
    {"pp1_policy", "rapl", "Balance Power Policy (PP1 domain)", sysFeatures_intel_pp1_policy_getter, sysFeatures_intel_pp1_policy_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_pp1_policy_test},
};

static _HWFeatureList intel_rapl_pp1_feature_list = {
    .num_features = MAX_INTEL_RAPL_PP1_FEATURES,
    .tester = intel_rapl_pp1_test,
    .features = intel_rapl_pp1_features,
};


int intel_rapl_psys_test();

int sysFeatures_intel_psys_energy_status_test();
int sysFeatures_intel_psys_energy_status_getter(LikwidDevice_t device, char** value);

int sysFeatures_intel_psys_energy_limit_test();
int sysFeatures_intel_psys_energy_limit_1_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_psys_energy_limit_1_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_psys_energy_limit_1_time_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_psys_energy_limit_1_time_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_psys_energy_limit_1_enable_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_psys_energy_limit_1_enable_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_psys_energy_limit_1_clamp_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_psys_energy_limit_1_clamp_setter(LikwidDevice_t device, char* value);

int sysFeatures_intel_psys_energy_limit_2_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_psys_energy_limit_2_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_psys_energy_limit_2_time_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_psys_energy_limit_2_time_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_psys_energy_limit_2_enable_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_psys_energy_limit_2_enable_setter(LikwidDevice_t device, char* value);
int sysFeatures_intel_psys_energy_limit_2_clamp_getter(LikwidDevice_t device, char** value);
int sysFeatures_intel_psys_energy_limit_2_clamp_setter(LikwidDevice_t device, char* value);



#define MAX_INTEL_RAPL_PSYS_FEATURES 9
static _HWFeature intel_rapl_psys_features[] = {
    {"psys_energy", "rapl", "Current energy consumtion (PSYS domain)", sysFeatures_intel_psys_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, sysFeatures_intel_psys_energy_status_test, "uJ"},
    {"psys_limit_1", "rapl", "Long-term energy limit (PSYS domain)", sysFeatures_intel_psys_energy_limit_1_getter, sysFeatures_intel_psys_energy_limit_1_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_psys_energy_limit_test, "mW"},
    {"psys_limit_1_time", "rapl", "Long-term time window (PSYS domain)", sysFeatures_intel_psys_energy_limit_1_time_getter, sysFeatures_intel_psys_energy_limit_1_time_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_psys_energy_limit_test, "ms"},
    {"psys_limit_1_enable", "rapl", "Status of long-term energy limit (PSYS domain)", sysFeatures_intel_psys_energy_limit_1_enable_getter, sysFeatures_intel_psys_energy_limit_1_enable_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_psys_energy_limit_test},
    {"psys_limit_1_clamp", "rapl", "Clamping status of long-term energy limit (PSYS domain)", sysFeatures_intel_psys_energy_limit_1_clamp_getter, sysFeatures_intel_psys_energy_limit_1_clamp_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_psys_energy_limit_test},
    {"psys_limit_2", "rapl", "Short-term energy limit (PSYS domain)", sysFeatures_intel_psys_energy_limit_2_getter, sysFeatures_intel_psys_energy_limit_2_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_psys_energy_limit_test, "mW"},
    {"psys_limit_2_time", "rapl", "Short-term time window (PSYS domain)", sysFeatures_intel_psys_energy_limit_2_time_getter, sysFeatures_intel_psys_energy_limit_2_time_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_psys_energy_limit_test, "ms"},
    {"psys_limit_2_enable", "rapl", "Status of short-term energy limit (PSYS domain)", sysFeatures_intel_psys_energy_limit_2_enable_getter, sysFeatures_intel_psys_energy_limit_2_enable_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_psys_energy_limit_test},
    {"psys_limit_2_clamp", "rapl", "Clamping status of short-term energy limit (PSYS domain)", sysFeatures_intel_psys_energy_limit_2_clamp_getter, sysFeatures_intel_psys_energy_limit_2_clamp_setter, DEVICE_TYPE_SOCKET, sysFeatures_intel_psys_energy_limit_test},
};


static _HWFeatureList intel_rapl_psys_feature_list = {
    .num_features = MAX_INTEL_RAPL_PSYS_FEATURES,
    .tester = intel_rapl_psys_test,
    .features = intel_rapl_psys_features,
};


int sysFeatures_init_intel_rapl(_HWFeatureList* out);


#endif /* HWFEATURES_X86_INTEL_UNCOREFREQ_H */
