#ifndef HWFEATURES_X86_AMD_RAPL_H
#define HWFEATURES_X86_AMD_RAPL_H


int amd_rapl_pkg_test();

int sysFeatures_amd_pkg_energy_status_test();
int sysFeatures_amd_pkg_energy_status_getter(LikwidDevice_t device, char** value);



#define MAX_AMD_RAPL_PKG_FEATURES 1
static _SysFeature amd_rapl_pkg_features[] = {
    {"pkg_energy", "rapl", "Current energy consumtion (PKG domain)", sysFeatures_amd_pkg_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, sysFeatures_amd_pkg_energy_status_test, "J"},
};

static _SysFeatureList amd_rapl_pkg_feature_list = {
    .num_features = MAX_AMD_RAPL_PKG_FEATURES,
    .tester = amd_rapl_pkg_test,
    .features = amd_rapl_pkg_features,
};

int amd_rapl_core_test();

int sysFeatures_amd_core_energy_status_test();
int sysFeatures_amd_core_energy_status_getter(LikwidDevice_t device, char** value);


#define MAX_AMD_RAPL_CORE_FEATURES 1
static _SysFeature amd_rapl_core_features[] = {
    {"core_energy", "rapl", "Current energy consumtion (Core domain)", sysFeatures_amd_core_energy_status_getter, NULL, DEVICE_TYPE_CORE, sysFeatures_amd_core_energy_status_test, "J"},
};

static _SysFeatureList amd_rapl_core_feature_list = {
    .num_features = MAX_AMD_RAPL_CORE_FEATURES,
    .tester = amd_rapl_core_test,
    .features = amd_rapl_core_features,
};

int amd_rapl_l3_test();

int sysFeatures_amd_l3_energy_status_test();
int sysFeatures_amd_l3_energy_status_getter(LikwidDevice_t device, char** value);



#define MAX_AMD_RAPL_L3_FEATURES 1
static _SysFeature amd_rapl_l3_features[] = {
    {"l3_energy", "rapl", "Current energy consumtion (L3 domain)", sysFeatures_amd_l3_energy_status_getter, NULL, DEVICE_TYPE_SOCKET, sysFeatures_amd_l3_energy_status_test, "J"},
};

static _SysFeatureList amd_rapl_l3_feature_list = {
    .num_features = MAX_AMD_RAPL_L3_FEATURES,
    .tester = amd_rapl_l3_test,
    .features = amd_rapl_l3_features,
};



int sysFeatures_init_amd_rapl(_SysFeatureList* out);


#endif /* HWFEATURES_X86_AMD_RAPL_H */
