#ifndef HWFEATURES_NUMABALANCING_H
#define HWFEATURES_NUMABALANCING_H



int numa_balancing_procfs_getter(LikwidDevice_t device, char** value, char* sysfsfile);

int numa_balancing_test();
int numa_balancing_state_getter(LikwidDevice_t device, char** value);
int numa_balancing_scan_delay_getter(LikwidDevice_t device, char** value);
int numa_balancing_scan_period_min_getter(LikwidDevice_t device, char** value);
int numa_balancing_scan_period_max_getter(LikwidDevice_t device, char** value);
int numa_balancing_scan_size_getter(LikwidDevice_t device, char** value);

#define MAX_NUMA_BALANCING_FEATURES 5
static _HWFeature numa_balancing_features[] = {
    {"numa_balancing", "os", "Current state of NUMA balancing", numa_balancing_state_getter, NULL, DEVICE_TYPE_NODE},
    {"numa_balancing_scan_delay_ms", "os", "Time between page scans", numa_balancing_scan_delay_getter, NULL, DEVICE_TYPE_NODE},
    {"numa_balancing_scan_period_min_ms", "os", "Minimal time for scan period", numa_balancing_scan_period_min_getter, NULL, DEVICE_TYPE_NODE},
    {"numa_balancing_scan_period_max_ms", "os", "Maximal time for scan period", numa_balancing_scan_period_max_getter, NULL, DEVICE_TYPE_NODE},
    {"numa_balancing_scan_size_mb", "os", "Scan size for NUMA balancing", numa_balancing_scan_size_getter, NULL, DEVICE_TYPE_NODE},
};

static _HWFeatureList numa_balancing_feature_list = {
    .num_features = MAX_NUMA_BALANCING_FEATURES,
    .tester = numa_balancing_test,
    .features = numa_balancing_features,
};


int sysFeatures_init_linux_numa_balancing(_HWFeatureList* out);


#endif /* HWFEATURES_NUMABALANCING_H */

