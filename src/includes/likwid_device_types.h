#ifndef LIKWID_DEVICE_TYPES_H
#define LIKWID_DEVICE_TYPES_H

typedef enum {
    DEVICE_TYPE_INVALID,
    DEVICE_TYPE_HWTHREAD,
    DEVICE_TYPE_CORE,
    DEVICE_TYPE_LLC,
    DEVICE_TYPE_NUMA,
    DEVICE_TYPE_DIE,
    DEVICE_TYPE_SOCKET,
    DEVICE_TYPE_NODE,
#ifdef LIKWID_WITH_NVMON
    DEVICE_TYPE_NVIDIA_GPU,
#endif
#ifdef LIKWID_WITH_ROCMON
    DEVICE_TYPE_AMD_GPU,
#endif
#ifdef LIKWID_WITH_XEMON
    DEVICE_TYPE_INTEL_GPU,
#endif
    MAX_DEVICE_TYPE,
} LikwidDeviceType;
#define MIN_DEVICE_TYPE DEVICE_TYPE_HWTHREAD


typedef struct {
    LikwidDeviceType type;
    union {
        struct {
            int id;
        } simple;
        struct {
            int16_t pci_domain;
            int8_t pci_bus;
            int8_t pci_dev;
            int8_t pci_func;
        } pci;
    } id;
    int internal_id;
} _LikwidDevice;
typedef _LikwidDevice* LikwidDevice_t;

typedef struct {
    int num_devices;
    LikwidDevice_t devices;
} _LikwidDeviceList;
typedef _LikwidDeviceList* LikwidDeviceList_t;

static char* LikwidDeviceTypeNames[MAX_DEVICE_TYPE] = {
    [DEVICE_TYPE_INVALID] = "invalid",
    [DEVICE_TYPE_HWTHREAD] = "hwthread",
    [DEVICE_TYPE_CORE] = "core",
    [DEVICE_TYPE_LLC] = "LLC",
    [DEVICE_TYPE_NUMA] = "numa",
    [DEVICE_TYPE_DIE] = "die",
    [DEVICE_TYPE_SOCKET] = "socket",
    [DEVICE_TYPE_NODE] = "node",
#ifdef LIKWID_WITH_NVMON
    [DEVICE_TYPE_NVIDIA_GPU] = "nvidia_gpu",
#endif
#ifdef LIKWID_WITH_ROCMON
    [DEVICE_TYPE_AMD_GPU] = "amd_gpu",
#endif
#ifdef LIKWID_WITH_XEMON
    [DEVICE_TYPE_INTEL_GPU] = "intel_gpu",
#endif
};

#endif /* LIKWID_DEVICE_TYPES_H */
