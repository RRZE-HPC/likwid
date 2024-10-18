#include <sysFeatures_nvml.h>

#include <dlfcn.h>
#include <nvml.h>
#include <stdbool.h>
#include <stdlib.h>

#include <error.h>
#include <sysFeatures_common.h>
#include <types.h>

/* Perhaps we should deduplicate these dynamically loaded functions at some point,
 * but we try to be as compatible as best by only loading the functions we actually need. */

#define DLSYM_AND_CHECK( dllib, name ) name##_ptr = dlsym( dllib, #name ); if ( dlerror() != NULL ) { return -1; }
#define NVML_CALL(call, args, handleerror)                                            \
    do {                                                                           \
        nvmlReturn_t _status = (*call##_ptr)args;                                         \
        if (_status != NVML_SUCCESS) {                                            \
            fprintf(stderr, "Error: function %s failed with error: '%s' (nvmlReturn=%d).\n", #call, nvmlErrorString_ptr(_status), _status);                    \
            handleerror;                                                             \
        }                                                                          \
    } while (0)

#define DECLAREFUNC_NVML(funcname, funcsig) nvmlReturn_t __attribute__((weak)) funcname funcsig;  static nvmlReturn_t ( *funcname##_ptr ) funcsig;

DECLAREFUNC_NVML(nvmlDeviceGetClock, (nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int* clockMHz));
DECLAREFUNC_NVML(nvmlDeviceGetClockInfo, (nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock));
DECLAREFUNC_NVML(nvmlDeviceGetCount_v2, (unsigned int *deviceCount));
DECLAREFUNC_NVML(nvmlDeviceGetDetailedEccErrors, (nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlEccErrorCounts_t* eccCounts));
DECLAREFUNC_NVML(nvmlDeviceGetEccMode, (nvmlDevice_t device, nvmlEnableState_t* current, nvmlEnableState_t* pending));
DECLAREFUNC_NVML(nvmlDeviceGetFanSpeed_v2, (nvmlDevice_t device, unsigned int fan, unsigned int* speed));
DECLAREFUNC_NVML(nvmlDeviceGetHandleByIndex_v2, (unsigned int  index, nvmlDevice_t* device));
DECLAREFUNC_NVML(nvmlDeviceGetHandleByPciBusId_v2, (const char *pciBusId, nvmlDevice_t* device));
DECLAREFUNC_NVML(nvmlDeviceGetInforomVersion, (nvmlDevice_t device, nvmlInforomObject_t object, char* version, unsigned int  length));
DECLAREFUNC_NVML(nvmlDeviceGetMemoryInfo, (nvmlDevice_t device, nvmlMemory_t* memory));
DECLAREFUNC_NVML(nvmlDeviceGetPciInfo_v3, (nvmlDevice_t device, nvmlPciInfo_t *pci));
DECLAREFUNC_NVML(nvmlDeviceGetPerformanceState, (nvmlDevice_t device, nvmlPstates_t* pState));
DECLAREFUNC_NVML(nvmlDeviceGetPowerManagementLimit, (nvmlDevice_t device, unsigned int* limit));
DECLAREFUNC_NVML(nvmlDeviceGetPowerManagementLimitConstraints, (nvmlDevice_t device, unsigned int* minLimit, unsigned int* maxLimit));
DECLAREFUNC_NVML(nvmlDeviceGetPowerUsage, (nvmlDevice_t device, unsigned int* power));
DECLAREFUNC_NVML(nvmlDeviceGetTemperature, (nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int* temp));
DECLAREFUNC_NVML(nvmlDeviceGetTotalEccErrors, (nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long* eccCounts));
DECLAREFUNC_NVML(nvmlDeviceGetUtilizationRates, (nvmlDevice_t device, nvmlUtilization_t* utilization));
__attribute__((weak)) const char *nvmlErrorString(nvmlReturn_t result);
static const char *(*nvmlErrorString_ptr)(nvmlReturn_t result);
DECLAREFUNC_NVML(nvmlInit_v2, (void));
DECLAREFUNC_NVML(nvmlShutdown, (void));

static const _SysFeatureList nvidia_gpu_feature_list;
static bool nvml_initialized = false;
static void *dl_nvml = NULL;

static void cleanup_nvml(void)
{
    if (!nvml_initialized)
        return;

    nvmlShutdown_ptr();
    dlclose(dl_nvml);
}

int likwid_sysft_init_nvml(_SysFeatureList *list)
{
    if (nvml_initialized)
        return 0;

    dl_nvml = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_nvml)
    {
        ERROR_PRINT(dlopen(libnvidia-ml.so) failed: %s, dlerror());
        return -ELIBACC;
    }

    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetClock);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetClockInfo);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetCount_v2);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetDetailedEccErrors);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetEccMode);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetFanSpeed_v2);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetHandleByIndex_v2);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetHandleByPciBusId_v2);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetInforomVersion);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetMemoryInfo);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPciInfo_v3);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPerformanceState);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPowerManagementLimit);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPowerManagementLimitConstraints);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPowerUsage);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetTotalEccErrors);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetTemperature);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetUtilizationRates);
    DLSYM_AND_CHECK(dl_nvml, nvmlErrorString);
    DLSYM_AND_CHECK(dl_nvml, nvmlInit_v2);
    DLSYM_AND_CHECK(dl_nvml, nvmlShutdown);

    nvmlReturn_t nverr = nvmlInit_v2_ptr();
    if (nverr != NVML_SUCCESS)
    {
        dlclose(dl_nvml);
        ERROR_PRINT(nvmlInit_v2() failed: %s, nvmlErrorString_ptr(nverr));
        return -EPERM;
    }

    atexit(cleanup_nvml);

    nvml_initialized = true;

    return likwid_sysft_register_features(list, &nvidia_gpu_feature_list);
}

static int nvidia_gpu_device_count_getter(const LikwidDevice_t device, char **value)
{
    if (device->type != DEVICE_TYPE_NODE)
        return -EINVAL;
    unsigned int deviceCount;
    NVML_CALL(nvmlDeviceGetCount_v2, (&deviceCount), return -EPERM);
    return likwid_sysft_uint64_to_string(deviceCount, value);
}

static int lw_device_to_nvml_device(const LikwidDevice_t lwDevice, nvmlDevice_t *nvmlDevice)
{
    if (lwDevice->type != DEVICE_TYPE_NVIDIA_GPU)
        return -EINVAL;
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    const uint16_t domain = lwDevice->id.pci.pci_domain;
    const uint8_t bus = lwDevice->id.pci.pci_bus;
    const uint8_t dev = lwDevice->id.pci.pci_dev;
    const uint8_t func = lwDevice->id.pci.pci_func;
    snprintf(busId, sizeof(busId), "%04x:%02x:%02x.%01x", domain, bus, dev, func);
    NVML_CALL(nvmlDeviceGetHandleByPciBusId_v2, (busId, nvmlDevice), return -EPERM);
    return 0;
}

static int nvidia_gpu_devices_available_getter(const LikwidDevice_t device, char **value)
{
    if (device->type != DEVICE_TYPE_NODE)
        return -EINVAL;
    unsigned int deviceCount;
    NVML_CALL(nvmlDeviceGetCount_v2, (&deviceCount), return -EPERM);

    if (deviceCount == 0)
    {
        char *nonestr = strdup("");
        if (!nonestr)
            return -ENOMEM;
        free(*value);
        *value = nonestr;
        return 0;
    }

    const size_t requiredBytes = NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE * deviceCount;
    char *newval = realloc(*value, requiredBytes);
    if (!newval)
        return -ENOMEM;

    for (unsigned i = 0; i < deviceCount; i++)
    {
        nvmlReturn_t err;
        nvmlDevice_t nvmlDevice;
        nvmlPciInfo_t pciInfo;
        if (nvmlDeviceGetHandleByIndex_v2_ptr(i, &nvmlDevice) != NVML_SUCCESS ||
            nvmlDeviceGetPciInfo_v3_ptr(nvmlDevice, &pciInfo) != NVML_SUCCESS)
        {
            snprintf(pciInfo.busId, sizeof(pciInfo.busId), "ERR");
        }

        /* Since the strings to copy are limited by NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE,
         * the buffer size should always be sufficient. */
        if (i == 0)
        {
            strcpy(newval, pciInfo.busId);
        }
        else
        {
            strcat(newval, " ");
            strcat(newval, pciInfo.busId);
        }
    }

    *value = newval;
    return 0;
}

static int nvidia_gpu_clock_info_getter(const LikwidDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, char **value)
{
    nvmlDevice_t nvmlDevice;
    int err = lw_device_to_nvml_device(device, &nvmlDevice);
    if (err < 0)
        return err;
    unsigned clockMHz;
    NVML_CALL(nvmlDeviceGetClock, (nvmlDevice, clockType, clockId, &clockMHz), return -EPERM);
    return likwid_sysft_uint64_to_string(clockMHz, value);
}

static int nvidia_gpu_gfx_clock_cur_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT, value);
}

static _SysFeature nvidia_gpu_features[] = {
    {"device_count", "nvml", "Number of GPUs on node. Not all GPUs may be accessible.", nvidia_gpu_device_count_getter, NULL, DEVICE_TYPE_NODE},
    {"devices_available", "nvml", "Available GPUs (PCI addresses)", nvidia_gpu_devices_available_getter, NULL, DEVICE_TYPE_NODE},
    {"gfx_clock_cur", "nvml", "Current clock speed (graphics domain)", nvidia_gpu_gfx_clock_cur_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"gfx_clock_app_target", "nvml", "Application target clock speed (graphics domain)"},
    {"gfx_clock_app_default", "nvml", "Application default clock speed (graphics domain)"},
    {"gfx_clock_boost_max", "nvml", "Application default clock speed (graphics domain)"},
    {"sm_clock_cur", "nvml", "Current clock speed (SM domain)"},
    {"sm_clock_app_target", "nvml", "Application target clock speed (SM domain)"},
    {"sm_clock_app_default", "nvml", "Application default clock speed (SM domain)"},
    {"sm_clock_boost_max", "nvml", "Application default clock speed (SM domain)"},
    {"dram_clock_cur", "nvml", "Current clock speed (memory domain)"},
    {"dram_clock_app_target", "nvml", "Application target clock speed (memory domain)"},
    {"dram_clock_app_default", "nvml", "Application default clock speed (memory domain)"},
    {"dram_clock_boost_max", "nvml", "Application default clock speed (memory domain)"},
    {"video_clock_cur", "nvml", "Current clock speed (video encoder/decoder domain)"},
    {"video_clock_app_target", "nvml", "Application target clock speed (video encoder/decoder domain)"},
    {"video_clock_app_default", "nvml", "Application default clock speed (video encoder/decoder domain)"},
    {"video_clock_boost_max", "nvml", "Application default clock speed (video encoder/decoder domain)"},
};

static const _SysFeatureList nvidia_gpu_feature_list = {
    .num_features = ARRAY_COUNT(nvidia_gpu_features),
    .features = nvidia_gpu_features,
};
