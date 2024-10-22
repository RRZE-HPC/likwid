#include <sysFeatures_nvml.h>

#include <math.h>
#include <dlfcn.h>
#include <nvml.h>
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>

#include <error.h>
#include <sysFeatures_common.h>
#include <types.h>

/* Perhaps we should deduplicate these dynamically loaded functions at some point,
 * but we try to be as compatible as best by only loading the functions we actually need. */

#define DLSYM_AND_CHECK(dllib, name) name##_ptr = dlsym(dllib, #name);  \
    do {                                                                \
        const char *err = dlerror();                                    \
        if (err) {                                                      \
            ERROR_PRINT(Error: dlsym on symbol '%s' failed with error: %s, #name, err); \
            return -EINVAL;                                             \
        }                                                               \
    } while (0)

#define NVML_CALL(func, ...)                            \
    do {                                                \
        nvmlReturn_t s = (*func##_ptr)(__VA_ARGS__);    \
        if (s != NVML_SUCCESS) {                        \
            ERROR_PRINT(Error: function %s failed with error: '%s' (nvmlReturn=%d).\n, #func, nvmlErrorString_ptr(s), s);   \
            return -EPERM;                              \
        }                                               \
    } while (0)

#define DECLAREFUNC_NVML(funcname, ...) nvmlReturn_t __attribute__((weak)) funcname(__VA_ARGS__);  static nvmlReturn_t (*funcname##_ptr)(__VA_ARGS__);

#define LWD_TO_NVMLD(lwd, nvmlDevice) nvmlDevice_t nvmlDevice;  \
    do {                                                        \
        int err = lw_device_to_nvml_device(lwd, &nvmlDevice);   \
        if (err < 0)                                            \
            return err;                                         \
    } while (0)

DECLAREFUNC_NVML(nvmlDeviceGetBAR1MemoryInfo, nvmlDevice_t device, nvmlBAR1Memory_t* bar1Memory);
DECLAREFUNC_NVML(nvmlDeviceGetBusType, nvmlDevice_t device, nvmlBusType_t *type);
DECLAREFUNC_NVML(nvmlDeviceGetClock, nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int* clockMHz);
DECLAREFUNC_NVML(nvmlDeviceGetClockInfo, nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock);
DECLAREFUNC_NVML(nvmlDeviceGetCurrPcieLinkGeneration, nvmlDevice_t device, unsigned int *currLinkGen);
DECLAREFUNC_NVML(nvmlDeviceGetCurrPcieLinkWidth, nvmlDevice_t device, unsigned int *currLinkGen);
DECLAREFUNC_NVML(nvmlDeviceGetCount_v2, unsigned int *deviceCount);
DECLAREFUNC_NVML(nvmlDeviceGetDecoderUtilization, nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs);
DECLAREFUNC_NVML(nvmlDeviceGetDefaultEccMode, nvmlDevice_t device, nvmlEnableState_t *defaultMode);
DECLAREFUNC_NVML(nvmlDeviceGetDetailedEccErrors, nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlEccErrorCounts_t* eccCounts);
DECLAREFUNC_NVML(nvmlDeviceGetEccMode, nvmlDevice_t device, nvmlEnableState_t* current, nvmlEnableState_t* pending);
DECLAREFUNC_NVML(nvmlDeviceGetEncoderCapacity, nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int *capacity);
DECLAREFUNC_NVML(nvmlDeviceGetEncoderUtilization, nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs);
DECLAREFUNC_NVML(nvmlDeviceGetEnforcedPowerLimit, nvmlDevice_t device, unsigned int *limit);
DECLAREFUNC_NVML(nvmlDeviceGetFanSpeed, nvmlDevice_t device, unsigned int *speed);
DECLAREFUNC_NVML(nvmlDeviceGetGpuMaxPcieLinkGeneration, nvmlDevice_t device, unsigned int *maxLinkGenDevice);
DECLAREFUNC_NVML(nvmlDeviceGetGpuOperationMode, nvmlDevice_t device, nvmlGpuOperationMode_t *current, nvmlGpuOperationMode_t *pending);
DECLAREFUNC_NVML(nvmlDeviceGetHandleByIndex_v2, unsigned int  index, nvmlDevice_t* device);
DECLAREFUNC_NVML(nvmlDeviceGetHandleByPciBusId_v2, const char *pciBusId, nvmlDevice_t* device);
DECLAREFUNC_NVML(nvmlDeviceGetInforomVersion, nvmlDevice_t device, nvmlInforomObject_t object, char* version, unsigned int  length);
DECLAREFUNC_NVML(nvmlDeviceGetJpgUtilization, nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs);
DECLAREFUNC_NVML(nvmlDeviceGetMaxPcieLinkGeneration, nvmlDevice_t device, unsigned int *maxSpeed);
DECLAREFUNC_NVML(nvmlDeviceGetMaxPcieLinkWidth, nvmlDevice_t device, unsigned int *width);
DECLAREFUNC_NVML(nvmlDeviceGetMemoryBusWidth, nvmlDevice_t device, unsigned int *busWidth);
DECLAREFUNC_NVML(nvmlDeviceGetMemoryInfo, nvmlDevice_t device, nvmlMemory_t* memory);
DECLAREFUNC_NVML(nvmlDeviceGetMultiGpuBoard, nvmlDevice_t device, unsigned int *multiGpuBool);
DECLAREFUNC_NVML(nvmlDeviceGetOfaUtilization, nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs);
DECLAREFUNC_NVML(nvmlDeviceGetPcieLinkMaxSpeed, nvmlDevice_t device, unsigned int *maxSpeed);
DECLAREFUNC_NVML(nvmlDeviceGetPciInfo_v3, nvmlDevice_t device, nvmlPciInfo_t *pci);
DECLAREFUNC_NVML(nvmlDeviceGetPcieReplayCounter, nvmlDevice_t device, unsigned int *value);
DECLAREFUNC_NVML(nvmlDeviceGetPcieSpeed, nvmlDevice_t device, unsigned int *pcieSpeed);
DECLAREFUNC_NVML(nvmlDeviceGetPcieThroughput, nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int *value);
DECLAREFUNC_NVML(nvmlDeviceGetPerformanceState, nvmlDevice_t device, nvmlPstates_t* pState);
DECLAREFUNC_NVML(nvmlDeviceGetPowerManagementDefaultLimit, nvmlDevice_t device, unsigned int* limit);
DECLAREFUNC_NVML(nvmlDeviceGetPowerManagementLimit, nvmlDevice_t device, unsigned int* limit);
DECLAREFUNC_NVML(nvmlDeviceGetPowerManagementLimitConstraints, nvmlDevice_t device, unsigned int* minLimit, unsigned int* maxLimit);
DECLAREFUNC_NVML(nvmlDeviceGetPowerUsage, nvmlDevice_t device, unsigned int* power);
DECLAREFUNC_NVML(nvmlDeviceGetTemperature, nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int* temp);
DECLAREFUNC_NVML(nvmlDeviceGetTemperatureThreshold, nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int* temp);
DECLAREFUNC_NVML(nvmlDeviceGetTotalEccErrors, nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long* eccCounts);
DECLAREFUNC_NVML(nvmlDeviceGetTotalEnergyConsumption, nvmlDevice_t device, unsigned long long *energy);
DECLAREFUNC_NVML(nvmlDeviceGetUtilizationRates, nvmlDevice_t device, nvmlUtilization_t* utilization);
DECLAREFUNC_NVML(nvmlDeviceGetViolationStatus, nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType, nvmlViolationTime_t* violTime);
DECLAREFUNC_NVML(nvmlDeviceSetEccMode, nvmlDevice_t device, nvmlEnableState_t ecc);
DECLAREFUNC_NVML(nvmlDeviceSetGpuOperationMode, nvmlDevice_t device, nvmlGpuOperationMode_t mode);
DECLAREFUNC_NVML(nvmlDeviceSetPowerManagementLimit, nvmlDevice_t device, unsigned int limit);
DECLAREFUNC_NVML(nvmlDeviceSetTemperatureThreshold, nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int *temp);
__attribute__((weak)) const char *nvmlErrorString(nvmlReturn_t result);
static const char *(*nvmlErrorString_ptr)(nvmlReturn_t result);
DECLAREFUNC_NVML(nvmlInit_v2, void);
DECLAREFUNC_NVML(nvmlShutdown, void);

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

    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetBAR1MemoryInfo);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetBusType);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetClock);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetClockInfo);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetCurrPcieLinkGeneration);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetCurrPcieLinkWidth);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetCount_v2);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetDecoderUtilization);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetDefaultEccMode);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetDetailedEccErrors);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetEccMode);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetEncoderCapacity);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetEncoderUtilization);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetEnforcedPowerLimit);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetFanSpeed);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetGpuMaxPcieLinkGeneration);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetGpuOperationMode);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetHandleByIndex_v2);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetHandleByPciBusId_v2);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetInforomVersion);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetJpgUtilization);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetMaxPcieLinkGeneration);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetMaxPcieLinkWidth);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetMemoryBusWidth);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetMemoryInfo);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetMultiGpuBoard);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetOfaUtilization);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPcieLinkMaxSpeed);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPciInfo_v3);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPcieReplayCounter);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPcieSpeed);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPcieThroughput);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPerformanceState);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPowerManagementDefaultLimit);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPowerManagementLimit);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPowerManagementLimitConstraints);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPowerUsage);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetTotalEccErrors);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetTotalEnergyConsumption);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetTemperature);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetTemperatureThreshold);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetUtilizationRates);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetViolationStatus);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceSetEccMode);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceSetGpuOperationMode);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceSetPowerManagementLimit);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceSetTemperatureThreshold);
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
    NVML_CALL(nvmlDeviceGetCount_v2, &deviceCount);
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
    NVML_CALL(nvmlDeviceGetHandleByPciBusId_v2, busId, nvmlDevice);
    return 0;
}

static int nvidia_gpu_devices_available_getter(const LikwidDevice_t device, char **value)
{
    if (device->type != DEVICE_TYPE_NODE)
        return -EINVAL;
    unsigned int deviceCount;
    NVML_CALL(nvmlDeviceGetCount_v2, &deviceCount);

    if (deviceCount == 0)
        return likwid_sysft_copystr("", value);

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
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned clockMHz;
    NVML_CALL(nvmlDeviceGetClock, nvmlDevice, clockType, clockId, &clockMHz);
    return likwid_sysft_uint64_to_string(clockMHz, value);
}

static int nvidia_gpu_gfx_clock_cur_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT, value);
}

static int nvidia_gpu_gfx_clock_app_target_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_APP_CLOCK_TARGET, value);
}

static int nvidia_gpu_gfx_clock_app_default_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_APP_CLOCK_DEFAULT, value);
}

static int nvidia_gpu_gfx_clock_boost_max_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CUSTOMER_BOOST_MAX, value);
}

static int nvidia_gpu_sm_clock_cur_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, value);
}

static int nvidia_gpu_sm_clock_app_target_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_SM, NVML_CLOCK_ID_APP_CLOCK_TARGET, value);
}

static int nvidia_gpu_sm_clock_app_default_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_SM, NVML_CLOCK_ID_APP_CLOCK_DEFAULT, value);
}

static int nvidia_gpu_sm_clock_boost_max_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_SM, NVML_CLOCK_ID_CUSTOMER_BOOST_MAX, value);
}

static int nvidia_gpu_dram_clock_cur_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, value);
}

static int nvidia_gpu_dram_clock_app_target_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_MEM, NVML_CLOCK_ID_APP_CLOCK_TARGET, value);
}

static int nvidia_gpu_dram_clock_app_default_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_MEM, NVML_CLOCK_ID_APP_CLOCK_DEFAULT, value);
}

static int nvidia_gpu_dram_clock_boost_max_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_MEM, NVML_CLOCK_ID_CUSTOMER_BOOST_MAX, value);
}

static int nvidia_gpu_video_clock_cur_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_VIDEO, NVML_CLOCK_ID_CURRENT, value);
}

static int nvidia_gpu_video_clock_app_target_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_VIDEO, NVML_CLOCK_ID_APP_CLOCK_TARGET, value);
}

static int nvidia_gpu_video_clock_app_default_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_VIDEO, NVML_CLOCK_ID_APP_CLOCK_DEFAULT, value);
}

static int nvidia_gpu_video_clock_boost_max_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_clock_info_getter(device, NVML_CLOCK_VIDEO, NVML_CLOCK_ID_CUSTOMER_BOOST_MAX, value);
}

static int nvidia_gpu_bar1_getter(const LikwidDevice_t device, int entry, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    nvmlBAR1Memory_t bar1Memory;
    NVML_CALL(nvmlDeviceGetBAR1MemoryInfo, nvmlDevice, &bar1Memory);
    if (entry == 0)
        return likwid_sysft_uint64_to_string(bar1Memory.bar1Free, value);
    if (entry == 1)
        return likwid_sysft_uint64_to_string(bar1Memory.bar1Used, value);
    return likwid_sysft_uint64_to_string(bar1Memory.bar1Total, value);
}

static int nvidia_gpu_bar1_free_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_bar1_getter(device, 0, value);
}

static int nvidia_gpu_bar1_used_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_bar1_getter(device, 1, value);
}

static int nvidia_gpu_bar1_total_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_bar1_getter(device, 2, value);
}

static int nvidia_gpu_bus_type_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    nvmlBusType_t type;
    NVML_CALL(nvmlDeviceGetBusType, nvmlDevice, &type);
    static const char *types[] = {
        "Unknown", "PCI", "PCIe", "FPCI", "AGP",
    };
    if (type >= ARRAY_COUNT(types))
        type = 0;
    return likwid_sysft_copystr(types[type], value);
}

static int nvidia_gpu_pcie_gen_cur_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int gen;
    NVML_CALL(nvmlDeviceGetCurrPcieLinkGeneration, nvmlDevice, &gen);
    return likwid_sysft_uint64_to_string(gen, value);
}

static int nvidia_gpu_pcie_gen_gpu_max_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int gen;
    NVML_CALL(nvmlDeviceGetGpuMaxPcieLinkGeneration, nvmlDevice, &gen);
    return likwid_sysft_uint64_to_string(gen, value);
}

static int nvidia_gpu_pcie_gen_sys_max_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int gen;
    NVML_CALL(nvmlDeviceGetMaxPcieLinkGeneration, nvmlDevice, &gen);
    return likwid_sysft_uint64_to_string(gen, value);
}

static int nvidia_gpu_pcie_width_cur_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int width;
    NVML_CALL(nvmlDeviceGetCurrPcieLinkWidth, nvmlDevice, &width);
    return likwid_sysft_uint64_to_string(width, value);
}

static int nvidia_gpu_pcie_width_sys_max_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int width;
    NVML_CALL(nvmlDeviceGetMaxPcieLinkWidth, nvmlDevice, &width);
    return likwid_sysft_uint64_to_string(width, value);
}

static int nvidia_gpu_pcie_speed_value(unsigned int pcieSpeed, char **value)
{
    const char *speeds[] = {
        "INVALID", "2500", "5000", "8000", "16000", "32000", "64000",
    };
    if (pcieSpeed >= ARRAY_COUNT(speeds))
        pcieSpeed = 0;
    return likwid_sysft_copystr(speeds[pcieSpeed], value);
}

static int nvidia_gpu_pcie_speed_max_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int maxSpeed;
    NVML_CALL(nvmlDeviceGetPcieLinkMaxSpeed, nvmlDevice, &maxSpeed);
    return nvidia_gpu_pcie_speed_value(maxSpeed, value);
}

static int nvidia_gpu_pcie_speed_cur_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int speed;
    NVML_CALL(nvmlDeviceGetPcieSpeed, nvmlDevice, &speed);
    return nvidia_gpu_pcie_speed_value(speed, value);
}

static int nvidia_gpu_pcie_replay_counter_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int replayCounter;
    NVML_CALL(nvmlDeviceGetPcieReplayCounter, nvmlDevice, &replayCounter);
    return likwid_sysft_uint64_to_string(replayCounter, value);
}

static int nvidia_gpu_pcie_throughput_getter(const LikwidDevice_t device, nvmlPcieUtilCounter_t counter, char **value)
{
    // TODO does this work right? appears to always return 0?
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int throughput;
    NVML_CALL(nvmlDeviceGetPcieThroughput, nvmlDevice, counter, &throughput);
    return likwid_sysft_uint64_to_string(throughput, value);
}

static int nvidia_gpu_pcie_throughput_tx_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_pcie_throughput_getter(device, NVML_PCIE_UTIL_TX_BYTES, value);
}

static int nvidia_gpu_pcie_throughput_rx_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_pcie_throughput_getter(device, NVML_PCIE_UTIL_RX_BYTES, value);
}

static int nvidia_gpu_decoder_util_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int utilization, dummy;
    NVML_CALL(nvmlDeviceGetDecoderUtilization, nvmlDevice, &utilization, &dummy);
    return likwid_sysft_uint64_to_string(utilization, value);
}

static int nvidia_gpu_ecc_default_enabled_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    nvmlEnableState_t enabled;
    NVML_CALL(nvmlDeviceGetDefaultEccMode, nvmlDevice, &enabled);
    return likwid_sysft_uint64_to_string(enabled == NVML_FEATURE_ENABLED, value);
}

static int nvidia_gpu_ecc_enabled_getter(const LikwidDevice_t device, bool getPending, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    nvmlEnableState_t current, pending;
    NVML_CALL(nvmlDeviceGetEccMode, nvmlDevice, &current, &pending);
    if (getPending)
        return likwid_sysft_uint64_to_string(pending == NVML_FEATURE_ENABLED, value);
    return likwid_sysft_uint64_to_string(current == NVML_FEATURE_ENABLED, value);
}

static int nvidia_gpu_ecc_enabled_setter(const LikwidDevice_t device, const char *value)
{
    uint64_t enabled;
    int err = likwid_sysft_string_to_uint64(value, &enabled);
    if (err < 0)
        return err;
    LWD_TO_NVMLD(device, nvmlDevice);
    NVML_CALL(nvmlDeviceSetEccMode, nvmlDevice, (enabled == 0) ? NVML_FEATURE_DISABLED : NVML_FEATURE_ENABLED);
    return 0;
}

static int nvidia_gpu_ecc_enabled_cur_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_enabled_getter(device, false, value);
}

static int nvidia_gpu_ecc_enabled_pend_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_enabled_getter(device, true, value);
}

typedef enum
{
    ECC_CORR_VOL,
    ECC_UNCORR_VOL,
    ECC_CORR_AGG,
    ECC_UNCORR_AGG,
} EccErrType;

typedef enum
{
    ECC_MEM_DRAM,
    ECC_MEM_L1,
    ECC_MEM_L2,
    ECC_MEM_REG,
} EccErrMem;

static int nvidia_gpu_ecc_error_getter(const LikwidDevice_t device, EccErrType type, EccErrMem mem, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    nvmlEccErrorCounts_t ecc;
    if (type == ECC_CORR_VOL)
        NVML_CALL(nvmlDeviceGetDetailedEccErrors, nvmlDevice, NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_VOLATILE_ECC, &ecc);
    else if (type == ECC_UNCORR_VOL)
        NVML_CALL(nvmlDeviceGetDetailedEccErrors, nvmlDevice, NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_VOLATILE_ECC, &ecc);
    else if (type == ECC_CORR_AGG)
        NVML_CALL(nvmlDeviceGetDetailedEccErrors, nvmlDevice, NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_AGGREGATE_ECC, &ecc);
    else
        NVML_CALL(nvmlDeviceGetDetailedEccErrors, nvmlDevice, NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_AGGREGATE_ECC, &ecc);
    if (mem == ECC_MEM_DRAM)
        return likwid_sysft_uint64_to_string(ecc.deviceMemory, value);
    if (mem == ECC_MEM_L1)
        return likwid_sysft_uint64_to_string(ecc.l1Cache, value);
    if (mem == ECC_MEM_L2)
        return likwid_sysft_uint64_to_string(ecc.l2Cache, value);
    return likwid_sysft_uint64_to_string(ecc.registerFile, value);
}

static int nvidia_gpu_ecc_error_dram_vol_corr_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_error_getter(device, ECC_CORR_VOL, ECC_MEM_DRAM, value);
}

static int nvidia_gpu_ecc_error_dram_vol_uncorr_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_error_getter(device, ECC_UNCORR_VOL, ECC_MEM_DRAM, value);
}

static int nvidia_gpu_ecc_error_dram_agg_corr_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_error_getter(device, ECC_CORR_AGG, ECC_MEM_DRAM, value);
}

static int nvidia_gpu_ecc_error_dram_agg_uncorr_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_error_getter(device, ECC_UNCORR_AGG, ECC_MEM_DRAM, value);
}

static int nvidia_gpu_ecc_error_l1_vol_corr_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_error_getter(device, ECC_CORR_VOL, ECC_MEM_L1, value);
}

static int nvidia_gpu_ecc_error_l1_vol_uncorr_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_error_getter(device, ECC_UNCORR_VOL, ECC_MEM_L1, value);
}

static int nvidia_gpu_ecc_error_l1_agg_corr_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_error_getter(device, ECC_CORR_AGG, ECC_MEM_L1, value);
}

static int nvidia_gpu_ecc_error_l1_agg_uncorr_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_error_getter(device, ECC_UNCORR_AGG, ECC_MEM_L1, value);
}

static int nvidia_gpu_ecc_error_l2_vol_corr_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_error_getter(device, ECC_CORR_VOL, ECC_MEM_L2, value);
}

static int nvidia_gpu_ecc_error_l2_vol_uncorr_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_error_getter(device, ECC_UNCORR_VOL, ECC_MEM_L2, value);
}

static int nvidia_gpu_ecc_error_l2_agg_corr_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_error_getter(device, ECC_CORR_AGG, ECC_MEM_L2, value);
}

static int nvidia_gpu_ecc_error_l2_agg_uncorr_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_error_getter(device, ECC_UNCORR_AGG, ECC_MEM_L2, value);
}

static int nvidia_gpu_ecc_error_reg_vol_corr_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_error_getter(device, ECC_CORR_VOL, ECC_MEM_REG, value);
}

static int nvidia_gpu_ecc_error_reg_vol_uncorr_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_error_getter(device, ECC_UNCORR_VOL, ECC_MEM_REG, value);
}

static int nvidia_gpu_ecc_error_reg_agg_corr_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_error_getter(device, ECC_CORR_AGG, ECC_MEM_REG, value);
}

static int nvidia_gpu_ecc_error_reg_agg_uncorr_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_ecc_error_getter(device, ECC_UNCORR_AGG, ECC_MEM_REG, value);
}

static int nvidia_gpu_fan_speed_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int fanSpeed;
    NVML_CALL(nvmlDeviceGetFanSpeed, nvmlDevice, &fanSpeed);
    return likwid_sysft_uint64_to_string(fanSpeed, value);
}

static int nvidia_gpu_pstate_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    nvmlPstates_t pstate;
    NVML_CALL(nvmlDeviceGetPerformanceState, nvmlDevice, &pstate);
    return likwid_sysft_uint64_to_string(pstate, value);
}

static int nvidia_gpu_power_limit_default_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int milliwatts;
    NVML_CALL(nvmlDeviceGetPowerManagementDefaultLimit, nvmlDevice, &milliwatts);
    return likwid_sysft_double_to_string(milliwatts / 1000.0, value);
}

static int nvidia_gpu_power_limit_cur_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int milliwatts;
    NVML_CALL(nvmlDeviceGetPowerManagementLimit, nvmlDevice, &milliwatts);
    return likwid_sysft_double_to_string(milliwatts / 1000.0, value);
}

static int nvidia_gpu_power_limit_cur_setter(const LikwidDevice_t device, const char *value)
{
    uint64_t watts;
    int err = likwid_sysft_string_to_uint64(value, &watts);
    if (err < 0)
        return err;
    const unsigned int pw = (unsigned int)round(watts * 1000.0);
    LWD_TO_NVMLD(device, nvmlDevice);
    NVML_CALL(nvmlDeviceSetPowerManagementLimit, nvmlDevice, pw);
    return 0;
}

static int nvidia_gpu_power_limit_minmax_getter(const LikwidDevice_t device, bool max, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int milliwattsMin, milliwattsMax;
    NVML_CALL(nvmlDeviceGetPowerManagementLimitConstraints, nvmlDevice, &milliwattsMin, &milliwattsMax);
    return likwid_sysft_double_to_string((max ? milliwattsMax : milliwattsMin) / 1000.0, value);
}

static int nvidia_gpu_power_limit_min_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_power_limit_minmax_getter(device, false, value);
}

static int nvidia_gpu_power_limit_max_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_power_limit_minmax_getter(device, true, value);
}

static int nvidia_gpu_power_limit_enf_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int milliwatts;
    NVML_CALL(nvmlDeviceGetEnforcedPowerLimit, nvmlDevice, &milliwatts);
    return likwid_sysft_double_to_string(milliwatts / 1000.0, value);
}

static int nvidia_gpu_power_cur_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int milliwatts;
    NVML_CALL(nvmlDeviceGetPowerUsage, nvmlDevice, &milliwatts);
    return likwid_sysft_double_to_string(milliwatts / 1000.0, value);
}

static int nvidia_gpu_encoder_capacity_getter(const LikwidDevice_t device, nvmlEncoderType_t type, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int capacity;
    NVML_CALL(nvmlDeviceGetEncoderCapacity, nvmlDevice, type, &capacity);
    return likwid_sysft_uint64_to_string(capacity, value);
}

static int nvidia_gpu_encoder_capacity_h264_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_encoder_capacity_getter(device, NVML_ENCODER_QUERY_H264, value);
}

static int nvidia_gpu_encoder_capacity_hevc_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_encoder_capacity_getter(device, NVML_ENCODER_QUERY_HEVC, value);
}

static int nvidia_gpu_encoder_capacity_av1_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_encoder_capacity_getter(device, NVML_ENCODER_QUERY_AV1, value);
}

static int nvidia_gpu_encoder_usage_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int utilization, periodUs;
    NVML_CALL(nvmlDeviceGetEncoderUtilization, nvmlDevice, &utilization, &periodUs);
    return likwid_sysft_uint64_to_string(utilization, value);
}

static const char *goms[] = {
    "all_on", "compute_only", "graphics_only",
};

static int nvidia_gpu_gom_getter(const LikwidDevice_t device, bool usePending, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    nvmlGpuOperationMode_t current, pending;
    NVML_CALL(nvmlDeviceGetGpuOperationMode, nvmlDevice, &current, &pending);
    nvmlGpuOperationMode_t gom = usePending ? pending : current;
    if (gom >= ARRAY_COUNT(goms))
        gom = 0;
    return likwid_sysft_copystr(goms[gom], value);
}

static int nvidia_gpu_gom_setter(const LikwidDevice_t device, const char *value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    for (size_t i = 0; i < ARRAY_COUNT(goms); i++)
    {
        if (strcmp(value, goms[i]) == 0)
        {
            NVML_CALL(nvmlDeviceSetGpuOperationMode, nvmlDevice, (int)i);
            return 0;
        }
    }
    return -EINVAL;
}

static int nvidia_gpu_gom_cur_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_gom_getter(device, false, value);
}

static int nvidia_gpu_gom_pend_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_gom_getter(device, true, value);
}

static int nvidia_gpu_jpg_usage_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int utilization, periodUs;
    NVML_CALL(nvmlDeviceGetJpgUtilization, nvmlDevice, &utilization, &periodUs);
    return likwid_sysft_uint64_to_string(utilization, value);
}

static int nvidia_gpu_dram_bus_width_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int busWidth;
    NVML_CALL(nvmlDeviceGetMemoryBusWidth, nvmlDevice, &busWidth);
    return likwid_sysft_uint64_to_string(busWidth, value);
}

static int nvidia_gpu_multi_gpu_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int multiGpuBool;
    NVML_CALL(nvmlDeviceGetMultiGpuBoard, nvmlDevice, &multiGpuBool);
    return likwid_sysft_uint64_to_string(multiGpuBool, value);
}

static int nvidia_gpu_temp_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int temp;
    NVML_CALL(nvmlDeviceGetTemperature, nvmlDevice, NVML_TEMPERATURE_GPU, &temp);
    return likwid_sysft_uint64_to_string(temp, value);
}

static int nvidia_gpu_temp_thresh_getter(const LikwidDevice_t device, nvmlTemperatureThresholds_t t, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int temp;
    NVML_CALL(nvmlDeviceGetTemperatureThreshold, nvmlDevice, t, &temp);
    return likwid_sysft_uint64_to_string(temp, value);
}

static int nvidia_gpu_temp_thresh_shut_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_temp_thresh_getter(device, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN, value);
}

static int nvidia_gpu_temp_thresh_slow_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_temp_thresh_getter(device, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, value);
}

static int nvidia_gpu_temp_thresh_mem_max_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_temp_thresh_getter(device, NVML_TEMPERATURE_THRESHOLD_MEM_MAX, value);
}

static int nvidia_gpu_temp_thresh_gpu_max_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_temp_thresh_getter(device, NVML_TEMPERATURE_THRESHOLD_GPU_MAX, value);
}

static int nvidia_gpu_temp_thresh_acou_min_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_temp_thresh_getter(device, NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MIN, value);
}

static int nvidia_gpu_temp_thresh_acou_cur_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_temp_thresh_getter(device, NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_CURR, value);
}

static int nvidia_gpu_temp_thresh_acou_max_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_temp_thresh_getter(device, NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MAX, value);
}

static int nvidia_gpu_temp_thresh_gps_cur_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_temp_thresh_getter(device, NVML_TEMPERATURE_THRESHOLD_GPS_CURR, value);
}

static int nvidia_gpu_temp_thresh_setter(const LikwidDevice_t device, nvmlTemperatureThresholds_t t, const char *value)
{
    uint64_t temp;
    int err = likwid_sysft_string_to_uint64(value, &temp);
    if (err < 0)
        return err;
    if (temp > INT_MAX)
        temp = INT_MAX;
    int arg = (int)temp;
    LWD_TO_NVMLD(device, nvmlDevice);
    NVML_CALL(nvmlDeviceSetTemperatureThreshold, nvmlDevice, t, &arg);
    return 0;
}

static int nvidia_gpu_temp_thresh_shut_setter(const LikwidDevice_t device, const char *value)
{
    return nvidia_gpu_temp_thresh_setter(device, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN, value);
}

static int nvidia_gpu_temp_thresh_slow_setter(const LikwidDevice_t device, const char *value)
{
    return nvidia_gpu_temp_thresh_setter(device, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, value);
}

static int nvidia_gpu_temp_thresh_mem_max_setter(const LikwidDevice_t device, const char *value)
{
    return nvidia_gpu_temp_thresh_setter(device, NVML_TEMPERATURE_THRESHOLD_MEM_MAX, value);
}

static int nvidia_gpu_temp_thresh_gpu_max_setter(const LikwidDevice_t device, const char *value)
{
    return nvidia_gpu_temp_thresh_setter(device, NVML_TEMPERATURE_THRESHOLD_GPU_MAX, value);
}

static int nvidia_gpu_temp_thresh_acou_min_setter(const LikwidDevice_t device, const char *value)
{
    return nvidia_gpu_temp_thresh_setter(device, NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MIN, value);
}

static int nvidia_gpu_temp_thresh_acou_cur_setter(const LikwidDevice_t device, const char *value)
{
    return nvidia_gpu_temp_thresh_setter(device, NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_CURR, value);
}

static int nvidia_gpu_temp_thresh_acou_max_setter(const LikwidDevice_t device, const char *value)
{
    return nvidia_gpu_temp_thresh_setter(device, NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MAX, value);
}

static int nvidia_gpu_temp_thresh_gps_cur_setter(const LikwidDevice_t device, const char *value)
{
    return nvidia_gpu_temp_thresh_setter(device, NVML_TEMPERATURE_THRESHOLD_GPS_CURR, value);
}

static int nvidia_gpu_ofa_usage_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned int usage, samplingPeriodUs;
    NVML_CALL(nvmlDeviceGetOfaUtilization, nvmlDevice, &usage, &samplingPeriodUs);
    return likwid_sysft_uint64_to_string(usage, value);
}

static int nvidia_gpu_energy_getter(const LikwidDevice_t device, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    unsigned long long energy;
    NVML_CALL(nvmlDeviceGetTotalEnergyConsumption, nvmlDevice, &energy);
    return likwid_sysft_double_to_string((double)energy / 1000.0, value);
}

static int nvidia_gpu_gpumem_usage_getter(const LikwidDevice_t device, bool gpu, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    nvmlUtilization_t util;
    NVML_CALL(nvmlDeviceGetUtilizationRates, nvmlDevice, &util);
    return likwid_sysft_uint64_to_string(gpu ? util.gpu : util.memory, value);
}

static int nvidia_gpu_gpu_usage_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_gpumem_usage_getter(device, true, value);
}

static int nvidia_gpu_dram_usage_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_gpumem_usage_getter(device, false, value);
}

static int nvidia_gpu_policy_time_getter(const LikwidDevice_t device, nvmlPerfPolicyType_t type, char **value)
{
    LWD_TO_NVMLD(device, nvmlDevice);
    nvmlViolationTime_t vt;
    NVML_CALL(nvmlDeviceGetViolationStatus, nvmlDevice, type, &vt);
    char timeBuf[64];
    snprintf(timeBuf, sizeof(timeBuf), "%llu:%llu", vt.referenceTime, vt.violationTime);
    return likwid_sysft_copystr(timeBuf, value);
}

static int nvidia_gpu_policy_time_power_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_policy_time_getter(device, NVML_PERF_POLICY_POWER, value);
}

static int nvidia_gpu_policy_time_therm_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_policy_time_getter(device, NVML_PERF_POLICY_THERMAL, value);
}

static int nvidia_gpu_policy_time_sync_boost_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_policy_time_getter(device, NVML_PERF_POLICY_SYNC_BOOST, value);
}

static int nvidia_gpu_policy_time_board_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_policy_time_getter(device, NVML_PERF_POLICY_BOARD_LIMIT, value);
}

static int nvidia_gpu_policy_time_lowutil_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_policy_time_getter(device, NVML_PERF_POLICY_LOW_UTILIZATION, value);
}

static int nvidia_gpu_policy_time_rel_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_policy_time_getter(device, NVML_PERF_POLICY_RELIABILITY, value);
}

static int nvidia_gpu_policy_time_total_app_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_policy_time_getter(device, NVML_PERF_POLICY_TOTAL_APP_CLOCKS, value);
}

static int nvidia_gpu_policy_time_total_base_getter(const LikwidDevice_t device, char **value)
{
    return nvidia_gpu_policy_time_getter(device, NVML_PERF_POLICY_TOTAL_BASE_CLOCKS, value);
}

static _SysFeature nvidia_gpu_features[] = {
    {"device_count", "nvml", "Number of GPUs on node. Not all GPUs may be accessible.", nvidia_gpu_device_count_getter, NULL, DEVICE_TYPE_NODE},
    {"devices_available", "nvml", "Available GPUs (PCI addresses)", nvidia_gpu_devices_available_getter, NULL, DEVICE_TYPE_NODE},
    {"gfx_clock_cur", "nvml", "Current clock speed (graphics domain)", nvidia_gpu_gfx_clock_cur_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"gfx_clock_app_target", "nvml", "Application target clock speed (graphics domain)", nvidia_gpu_gfx_clock_app_target_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"gfx_clock_app_default", "nvml", "Application default clock speed (graphics domain)", nvidia_gpu_gfx_clock_app_default_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"gfx_clock_boost_max", "nvml", "Application default clock speed (graphics domain)", nvidia_gpu_gfx_clock_boost_max_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"sm_clock_cur", "nvml", "Current clock speed (SM domain)", nvidia_gpu_sm_clock_cur_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"sm_clock_app_target", "nvml", "Application target clock speed (SM domain)", nvidia_gpu_sm_clock_app_target_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"sm_clock_app_default", "nvml", "Application default clock speed (SM domain)", nvidia_gpu_sm_clock_app_default_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"sm_clock_boost_max", "nvml", "Application default clock speed (SM domain)", nvidia_gpu_sm_clock_boost_max_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"dram_clock_cur", "nvml", "Current clock speed (memory domain)", nvidia_gpu_dram_clock_cur_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"dram_clock_app_target", "nvml", "Application target clock speed (memory domain)", nvidia_gpu_dram_clock_app_target_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"dram_clock_app_default", "nvml", "Application default clock speed (memory domain)", nvidia_gpu_dram_clock_app_default_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"dram_clock_boost_max", "nvml", "Application default clock speed (memory domain)", nvidia_gpu_dram_clock_boost_max_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"video_clock_cur", "nvml", "Current clock speed (video encoder/decoder domain)", nvidia_gpu_video_clock_cur_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"video_clock_app_target", "nvml", "Application target clock speed (video encoder/decoder domain)", nvidia_gpu_video_clock_app_target_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"video_clock_app_default", "nvml", "Application default clock speed (video encoder/decoder domain)", nvidia_gpu_video_clock_app_default_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"video_clock_boost_max", "nvml", "Application default clock speed (video encoder/decoder domain)", nvidia_gpu_video_clock_boost_max_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MHz"},
    {"pci_bar1_free", "nvml", "Unallocated BAR1 memory", nvidia_gpu_bar1_free_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "Byte"},
    {"pci_bar1_used", "nvml", "Allocated BAR1 memory", nvidia_gpu_bar1_used_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "Byte"},
    {"pci_bar1_total", "nvml", "Total BAR1 memory", nvidia_gpu_bar1_total_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "Byte"},
    {"bus_type", "nvml", "GPU Bus Type", nvidia_gpu_bus_type_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"pcie_link_gen_cur", "nvml", "Current PCIe link generation", nvidia_gpu_pcie_gen_cur_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"pcie_link_gen_gpu_max", "nvml", "Maximum PCIe link generation (GPU only)", nvidia_gpu_pcie_gen_gpu_max_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"pcie_link_gen_sys_max", "nvml", "Maximum PCIe link generation (GPU/system combination)", nvidia_gpu_pcie_gen_sys_max_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"pcie_link_width_cur", "nvml", "Current PCIe link width (GPU only)", nvidia_gpu_pcie_width_cur_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"pcie_link_width_sys_max", "nvml", "Maximum PCIe link width (CPU/system combination)", nvidia_gpu_pcie_width_sys_max_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"pcie_link_rate_cur", "nvml", "Current PCIe link speed", nvidia_gpu_pcie_speed_cur_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MB/s"},
    {"pcie_link_rate_max", "nvml", "Maximum PCIe link speed", nvidia_gpu_pcie_speed_max_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "MB/s"},
    {"pcie_replay_counter", "nvml", "PCIe replay counter", nvidia_gpu_pcie_replay_counter_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"pcie_throughput_tx", "nvml", "Current PCIe throughput (TX)", nvidia_gpu_pcie_throughput_tx_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "KB/s"},
    {"pcie_throughput_rx", "nvml", "Current PCIe throughput (RX)", nvidia_gpu_pcie_throughput_rx_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "KB/s"},
    {"decoder_usage", "nvml", "Current decoder usage", nvidia_gpu_decoder_util_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_enabled_default", "nvml", "DRAM ECC enable default", nvidia_gpu_ecc_default_enabled_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_enabled_cur", "nvml", "DRAM ECC currently enabled", nvidia_gpu_ecc_enabled_cur_getter, nvidia_gpu_ecc_enabled_setter, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_enabled_pend", "nvml", "DRAM ECC enable pending (after reboot)", nvidia_gpu_ecc_enabled_pend_getter, nvidia_gpu_ecc_enabled_setter, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_dram_volatile_corr_err_count", "nvml", "DRAM ECC corrected errors (GPU lifetime aggregate)", nvidia_gpu_ecc_error_dram_vol_corr_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_dram_volatile_uncorr_err_count", "nvml", "DRAM ECC corrected errors (since reboot)", nvidia_gpu_ecc_error_dram_vol_uncorr_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_dram_aggregate_corr_err_count", "nvml", "DRAM ECC corrected errors (GPU lifetime aggregate)", nvidia_gpu_ecc_error_dram_agg_corr_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_dram_aggregate_uncorr_err_count", "nvml", "DRAM ECC corrected errors (since reboot)", nvidia_gpu_ecc_error_dram_agg_uncorr_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_l1_volatile_corr_err_count", "nvml", "L1 ECC corrected errors (GPU lifetime aggregate)", nvidia_gpu_ecc_error_l1_vol_corr_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_l1_volatile_uncorr_err_count", "nvml", "L1 ECC corrected errors (since reboot)", nvidia_gpu_ecc_error_l1_vol_uncorr_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_l1_aggregate_corr_err_count", "nvml", "L1 ECC corrected errors (GPU lifetime aggregate)", nvidia_gpu_ecc_error_l1_agg_corr_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_l1_aggregate_uncorr_err_count", "nvml", "L1 ECC corrected errors (since reboot)", nvidia_gpu_ecc_error_l1_agg_uncorr_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_l2_volatile_corr_err_count", "nvml", "L2 ECC corrected errors (GPU lifetime aggregate)", nvidia_gpu_ecc_error_l2_vol_corr_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_l2_volatile_uncorr_err_count", "nvml", "L2 ECC corrected errors (since reboot)", nvidia_gpu_ecc_error_l2_vol_uncorr_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_l2_aggregate_corr_err_count", "nvml", "L2 ECC corrected errors (GPU lifetime aggregate)", nvidia_gpu_ecc_error_l2_agg_corr_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_l2_aggregate_uncorr_err_count", "nvml", "L2 ECC corrected errors (since reboot)", nvidia_gpu_ecc_error_l2_agg_uncorr_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_reg_volatile_corr_err_count", "nvml", "REG ECC corrected errors (GPU lifetime aggregate)", nvidia_gpu_ecc_error_reg_vol_corr_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_reg_volatile_uncorr_err_count", "nvml", "REG ECC corrected errors (since reboot)", nvidia_gpu_ecc_error_reg_vol_uncorr_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_reg_aggregate_corr_err_count", "nvml", "REG ECC corrected errors (GPU lifetime aggregate)", nvidia_gpu_ecc_error_reg_agg_corr_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"ecc_reg_aggregate_uncorr_err_count", "nvml", "REG ECC corrected errors (since reboot)", nvidia_gpu_ecc_error_reg_agg_uncorr_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"fan_speed", "nvml", "Fan speed", nvidia_gpu_fan_speed_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "%"},
    {"pstate_cur", "nvml", "Current performance state", nvidia_gpu_pstate_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"power_limit_default", "nvml", "Default power limit", nvidia_gpu_power_limit_default_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "W"},
    {"power_limit_cur", "nvml", "Current power limit", nvidia_gpu_power_limit_cur_getter, nvidia_gpu_power_limit_cur_setter, DEVICE_TYPE_NVIDIA_GPU, NULL, "W"},
    {"power_limit_min", "nvml", "Minimum power limit", nvidia_gpu_power_limit_min_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "W"},
    {"power_limit_max", "nvml", "Maximum power limit", nvidia_gpu_power_limit_max_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "W"},
    {"power_limit_enf", "nvml", "Enforced (resulting) power limit", nvidia_gpu_power_limit_enf_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "W"},
    {"power_usage", "nvml", "Current power usage", nvidia_gpu_power_cur_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "W"},
    {"encoder_cap_h264", "nvml", "Video encoder H264 capacity", nvidia_gpu_encoder_capacity_h264_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "%"},
    {"encoder_cap_hevc", "nvml", "Video encoder HEVC capacity", nvidia_gpu_encoder_capacity_hevc_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "%"},
    {"encoder_cap_av1", "nvml", "Video encoder AV1 capacity", nvidia_gpu_encoder_capacity_av1_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "%"},
    {"encoder_usage", "nvml", "Video encoder usage", nvidia_gpu_encoder_usage_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "%"},
    {"gom_cur", "nvml", "Current GPU operation mode (all_on, compute_only, graphics_only)", nvidia_gpu_gom_cur_getter, nvidia_gpu_gom_setter, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"gom_pend", "nvml", "Pending (after reboot) GPU operation mode (all_on, compute_only, graphics_only)", nvidia_gpu_gom_pend_getter, nvidia_gpu_gom_setter, DEVICE_TYPE_NVIDIA_GPU, NULL},
    {"jpg_usage", "nvml", "JPG usage", nvidia_gpu_jpg_usage_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "%"},
    {"dram_bus_width", "nvml", "DRAM bus width", nvidia_gpu_dram_bus_width_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "Bit"},
    {"multi_gpu", "nvml", "Multi GPU board", nvidia_gpu_multi_gpu_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "bool"},
    {"temp_gpu_cur", "nvml", "Current GPU temperature", nvidia_gpu_temp_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "degrees C"},
    {"temp_thresh_shut", "nvml", "Shutdown temperature threshold", nvidia_gpu_temp_thresh_shut_getter, nvidia_gpu_temp_thresh_shut_setter, DEVICE_TYPE_NVIDIA_GPU, NULL, "degrees C"},
    {"temp_thresh_slow", "nvml", "Slowdown Temperature threshold", nvidia_gpu_temp_thresh_slow_getter, nvidia_gpu_temp_thresh_slow_setter, DEVICE_TYPE_NVIDIA_GPU, NULL, "degrees C"},
    {"temp_thresh_mem_max", "nvml", "Maximum memory temperature threshold", nvidia_gpu_temp_thresh_mem_max_getter, nvidia_gpu_temp_thresh_mem_max_setter, DEVICE_TYPE_NVIDIA_GPU, NULL, "degrees C"},
    {"temp_thresh_gpu_max", "nvml", "Maximum GPU temperature threshold", nvidia_gpu_temp_thresh_gpu_max_getter, nvidia_gpu_temp_thresh_gpu_max_setter, DEVICE_TYPE_NVIDIA_GPU, NULL, "degrees C"},
    {"temp_thresh_acou_min", "nvml", "Minimum acoustic temperature threshold", nvidia_gpu_temp_thresh_acou_min_getter, nvidia_gpu_temp_thresh_acou_min_setter, DEVICE_TYPE_NVIDIA_GPU, NULL, "degrees C"},
    {"temp_thresh_acou_cur", "nvml", "Current acoustic temperature threshold", nvidia_gpu_temp_thresh_acou_cur_getter, nvidia_gpu_temp_thresh_acou_cur_setter, DEVICE_TYPE_NVIDIA_GPU, NULL, "degrees C"},
    {"temp_thresh_acou_max", "nvml", "Maximum acoustic temperature threshold", nvidia_gpu_temp_thresh_acou_max_getter, nvidia_gpu_temp_thresh_acou_max_setter, DEVICE_TYPE_NVIDIA_GPU, NULL, "degrees C"},
    {"temp_thresh_gps_cur", "nvml", "Current GPS temperature threshold", nvidia_gpu_temp_thresh_gps_cur_getter, nvidia_gpu_temp_thresh_gps_cur_setter, DEVICE_TYPE_NVIDIA_GPU, NULL, "degrees C"},
    {"ofa_usage", "nvml", "Optical Flow Accelerator usage", nvidia_gpu_ofa_usage_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "%"},
    {"energy", "nvml", "Energy consumed since driver reload", nvidia_gpu_energy_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "J"},
    {"gpu_usage", "nvml", "GPU usage", nvidia_gpu_gpu_usage_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "%"},
    {"dram_bw_usage", "nvml", "DRAM bandwidth usage", nvidia_gpu_dram_usage_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "%"},
    {"gpu_perf_pol_power", "nvml", "Time:duration below app clock due to power limit", nvidia_gpu_policy_time_power_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "us:ns"},
    {"gpu_perf_pol_therm", "nvml", "Time:duration below app clock due to thermal limit", nvidia_gpu_policy_time_therm_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "us:ns"},
    {"gpu_perf_pol_syncboost", "nvml", "Time:duration below app clock due to sync boost", nvidia_gpu_policy_time_sync_boost_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "us:ns"},
    {"gpu_perf_pol_boardlimit", "nvml", "Time:duration below app clock due to board limit", nvidia_gpu_policy_time_board_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "us:ns"},
    {"gpu_perf_pol_lowutil", "nvml", "Time:duration below app clock due to low utilization", nvidia_gpu_policy_time_lowutil_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "us:ns"},
    {"gpu_perf_pol_reliab", "nvml", "Time:duration below app clock due to board reliability limit", nvidia_gpu_policy_time_rel_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "us:ns"},
    {"gpu_perf_pol_totalapp", "nvml", "Total time:duration below app clock", nvidia_gpu_policy_time_total_app_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "us:ns"},
    {"gpu_perf_pol_totalbase", "nvml", "Total time:duration below base clock", nvidia_gpu_policy_time_total_base_getter, NULL, DEVICE_TYPE_NVIDIA_GPU, NULL, "us:ns"},
};

static const _SysFeatureList nvidia_gpu_feature_list = {
    .num_features = ARRAY_COUNT(nvidia_gpu_features),
    .features = nvidia_gpu_features,
};
