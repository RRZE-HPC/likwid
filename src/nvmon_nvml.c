/*
 * =======================================================================================
 *
 *      Filename:  nvmon_nvml.c
 *
 *      Description:  NVML implementation of the performance monitoring module
 *                    for NVIDIA GPUs
 *
 *      Version:   5.3
 *      Released:  10.11.2023
 *
 *      Author:   Thomas Gruber (tg), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2023 RRZE, University Erlangen-Nuremberg
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include <sys/types.h>

#include <dlfcn.h>
#include <nvml.h>
#include <cupti.h>

#include <likwid.h>
#include <error.h>
#include <nvmon_types.h>
#include <libnvctr_types.h>

typedef enum {
    FEATURE_CLOCK_INFO          = 1,
    FEATURE_ECC_LOCAL_ERRORS    = 2,
    FEATURE_FAN_SPEED           = 4,
    FEATURE_MAX_CLOCK           = 8,
    FEATURE_MEMORY_INFO         = 16,
    FEATURE_PERF_STATES         = 32,
    FEATURE_POWER               = 64,
    FEATURE_TEMP                = 128,
    FEATURE_ECC_TOTAL_ERRORS    = 256,
    FEATURE_UTILIZATION         = 512,
    FEATURE_POWER_MANAGEMENT    = 1024,
    FEATURE_NVML_POWER_MANAGEMENT_LIMIT_CONSTRAINT = 2048,
} NvmlFeature;

typedef enum {
    LOCAL_ECC_REGFILE = 0,
    LOCAL_ECC_L1,
    LOCAL_ECC_L2,
    LOCAL_ECC_MEM,
} NvmlEccErrorCount;

typedef enum {
    MEMORY_FREE = 0,
    MEMORY_TOTAL,
    MEMORY_USED,
} NvmlMemoryType;

typedef enum {
    LIMIT_MIN = 0,
    LIMIT_MAX,
} NvmlPowerLimit;

typedef enum {
    UTILIZATION_GPU = 0,
    UTILIZATION_MEMORY,
} NvmlUtilization;

typedef struct {
    double fullValue;
    double lastValue;
} NvmlEventResult;

struct NvmlEvent_struct;
typedef int (*NvmlMeasureFunc)(nvmlDevice_t device, struct NvmlEvent_struct* event, NvmlEventResult* result);

#define LIKWID_NVML_NAME_LEN 40
#define LIKWID_NVML_DESC_LEN 50
typedef struct NvmlEvent_struct {
    char name[LIKWID_NVML_NAME_LEN];
    char description[LIKWID_NVML_DESC_LEN];
    NvmlMeasureFunc measureFunc;

    union {
        nvmlClockType_t clock;
        struct {
            nvmlMemoryErrorType_t type;
            NvmlEccErrorCount counter;
        } ecc;
        unsigned int fan;
        NvmlMemoryType memory;
        nvmlTemperatureSensors_t tempSensor;
        NvmlPowerLimit powerLimit;
        NvmlUtilization utilization;
    } options;
} NvmlEvent;

typedef struct {
    int numEvents;
    NvmlEvent* events;
    NvmlEventResult* results;
} NvmlEventSet;

typedef struct {
    NvmonDevice* nvDevice;
    nvmlDevice_t nvmlDevice;

    int numAllEvents;
    NvmlEvent* allEvents;

    int activeEventSet;
    int numEventSets;
    NvmlEventSet* eventSets;

    uint32_t features;
    unsigned int numFans;

    // Timestamps in ns
    struct {
        uint64_t start;
        uint64_t read;
        uint64_t stop;
    } time;
} NvmlDevice;

typedef struct {
    int numDevices;
    NvmlDevice* devices;
} NvmlContext;


// Variables
static int nvml_initialized = 0;
static void* dl_nvml = NULL;
static void* dl_cupti = NULL;
static NvmlContext nvmlContext;


// Macros
#define FREE_IF_NOT_NULL(x) if (x) { free(x); }
#define DLSYM_AND_CHECK( dllib, name ) name##_ptr = dlsym( dllib, #name ); if ( dlerror() != NULL ) { return -1; }
#define NVML_CALL(call, args, handleerror)                                            \
    do {                                                                           \
        nvmlReturn_t _status = (*call##_ptr)args;                                         \
        if (_status != NVML_SUCCESS) {                                            \
            fprintf(stderr, "Error: function %s failed with error %d.\n", #call, _status);                    \
            handleerror;                                                             \
        }                                                                          \
    } while (0)
#define CUPTI_CALL(call, args, handleerror)                                            \
    do {                                                                \
        CUptiResult _status = (*call##_ptr)args;                                  \
        if (_status != CUPTI_SUCCESS) {                                 \
            const char *errstr;                                         \
            (*cuptiGetResultString)(_status, &errstr);               \
            fprintf(stderr, "Error: function %s failed with error %s.\n", #call, errstr); \
            handleerror;                                                \
        }                                                               \
    } while (0)

// NVML function declarations
#define NVMLWEAK __attribute__(( weak ))
#define DECLAREFUNC_NVML(funcname, funcsig) nvmlReturn_t NVMLWEAK funcname funcsig;  nvmlReturn_t ( *funcname##_ptr ) funcsig;

DECLAREFUNC_NVML(nvmlInit_v2, (void));
DECLAREFUNC_NVML(nvmlShutdown, (void));
DECLAREFUNC_NVML(nvmlDeviceGetHandleByIndex_v2, (unsigned int  index, nvmlDevice_t* device));
DECLAREFUNC_NVML(nvmlDeviceGetClockInfo, (nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock));
DECLAREFUNC_NVML(nvmlDeviceGetInforomVersion, (nvmlDevice_t device, nvmlInforomObject_t object, char* version, unsigned int  length));
DECLAREFUNC_NVML(nvmlDeviceGetEccMode, (nvmlDevice_t device, nvmlEnableState_t* current, nvmlEnableState_t* pending));
DECLAREFUNC_NVML(nvmlDeviceGetDetailedEccErrors, (nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlEccErrorCounts_t* eccCounts));
DECLAREFUNC_NVML(nvmlDeviceGetTotalEccErrors, (nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long* eccCounts));
DECLAREFUNC_NVML(nvmlDeviceGetFanSpeed_v2, (nvmlDevice_t device, unsigned int fan, unsigned int* speed));
DECLAREFUNC_NVML(nvmlDeviceGetClock, (nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int* clockMHz));
DECLAREFUNC_NVML(nvmlDeviceGetMemoryInfo, (nvmlDevice_t device, nvmlMemory_t* memory));
DECLAREFUNC_NVML(nvmlDeviceGetPerformanceState, (nvmlDevice_t device, nvmlPstates_t* pState));
DECLAREFUNC_NVML(nvmlDeviceGetPowerUsage, (nvmlDevice_t device, unsigned int* power));
DECLAREFUNC_NVML(nvmlDeviceGetTemperature, (nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int* temp));
DECLAREFUNC_NVML(nvmlDeviceGetPowerManagementLimit, (nvmlDevice_t device, unsigned int* limit));
DECLAREFUNC_NVML(nvmlDeviceGetPowerManagementLimitConstraints, (nvmlDevice_t device, unsigned int* minLimit, unsigned int* maxLimit));
DECLAREFUNC_NVML(nvmlDeviceGetUtilizationRates, (nvmlDevice_t device, nvmlUtilization_t* utilization));

// CUPTI function declarations
#define CUPTIWEAK __attribute__(( weak ))
#define DECLAREFUNC_CUPTI(funcname, funcsig) CUptiResult CUPTIWEAK funcname funcsig;  CUptiResult( *funcname##_ptr ) funcsig;

DECLAREFUNC_CUPTI(cuptiGetTimestamp, (uint64_t * timestamp));
DECLAREFUNC_CUPTI(cuptiGetResultString, (CUptiResult result, const char **str));


// ----------------------------------------------------
//   Wrapper functions
// ----------------------------------------------------

static void
_nvml_resultAddMeasurement(NvmlEventResult* result, double value)
{
    result->lastValue = value;
    result->fullValue += value;
}

static int
_nvml_wrapper_getClockInfo(nvmlDevice_t device, NvmlEvent* event, NvmlEventResult* result)
{
    unsigned int clock;

    NVML_CALL(nvmlDeviceGetClockInfo, (device, event->options.clock, &clock), return -1);
    _nvml_resultAddMeasurement(result, clock);

    return 0;
}


static int
_nvml_wrapper_getMaxClock(nvmlDevice_t device, NvmlEvent* event, NvmlEventResult* result)
{
    unsigned int clock;

    NVML_CALL(nvmlDeviceGetClock, (device, event->options.clock, NVML_CLOCK_ID_CUSTOMER_BOOST_MAX, &clock), return -1);
    _nvml_resultAddMeasurement(result, clock);

    return 0;
}


static int
_nvml_wrapper_getEccLocalErrors(nvmlDevice_t device, NvmlEvent* event, NvmlEventResult* result)
{
    nvmlEccErrorCounts_t counts;

    NVML_CALL(nvmlDeviceGetDetailedEccErrors, (device, event->options.ecc.type, NVML_VOLATILE_ECC, &counts), return -1);
    switch (event->options.ecc.counter)
    {
    case LOCAL_ECC_L1:      _nvml_resultAddMeasurement(result, counts.l1Cache);         break;
    case LOCAL_ECC_L2:      _nvml_resultAddMeasurement(result, counts.l2Cache);         break;
    case LOCAL_ECC_MEM:     _nvml_resultAddMeasurement(result, counts.deviceMemory);    break;
    case LOCAL_ECC_REGFILE: _nvml_resultAddMeasurement(result, counts.registerFile);    break;
    default:                return -1;
    }

    return 0;
}


static int
_nvml_wrapper_getEccTotalErrors(nvmlDevice_t device, NvmlEvent* event, NvmlEventResult* result)
{
    unsigned long long count;

    NVML_CALL(nvmlDeviceGetTotalEccErrors, (device, event->options.ecc.type, NVML_VOLATILE_ECC, &count), return -1);
    _nvml_resultAddMeasurement(result, count);

    return 0;
}


static int
_nvml_wrapper_getFanSpeed(nvmlDevice_t device, NvmlEvent* event, NvmlEventResult* result)
{
    unsigned int speed;

    NVML_CALL(nvmlDeviceGetFanSpeed_v2, (device, event->options.fan, &speed), return -1);
    _nvml_resultAddMeasurement(result, speed);
    
    return 0;
}


static int
_nvml_wrapper_getMemoryInfo(nvmlDevice_t device, NvmlEvent* event, NvmlEventResult* result)
{
    nvmlMemory_t memory;

    NVML_CALL(nvmlDeviceGetMemoryInfo, (device, &memory), return -1);
    switch (event->options.memory)
    {
    case MEMORY_FREE:   _nvml_resultAddMeasurement(result, memory.free);    break;
    case MEMORY_TOTAL:  _nvml_resultAddMeasurement(result, memory.total);   break;
    case MEMORY_USED:   _nvml_resultAddMeasurement(result, memory.used);    break;
    default:            return -1;
    }
    
    return 0;
}


static int
_nvml_wrapper_getPerformanceState(nvmlDevice_t device, NvmlEvent* event, NvmlEventResult* result)
{
    nvmlPstates_t state;

    NVML_CALL(nvmlDeviceGetPerformanceState, (device, &state), return -1);
    _nvml_resultAddMeasurement(result, state);

    return 0;
}


static int
_nvml_wrapper_getPowerUsage(nvmlDevice_t device, NvmlEvent* event, NvmlEventResult* result)
{
    unsigned int power;

    NVML_CALL(nvmlDeviceGetPowerUsage, (device, &power), return -1);
    _nvml_resultAddMeasurement(result, power);

    return 0;
}


static int
_nvml_wrapper_getTemperature(nvmlDevice_t device, NvmlEvent* event, NvmlEventResult* result)
{
    unsigned int temp;

    NVML_CALL(nvmlDeviceGetTemperature, (device, event->options.tempSensor, &temp), return -1);
    _nvml_resultAddMeasurement(result, temp);

    return 0;
}


static int
_nvml_wrapper_getPowerManagementLimit(nvmlDevice_t device, NvmlEvent* event, NvmlEventResult* result)
{
    unsigned int limit;

    NVML_CALL(nvmlDeviceGetPowerManagementLimit, (device, &limit), return -1);
    _nvml_resultAddMeasurement(result, limit);

    return 0;
}


static int
_nvml_wrapper_getPowerManagementLimitConstraints(nvmlDevice_t device, NvmlEvent* event, NvmlEventResult* result)
{
    unsigned int maxLimit;
    unsigned int minLimit;

    NVML_CALL(nvmlDeviceGetPowerManagementLimitConstraints, (device, &minLimit, &maxLimit), return -1);
    if (event->options.powerLimit == LIMIT_MIN)
    {
        _nvml_resultAddMeasurement(result, minLimit);
    }
    else if (event->options.powerLimit == LIMIT_MAX)
    {
        _nvml_resultAddMeasurement(result, maxLimit);
    }

    return 0;
}


static int
_nvml_wrapper_getUtilization(nvmlDevice_t device, NvmlEvent* event, NvmlEventResult* result)
{
    nvmlUtilization_t utilization;

    NVML_CALL(nvmlDeviceGetUtilizationRates, (device, &utilization), return -1);
    switch (event->options.utilization)
    {
    case UTILIZATION_GPU:       _nvml_resultAddMeasurement(result, utilization.gpu);        break;
    case UTILIZATION_MEMORY:    _nvml_resultAddMeasurement(result, utilization.memory);     break;
    default:                    return -1;
    }
    
    return 0;
}


// ----------------------------------------------------
//   Helper functions
// ----------------------------------------------------

static int
_nvml_linkLibraries()
{
    // Load NVML libary and link functions
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Init NVML Libaries);
    dl_nvml = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_nvml || dlerror() != NULL)
    {
        fprintf(stderr, "NVML library libnvidia-ml.so not found.");
        return -1;
    }

    DLSYM_AND_CHECK(dl_nvml, nvmlInit_v2);
    DLSYM_AND_CHECK(dl_nvml, nvmlShutdown);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetHandleByIndex_v2);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetClockInfo);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetInforomVersion);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetEccMode);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetDetailedEccErrors);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetTotalEccErrors);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetFanSpeed_v2);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetClock);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetMemoryInfo);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPerformanceState);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPowerUsage);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetTemperature);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPowerManagementLimit);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetPowerManagementLimitConstraints);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetUtilizationRates);

    // Load CUPTI library and link functions
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Init NVML Libaries);
    dl_cupti = dlopen("libcupti.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_cupti || dlerror() != NULL)
    {
        fprintf(stderr, "CUPTI library libcupti.so not found.");
        return -1;
    }

    DLSYM_AND_CHECK(dl_cupti, cuptiGetTimestamp);
    DLSYM_AND_CHECK(dl_cupti, cuptiGetResultString);

    return 0;
}


static int
_nvml_getEventsForDevice(NvmlDevice* device)
{
    NvmlEvent* event = device->allEvents;

    if (device->features & FEATURE_CLOCK_INFO)
    {
        snprintf(event->name, LIKWID_NVML_NAME_LEN, "CLOCK_GRAPHICS");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Graphics clock domain in MHz");
        event->measureFunc = &_nvml_wrapper_getClockInfo;
        event->options.clock = NVML_CLOCK_GRAPHICS;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "CLOCK_SM");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "SM clock domain in MHz");
        event->measureFunc = &_nvml_wrapper_getClockInfo;
        event->options.clock = NVML_CLOCK_SM;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "CLOCK_MEM");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Memory clock domain in MHz");
        event->measureFunc = &_nvml_wrapper_getClockInfo;
        event->options.clock = NVML_CLOCK_MEM;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "CLOCK_VIDEO");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Video clock domain in MHz");
        event->measureFunc = &_nvml_wrapper_getClockInfo;
        event->options.clock = NVML_CLOCK_VIDEO;
        event++;
    }

    if (device->features & FEATURE_MAX_CLOCK)
    {
        snprintf(event->name, LIKWID_NVML_NAME_LEN, "MAX_CLOCK_GRAPHICS");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Maximum graphics clock domain in MHz");
        event->measureFunc = &_nvml_wrapper_getMaxClock;
        event->options.clock = NVML_CLOCK_GRAPHICS;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "MAX_CLOCK_SM");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Maximum SM clock domain in MHz");
        event->measureFunc = &_nvml_wrapper_getMaxClock;
        event->options.clock = NVML_CLOCK_SM;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "MAX_CLOCK_MEM");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Maximum memory clock domain in MHz");
        event->measureFunc = &_nvml_wrapper_getMaxClock;
        event->options.clock = NVML_CLOCK_MEM;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "MAX_CLOCK_VIDEO");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Maximum video clock domain in MHz");
        event->measureFunc = &_nvml_wrapper_getClockInfo;
        event->options.clock = NVML_CLOCK_VIDEO;
        event++;
    }

    if (device->features & FEATURE_ECC_LOCAL_ERRORS)
    {
        // Single bit errors
        snprintf(event->name, LIKWID_NVML_NAME_LEN, "L1_LOCAL_ECC_ERRORS_SINGLE_BIT");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "L1 cache single bit ECC errors");
        event->measureFunc = &_nvml_wrapper_getEccLocalErrors;
        event->options.ecc.type = NVML_MEMORY_ERROR_TYPE_CORRECTED;
        event->options.ecc.counter = LOCAL_ECC_L1;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "L2_LOCAL_ECC_ERRORS_SINGLE_BIT");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "L2 cache single bit ECC errors");
        event->measureFunc = &_nvml_wrapper_getEccLocalErrors;
        event->options.ecc.type = NVML_MEMORY_ERROR_TYPE_CORRECTED;
        event->options.ecc.counter = LOCAL_ECC_L2;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "MEM_LOCAL_ECC_ERRORS_SINGLE_BIT");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Memory single bit ECC errors");
        event->measureFunc = &_nvml_wrapper_getEccLocalErrors;
        event->options.ecc.type = NVML_MEMORY_ERROR_TYPE_CORRECTED;
        event->options.ecc.counter = LOCAL_ECC_MEM;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "REGFILE_LOCAL_ECC_ERRORS_SINGLE_BIT");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Register file single bit ECC errors");
        event->measureFunc = &_nvml_wrapper_getEccLocalErrors;
        event->options.ecc.type = NVML_MEMORY_ERROR_TYPE_CORRECTED;
        event->options.ecc.counter = LOCAL_ECC_REGFILE;
        event++;

        // Double bit errors
        snprintf(event->name, LIKWID_NVML_NAME_LEN, "L1_LOCAL_ECC_ERRORS_DOUBLE_BIT");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "L1 cache double bit ECC errors");
        event->measureFunc = &_nvml_wrapper_getEccLocalErrors;
        event->options.ecc.type = NVML_MEMORY_ERROR_TYPE_UNCORRECTED;
        event->options.ecc.counter = LOCAL_ECC_L1;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "L2_LOCAL_ECC_ERRORS_DOUBLE_BIT");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "L2 cache double bit ECC errors");
        event->measureFunc = &_nvml_wrapper_getEccLocalErrors;
        event->options.ecc.type = NVML_MEMORY_ERROR_TYPE_UNCORRECTED;
        event->options.ecc.counter = LOCAL_ECC_L2;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "MEM_LOCAL_ECC_ERRORS_DOUBLE_BIT");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Memory double bit ECC errors");
        event->measureFunc = &_nvml_wrapper_getEccLocalErrors;
        event->options.ecc.type = NVML_MEMORY_ERROR_TYPE_UNCORRECTED;
        event->options.ecc.counter = LOCAL_ECC_MEM;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "REGFILE_LOCAL_ECC_ERRORS_DOUBLE_BIT");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Register file double bit ECC errors");
        event->measureFunc = &_nvml_wrapper_getEccLocalErrors;
        event->options.ecc.type = NVML_MEMORY_ERROR_TYPE_UNCORRECTED;
        event->options.ecc.counter = LOCAL_ECC_REGFILE;
        event++;
    }

    if (device->features & FEATURE_ECC_TOTAL_ERRORS)
    {
        snprintf(event->name, LIKWID_NVML_NAME_LEN, "TOTAL_ECC_ERRORS_SINGLE_BIT");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Total single bit ECC errors");
        event->measureFunc = &_nvml_wrapper_getEccTotalErrors;
        event->options.ecc.type = NVML_MEMORY_ERROR_TYPE_CORRECTED;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "TOTAL_ECC_ERRORS_DOUBLE_BIT");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Total double bit ECC errors");
        event->measureFunc = &_nvml_wrapper_getEccTotalErrors;
        event->options.ecc.type = NVML_MEMORY_ERROR_TYPE_UNCORRECTED;
        event++;
    }

    if (device->features & FEATURE_FAN_SPEED)
    {
        for (int i = 0; i < device->numFans; i++)
        {
            snprintf(event->name, LIKWID_NVML_NAME_LEN, "FAN_SPEED[%d]", i);
            snprintf(event->description, LIKWID_NVML_DESC_LEN, "Indended fan speed represented as a percentage of the maximum noise tolerance fan speed. May exceed 100 in certain cases");
            event->measureFunc = &_nvml_wrapper_getFanSpeed;
            event->options.fan = i;
            event++;
        }
    }

    if (device->features & FEATURE_MEMORY_INFO)
    {
        snprintf(event->name, LIKWID_NVML_NAME_LEN, "FREE_MEMORY");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Unallocated FB memory (in bytes)");
        event->measureFunc = &_nvml_wrapper_getMemoryInfo;
        event->options.memory = MEMORY_FREE;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "TOTAL_MEMORY");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Total installed FB memory (in bytes)");
        event->measureFunc = &_nvml_wrapper_getMemoryInfo;
        event->options.memory = MEMORY_TOTAL;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "USED_MEMORY");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Allocated FB memory (in bytes). Note that the driver/GPU always sets aside a small amount of memory for bookkeeping");
        event->measureFunc = &_nvml_wrapper_getMemoryInfo;
        event->options.memory = MEMORY_USED;
        event++;
    }

    if (device->features & FEATURE_PERF_STATES)
    {
        snprintf(event->name, LIKWID_NVML_NAME_LEN, "PERF_STATE");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Current performance state for the device");
        event->measureFunc = &_nvml_wrapper_getPerformanceState;
        event++;
    }

    if (device->features & FEATURE_POWER)
    {
        snprintf(event->name, LIKWID_NVML_NAME_LEN, "POWER_USAGE");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Power usage for this GPU in milliwatts and its associated circuitry (e.g. memory)");
        event->measureFunc = &_nvml_wrapper_getPowerUsage;
        event++;
    }

    if (device->features & FEATURE_TEMP)
    {
        snprintf(event->name, LIKWID_NVML_NAME_LEN, "TEMP_GPU");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Current temperature readings for the device, in degrees C");
        event->measureFunc = &_nvml_wrapper_getPowerUsage;
        event->options.tempSensor = NVML_TEMPERATURE_GPU;
        event++;
    }

    if (device->features & FEATURE_POWER_MANAGEMENT)
    {
        snprintf(event->name, LIKWID_NVML_NAME_LEN, "POWER_LIMIT");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Power management limit associated with this device in milliwatts");
        event->measureFunc = &_nvml_wrapper_getPowerManagementLimit;
        event++;
    }

    if (device->features & FEATURE_NVML_POWER_MANAGEMENT_LIMIT_CONSTRAINT)
    {
        snprintf(event->name, LIKWID_NVML_NAME_LEN, "POWER_LIMIT_MIN");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Minimum power management limit in milliwatts");
        event->measureFunc = &_nvml_wrapper_getPowerManagementLimitConstraints;
        event->options.powerLimit = LIMIT_MIN;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "POWER_LIMIT_MAX");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Maximum power management limit in milliwatts");
        event->measureFunc = &_nvml_wrapper_getPowerManagementLimitConstraints;
        event->options.powerLimit = LIMIT_MAX;
        event++;
    }

    if (device->features & FEATURE_UTILIZATION)
    {
        snprintf(event->name, LIKWID_NVML_NAME_LEN, "GPU_UTILIZATION");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Percent of time over the past sample period during which one or more kernels was executing on the GPU");
        event->measureFunc = &_nvml_wrapper_getUtilization;
        event->options.utilization = UTILIZATION_GPU;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "MEMORY_UTILIZATION");
        snprintf(event->description, LIKWID_NVML_DESC_LEN, "Percent of time over the past sample period during which global (device) memory was being read or written");
        event->measureFunc = &_nvml_wrapper_getUtilization;
        event->options.utilization = UTILIZATION_MEMORY;
        event++;
    }

    return 0;
}


static void
_nvml_getEccFeaturesOfDevice(NvmlDevice* device)
{
    char inforomECC[16];
    float ecc_version = 0;
    nvmlEnableState_t mode = NVML_FEATURE_DISABLED;
    nvmlEnableState_t pendingmode = NVML_FEATURE_DISABLED;

    /*
    For Tesla and Quadro products from Fermi and Kepler families.
    requires NVML_INFOROM_ECC 2.0 or higher for location-based counts
    requires NVML_INFOROM_ECC 1.0 or higher for all other ECC counts
    requires ECC mode to be enabled.
    */

    // Query ecc version
    if ((*nvmlDeviceGetInforomVersion_ptr)(device->nvmlDevice, NVML_INFOROM_ECC, inforomECC, 16) == NVML_SUCCESS)
    {
        ecc_version = strtof(inforomECC, NULL);
    }

    // Query ecc mode
    if ((*nvmlDeviceGetEccMode_ptr)(device->nvmlDevice, &mode, &pendingmode) == NVML_SUCCESS) {
        if (mode == NVML_FEATURE_ENABLED) {
            if (ecc_version >= 2.0) {
                device->features |= FEATURE_ECC_LOCAL_ERRORS;
                device->numAllEvents += 8; /* {single bit, two bit errors} x { reg, l1, l2, memory } */
            }
            if (ecc_version >= 1.0) {
                device->features |= FEATURE_ECC_TOTAL_ERRORS;
                device->numAllEvents += 2; /* single bit errors, double bit errors */
            }
        }
    }
}


static int
_nvml_getFeaturesOfDevice(NvmlDevice* device)
{
    unsigned int value;

    /*
    Features copied from PAPI nvml component (https://bitbucket.org/icl/papi/src/master/src/components/nvml/linux-nvml.c)
    */

    // Reset state
    device->features = 0;
    device->numAllEvents = 0;

    // Check FEATURE_CLOCK_INFO
    if ((*nvmlDeviceGetClockInfo_ptr)(device->nvmlDevice, NVML_CLOCK_GRAPHICS, &value) == NVML_SUCCESS)
    {
        device->features |= FEATURE_CLOCK_INFO;
        device->numAllEvents += 4;
    }

    // Check FEATURE_MAX_CLOCK
    if ((*nvmlDeviceGetClock_ptr)(device->nvmlDevice, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CUSTOMER_BOOST_MAX, &value) == NVML_SUCCESS)
    {
        device->features |= FEATURE_MAX_CLOCK;
        device->numAllEvents += 4;
    }

    // Check ECC features
    _nvml_getEccFeaturesOfDevice(device);

    // Check FEATURE_FAN_SPEED
    while (1)
    {
        if ((*nvmlDeviceGetFanSpeed_v2_ptr)(device->nvmlDevice, device->numFans, &value) != NVML_SUCCESS)
        {
            break;
        }
        device->features |= FEATURE_FAN_SPEED;
        device->numAllEvents += 1;
        device->numFans++;
    }

    // All products support FEATURE_MEMORY_INFO
    device->features |= FEATURE_MEMORY_INFO;
    device->numAllEvents += 3;

    // Check FEATURE_PERF_STATES
    nvmlPstates_t state;
    if ((*nvmlDeviceGetPerformanceState_ptr)(device->nvmlDevice, &state) == NVML_SUCCESS)
    {
        device->features |= FEATURE_PERF_STATES;
        device->numAllEvents += 1;
    }

    // Check FEATURE_POWER
    if ((*nvmlDeviceGetPowerUsage_ptr)(device->nvmlDevice, &value) == NVML_SUCCESS)
    {
        device->features |= FEATURE_POWER;
        device->numAllEvents += 1;
    }

    // Check FEATURE_TEMP
    if ((*nvmlDeviceGetTemperature_ptr)(device->nvmlDevice, NVML_TEMPERATURE_GPU, &value) == NVML_SUCCESS)
    {
        device->features |= FEATURE_TEMP;
        device->numAllEvents += 1;
    }

    // Check FEATURE_POWER_MANAGEMENT
    if ((*nvmlDeviceGetPowerManagementLimit_ptr)(device->nvmlDevice, &value) == NVML_SUCCESS)
    {
        device->features |= FEATURE_POWER_MANAGEMENT;
        device->numAllEvents += 1;
    }

    // Check FEATURE_NVML_POWER_MANAGEMENT_LIMIT_CONSTRAINT
    unsigned int minLimit, maxLimit;
    if ((*nvmlDeviceGetPowerManagementLimitConstraints_ptr)(device->nvmlDevice, &minLimit, &maxLimit) == NVML_SUCCESS)
    {
        device->features |= FEATURE_NVML_POWER_MANAGEMENT_LIMIT_CONSTRAINT;
        device->numAllEvents += 2;
    }

    // Check FEATURE_UTILIZATION
    nvmlUtilization_t utilization;
    if ((*nvmlDeviceGetUtilizationRates_ptr)(device->nvmlDevice, &utilization) == NVML_SUCCESS)
    {
        device->features |= FEATURE_UTILIZATION;
        device->numAllEvents += 2;
    }

    return 0;
}


static int
_nvml_createDevice(int idx, NvmlDevice* device)
{
    int ret;

    // Set corresponding nvmon device
    device->nvDevice = &nvGroupSet->gpus[idx];
    device->activeEventSet = 0;
    device->numEventSets = 0;
    device->eventSets = NULL;
    device->numFans = 0;

    // Get NVML device handle
    NVML_CALL(nvmlDeviceGetHandleByIndex_v2, (device->nvDevice->deviceId, &device->nvmlDevice), {
        ERROR_PRINT(Failed to get device handle for GPU %d, device->nvDevice->deviceId);
        return -1;
    });

    ret = _nvml_getFeaturesOfDevice(device);
    if (ret < 0) return ret;

    // Allocate memory for event list
    device->allEvents = (NvmlEvent*) malloc(device->numAllEvents * sizeof(NvmlEvent));
    if (device->allEvents == NULL)
    {
        ERROR_PRINT(Failed to allocate memory for event list of GPU %d, device->nvDevice->deviceId);
        return -ENOMEM;
    }

    ret = _nvml_getEventsForDevice(device);
    if (ret < 0) return ret;

    return 0;
}


static int
_nvml_readCounters(void (*saveTimestamp)(NvmlDevice* device, uint64_t timestamp), void (*afterMeasure)(NvmlEventResult* result))
{
    int ret;

    // Get timestamp
    uint64_t timestamp;
    CUPTI_CALL(cuptiGetTimestamp, (&timestamp), return -EFAULT);
    if (ret < 0)
    {
        return -EFAULT;
    }

    for (int i = 0; i < nvmlContext.numDevices; i++)
    {
        NvmlDevice* device = &nvmlContext.devices[i];
        NvmlEventSet* eventSet = &device->eventSets[device->activeEventSet];

        // Save timestamp
        if (saveTimestamp)
        {
            saveTimestamp(device, timestamp);
        }

        // Read value of each event
        for (int i = 0; i < eventSet->numEvents; i++)
        {
            NvmlEvent* event = &eventSet->events[i];
            NvmlEventResult* result = &eventSet->results[i];
            if (event->measureFunc)
            {
                ret = event->measureFunc(device->nvmlDevice, event, result);
                if (ret < 0) return ret;

                if (afterMeasure)
                {
                    afterMeasure(result);
                }
            }
        }
    }

    return 0;
}


static void
_nvml_saveStartTime(NvmlDevice* device, uint64_t timestamp)
{
    device->time.start = timestamp;
    device->time.read = timestamp;
}


static void
_nvml_resetFullValue(NvmlEventResult* result)
{
    result->fullValue = 0;
}


static void
_nvml_saveReadTime(NvmlDevice* device, uint64_t timestamp)
{
    device->time.read = timestamp;
}


static void
_nvml_saveStopTime(NvmlDevice* device, uint64_t timestamp)
{
    device->time.stop = timestamp;
}


// ----------------------------------------------------
//   Exported functions
// ----------------------------------------------------

int
nvml_init()
{
    int ret;

    if (nvml_initialized == 1)
    {
        return 0;
    }

    ret = _nvml_linkLibraries();
    if (ret < 0)
    {
        ERROR_PLAIN_PRINT(Failed to link libraries);
        return -1;
    }

    // Allocate space for nvml specific structures
    nvmlContext.numDevices = nvGroupSet->numberOfGPUs;
    nvmlContext.devices = (NvmlDevice*) malloc(nvmlContext.numDevices * sizeof(NvmlDevice));
    if (nvmlContext.devices == NULL)
    {   
        ERROR_PLAIN_PRINT(Cannot allocate NVML device structures);
        return -ENOMEM;
    }

    // Init NVML
    NVML_CALL(nvmlInit_v2, (), return -1);

    // Do device specific setup
    for (int i = 0; i < nvmlContext.numDevices; i++)
    {
        NvmlDevice* device = &nvmlContext.devices[i];
        ret = _nvml_createDevice(i, device);
        if (ret < 0)
        {
            ERROR_PRINT(Failed to create device #%d, i);
            return ret;
        }
    }

    nvml_initialized = 1;
    return 0;
}


void
nvml_finalize()
{
    if (nvmlContext.devices)
    {
        for (int i = 0; i < nvmlContext.numDevices; i++)
        {
            NvmlDevice* device = &nvmlContext.devices[i];

            FREE_IF_NOT_NULL(device->allEvents);
            for (int j = 0; j < device->numEventSets; j++)
            {
                FREE_IF_NOT_NULL(device->eventSets[j].events);
                FREE_IF_NOT_NULL(device->eventSets[j].results);
            }
            FREE_IF_NOT_NULL(device->eventSets);
        }
        free(nvmlContext.devices);
    }

    // Shutdown NVML
    NVML_CALL(nvmlShutdown, (), return);
}


int
nvml_addEventSet(char** events, int numEvents)
{
    // Allocate memory for event results
    for (int i = 0; i < nvmlContext.numDevices; i++)
    {
        NvmlDevice* device = &nvmlContext.devices[i];

        // Allocate new event set in device
        NvmlEvent* tmpEvents = (NvmlEvent*) malloc(numEvents * sizeof(NvmlEvent));
        if (tmpEvents == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate events for new event set);
            return -ENOMEM;
        }
        NvmlEventResult* tmpResults = (NvmlEventResult*) malloc(numEvents * sizeof(NvmlEventResult));
        if (tmpResults == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate event results);
            free(tmpEvents);
            return -ENOMEM;
        }
        NvmlEventSet* tmpEventSets = (NvmlEventSet*) realloc(device->eventSets, (device->numEventSets+1) * sizeof(NvmlEventSet));
        if (tmpEventSets == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate new event set);
            free(tmpEvents);
            free(tmpResults);
            return -ENOMEM;
        }

        // Copy event information
        for (int j = 0; j < numEvents; j++)
        {
            // Search for it in allEvents
            int idx = -1;
            for (int k = 0; k < device->numAllEvents; k++)
            {
                if (strcmp(device->allEvents[k].name, events[j]) == 0)
                {
                    idx = k;
                    break;
                }
            }

            // Check if event was found
            if (idx < 0)
            {
                ERROR_PRINT(Could not find event %s, events[j]);
                return -EINVAL;
            }

            // Copy whole event into activeEvents array
            memcpy(&tmpEvents[j], &device->allEvents[idx], sizeof(NvmlEvent));
        }

        device->eventSets = tmpEventSets;
        device->eventSets[device->numEventSets].numEvents = numEvents;
        device->eventSets[device->numEventSets].events = tmpEvents;
        device->eventSets[device->numEventSets].results = tmpResults;
        device->numEventSets++;
    }

    return 0;
}


int
nvml_setupCounters(int gid)
{
    // Update active events of each device
    for (int i = 0; i < nvmlContext.numDevices; i++)
    {
        nvmlContext.devices[i].activeEventSet = gid;
    }

    return 0;
}


// Strings inside structure are only valid as long as nvmon/nvml is initialized
int
nvml_getEventsOfGpu(int gpuId, NvmonEventList_t* output)
{
    int gpuIdx = -1;

    // Find index with given gpuId
    for (int i = 0; i < nvmlContext.numDevices; i++)
    {
        if (nvmlContext.devices[i].nvDevice->deviceId == gpuId)
        {
            gpuIdx = i;
            break;
        }
    }
    if (gpuIdx < 0)
    {
        return -EINVAL;
    }

    // Get device handle
    NvmlDevice* device = &nvmlContext.devices[gpuIdx];

    // Allocate space for output structure
    NvmonEventListEntry* entries = (NvmonEventListEntry*) malloc(device->numAllEvents * sizeof(NvmonEventListEntry));
    if (entries == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate event list entries);
        return -ENOMEM;
    }
    NvmonEventList* list = (NvmonEventList*) malloc(sizeof(NvmonEventList));
    if (list == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate event list);
        free(entries);
        return -ENOMEM;
    }

    // Fill structure
    for (int i = 0; i < device->numAllEvents; i++)
    {
        NvmlEvent* event = &device->allEvents[i];
        NvmonEventListEntry* entry = &entries[i];
        int len;

        entry->name = event->name;
        entry->desc = "No description"; // TODO: Add event descriptions
        entry->limit = "GPU";
    }

    list->events = entries;
    list->numEvents = device->numAllEvents;
    *output = list;

    return 0;
}


void
nvml_returnEventsOfGpu(NvmonEventList_t list)
{
    if (list == NULL)
    {
        return;
    }

    if (list->events != NULL && list->numEvents > 0)
    {
        // Event entries do not have owned strings, so nothing to free per entry
        free(list->events);
    }

    free(list);
}


int
nvml_startCounters()
{
    int ret;

    // Ensure nvml is initialized
    if (!nvml_initialized)
    {
        return -EFAULT;
    }

    // Read initial counter values and reset full value
    ret = _nvml_readCounters(_nvml_saveStartTime, _nvml_resetFullValue);
    if (ret < 0) return ret;

    return 0;
}


int
nvml_stopCounters()
{
    int ret;

    // Ensure nvml is initialized
    if (!nvml_initialized)
    {
        return -EFAULT;
    }

    // Read counters
    ret = _nvml_readCounters(_nvml_saveStopTime, NULL);
    if (ret < 0) return ret;

    return 0;
}


int
nvml_readCounters()
{
    int ret;

    // Ensure nvml is initialized
    if (!nvml_initialized)
    {
        return -EFAULT;
    }

    // Read counters
    ret = _nvml_readCounters(_nvml_saveReadTime, NULL);
    if (ret < 0) return ret;

    return 0;
}


int
nvml_getNumberOfEvents(int groupId)
{
    // Ensure nvml is initialized
    if (!nvml_initialized)
    {
        return -EFAULT;
    }

    // Verify that at least one device is registered
    if (nvmlContext.numDevices < 1)
    {
        return 0; // No events registered
    }

    // Verify groupId
    NvmlDevice* device = &nvmlContext.devices[0];
    if (groupId < 0 || groupId >= device->numEventSets)
    {
        return -EINVAL;
    }

    // Events are the same on all devices, take the first
    return device->eventSets[groupId].numEvents;
}


double
nvml_getResult(int gpuIdx, int groupId, int eventId)
{
    // Ensure nvml is initialized
    if (!nvml_initialized)
    {
        return -EFAULT;
    }

    // Validate gpuIdx
    if (gpuIdx < 0 || gpuIdx >= nvmlContext.numDevices)
    {
        return -EINVAL;
    }

    // Validate groupId
    NvmlDevice* device = &nvmlContext.devices[gpuIdx];
    if (groupId < 0 || groupId >= device->numEventSets)
    {
        return -EINVAL;
    }

    // Validate eventId
    NvmlEventSet* eventSet = &device->eventSets[groupId];
    if (eventId < 0 || eventId >= eventSet->numEvents)
    {
        return -EINVAL;
    }

    // Return result
    return eventSet->results[eventId].fullValue;
}


double
nvml_getLastResult(int gpuIdx, int groupId, int eventId)
{
    // Ensure nvml is initialized
    if (!nvml_initialized)
    {
        return -EFAULT;
    }

    // Validate gpuIdx
    if (gpuIdx < 0 || gpuIdx >= nvmlContext.numDevices)
    {
        return -EINVAL;
    }

    // Validate groupId
    NvmlDevice* device = &nvmlContext.devices[gpuIdx];
    if (groupId < 0 || groupId >= device->numEventSets)
    {
        return -EINVAL;
    }

    // Validate eventId
    NvmlEventSet* eventSet = &device->eventSets[groupId];
    if (eventId < 0 || eventId >= eventSet->numEvents)
    {
        return -EINVAL;
    }

    // Return result
    return eventSet->results[eventId].lastValue;
}


double 
nvml_getTimeOfGroup(int groupId)
{
    double time = 0;

    // Ensure nvml is initialized
    if (!nvml_initialized)
    {
        return -EFAULT;
    }

    // Validate gpuIdx
    if (groupId < 0 || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return -EINVAL;
    }

    // Get largest time measured
    for (int i = 0; i < nvmlContext.numDevices; i++)
    {
        time = MAX(time, (double)(nvmlContext.devices[i].time.stop - nvmlContext.devices[i].time.start));
    }

    // Return time as seconds
    return time*1E-9;
}


double 
nvml_getLastTimeOfGroup(int groupId)
{
    double time = 0;

    // Ensure nvml is initialized
    if (!nvml_initialized)
    {
        return -EFAULT;
    }

    // Validate gpuIdx
    if (groupId < 0 || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return -EINVAL;
    }

    // Get largest time measured
    for (int i = 0; i < nvmlContext.numDevices; i++)
    {
        time = MAX(time, (double)(nvmlContext.devices[i].time.stop - nvmlContext.devices[i].time.read));
    }

    // Return time as seconds
    return time*1E-9;
}


double
nvml_getTimeToLastReadOfGroup(int groupId)
{
    double time = 0;

    // Ensure nvml is initialized
    if (!nvml_initialized)
    {
        return -EFAULT;
    }

    // Validate gpuIdx
    if (groupId < 0 || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return -EINVAL;
    }

    // Get largest time measured
    for (int i = 0; i < nvmlContext.numDevices; i++)
    {
        time = MAX(time, (double)(nvmlContext.devices[i].time.read - nvmlContext.devices[i].time.start));
    }

    // Return time as seconds
    return time*1E-9;
}