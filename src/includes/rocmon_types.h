/*
 * =======================================================================================
 *
 *      Filename:  rocmon_types.h
 *
 *      Description:  Header File of rocmon module.
 *                    Configures and reads out performance counters
 *                    on AMD GPUs. Supports multi GPUs.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tg), thomas.gruber@googlemail.com
 *                Various people at HPE
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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
#ifndef LIKWID_ROCMON_TYPES_H
#define LIKWID_ROCMON_TYPES_H

#include <likwid.h>
// #include <hsa.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/context.h>
#include <map.h>
#include <bstrlib.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    double lastValue;
    double fullValue;
} RocmonEventResult;

typedef struct {
    RocmonEventResult *eventResults; // First rocprofiler results, then SMI results
    size_t numEventResults;
} RocmonEventResultList;

struct RocmonSmiEvent_struct;
typedef int (*RocmonSmiMeasureFunc)(uint32_t rsmiDevId, struct RocmonSmiEvent_struct* event, RocmonEventResult* result);

typedef enum {
    ROCMON_SMI_EVENT_TYPE_NORMAL = 0,
    ROCMON_SMI_EVENT_TYPE_VARIANT,
    ROCMON_SMI_EVENT_TYPE_SUBVARIANT,
    ROCMON_SMI_EVENT_TYPE_INSTANCES
} RocmonSmiEventType;

typedef struct RocmonSmiEvent_struct {
    char name[64];
    uint64_t variant;
    uint64_t subvariant;
    uint64_t extra;
    RocmonSmiEventType type;
    RocmonSmiMeasureFunc measureFunc;
} RocmonSmiEvent;

typedef struct {
    RocmonSmiEvent* entries;
    size_t numEntries;
} RocmonSmiEventList;

typedef struct {
    rocprofiler_counter_info_v1_t counterInfo;
    // TODO, do we need anything else here?
} RocmonRprEvent;

typedef struct {
    // 'enabled' is true if the device is an enabled HIP device.
    // Because we cannot init HIP before rocprofiler-sdk, we have allocate
    // all devices. But you can later check this flag, whether to actually
    // monitor the device or not.
    bool enabled;

    // HIP stuff
    int hipDeviceId;
    hipDeviceProp_t hipProps;

    // ROCm SMI stuff
    uint32_t rsmiDeviceId;
    uint32_t pciDomain;
    uint32_t pciLocation;

    // rocprofiler-sdk stuff
    const rocprofiler_agent_v0_t *rocprofAgent;
    rocprofiler_buffer_id_t rocprofBuf;
    rocprofiler_callback_thread_t rocprofThrd;

    // Available rocprofiler events
    // event_name (const char *) -> event (RocmonRprEvent *)
    Map_t availableRprEvents;

    // ROCm SMI events (available on hardware)
    // event_name (const char *) -> event (RocmonSmiEvent *)
    Map_t availableSmiEvents;

    // Currently configured rocprofiler events (bound to context)
    rocprofiler_counter_id_t *activeRprEvents;
    size_t numActiveRprEvents;

    // ROCm SMI events (currently enabled)
    RocmonSmiEvent* activeSmiEvents;
    size_t numActiveSmiEvents;

    // Results for all events in all event sets
    RocmonEventResultList* groupResults;
    size_t numGroupResults;

    // Temporary results, which are written by the buffer callback
    // On read, they are transferred to the respective groupResults
    // counter_id (const char *) -> double
    Map_t callbackRprResults;
    pthread_mutex_t callbackRprMutex;
} RocmonDevice;

typedef struct {
    GroupInfo groupInfo;

    // Timestamps in ns
    struct {
        uint64_t start;
        uint64_t read;
        uint64_t stop;
    } time;
} RocmonGroupInfo;

typedef struct {
    // Event Groups
    RocmonGroupInfo *groups;
    size_t          numGroups;       // Number of groups
    size_t          activeGroupIdx;  // Currently active group

    // Devices
    rocprofiler_context_id_t rocprofCtx;
    RocmonDevice             *devices;
    size_t                   numDevices;

    // Devices (HIP only)
    size_t *hipDeviceIdxToRocmonDeviceIdx;
    size_t numHipDeviceIdxToRocmonDeviceIdx;

    // System information
    long double hsa_timestamp_factor; // hsa_timestamp * hsa_timestamp_factor = timestamp_in_ns

    // ROCm SMI events (implemented by LIKWID)
    // label_name (const char *) -> event_list (RocmonSmiEventList *)
    Map_t implementedSmiEvents;
} RocmonContext;

typedef struct {
    RocmonEventResult   *counterValues;
    size_t              numCounterValues;
} RocmarkerGpuResultList;

typedef struct {
    char                *tag;
    int                 groupId;     // event set group ID which the region uses
    bool                started;     // Is this region currently executing?
    size_t              execCount;  // times this region was started and stopped
    uint64_t            lastStartTime;   // timestamp when the region started measuring last
    uint64_t            lastStopTime;    // timestamp when the region stopped measuring last
    uint64_t            totalTime;      // total time spent in region
    RocmarkerGpuResultList *gpuResults; // array of n elements, where n is == numHipDeviceIds
} RocmarkerRegion;

typedef struct {
    char *name;
    char *formula;
} RocmarkerMetric;

typedef struct {
    char *eventName;
    char *counterName;
} RocmarkerEvent;

typedef struct {
    RocmarkerEvent *events;
    size_t numEvents;

    RocmarkerMetric *metrics;
    size_t numMetrics;

    int groupId;
} RocmarkerGroup;

typedef struct {
    // Capture the thread, which the rocmon Marker API was initialized with.
    // Only this thread may call functions of the marker API.
    pid_t main_tid;

    // GPU device IDs
    int    *hipDeviceIds;
    size_t numHipDeviceIds;

    // group info
    RocmarkerGroup *groups;
    size_t numGroups;
    size_t activeGroupIdx;

    // Region information and results
    // event_name (const char *) -> event (RocmarkerRegion *)
    Map_t regions;
} RocmarkerContext;

#endif /* LIKWID_ROCMON_TYPES_H */
