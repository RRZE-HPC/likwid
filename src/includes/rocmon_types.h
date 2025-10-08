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

typedef struct {
    double lastValue;
    double fullValue;
} RocmonEventResult;

typedef struct {
    RocmonEventResult* results; // First rocprofiler results, then SMI results
    size_t numResults;
} RocmonEventResultList;



struct RocmonSmiEvent_struct;
typedef int (*RocmonSmiMeasureFunc)(int deviceId, struct RocmonSmiEvent_struct* event, RocmonEventResult* result);

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
    int hipDeviceIdx; // HIP device id

    uint32_t rsmiDeviceId;
    const rocprofiler_agent_v0_t *rocprofAgent;
    hipDeviceProp_t hipProps;

    rocprofiler_context_id_t rocprof_ctx;

    // Available rocprofiler events
    Map_t availableRocprofEvents;

    // ROCm SMI events (available on hardware)
    // event_name (const char *) -> event (RocmonSmiEvent *)
    Map_t availableSmiEvents;

    // Currently configured rocprofiler events (bound to context)
    //XYZrocprofiler_feature_t* activeRocEvents;
    size_t numActiveRocEvents;

    // ROCm SMI events (currently enabled)
    RocmonSmiEvent* activeSmiEvents;
    size_t numActiveSmiEvents;

    // Results for all events in all event sets
    RocmonEventResultList* groupResults;
    size_t numGroupResults;

    // Timestamps in ns
    struct {
        uint64_t start;
        uint64_t read;
        uint64_t stop;
    } time;
} RocmonDevice;

typedef struct {
    // Event Groups
    GroupInfo   *groups;
    int         numGroups;       // Number of allocated groups
    int         numActiveGroups; // Number of used groups
    int         activeGroup;     // Currently active group

    // Devices (HSA agents)
    rocprofiler_context_id_t rocprof_ctx;
    RocmonDevice             *devices;
    size_t                   numDevices;

    // System information
    long double hsa_timestamp_factor; // hsa_timestamp * hsa_timestamp_factor = timestamp_in_ns

    // ROCm SMI events (implemented by LIKWID)
    // label_name (const char *) -> event_list (RocmonSmiEventList *)
    Map_t implementedSmiEvents;
} RocmonContext;

extern RocmonContext *rocmon_context;


typedef struct {
    bstring  tag;
    int groupID;
    int gpuCount;
    int eventCount;
    double*  time;
    uint32_t*  count;
    int* gpulist;
    double** counters;
} LikwidRocmResults;
#endif /* LIKWID_ROCMON_TYPES_H */
