/*
 * =======================================================================================
 *
 *      Filename:  nvmon_types.h
 *
 *      Description:  Header File of nvmon module.
 *                    Configures and reads out performance counters
 *                    on NVIDIA GPUs. Supports multi GPUs.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tg), thomas.gruber@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2019 RRZE, University Erlangen-Nuremberg
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
#ifndef ROCPROFILER_VERSION_MAJOR
#ifdef HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE
#undef HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE
#endif
#include <rocprofiler/rocprofiler.h>
#endif
#include <amd_smi/amdsmi.h>
#if AMDSMI_LIB_VERSION_YEAR == 23 && AMDSMI_LIB_VERSION_MAJOR == 4 && AMDSMI_LIB_VERSION_MINOR == 0 && AMDSMI_LIB_VERSION_RELEASE == 0
typedef struct metrics_table_header_t metrics_table_header_t;
#endif
#include <rocm_smi/rocm_smi.h>
#include <map.h>

typedef struct {
    double lastValue;
    double fullValue;
} RocmonEventResult;

typedef struct {
    RocmonEventResult* results; // First rocprofiler results, then SMI results
    int numResults;
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
    char name[40];
    uint64_t variant;
    uint64_t subvariant;
    uint64_t extra;
    int instances;
    RocmonSmiEventType type;
    RocmonSmiMeasureFunc measureFunc;
} RocmonSmiEvent;

typedef struct {
    RocmonSmiEvent* entries;
    int numEntries;
} RocmonSmiEventList;

typedef struct {
    int deviceId; // LIKWID device id

    hsa_agent_t hsa_agent;  // HSA agent handle for this device
    rocprofiler_t* context; // Rocprofiler context (has activeEvents configured)

    // Available rocprofiler metrics
    rocprofiler_info_data_t* rocMetrics;
    int numRocMetrics;

    // Available ROCm SMI events
    Map_t smiMetrics;

    // Currently configured rocprofiler events (bound to context)
    rocprofiler_feature_t* activeRocEvents;
    int numActiveRocEvents;

    // Currently configured ROCm SMI events
    RocmonSmiEvent* activeSmiEvents;
    int numActiveSmiEvents;

    // Results for all events in all event sets
    RocmonEventResultList* groupResults;
    int numGroupResults;

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
    RocmonDevice    *devices;
    int             numDevices;

    // System information
    long double hsa_timestamp_factor; // hsa_timestamp * hsa_timestamp_factor = timestamp_in_ns

    // ROCm SMI events
    Map_t smiEvents;
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
