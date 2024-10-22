/*
 * =======================================================================================
 *
 *      Filename:  rocmon_common_types.h
 *
 *      Description:  Header File of rocmon for v1 and sdk backend.
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
#ifndef LIKWID_ROCMON_COMMON_TYPES_H
#define LIKWID_ROCMON_COMMON_TYPES_H

#include <map.h>

#include <amd_smi/amdsmi.h>
#if AMDSMI_LIB_VERSION_YEAR == 23 && AMDSMI_LIB_VERSION_MAJOR == 4 && AMDSMI_LIB_VERSION_MINOR == 0 && AMDSMI_LIB_VERSION_RELEASE == 0
typedef struct metrics_table_header_t metrics_table_header_t;
#endif
#include <rocm_smi/rocm_smi.h>
#include <hsa.h>
#include <hsa/hsa_ext_amd.h>
#ifdef ROCPROFILER_EXPORT
#undef ROCPROFILER_EXPORT
#endif
#ifdef ROCPROFILER_IMPORT
#undef ROCPROFILER_IMPORT
#endif
#ifdef ROCPROFILER_VERSION_MAJOR
#undef ROCPROFILER_VERSION_MAJOR
#endif
#ifdef ROCPROFILER_VERSION_MINOR
#undef ROCPROFILER_VERSION_MINOR
#endif
#ifdef ROCPROFILER_API
#undef ROCPROFILER_API
#endif
#include <rocprofiler/rocprofiler.h>
#ifdef LIKWID_ROCPROF_SDK
#ifdef ROCPROFILER_EXPORT
#undef ROCPROFILER_EXPORT
#endif
#ifdef ROCPROFILER_IMPORT
#undef ROCPROFILER_IMPORT
#endif
#ifdef ROCPROFILER_VERSION_MAJOR
#undef ROCPROFILER_VERSION_MAJOR
#endif
#ifdef ROCPROFILER_VERSION_MINOR
#undef ROCPROFILER_VERSION_MINOR
#endif
#ifdef ROCPROFILER_API
#undef ROCPROFILER_API
#endif
#include <rocprofiler-sdk/rocprofiler.h>
/*#ifdef ROCPROFILER_EXPORT*/
/*#undef ROCPROFILER_EXPORT*/
/*#endif*/
/*#ifdef ROCPROFILER_IMPORT*/
/*#undef ROCPROFILER_IMPORT*/
/*#endif*/
/*#ifdef ROCPROFILER_VERSION_MAJOR*/
/*#undef ROCPROFILER_VERSION_MAJOR*/
/*#endif*/
/*#ifdef ROCPROFILER_VERSION_MINOR*/
/*#undef ROCPROFILER_VERSION_MINOR*/
/*#endif*/
/*#ifdef ROCPROFILER_API*/
/*#undef ROCPROFILER_API*/
/*#endif*/
#include <rocprofiler-sdk/registration.h>
#endif



#ifndef ROCMWEAK
#define ROCMWEAK __attribute__(( weak ))
#endif
#ifndef FREE_IF_NOT_NULL
#define FREE_IF_NOT_NULL(var) if ( var ) { free( var ); var = NULL; }
#endif
/*#ifndef ARRAY_COUNT*/
/*#define ARRAY_COUNT(arr) (sizeof(arr) / sizeof((arr)[0]))*/
/*#endif*/
/*#ifndef SIZEOF_STRUCT_MEMBER*/
/*#define SIZEOF_STRUCT_MEMBER(type, member) (sizeof(((type *) NULL)->member))*/
/*#endif*/

typedef struct {
    double lastValue;
    double fullValue;
} RocmonEventResult;

typedef struct {
    RocmonEventResult* results; // First rocprofiler results, then SMI results
    int numResults;
} RocmonEventResultList;

#include <rocmon_smi_types.h>
#include <rocmon_sdk_types.h>

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

typedef struct {
    int deviceId; // LIKWID device id
    int rocprof_v1;
    int activeGroup;

    // Rocprofiler V1
    hsa_agent_t hsa_agent;  // HSA agent handle for this device
    rocprofiler_t* v1_context; // Rocprofiler context (has activeEvents configured)
#ifdef LIKWID_ROCPROF_SDK
    // Rocprofiler SDK
    rocprofiler_agent_t agent;
    rocprofiler_context_id_t sdk_context; // Rocprofiler context (has activeEvents configured)
    rocprofiler_buffer_id_t buffer;
    rocprofiler_callback_thread_t thread;
#endif

    // Available rocprofiler metrics
    rocprofiler_info_data_t* v1_rocMetrics;
#ifdef LIKWID_ROCPROF_SDK
    rocprofiler_counter_info_v0_t* sdk_rocMetrics;
#endif
    int numRocMetrics;

    // Available ROCm SMI events
    Map_t smiMetrics;

    // Currently configured rocprofiler events (bound to context)
    rocprofiler_feature_t* v1_activeRocEvents;
#ifdef LIKWID_ROCPROF_SDK
    rocprofiler_counter_info_v0_t* sdk_activeRocEvents;
#endif
    int numActiveRocEvents;

    // Currently configured ROCm SMI events
    RocmonSmiEvent* activeSmiEvents;
    int numActiveSmiEvents;

    // Results for all events in all event sets
    RocmonEventResultList* groupResults;
    int numGroupResults;

#ifdef LIKWID_ROCPROF_SDK
    rocprofiler_profile_config_id_t* profiles;
    int numProfiles;
#endif

    // Timestamps in ns
    struct {
        uint64_t start;
        uint64_t read;
        uint64_t stop;
    } time;

    // buffer?
} RocmonDevice;

typedef enum {
    ROCMON_STATE_FINALIZED = 0,
    ROCMON_STATE_INITIALIZED,
    ROCMON_STATE_SETUP,
    ROCMON_STATE_RUNNING,
    ROCMON_STATE_STOPPED,
    MAX_ROCMON_STATE,
} RocmonContextState;
#define MIN_ROCMON_STATE ROCMON_STATE_FINALIZED

typedef struct {
    int         numGroups;       // Number of allocated groups
    int         numActiveGroups; // Number of used groups
    int         activeGroup;     // Currently active group
    GroupInfo   *groups;

    // Devices (HSA agents)
    RocmonDevice    *devices;
    int             numDevices;

    // System information
    long double hsa_timestamp_factor; // hsa_timestamp * hsa_timestamp_factor = timestamp_in_ns

    // Rocprofiler SDK agents with buffers
#ifdef LIKWID_ROCPROF_SDK
    int num_sdk_agents;
    RocprofilerSdkAgentData* agents;
#endif

    // ROCm SMI events
    Map_t smiEvents;

    // Use legacy rocprofiler v1
    int         use_rocprofiler_v1:1;
    RocmonContextState state;
} RocmonContext;

//extern static RocmonContext* rocmon_context;


#endif /* LIKWID_ROCMON_COMMON_TYPES_H */
