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
#include <rocprofiler.h>

typedef struct {
    double lastValue;
    double fullValue;
} RocmonEventResult;

typedef struct {
    RocmonEventResult* results;
    int numResults;
} RocmonEventResultList;

typedef struct {
    int deviceId; // LIKWID device id

    hsa_agent_t hsa_agent;  // HSA agent handle for this device
    rocprofiler_t* context; // Rocprofiler context (has activeEvents configured)

    // Available rocprofiler metrics
    rocprofiler_info_data_t* allMetrics;
    int numAllMetrics;

    // Currently configured events (bound to context)
    rocprofiler_feature_t* activeEvents;
    int numActiveEvents;

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
} RocmonContext;

extern RocmonContext *rocmon_context;

#endif /* LIKWID_ROCMON_TYPES_H */