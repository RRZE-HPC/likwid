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
#include <hsa/hsa.h>
#include <rocprofiler.h>

typedef struct {
    int deviceId;
    hsa_agent_t hsa_agent;
    int numAllMetrics;
    rocprofiler_info_data_t* allMetrics;
    rocprofiler_t* context; 
    int numActiveEvents;
    rocprofiler_feature_t* activeEvents;
    uint32_t rocprofilerGroupCount;
} RocmonDevice;

typedef struct {
    int             numGroups;
    int             numActiveGroups;
    GroupInfo       *groups;
    int             numDevices;
    RocmonDevice    *devices; // HSA agents
} RocmonContext;

extern RocmonContext *rocmon_context;

#endif /* LIKWID_ROCMON_TYPES_H */