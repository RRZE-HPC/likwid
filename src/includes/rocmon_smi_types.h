/*
 * =======================================================================================
 *
 *      Filename:  rocmon_smi_types.h
 *
 *      Description:  Header File of rocmon for smi backend.
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
#ifndef LIKWID_ROCMON_SMI_TYPES_H
#define LIKWID_ROCMON_SMI_TYPES_H

#include <amd_smi/amdsmi.h>
#if AMDSMI_LIB_VERSION_YEAR == 23 && AMDSMI_LIB_VERSION_MAJOR == 4 && AMDSMI_LIB_VERSION_MINOR == 0 && AMDSMI_LIB_VERSION_RELEASE == 0
typedef struct metrics_table_header_t metrics_table_header_t;
#endif
#include <rocm_smi/rocm_smi.h>
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
#include <rocmon_common_types.h>

struct RocmonSmiEvent_struct;
typedef int (*RocmonSmiMeasureFunc)(int deviceId, struct RocmonSmiEvent_struct* event, RocmonEventResult* result);

typedef enum {
    ROCMON_SMI_EVENT_TYPE_NORMAL = 0,
    ROCMON_SMI_EVENT_TYPE_VARIANT,
    ROCMON_SMI_EVENT_TYPE_SUBVARIANT,
    ROCMON_SMI_EVENT_TYPE_INSTANCES
} RocmonSmiEventType;

#define MAX_ROCMON_SMI_EVENT_NAME 40
typedef struct RocmonSmiEvent_struct {
    char name[MAX_ROCMON_SMI_EVENT_NAME];
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

#endif /* LIKWID_ROCMON_SMI_TYPES_H */
