/*
 * =======================================================================================
 *
 *      Filename:  perfmon_types.h
 *
 *      Description:  Header File of perfmon module.
 *                    Configures and reads out performance counters
 *                    on x86 based architectures. Supports multi threading.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2013 Jan Treibig 
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

#ifndef PERFMON_TYPES_H
#define PERFMON_TYPES_H

#include <bstrlib.h>
#include <timer.h>

#define MAX_EVENT_OPTIONS 4

/* #####   EXPORTED TYPE DEFINITIONS   #################################### */




/////////////////////////////////////////////
typedef struct {
    int             thread_id;
    int             processorId;
} PerfmonThread;

typedef enum {
    EVENT_OPTION_NONE = 0,
    EVENT_OPTION_OPCODE,
    EVENT_OPTION_MATCH0,
    EVENT_OPTION_MATCH1,
    EVENT_OPTION_MASK0,
    EVENT_OPTION_MASK1,
    EVENT_OPTION_NID,
    EVENT_OPTION_TID,
    EVENT_OPTION_STATE,
    EVENT_OPTION_EDGE,
    EVENT_OPTION_THRESHOLD,
    EVENT_OPTION_INVERT,
    EVENT_OPTION_COUNT_KERNEL,
    EVENT_OPTION_OCCUPANCY,
    EVENT_OPTION_OCCUPANCY_FILTER,
    EVENT_OPTION_OCCUPANCY_EDGE,
    EVENT_OPTION_OCCUPANCY_INVERT,
    NUM_EVENT_OPTIONS
} EventOptionType;

static char* eventOptionTypeName[NUM_EVENT_OPTIONS] = {
    "NONE",
    "OPCODE",
    "MATCH0",
    "MATCH1",
    "MASK0",
    "MASK1",
    "NID",
    "TID",
    "STATE",
    "EDGEDETECT",
    "THRESHOLD",
    "INVERT",
    "COUNT_KERNEL",
    "OCCUPANCY",
    "OCCUPANCY_FILTER",
    "OCCUPANCY_EDGEDETECT",
    "OCCUPANCY_INVERT"
};

#define OPTIONS_TYPE_MASK(type) (type == EVENT_OPTION_NONE ? 0x0 : (1<<type))
#define EVENT_OPTION_NONE_MASK 0x0
#define EVENT_OPTION_OPCODE_MASK (1<<EVENT_OPTION_OPCODE)
#define EVENT_OPTION_MATCH0_MASK (1<<EVENT_OPTION_MATCH0)
#define EVENT_OPTION_MATCH1_MASK (1<<EVENT_OPTION_MATCH1)
#define EVENT_OPTION_MASK0_MASK (1<<EVENT_OPTION_MASK0)
#define EVENT_OPTION_MASK1_MASK (1<<EVENT_OPTION_MASK1)
#define EVENT_OPTION_NID_MASK (1<<EVENT_OPTION_NID)
#define EVENT_OPTION_TID_MASK (1<<EVENT_OPTION_TID)
#define EVENT_OPTION_STATE_MASK (1<<EVENT_OPTION_STATE)
#define EVENT_OPTION_EDGE_MASK (1<<EVENT_OPTION_EDGE)
#define EVENT_OPTION_THRESHOLD_MASK (1<<EVENT_OPTION_THRESHOLD)
#define EVENT_OPTION_INVERT_MASK (1<<EVENT_OPTION_INVERT)
#define EVENT_OPTION_COUNT_KERNEL_MASK (1<<EVENT_OPTION_COUNT_KERNEL)
#define EVENT_OPTION_OCCUPANCY_MASK (1<<EVENT_OPTION_OCCUPANCY)
#define EVENT_OPTION_OCCUPANCY_FILTER_MASK (1<<EVENT_OPTION_OCCUPANCY_FILTER)
#define EVENT_OPTION_OCCUPANCY_EDGE_MASK (1<<EVENT_OPTION_OCCUPANCY_EDGE)
#define EVENT_OPTION_OCCUPANCY_INVERT_MASK (1<<EVENT_OPTION_OCCUPANCY_INVERT)

typedef struct {
    EventOptionType      type;
    uint32_t             value;
} PerfmonEventOption;

typedef struct {
    const char*     name;
    const char*     limit;
    uint16_t        eventId;
    uint8_t         umask;
    uint8_t         cfgBits;
    uint8_t         cmask;
    uint8_t         numberOfOptions;
    uint64_t        optionMask;
    PerfmonEventOption options[MAX_EVENT_OPTIONS];
} PerfmonEvent;

typedef struct {
    int         init;
    int         id;
    int         overflows;
    uint64_t    startData;
    uint64_t    counterData;
} PerfmonCounter;

typedef struct {
    PerfmonEvent        event;
    RegisterIndex       index;
    PerfmonCounter*     threadCounter;
} PerfmonEventSetEntry;

typedef struct {
    int                   numberOfEvents;
    PerfmonEventSetEntry* events;
    TimerData             timer;
    double                rdtscTime;
    uint64_t              regTypeMask;
} PerfmonEventSet;

typedef struct {
    int              numberOfGroups;
    int              numberOfActiveGroups;
    int              activeGroup;
    PerfmonEventSet* groups;
    int              numberOfThreads;
    PerfmonThread*   threads;
} PerfmonGroupSet;



#endif /*PERFMON_TYPES_H*/
