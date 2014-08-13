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
    EVENT_OPTION_ADDR,
    EVENT_OPTION_NID,
    EVENT_OPTION_TID,
    EVENT_OPTION_STATE,
    EVENT_OPTION_EDGE,
    EVENT_OPTION_THRESHOLD,
    EVENT_OPTION_INVERT,
    EVENT_OPTION_COUNT_KERNEL,
    EVENT_OPTION_OCCUPANCY
} EventOptionType;

#define EVENT_MASK_NONE 0x0
#define EVENT_MASK_OPCODE (1<<EVENT_OPTION_OPCODE)
#define EVENT_MASK_ADDR (1<<EVENT_OPTION_ADDR)
#define EVENT_MASK_NID (1<<EVENT_OPTION_NID)
#define EVENT_MASK_TID (1<<EVENT_OPTION_TID)
#define EVENT_MASK_STATE (1<<EVENT_OPTION_STATE)
#define EVENT_MASK_EDGE (1<<EVENT_OPTION_EDGE)
#define EVENT_MASK_THRESHOLD (1<<EVENT_OPTION_THRESHOLD)
#define EVENT_MASK_INVERT (1<<EVENT_OPTION_INVERT)
#define EVENT_MASK_COUNT_KERNEL (1<<EVENT_OPTION_COUNT_KERNEL)
#define EVENT_MASK_OCCUPANCY (1<<EVENT_OPTION_OCCUPANCY)

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
