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

/* #####   EXPORTED TYPE DEFINITIONS   #################################### */




/////////////////////////////////////////////
typedef struct {
    int             thread_id;
    int             processorId;
} PerfmonThread;

typedef struct {
    const char*     name;
    const char*     limit;
    uint16_t        eventId;
    uint8_t         umask;
    uint8_t         cfgBits;
    uint8_t         cmask;
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
    uint8_t               measureFixed;
    uint8_t               measurePMC;
    uint8_t               measurePMCUncore;
    uint8_t               measurePCIUncore;
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
