/*
 * ===========================================================================
 *
 *      Filename:  perfmon_types.h
 *
 *      Description:  Header File of perfmon module.
 *                    Configures and reads out performance counters
 *                    on x86 based architectures. Supports multi threading.
 *
 *      Version:  <VERSION>
 *      Created:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Company:  RRZE Erlangen
 *      Project:  likwid
 *      Copyright:  Copyright (c) 2010, Jan Treibig
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License, v2, as
 *      published by the Free Software Foundation
 *     
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *     
 *      You should have received a copy of the GNU General Public License
 *      along with this program; if not, write to the Free Software
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * ===========================================================================
 */

#ifndef PERFMON_TYPES_H
#define PERFMON_TYPES_H

#include <bstrlib.h>
#include <perfmon_group_types.h>

/* #####   EXPORTED TYPE DEFINITIONS   #################################### */

typedef enum {
    PMC0 = 0,
    PMC1,
    PMC2,
    PMC3,
    PMC4,
    PMC5,
    PMC6,
    PMC7,
    PMC8,
    PMC9,
    PMC10,
    PMC11,
    PMC12,
    PMC13,
    PMC14,
    NUM_PMC} PerfmonCounterIndex;


typedef enum {
    PMC = 0,
    FIXED,
    UNCORE} PerfmonType;

typedef struct {
    char* key;
    PerfmonCounterIndex index;
} PerfmonCounterMap;

typedef struct {
    char* key;
    PerfmonGroup index;
    char* info;
    char* config;
} PerfmonGroupMap;

typedef struct {
    char* key;
    char* msg;
} PerfmonGroupHelp;


typedef struct {
    PerfmonType  type;
    int       init;
    uint64_t  configRegister;
    uint64_t  counterRegister;
    uint64_t  counterData;
} PerfmonCounter;

typedef struct {
    int processorId;
    PerfmonCounter counters[NUM_PMC];
} PerfmonThread;

typedef struct {
    char*    name;
    char*    limit;
    uint32_t eventId;
    uint32_t umask;
} PerfmonEvent;

typedef struct {
    PerfmonEvent event;
    PerfmonCounterIndex index;
    double* result;
} PerfmonEventSetEntry;

typedef struct {
    int numberOfEvents;
    PerfmonEventSetEntry* events;
} PerfmonEventSet;


typedef struct {
    bstring label;
    double* value;
} PerfmonResult;

typedef struct {
    bstrList* header;
    int numRows;
    int numColumns;
    PerfmonResult* rows;
} PerfmonResultTable;


#endif /*PERFMON_TYPES_H*/
