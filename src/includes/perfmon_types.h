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
 *      Copyright (C) 2012 Jan Treibig 
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
#include <perfmon_group_types.h>

/* #####   EXPORTED TYPE DEFINITIONS   #################################### */

typedef enum {
    PMC0 = 0,
    PMC1, PMC2, PMC3, PMC4, PMC5, PMC6,
    PMC7, PMC8, PMC9, PMC10, PMC11, PMC12,
    PMC13, PMC14, PMC15, PMC16, PMC17, PMC18,
    PMC19, PMC20, PMC21, PMC22, PMC23, PMC24,
    PMC25, PMC26, PMC27, PMC28, PMC29, PMC30,
    PMC31, PMC32, PMC33, PMC34, PMC35, PMC36,
    PMC37, PMC38, PMC39, PMC40, PMC41, PMC42,
    PMC43, PMC44, PMC45, PMC46, PMC47,
    NUM_PMC} PerfmonCounterIndex;

typedef enum {
    PMC = 0,
    FIXED,
    UNCORE,
    MBOX0,
    MBOX1,
    MBOX2,
    MBOX3,
    MBOXFIX,
    BBOX0,
    BBOX1,
    RBOX0,
    RBOX1,
    WBOX,
    SBOX0,
    SBOX1,
    SBOX2,
    CBOX,
    PBOX,
    POWER,
    NUM_UNITS} PerfmonType;

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

/* only used in westmereEX at the moment */
typedef struct {
    uint32_t  ctrlRegister;
    uint32_t  statusRegister;
    uint32_t  ovflRegister;
} PerfmonUnit;

typedef struct {
    PerfmonType  type;
    int       init;
    int       id;
    uint64_t  configRegister;
    uint64_t  counterRegister;
    uint64_t  counterRegister2;
    uint64_t  counterData;
} PerfmonCounter;

typedef struct {
    int processorId;
    PerfmonCounter counters[NUM_PMC];
} PerfmonThread;

typedef struct {
    const char*    name;
    const char*    limit;
    uint16_t eventId;
    uint8_t umask;
    uint8_t cfgBits;
    uint8_t cmask;
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
