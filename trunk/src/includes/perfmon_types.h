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
#include <perfmon_group_types.h>
#include <timer.h>

/* #####   EXPORTED TYPE DEFINITIONS   #################################### */

typedef enum {
    PMC0 = 0,
    PMC1 , PMC2 , PMC3 , PMC4 , PMC5 , PMC6,
    PMC7 , PMC8 , PMC9 , PMC10, PMC11, PMC12,
    PMC13, PMC14, PMC15, PMC16, PMC17, PMC18,
    PMC19, PMC20, PMC21, PMC22, PMC23, PMC24,
    PMC25, PMC26, PMC27, PMC28, PMC29, PMC30,
    PMC31, PMC32, PMC33, PMC34, PMC35, PMC36,
    PMC37, PMC38, PMC39, PMC40, PMC41, PMC42,
    PMC43, PMC44, PMC45, PMC46, PMC47, PMC48,
    PMC49, PMC50, PMC51, PMC52, PMC53, PMC54,
    PMC55, PMC56, PMC57, PMC58, PMC59, PMC60,
    PMC61, PMC62, PMC63, PMC64, PMC65, PMC66,
    PMC67, PMC68, PMC69, PMC70, PMC71, PMC72,
    PMC73, PMC74, PMC75, PMC76, PMC77, PMC78,
    PMC79, PMC80, PMC81, PMC82, PMC83, PMC84,
    NUM_PMC} PerfmonCounterIndex;

typedef enum {
    PMC = 0,
    FIXED, THERMAL, UNCORE,
    MBOX0, MBOX1, MBOX2, MBOX3, MBOXFIX,
    BBOX0, BBOX1,
    RBOX0, RBOX1,
    WBOX,
    SBOX0, SBOX1, SBOX2,
    CBOX0, CBOX1, CBOX2, CBOX3, CBOX4,
    CBOX5, CBOX6, CBOX7, CBOX8, CBOX9,
    CBOX10, CBOX11,
    PBOX, POWER,
    NUM_UNITS} PerfmonType;

typedef struct {
    char*               key;
    PerfmonCounterIndex index;
    PerfmonType         type;
    uint64_t            configRegister;
    uint64_t            counterRegister;
    uint64_t            counterRegister2;
    PciDeviceIndex      device;
} PerfmonCounterMap;

typedef struct {
    char*           key;
    PerfmonGroup    index;
    int             isUncore;
    char*           info;
    char*           config;
} PerfmonGroupMap;

typedef struct {
    char*           key;
    char*           msg;
} PerfmonGroupHelp;

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
    uint64_t    startData;
    uint64_t    counterData;
} PerfmonCounter;

typedef struct {
    PerfmonEvent        event;
    PerfmonCounterIndex index;
    PerfmonCounter*     threadCounter;
} PerfmonEventSetEntry;

typedef struct {
    int                   numberOfEvents;
    PerfmonEventSetEntry* events;
    TimerData             timer;
    double                rdtscTime;
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
