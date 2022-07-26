/*
 * =======================================================================================
 *
 *      Filename:  libperfctr_types.h
 *
 *      Description:  Types file for libperfctr module.
 *
 *      Version:   5.2.2
 *      Released:  26.07.2022
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg
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
#ifndef LIBPERFCTR_H
#define LIBPERFCTR_H

#include <bstrlib.h>
#include <map.h>

typedef enum LikwidThreadStates {
    MARKER_STATE_NEW,
    MARKER_STATE_START,
    MARKER_STATE_STOP
} LikwidThreadStates;

typedef struct LikwidThreadResults{
    bstring  label;
    uint32_t index;
    double time;
    TimerData startTime;
    int groupID;
    int cpuID;
    uint32_t count;
    double StartPMcounters[NUM_PMC];
    int StartOverflows[NUM_PMC];
    double PMcounters[NUM_PMC];
    LikwidThreadStates state;
} LikwidThreadResults;

typedef struct {
    int thread_id;
    int cpu_id;
    uint64_t last;
    LikwidThreadResults* last_res;
    Map_t regions;
    uint64_t _padding[4];
} GroupThreadsMap;

typedef struct {
    int numberOfThreads;
    GroupThreadsMap *threads;
} GroupThreads;

typedef struct {
    int numberOfGroups;
    GroupThreads *groups;
} MarkerGroups;

typedef struct {
    bstring  tag;
    int groupID;
    int threadCount;
    int eventCount;
    double*  time;
    uint32_t*  count;
    int* cpulist;
    double** counters;
} LikwidResults;

#endif /*LIBPERFCTR_H*/
