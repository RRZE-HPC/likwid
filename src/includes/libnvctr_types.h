/*
 * =======================================================================================
 *
 *      Filename:  libnvctr_types.h
 *
 *      Description:  Types file for libnvctr module.
 *
 *      Version:   5.1
 *      Released:  16.11.2020
 *
 *      Author:   Thomas Gruber (tg), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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
#ifndef LIBNVCTR_H
#define LIBNVCTR_H

#include <bstrlib.h>

#define MAX_NUM_GPUEVENTS 30

typedef enum LikwidGpuStates {
    GPUMARKER_STATE_NEW,
    GPUMARKER_STATE_START,
    GPUMARKER_STATE_STOP
} LikwidGpuStates;

typedef struct LikwidGpuResults {
    bstring  label;
    double time;
    TimerData startTime;
    int groupID;
    int gpuID;
    uint32_t count;
    int nevents;
    double PMcounters[MAX_NUM_GPUEVENTS];
    double StartPMcounters[MAX_NUM_GPUEVENTS];
    LikwidGpuStates state;
} LikwidGpuResults;

typedef struct {
    bstring  tag;
    int groupID;
    int gpuCount;
    int eventCount;
    double*  time;
    uint32_t*  count;
    int* gpulist;
    double** counters;
} LikwidNvResults;

#endif /* LIBNVCTR_H */
