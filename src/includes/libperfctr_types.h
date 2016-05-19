/*
 * =======================================================================================
 *
 *      Filename:  libperfctr_types.h
 *
 *      Description:  Types file for libperfctr module.
 *
 *      Version:   4.1
 *      Released:  19.5.2016
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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

typedef struct LikwidThreadResults{
    bstring  label;
    double time;
    TimerData startTime;
    int groupID;
    int cpuID;
    uint32_t count;
    double StartPMcounters[NUM_PMC];
    int StartOverflows[NUM_PMC];
    double PMcounters[NUM_PMC];
} LikwidThreadResults;

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
