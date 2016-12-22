/*
 * =======================================================================================
 *
 *      Filename:  perfmon_perf.h
 *
 *      Description: Header file of example perfmon module for software events using
 *                   the perf_event interface
 *
 *      Version:   4.2
 *      Released:  22.12.2016
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
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

#ifndef PERFMON_PERF_H
#define PERFMON_PERF_H

#include <perfmon_types.h>

#define MAX_SW_EVENTS 9


extern int init_perf_event(int cpu_id);

extern int setup_perf_event(int cpu_id, PerfmonEvent *event);

extern int read_perf_event(int cpu_id, uint64_t eventID, uint64_t *data);

extern int stop_perf_event(int cpu_id, uint64_t eventID);
extern int stop_all_perf_event(int cpu_id);

extern int clear_perf_event(int cpu_id, uint64_t eventID);
extern int clear_all_perf_event(int cpu_id);

extern int start_perf_event(int cpu_id, uint64_t eventID);
extern int start_all_perf_event(int cpu_id);

extern int close_perf_event(int cpu_id, uint64_t eventID);

extern int finalize_perf_event(int cpu_id);

#endif
