/*
 * =======================================================================================
 *
 *      Filename:  perfmon.h
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

#ifndef PERFMON_H
#define PERFMON_H

#include <bstrlib.h>
#include <types.h>
#include <topology.h>

extern int perfmon_verbose;
extern PerfmonGroupSet *groupSet;

extern int (*perfmon_startCountersThread) (int thread_id, PerfmonEventSet* eventSet);
extern int (*perfmon_stopCountersThread) (int thread_id, PerfmonEventSet* eventSet);
extern int  (*perfmon_getIndex) (bstring reg, PerfmonCounterIndex* index);
extern int (*perfmon_setupCounterThread) (int thread_id, PerfmonEventSet* eventSet);

extern int perfmon_addEventSet(char* eventCString);
extern int perfmon_setupCounters(int groupId);
extern int perfmon_startCounters(void);
extern int perfmon_stopCounters(void);
extern int perfmon_readCounters(void);
int perfmon_initThread(int thread_id, int cpu_id);
extern int perfmon_init(int nrThreads, int threadsToCpu[]);
extern void perfmon_finalize(void);
uint64_t perfmon_getResults(int groupId, int eventId, int threadId);
void perfmon_switchActiveGroup(int new_group);

#if 0
extern void perfmon_initEventSet(StrUtilEventSet* eventSetConfig, PerfmonEventSet* set);
extern void perfmon_setCSVMode(int v);
extern void perfmon_printAvailableGroups(void);
extern void perfmon_printGroupHelp(bstring group);
extern void perfmon_init(int numThreads, int threads[],FILE* outstream);
extern void perfmon_finalize(void);
extern void perfmon_setupEventSet(bstring eventString, BitMask* mask);
extern double perfmon_getEventResult(int thread, int index);
extern int perfmon_setupEventSetC(char* eventCString, const char*** eventnames);
extern void perfmon_setupCounters(void);
extern void perfmon_startCounters(void);
extern void perfmon_stopCounters(void);
extern void perfmon_readCounters(void);
extern double perfmon_getResult(int threadId, char* counterString);
extern void perfmon_printMarkerResults(bstring filepath);
extern void perfmon_logCounterResults(double time);
extern void perfmon_printCounterResults(void);
extern void perfmon_printCounters(void);
extern void perfmon_printEvents(void);
#endif

#endif /*PERFMON_H*/
