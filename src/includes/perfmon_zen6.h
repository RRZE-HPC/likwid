/*
 * =======================================================================================
 *
 *      Filename:  perfmon_zen6.h
 *
 *      Description:  Header file of perfmon module for AMD Family 1A (ZEN6)
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tg), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2017 RRZE, University Erlangen-Nuremberg
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
#ifndef PERFMON_ZEN6_H
#define PERFMON_ZEN6_H

#include <perfmon_zen6_events.h>
#include <perfmon_zen6_counters.h>
#include <error.h>
#include <affinity.h>
#include <cpuid.h>

static int perfmon_numCountersZen6 = NUM_COUNTERS_ZEN6;
static int perfmon_numArchEventsZen6 = NUM_ARCH_EVENTS_ZEN6;


int perfmon_init_zen6(int cpu_id)
{
    lock_acquire((int*) &socket_lock[affinity_thread2socket_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &core_lock[affinity_thread2core_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &sharedl3_lock[affinity_thread2sharedl3_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &numa_lock[affinity_thread2numa_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &die_lock[affinity_thread2die_lookup[cpu_id]], cpu_id);
    return 0;
}

int perfmon_setupCounterThread_zen6(int thread_id, PerfmonEventSet* eventSet)
{
    return 0;
}

int perfmon_startCountersThread_zen6(int thread_id, PerfmonEventSet* eventSet)
{
    return 0;
}

int perfmon_stopCountersThread_zen6(int thread_id, PerfmonEventSet* eventSet)
{
    return 0;
}

int perfmon_readCountersThread_zen6(int thread_id, PerfmonEventSet* eventSet)
{
    return 0;
}

int perfmon_finalizeCountersThread_zen6(int thread_id, PerfmonEventSet* eventSet)
{
    return 0;
}


#endif // PERFMON_ZEN6_H
