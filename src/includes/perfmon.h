/*
 * =======================================================================================
 *
 *      Filename:  perfmon.h
 *
 *      Description:  Header File of perfmon module.
 *                    Configures and reads out performance counters
 *                    on x86 based architectures. Supports multi threading.
 *
 *      Version:   5.1.0
 *      Released:  20.11.2020
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2020 RRZE, University Erlangen-Nuremberg
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

#include <types.h>
#include <likwid.h>

#define FREEZE_FLAG_ONLYFREEZE 0x0ULL
#define FREEZE_FLAG_CLEAR_CTR (1ULL<<1)
#define FREEZE_FLAG_CLEAR_CTL (1ULL<<0)

extern uint64_t **currentConfig;

extern int (*perfmon_startCountersThread) (int thread_id, PerfmonEventSet* eventSet);
extern int (*perfmon_stopCountersThread) (int thread_id, PerfmonEventSet* eventSet);
extern int (*perfmon_setupCountersThread) (int thread_id, PerfmonEventSet* eventSet);
extern int (*perfmon_readCountersThread) (int thread_id, PerfmonEventSet* eventSet);
extern int (*perfmon_finalizeCountersThread) (int thread_id, PerfmonEventSet* eventSet);
extern int (*initThreadArch) (int cpu_id);

/* Internal helpers */
extern int getCounterTypeOffset(int index);
extern uint64_t perfmon_getMaxCounterValue(RegisterType type);
extern char** getArchRegisterTypeNames();

#endif /*PERFMON_H*/
