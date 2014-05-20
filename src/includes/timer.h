/*
 * =======================================================================================
 *      Filename:  timer.h
 *
 *      Description:  Measure runtime with getTimeOfday and rdtsc.
 *
 *      A C module to measure runtime. There are two methods: with gettimeofday
 *      for longer time periods (over 0.5 sec) and with rdtsc (read time stamp
 *      counter) for shorter periods. There is a variation for measurements
 *      with rdtsc of 100 cycles in the worst case. Therefore sensible
 *      measurements should be over 1000 cycles.
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

#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>
#include <timer_types.h>

#define RDTSC2(cpu_c) \
__asm__ volatile( "rdtsc\n\t"           \
"movl %%eax, %0\n\t"  \
"movl %%edx, %1\n\t"  \
: "=r" ((cpu_c).int32.lo), "=r" ((cpu_c).int32.hi) \
: : "%eax", "%edx")

#define RDTSC(cpu_c) \
__asm__ volatile("xor %%eax,%%eax\n\t"           \
"cpuid\n\t"           \
"rdtsc\n\t"           \
"movl %%eax, %0\n\t"  \
"movl %%edx, %1\n\t"  \
: "=r" ((cpu_c).int32.lo), "=r" ((cpu_c).int32.hi) \
: : "%eax","%ebx","%ecx","%edx")



/**
 * @brief  Initialize timer module
 *
 * Determines processor clock and cycles for cpuid.
 */
extern void timer_init(void );

/**
 * @brief  Start timer measurement with getTimeofDay
 * @param  time  Reference to struct holding the timestamps
 */
extern void timer_start(TimerData* time);

/**
 * @brief  Stop timer measurement with getTimeofDay
 * @param  time Reference to struct holding the timestamps
 */
extern void timer_stop(TimerData* time);

/**
 * @brief  Get timer measurement with getTimeofDay
 * @param  time Reference to struct holding the timestamps
 * @return Time duration between start and stop in seconds
 */
extern float timer_print(TimerData* timer);

/**
 * @brief  Start cycles measurement with rdtsc
 * @param cycles Reference to struct holding the timestamps 
 */
extern void timer_startCycles(CyclesData* cycles);

/**
 * @brief  Stop cycles measurement with rdtsc
 * @param cycles Reference to struct holding the timestamps 
 */
extern void timer_stopCycles(CyclesData* cycles);


/**
 * @brief  Get time of cycles measurement
 * @param cycles Reference to struct holding the timestamps 
 * @return Timer duration between start and stop in seconds
 */
extern float timer_printCyclesTime(CyclesData* cycles);

/**
 * @brief  Get cycles of cycles measurement
 * @param cycles Reference to struct holding the timestamps 
 * @return cycles between start and stop
 */
extern uint64_t timer_printCycles(CyclesData* cycles);

/**
 * @brief  Get Clock rate of cpu in Hertz 
 * @return clock rate of cpu
 */
extern uint64_t timer_getCpuClock(void);

/**
 * @brief  Get cycles for cpuid
 * @return cycles for cpuid
 */
extern uint64_t timer_getCpuidCycles(void);

#endif /* TIMER_H */
