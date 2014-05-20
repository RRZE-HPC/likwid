/*
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
 *      Version:  <VERSION>
 *      Created:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Company:  RRZE Erlangen
 *      Project:  likwid
 *      Copyright:  Copyright (c) 2010, Jan Treibig
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License, v2, as
 *      published by the Free Software Foundation
 *     
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *     
 *      You should have received a copy of the GNU General Public License
 *      along with this program; if not, write to the Free Software
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *

 *
 * ============================================================================
 */
#ifndef TIMER_H
#define TIMER_H

#include <timer_types.h>


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
