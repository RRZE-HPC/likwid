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
 *      Version:   5.3
 *      Released:  10.11.2023
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2023 RRZE, University Erlangen-Nuremberg
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

#include <types.h>

extern void timer_init( void );
extern double timer_print( const TimerData* );
extern uint64_t timer_printCycles( const TimerData* );
extern uint64_t timer_getCpuClock( void );
extern uint64_t timer_getCpuClockCurrent( int cpu_id );
extern uint64_t timer_getCycleClock( void );
extern uint64_t timer_getBaseline( void );

extern void timer_start( TimerData* );
extern void timer_stop ( TimerData* );

#endif /* TIMER_H */
