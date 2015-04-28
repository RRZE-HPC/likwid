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
#ifndef TIMER_H
#define TIMER_H

#include <types.h>

#define RDTSC(cpu_c) \
__asm__ volatile("xor %%eax,%%eax\n\t"           \
"cpuid\n\t"           \
"rdtsc\n\t"           \
"movl %%eax, %0\n\t"  \
"movl %%edx, %1\n\t"  \
: "=r" ((cpu_c).int32.lo), "=r" ((cpu_c).int32.hi) \
: : "%eax","%ebx","%ecx","%edx")

#define RDTSC_CR(cpu_c) \
__asm__ volatile(   \
"rdtsc\n\t"           \
"movl %%eax, %0\n\t"  \
"movl %%edx, %1\n\t"  \
: "=r" ((cpu_c).int32.lo), "=r" ((cpu_c).int32.hi) \
: : "%eax","%ebx","%ecx","%edx")

#define RDTSCP(cpu_c) \
__asm__ volatile(     \
"rdtscp\n\t"          \
"movl %%eax, %0\n\t"  \
"movl %%edx, %1\n\t"  \
"cpuid\n\t"           \
: "=r" ((cpu_c).int32.lo), "=r" ((cpu_c).int32.hi) \
: : "%eax","%ebx","%ecx","%edx")

#ifdef HAS_RDTSCP
#define RDTSC_STOP(cpu_c) RDTSCP(cpu_c);
#else
#define RDTSC_STOP(cpu_c) RDTSC_CR(cpu_c);
#endif


extern void timer_init( void );
extern double timer_print( TimerData* );
extern uint64_t timer_printCycles( TimerData* );
extern uint64_t timer_getCpuClock( void );
extern uint64_t timer_getBaseline( void );

extern void timer_start( TimerData* );
extern void timer_stop ( TimerData* );





#endif /* TIMER_H */
