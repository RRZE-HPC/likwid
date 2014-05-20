/*
 * =======================================================================================
 *
 *      Filename:  timer_types.h
 *
 *      Description:  Types file for timer module.
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

#ifndef TIMER_TYPES_H
#define TIMER_TYPES_H

#include <stdint.h>
#include <sys/time.h>


/**
 * @brief Union for cycles measurements
 * 
 * Union with either one 64 bit unsigned integer or a struct
 * of two 32 bit unsigned integers used as register eax and ebx
 * in call to rdtsc.
 */
typedef union
{  
    uint64_t int64;                   /** 64 bit unsigned integer fo cycles */
    struct {uint32_t lo, hi;} int32;  /** two 32 bit unsigned integers used
                                        for register values */
} TscCounter;

/**
 * @brief Struct holding the start and stop timestamp
 * 
 * A reference to this struct is passed to the timer functions and hold the
 * timestamps.
 */
typedef struct {
    struct timeval before;
    struct timeval after;
} TimerData;

/**
 * @brief Struct holding the start, stop and base timestamps
 * 
 * A reference to this struct is passed to the cycles functions and holds
 * the start and stop timestamp. Additionally base holds a possible overhead
 * for empty measurement or cpuid serialization.
 */

typedef struct {
    TscCounter start;
    TscCounter stop;
    uint64_t base;
} CyclesData;


#endif /*TIMER_TYPES_H*/
