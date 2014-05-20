/*
 * ===========================================================================
 *
 *      Filename:  timertypes.h
 *
 *      Description:  Types file for timer module.
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
 * ===========================================================================
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
