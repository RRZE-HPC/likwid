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
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
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
#ifndef TIMER_TYPES_H
#define TIMER_TYPES_H

#include <stdint.h>

typedef union {
    uint64_t int64;
    struct {
        uint32_t lo, hi;
    } int32;
} TscCounter;

typedef struct {
    TscCounter start;
    TscCounter stop;
} TimerData;

#endif /*TIMER_TYPES_H*/
