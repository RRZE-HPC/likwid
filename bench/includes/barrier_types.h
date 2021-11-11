/*
 * =======================================================================================
 *
 *      Filename:  barrier_types.h
 *
 *      Description:  Type Definitions for barrier Module
 *
 *      Version:   5.2.1
 *      Released:  11.11.2021
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 NHR@FAU, University Erlangen-Nuremberg
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
#ifndef BARRIER_TYPES_H
#define BARRIER_TYPES_H

#include <stdint.h>

typedef struct {
    int        numberOfThreads;
    int        offset;
    int        val;
    int*       index;
    volatile int*  bval;
} BarrierData;

typedef struct {
    int*       groupBval;
    int        numberOfThreads;
} BarrierGroup;

#endif /*BARRIER_TYPES_H*/
