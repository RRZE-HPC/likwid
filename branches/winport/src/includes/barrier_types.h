/*
 * ===========================================================================
 *
 *      Filename:  barrier_types.h
 *
 *      Description:  Type Definitions for barrier Module
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
