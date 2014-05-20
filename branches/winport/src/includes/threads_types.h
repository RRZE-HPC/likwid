/*
 * ===========================================================================
 *
 *      Filename:  threads_types.h
 *
 *      Description:  Types file for threads module.
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

#ifndef THREADS_TYPES_H
#define THREADS_TYPES_H

#include <stdint.h>

typedef struct {
    int        globalNumberOfThreads;
    int        numberOfThreads;
    int        globalThreadId;
    int        threadId;
    int        numberOfGroups;
    int        groupId;
    double      time;
    uint64_t   cycles;
    ThreadUserData data;
} ThreadData;

typedef struct {
    int        numberOfThreads;
    int*       threadIds;
} ThreadGroup;

typedef void (*threads_copyDataFunc)(ThreadUserData* src,ThreadUserData* dst);

#endif /*THREADS_TYPES_H*/
