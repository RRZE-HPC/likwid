/*
 * =======================================================================================
 *
 *      Filename:  threads_types.h
 *
 *      Description:  Types file for threads module.
 *
 *      Version:   4.0
 *      Released:  16.6.2015
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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

#ifndef THREADS_TYPES_H
#define THREADS_TYPES_H

#include <stdint.h>
#include <test_types.h>

typedef struct {
    int        globalNumberOfThreads;
    int        numberOfThreads;
    int        globalThreadId;
    int        threadId;
    int        numberOfGroups;
    int        groupId;
    double     time;
    uint64_t   cycles;
    ThreadUserData data;
} ThreadData;

typedef struct {
    int        numberOfThreads;
    int*       threadIds;
} ThreadGroup;

typedef void (*threads_copyDataFunc)(ThreadUserData* src,ThreadUserData* dst);

#endif /*THREADS_TYPES_H*/
