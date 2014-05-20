/*
 * =======================================================================================
 *
 *      Filename:  strUtil_types.h
 *
 *      Description:  Types file for strUtil module.
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

#ifndef STRUTIL_TYPES_H
#define STRUTIL_TYPES_H

#include  <bstrlib.h>


typedef struct {
    bstring eventName;
    bstring counterName;
} StrUtilEvent;

typedef struct {
    StrUtilEvent* events;
    int numberOfEvents;
} StrUtilEventSet;

typedef struct {
    bstring domain;
    int offset;
    void* ptr;
} Stream;

typedef struct {
    uint32_t numberOfThreads;
    int* processorIds;
    uint64_t size;
    Stream* streams;
} Workgroup;


#endif /*STRUTIL_TYPES_H*/
