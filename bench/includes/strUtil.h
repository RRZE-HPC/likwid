/*
 * =======================================================================================
 *      Filename:  strUtil.h
 *
 *      Description:  Some sting functions
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
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
#ifndef STRUTIL_H
#define STRUTIL_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <bstrlib.h>
#include <likwid.h>

#include <test_types.h>

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


extern void bstr_to_workgroup(Workgroup* group, const_bstring str, DataType type, int numberOfStreams);

#endif
