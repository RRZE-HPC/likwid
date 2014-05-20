/*
 * ===========================================================================
 *
 *      Filename:  test_types.h
 *
 *      Description:  Type definitions for benchmarking framework
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

#ifndef TEST_TYPES_H
#define TEST_TYPES_H

#include <stdint.h>
#include <bstrlib.h>

typedef void (*FuncPrototype)();

typedef enum {
    SINGLE = 0,
    DOUBLE} DataType;

typedef enum {
    STREAM_1 = 1,
    STREAM_2,
    STREAM_3,
    STREAM_4,
    STREAM_5,
    STREAM_6,
    STREAM_7,
    STREAM_8,
    STREAM_9,
    STREAM_10,
    STREAM_11,
    STREAM_12,
    MAX_STREAMS} Pattern;

typedef struct {
    char* name;
    Pattern streams;
    DataType type ;
    int stride;
    FuncPrototype kernel;
    int  flops;
    int  bytes;
} TestCase;

typedef struct {
    uint32_t   size;
    uint32_t   iter;
    const TestCase* test;
    uint64_t   cycles;
    uint32_t numberOfThreads;
    int* processors;
    void** streams;
} ThreadUserData;

#endif /*TEST_TYPES_H*/
