/*
 * =======================================================================================
 *
 *      Filename:  test_types.h
 *
 *      Description:  Type definitions for benchmarking framework
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
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
    STREAM_13,
    STREAM_14,
    STREAM_15,
    STREAM_16,
    STREAM_17,
    STREAM_18,
    STREAM_19,
    STREAM_20,
    STREAM_21,
    STREAM_22,
    STREAM_23,
    STREAM_24,
    STREAM_25,
    STREAM_26,
    STREAM_27,
    STREAM_28,
    STREAM_29,
    STREAM_30,
    STREAM_31,
    STREAM_32,
    STREAM_33,
    STREAM_34,
    STREAM_35,
    STREAM_36,
    STREAM_37,
    STREAM_38,
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
    uint64_t   size;
    uint32_t   iter;
    const TestCase* test;
    uint64_t   cycles;
    uint32_t numberOfThreads;
    int* processors;
    void** streams;
} ThreadUserData;

#endif /*TEST_TYPES_H*/
