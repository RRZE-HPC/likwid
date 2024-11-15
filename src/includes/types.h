/*
 * =======================================================================================
 *
 *      Filename:  types.h
 *
 *      Description:  Global  Types file
 *
 *      Version:   5.4.0
 *      Released:  15.11.2024
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2024 RRZE, University Erlangen-Nuremberg
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
#ifndef TYPES_H
#define TYPES_H

/* #####   HEADER FILE INCLUDES   ######################################### */
#include <stdint.h>
#include <bstrlib.h>

#include <access_client_types.h>
#include <registers_types.h>
#include <pci_types.h>
#include <power_types.h>
#include <thermal_types.h>
#include <tree_types.h>
#include <topology_types.h>
#include <perfmon_types.h>
#include <libperfctr_types.h>

typedef struct {
    uint64_t mask[2];
} BitMask;

/* #####   EXPORTED MACROS   ############################################## */

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif

#ifndef GLIB_MAJOR_VERSION
#define TRUE  1
#define FALSE 0
#endif

#define HLINE "-------------------------------------------------------------\n"
#define SLINE "*************************************************************\n"

#define LLU_CAST  (unsigned long long)

// borrowed from https://www.scs.stanford.edu/~dm/blog/va-opt.html
#define PARENS ()
#define EXPAND(...) EXPAND3(EXPAND3(EXPAND3(__VA_ARGS__)))
#define EXPAND3(...) EXPAND2(EXPAND2(EXPAND2(__VA_ARGS__)))
#define EXPAND2(...) EXPAND1(EXPAND1(EXPAND1(__VA_ARGS__)))
#define EXPAND1(...) __VA_ARGS__
#define FOR_EACH(macro, ...)                                                  \
      __VA_OPT__(EXPAND(FOR_EACH_HELPER(macro, __VA_ARGS__)))
#define FOR_EACH_HELPER(macro, a1, ...)                                       \
      macro(a1) __VA_OPT__(FOR_EACH_AGAIN PARENS (macro, __VA_ARGS__))
#define FOR_EACH_AGAIN() FOR_EACH_HELPER
#define STRINGIFY(x) #x
#define TOSTRING(...) FOR_EACH(STRINGIFY, __VA_ARGS__)

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

#define ARRAY_COUNT(arr) (sizeof(arr) / sizeof((arr)[0]))

#endif /*TYPES_H*/
