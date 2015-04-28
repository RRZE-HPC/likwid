/*
 * =======================================================================================
 *
 *      Filename:  types.h
 *
 *      Description:  Global  Types file
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

#ifndef TYPES_H
#define TYPES_H


/* #####   HEADER FILE INCLUDES   ######################################### */
#include <stdint.h>
#include <bstrlib.h>

#include <accessClient_types.h>
#include <registers_types.h>
#include <pci_types.h>
#include <power_types.h>
#include <thermal_types.h>
//#include <strUtil_types.h>
#include <test_types.h>
//#include <timer_types.h>
#include <tree_types.h>
#include <topology_types.h>
#include <perfmon_types.h>
#include <libperfctr_types.h>
#include <barrier_types.h>
#include <cpuFeatures_types.h>
#include <multiplex_types.h>


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

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

#endif /*TYPES_H*/
