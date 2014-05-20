/*
 * ===========================================================================
 *
 *      Filename:  types.h
 *
 *      Description:  Global  Types file
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


#ifndef TYPES_H
#define TYPES_H


/* #####   HEADER FILE INCLUDES   ######################################### */
#include <stdint.h>

#include <strUtil_types.h>
#include <libperfctr_types.h>
#include <timer_types.h>
#include <tree_types.h>
#include <cpuid_types.h>
#include <cpuFeatures_types.h>
#include <asciiBoxes_types.h>
#include <asciiTable_types.h>
#include <perfmon_types.h>
#include <multiplex_types.h>


/* #####   EXPORTED MACROS   ############################################## */

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif

#define TRUE  1
#define FALSE 0

#define HLINE "-------------------------------------------------------------\n"
#define SLINE "*************************************************************\n"

#define LLU_CAST  (unsigned long long)

#endif /*TYPES_H*/
