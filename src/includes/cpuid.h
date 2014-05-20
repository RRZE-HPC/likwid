/*
 * ===========================================================================
 *
 *      Filename:  cpuid.h
 *
 *      Description:  Header File cpuid Module. 
 *                    Reads out cpuid information and initilaizes a global 
 *                    data structure cpuid_info.
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

/** \file cpuid.h
 * Header of cpuid module
 *
 */
#ifndef CPUID_H
#define CPUID_H
#include <types.h>

/* Intel P6 */
#define PENTIUM_M_BANIAS 0x09U
#define PENTIUM_M_DOTHAN 0x0DU
#define CORE_DUO  0x0EU
#define CORE2_65  0x0FU
#define CORE2_45  0x17U
#define ATOM      0x1CU
#define NEHALEM   0x1AU
#define NEHALEM_BLOOMFIELD   0x1AU
#define NEHALEM_LYNNFIELD    0x1EU
#define NEHALEM_WESTMERE     0x2CU
#define NEHALEM_WESTMERE_M   0x25U
#define NEHALEM_EX   0x2EU
#define XEON_MP   0x1DU

/* AMD K10 */
#define BARCELONA 0x02U
#define SHANGHAI  0x04U
#define ISTANBUL  0x08U
#define MAGNYCOURS 0x09U

/* AMD K8 */
#define OPTERON_SC_1MB  0x05U
#define OPTERON_DC_E  0x21U
#define OPTERON_DC_F  0x41U
#define ATHLON64_X2   0x43U
#define ATHLON64_X2_F 0x4BU
#define ATHLON64_F1   0x4FU
#define ATHLON64_F2   0x5FU
#define ATHLON64_X2_G 0x6BU
#define ATHLON64_G1   0x6FU
#define ATHLON64_G2   0x7FU


#define  P6_FAMILY        0x6U
#define  NETBURST_FAMILY  0xFFU
#define  K10_FAMILY       0x10U
#define  K8_FAMILY        0xFU

/** Structure holding cpuid information
 *
 */
extern CpuInfo cpuid_info;
extern CpuTopology cpuid_topology;

/** Init routine to intialize global structure.
 *
 *  Determines: 
 *  - cpu family
 *  - cpu model
 *  - cpu stepping
 *  - cpu clock
 *  - Instruction Set Extension Flags
 *  - Performance counter features (Intel P6 only)
 *
 */
extern void cpuid_init (void);
extern void cpuid_initTopology (void);
extern void cpuid_initCacheTopology (void);

#endif /*CPUID_H*/
