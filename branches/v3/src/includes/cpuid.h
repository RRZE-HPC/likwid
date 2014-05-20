/*
 * =======================================================================================
 *
 *      Filename:  cpuid.h
 *
 *      Description:  Header File cpuid Module. 
 *                    Reads out cpuid information and initilaizes a global 
 *                    data structure cpuid_info.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2012 Jan Treibig 
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

#ifndef CPUID_H
#define CPUID_H

#include <types.h>

/* Intel P6 */
#define PENTIUM_M_BANIAS     0x09U
#define PENTIUM_M_DOTHAN     0x0DU
#define CORE_DUO             0x0EU
#define CORE2_65             0x0FU
#define CORE2_45             0x17U
#define ATOM                 0x1CU
#define NEHALEM              0x1AU
#define NEHALEM_BLOOMFIELD   0x1AU
#define NEHALEM_LYNNFIELD    0x1EU
#define NEHALEM_LYNNFIELD_M  0x1FU
#define NEHALEM_WESTMERE     0x2CU
#define NEHALEM_WESTMERE_M   0x25U
#define SANDYBRIDGE          0x2AU
#define SANDYBRIDGE_EP       0x2DU
#define IVYBRIDGE            0x3AU
#define NEHALEM_EX           0x2EU
#define WESTMERE_EX          0x2FU
#define XEON_MP              0x1DU

/* Intel MIC */
#define XEON_PHI           0x01U

/* AMD K10 */
#define BARCELONA      0x02U
#define SHANGHAI       0x04U
#define ISTANBUL       0x08U
#define MAGNYCOURS     0x09U

/* AMD K8 */
#define OPTERON_SC_1MB  0x05U
#define OPTERON_DC_E    0x21U
#define OPTERON_DC_F    0x41U
#define ATHLON64_X2     0x43U
#define ATHLON64_X2_F   0x4BU
#define ATHLON64_F1     0x4FU
#define ATHLON64_F2     0x5FU
#define ATHLON64_X2_G   0x6BU
#define ATHLON64_G1     0x6FU
#define ATHLON64_G2     0x7FU


#define  P6_FAMILY        0x6U
#define  MIC_FAMILY       0xBU
#define  NETBURST_FAMILY  0xFFU
#define  K15_FAMILY       0x15U
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
extern int  cpuid_isInCpuset(void);

#endif /*CPUID_H*/
