/*
 * ===========================================================================
 *
 *      Filename:  registers.h
 *
 *      Description:  Register Defines for the perfmon module
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


#ifndef REGISTERS_H
#define REGISTERS_H

/*
 * INTEL
 */

/* MSR Registers  */
/* 3 80 bit fixed counters */
#define MSR_PERF_FIXED_CTR_CTRL   0x38D
#define MSR_PERF_FIXED_CTR0       0x309  /* Instr_Retired.Any */
#define MSR_PERF_FIXED_CTR1       0x30A  /* CPU_CLK_UNHALTED.CORE */
#define MSR_PERF_FIXED_CTR2       0x30B  /* CPU_CLK_UNHALTED.REF */
/* 4 40/48 bit configurable counters */
/* Perfmon V1 */
#define MSR_PERFEVTSEL0           0x186
#define MSR_PERFEVTSEL1           0x187
#define MSR_PERFEVTSEL2           0x188
#define MSR_PERFEVTSEL3           0x189
#define MSR_PMC0                  0x0C1
#define MSR_PMC1                  0x0C2
#define MSR_PMC2                  0x0C3
#define MSR_PMC3                  0x0C4
/* Perfmon V2 */
#define MSR_PERF_GLOBAL_CTRL      0x38F
#define MSR_PERF_GLOBAL_STATUS    0x38E
#define MSR_PERF_GLOBAL_OVF_CTRL  0x390
#define MSR_PEBS_ENABLE           0x3F1
/* Perfmon V3 */
#define MSR_OFFCORE_RSP0              0x1A6
#define MSR_UNCORE_PERF_GLOBAL_CTRL       0x391
#define MSR_UNCORE_PERF_GLOBAL_STATUS     0x392
#define MSR_UNCORE_PERF_GLOBAL_OVF_CTRL   0x393
#define MSR_UNCORE_FIXED_CTR0             0x394  /* Uncore clock cycles */
#define MSR_UNCORE_FIXED_CTR_CTRL         0x395
#define MSR_UNCORE_ADDR_OPCODE_MATCH      0x396 
#define MSR_UNCORE_PERFEVTSEL0         0x3C0
#define MSR_UNCORE_PERFEVTSEL1         0x3C1
#define MSR_UNCORE_PERFEVTSEL2         0x3C2
#define MSR_UNCORE_PERFEVTSEL3         0x3C3
#define MSR_UNCORE_PERFEVTSEL4         0x3C4
#define MSR_UNCORE_PERFEVTSEL5         0x3C5
#define MSR_UNCORE_PERFEVTSEL6         0x3C6
#define MSR_UNCORE_PERFEVTSEL7         0x3C7
#define MSR_UNCORE_PMC0                0x3B0
#define MSR_UNCORE_PMC1                0x3B1
#define MSR_UNCORE_PMC2                0x3B2
#define MSR_UNCORE_PMC3                0x3B3
#define MSR_UNCORE_PMC4                0x3B4
#define MSR_UNCORE_PMC5                0x3B5
#define MSR_UNCORE_PMC6                0x3B6
#define MSR_UNCORE_PMC7                0x3B7

/*
 * AMD
 */

#define MSR_AMD_PERFEVTSEL0           0xC0010000
#define MSR_AMD_PERFEVTSEL1           0xC0010001
#define MSR_AMD_PERFEVTSEL2           0xC0010002
#define MSR_AMD_PERFEVTSEL3           0xC0010003
#define MSR_AMD_PMC0                  0xC0010004
#define MSR_AMD_PMC1                  0xC0010005
#define MSR_AMD_PMC2                  0xC0010006
#define MSR_AMD_PMC3                  0xC0010007


#endif /* REGISTERS_H */

