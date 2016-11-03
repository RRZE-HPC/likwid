/*
 * =======================================================================================
 *
 *      Filename:  registers.h
 *
 *      Description:  Register Defines for the perfmon module
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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
#define MSR_PERFEVTSEL4           0x18A
#define MSR_PERFEVTSEL5           0x18B
#define MSR_PERFEVTSEL6           0x18C
#define MSR_PERFEVTSEL7           0x18D
#define MSR_PMC0                  0x0C1
#define MSR_PMC1                  0x0C2
#define MSR_PMC2                  0x0C3
#define MSR_PMC3                  0x0C4
#define MSR_PMC4                  0x0C5
#define MSR_PMC5                  0x0C6
#define MSR_PMC6                  0x0C7
#define MSR_PMC7                  0x0C8
/* Perfmon V2 */
#define MSR_PERF_GLOBAL_CTRL      0x38F
#define MSR_PERF_GLOBAL_STATUS    0x38E
#define MSR_PERF_GLOBAL_OVF_CTRL  0x390
#define MSR_PEBS_ENABLE           0x3F1
#define MSR_PEBS_LD_LAT           0x3F6
/* Perfmon V3 */
#define MSR_OFFCORE_RESP0              0x1A6
#define MSR_OFFCORE_RESP1              0x1A7
#define MSR_UNCORE_PERF_GLOBAL_CTRL       0x391
#define MSR_UNCORE_PERF_GLOBAL_STATUS     0x392
#define MSR_UNCORE_PERF_GLOBAL_OVF_CTRL   0x393
#define MSR_UNCORE_FIXED_CTR0             0x394  /* Uncore clock cycles */
#define MSR_UNCORE_FIXED_CTR_CTRL         0x395 /*FIXME: Is this correct? */
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
 * Perfmon V3 (starting with Haswell, according to
 * Intel software developers guide also for SandyBridge,
 * IvyBridge not mentioned in this section)
 */
#define MSR_UNC_PERF_GLOBAL_CTRL       0x391
#define MSR_UNC_PERF_GLOBAL_STATUS     0x392
#define MSR_UNC_PERF_GLOBAL_OVF_CTRL   0x393
#define MSR_UNC_PERF_FIXED_CTRL        0x394
#define MSR_UNC_PERF_FIXED_CTR         0x395
#define MSR_UNC_ARB_PERFEVTSEL0        0x3B2
#define MSR_UNC_ARB_PERFEVTSEL1        0x3B3
#define MSR_UNC_ARB_CTR0               0x3B0
#define MSR_UNC_ARB_CTR1               0x3B1
#define MSR_UNC_CBO_CONFIG             0x396
#define MSR_UNC_CBO_0_PERFEVTSEL0      0x700
#define MSR_UNC_CBO_0_PERFEVTSEL1      0x701
#define MSR_UNC_CBO_0_CTR0             0x706
#define MSR_UNC_CBO_0_CTR1             0x707
#define MSR_UNC_CBO_1_PERFEVTSEL0      0x710
#define MSR_UNC_CBO_1_PERFEVTSEL1      0x711
#define MSR_UNC_CBO_1_CTR0             0x716
#define MSR_UNC_CBO_1_CTR1             0x717
#define MSR_UNC_CBO_2_PERFEVTSEL0      0x720
#define MSR_UNC_CBO_2_PERFEVTSEL1      0x721
#define MSR_UNC_CBO_2_CTR0             0x726
#define MSR_UNC_CBO_2_CTR1             0x727
#define MSR_UNC_CBO_3_PERFEVTSEL0      0x730
#define MSR_UNC_CBO_3_PERFEVTSEL1      0x731
#define MSR_UNC_CBO_3_CTR0             0x736
#define MSR_UNC_CBO_3_CTR1             0x737
/* Perfmon V4 starting with Skylake */
#define MSR_V4_PERF_GLOBAL_STATUS       0x38E
#define MSR_V4_PERF_GLOBAL_STATUS_SET   0x391
#define MSR_V4_PERF_GLOBAL_STATUS_RESET 0x390
#define MSR_V4_PERF_GLOBAL_INUSE        0x392
#define MSR_V4_PEBS_FRONTEND            0x3F7
#define MSR_V4_UNC_PERF_GLOBAL_CTRL     0xE01
#define MSR_V4_UNC_PERF_GLOBAL_STATUS   0xE02
#define MSR_V4_UNC_PERF_FIXED_CTRL      0x394
#define MSR_V4_UNC_PERF_FIXED_CTR       0x395
#define MSR_V4_ARB_PERF_FIXED_CTRL0      0x3B2
#define MSR_V4_ARB_PERF_FIXED_CTR0       0x3B0
#define MSR_V4_ARB_PERF_FIXED_CTRL1      0x3B3
#define MSR_V4_ARB_PERF_FIXED_CTR1       0x3B1
#define MSR_V4_C0_PERF_FIXED_CTRL0      0x700
#define MSR_V4_C0_PERF_FIXED_CTR0       0x706
#define MSR_V4_C0_PERF_FIXED_CTRL1      0x701
#define MSR_V4_C0_PERF_FIXED_CTR1       0x707
#define MSR_V4_C1_PERF_FIXED_CTRL0      0x710
#define MSR_V4_C1_PERF_FIXED_CTR0       0x716
#define MSR_V4_C1_PERF_FIXED_CTRL1      0x711
#define MSR_V4_C1_PERF_FIXED_CTR1       0x717
#define MSR_V4_C2_PERF_FIXED_CTRL0      0x720
#define MSR_V4_C2_PERF_FIXED_CTR0       0x726
#define MSR_V4_C2_PERF_FIXED_CTRL1      0x721
#define MSR_V4_C2_PERF_FIXED_CTR1       0x727
#define MSR_V4_C3_PERF_FIXED_CTRL0      0x730
#define MSR_V4_C3_PERF_FIXED_CTR0       0x736
#define MSR_V4_C3_PERF_FIXED_CTRL1      0x731
#define MSR_V4_C3_PERF_FIXED_CTR1       0x737
/* V4 Uncore registers the same as in V3 */

/* Xeon Phi */
#define MSR_MIC_TSC                   0x010
#define MSR_MIC_PERFEVTSEL0           0x028
#define MSR_MIC_PERFEVTSEL1           0x029
#define MSR_MIC_PMC0                  0x020
#define MSR_MIC_PMC1                  0x021
#define MSR_MIC_SPFLT_CONTROL         0x02C
#define MSR_MIC_PERF_GLOBAL_STATUS    0x02D
#define MSR_MIC_PERF_GLOBAL_OVF_CTRL  0x02E
#define MSR_MIC_PERF_GLOBAL_CTRL      0x02F
/* Xeon Phi (Knights Landing)*/
#define MSR_MIC2_PMC0                 0xC1
#define MSR_MIC2_PMC1                 0xC2
#define MSR_MIC2_PERFEVTSEL0          0x186
#define MSR_MIC2_PERFEVTSEL1          0x187
#define MSR_MIC2_TURBO_RATIO_LIMIT    0x1AD
#define MSR_MIC2_SPFLT_CONTROL	      0x02C
#define MSR_MIC2_PERF_GLOBAL_STATUS    0x02D
#define MSR_MIC2_PERF_GLOBAL_OVF_CTRL  0x02E
#define MSR_MIC2_PERF_GLOBAL_CTRL      0x02F
/* Xeon Phi (Knights Landing) UBOX*/
#define MSR_MIC2_U_GLOBAL_CTRL      0x700
#define MSR_MIC2_U_GLOBAL_STATUS    0x701
#define MSR_MIC2_U_CONFIG           0x702
#define MSR_MIC2_U_FIXED_CTRL	      0x703
#define MSR_MIC2_U_FIXED_CTR	      0x704
#define MSR_MIC2_U_CTRL0	      0x705
#define MSR_MIC2_U_CTRL1	      0x706
#define MSR_MIC2_U_OVFL	      0x708
#define MSR_MIC2_U_CTR0	      0x709
#define MSR_MIC2_U_CTR1	      0x70A
/* Xeon Phi (Knights Landing) WBOX*/
#define MSR_MIC2_PCU_GLOBAL_CTRL    0x710
#define MSR_MIC2_PCU_CTRL0   0x711
#define MSR_MIC2_PCU_CTRL1   0x712
#define MSR_MIC2_PCU_CTRL2   0x713
#define MSR_MIC2_PCU_CTRL3   0x714
#define MSR_MIC2_PCU_CTR0    0x717
#define MSR_MIC2_PCU_CTR1    0x718
#define MSR_MIC2_PCU_CTR2    0x719
#define MSR_MIC2_PCU_CTR3    0x71A
/* Xeon Phi (Knights Landing) Cache boxes*/
#define MSR_MIC2_C0_GLOBAL_CTRL       0xE00
#define MSR_MIC2_C0_CTRL0             0xE01
#define MSR_MIC2_C0_CTRL1             0xE02
#define MSR_MIC2_C0_CTRL2             0xE03
#define MSR_MIC2_C0_CTRL3             0xE04
#define MSR_MIC2_C0_CTR0             0xE08
#define MSR_MIC2_C0_CTR1             0xE09
#define MSR_MIC2_C0_CTR2             0xE0A
#define MSR_MIC2_C0_CTR3             0xE0B
#define MSR_MIC2_C0_FILTER0          0xE05
#define MSR_MIC2_C0_FILTER1          0xE06
#define MSR_MIC2_C0_STATUS           0xE07
#define MSR_MIC2_C1_GLOBAL_CTRL       0xE0C
#define MSR_MIC2_C1_CTRL0             0xE0D
#define MSR_MIC2_C1_CTRL1             0xE0E
#define MSR_MIC2_C1_CTRL2             0xE0F
#define MSR_MIC2_C1_CTRL3             0xE10
#define MSR_MIC2_C1_CTR0             0xE14
#define MSR_MIC2_C1_CTR1             0xE15
#define MSR_MIC2_C1_CTR2             0xE16
#define MSR_MIC2_C1_CTR3             0xE17
#define MSR_MIC2_C1_FILTER0          0xE11
#define MSR_MIC2_C1_FILTER1          0xE12
#define MSR_MIC2_C1_STATUS           0xE13
#define MSR_MIC2_C2_GLOBAL_CTRL       0xE18
#define MSR_MIC2_C2_CTRL0             0xE19
#define MSR_MIC2_C2_CTRL1             0xE1A
#define MSR_MIC2_C2_CTRL2             0xE1B
#define MSR_MIC2_C2_CTRL3             0xE1C
#define MSR_MIC2_C2_CTR0             0xE20
#define MSR_MIC2_C2_CTR1             0xE21
#define MSR_MIC2_C2_CTR2             0xE22
#define MSR_MIC2_C2_CTR3             0xE23
#define MSR_MIC2_C2_FILTER0          0xE1D
#define MSR_MIC2_C2_FILTER1          0xE1E
#define MSR_MIC2_C2_STATUS           0xE1F
#define MSR_MIC2_C3_GLOBAL_CTRL       0xE24
#define MSR_MIC2_C3_CTRL0             0xE25
#define MSR_MIC2_C3_CTRL1             0xE26
#define MSR_MIC2_C3_CTRL2             0xE27
#define MSR_MIC2_C3_CTRL3             0xE28
#define MSR_MIC2_C3_CTR0             0xE2C
#define MSR_MIC2_C3_CTR1             0xE2D
#define MSR_MIC2_C3_CTR2             0xE2E
#define MSR_MIC2_C3_CTR3             0xE2F
#define MSR_MIC2_C3_FILTER0          0xE29
#define MSR_MIC2_C3_FILTER1          0xE2A
#define MSR_MIC2_C3_STATUS           0xE2B
#define MSR_MIC2_C4_GLOBAL_CTRL       0xE30
#define MSR_MIC2_C4_CTRL0             0xE31
#define MSR_MIC2_C4_CTRL1             0xE32
#define MSR_MIC2_C4_CTRL2             0xE33
#define MSR_MIC2_C4_CTRL3             0xE34
#define MSR_MIC2_C4_CTR0             0xE38
#define MSR_MIC2_C4_CTR1             0xE39
#define MSR_MIC2_C4_CTR2             0xE3A
#define MSR_MIC2_C4_CTR3             0xE3B
#define MSR_MIC2_C4_FILTER0          0xE35
#define MSR_MIC2_C4_FILTER1          0xE36
#define MSR_MIC2_C4_STATUS           0xE37
#define MSR_MIC2_C5_GLOBAL_CTRL       0xE3C
#define MSR_MIC2_C5_CTRL0             0xE3D
#define MSR_MIC2_C5_CTRL1             0xE3E
#define MSR_MIC2_C5_CTRL2             0xE3F
#define MSR_MIC2_C5_CTRL3             0xE40
#define MSR_MIC2_C5_CTR0             0xE44
#define MSR_MIC2_C5_CTR1             0xE45
#define MSR_MIC2_C5_CTR2             0xE46
#define MSR_MIC2_C5_CTR3             0xE47
#define MSR_MIC2_C5_FILTER0          0xE41
#define MSR_MIC2_C5_FILTER1          0xE42
#define MSR_MIC2_C5_STATUS           0xE43
#define MSR_MIC2_C6_GLOBAL_CTRL       0xE48
#define MSR_MIC2_C6_CTRL0             0xE49
#define MSR_MIC2_C6_CTRL1             0xE4A
#define MSR_MIC2_C6_CTRL2             0xE4B
#define MSR_MIC2_C6_CTRL3             0xE4C
#define MSR_MIC2_C6_CTR0             0xE50
#define MSR_MIC2_C6_CTR1             0xE51
#define MSR_MIC2_C6_CTR2             0xE52
#define MSR_MIC2_C6_CTR3             0xE53
#define MSR_MIC2_C6_FILTER0          0xE4D
#define MSR_MIC2_C6_FILTER1          0xE4E
#define MSR_MIC2_C6_STATUS           0xE4F
#define MSR_MIC2_C7_GLOBAL_CTRL       0xE54
#define MSR_MIC2_C7_CTRL0             0xE55
#define MSR_MIC2_C7_CTRL1             0xE56
#define MSR_MIC2_C7_CTRL2             0xE57
#define MSR_MIC2_C7_CTRL3             0xE58
#define MSR_MIC2_C7_CTR0             0xE5C
#define MSR_MIC2_C7_CTR1             0xE5D
#define MSR_MIC2_C7_CTR2             0xE5E
#define MSR_MIC2_C7_CTR3             0xE5F
#define MSR_MIC2_C7_FILTER0          0xE59
#define MSR_MIC2_C7_FILTER1          0xE5A
#define MSR_MIC2_C7_STATUS           0xE5B
#define MSR_MIC2_C8_GLOBAL_CTRL       0xE60
#define MSR_MIC2_C8_CTRL0             0xE61
#define MSR_MIC2_C8_CTRL1             0xE62
#define MSR_MIC2_C8_CTRL2             0xE63
#define MSR_MIC2_C8_CTRL3             0xE64
#define MSR_MIC2_C8_CTR0             0xE68
#define MSR_MIC2_C8_CTR1             0xE69
#define MSR_MIC2_C8_CTR2             0xE6A
#define MSR_MIC2_C8_CTR3             0xE6B
#define MSR_MIC2_C8_FILTER0          0xE65
#define MSR_MIC2_C8_FILTER1          0xE66
#define MSR_MIC2_C8_STATUS           0xE67
#define MSR_MIC2_C9_GLOBAL_CTRL       0xE6C
#define MSR_MIC2_C9_CTRL0             0xE6D
#define MSR_MIC2_C9_CTRL1             0xE6E
#define MSR_MIC2_C9_CTRL2             0xE6F
#define MSR_MIC2_C9_CTRL3             0xE70
#define MSR_MIC2_C9_CTR0             0xE74
#define MSR_MIC2_C9_CTR1             0xE75
#define MSR_MIC2_C9_CTR2             0xE76
#define MSR_MIC2_C9_CTR3             0xE77
#define MSR_MIC2_C9_FILTER0          0xE71
#define MSR_MIC2_C9_FILTER1          0xE72
#define MSR_MIC2_C9_STATUS           0xE73
#define MSR_MIC2_C10_GLOBAL_CTRL       0xE78
#define MSR_MIC2_C10_CTRL0             0xE79
#define MSR_MIC2_C10_CTRL1             0xE7A
#define MSR_MIC2_C10_CTRL2             0xE7B
#define MSR_MIC2_C10_CTRL3             0xE7C
#define MSR_MIC2_C10_CTR0             0xE80
#define MSR_MIC2_C10_CTR1             0xE81
#define MSR_MIC2_C10_CTR2             0xE82
#define MSR_MIC2_C10_CTR3             0xE83
#define MSR_MIC2_C10_FILTER0          0xE7D
#define MSR_MIC2_C10_FILTER1          0xE7E
#define MSR_MIC2_C10_STATUS           0xE7F
#define MSR_MIC2_C11_GLOBAL_CTRL       0xE84
#define MSR_MIC2_C11_CTRL0             0xE85
#define MSR_MIC2_C11_CTRL1             0xE86
#define MSR_MIC2_C11_CTRL2             0xE87
#define MSR_MIC2_C11_CTRL3             0xE88
#define MSR_MIC2_C11_CTR0             0xE8C
#define MSR_MIC2_C11_CTR1             0xE8D
#define MSR_MIC2_C11_CTR2             0xE8E
#define MSR_MIC2_C11_CTR3             0xE8F
#define MSR_MIC2_C11_FILTER0          0xE89
#define MSR_MIC2_C11_FILTER1          0xE8A
#define MSR_MIC2_C11_STATUS           0xE8B
#define MSR_MIC2_C12_GLOBAL_CTRL       0xE90
#define MSR_MIC2_C12_CTRL0             0xE91
#define MSR_MIC2_C12_CTRL1             0xE92
#define MSR_MIC2_C12_CTRL2             0xE93
#define MSR_MIC2_C12_CTRL3             0xE94
#define MSR_MIC2_C12_CTR0             0xE98
#define MSR_MIC2_C12_CTR1             0xE99
#define MSR_MIC2_C12_CTR2             0xE9A
#define MSR_MIC2_C12_CTR3             0xE9B
#define MSR_MIC2_C12_FILTER0          0xE95
#define MSR_MIC2_C12_FILTER1          0xE96
#define MSR_MIC2_C12_STATUS           0xE97
#define MSR_MIC2_C13_GLOBAL_CTRL       0xE9C
#define MSR_MIC2_C13_CTRL0             0xE9D
#define MSR_MIC2_C13_CTRL1             0xE9E
#define MSR_MIC2_C13_CTRL2             0xE9F
#define MSR_MIC2_C13_CTRL3             0xEA0
#define MSR_MIC2_C13_CTR0             0xEA4
#define MSR_MIC2_C13_CTR1             0xEA5
#define MSR_MIC2_C13_CTR2             0xEA6
#define MSR_MIC2_C13_CTR3             0xEA7
#define MSR_MIC2_C13_FILTER0          0xEA1
#define MSR_MIC2_C13_FILTER1          0xEA2
#define MSR_MIC2_C13_STATUS           0xEA3
#define MSR_MIC2_C14_GLOBAL_CTRL       0xEA8
#define MSR_MIC2_C14_CTRL0             0xEA9
#define MSR_MIC2_C14_CTRL1             0xEAA
#define MSR_MIC2_C14_CTRL2             0xEAB
#define MSR_MIC2_C14_CTRL3             0xEAC
#define MSR_MIC2_C14_CTR0             0xEB0
#define MSR_MIC2_C14_CTR1             0xEB1
#define MSR_MIC2_C14_CTR2             0xEB2
#define MSR_MIC2_C14_CTR3             0xEB3
#define MSR_MIC2_C14_FILTER0          0xEAD
#define MSR_MIC2_C14_FILTER1          0xEAE
#define MSR_MIC2_C14_STATUS           0xEAF
#define MSR_MIC2_C15_GLOBAL_CTRL       0xEB4
#define MSR_MIC2_C15_CTRL0             0xEB5
#define MSR_MIC2_C15_CTRL1             0xEB6
#define MSR_MIC2_C15_CTRL2             0xEB7
#define MSR_MIC2_C15_CTRL3             0xEB8
#define MSR_MIC2_C15_CTR0             0xEBC
#define MSR_MIC2_C15_CTR1             0xEBD
#define MSR_MIC2_C15_CTR2             0xEBE
#define MSR_MIC2_C15_CTR3             0xEBF
#define MSR_MIC2_C15_FILTER0          0xEB9
#define MSR_MIC2_C15_FILTER1          0xEBA
#define MSR_MIC2_C15_STATUS           0xEBB
#define MSR_MIC2_C16_GLOBAL_CTRL       0xEC0
#define MSR_MIC2_C16_CTRL0             0xEC1
#define MSR_MIC2_C16_CTRL1             0xEC2
#define MSR_MIC2_C16_CTRL2             0xEC3
#define MSR_MIC2_C16_CTRL3             0xEC4
#define MSR_MIC2_C16_CTR0             0xEC8
#define MSR_MIC2_C16_CTR1             0xEC9
#define MSR_MIC2_C16_CTR2             0xECA
#define MSR_MIC2_C16_CTR3             0xECB
#define MSR_MIC2_C16_FILTER0          0xEC5
#define MSR_MIC2_C16_FILTER1          0xEC6
#define MSR_MIC2_C16_STATUS           0xEC7
#define MSR_MIC2_C17_GLOBAL_CTRL       0xECC
#define MSR_MIC2_C17_CTRL0             0xECD
#define MSR_MIC2_C17_CTRL1             0xECE
#define MSR_MIC2_C17_CTRL2             0xECF
#define MSR_MIC2_C17_CTRL3             0xED0
#define MSR_MIC2_C17_CTR0             0xED4
#define MSR_MIC2_C17_CTR1             0xED5
#define MSR_MIC2_C17_CTR2             0xED6
#define MSR_MIC2_C17_CTR3             0xED7
#define MSR_MIC2_C17_FILTER0          0xED1
#define MSR_MIC2_C17_FILTER1          0xED2
#define MSR_MIC2_C17_STATUS           0xED3
#define MSR_MIC2_C18_GLOBAL_CTRL       0xED8
#define MSR_MIC2_C18_CTRL0             0xED9
#define MSR_MIC2_C18_CTRL1             0xEDA
#define MSR_MIC2_C18_CTRL2             0xEDB
#define MSR_MIC2_C18_CTRL3             0xED0
#define MSR_MIC2_C18_CTR0             0xEDF
#define MSR_MIC2_C18_CTR1             0xEE0
#define MSR_MIC2_C18_CTR2             0xEE1
#define MSR_MIC2_C18_CTR3             0xEE2
#define MSR_MIC2_C18_FILTER0          0xEE3
#define MSR_MIC2_C18_FILTER1          0xEDD
#define MSR_MIC2_C18_STATUS           0xEDE
#define MSR_MIC2_C19_GLOBAL_CTRL       0xEE4
#define MSR_MIC2_C19_CTRL0             0xEE5
#define MSR_MIC2_C19_CTRL1             0xEE6
#define MSR_MIC2_C19_CTRL2             0xEE7
#define MSR_MIC2_C19_CTRL3             0xEE8
#define MSR_MIC2_C19_CTR0             0xEEC
#define MSR_MIC2_C19_CTR1             0xEED
#define MSR_MIC2_C19_CTR2             0xEEE
#define MSR_MIC2_C19_CTR3             0xEEF
#define MSR_MIC2_C19_FILTER0          0xEE9
#define MSR_MIC2_C19_FILTER1          0xEEA
#define MSR_MIC2_C19_STATUS           0xEEB
#define MSR_MIC2_C20_GLOBAL_CTRL       0xEF0
#define MSR_MIC2_C20_CTRL0             0xEF1
#define MSR_MIC2_C20_CTRL1             0xEF2
#define MSR_MIC2_C20_CTRL2             0xEF3
#define MSR_MIC2_C20_CTRL3             0xEF4
#define MSR_MIC2_C20_CTR0             0xEF8
#define MSR_MIC2_C20_CTR1             0xEF9
#define MSR_MIC2_C20_CTR2             0xEFA
#define MSR_MIC2_C20_CTR3             0xEFB
#define MSR_MIC2_C20_FILTER0          0xEF5
#define MSR_MIC2_C20_FILTER1          0xEF6
#define MSR_MIC2_C20_STATUS           0xEF7
#define MSR_MIC2_C21_GLOBAL_CTRL       0xEFC
#define MSR_MIC2_C21_CTRL0             0xEFD
#define MSR_MIC2_C21_CTRL1             0xEFE
#define MSR_MIC2_C21_CTRL2             0xEFF
#define MSR_MIC2_C21_CTRL3             0xF00
#define MSR_MIC2_C21_CTR0             0xF04
#define MSR_MIC2_C21_CTR1             0xF05
#define MSR_MIC2_C21_CTR2             0xF06
#define MSR_MIC2_C21_CTR3             0xF07
#define MSR_MIC2_C21_FILTER0          0xF01
#define MSR_MIC2_C21_FILTER1          0xF02
#define MSR_MIC2_C21_STATUS           0xF03
#define MSR_MIC2_C22_GLOBAL_CTRL       0xF08
#define MSR_MIC2_C22_CTRL0             0xF09
#define MSR_MIC2_C22_CTRL1             0xF0A
#define MSR_MIC2_C22_CTRL2             0xF0B
#define MSR_MIC2_C22_CTRL3             0xF0C
#define MSR_MIC2_C22_CTR0             0xF10
#define MSR_MIC2_C22_CTR1             0xF11
#define MSR_MIC2_C22_CTR2             0xF12
#define MSR_MIC2_C22_CTR3             0xF13
#define MSR_MIC2_C22_FILTER0          0xF0D
#define MSR_MIC2_C22_FILTER1          0xF0E
#define MSR_MIC2_C22_STATUS           0xF0F
#define MSR_MIC2_C23_GLOBAL_CTRL       0xF14
#define MSR_MIC2_C23_CTRL0             0xF15
#define MSR_MIC2_C23_CTRL1             0xF16
#define MSR_MIC2_C23_CTRL2             0xF17
#define MSR_MIC2_C23_CTRL3             0xF18
#define MSR_MIC2_C23_CTR0             0xF1C
#define MSR_MIC2_C23_CTR1             0xF1D
#define MSR_MIC2_C23_CTR2             0xF1E
#define MSR_MIC2_C23_CTR3             0xF1F
#define MSR_MIC2_C23_FILTER0          0xF19
#define MSR_MIC2_C23_FILTER1          0xF1A
#define MSR_MIC2_C23_STATUS           0xF1B
#define MSR_MIC2_C24_GLOBAL_CTRL       0xF20
#define MSR_MIC2_C24_CTRL0             0xF21
#define MSR_MIC2_C24_CTRL1             0xF22
#define MSR_MIC2_C24_CTRL2             0xF23
#define MSR_MIC2_C24_CTRL3             0xF24
#define MSR_MIC2_C24_CTR0             0xF28
#define MSR_MIC2_C24_CTR1             0xF29
#define MSR_MIC2_C24_CTR2             0xF2A
#define MSR_MIC2_C24_CTR3             0xF2B
#define MSR_MIC2_C24_FILTER0          0xF25
#define MSR_MIC2_C24_FILTER1          0xF26
#define MSR_MIC2_C24_STATUS           0xF27
#define MSR_MIC2_C25_GLOBAL_CTRL       0xF2C
#define MSR_MIC2_C25_CTRL0             0xF2D
#define MSR_MIC2_C25_CTRL1             0xF2E
#define MSR_MIC2_C25_CTRL2             0xF2F
#define MSR_MIC2_C25_CTRL3             0xF30
#define MSR_MIC2_C25_CTR0             0xF34
#define MSR_MIC2_C25_CTR1             0xF35
#define MSR_MIC2_C25_CTR2             0xF36
#define MSR_MIC2_C25_CTR3             0xF37
#define MSR_MIC2_C25_FILTER0          0xF31
#define MSR_MIC2_C25_FILTER1          0xF32
#define MSR_MIC2_C25_STATUS           0xF33
#define MSR_MIC2_C26_GLOBAL_CTRL       0xF38
#define MSR_MIC2_C26_CTRL0             0xF39
#define MSR_MIC2_C26_CTRL1             0xF3A
#define MSR_MIC2_C26_CTRL2             0xF3B
#define MSR_MIC2_C26_CTRL3             0xF3C
#define MSR_MIC2_C26_CTR0             0xF40
#define MSR_MIC2_C26_CTR1             0xF41
#define MSR_MIC2_C26_CTR2             0xF42
#define MSR_MIC2_C26_CTR3             0xF43
#define MSR_MIC2_C26_FILTER0          0xF3D
#define MSR_MIC2_C26_FILTER1          0xF3E
#define MSR_MIC2_C26_STATUS           0xF3F
#define MSR_MIC2_C27_GLOBAL_CTRL       0xF44
#define MSR_MIC2_C27_CTRL0             0xF45
#define MSR_MIC2_C27_CTRL1             0xF46
#define MSR_MIC2_C27_CTRL2             0xF47
#define MSR_MIC2_C27_CTRL3             0xF48
#define MSR_MIC2_C27_CTR0             0xF4C
#define MSR_MIC2_C27_CTR1             0xF4D
#define MSR_MIC2_C27_CTR2             0xF4E
#define MSR_MIC2_C27_CTR3             0xF4F
#define MSR_MIC2_C27_FILTER0          0xF49
#define MSR_MIC2_C27_FILTER1          0xF4A
#define MSR_MIC2_C27_STATUS           0xF4B
#define MSR_MIC2_C28_GLOBAL_CTRL       0xF50
#define MSR_MIC2_C28_CTRL0             0xF51
#define MSR_MIC2_C28_CTRL1             0xF52
#define MSR_MIC2_C28_CTRL2             0xF53
#define MSR_MIC2_C28_CTRL3             0xF54
#define MSR_MIC2_C28_CTR0             0xF58
#define MSR_MIC2_C28_CTR1             0xF59
#define MSR_MIC2_C28_CTR2             0xF5A
#define MSR_MIC2_C28_CTR3             0xF5B
#define MSR_MIC2_C28_FILTER0          0xF55
#define MSR_MIC2_C28_FILTER1          0xF56
#define MSR_MIC2_C28_STATUS           0xF57
#define MSR_MIC2_C29_GLOBAL_CTRL       0xF5C
#define MSR_MIC2_C29_CTRL0             0xF5D
#define MSR_MIC2_C29_CTRL1             0xF5E
#define MSR_MIC2_C29_CTRL2             0xF5F
#define MSR_MIC2_C29_CTRL3             0xF60
#define MSR_MIC2_C29_CTR0             0xF64
#define MSR_MIC2_C29_CTR1             0xF65
#define MSR_MIC2_C29_CTR2             0xF66
#define MSR_MIC2_C29_CTR3             0xF67
#define MSR_MIC2_C29_FILTER0          0xF61
#define MSR_MIC2_C29_FILTER1          0xF62
#define MSR_MIC2_C29_STATUS           0xF63
#define MSR_MIC2_C30_GLOBAL_CTRL       0xF68
#define MSR_MIC2_C30_CTRL0             0xF69
#define MSR_MIC2_C30_CTRL1             0xF6A
#define MSR_MIC2_C30_CTRL2             0xF6B
#define MSR_MIC2_C30_CTRL3             0xF6C
#define MSR_MIC2_C30_CTR0             0xF70
#define MSR_MIC2_C30_CTR1             0xF71
#define MSR_MIC2_C30_CTR2             0xF72
#define MSR_MIC2_C30_CTR3             0xF73
#define MSR_MIC2_C30_FILTER0          0xF6D
#define MSR_MIC2_C30_FILTER1          0xF6E
#define MSR_MIC2_C30_STATUS           0xF6F
#define MSR_MIC2_C31_GLOBAL_CTRL       0xF74
#define MSR_MIC2_C31_CTRL0             0xF75
#define MSR_MIC2_C31_CTRL1             0xF76
#define MSR_MIC2_C31_CTRL2             0xF77
#define MSR_MIC2_C31_CTRL3             0xF78
#define MSR_MIC2_C31_CTR0             0xF7C
#define MSR_MIC2_C31_CTR1             0xF7D
#define MSR_MIC2_C31_CTR2             0xF7E
#define MSR_MIC2_C31_CTR3             0xF7F
#define MSR_MIC2_C31_FILTER0          0xF79
#define MSR_MIC2_C31_FILTER1          0xF7A
#define MSR_MIC2_C31_STATUS           0xF7B
#define MSR_MIC2_C32_GLOBAL_CTRL       0xF80
#define MSR_MIC2_C32_CTRL0             0xF81
#define MSR_MIC2_C32_CTRL1             0xF82
#define MSR_MIC2_C32_CTRL2             0xF83
#define MSR_MIC2_C32_CTRL3             0xF84
#define MSR_MIC2_C32_CTR0             0xF88
#define MSR_MIC2_C32_CTR1             0xF89
#define MSR_MIC2_C32_CTR2             0xF8A
#define MSR_MIC2_C32_CTR3             0xF8B
#define MSR_MIC2_C32_FILTER0          0xF85
#define MSR_MIC2_C32_FILTER1          0xF86
#define MSR_MIC2_C32_STATUS           0xF87
#define MSR_MIC2_C33_GLOBAL_CTRL       0xF8C
#define MSR_MIC2_C33_CTRL0             0xF8D
#define MSR_MIC2_C33_CTRL1             0xF8E
#define MSR_MIC2_C33_CTRL2             0xF8F
#define MSR_MIC2_C33_CTRL3             0xF90
#define MSR_MIC2_C33_CTR0             0xF94
#define MSR_MIC2_C33_CTR1             0xF95
#define MSR_MIC2_C33_CTR2             0xF96
#define MSR_MIC2_C33_CTR3             0xF97
#define MSR_MIC2_C33_FILTER0          0xF91
#define MSR_MIC2_C33_FILTER1          0xF92
#define MSR_MIC2_C33_STATUS           0xF93
#define MSR_MIC2_C34_GLOBAL_CTRL       0xF98
#define MSR_MIC2_C34_CTRL0             0xF99
#define MSR_MIC2_C34_CTRL1             0xF9A
#define MSR_MIC2_C34_CTRL2             0xF9B
#define MSR_MIC2_C34_CTRL3             0xF9C
#define MSR_MIC2_C34_CTR0             0xFA0
#define MSR_MIC2_C34_CTR1             0xFA1
#define MSR_MIC2_C34_CTR2             0xFA2
#define MSR_MIC2_C34_CTR3             0xFA3
#define MSR_MIC2_C34_FILTER0          0xF9D
#define MSR_MIC2_C34_FILTER1          0xF9E
#define MSR_MIC2_C34_STATUS           0xF9F
#define MSR_MIC2_C35_GLOBAL_CTRL       0xFA4
#define MSR_MIC2_C35_CTRL0             0xFA5
#define MSR_MIC2_C35_CTRL1             0xFA6
#define MSR_MIC2_C35_CTRL2             0xFA7
#define MSR_MIC2_C35_CTRL3             0xFA8
#define MSR_MIC2_C35_CTR0             0xFAC
#define MSR_MIC2_C35_CTR1             0xFAD
#define MSR_MIC2_C35_CTR2             0xFAE
#define MSR_MIC2_C35_CTR3             0xFAF
#define MSR_MIC2_C35_FILTER0          0xFA9
#define MSR_MIC2_C35_FILTER1          0xFAA
#define MSR_MIC2_C35_STATUS           0xFAB
#define MSR_MIC2_C36_GLOBAL_CTRL       0xFB0
#define MSR_MIC2_C36_CTRL0             0xFB1
#define MSR_MIC2_C36_CTRL1             0xFB2
#define MSR_MIC2_C36_CTRL2             0xFB3
#define MSR_MIC2_C36_CTRL3             0xFB4
#define MSR_MIC2_C36_CTR0             0xFB8
#define MSR_MIC2_C36_CTR1             0xFB9
#define MSR_MIC2_C36_CTR2             0xFBA
#define MSR_MIC2_C36_CTR3             0xFBB
#define MSR_MIC2_C36_FILTER0          0xFB5
#define MSR_MIC2_C36_FILTER1          0xFB6
#define MSR_MIC2_C36_STATUS           0xFB7
#define MSR_MIC2_C37_GLOBAL_CTRL       0xFBC
#define MSR_MIC2_C37_CTRL0             0xFBD
#define MSR_MIC2_C37_CTRL1             0xFBE
#define MSR_MIC2_C37_CTRL2             0xFBF
#define MSR_MIC2_C37_CTRL3             0xFC0
#define MSR_MIC2_C37_CTR0             0xFC4
#define MSR_MIC2_C37_CTR1             0xFC5
#define MSR_MIC2_C37_CTR2             0xFC6
#define MSR_MIC2_C37_CTR3             0xFC7
#define MSR_MIC2_C37_FILTER0          0xFC1
#define MSR_MIC2_C37_FILTER1          0xFC2
#define MSR_MIC2_C37_STATUS           0xFC3
/* Xeon Phi (Knights Landing) Embedded DRAM controller aka High Bandwidth Memory */
#define PCI_MIC2_EDC_U_CTR0_A	     0x404
#define PCI_MIC2_EDC_U_CTR0_B	     0x400
#define PCI_MIC2_EDC_U_CTR1_A	     0x40C
#define PCI_MIC2_EDC_U_CTR1_B	     0x408
#define PCI_MIC2_EDC_U_CTR2_A	     0x414
#define PCI_MIC2_EDC_U_CTR2_B	     0x410
#define PCI_MIC2_EDC_U_CTR3_A	     0x41C
#define PCI_MIC2_EDC_U_CTR3_B	     0x418
#define PCI_MIC2_EDC_U_CTRL0	     0x420
#define PCI_MIC2_EDC_U_CTRL1	     0x424
#define PCI_MIC2_EDC_U_CTRL2	     0x428
#define PCI_MIC2_EDC_U_CTRL3	     0x42C
#define PCI_MIC2_EDC_U_BOX_CTRL	     0x430
#define PCI_MIC2_EDC_U_BOX_STATUS    0x434
#define PCI_MIC2_EDC_U_FIXED_CTR_A   0x450
#define PCI_MIC2_EDC_U_FIXED_CTR_B   0x44C
#define PCI_MIC2_EDC_U_FIXED_CTRL    0x454
#define PCI_MIC2_EDC_D_CTR0_A	     0xA04
#define PCI_MIC2_EDC_D_CTR0_B	     0xA00
#define PCI_MIC2_EDC_D_CTR1_A	     0xA0C
#define PCI_MIC2_EDC_D_CTR1_B	     0xA08
#define PCI_MIC2_EDC_D_CTR2_A	     0xA14
#define PCI_MIC2_EDC_D_CTR2_B	     0xA10
#define PCI_MIC2_EDC_D_CTR3_A	     0xA1C
#define PCI_MIC2_EDC_D_CTR3_B	     0xA18
#define PCI_MIC2_EDC_D_CTRL0	     0xA20
#define PCI_MIC2_EDC_D_CTRL1	     0xA24
#define PCI_MIC2_EDC_D_CTRL2	     0xA28
#define PCI_MIC2_EDC_D_CTRL3	     0xA2C
#define PCI_MIC2_EDC_D_BOX_CTRL	     0xA30
#define PCI_MIC2_EDC_D_BOX_STATUS    0xA34
#define PCI_MIC2_EDC_D_FIXED_CTR_A   0xA40
#define PCI_MIC2_EDC_D_FIXED_CTR_B   0xA3C
#define PCI_MIC2_EDC_D_FIXED_CTRL    0xA44
/* Xeon Phi (Knights Landing) Memory controller*/
#define PCI_MIC2_MC_U_CTR0_A	     0x404
#define PCI_MIC2_MC_U_CTR0_B	     0x400
#define PCI_MIC2_MC_U_CTR1_A	     0x40C
#define PCI_MIC2_MC_U_CTR1_B	     0x408
#define PCI_MIC2_MC_U_CTR2_A	     0x414
#define PCI_MIC2_MC_U_CTR2_B	     0x410
#define PCI_MIC2_MC_U_CTR3_A	     0x41C
#define PCI_MIC2_MC_U_CTR3_B	     0x418
#define PCI_MIC2_MC_U_CTRL0	     0x420
#define PCI_MIC2_MC_U_CTRL1	     0x424
#define PCI_MIC2_MC_U_CTRL2	     0x428
#define PCI_MIC2_MC_U_CTRL3	     0x42C
#define PCI_MIC2_MC_U_BOX_CTRL	     0x430
#define PCI_MIC2_MC_U_BOX_STATUS    0x434
#define PCI_MIC2_MC_U_FIXED_CTR_A   0x450
#define PCI_MIC2_MC_U_FIXED_CTR_B   0x44C
#define PCI_MIC2_MC_U_FIXED_CTRL    0x454
#define PCI_MIC2_MC_D_CTR0_A	     0xB04
#define PCI_MIC2_MC_D_CTR0_B	     0xB00
#define PCI_MIC2_MC_D_CTR1_A	     0xB0C
#define PCI_MIC2_MC_D_CTR1_B	     0xB08
#define PCI_MIC2_MC_D_CTR2_A	     0xB14
#define PCI_MIC2_MC_D_CTR2_B	     0xB10
#define PCI_MIC2_MC_D_CTR3_A	     0xB1C
#define PCI_MIC2_MC_D_CTR3_B	     0xB18
#define PCI_MIC2_MC_D_CTRL0	     0xB20
#define PCI_MIC2_MC_D_CTRL1	     0xB24
#define PCI_MIC2_MC_D_CTRL2	     0xB28
#define PCI_MIC2_MC_D_CTRL3	     0xB2C
#define PCI_MIC2_MC_D_BOX_CTRL	     0xB30
#define PCI_MIC2_MC_D_BOX_STATUS    0xB34
#define PCI_MIC2_MC_D_FIXED_CTR_A   0xB40
#define PCI_MIC2_MC_D_FIXED_CTR_B   0xB3C
#define PCI_MIC2_MC_D_FIXED_CTRL    0xB44
/* Xeon Phi (Knights Landing) M2PCIE */
#define PCI_MIC2_M2PCIE_CTR0_A	    0xA4
#define PCI_MIC2_M2PCIE_CTR0_B	    0xA0
#define PCI_MIC2_M2PCIE_CTR1_A	    0xAC
#define PCI_MIC2_M2PCIE_CTR1_B	    0xA8
#define PCI_MIC2_M2PCIE_CTR2_A	    0xB4
#define PCI_MIC2_M2PCIE_CTR2_B	    0xB0
#define PCI_MIC2_M2PCIE_CTR3_A	    0xBC
#define PCI_MIC2_M2PCIE_CTR3_B	    0xB8
#define PCI_MIC2_M2PCIE_CTRL0	    0xD8
#define PCI_MIC2_M2PCIE_CTRL1	    0xDC
#define PCI_MIC2_M2PCIE_CTRL2	    0xE0
#define PCI_MIC2_M2PCIE_CTRL3	    0xE4
#define PCI_MIC2_M2PCIE_BOX_CTRL    0xF4
#define PCI_MIC2_M2PCIE_BOX_STATUS  0xF8
/* Xeon Phi (Knights Landing) IRP */
#define PCI_MIC2_IRP_CTR0	    0xA0
#define PCI_MIC2_IRP_CTR1	    0xA8
#define PCI_MIC2_IRP_CTRL0	    0xD8
#define PCI_MIC2_IRP_CTRL1	    0xDC
#define PCI_MIC2_IRP_BOX_CTRL	    0xF0
#define PCI_MIC2_IRP_BOX_STATUS	    0xF4

/* Core v1/v2 type uncore
 * Naming following Intel Uncore Performance Monitoring Guide
 * Ref. Nr. 327043-001 and 329468-001
 * */

/* CBo Performance Monitoring */
#define MSR_UNC_C0_PMON_CTR0           0xD16
#define MSR_UNC_C0_PMON_CTR1           0xD17
#define MSR_UNC_C0_PMON_CTR2           0xD18
#define MSR_UNC_C0_PMON_CTR3           0xD19
#define MSR_UNC_C0_PMON_CTL0           0xD10
#define MSR_UNC_C0_PMON_CTL1           0xD11
#define MSR_UNC_C0_PMON_CTL2           0xD12
#define MSR_UNC_C0_PMON_CTL3           0xD13
#define MSR_UNC_C0_PMON_BOX_FILTER     0xD14
#define MSR_UNC_C0_PMON_BOX_FILTER1    0xD1A
#define MSR_UNC_C0_PMON_BOX_CTL        0xD04

#define MSR_UNC_C1_PMON_CTR0           0xD36
#define MSR_UNC_C1_PMON_CTR1           0xD37
#define MSR_UNC_C1_PMON_CTR2           0xD38
#define MSR_UNC_C1_PMON_CTR3           0xD39
#define MSR_UNC_C1_PMON_CTL0           0xD30
#define MSR_UNC_C1_PMON_CTL1           0xD31
#define MSR_UNC_C1_PMON_CTL2           0xD32
#define MSR_UNC_C1_PMON_CTL3           0xD33
#define MSR_UNC_C1_PMON_BOX_FILTER     0xD34
#define MSR_UNC_C1_PMON_BOX_FILTER1    0xD3A
#define MSR_UNC_C1_PMON_BOX_CTL        0xD24

#define MSR_UNC_C2_PMON_CTR0           0xD56
#define MSR_UNC_C2_PMON_CTR1           0xD57
#define MSR_UNC_C2_PMON_CTR2           0xD58
#define MSR_UNC_C2_PMON_CTR3           0xD59
#define MSR_UNC_C2_PMON_CTL0           0xD50
#define MSR_UNC_C2_PMON_CTL1           0xD51
#define MSR_UNC_C2_PMON_CTL2           0xD52
#define MSR_UNC_C2_PMON_CTL3           0xD53
#define MSR_UNC_C2_PMON_BOX_FILTER     0xD54
#define MSR_UNC_C2_PMON_BOX_FILTER1    0xD5A
#define MSR_UNC_C2_PMON_BOX_CTL        0xD44

#define MSR_UNC_C3_PMON_CTR0           0xD76
#define MSR_UNC_C3_PMON_CTR1           0xD77
#define MSR_UNC_C3_PMON_CTR2           0xD78
#define MSR_UNC_C3_PMON_CTR3           0xD79
#define MSR_UNC_C3_PMON_CTL0           0xD70
#define MSR_UNC_C3_PMON_CTL1           0xD71
#define MSR_UNC_C3_PMON_CTL2           0xD72
#define MSR_UNC_C3_PMON_CTL3           0xD73
#define MSR_UNC_C3_PMON_BOX_FILTER     0xD74
#define MSR_UNC_C3_PMON_BOX_FILTER1    0xD7A
#define MSR_UNC_C3_PMON_BOX_CTL        0xD64

#define MSR_UNC_C4_PMON_CTR0           0xD96
#define MSR_UNC_C4_PMON_CTR1           0xD97
#define MSR_UNC_C4_PMON_CTR2           0xD98
#define MSR_UNC_C4_PMON_CTR3           0xD99
#define MSR_UNC_C4_PMON_CTL0           0xD90
#define MSR_UNC_C4_PMON_CTL1           0xD91
#define MSR_UNC_C4_PMON_CTL2           0xD92
#define MSR_UNC_C4_PMON_CTL3           0xD93
#define MSR_UNC_C4_PMON_BOX_FILTER     0xD94
#define MSR_UNC_C4_PMON_BOX_FILTER1    0xD9A
#define MSR_UNC_C4_PMON_BOX_CTL        0xD84

#define MSR_UNC_C5_PMON_CTR0           0xDB6
#define MSR_UNC_C5_PMON_CTR1           0xDB7
#define MSR_UNC_C5_PMON_CTR2           0xDB8
#define MSR_UNC_C5_PMON_CTR3           0xDB9
#define MSR_UNC_C5_PMON_CTL0           0xDB0
#define MSR_UNC_C5_PMON_CTL1           0xDB1
#define MSR_UNC_C5_PMON_CTL2           0xDB2
#define MSR_UNC_C5_PMON_CTL3           0xDB3
#define MSR_UNC_C5_PMON_BOX_FILTER     0xDB4
#define MSR_UNC_C5_PMON_BOX_FILTER1    0xDBA
#define MSR_UNC_C5_PMON_BOX_CTL        0xDA4

#define MSR_UNC_C6_PMON_CTR0           0xDD6
#define MSR_UNC_C6_PMON_CTR1           0xDD7
#define MSR_UNC_C6_PMON_CTR2           0xDD8
#define MSR_UNC_C6_PMON_CTR3           0xDD9
#define MSR_UNC_C6_PMON_CTL0           0xDD0
#define MSR_UNC_C6_PMON_CTL1           0xDD1
#define MSR_UNC_C6_PMON_CTL2           0xDD2
#define MSR_UNC_C6_PMON_CTL3           0xDD3
#define MSR_UNC_C6_PMON_BOX_FILTER     0xDD4
#define MSR_UNC_C6_PMON_BOX_FILTER1    0xDDA
#define MSR_UNC_C6_PMON_BOX_CTL        0xDC4

#define MSR_UNC_C7_PMON_CTR0           0xDF6
#define MSR_UNC_C7_PMON_CTR1           0xDF7
#define MSR_UNC_C7_PMON_CTR2           0xDF8
#define MSR_UNC_C7_PMON_CTR3           0xDF9
#define MSR_UNC_C7_PMON_CTL0           0xDF0
#define MSR_UNC_C7_PMON_CTL1           0xDF1
#define MSR_UNC_C7_PMON_CTL2           0xDF2
#define MSR_UNC_C7_PMON_CTL3           0xDF3
#define MSR_UNC_C7_PMON_BOX_FILTER     0xDF4
#define MSR_UNC_C7_PMON_BOX_FILTER1    0xDFA
#define MSR_UNC_C7_PMON_BOX_CTL        0xDE4

#define MSR_UNC_C8_PMON_CTR0           0xE16
#define MSR_UNC_C8_PMON_CTR1           0xE17
#define MSR_UNC_C8_PMON_CTR2           0xE18
#define MSR_UNC_C8_PMON_CTR3           0xE19
#define MSR_UNC_C8_PMON_CTL0           0xE10
#define MSR_UNC_C8_PMON_CTL1           0xE11
#define MSR_UNC_C8_PMON_CTL2           0xE12
#define MSR_UNC_C8_PMON_CTL3           0xE13
#define MSR_UNC_C8_PMON_BOX_FILTER     0xE14
#define MSR_UNC_C8_PMON_BOX_FILTER1    0xE1A
#define MSR_UNC_C8_PMON_BOX_CTL        0xE04

#define MSR_UNC_C9_PMON_CTR0           0xE36
#define MSR_UNC_C9_PMON_CTR1           0xE37
#define MSR_UNC_C9_PMON_CTR2           0xE38
#define MSR_UNC_C9_PMON_CTR3           0xE39
#define MSR_UNC_C9_PMON_CTL0           0xE30
#define MSR_UNC_C9_PMON_CTL1           0xE31
#define MSR_UNC_C9_PMON_CTL2           0xE32
#define MSR_UNC_C9_PMON_CTL3           0xE33
#define MSR_UNC_C9_PMON_BOX_FILTER     0xE34
#define MSR_UNC_C9_PMON_BOX_FILTER1    0xE3A
#define MSR_UNC_C9_PMON_BOX_CTL        0xE24

#define MSR_UNC_C10_PMON_CTR0           0xE56
#define MSR_UNC_C10_PMON_CTR1           0xE57
#define MSR_UNC_C10_PMON_CTR2           0xE58
#define MSR_UNC_C10_PMON_CTR3           0xE59
#define MSR_UNC_C10_PMON_CTL0           0xE50
#define MSR_UNC_C10_PMON_CTL1           0xE51
#define MSR_UNC_C10_PMON_CTL2           0xE52
#define MSR_UNC_C10_PMON_CTL3           0xE53
#define MSR_UNC_C10_PMON_BOX_FILTER     0xE54
#define MSR_UNC_C10_PMON_BOX_FILTER1    0xE5A
#define MSR_UNC_C10_PMON_BOX_CTL        0xE44

#define MSR_UNC_C11_PMON_CTR0           0xE76
#define MSR_UNC_C11_PMON_CTR1           0xE77
#define MSR_UNC_C11_PMON_CTR2           0xE78
#define MSR_UNC_C11_PMON_CTR3           0xE79
#define MSR_UNC_C11_PMON_CTL0           0xE70
#define MSR_UNC_C11_PMON_CTL1           0xE71
#define MSR_UNC_C11_PMON_CTL2           0xE72
#define MSR_UNC_C11_PMON_CTL3           0xE73
#define MSR_UNC_C11_PMON_BOX_FILTER     0xE74
#define MSR_UNC_C11_PMON_BOX_FILTER1    0xE7A
#define MSR_UNC_C11_PMON_BOX_CTL        0xE64

#define MSR_UNC_C12_PMON_CTR0           0xE96
#define MSR_UNC_C12_PMON_CTR1           0xE97
#define MSR_UNC_C12_PMON_CTR2           0xE98
#define MSR_UNC_C12_PMON_CTR3           0xE99
#define MSR_UNC_C12_PMON_CTL0           0xE90
#define MSR_UNC_C12_PMON_CTL1           0xE91
#define MSR_UNC_C12_PMON_CTL2           0xE92
#define MSR_UNC_C12_PMON_CTL3           0xE93
#define MSR_UNC_C12_PMON_BOX_FILTER     0xE94
#define MSR_UNC_C12_PMON_BOX_FILTER1    0xE9A
#define MSR_UNC_C12_PMON_BOX_CTL        0xE84

#define MSR_UNC_C13_PMON_CTR0           0xEB6
#define MSR_UNC_C13_PMON_CTR1           0xEB7
#define MSR_UNC_C13_PMON_CTR2           0xEB8
#define MSR_UNC_C13_PMON_CTR3           0xEB9
#define MSR_UNC_C13_PMON_CTL0           0xEB0
#define MSR_UNC_C13_PMON_CTL1           0xEB1
#define MSR_UNC_C13_PMON_CTL2           0xEB2
#define MSR_UNC_C13_PMON_CTL3           0xEB3
#define MSR_UNC_C13_PMON_BOX_FILTER     0xEB4
#define MSR_UNC_C13_PMON_BOX_FILTER1    0xEBA
#define MSR_UNC_C13_PMON_BOX_CTL        0xEA4

#define MSR_UNC_C14_PMON_CTR0           0xED6
#define MSR_UNC_C14_PMON_CTR1           0xED7
#define MSR_UNC_C14_PMON_CTR2           0xED8
#define MSR_UNC_C14_PMON_CTR3           0xED9
#define MSR_UNC_C14_PMON_CTL0           0xED0
#define MSR_UNC_C14_PMON_CTL1           0xED1
#define MSR_UNC_C14_PMON_CTL2           0xED2
#define MSR_UNC_C14_PMON_CTL3           0xED3
#define MSR_UNC_C14_PMON_BOX_FILTER     0xED4
#define MSR_UNC_C14_PMON_BOX_FILTER1    0xEDA
#define MSR_UNC_C14_PMON_BOX_CTL        0xEC4

/* PCU (Power Control) Performance Monitoring */

#define MSR_UNC_PCU_PMON_CTR0           0xC36
#define MSR_UNC_PCU_PMON_CTR1           0xC37
#define MSR_UNC_PCU_PMON_CTR2           0xC38
#define MSR_UNC_PCU_PMON_CTR3           0xC39
#define MSR_UNC_PCU_PMON_CTL0           0xC30
#define MSR_UNC_PCU_PMON_CTL1           0xC31
#define MSR_UNC_PCU_PMON_CTL2           0xC32
#define MSR_UNC_PCU_PMON_CTL3           0xC33
#define MSR_UNC_PCU_PMON_BOX_FILTER     0xC34
#define MSR_UNC_PCU_PMON_BOX_CTL        0xC24
#define MSR_UNC_PCU_PMON_BOX_STATUS     0xC35
#define MSR_UNC_PCU_PMON_FIXED_CTR0     0x3FC
#define MSR_UNC_PCU_PMON_FIXED_CTR1     0x3FD

/* UBox Performance Monitoring */

#define MSR_UNC_U_PMON_CTR0             0xC16
#define MSR_UNC_U_PMON_CTR1             0xC17
#define MSR_UNC_U_PMON_CTL0             0xC10
#define MSR_UNC_U_PMON_CTL1             0xC11
#define MSR_UNC_U_UCLK_FIXED_CTR        0xC09
#define MSR_UNC_U_UCLK_FIXED_CTL        0xC08
#define MSR_UNC_U_PMON_BOX_STATUS       0xC15
#define MSR_UNC_U_PMON_GLOBAL_STATUS    0xC01
#define MSR_UNC_U_PMON_GLOBAL_CTL       0xC00
#define MSR_UNC_U_PMON_GLOBAL_CONFIG    0xC06

/* HA Box Performance Monitoring */

#define PCI_UNC_HA_PMON_BOX_CTL         0xF4
#define PCI_UNC_HA_PMON_BOX_STATUS      0xF8
#define PCI_UNC_HA_PMON_CTL_0           0xD8
#define PCI_UNC_HA_PMON_CTL_1           0xDC
#define PCI_UNC_HA_PMON_CTL_2           0xE0
#define PCI_UNC_HA_PMON_CTL_3           0xE4
#define PCI_UNC_HA_PMON_CTR_0_A         0xA4
#define PCI_UNC_HA_PMON_CTR_1_A         0xAC
#define PCI_UNC_HA_PMON_CTR_2_A         0xB4
#define PCI_UNC_HA_PMON_CTR_3_A         0xBC
#define PCI_UNC_HA_PMON_CTR_0_B         0xA0
#define PCI_UNC_HA_PMON_CTR_1_B         0xA8
#define PCI_UNC_HA_PMON_CTR_2_B         0xB0
#define PCI_UNC_HA_PMON_CTR_3_B         0xB8
#define PCI_UNC_HA_PMON_OPCODEMATCH     0x48
#define PCI_UNC_HA_PMON_ADDRMATCH0      0x40
#define PCI_UNC_HA_PMON_ADDRMATCH1      0x44

/* iMC Box Performance Monitoring */

#define PCI_UNC_MC_PMON_BOX_CTL         0xF4
#define PCI_UNC_MC_PMON_BOX_STATUS      0xF8
#define PCI_UNC_MC_PMON_FIXED_CTL       0xF0
#define PCI_UNC_MC_PMON_CTL_0           0xD8
#define PCI_UNC_MC_PMON_CTL_1           0xDC
#define PCI_UNC_MC_PMON_CTL_2           0xE0
#define PCI_UNC_MC_PMON_CTL_3           0xE4
#define PCI_UNC_MC_PMON_FIXED_CTR_A     0xD4
#define PCI_UNC_MC_PMON_FIXED_CTR_B     0xD0
#define PCI_UNC_MC_PMON_CTR_0_A         0xA4
#define PCI_UNC_MC_PMON_CTR_1_A         0xAC
#define PCI_UNC_MC_PMON_CTR_2_A         0xB4
#define PCI_UNC_MC_PMON_CTR_3_A         0xBC
#define PCI_UNC_MC_PMON_CTR_0_B         0xA0
#define PCI_UNC_MC_PMON_CTR_1_B         0xA8
#define PCI_UNC_MC_PMON_CTR_2_B         0xB0
#define PCI_UNC_MC_PMON_CTR_3_B         0xB8

/* IRP Performance Monitoring */
#define PCI_UNC_IRP_PMON_BOX_STATUS     0xF8
#define PCI_UNC_IRP_PMON_BOX_CTL        0xF4
#define PCI_UNC_IRP0_PMON_CTL_0         0xD8
#define PCI_UNC_IRP0_PMON_CTL_1         0xDC
#define PCI_UNC_IRP0_PMON_CTR_0         0xA0
#define PCI_UNC_IRP0_PMON_CTR_1         0xB0
#define PCI_UNC_IRP1_PMON_CTL_0         0xE0
#define PCI_UNC_IRP1_PMON_CTL_1         0xE4
#define PCI_UNC_IRP1_PMON_CTR_0         0xB8
#define PCI_UNC_IRP1_PMON_CTR_1         0xC0

/* QPI Box Performance Monitoring */

#define PCI_UNC_QPI_PMON_BOX_CTL         0xF4
#define PCI_UNC_QPI_PMON_BOX_STATUS      0xF8
#define PCI_UNC_QPI_PMON_CTL_0           0xD8
#define PCI_UNC_QPI_PMON_CTL_1           0xDC
#define PCI_UNC_QPI_PMON_CTL_2           0xE0
#define PCI_UNC_QPI_PMON_CTL_3           0xE4
#define PCI_UNC_QPI_PMON_CTR_0_A         0xA4
#define PCI_UNC_QPI_PMON_CTR_1_A         0xAC
#define PCI_UNC_QPI_PMON_CTR_2_A         0xB4
#define PCI_UNC_QPI_PMON_CTR_3_A         0xBC
#define PCI_UNC_QPI_PMON_CTR_0_B         0xA0
#define PCI_UNC_QPI_PMON_CTR_1_B         0xA8
#define PCI_UNC_QPI_PMON_CTR_2_B         0xB0
#define PCI_UNC_QPI_PMON_CTR_3_B         0xB8
#define PCI_UNC_QPI_PMON_MASK_0          0x238
#define PCI_UNC_QPI_PMON_MASK_1          0x23C
#define PCI_UNC_QPI_PMON_MATCH_0         0x228
#define PCI_UNC_QPI_PMON_MATCH_1         0x22C
#define PCI_UNC_QPI_RATE_STATUS          0xD4

/* R2PCIE Box Performance Monitoring */

#define PCI_UNC_R2PCIE_PMON_BOX_CTL         0xF4
#define PCI_UNC_R2PCIE_PMON_BOX_STATUS      0xF8
#define PCI_UNC_R2PCIE_PMON_CTL_0           0xD8
#define PCI_UNC_R2PCIE_PMON_CTL_1           0xDC
#define PCI_UNC_R2PCIE_PMON_CTL_2           0xE0
#define PCI_UNC_R2PCIE_PMON_CTL_3           0xE4
#define PCI_UNC_R2PCIE_PMON_CTR_0_A         0xA4
#define PCI_UNC_R2PCIE_PMON_CTR_1_A         0xAC
#define PCI_UNC_R2PCIE_PMON_CTR_2_A         0xB4
#define PCI_UNC_R2PCIE_PMON_CTR_3_A         0xBC
#define PCI_UNC_R2PCIE_PMON_CTR_0_B         0xA0
#define PCI_UNC_R2PCIE_PMON_CTR_1_B         0xA8
#define PCI_UNC_R2PCIE_PMON_CTR_2_B         0xB0
#define PCI_UNC_R2PCIE_PMON_CTR_3_B         0xB8

/* R3QPI Box Performance Monitoring */

#define PCI_UNC_R3QPI_PMON_BOX_CTL         0xF4
#define PCI_UNC_R3QPI_PMON_BOX_STATUS      0xF8
#define PCI_UNC_R3QPI_PMON_CTL_0           0xD8
#define PCI_UNC_R3QPI_PMON_CTL_1           0xDC
#define PCI_UNC_R3QPI_PMON_CTL_2           0xE0
#define PCI_UNC_R3QPI_PMON_CTR_0_A         0xA4
#define PCI_UNC_R3QPI_PMON_CTR_1_A         0xAC
#define PCI_UNC_R3QPI_PMON_CTR_2_A         0xB4
#define PCI_UNC_R3QPI_PMON_CTR_0_B         0xA0
#define PCI_UNC_R3QPI_PMON_CTR_1_B         0xA8
#define PCI_UNC_R3QPI_PMON_CTR_2_B         0xB0

/* ########################################################## */
/* Core v3 type uncore
 * Naming following Intel Uncore Performance Monitoring Guide
 * Ref. Nr. 331051-001
 * */

/* UBox Performance Monitoring */
#define MSR_UNC_V3_U_PMON_CTR0             0x709
#define MSR_UNC_V3_U_PMON_CTR1             0x70A
#define MSR_UNC_V3_U_PMON_CTL0             0x705
#define MSR_UNC_V3_U_PMON_CTL1             0x706
#define MSR_UNC_V3_U_UCLK_FIXED_CTR        0x704
#define MSR_UNC_V3_U_UCLK_FIXED_CTL        0x703
#define MSR_UNC_V3_U_PMON_BOX_STATUS       0x708
#define MSR_UNC_V3_U_PMON_GLOBAL_STATUS    0x701
#define MSR_UNC_V3_U_PMON_GLOBAL_CTL       0x700
#define MSR_UNC_V3_U_PMON_GLOBAL_CONFIG    0x702

/* CBox Performance Monitoring */
#define MSR_UNC_V3_C0_PMON_BOX_CTL         0xE00
#define MSR_UNC_V3_C0_PMON_BOX_STATUS      0xE07
#define MSR_UNC_V3_C0_PMON_BOX_FILTER0     0xE05
#define MSR_UNC_V3_C0_PMON_BOX_FILTER1     0xE06
#define MSR_UNC_V3_C0_PMON_CTL0            0xE01
#define MSR_UNC_V3_C0_PMON_CTL1            0xE02
#define MSR_UNC_V3_C0_PMON_CTL2            0xE03
#define MSR_UNC_V3_C0_PMON_CTL3            0xE04
#define MSR_UNC_V3_C0_PMON_CTR0            0xE08
#define MSR_UNC_V3_C0_PMON_CTR1            0xE09
#define MSR_UNC_V3_C0_PMON_CTR2            0xE0A
#define MSR_UNC_V3_C0_PMON_CTR3            0xE0B

#define MSR_UNC_V3_C1_PMON_BOX_CTL         0xE10
#define MSR_UNC_V3_C1_PMON_BOX_STATUS      0xE17
#define MSR_UNC_V3_C1_PMON_BOX_FILTER0     0xE15
#define MSR_UNC_V3_C1_PMON_BOX_FILTER1     0xE16
#define MSR_UNC_V3_C1_PMON_CTL0            0xE11
#define MSR_UNC_V3_C1_PMON_CTL1            0xE12
#define MSR_UNC_V3_C1_PMON_CTL2            0xE13
#define MSR_UNC_V3_C1_PMON_CTL3            0xE14
#define MSR_UNC_V3_C1_PMON_CTR0            0xE18
#define MSR_UNC_V3_C1_PMON_CTR1            0xE19
#define MSR_UNC_V3_C1_PMON_CTR2            0xE1A
#define MSR_UNC_V3_C1_PMON_CTR3            0xE1B

#define MSR_UNC_V3_C2_PMON_BOX_CTL         0xE20
#define MSR_UNC_V3_C2_PMON_BOX_STATUS      0xE27
#define MSR_UNC_V3_C2_PMON_BOX_FILTER0     0xE25
#define MSR_UNC_V3_C2_PMON_BOX_FILTER1     0xE26
#define MSR_UNC_V3_C2_PMON_CTL0            0xE21
#define MSR_UNC_V3_C2_PMON_CTL1            0xE22
#define MSR_UNC_V3_C2_PMON_CTL2            0xE23
#define MSR_UNC_V3_C2_PMON_CTL3            0xE24
#define MSR_UNC_V3_C2_PMON_CTR0            0xE28
#define MSR_UNC_V3_C2_PMON_CTR1            0xE29
#define MSR_UNC_V3_C2_PMON_CTR2            0xE2A
#define MSR_UNC_V3_C2_PMON_CTR3            0xE2B

#define MSR_UNC_V3_C3_PMON_BOX_CTL         0xE30
#define MSR_UNC_V3_C3_PMON_BOX_STATUS      0xE37
#define MSR_UNC_V3_C3_PMON_BOX_FILTER0     0xE35
#define MSR_UNC_V3_C3_PMON_BOX_FILTER1     0xE36
#define MSR_UNC_V3_C3_PMON_CTL0            0xE31
#define MSR_UNC_V3_C3_PMON_CTL1            0xE32
#define MSR_UNC_V3_C3_PMON_CTL2            0xE33
#define MSR_UNC_V3_C3_PMON_CTL3            0xE34
#define MSR_UNC_V3_C3_PMON_CTR0            0xE38
#define MSR_UNC_V3_C3_PMON_CTR1            0xE39
#define MSR_UNC_V3_C3_PMON_CTR2            0xE3A
#define MSR_UNC_V3_C3_PMON_CTR3            0xE3B

#define MSR_UNC_V3_C4_PMON_BOX_CTL         0xE40
#define MSR_UNC_V3_C4_PMON_BOX_STATUS      0xE47
#define MSR_UNC_V3_C4_PMON_BOX_FILTER0     0xE45
#define MSR_UNC_V3_C4_PMON_BOX_FILTER1     0xE46
#define MSR_UNC_V3_C4_PMON_CTL0            0xE41
#define MSR_UNC_V3_C4_PMON_CTL1            0xE42
#define MSR_UNC_V3_C4_PMON_CTL2            0xE43
#define MSR_UNC_V3_C4_PMON_CTL3            0xE44
#define MSR_UNC_V3_C4_PMON_CTR0            0xE48
#define MSR_UNC_V3_C4_PMON_CTR1            0xE49
#define MSR_UNC_V3_C4_PMON_CTR2            0xE4A
#define MSR_UNC_V3_C4_PMON_CTR3            0xE4B

#define MSR_UNC_V3_C5_PMON_BOX_CTL         0xE50
#define MSR_UNC_V3_C5_PMON_BOX_STATUS      0xE57
#define MSR_UNC_V3_C5_PMON_BOX_FILTER0     0xE55
#define MSR_UNC_V3_C5_PMON_BOX_FILTER1     0xE56
#define MSR_UNC_V3_C5_PMON_CTL0            0xE51
#define MSR_UNC_V3_C5_PMON_CTL1            0xE52
#define MSR_UNC_V3_C5_PMON_CTL2            0xE53
#define MSR_UNC_V3_C5_PMON_CTL3            0xE54
#define MSR_UNC_V3_C5_PMON_CTR0            0xE58
#define MSR_UNC_V3_C5_PMON_CTR1            0xE59
#define MSR_UNC_V3_C5_PMON_CTR2            0xE5A
#define MSR_UNC_V3_C5_PMON_CTR3            0xE5B

#define MSR_UNC_V3_C6_PMON_BOX_CTL         0xE60
#define MSR_UNC_V3_C6_PMON_BOX_STATUS      0xE67
#define MSR_UNC_V3_C6_PMON_BOX_FILTER0     0xE65
#define MSR_UNC_V3_C6_PMON_BOX_FILTER1     0xE66
#define MSR_UNC_V3_C6_PMON_CTL0            0xE61
#define MSR_UNC_V3_C6_PMON_CTL1            0xE62
#define MSR_UNC_V3_C6_PMON_CTL2            0xE63
#define MSR_UNC_V3_C6_PMON_CTL3            0xE64
#define MSR_UNC_V3_C6_PMON_CTR0            0xE68
#define MSR_UNC_V3_C6_PMON_CTR1            0xE69
#define MSR_UNC_V3_C6_PMON_CTR2            0xE6A
#define MSR_UNC_V3_C6_PMON_CTR3            0xE6B

#define MSR_UNC_V3_C7_PMON_BOX_CTL         0xE70
#define MSR_UNC_V3_C7_PMON_BOX_STATUS      0xE77
#define MSR_UNC_V3_C7_PMON_BOX_FILTER0     0xE75
#define MSR_UNC_V3_C7_PMON_BOX_FILTER1     0xE76
#define MSR_UNC_V3_C7_PMON_CTL0            0xE71
#define MSR_UNC_V3_C7_PMON_CTL1            0xE72
#define MSR_UNC_V3_C7_PMON_CTL2            0xE73
#define MSR_UNC_V3_C7_PMON_CTL3            0xE74
#define MSR_UNC_V3_C7_PMON_CTR0            0xE78
#define MSR_UNC_V3_C7_PMON_CTR1            0xE79
#define MSR_UNC_V3_C7_PMON_CTR2            0xE7A
#define MSR_UNC_V3_C7_PMON_CTR3            0xE7B

#define MSR_UNC_V3_C8_PMON_BOX_CTL         0xE80
#define MSR_UNC_V3_C8_PMON_BOX_STATUS      0xE87
#define MSR_UNC_V3_C8_PMON_BOX_FILTER0     0xE85
#define MSR_UNC_V3_C8_PMON_BOX_FILTER1     0xE86
#define MSR_UNC_V3_C8_PMON_CTL0            0xE81
#define MSR_UNC_V3_C8_PMON_CTL1            0xE82
#define MSR_UNC_V3_C8_PMON_CTL2            0xE83
#define MSR_UNC_V3_C8_PMON_CTL3            0xE84
#define MSR_UNC_V3_C8_PMON_CTR0            0xE88
#define MSR_UNC_V3_C8_PMON_CTR1            0xE89
#define MSR_UNC_V3_C8_PMON_CTR2            0xE8A
#define MSR_UNC_V3_C8_PMON_CTR3            0xE8B

#define MSR_UNC_V3_C9_PMON_BOX_CTL         0xE90
#define MSR_UNC_V3_C9_PMON_BOX_STATUS      0xE97
#define MSR_UNC_V3_C9_PMON_BOX_FILTER0     0xE95
#define MSR_UNC_V3_C9_PMON_BOX_FILTER1     0xE96
#define MSR_UNC_V3_C9_PMON_CTL0            0xE91
#define MSR_UNC_V3_C9_PMON_CTL1            0xE92
#define MSR_UNC_V3_C9_PMON_CTL2            0xE93
#define MSR_UNC_V3_C9_PMON_CTL3            0xE94
#define MSR_UNC_V3_C9_PMON_CTR0            0xE98
#define MSR_UNC_V3_C9_PMON_CTR1            0xE99
#define MSR_UNC_V3_C9_PMON_CTR2            0xE9A
#define MSR_UNC_V3_C9_PMON_CTR3            0xE9B

#define MSR_UNC_V3_C10_PMON_BOX_CTL        0xEA0
#define MSR_UNC_V3_C10_PMON_BOX_STATUS     0xEA7
#define MSR_UNC_V3_C10_PMON_BOX_FILTER0    0xEA5
#define MSR_UNC_V3_C10_PMON_BOX_FILTER1    0xEA6
#define MSR_UNC_V3_C10_PMON_CTL0           0xEA1
#define MSR_UNC_V3_C10_PMON_CTL1           0xEA2
#define MSR_UNC_V3_C10_PMON_CTL2           0xEA3
#define MSR_UNC_V3_C10_PMON_CTL3           0xEA4
#define MSR_UNC_V3_C10_PMON_CTR0           0xEA8
#define MSR_UNC_V3_C10_PMON_CTR1           0xEA9
#define MSR_UNC_V3_C10_PMON_CTR2           0xEAA
#define MSR_UNC_V3_C10_PMON_CTR3           0xEAB

#define MSR_UNC_V3_C11_PMON_BOX_CTL        0xEB0
#define MSR_UNC_V3_C11_PMON_BOX_STATUS     0xEB7
#define MSR_UNC_V3_C11_PMON_BOX_FILTER0    0xEB5
#define MSR_UNC_V3_C11_PMON_BOX_FILTER1    0xEB6
#define MSR_UNC_V3_C11_PMON_CTL0           0xEB1
#define MSR_UNC_V3_C11_PMON_CTL1           0xEB2
#define MSR_UNC_V3_C11_PMON_CTL2           0xEB3
#define MSR_UNC_V3_C11_PMON_CTL3           0xEB4
#define MSR_UNC_V3_C11_PMON_CTR0           0xEB8
#define MSR_UNC_V3_C11_PMON_CTR1           0xEB9
#define MSR_UNC_V3_C11_PMON_CTR2           0xEBA
#define MSR_UNC_V3_C11_PMON_CTR3           0xEBB

#define MSR_UNC_V3_C12_PMON_BOX_CTL        0xEC0
#define MSR_UNC_V3_C12_PMON_BOX_STATUS     0xEC7
#define MSR_UNC_V3_C12_PMON_BOX_FILTER0    0xEC5
#define MSR_UNC_V3_C12_PMON_BOX_FILTER1    0xEC6
#define MSR_UNC_V3_C12_PMON_CTL0           0xEC1
#define MSR_UNC_V3_C12_PMON_CTL1           0xEC2
#define MSR_UNC_V3_C12_PMON_CTL2           0xEC3
#define MSR_UNC_V3_C12_PMON_CTL3           0xEC4
#define MSR_UNC_V3_C12_PMON_CTR0           0xEC8
#define MSR_UNC_V3_C12_PMON_CTR1           0xEC9
#define MSR_UNC_V3_C12_PMON_CTR2           0xECA
#define MSR_UNC_V3_C12_PMON_CTR3           0xECB

#define MSR_UNC_V3_C13_PMON_BOX_CTL        0xED0
#define MSR_UNC_V3_C13_PMON_BOX_STATUS     0xED7
#define MSR_UNC_V3_C13_PMON_BOX_FILTER0    0xED5
#define MSR_UNC_V3_C13_PMON_BOX_FILTER1    0xED6
#define MSR_UNC_V3_C13_PMON_CTL0           0xED1
#define MSR_UNC_V3_C13_PMON_CTL1           0xED2
#define MSR_UNC_V3_C13_PMON_CTL2           0xED3
#define MSR_UNC_V3_C13_PMON_CTL3           0xED4
#define MSR_UNC_V3_C13_PMON_CTR0           0xED8
#define MSR_UNC_V3_C13_PMON_CTR1           0xED9
#define MSR_UNC_V3_C13_PMON_CTR2           0xEDA
#define MSR_UNC_V3_C13_PMON_CTR3           0xEDB

#define MSR_UNC_V3_C14_PMON_BOX_CTL        0xEE0
#define MSR_UNC_V3_C14_PMON_BOX_STATUS     0xEE7
#define MSR_UNC_V3_C14_PMON_BOX_FILTER0    0xEE5
#define MSR_UNC_V3_C14_PMON_BOX_FILTER1    0xEE6
#define MSR_UNC_V3_C14_PMON_CTL0           0xEE1
#define MSR_UNC_V3_C14_PMON_CTL1           0xEE2
#define MSR_UNC_V3_C14_PMON_CTL2           0xEE3
#define MSR_UNC_V3_C14_PMON_CTL3           0xEE4
#define MSR_UNC_V3_C14_PMON_CTR0           0xEE8
#define MSR_UNC_V3_C14_PMON_CTR1           0xEE9
#define MSR_UNC_V3_C14_PMON_CTR2           0xEEA
#define MSR_UNC_V3_C14_PMON_CTR3           0xEEB

#define MSR_UNC_V3_C15_PMON_BOX_CTL        0xEF0
#define MSR_UNC_V3_C15_PMON_BOX_STATUS     0xEF7
#define MSR_UNC_V3_C15_PMON_BOX_FILTER0    0xEF5
#define MSR_UNC_V3_C15_PMON_BOX_FILTER1    0xEF6
#define MSR_UNC_V3_C15_PMON_CTL0           0xEF1
#define MSR_UNC_V3_C15_PMON_CTL1           0xEF2
#define MSR_UNC_V3_C15_PMON_CTL2           0xEF3
#define MSR_UNC_V3_C15_PMON_CTL3           0xEF4
#define MSR_UNC_V3_C15_PMON_CTR0           0xEF8
#define MSR_UNC_V3_C15_PMON_CTR1           0xEF9
#define MSR_UNC_V3_C15_PMON_CTR2           0xEFA
#define MSR_UNC_V3_C15_PMON_CTR3           0xEFB

#define MSR_UNC_V3_C16_PMON_BOX_CTL        0xF00
#define MSR_UNC_V3_C16_PMON_BOX_STATUS     0xF07
#define MSR_UNC_V3_C16_PMON_BOX_FILTER0    0xF05
#define MSR_UNC_V3_C16_PMON_BOX_FILTER1    0xF06
#define MSR_UNC_V3_C16_PMON_CTL0           0xF01
#define MSR_UNC_V3_C16_PMON_CTL1           0xF02
#define MSR_UNC_V3_C16_PMON_CTL2           0xF03
#define MSR_UNC_V3_C16_PMON_CTL3           0xF04
#define MSR_UNC_V3_C16_PMON_CTR0           0xF08
#define MSR_UNC_V3_C16_PMON_CTR1           0xF09
#define MSR_UNC_V3_C16_PMON_CTR2           0xF0A
#define MSR_UNC_V3_C16_PMON_CTR3           0xF0B

#define MSR_UNC_V3_C17_PMON_BOX_CTL        0xF10
#define MSR_UNC_V3_C17_PMON_BOX_STATUS     0xF17
#define MSR_UNC_V3_C17_PMON_BOX_FILTER0    0xF15
#define MSR_UNC_V3_C17_PMON_BOX_FILTER1    0xF16
#define MSR_UNC_V3_C17_PMON_CTL0           0xF11
#define MSR_UNC_V3_C17_PMON_CTL1           0xF12
#define MSR_UNC_V3_C17_PMON_CTL2           0xF13
#define MSR_UNC_V3_C17_PMON_CTL3           0xF14
#define MSR_UNC_V3_C17_PMON_CTR0           0xF18
#define MSR_UNC_V3_C17_PMON_CTR1           0xF19
#define MSR_UNC_V3_C17_PMON_CTR2           0xF1A
#define MSR_UNC_V3_C17_PMON_CTR3           0xF1B

#define MSR_UNC_V3_C18_PMON_BOX_CTL        0xF20
#define MSR_UNC_V3_C18_PMON_BOX_STATUS     0xF27
#define MSR_UNC_V3_C18_PMON_BOX_FILTER0    0xF25
#define MSR_UNC_V3_C18_PMON_BOX_FILTER1    0xF26
#define MSR_UNC_V3_C18_PMON_CTL0           0xF21
#define MSR_UNC_V3_C18_PMON_CTL1           0xF22
#define MSR_UNC_V3_C18_PMON_CTL2           0xF23
#define MSR_UNC_V3_C18_PMON_CTL3           0xF24
#define MSR_UNC_V3_C18_PMON_CTR0           0xF28
#define MSR_UNC_V3_C18_PMON_CTR1           0xF29
#define MSR_UNC_V3_C18_PMON_CTR2           0xF2A
#define MSR_UNC_V3_C18_PMON_CTR3           0xF2B

#define MSR_UNC_V3_C19_PMON_BOX_CTL        0xF30
#define MSR_UNC_V3_C19_PMON_BOX_STATUS     0xF37
#define MSR_UNC_V3_C19_PMON_BOX_FILTER0    0xF35
#define MSR_UNC_V3_C19_PMON_BOX_FILTER1    0xF36
#define MSR_UNC_V3_C19_PMON_CTL0           0xF31
#define MSR_UNC_V3_C19_PMON_CTL1           0xF32
#define MSR_UNC_V3_C19_PMON_CTL2           0xF33
#define MSR_UNC_V3_C19_PMON_CTL3           0xF34
#define MSR_UNC_V3_C19_PMON_CTR0           0xF38
#define MSR_UNC_V3_C19_PMON_CTR1           0xF39
#define MSR_UNC_V3_C19_PMON_CTR2           0xF3A
#define MSR_UNC_V3_C19_PMON_CTR3           0xF3B

#define MSR_UNC_V3_C20_PMON_BOX_CTL        0xF40
#define MSR_UNC_V3_C20_PMON_BOX_STATUS     0xF47
#define MSR_UNC_V3_C20_PMON_BOX_FILTER0    0xF45
#define MSR_UNC_V3_C20_PMON_BOX_FILTER1    0xF46
#define MSR_UNC_V3_C20_PMON_CTL0           0xF41
#define MSR_UNC_V3_C20_PMON_CTL1           0xF42
#define MSR_UNC_V3_C20_PMON_CTL2           0xF43
#define MSR_UNC_V3_C20_PMON_CTL3           0xF44
#define MSR_UNC_V3_C20_PMON_CTR0           0xF48
#define MSR_UNC_V3_C20_PMON_CTR1           0xF49
#define MSR_UNC_V3_C20_PMON_CTR2           0xF4A
#define MSR_UNC_V3_C20_PMON_CTR3           0xF4B

#define MSR_UNC_V3_C21_PMON_BOX_CTL        0xF50
#define MSR_UNC_V3_C21_PMON_BOX_STATUS     0xF57
#define MSR_UNC_V3_C21_PMON_BOX_FILTER0    0xF55
#define MSR_UNC_V3_C21_PMON_BOX_FILTER1    0xF56
#define MSR_UNC_V3_C21_PMON_CTL0           0xF51
#define MSR_UNC_V3_C21_PMON_CTL1           0xF52
#define MSR_UNC_V3_C21_PMON_CTL2           0xF53
#define MSR_UNC_V3_C21_PMON_CTL3           0xF54
#define MSR_UNC_V3_C21_PMON_CTR0           0xF58
#define MSR_UNC_V3_C21_PMON_CTR1           0xF59
#define MSR_UNC_V3_C21_PMON_CTR2           0xF5A
#define MSR_UNC_V3_C21_PMON_CTR3           0xF5B

#define MSR_UNC_V3_C22_PMON_BOX_CTL        0xF60
#define MSR_UNC_V3_C22_PMON_BOX_STATUS     0xF67
#define MSR_UNC_V3_C22_PMON_BOX_FILTER0    0xF65
#define MSR_UNC_V3_C22_PMON_BOX_FILTER1    0xF66
#define MSR_UNC_V3_C22_PMON_CTL0           0xF61
#define MSR_UNC_V3_C22_PMON_CTL1           0xF62
#define MSR_UNC_V3_C22_PMON_CTL2           0xF63
#define MSR_UNC_V3_C22_PMON_CTL3           0xF64
#define MSR_UNC_V3_C22_PMON_CTR0           0xF68
#define MSR_UNC_V3_C22_PMON_CTR1           0xF69
#define MSR_UNC_V3_C22_PMON_CTR2           0xF6A
#define MSR_UNC_V3_C22_PMON_CTR3           0xF6B

#define MSR_UNC_V3_C23_PMON_BOX_CTL        0xF70
#define MSR_UNC_V3_C23_PMON_BOX_STATUS     0xF77
#define MSR_UNC_V3_C23_PMON_BOX_FILTER0    0xF75
#define MSR_UNC_V3_C23_PMON_BOX_FILTER1    0xF76
#define MSR_UNC_V3_C23_PMON_CTL0           0xF71
#define MSR_UNC_V3_C23_PMON_CTL1           0xF72
#define MSR_UNC_V3_C23_PMON_CTL2           0xF73
#define MSR_UNC_V3_C23_PMON_CTL3           0xF74
#define MSR_UNC_V3_C23_PMON_CTR0           0xF78
#define MSR_UNC_V3_C23_PMON_CTR1           0xF79
#define MSR_UNC_V3_C23_PMON_CTR2           0xF7A
#define MSR_UNC_V3_C23_PMON_CTR3           0xF7B

/* Sbox */
#define MSR_UNC_V3_S0_PMON_BOX_CTL         0x720
#define MSR_UNC_V3_S0_PMON_BOX_STATUS      0x725
#define MSR_UNC_V3_S0_PMON_CTL_0           0x721
#define MSR_UNC_V3_S0_PMON_CTL_1           0x722
#define MSR_UNC_V3_S0_PMON_CTL_2           0x723
#define MSR_UNC_V3_S0_PMON_CTL_3           0x724
#define MSR_UNC_V3_S0_PMON_CTR_0           0x726
#define MSR_UNC_V3_S0_PMON_CTR_1           0x727
#define MSR_UNC_V3_S0_PMON_CTR_2           0x728
#define MSR_UNC_V3_S0_PMON_CTR_3           0x729

#define MSR_UNC_V3_S1_PMON_BOX_CTL         0x72A
#define MSR_UNC_V3_S1_PMON_BOX_STATUS      0x72F
#define MSR_UNC_V3_S1_PMON_CTL_0           0x72B
#define MSR_UNC_V3_S1_PMON_CTL_1           0x72C
#define MSR_UNC_V3_S1_PMON_CTL_2           0x72D
#define MSR_UNC_V3_S1_PMON_CTL_3           0x72E
#define MSR_UNC_V3_S1_PMON_CTR_0           0x730
#define MSR_UNC_V3_S1_PMON_CTR_1           0x731
#define MSR_UNC_V3_S1_PMON_CTR_2           0x732
#define MSR_UNC_V3_S1_PMON_CTR_3           0x733

#define MSR_UNC_V3_S2_PMON_BOX_CTL         0x734
#define MSR_UNC_V3_S2_PMON_BOX_STATUS      0x739
#define MSR_UNC_V3_S2_PMON_CTL_0           0x735
#define MSR_UNC_V3_S2_PMON_CTL_1           0x736
#define MSR_UNC_V3_S2_PMON_CTL_2           0x737
#define MSR_UNC_V3_S2_PMON_CTL_3           0x738
#define MSR_UNC_V3_S2_PMON_CTR_0           0x73A
#define MSR_UNC_V3_S2_PMON_CTR_1           0x73B
#define MSR_UNC_V3_S2_PMON_CTR_2           0x73C
#define MSR_UNC_V3_S2_PMON_CTR_3           0x73D

#define MSR_UNC_V3_S3_PMON_BOX_CTL         0x73E
#define MSR_UNC_V3_S3_PMON_BOX_STATUS      0x743
#define MSR_UNC_V3_S3_PMON_CTL_0           0x73F
#define MSR_UNC_V3_S3_PMON_CTL_1           0x740
#define MSR_UNC_V3_S3_PMON_CTL_2           0x741
#define MSR_UNC_V3_S3_PMON_CTL_3           0x742
#define MSR_UNC_V3_S3_PMON_CTR_0           0x744
#define MSR_UNC_V3_S3_PMON_CTR_1           0x745
#define MSR_UNC_V3_S3_PMON_CTR_2           0x746
#define MSR_UNC_V3_S3_PMON_CTR_3           0x747

/* V3 HA similar to V1/V2 */
/* V3 iMC similar to V1/V2 */


/* PCU (Power Control) Performance Monitoring */

#define MSR_UNC_V3_PCU_PMON_CTR0           0x717
#define MSR_UNC_V3_PCU_PMON_CTR1           0x718
#define MSR_UNC_V3_PCU_PMON_CTR2           0x719
#define MSR_UNC_V3_PCU_PMON_CTR3           0x71A
#define MSR_UNC_V3_PCU_PMON_CTL0           0x711
#define MSR_UNC_V3_PCU_PMON_CTL1           0x712
#define MSR_UNC_V3_PCU_PMON_CTL2           0x713
#define MSR_UNC_V3_PCU_PMON_CTL3           0x714
#define MSR_UNC_V3_PCU_PMON_BOX_FILTER     0x715
#define MSR_UNC_V3_PCU_PMON_BOX_CTL        0x710
#define MSR_UNC_V3_PCU_PMON_BOX_STATUS     0x716
#define MSR_UNC_V3_PCU_CC6_CTR             0x3FD
#define MSR_UNC_V3_PCU_CC3_CTR             0x3FC
#define MSR_UNC_V3_PCU_PC2_CTR             0x60D
#define MSR_UNC_V3_PCU_PC3_CTR             0x3F8

/* V3 QPI Box Performance Monitoring, mostly similar to V1/V2 */

#define PCI_UNC_V3_QPI_PMON_BOX_CTL         0xF4
#define PCI_UNC_V3_QPI_PMON_BOX_STATUS      0xF8
#define PCI_UNC_V3_QPI_PMON_CTL_0           0xD8
#define PCI_UNC_V3_QPI_PMON_CTL_1           0xDC
#define PCI_UNC_V3_QPI_PMON_CTL_2           0xE0
#define PCI_UNC_V3_QPI_PMON_CTL_3           0xE4
#define PCI_UNC_V3_QPI_PMON_CTR_0_A         0xA4
#define PCI_UNC_V3_QPI_PMON_CTR_1_A         0xAC
#define PCI_UNC_V3_QPI_PMON_CTR_2_A         0xB4
#define PCI_UNC_V3_QPI_PMON_CTR_3_A         0xBC
#define PCI_UNC_V3_QPI_PMON_CTR_0_B         0xA0
#define PCI_UNC_V3_QPI_PMON_CTR_1_B         0xA8
#define PCI_UNC_V3_QPI_PMON_CTR_2_B         0xB0
#define PCI_UNC_V3_QPI_PMON_CTR_3_B         0xB8
#define PCI_UNC_V3_QPI_PMON_RX_MASK_0          0x238
#define PCI_UNC_V3_QPI_PMON_RX_MASK_1          0x23C
#define PCI_UNC_V3_QPI_PMON_RX_MATCH_0         0x228
#define PCI_UNC_V3_QPI_PMON_RX_MATCH_1         0x22C
#define PCI_UNC_V3_QPI_PMON_TX_MASK_0          0x210
#define PCI_UNC_V3_QPI_PMON_TX_MASK_1          0x214
#define PCI_UNC_V3_QPI_PMON_TX_MATCH_0         0x200
#define PCI_UNC_V3_QPI_PMON_TX_MATCH_1         0x204
#define PCI_UNC_V3_QPI_RATE_STATUS          0xD4
#define PCI_UNC_V3_QPI_LINK_LLR             0xD0
#define PCI_UNC_V3_QPI_LINK_IDLE            0xC8


/* V3 R2PCIE Box Performance Monitoring similar to V1/V2 */

/* V3 R3QPI Box Performance Monitoring similar to V1/V2 */

/* ########################################################## */

/* EX type uncore */
/* U box - System Config Controller */
#define MSR_U_PMON_GLOBAL_CTRL          0xC00
#define MSR_U_PMON_GLOBAL_STATUS        0xC01
#define MSR_U_PMON_GLOBAL_OVF_CTRL      0xC02
#define MSR_U_PMON_GLOBAL_EVNT_SEL      0xC10
#define MSR_U_PMON_GLOBAL_CTR           0xC11
/* B box 0 - Home Agent 0 */
#define MSR_B0_PMON_BOX_CTRL            0xC20
#define MSR_B0_PMON_BOX_STATUS          0xC21
#define MSR_B0_PMON_BOX_OVF_CTRL        0xC22
#define MSR_B0_PMON_EVNT_SEL0           0xC30
#define MSR_B0_PMON_CTR0                0xC31
#define MSR_B0_PMON_EVNT_SEL1           0xC32
#define MSR_B0_PMON_CTR1                0xC33
#define MSR_B0_PMON_EVNT_SEL2           0xC34
#define MSR_B0_PMON_CTR2                0xC35
#define MSR_B0_PMON_EVNT_SEL3           0xC36
#define MSR_B0_PMON_CTR3                0xC37
/* S box 0 - Caching Agent 0 */
#define MSR_S0_PMON_BOX_CTRL            0xC40
#define MSR_S0_PMON_BOX_STATUS          0xC41
#define MSR_S0_PMON_BOX_OVF_CTRL        0xC42
#define MSR_S0_PMON_EVNT_SEL0           0xC50
#define MSR_S0_PMON_CTR0                0xC51
#define MSR_S0_PMON_EVNT_SEL1           0xC52
#define MSR_S0_PMON_CTR1                0xC53
#define MSR_S0_PMON_EVNT_SEL2           0xC54
#define MSR_S0_PMON_CTR2                0xC55
#define MSR_S0_PMON_EVNT_SEL3           0xC56
#define MSR_S0_PMON_CTR3                0xC57
/* B box 1 - Home Agent 1 */
#define MSR_B1_PMON_BOX_CTRL            0xC60
#define MSR_B1_PMON_BOX_STATUS          0xC61
#define MSR_B1_PMON_BOX_OVF_CTRL        0xC62
#define MSR_B1_PMON_EVNT_SEL0           0xC70
#define MSR_B1_PMON_CTR0                0xC71
#define MSR_B1_PMON_EVNT_SEL1           0xC72
#define MSR_B1_PMON_CTR1                0xC73
#define MSR_B1_PMON_EVNT_SEL2           0xC74
#define MSR_B1_PMON_CTR2                0xC75
#define MSR_B1_PMON_EVNT_SEL3           0xC76
#define MSR_B1_PMON_CTR3                0xC77
/* W box  - Power Controller */
#define MSR_W_PMON_BOX_CTRL             0xC80
#define MSR_W_PMON_BOX_STATUS           0xC81
#define MSR_W_PMON_BOX_OVF_CTRL         0xC82
#define MSR_W_PMON_FIXED_CTR_CTL        0x395
#define MSR_W_PMON_FIXED_CTR            0x394
#define MSR_W_PMON_BOX_OVF_CTRL         0xC82
#define MSR_W_PMON_EVNT_SEL0            0xC90
#define MSR_W_PMON_CTR0                 0xC91
#define MSR_W_PMON_EVNT_SEL1            0xC92
#define MSR_W_PMON_CTR1                 0xC93
#define MSR_W_PMON_EVNT_SEL2            0xC94
#define MSR_W_PMON_CTR2                 0xC95
#define MSR_W_PMON_EVNT_SEL3            0xC96
#define MSR_W_PMON_CTR3                 0xC97
/* M box 0 - Memory Controller 0 */
#define MSR_M0_PMON_BOX_CTRL            0xCA0
#define MSR_M0_PMON_BOX_STATUS          0xCA1
#define MSR_M0_PMON_BOX_OVF_CTRL        0xCA2
#define MSR_M0_PMON_TIMESTAMP           0xCA4
#define MSR_M0_PMON_DSP                 0xCA5
#define MSR_M0_PMON_ISS                 0xCA6
#define MSR_M0_PMON_MAP                 0xCA7
#define MSR_M0_PMON_MSC_THR             0xCA8
#define MSR_M0_PMON_PGT                 0xCA9
#define MSR_M0_PMON_PLD                 0xCAA
#define MSR_M0_PMON_ZDP                 0xCAB
#define MSR_M0_PMON_EVNT_SEL0           0xCB0
#define MSR_M0_PMON_CTR0                0xCB1
#define MSR_M0_PMON_EVNT_SEL1           0xCB2
#define MSR_M0_PMON_CTR1                0xCB3
#define MSR_M0_PMON_EVNT_SEL2           0xCB4
#define MSR_M0_PMON_CTR2                0xCB5
#define MSR_M0_PMON_EVNT_SEL3           0xCB6
#define MSR_M0_PMON_CTR3                0xCB7
#define MSR_M0_PMON_EVNT_SEL4           0xCB8
#define MSR_M0_PMON_CTR4                0xCB9
#define MSR_M0_PMON_EVNT_SEL5           0xCBA
#define MSR_M0_PMON_CTR5                0xCBB
/* S box 1 - Caching Agent 1 */
#define MSR_S1_PMON_BOX_CTRL            0xCC0
#define MSR_S1_PMON_BOX_STATUS          0xCC1
#define MSR_S1_PMON_BOX_OVF_CTRL        0xCC2
#define MSR_S1_PMON_EVNT_SEL0           0xCD0
#define MSR_S1_PMON_CTR0                0xCD1
#define MSR_S1_PMON_EVNT_SEL1           0xCD2
#define MSR_S1_PMON_CTR1                0xCD3
#define MSR_S1_PMON_EVNT_SEL2           0xCD4
#define MSR_S1_PMON_CTR2                0xCD5
#define MSR_S1_PMON_EVNT_SEL3           0xCD6
#define MSR_S1_PMON_CTR3                0xCD7
/* M box 1 - Memory Controller 1 */
#define MSR_M1_PMON_BOX_CTRL            0xCE0
#define MSR_M1_PMON_BOX_STATUS          0xCE1
#define MSR_M1_PMON_BOX_OVF_CTRL        0xCE2
#define MSR_M1_PMON_TIMESTAMP           0xCE4
#define MSR_M1_PMON_DSP                 0xCE5
#define MSR_M1_PMON_ISS                 0xCE6
#define MSR_M1_PMON_MAP                 0xCE7
#define MSR_M1_PMON_MSC_THR             0xCE8
#define MSR_M1_PMON_PGT                 0xCE9
#define MSR_M1_PMON_PLD                 0xCEA
#define MSR_M1_PMON_ZDP                 0xCEB
#define MSR_M1_PMON_EVNT_SEL0           0xCF0
#define MSR_M1_PMON_CTR0                0xCF1
#define MSR_M1_PMON_EVNT_SEL1           0xCF2
#define MSR_M1_PMON_CTR1                0xCF3
#define MSR_M1_PMON_EVNT_SEL2           0xCF4
#define MSR_M1_PMON_CTR2                0xCF5
#define MSR_M1_PMON_EVNT_SEL3           0xCF6
#define MSR_M1_PMON_CTR3                0xCB7
#define MSR_M1_PMON_EVNT_SEL4           0xCF8
#define MSR_M1_PMON_CTR4                0xCF9
#define MSR_M1_PMON_EVNT_SEL5           0xCFA
#define MSR_M1_PMON_CTR5                0xCFB
/* C box 0 - Coherence Engine core 0 */
#define MSR_C0_PMON_BOX_CTRL            0xD00
#define MSR_C0_PMON_BOX_STATUS          0xD01
#define MSR_C0_PMON_BOX_OVF_CTRL        0xD02
#define MSR_C0_PMON_EVNT_SEL0           0xD10
#define MSR_C0_PMON_CTR0                0xD11
#define MSR_C0_PMON_EVNT_SEL1           0xD12
#define MSR_C0_PMON_CTR1                0xD13
#define MSR_C0_PMON_EVNT_SEL2           0xD14
#define MSR_C0_PMON_CTR2                0xD15
#define MSR_C0_PMON_EVNT_SEL3           0xD16
#define MSR_C0_PMON_CTR3                0xD17
#define MSR_C0_PMON_EVNT_SEL4           0xD18
#define MSR_C0_PMON_CTR4                0xD19
#define MSR_C0_PMON_EVNT_SEL5           0xD1A
#define MSR_C0_PMON_CTR5                0xD1B
/* C box 4 - Coherence Engine core 4 */
#define MSR_C4_PMON_BOX_CTRL            0xD20
#define MSR_C4_PMON_BOX_STATUS          0xD21
#define MSR_C4_PMON_BOX_OVF_CTRL        0xD22
#define MSR_C4_PMON_EVNT_SEL0           0xD30
#define MSR_C4_PMON_CTR0                0xD31
#define MSR_C4_PMON_EVNT_SEL1           0xD32
#define MSR_C4_PMON_CTR1                0xD33
#define MSR_C4_PMON_EVNT_SEL2           0xD34
#define MSR_C4_PMON_CTR2                0xD35
#define MSR_C4_PMON_EVNT_SEL3           0xD36
#define MSR_C4_PMON_CTR3                0xD37
#define MSR_C4_PMON_EVNT_SEL4           0xD38
#define MSR_C4_PMON_CTR4                0xD39
#define MSR_C4_PMON_EVNT_SEL5           0xD3A
#define MSR_C4_PMON_CTR5                0xD3B
/* C box 2 - Coherence Engine core 2 */
#define MSR_C2_PMON_BOX_CTRL            0xD40
#define MSR_C2_PMON_BOX_STATUS          0xD41
#define MSR_C2_PMON_BOX_OVF_CTRL        0xD42
#define MSR_C2_PMON_EVNT_SEL0           0xD50
#define MSR_C2_PMON_CTR0                0xD51
#define MSR_C2_PMON_EVNT_SEL1           0xD52
#define MSR_C2_PMON_CTR1                0xD53
#define MSR_C2_PMON_EVNT_SEL2           0xD54
#define MSR_C2_PMON_CTR2                0xD55
#define MSR_C2_PMON_EVNT_SEL3           0xD56
#define MSR_C2_PMON_CTR3                0xD57
#define MSR_C2_PMON_EVNT_SEL4           0xD58
#define MSR_C2_PMON_CTR4                0xD59
#define MSR_C2_PMON_EVNT_SEL5           0xD5A
#define MSR_C2_PMON_CTR5                0xD5B
/* C box 6 - Coherence Engine core 6 */
#define MSR_C6_PMON_BOX_CTRL            0xD60
#define MSR_C6_PMON_BOX_STATUS          0xD61
#define MSR_C6_PMON_BOX_OVF_CTRL        0xD62
#define MSR_C6_PMON_EVNT_SEL0           0xD70
#define MSR_C6_PMON_CTR0                0xD71
#define MSR_C6_PMON_EVNT_SEL1           0xD72
#define MSR_C6_PMON_CTR1                0xD73
#define MSR_C6_PMON_EVNT_SEL2           0xD74
#define MSR_C6_PMON_CTR2                0xD75
#define MSR_C6_PMON_EVNT_SEL3           0xD76
#define MSR_C6_PMON_CTR3                0xD77
#define MSR_C6_PMON_EVNT_SEL4           0xD78
#define MSR_C6_PMON_CTR4                0xD79
#define MSR_C6_PMON_EVNT_SEL5           0xD7A
#define MSR_C6_PMON_CTR5                0xD7B
/* C box 1 - Coherence Engine core 1 */
#define MSR_C1_PMON_BOX_CTRL            0xD80
#define MSR_C1_PMON_BOX_STATUS          0xD81
#define MSR_C1_PMON_BOX_OVF_CTRL        0xD82
#define MSR_C1_PMON_EVNT_SEL0           0xD90
#define MSR_C1_PMON_CTR0                0xD91
#define MSR_C1_PMON_EVNT_SEL1           0xD92
#define MSR_C1_PMON_CTR1                0xD93
#define MSR_C1_PMON_EVNT_SEL2           0xD94
#define MSR_C1_PMON_CTR2                0xD95
#define MSR_C1_PMON_EVNT_SEL3           0xD96
#define MSR_C1_PMON_CTR3                0xD97
#define MSR_C1_PMON_EVNT_SEL4           0xD98
#define MSR_C1_PMON_CTR4                0xD99
#define MSR_C1_PMON_EVNT_SEL5           0xD9A
#define MSR_C1_PMON_CTR5                0xD9B
/* C box 5 - Coherence Engine core 5 */
#define MSR_C5_PMON_BOX_CTRL            0xDA0
#define MSR_C5_PMON_BOX_STATUS          0xDA1
#define MSR_C5_PMON_BOX_OVF_CTRL        0xDA2
#define MSR_C5_PMON_EVNT_SEL0           0xDB0
#define MSR_C5_PMON_CTR0                0xDB1
#define MSR_C5_PMON_EVNT_SEL1           0xDB2
#define MSR_C5_PMON_CTR1                0xDB3
#define MSR_C5_PMON_EVNT_SEL2           0xDB4
#define MSR_C5_PMON_CTR2                0xDB5
#define MSR_C5_PMON_EVNT_SEL3           0xDB6
#define MSR_C5_PMON_CTR3                0xDB7
#define MSR_C5_PMON_EVNT_SEL4           0xDB8
#define MSR_C5_PMON_CTR4                0xDB9
#define MSR_C5_PMON_EVNT_SEL5           0xDBA
#define MSR_C5_PMON_CTR5                0xDBB
/* C box 3 - Coherence Engine core 3 */
#define MSR_C3_PMON_BOX_CTRL            0xDC0
#define MSR_C3_PMON_BOX_STATUS          0xDC1
#define MSR_C3_PMON_BOX_OVF_CTRL        0xDC2
#define MSR_C3_PMON_EVNT_SEL0           0xDD0
#define MSR_C3_PMON_CTR0                0xDD1
#define MSR_C3_PMON_EVNT_SEL1           0xDD2
#define MSR_C3_PMON_CTR1                0xDD3
#define MSR_C3_PMON_EVNT_SEL2           0xDD4
#define MSR_C3_PMON_CTR2                0xDD5
#define MSR_C3_PMON_EVNT_SEL3           0xDD6
#define MSR_C3_PMON_CTR3                0xDD7
#define MSR_C3_PMON_EVNT_SEL4           0xDD8
#define MSR_C3_PMON_CTR4                0xDD9
#define MSR_C3_PMON_EVNT_SEL5           0xDDA
#define MSR_C3_PMON_CTR5                0xDDB
/* C box 7 - Coherence Engine core 7 */
#define MSR_C7_PMON_BOX_CTRL            0xDE0
#define MSR_C7_PMON_BOX_STATUS          0xDE1
#define MSR_C7_PMON_BOX_OVF_CTRL        0xDE2
#define MSR_C7_PMON_EVNT_SEL0           0xDF0
#define MSR_C7_PMON_CTR0                0xDF1
#define MSR_C7_PMON_EVNT_SEL1           0xDF2
#define MSR_C7_PMON_CTR1                0xDF3
#define MSR_C7_PMON_EVNT_SEL2           0xDF4
#define MSR_C7_PMON_CTR2                0xDF5
#define MSR_C7_PMON_EVNT_SEL3           0xDF6
#define MSR_C7_PMON_CTR3                0xDF7
#define MSR_C7_PMON_EVNT_SEL4           0xDF8
#define MSR_C7_PMON_CTR4                0xDF9
#define MSR_C7_PMON_EVNT_SEL5           0xDFA
#define MSR_C7_PMON_CTR5                0xDFB
/* C box 8 - Coherence Engine core 8 */
#define MSR_C8_PMON_BOX_CTRL            0xF40
#define MSR_C8_PMON_BOX_STATUS          0xF41
#define MSR_C8_PMON_BOX_OVF_CTRL        0xF42
#define MSR_C8_PMON_EVNT_SEL0           0xF50
#define MSR_C8_PMON_CTR0                0xF51
#define MSR_C8_PMON_EVNT_SEL1           0xF52
#define MSR_C8_PMON_CTR1                0xF53
#define MSR_C8_PMON_EVNT_SEL2           0xF54
#define MSR_C8_PMON_CTR2                0xF55
#define MSR_C8_PMON_EVNT_SEL3           0xF56
#define MSR_C8_PMON_CTR3                0xF57
#define MSR_C8_PMON_EVNT_SEL4           0xF58
#define MSR_C8_PMON_CTR4                0xF59
#define MSR_C8_PMON_EVNT_SEL5           0xF5A
#define MSR_C8_PMON_CTR5                0xF5B
/* C box 9 - Coherence Engine core 9 */
#define MSR_C9_PMON_BOX_CTRL            0xFC0
#define MSR_C9_PMON_BOX_STATUS          0xFC1
#define MSR_C9_PMON_BOX_OVF_CTRL        0xFC2
#define MSR_C9_PMON_EVNT_SEL0           0xFD0
#define MSR_C9_PMON_CTR0                0xFD1
#define MSR_C9_PMON_EVNT_SEL1           0xFD2
#define MSR_C9_PMON_CTR1                0xFD3
#define MSR_C9_PMON_EVNT_SEL2           0xFD4
#define MSR_C9_PMON_CTR2                0xFD5
#define MSR_C9_PMON_EVNT_SEL3           0xFD6
#define MSR_C9_PMON_CTR3                0xFD7
#define MSR_C9_PMON_EVNT_SEL4           0xFD8
#define MSR_C9_PMON_CTR4                0xFD9
#define MSR_C9_PMON_EVNT_SEL5           0xFDA
#define MSR_C9_PMON_CTR5                0xFDB
/* R box 0 - Router 0 */
#define MSR_R0_PMON_BOX_CTRL            0xE00
#define MSR_R0_PMON_BOX_STATUS          0xE01
#define MSR_R0_PMON_BOX_OVF_CTRL        0xE02
#define MSR_R0_PMON_IPERF0_P0           0xE04
#define MSR_R0_PMON_IPERF0_P1           0xE05
#define MSR_R0_PMON_IPERF0_P2           0xE06
#define MSR_R0_PMON_IPERF0_P3           0xE07
#define MSR_R0_PMON_IPERF1_P0           0xE24
#define MSR_R0_PMON_IPERF1_P1           0xE25
#define MSR_R0_PMON_IPERF1_P2           0xE26
#define MSR_R0_PMON_IPERF1_P3           0xE27
#define MSR_R0_PMON_QLX_P0              0xE0C
#define MSR_R0_PMON_QLX_P1              0xE0D
#define MSR_R0_PMON_QLX_P2              0xE0E
#define MSR_R0_PMON_QLX_P3              0xE0F
#define MSR_R0_PMON_EVNT_SEL0           0xE10
#define MSR_R0_PMON_CTR0                0xE11
#define MSR_R0_PMON_EVNT_SEL1           0xE12
#define MSR_R0_PMON_CTR1                0xE13
#define MSR_R0_PMON_EVNT_SEL2           0xE14
#define MSR_R0_PMON_CTR2                0xE15
#define MSR_R0_PMON_EVNT_SEL3           0xE16
#define MSR_R0_PMON_CTR3                0xE17
#define MSR_R0_PMON_EVNT_SEL4           0xE18
#define MSR_R0_PMON_CTR4                0xE19
#define MSR_R0_PMON_EVNT_SEL5           0xE1A
#define MSR_R0_PMON_CTR5                0xE1B
#define MSR_R0_PMON_EVNT_SEL6           0xE1C
#define MSR_R0_PMON_CTR6                0xE1D
#define MSR_R0_PMON_EVNT_SEL7           0xE1E
#define MSR_R0_PMON_CTR7                0xE1F
/* R box 1 - Router 1 */
#define MSR_R1_PMON_BOX_CTRL            0xE20
#define MSR_R1_PMON_BOX_STATUS          0xE21
#define MSR_R1_PMON_BOX_OVF_CTRL        0xE22
#define MSR_R1_PMON_IPERF0_P0           0xE08
#define MSR_R1_PMON_IPERF0_P1           0xE09
#define MSR_R1_PMON_IPERF0_P2           0xE0A
#define MSR_R1_PMON_IPERF0_P3           0xE0B
#define MSR_R1_PMON_IPERF1_P0           0xE28
#define MSR_R1_PMON_IPERF1_P1           0xE29
#define MSR_R1_PMON_IPERF1_P2           0xE2A
#define MSR_R1_PMON_IPERF1_P3           0xE2B
#define MSR_R1_PMON_QLX_P0              0xE2C
#define MSR_R1_PMON_QLX_P1              0xE2D
#define MSR_R1_PMON_QLX_P2              0xE2E
#define MSR_R1_PMON_QLX_P3              0xE2F
#define MSR_R1_PMON_EVNT_SEL8           0xE30
#define MSR_R1_PMON_CTR8                0xE31
#define MSR_R1_PMON_EVNT_SEL9           0xE32
#define MSR_R1_PMON_CTR9                0xE33
#define MSR_R1_PMON_EVNT_SEL10          0xE34
#define MSR_R1_PMON_CTR10               0xE35
#define MSR_R1_PMON_EVNT_SEL11          0xE36
#define MSR_R1_PMON_CTR11               0xE37
#define MSR_R1_PMON_EVNT_SEL12          0xE38
#define MSR_R1_PMON_CTR12               0xE39
#define MSR_R1_PMON_EVNT_SEL13          0xE3A
#define MSR_R1_PMON_CTR13               0xE3B
#define MSR_R1_PMON_EVNT_SEL14          0xE3C
#define MSR_R1_PMON_CTR14               0xE3D
#define MSR_R1_PMON_EVNT_SEL15          0xE3E
#define MSR_R1_PMON_CTR15               0xE3F
/* Match/Mask MSRs */
#define MSR_B0_PMON_MATCH               0xE45
#define MSR_B0_PMON_MASK                0xE46
#define MSR_S0_PMON_MM_CFG              0xE49
#define MSR_S0_PMON_MATCH               0xE49
#define MSR_S0_PMON_MASK                0xE4A
#define MSR_B1_PMON_MATCH               0xE4D
#define MSR_B1_PMON_MASK                0xE4E
#define MSR_M0_PMON_MM_CONFIG           0xE54
#define MSR_M0_PMON_ADDR_MATCH          0xE55
#define MSR_M0_PMON_ADDR_MASK           0xE56
#define MSR_S1_PMON_MM_CFG              0xE58
#define MSR_S1_PMON_MATCH               0xE59
#define MSR_S1_PMON_MASK                0xE5A
#define MSR_M1_PMON_MM_CONFIG           0xE5C
#define MSR_M1_PMON_ADDR_MATCH          0xE5D
#define MSR_M1_PMON_ADDR_MASK           0xE5E
/* Power interfaces, RAPL */
#define MSR_RAPL_POWER_UNIT             0x606
#define MSR_PKG_RAPL_POWER_LIMIT        0x610
#define MSR_PKG_ENERGY_STATUS           0x611
#define MSR_PKG_PERF_STATUS             0x613
#define MSR_PKG_POWER_INFO              0x614
#define MSR_PP0_RAPL_POWER_LIMIT        0x638
#define MSR_PP0_ENERGY_STATUS           0x639
#define MSR_PP0_ENERGY_POLICY           0x63A
#define MSR_PP0_PERF_STATUS             0x63B
#define MSR_PP1_RAPL_POWER_LIMIT        0x640
#define MSR_PP1_ENERGY_STATUS           0x641
#define MSR_PP1_ENERGY_POLICY           0x642
#define MSR_DRAM_RAPL_POWER_LIMIT       0x618
#define MSR_DRAM_ENERGY_STATUS          0x619
#define MSR_DRAM_PERF_STATUS            0x61B
#define MSR_DRAM_POWER_INFO             0x61C
#define MSR_PLATFORM_ENERGY_STATUS      0x64D
#define MSR_PLATFORM_POWER_LIMIT        0x65C

/* Intel Silvermont's RAPL registers */
#define MSR_PKG_POWER_INFO_SILVERMONT   0x66E

/* TM/TM2 interface */
#define IA32_THERM_STATUS               0x19C
#define IA32_PACKAGE_THERM_STATUS       0x1B1
#define MSR_TEMPERATURE_TARGET          0x1A2

/* Turbo Boost Interface */
#define MSR_IA32_MISC_ENABLE            0x1A0
#define MSR_PREFETCH_ENABLE             0x1A4
#define MSR_PLATFORM_INFO               0x0CE
#define MSR_TURBO_POWER_CURRENT_LIMIT   0x1AC
#define MSR_TURBO_RATIO_LIMIT           0x1AD
#define MSR_TURBO_RATIO_LIMIT1          0x1AE
#define MSR_TURBO_RATIO_LIMIT2          0x1AF
#define MSR_TURBO_RATIO_LIMIT3          0x1AC

/* MISC Intel register */
#define MSR_MPERF                       0xE7
#define MSR_APERF                       0xE8
#define MSR_PPERF                       0x64E
#define MSR_WEIGHTED_CORE_C0            0x658
#define MSR_ANY_CORE_C0                 0x659
#define MSR_ANY_GFXE_C0                 0x65A
#define MSR_CORE_GFXE_OVERLAP_C0        0x65B
#define MSR_UNCORE_FREQ                 0x620
#define MSR_FSB_FREQ                    0xCD
#define MSR_ENERGY_PERF_BIAS            0x1B0
#define MSR_ALT_PEBS                    0x39C
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

/* 0x15 Interlagos */

#define MSR_AMD15_PERFEVTSEL0           0xC0010200
#define MSR_AMD15_PERFEVTSEL1           0xC0010202
#define MSR_AMD15_PERFEVTSEL2           0xC0010204
#define MSR_AMD15_PERFEVTSEL3           0xC0010206
#define MSR_AMD15_PERFEVTSEL4           0xC0010208
#define MSR_AMD15_PERFEVTSEL5           0xC001020A

#define MSR_AMD15_PMC0                  0xC0010201
#define MSR_AMD15_PMC1                  0xC0010203
#define MSR_AMD15_PMC2                  0xC0010205
#define MSR_AMD15_PMC3                  0xC0010207
#define MSR_AMD15_PMC4                  0xC0010209
#define MSR_AMD15_PMC5                  0xC001020B

#define MSR_AMD15_NB_PERFEVTSEL0         0xC0010240
#define MSR_AMD15_NB_PERFEVTSEL1         0xC0010242
#define MSR_AMD15_NB_PERFEVTSEL2         0xC0010244
#define MSR_AMD15_NB_PERFEVTSEL3         0xC0010246

#define MSR_AMD15_NB_PMC0               0xC0010241
#define MSR_AMD15_NB_PMC1               0xC0010243
#define MSR_AMD15_NB_PMC2               0xC0010245
#define MSR_AMD15_NB_PMC3               0xC0010247

/* AMD 0x16 */
#define MSR_AMD16_PERFEVTSEL0           0xC0010000
#define MSR_AMD16_PERFEVTSEL1           0xC0010001
#define MSR_AMD16_PERFEVTSEL2           0xC0010002
#define MSR_AMD16_PERFEVTSEL3           0xC0010003
#define MSR_AMD16_PMC0                  0xC0010004
#define MSR_AMD16_PMC1                  0xC0010005
#define MSR_AMD16_PMC2                  0xC0010006
#define MSR_AMD16_PMC3                  0xC0010007

#define MSR_AMD16_L2_PERFEVTSEL0        0xC0010230
#define MSR_AMD16_L2_PERFEVTSEL1        0xC0010232
#define MSR_AMD16_L2_PERFEVTSEL2        0xC0010234
#define MSR_AMD16_L2_PERFEVTSEL3        0xC0010236
#define MSR_AMD16_L2_PMC0               0xC0010231
#define MSR_AMD16_L2_PMC1               0xC0010233
#define MSR_AMD16_L2_PMC2               0xC0010235
#define MSR_AMD16_L2_PMC3               0xC0010237

#define MSR_AMD16_NB_PERFEVTSEL0        0xC0010240
#define MSR_AMD16_NB_PERFEVTSEL1        0xC0010242
#define MSR_AMD16_NB_PERFEVTSEL2        0xC0010244
#define MSR_AMD16_NB_PERFEVTSEL3        0xC0010246
#define MSR_AMD16_NB_PMC0               0xC0010241
#define MSR_AMD16_NB_PMC1               0xC0010243
#define MSR_AMD16_NB_PMC2               0xC0010245
#define MSR_AMD16_NB_PMC3               0xC0010247

/* IBM POWER8 */
#define IBM_MMCR0			0x0
#define IBM_MMCR1			0x1
#define IBM_MMCRA			0x2
#define IBM_MMCRC			0x3
#define IBM_PMC0			0x4
#define IBM_PMC1			0x5
#define IBM_PMC2			0x6
#define IBM_PMC3			0x7
#define IBM_PMC4			0x8
#define IBM_PMC5			0x9
#define IBM_PIR				0xA
#define IBM_PURR			0xB
#define IBM_SPURR			0xC
#define IBM_DSCR			0xD

#endif /* REGISTERS_H */
