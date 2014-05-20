/*
 * =======================================================================================
 *
 *      Filename:  perfmon_westmereEX_counters.h
 *
 *      Description: Counter Header File of perfmon module for Westmere EX.
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

#define NUM_COUNTERS_CORE_WESTMEREEX 7
#define NUM_COUNTERS_UNCORE_WESTMEREEX 48
#define NUM_COUNTERS_WESTMEREEX 48

static PerfmonCounterMap westmereEX_counter_map[NUM_COUNTERS_WESTMEREEX] = {
    /* Fixed Counters: instructions retired, cycles unhalted core */
    {"FIXC0", PMC0, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR0, 0, 0},
    {"FIXC1", PMC1, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR1, 0, 0},
    {"FIXC2", PMC2, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR2, 0, 0},
    /* PMC Counters: 4 48bit wide */
    {"PMC0", PMC3, PMC, MSR_PERFEVTSEL0, MSR_PMC0, 0, 0},
    {"PMC1", PMC4, PMC, MSR_PERFEVTSEL1, MSR_PMC1, 0, 0},
    {"PMC2", PMC5, PMC, MSR_PERFEVTSEL2, MSR_PMC2, 0, 0},
    {"PMC3", PMC6, PMC, MSR_PERFEVTSEL3, MSR_PMC3, 0, 0},
    /* MBOX */
    {"MBOX0C0",PMC7, MBOX0, MSR_M0_PMON_EVNT_SEL0, MSR_M0_PMON_CTR0, 0, 0},
    {"MBOX0C1",PMC8, MBOX0, MSR_M0_PMON_EVNT_SEL1, MSR_M0_PMON_CTR1, 0, 0},
    {"MBOX0C2",PMC9, MBOX0, MSR_M0_PMON_EVNT_SEL2, MSR_M0_PMON_CTR2, 0, 0},
    {"MBOX0C3",PMC10, MBOX0, MSR_M0_PMON_EVNT_SEL3, MSR_M0_PMON_CTR3, 0, 0},
    {"MBOX0C4",PMC11, MBOX0, MSR_M0_PMON_EVNT_SEL4, MSR_M0_PMON_CTR4, 0, 0},
    {"MBOX0C5",PMC12, MBOX0, MSR_M0_PMON_EVNT_SEL5, MSR_M0_PMON_CTR5, 0, 0},
    {"MBOX1C0",PMC13, MBOX1, MSR_M1_PMON_EVNT_SEL0, MSR_M1_PMON_CTR0, 0, 0},
    {"MBOX1C1",PMC14, MBOX1, MSR_M1_PMON_EVNT_SEL1, MSR_M1_PMON_CTR1, 0, 0},
    {"MBOX1C2",PMC15, MBOX1, MSR_M1_PMON_EVNT_SEL2, MSR_M1_PMON_CTR2, 0, 0},
    {"MBOX1C3",PMC16, MBOX1, MSR_M1_PMON_EVNT_SEL3, MSR_M1_PMON_CTR3, 0, 0},
    {"MBOX1C4",PMC17, MBOX1, MSR_M1_PMON_EVNT_SEL4, MSR_M1_PMON_CTR4, 0, 0},
    {"MBOX1C5",PMC18, MBOX1, MSR_M1_PMON_EVNT_SEL5, MSR_M1_PMON_CTR5, 0, 0},
    /* BBOX */
    {"BBOX0C0",PMC19, BBOX0, MSR_B0_PMON_EVNT_SEL0, MSR_B0_PMON_CTR0, 0, 0},
    {"BBOX0C1",PMC20, BBOX0, MSR_B0_PMON_EVNT_SEL1, MSR_B0_PMON_CTR1, 0, 0},
    {"BBOX0C2",PMC21, BBOX0, MSR_B0_PMON_EVNT_SEL2, MSR_B0_PMON_CTR2, 0, 0},
    {"BBOX0C3",PMC22, BBOX0, MSR_B0_PMON_EVNT_SEL3, MSR_B0_PMON_CTR3, 0, 0},
    {"BBOX1C0",PMC23, BBOX1, MSR_B1_PMON_EVNT_SEL0, MSR_B1_PMON_CTR0, 0, 0},
    {"BBOX1C1",PMC24, BBOX1, MSR_B1_PMON_EVNT_SEL1, MSR_B1_PMON_CTR1, 0, 0},
    {"BBOX1C2",PMC25, BBOX1, MSR_B1_PMON_EVNT_SEL2, MSR_B1_PMON_CTR2, 0, 0},
    {"BBOX1C3",PMC26, BBOX1, MSR_B1_PMON_EVNT_SEL3, MSR_B1_PMON_CTR3, 0, 0},
    /* RBOX */
    {"RBOX0C0",PMC27, RBOX0, MSR_R0_PMON_EVNT_SEL0, MSR_R0_PMON_CTR0, 0, 0},
    {"RBOX0C1",PMC28, RBOX0, MSR_R0_PMON_EVNT_SEL1, MSR_R0_PMON_CTR1, 0, 0},
    {"RBOX0C2",PMC29, RBOX0, MSR_R0_PMON_EVNT_SEL2, MSR_R0_PMON_CTR2, 0, 0},
    {"RBOX0C3",PMC30, RBOX0, MSR_R0_PMON_EVNT_SEL3, MSR_R0_PMON_CTR3, 0, 0},
    {"RBOX0C4",PMC31, RBOX0, MSR_R0_PMON_EVNT_SEL4, MSR_R0_PMON_CTR4, 0, 0},
    {"RBOX0C5",PMC32, RBOX0, MSR_R0_PMON_EVNT_SEL5, MSR_R0_PMON_CTR5, 0, 0},
    {"RBOX0C6",PMC33, RBOX0, MSR_R0_PMON_EVNT_SEL6, MSR_R0_PMON_CTR6, 0, 0},
    {"RBOX0C7",PMC34, RBOX0, MSR_R0_PMON_EVNT_SEL7, MSR_R0_PMON_CTR7, 0, 0},
    {"RBOX1C0",PMC35, RBOX1, MSR_R1_PMON_EVNT_SEL8, MSR_R1_PMON_CTR8, 0, 0},
    {"RBOX1C1",PMC36, RBOX1, MSR_R1_PMON_EVNT_SEL9, MSR_R1_PMON_CTR9, 0, 0},
    {"RBOX1C2",PMC37, RBOX1, MSR_R1_PMON_EVNT_SEL10, MSR_R1_PMON_CTR10, 0, 0},
    {"RBOX1C3",PMC38, RBOX1, MSR_R1_PMON_EVNT_SEL11, MSR_R1_PMON_CTR11, 0, 0},
    {"RBOX1C4",PMC39, RBOX1, MSR_R1_PMON_EVNT_SEL12, MSR_R1_PMON_CTR12, 0, 0},
    {"RBOX1C5",PMC40, RBOX1, MSR_R1_PMON_EVNT_SEL13, MSR_R1_PMON_CTR13, 0, 0},
    {"RBOX1C6",PMC41, RBOX1, MSR_R1_PMON_EVNT_SEL14, MSR_R1_PMON_CTR14, 0, 0},
    {"RBOX1C7",PMC42, RBOX1, MSR_R1_PMON_EVNT_SEL15, MSR_R1_PMON_CTR15, 0, 0},
    /* WBOX */
    {"WBOX0",PMC43, WBOX, MSR_W_PMON_EVNT_SEL0, MSR_W_PMON_CTR0, 0, 0},
    {"WBOX1",PMC44, WBOX, MSR_W_PMON_EVNT_SEL1, MSR_W_PMON_CTR1, 0, 0},
    {"WBOX2",PMC45, WBOX, MSR_W_PMON_EVNT_SEL2, MSR_W_PMON_CTR2, 0, 0},
    {"WBOX3",PMC46, WBOX, MSR_W_PMON_EVNT_SEL3, MSR_W_PMON_CTR3, 0, 0},
    {"WBOX4",PMC47, WBOX, MSR_W_PMON_FIXED_CTR_CTL, MSR_W_PMON_FIXED_CTR, 0, 0}
};

