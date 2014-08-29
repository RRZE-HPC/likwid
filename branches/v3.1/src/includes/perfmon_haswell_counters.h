/*
 * =======================================================================================
 *
 *      Filename:  perfmon_haswell_counters.h
 *
 *      Description:  Counter Header File of perfmon module for Haswell.
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

#define NUM_COUNTERS_HASWELL 23
#define NUM_COUNTERS_UNCORE_HASWELL 15
#define NUM_COUNTERS_CORE_HASWELL 8

static PerfmonCounterMap haswell_counter_map[NUM_COUNTERS_HASWELL] = {
    /* Fixed Counters: instructions retired, cycles unhalted core */
    {"FIXC0", PMC0, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR0, 0, 0},
    {"FIXC1", PMC1, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR1, 0, 0},
    {"FIXC2", PMC2, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR2, 0, 0},
    /* PMC Counters: 4 48bit wide */
    {"PMC0", PMC3, PMC, MSR_PERFEVTSEL0, MSR_PMC0, 0, 0},
    {"PMC1", PMC4, PMC, MSR_PERFEVTSEL1, MSR_PMC1, 0, 0},
    {"PMC2", PMC5, PMC, MSR_PERFEVTSEL2, MSR_PMC2, 0, 0},
    {"PMC3", PMC6, PMC, MSR_PERFEVTSEL3, MSR_PMC3, 0, 0},
    /* Temperature Sensor*/
    {"TMP0", PMC7, THERMAL, 0, 0, 0, 0},
    /* RAPL counters */
    {"PWR0", PMC8, POWER, 0, MSR_PKG_ENERGY_STATUS, 0, 0},
    {"PWR1", PMC9, POWER, 0, MSR_PP0_ENERGY_STATUS, 0, 0},
    {"PWR2", PMC10, POWER, 0, MSR_PP1_ENERGY_STATUS,  0, 0},
    {"PWR3", PMC11, POWER, 0, MSR_DRAM_ENERGY_STATUS,  0, 0},
    {"CBOX0C0", PMC12, CBOX0, MSR_UNC_CBO_0_PERFEVTSEL0, MSR_UNC_CBO_0_CTR0, 0, 0},
    {"CBOX0C1", PMC13, CBOX0, MSR_UNC_CBO_0_PERFEVTSEL1, MSR_UNC_CBO_0_CTR1, 0, 0},
    {"CBOX1C0", PMC14, CBOX1, MSR_UNC_CBO_1_PERFEVTSEL0, MSR_UNC_CBO_1_CTR0, 0, 0},
    {"CBOX1C1", PMC15, CBOX1, MSR_UNC_CBO_1_PERFEVTSEL1, MSR_UNC_CBO_1_CTR1, 0, 0},
    {"CBOX2C0", PMC16, CBOX2, MSR_UNC_CBO_2_PERFEVTSEL0, MSR_UNC_CBO_2_CTR0, 0, 0},
    {"CBOX2C1", PMC17, CBOX2, MSR_UNC_CBO_2_PERFEVTSEL1, MSR_UNC_CBO_2_CTR1, 0, 0},
    {"CBOX3C0", PMC18, CBOX3, MSR_UNC_CBO_3_PERFEVTSEL0, MSR_UNC_CBO_3_CTR0, 0, 0},
    {"CBOX3C1", PMC19, CBOX3, MSR_UNC_CBO_3_PERFEVTSEL1, MSR_UNC_CBO_3_CTR1, 0, 0},
    {"UBOX0", PMC20, UBOX, MSR_UNC_ARB_PERFEVTSEL0, MSR_UNC_ARB_CTR0, 0, 0},
    {"UBOX1", PMC21, UBOX, MSR_UNC_ARB_PERFEVTSEL1, MSR_UNC_ARB_CTR1, 0, 0},
    {"UBOXFIX", PMC22, UBOX, MSR_UNC_PERF_FIXED_CTRL, MSR_UNC_PERF_FIXED_CTR, 0, 0},
};

