/*
 * =======================================================================================
 *
 *      Filename:  perfmon_nehalem_counters.h
 *
 *      Description:  Counter Header File of perfmon module for Nehalem.
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

#define NUM_COUNTERS_CORE_NEHALEM 7
#define NUM_COUNTERS_UNCORE_NEHALEM 15
#define NUM_COUNTERS_NEHALEM 15

static PerfmonCounterMap nehalem_counter_map[NUM_COUNTERS_NEHALEM] = {
    /* Fixed Counters: instructions retired, cycles unhalted core */
    {"FIXC0",PMC0, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR0, 0, 0},
    {"FIXC1",PMC1, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR1, 0, 0},
    {"FIXC2",PMC2, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR2, 0, 0},
    /* PMC Counters: 4 48bit wide */
    {"PMC0",PMC3, PMC, MSR_PERFEVTSEL0, MSR_PMC0, 0, 0},
    {"PMC1",PMC4, PMC, MSR_PERFEVTSEL1, MSR_PMC1, 0, 0},
    {"PMC2",PMC5, PMC, MSR_PERFEVTSEL2, MSR_PMC2, 0, 0},
    {"PMC3",PMC6, PMC, MSR_PERFEVTSEL3, MSR_PMC3, 0, 0},
    /* Uncore PMC Counters: 8 48bit wide */
    {"UPMC0",PMC7,  UNCORE, MSR_UNCORE_PERFEVTSEL0, MSR_UNCORE_PMC0, 0, 0},
    {"UPMC1",PMC8,  UNCORE, MSR_UNCORE_PERFEVTSEL1, MSR_UNCORE_PMC1, 0, 0},
    {"UPMC2",PMC9,  UNCORE, MSR_UNCORE_PERFEVTSEL2, MSR_UNCORE_PMC2, 0, 0},
    {"UPMC3",PMC10, UNCORE, MSR_UNCORE_PERFEVTSEL3, MSR_UNCORE_PMC3, 0, 0},
    {"UPMC4",PMC11, UNCORE, MSR_UNCORE_PERFEVTSEL4, MSR_UNCORE_PMC4, 0, 0},
    {"UPMC5",PMC12, UNCORE, MSR_UNCORE_PERFEVTSEL5, MSR_UNCORE_PMC5, 0, 0},
    {"UPMC6",PMC13, UNCORE, MSR_UNCORE_PERFEVTSEL6, MSR_UNCORE_PMC6, 0, 0},
    {"UPMC7",PMC14, UNCORE, MSR_UNCORE_PERFEVTSEL7, MSR_UNCORE_PMC7, 0, 0}
};

