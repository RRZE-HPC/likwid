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

#define NUM_COUNTERS_CORE_NEHALEM 7
#define NUM_COUNTERS_UNCORE_NEHALEM 16
#define NUM_COUNTERS_NEHALEM 16

#define NEH_VALID_OPTIONS_FIXED EVENT_OPTION_COUNT_KERNEL_MASK|EVENT_OPTION_ANYTHREAD_MASK
#define NEH_VALID_OPTIONS_PMC EVENT_OPTION_COUNT_KERNEL_MASK|EVENT_OPTION_ANYTHREAD_MASK|EVENT_OPTION_EDGE_MASK|EVENT_OPTION_INVERT_MASK|EVENT_OPTION_THRESHOLD_MASK
#define NEH_VALID_OPTIONS_UNCORE EVENT_OPTION_ANYTHREAD_MASK|EVENT_OPTION_EDGE_MASK|EVENT_OPTION_INVERT_MASK|EVENT_OPTION_THRESHOLD_MASK|EVENT_OPTION_MATCH0_MASK|EVENT_OPTION_OPCODE_MASK

static RegisterMap nehalem_counter_map[NUM_COUNTERS_NEHALEM] = {
    /* Fixed Counters: instructions retired, cycles unhalted core */
    {"FIXC0",PMC0, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR0, 0, 0, NEH_VALID_OPTIONS_FIXED},
    {"FIXC1",PMC1, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR1, 0, 0, NEH_VALID_OPTIONS_FIXED},
    {"FIXC2",PMC2, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR2, 0, 0, NEH_VALID_OPTIONS_FIXED},
    /* PMC Counters: 4 48bit wide */
    {"PMC0",PMC3, PMC, MSR_PERFEVTSEL0, MSR_PMC0, 0, 0, NEH_VALID_OPTIONS_PMC},
    {"PMC1",PMC4, PMC, MSR_PERFEVTSEL1, MSR_PMC1, 0, 0, NEH_VALID_OPTIONS_PMC},
    {"PMC2",PMC5, PMC, MSR_PERFEVTSEL2, MSR_PMC2, 0, 0, NEH_VALID_OPTIONS_PMC},
    {"PMC3",PMC6, PMC, MSR_PERFEVTSEL3, MSR_PMC3, 0, 0, NEH_VALID_OPTIONS_PMC},
    /* Uncore PMC Counters: 8 48bit wide */
    {"UPMC0",PMC7,  UNCORE, MSR_UNCORE_PERFEVTSEL0, MSR_UNCORE_PMC0, 0, 0, NEH_VALID_OPTIONS_UNCORE},
    {"UPMC1",PMC8,  UNCORE, MSR_UNCORE_PERFEVTSEL1, MSR_UNCORE_PMC1, 0, 0, NEH_VALID_OPTIONS_UNCORE},
    {"UPMC2",PMC9,  UNCORE, MSR_UNCORE_PERFEVTSEL2, MSR_UNCORE_PMC2, 0, 0, NEH_VALID_OPTIONS_UNCORE},
    {"UPMC3",PMC10, UNCORE, MSR_UNCORE_PERFEVTSEL3, MSR_UNCORE_PMC3, 0, 0, NEH_VALID_OPTIONS_UNCORE},
    {"UPMC4",PMC11, UNCORE, MSR_UNCORE_PERFEVTSEL4, MSR_UNCORE_PMC4, 0, 0, NEH_VALID_OPTIONS_UNCORE},
    {"UPMC5",PMC12, UNCORE, MSR_UNCORE_PERFEVTSEL5, MSR_UNCORE_PMC5, 0, 0, NEH_VALID_OPTIONS_UNCORE},
    {"UPMC6",PMC13, UNCORE, MSR_UNCORE_PERFEVTSEL6, MSR_UNCORE_PMC6, 0, 0, NEH_VALID_OPTIONS_UNCORE},
    {"UPMC7",PMC14, UNCORE, MSR_UNCORE_PERFEVTSEL7, MSR_UNCORE_PMC7, 0, 0, NEH_VALID_OPTIONS_UNCORE},
    {"UPMCFIX",PMC15, UNCORE, MSR_UNCORE_FIXED_CTR_CTRL, MSR_UNCORE_FIXED_CTR0, 0, 0, EVENT_OPTION_NONE_MASK}
};

static BoxMap nehalem_box_map[NUM_UNITS] = {
    [PMC] = {MSR_PERF_GLOBAL_CTRL, MSR_PERF_GLOBAL_STATUS, MSR_PERF_GLOBAL_OVF_CTRL, -1, 0, 0, 48},
    [FIXED] =  {MSR_PERF_GLOBAL_CTRL, MSR_PERF_GLOBAL_STATUS, MSR_PERF_GLOBAL_OVF_CTRL, -1, 0, 0, 48},
    [UNCORE] = {MSR_UNCORE_PERF_GLOBAL_CTRL, MSR_UNCORE_PERF_GLOBAL_STATUS, MSR_UNCORE_PERF_GLOBAL_OVF_CTRL, -1, 0, 0, 48}
};

