/*
 * =======================================================================================
 *
 *      Filename:  perfmon_ivybridge_counters.h
 *
 *      Description: Counter header file of perfmon module for Ivy Bridge.
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
#include <registers.h>

#define NUM_COUNTERS_CORE_SILVERMONT 6
#define NUM_COUNTERS_UNCORE_SILVERMONT 0
#define NUM_COUNTERS_SILVERMONT 8

#define SVM_VALID_OPTIONS_FIXED EVENT_OPTION_COUNT_KERNEL_MASK|EVENT_OPTION_ANYTHREAD_MASK
#define SVM_VALID_OPTIONS_PMC EVENT_OPTION_EDGE_MASK|EVENT_OPTION_COUNT_KERNEL_MASK|EVENT_OPTION_INVERT_MASK|EVENT_OPTION_ANYTHREAD_MASK|EVENT_OPTION_THRESHOLD_MASK

static RegisterMap silvermont_counter_map[NUM_COUNTERS_SILVERMONT] = {
    /* Fixed Counters: instructions retired, cycles unhalted core */
    {"FIXC0", PMC0, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR0, 0, 0, SVM_VALID_OPTIONS_FIXED},
    {"FIXC1", PMC1, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR1, 0, 0, SVM_VALID_OPTIONS_FIXED},
    {"FIXC2", PMC2, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR2, 0, 0, SVM_VALID_OPTIONS_FIXED},
    /* PMC Counters: 4 48bit wide */
    {"PMC0", PMC3, PMC, MSR_PERFEVTSEL0, MSR_PMC0, MSR_OFFCORE_RESP0, 0, SVM_VALID_OPTIONS_PMC},
    {"PMC1", PMC4, PMC, MSR_PERFEVTSEL1, MSR_PMC1, MSR_OFFCORE_RESP1, 0, SVM_VALID_OPTIONS_PMC},
    /* Temperature Sensor*/
    {"TMP0", PMC5, THERMAL, 0, IA32_THERM_STATUS, 0, 0},
    /* RAPL counters */
    {"PWR0", PMC6, POWER, 0, MSR_PKG_ENERGY_STATUS, 0, 0},
    {"PWR1", PMC7, POWER, 0, MSR_PP0_ENERGY_STATUS, 0, 0}
};

static BoxMap silvermont_box_map[NUM_UNITS] = {
    [PMC] = {MSR_PERF_GLOBAL_CTRL, MSR_PERF_GLOBAL_STATUS, MSR_PERF_GLOBAL_OVF_CTRL, -1, 0, 0, 48},
    [FIXED] = {MSR_PERF_GLOBAL_CTRL, MSR_PERF_GLOBAL_STATUS, MSR_PERF_GLOBAL_OVF_CTRL, -1, 0, 0, 48},
    [THERMAL] = {0, 0, 0, 0, 0, MSR_DEV, 8},
    [POWER] = {0, 0, 0, 0, 0, MSR_DEV, 32}
};
