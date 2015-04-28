/*
 * =======================================================================================
 *
 *      Filename:  perfmon_broadwell_counters.h
 *
 *      Description:  Counter Header File of perfmon module for Broadwell.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *               Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 Jan Treibig and Thomas Roehl
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

#define NUM_COUNTERS_BROADWELL 12
#define NUM_COUNTERS_CORE_BROADWELL 8
#define NUM_COUNTERS_UNCORE_BROADWELL 12

#define BDW_VALID_OPTIONS_FIXED EVENT_OPTION_ANYTHREAD_MASK|EVENT_OPTION_COUNT_KERNEL_MASK
#define BDW_VALID_OPTIONS_PMC EVENT_OPTION_EDGE_MASK|EVENT_OPTION_COUNT_KERNEL_MASK|EVENT_OPTION_INVERT_MASK| \
            EVENT_OPTION_ANYTHREAD_MASK|EVENT_OPTION_IN_TRANS_MASK|EVENT_OPTION_THRESHOLD_MASK

static RegisterMap broadwell_counter_map[NUM_COUNTERS_BROADWELL] = {
    /* Fixed Counters: instructions retired, cycles unhalted core */
    {"FIXC0", PMC0, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR0, 0, 0, BDW_VALID_OPTIONS_FIXED},
    {"FIXC1", PMC1, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR1, 0, 0, BDW_VALID_OPTIONS_FIXED},
    {"FIXC2", PMC2, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR2, 0, 0, BDW_VALID_OPTIONS_FIXED},
    /* PMC Counters: 4 48bit wide */
    {"PMC0", PMC3, PMC, MSR_PERFEVTSEL0, MSR_PMC0, 0, 0, BDW_VALID_OPTIONS_PMC},
    {"PMC1", PMC4, PMC, MSR_PERFEVTSEL1, MSR_PMC1, 0, 0, BDW_VALID_OPTIONS_PMC},
    {"PMC2", PMC5, PMC, MSR_PERFEVTSEL2, MSR_PMC2, 0, 0, BDW_VALID_OPTIONS_PMC|EVENT_OPTION_IN_TRANS_ABORT_MASK},
    {"PMC3", PMC6, PMC, MSR_PERFEVTSEL3, MSR_PMC3, 0, 0, BDW_VALID_OPTIONS_PMC},
    /* Temperature Sensor*/
    {"TMP0", PMC7, THERMAL, 0, IA32_THERM_STATUS, 0, 0, EVENT_OPTION_NONE_MASK},
    /* RAPL counters */
    {"PWR0", PMC8, POWER, 0, MSR_PKG_ENERGY_STATUS, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PWR1", PMC9, POWER, 0, MSR_PP0_ENERGY_STATUS, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PWR2", PMC10, POWER, 0, MSR_PP1_ENERGY_STATUS,  0, 0, EVENT_OPTION_NONE_MASK},
    {"PWR3", PMC11, POWER, 0, MSR_DRAM_ENERGY_STATUS,  0, 0, EVENT_OPTION_NONE_MASK},
};


static BoxMap broadwell_box_map[NUM_UNITS] = {
    [PMC] = {MSR_PERF_GLOBAL_CTRL, MSR_PERF_GLOBAL_STATUS, MSR_PERF_GLOBAL_OVF_CTRL, 0, 0, 0, 48},
    [THERMAL] = {0, 0, 0, 0, 0, 0, 8},
    [FIXED] =  {MSR_PERF_GLOBAL_CTRL, MSR_PERF_GLOBAL_STATUS, MSR_PERF_GLOBAL_OVF_CTRL, 0, 0, 0, 48},
    [POWER] = {0, 0, 0, 0, 0, 0, 32},
};
