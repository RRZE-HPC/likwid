/*
 * =======================================================================================
 *
 *      Filename:  perfmon_tigerlake_counters.h
 *
 *      Description:  Counter Header File of perfmon module for Intel Tigerlake.
 *
 *      Version:   5.1
 *      Released:  16.11.2020
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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

#define NUM_COUNTERS_TIGERLAKE 23
#define NUM_COUNTERS_CORE_TIGERLAKE 18
#define NUM_COUNTERS_UNCORE_TIGERLAKE 5

#define TGL_VALID_OPTIONS_FIXED EVENT_OPTION_COUNT_KERNEL_MASK
#define TGL_VALID_OPTIONS_PMC EVENT_OPTION_EDGE_MASK|EVENT_OPTION_COUNT_KERNEL_MASK|EVENT_OPTION_INVERT_MASK|EVENT_OPTION_IN_TRANS_MASK|EVENT_OPTION_THRESHOLD_MASK

static RegisterMap tigerlake_counter_map[NUM_COUNTERS_TIGERLAKE] = {
    /* Fixed Counters: instructions retired, cycles unhalted core */
    {"FIXC0", PMC0, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR0, 0, 0, TGL_VALID_OPTIONS_FIXED},
    {"FIXC1", PMC1, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR1, 0, 0, TGL_VALID_OPTIONS_FIXED},
    {"FIXC2", PMC2, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR2, 0, 0, TGL_VALID_OPTIONS_FIXED},
    {"FIXC3", PMC3, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR3, 0, 0, TGL_VALID_OPTIONS_FIXED},
    /* PMC Counters: 4 48bit wide */
    {"PMC0", PMC4, PMC, MSR_PERFEVTSEL0, MSR_PMC0, 0, 0, TGL_VALID_OPTIONS_PMC},
    {"PMC1", PMC5, PMC, MSR_PERFEVTSEL1, MSR_PMC1, 0, 0, TGL_VALID_OPTIONS_PMC},
    {"PMC2", PMC6, PMC, MSR_PERFEVTSEL2, MSR_PMC2, 0, 0, TGL_VALID_OPTIONS_PMC|EVENT_OPTION_IN_TRANS_ABORT_MASK},
    {"PMC3", PMC7, PMC, MSR_PERFEVTSEL3, MSR_PMC3, 0, 0, TGL_VALID_OPTIONS_PMC},
    /* Additional PMC Counters if HyperThreading is not active: 4 48bit wide */
    {"PMC4", PMC8, PMC, MSR_PERFEVTSEL4, MSR_PMC4, 0, 0, TGL_VALID_OPTIONS_PMC},
    {"PMC5", PMC9, PMC, MSR_PERFEVTSEL5, MSR_PMC5, 0, 0, TGL_VALID_OPTIONS_PMC},
    {"PMC6", PMC10, PMC, MSR_PERFEVTSEL6, MSR_PMC6, 0, 0, TGL_VALID_OPTIONS_PMC|EVENT_OPTION_IN_TRANS_ABORT_MASK},
    {"PMC7", PMC11, PMC, MSR_PERFEVTSEL7, MSR_PMC7, 0, 0, TGL_VALID_OPTIONS_PMC},
    /* Temperature Sensor*/
    {"TMP0", PMC12, THERMAL, 0, IA32_THERM_STATUS, 0, 0, EVENT_OPTION_NONE_MASK},
    /* Vcore Status*/
    {"VTG0", PMC13, VOLTAGE, 0, MSR_PERF_STATUS, 0, 0, EVENT_OPTION_NONE_MASK},
    /* Intel Performance Metrics (first level of TMA tree) */
    {"TMA0", PMC14, METRICS, 0, MSR_PERF_METRICS, 0, 0, EVENT_OPTION_NONE_MASK}, 
    {"TMA1", PMC15, METRICS, 0, MSR_PERF_METRICS, 0, 0, EVENT_OPTION_NONE_MASK}, 
    {"TMA2", PMC16, METRICS, 0, MSR_PERF_METRICS, 0, 0, EVENT_OPTION_NONE_MASK}, 
    {"TMA3", PMC17, METRICS, 0, MSR_PERF_METRICS, 0, 0, EVENT_OPTION_NONE_MASK},
    /* RAPL counters */
    {"PWR0", PMC18, POWER, 0, MSR_PKG_ENERGY_STATUS, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PWR1", PMC19, POWER, 0, MSR_PP0_ENERGY_STATUS, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PWR2", PMC20, POWER, 0, MSR_PP1_ENERGY_STATUS,  0, 0, EVENT_OPTION_NONE_MASK},
    {"PWR3", PMC21, POWER, 0, MSR_DRAM_ENERGY_STATUS,  0, 0, EVENT_OPTION_NONE_MASK},
    {"PWR4", PMC22, POWER, 0, MSR_PLATFORM_ENERGY_STATUS,  0, 0, EVENT_OPTION_NONE_MASK},
};


static BoxMap tigerlake_box_map[NUM_UNITS] = {
    [PMC] = {MSR_PERF_GLOBAL_CTRL, MSR_V4_PERF_GLOBAL_STATUS, MSR_V4_PERF_GLOBAL_STATUS_RESET, 0, 0, 0, 48},
    [FIXED] =  {MSR_PERF_GLOBAL_CTRL, MSR_V4_PERF_GLOBAL_STATUS, MSR_V4_PERF_GLOBAL_STATUS_RESET, 0, 0, 0, 48},
    [THERMAL] = {0, 0, 0, 0, 0, 0, 8},
    [VOLTAGE] = {0, 0, 0, 0, 0, 0, 16},
    [POWER] = {0, 0, 0, 0, 0, 0, 32},
    [METRICS] = {0, MSR_V4_PERF_GLOBAL_STATUS, MSR_V4_PERF_GLOBAL_STATUS, 48, 0, 0, 8},
};

static PciDevice tigerlake_pci_devices[MAX_NUM_PCI_DEVICES] = {
 [MSR_DEV] = {NODEVTYPE, "", "MSR", ""},
 [PCI_IMC_DEVICE_0_CH_0] = {IMC, "00.0", "MMAP_IMC_DEVICE", "MBOX0", 0x0},
};
