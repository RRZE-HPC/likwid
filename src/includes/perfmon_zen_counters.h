/*
 * =======================================================================================
 *
 *      Filename:  perfmon_zen_counters.h
 *
 *      Description:  Counter Header File of perfmon module for AMD Family 17
 *
 *      Version:   4.3.2
 *      Released:  12.04.2018
 *
 *      Author:   Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2018 RRZE, University Erlangen-Nuremberg
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

#define NUM_COUNTERS_ZEN 19
#define NUM_COUNTERS_CORE_ZEN 7

#define AMD_K17_ENABLE_BIT 22

#define AMD_K17_INST_RETIRE_ENABLE_BIT 30

#define AMD_K17_PMC_INVERT_BIT 23
#define AMD_K17_PMC_EDGE_BIT 18
#define AMD_K17_PMC_KERNEL_BIT 17
#define AMD_K17_PMC_USER_BIT 16
#define AMD_K17_PMC_THRES_SHIFT 24
#define AMD_K17_PMC_THRES_MASK 0x7FULL
#define AMD_K17_PMC_HOST_BIT 41
#define AMD_K17_PMC_GUEST_BIT 40

#define AMD_K17_PMC_UNIT_SHIFT 8
#define AMD_K17_PMC_UNIT_MASK 0xFFULL
#define AMD_K17_PMC_EVSEL_SHIFT 0
#define AMD_K17_PMC_EVSEL_MASK 0xFFULL
#define AMD_K17_PMC_EVSEL_SHIFT2 32
#define AMD_K17_PMC_EVSEL_MASK2 0xFULL

#define AMD_K17_L3_UNIT_SHIFT 8
#define AMD_K17_L3_UNIT_MASK 0xFFULL
#define AMD_K17_L3_EVSEL_SHIFT 0
#define AMD_K17_L3_EVSEL_MASK 0xFFULL
#define AMD_K17_L3_TID_SHIFT 56
#define AMD_K17_L3_TID_MASK 0xFFULL
#define AMD_K17_L3_SLICE_SHIFT 48
#define AMD_K17_L3_SLICE_MASK 0xFULL

#define ZEN_VALID_OPTIONS_PMC EVENT_OPTION_EDGE_MASK|EVENT_OPTION_COUNT_KERNEL_MASK|EVENT_OPTION_INVERT_MASK|EVENT_OPTION_THRESHOLD_MASK
#define ZEN_VALID_OPTIONS_L3 EVENT_OPTION_TID_MASK|EVENT_OPTION_MATCH0_MASK

static RegisterMap zen_counter_map[NUM_COUNTERS_ZEN] = {
    /* Fixed counters */
    {"FIXC0", PMC0, FIXED, MSR_AMD17_HW_CONFIG, MSR_AMD17_RO_INST_RETIRED_CTR, 0, 0, 0},
    {"FIXC1", PMC1, FIXED, 0, MSR_AMD17_RO_APERF, 0, 0, 0},
    {"FIXC2", PMC2, FIXED, 0, MSR_AMD17_RO_MPERF, 0, 0, 0},
    /* Core counters */
    {"PMC0",PMC3, PMC, MSR_AMD17_PERFEVTSEL0, MSR_AMD17_PMC0, 0, 0, ZEN_VALID_OPTIONS_PMC},
    {"PMC1",PMC4, PMC, MSR_AMD17_PERFEVTSEL1, MSR_AMD17_PMC1, 0, 0, ZEN_VALID_OPTIONS_PMC},
    {"PMC2",PMC5, PMC, MSR_AMD17_PERFEVTSEL2, MSR_AMD17_PMC2, 0, 0, ZEN_VALID_OPTIONS_PMC},
    {"PMC3",PMC6, PMC, MSR_AMD17_PERFEVTSEL3, MSR_AMD17_PMC3, 0, 0, ZEN_VALID_OPTIONS_PMC},
    /* L3 cache counters */
    {"CPMC0",PMC7, CBOX0, MSR_AMD17_L3_PERFEVTSEL0, MSR_AMD17_L3_PMC0, 0, 0, ZEN_VALID_OPTIONS_L3},
    {"CPMC1",PMC8, CBOX0, MSR_AMD17_L3_PERFEVTSEL1, MSR_AMD17_L3_PMC1, 0, 0, ZEN_VALID_OPTIONS_L3},
    {"CPMC2",PMC9, CBOX0, MSR_AMD17_L3_PERFEVTSEL2, MSR_AMD17_L3_PMC2, 0, 0, ZEN_VALID_OPTIONS_L3},
    {"CPMC3",PMC10, CBOX0, MSR_AMD17_L3_PERFEVTSEL3, MSR_AMD17_L3_PMC3, 0, 0, ZEN_VALID_OPTIONS_L3},
    {"CPMC4",PMC11, CBOX0, MSR_AMD17_L3_PERFEVTSEL4, MSR_AMD17_L3_PMC4, 0, 0, ZEN_VALID_OPTIONS_L3},
    {"CPMC5",PMC12, CBOX0, MSR_AMD17_L3_PERFEVTSEL5, MSR_AMD17_L3_PMC5, 0, 0, ZEN_VALID_OPTIONS_L3},
    /* Energy counters */
    {"PWR0", PMC13, POWER, 0, MSR_AMD17_RAPL_CORE_STATUS, 0, 0},
    {"PWR1", PMC14, POWER, 0, MSR_AMD17_RAPL_PKG_STATUS, 0, 0},
    /* Northbridge counters */
    {"UPMC0",PMC15, UNCORE, MSR_AMD16_NB_PERFEVTSEL0, MSR_AMD16_NB_PMC0, 0, 0},
    {"UPMC1",PMC16, UNCORE, MSR_AMD16_NB_PERFEVTSEL1, MSR_AMD16_NB_PMC1, 0, 0},
    {"UPMC2",PMC17, UNCORE, MSR_AMD16_NB_PERFEVTSEL2, MSR_AMD16_NB_PMC2, 0, 0},
    {"UPMC3",PMC18, UNCORE, MSR_AMD16_NB_PERFEVTSEL3, MSR_AMD16_NB_PMC3, 0, 0}
};

static BoxMap zen_box_map[NUM_UNITS] = {
    [FIXED] = {0, 0, 0, 0, 0, 0, 64},
    [PMC] = {0, 0, 0, 0, 0, 0, 48},
    [CBOX0] = {0, 0, 0, 0, 0, 0, 48},
    [UNCORE] = {0, 0, 0, 0, 0, 0, 48},
    [POWER] = {0, 0, 0, 0, 0, 0, 32},
};

static char* zen_translate_types[NUM_UNITS] = {
    [FIXED] = "/sys/bus/event_source/devices/cpu",
    [PMC] = "/sys/bus/event_source/devices/cpu",
    [POWER] = "/sys/bus/event_source/devices/power",
    [CBOX0] = "/sys/bus/event_source/devices/amd_l2",
    [UNCORE] = "/sys/bus/event_source/devices/amd_nb",
};
