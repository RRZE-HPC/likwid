/*
 * =======================================================================================
 *
 *      Filename:  perfmon_zen2_counters.h
 *
 *      Description:  Counter Header File of perfmon module for AMD Family 17 (Zen2)
 *
 *      Version:   5.4.0
 *      Released:  15.11.2024
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2024 RRZE, University Erlangen-Nuremberg
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

#define NUM_COUNTERS_ZEN2 21
#define NUM_COUNTERS_CORE_ZEN2 9

#define AMD_K17_DF_EVSEL_SHIFT  0
#define AMD_K17_DF_EVSEL_MASK   0xFFULL
#define AMD_K17_DF_EVSEL_SHIFT1 32
#define AMD_K17_DF_EVSEL_MASK1  0xFULL
#define AMD_K17_DF_EVSEL_SHIFT2 59
#define AMD_K17_DF_EVSEL_MASK2  0x3ULL
#define AMD_K17_DF_UNIT_SHIFT   8
#define AMD_K17_DF_UNIT_MASK    0xFFULL

#define ZEN2_VALID_OPTIONS_PMC EVENT_OPTION_EDGE_MASK|EVENT_OPTION_COUNT_KERNEL_MASK|EVENT_OPTION_INVERT_MASK|EVENT_OPTION_THRESHOLD_MASK
#define ZEN2_VALID_OPTIONS_L3 EVENT_OPTION_TID_MASK|EVENT_OPTION_MATCH0_MASK

static RegisterMap zen2_counter_map[NUM_COUNTERS_ZEN2] = {
    /* Fixed counters */
    {"FIXC0", PMC0, FIXED, MSR_AMD17_HW_CONFIG, MSR_AMD17_RO_INST_RETIRED_CTR, 0, 0, 0},
    {"FIXC1", PMC1, FIXED, 0, MSR_AMD17_RO_APERF, 0, 0, 0},
    {"FIXC2", PMC2, FIXED, 0, MSR_AMD17_RO_MPERF, 0, 0, 0},
    /* Core counters */
    {"PMC0",PMC3, PMC, MSR_AMD17_PERFEVTSEL0, MSR_AMD17_PMC0, 0, 0, ZEN2_VALID_OPTIONS_PMC},
    {"PMC1",PMC4, PMC, MSR_AMD17_PERFEVTSEL1, MSR_AMD17_PMC1, 0, 0, ZEN2_VALID_OPTIONS_PMC},
    {"PMC2",PMC5, PMC, MSR_AMD17_PERFEVTSEL2, MSR_AMD17_PMC2, 0, 0, ZEN2_VALID_OPTIONS_PMC},
    {"PMC3",PMC6, PMC, MSR_AMD17_PERFEVTSEL3, MSR_AMD17_PMC3, 0, 0, ZEN2_VALID_OPTIONS_PMC},
    {"PMC4",PMC7, PMC, MSR_AMD17_2_PERFEVTSEL4, MSR_AMD17_2_PMC4, 0, 0, ZEN2_VALID_OPTIONS_PMC},
    {"PMC5",PMC8, PMC, MSR_AMD17_2_PERFEVTSEL5, MSR_AMD17_2_PMC5, 0, 0, ZEN2_VALID_OPTIONS_PMC},
    /* L3 cache counters */
    {"CPMC0",PMC9, CBOX0, MSR_AMD17_L3_PERFEVTSEL0, MSR_AMD17_L3_PMC0, 0, 0, ZEN2_VALID_OPTIONS_L3},
    {"CPMC1",PMC10, CBOX0, MSR_AMD17_L3_PERFEVTSEL1, MSR_AMD17_L3_PMC1, 0, 0, ZEN2_VALID_OPTIONS_L3},
    {"CPMC2",PMC11, CBOX0, MSR_AMD17_L3_PERFEVTSEL2, MSR_AMD17_L3_PMC2, 0, 0, ZEN2_VALID_OPTIONS_L3},
    {"CPMC3",PMC12, CBOX0, MSR_AMD17_L3_PERFEVTSEL3, MSR_AMD17_L3_PMC3, 0, 0, ZEN2_VALID_OPTIONS_L3},
    {"CPMC4",PMC13, CBOX0, MSR_AMD17_L3_PERFEVTSEL4, MSR_AMD17_L3_PMC4, 0, 0, ZEN2_VALID_OPTIONS_L3},
    {"CPMC5",PMC14, CBOX0, MSR_AMD17_L3_PERFEVTSEL5, MSR_AMD17_L3_PMC5, 0, 0, ZEN2_VALID_OPTIONS_L3},
    /* Energy counters */
    {"PWR0", PMC15, POWER, 0, MSR_AMD17_RAPL_CORE_STATUS, 0, 0},
    {"PWR1", PMC16, POWER, 0, MSR_AMD17_RAPL_PKG_STATUS, 0, 0},
    /* Data fabric counters */
    {"DFC0",PMC17, MBOX0, MSR_AMD17_2_DF_PERFEVTSEL0, MSR_AMD17_2_DF_PMC0, 0, 0},
    {"DFC1",PMC18, MBOX0, MSR_AMD17_2_DF_PERFEVTSEL1, MSR_AMD17_2_DF_PMC1, 0, 0},
    {"DFC2",PMC19, MBOX0, MSR_AMD17_2_DF_PERFEVTSEL2, MSR_AMD17_2_DF_PMC2, 0, 0},
    {"DFC3",PMC20, MBOX0, MSR_AMD17_2_DF_PERFEVTSEL3, MSR_AMD17_2_DF_PMC3, 0, 0},
};

static BoxMap zen2_box_map[NUM_UNITS] = {
    [FIXED] = {0, 0, 0, 0, 0, 0, 64},
    [PMC] = {0, 0, 0, 0, 0, 0, 48},
    [CBOX0] = {0, 0, 0, 0, 0, 0, 48},
    [MBOX0] = {0, 0, 0, 0, 0, 0, 48},
    [POWER] = {0, 0, 0, 0, 0, 0, 32},
};

static char* zen2_translate_types[NUM_UNITS] = {
    [FIXED] = "/sys/bus/event_source/devices/msr",
    [PMC] = "/sys/bus/event_source/devices/cpu",
    [POWER] = "/sys/bus/event_source/devices/power",
    [CBOX0] = "/sys/bus/event_source/devices/amd_l3",
    [MBOX0] = "/sys/bus/event_source/devices/amd_df",
};
