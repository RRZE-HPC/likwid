/*
 * =======================================================================================
 *
 *      Filename:  perfmon_zen5_counters.h
 *
 *      Description:  Counter Header File of perfmon module for AMD Family 1A (Zen5)
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2017 RRZE, University Erlangen-Nuremberg
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
#ifndef PERFMON_ZEN5_COUNTERS_H
#define PERFMON_ZEN5_COUNTERS_H

#ifdef LIKWID_USE_PERFEVENT
#define NUM_COUNTERS_ZEN5 32
#else
#define NUM_COUNTERS_ZEN5 33
#endif
#define NUM_COUNTERS_CORE_ZEN5 9

#define AMD_K1A_UMC_EVSEL_MASK 0xFF
#define AMD_K1A_UMC_EVSEL_SHIFT 0x0
#define AMD_K1A_UMC_RWMASK_MASK 0x3
#define AMD_K1A_UMC_RWMASK_SHIFT 0x8
#define AMD_K1A_UMC_ENABLE_BIT 31
#define AMD_K1A_UMC_MAX_UNITS 12

#define ZEN5_VALID_OPTIONS_PMC EVENT_OPTION_EDGE_MASK|EVENT_OPTION_COUNT_KERNEL_MASK|EVENT_OPTION_INVERT_MASK|EVENT_OPTION_THRESHOLD_MASK
#define ZEN5_VALID_OPTIONS_L3 EVENT_OPTION_TID_MASK|EVENT_OPTION_CID_MASK|EVENT_OPTION_SLICE_MASK
#define ZEN5_VALID_OPTIONS_UMC EVENT_OPTION_MASK0_MASK

static RegisterMap zen5_counter_map[NUM_COUNTERS_ZEN5] = {
    /* Fixed counters */
    {"FIXC0", PMC0, FIXED, MSR_AMD17_HW_CONFIG, MSR_AMD17_RO_INST_RETIRED_CTR, 0, 0, EVENT_OPTION_NONE_MASK},
    {"FIXC1", PMC1, FIXED, 0, MSR_AMD17_RO_APERF, 0, 0, EVENT_OPTION_NONE_MASK},
    {"FIXC2", PMC2, FIXED, 0, MSR_AMD17_RO_MPERF, 0, 0, EVENT_OPTION_NONE_MASK},
    /* Core counters */
    {"PMC0",PMC3, PMC, MSR_AMD17_PERFEVTSEL0, MSR_AMD17_PMC0, 0, 0, ZEN5_VALID_OPTIONS_PMC},
    {"PMC1",PMC4, PMC, MSR_AMD17_PERFEVTSEL1, MSR_AMD17_PMC1, 0, 0, ZEN5_VALID_OPTIONS_PMC},
    {"PMC2",PMC5, PMC, MSR_AMD17_PERFEVTSEL2, MSR_AMD17_PMC2, 0, 0, ZEN5_VALID_OPTIONS_PMC},
    {"PMC3",PMC6, PMC, MSR_AMD17_PERFEVTSEL3, MSR_AMD17_PMC3, 0, 0, ZEN5_VALID_OPTIONS_PMC},
    {"PMC4",PMC7, PMC, MSR_AMD17_2_PERFEVTSEL4, MSR_AMD17_2_PMC4, 0, 0, ZEN5_VALID_OPTIONS_PMC},
    {"PMC5",PMC8, PMC, MSR_AMD17_2_PERFEVTSEL5, MSR_AMD17_2_PMC5, 0, 0, ZEN5_VALID_OPTIONS_PMC},
    /* L3 cache counters */
    {"CPMC0",PMC9, CBOX0, MSR_AMD17_L3_PERFEVTSEL0, MSR_AMD17_L3_PMC0, 0, 0, ZEN5_VALID_OPTIONS_L3},
    {"CPMC1",PMC10, CBOX0, MSR_AMD17_L3_PERFEVTSEL1, MSR_AMD17_L3_PMC1, 0, 0, ZEN5_VALID_OPTIONS_L3},
    {"CPMC2",PMC11, CBOX0, MSR_AMD17_L3_PERFEVTSEL2, MSR_AMD17_L3_PMC2, 0, 0, ZEN5_VALID_OPTIONS_L3},
    {"CPMC3",PMC12, CBOX0, MSR_AMD17_L3_PERFEVTSEL3, MSR_AMD17_L3_PMC3, 0, 0, ZEN5_VALID_OPTIONS_L3},
    {"CPMC4",PMC13, CBOX0, MSR_AMD17_L3_PERFEVTSEL4, MSR_AMD17_L3_PMC4, 0, 0, ZEN5_VALID_OPTIONS_L3},
    {"CPMC5",PMC14, CBOX0, MSR_AMD17_L3_PERFEVTSEL5, MSR_AMD17_L3_PMC5, 0, 0, ZEN5_VALID_OPTIONS_L3},
    /* Data fabric counters */
    {"DFC0",PMC15, MBOX0, MSR_AMD19_DF_PERFEVTSEL0, MSR_AMD19_DF_PMC0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC1",PMC16, MBOX0, MSR_AMD19_DF_PERFEVTSEL1, MSR_AMD19_DF_PMC1, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC2",PMC17, MBOX0, MSR_AMD19_DF_PERFEVTSEL2, MSR_AMD19_DF_PMC2, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC3",PMC18, MBOX0, MSR_AMD19_DF_PERFEVTSEL3, MSR_AMD19_DF_PMC3, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC4",PMC19, MBOX0, MSR_AMD19_DF_PERFEVTSEL4, MSR_AMD19_DF_PMC4, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC5",PMC20, MBOX0, MSR_AMD19_DF_PERFEVTSEL5, MSR_AMD19_DF_PMC5, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC6",PMC21, MBOX0, MSR_AMD19_DF_PERFEVTSEL6, MSR_AMD19_DF_PMC6, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC7",PMC22, MBOX0, MSR_AMD19_DF_PERFEVTSEL7, MSR_AMD19_DF_PMC7, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC8",PMC23, MBOX0, MSR_AMD19_DF_PERFEVTSEL8, MSR_AMD19_DF_PMC8, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC9",PMC24, MBOX0, MSR_AMD19_DF_PERFEVTSEL9, MSR_AMD19_DF_PMC9, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC10",PMC25, MBOX0, MSR_AMD19_DF_PERFEVTSEL10, MSR_AMD19_DF_PMC10, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC11",PMC26, MBOX0, MSR_AMD19_DF_PERFEVTSEL11, MSR_AMD19_DF_PMC11, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC12",PMC27, MBOX0, MSR_AMD19_DF_PERFEVTSEL12, MSR_AMD19_DF_PMC12, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC13",PMC28, MBOX0, MSR_AMD19_DF_PERFEVTSEL13, MSR_AMD19_DF_PMC13, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC14",PMC29, MBOX0, MSR_AMD19_DF_PERFEVTSEL14, MSR_AMD19_DF_PMC14, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC15",PMC30, MBOX0, MSR_AMD19_DF_PERFEVTSEL15, MSR_AMD19_DF_PMC15, 0, 0, EVENT_OPTION_NONE_MASK},
    /* Energy counters */
#ifdef LIKWID_USE_PERFEVENT
    {"PWR1", PMC31, POWER, 0x0, 0x0, 0x0, 0x0, EVENT_OPTION_NONE_MASK},
#else
    {"PWR0", PMC31, POWER, 0, MSR_AMD1A_RAPL_CORE_STATUS, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PWR2", PMC32, POWER, 0, MSR_AMD1A_RAPL_L3_STATUS, 0, 0, EVENT_OPTION_NONE_MASK},
#endif
    /* UMC Performance counters are added at runtime depending on values from CPUID */
};

static BoxMap zen5_box_map[NUM_UNITS] = {
    [FIXED] = {0, 0, 0, 0, 0, 0, 64, 0, 0},
    [PMC] = {0, 0, 0, 0, 0, 0, 48, 0, 0},
    [CBOX0] = {0, 0, 0, 0, 0, 0, 48, 0, 0},
    [MBOX0] = {0, 0, 0, 0, 0, 0, 48, 0, 0},
    [POWER] = {0, 0, 0, 0, 0, 0, 64, 0, 0},
    [BBOX0 ... BBOX23] = {0, 0, 0, 0, 0, 0, 48, 0, 0},
};

static char* zen5_translate_types[NUM_UNITS] = {
    [FIXED] = "/sys/bus/event_source/devices/msr",
    [PMC] = "/sys/bus/event_source/devices/cpu",
    [POWER] = "/sys/bus/event_source/devices/power",
    [CBOX0] = "/sys/bus/event_source/devices/amd_l3",
    [MBOX0] = "/sys/bus/event_source/devices/amd_df",
    /* UMC Performance counters for socket 0 */
    [BBOX0] = "/sys/bus/event_source/devices/amd_umc_0",
    [BBOX1] = "/sys/bus/event_source/devices/amd_umc_1",
    [BBOX2] = "/sys/bus/event_source/devices/amd_umc_2",
    [BBOX3] = "/sys/bus/event_source/devices/amd_umc_3",
    [BBOX4] = "/sys/bus/event_source/devices/amd_umc_4",
    [BBOX5] = "/sys/bus/event_source/devices/amd_umc_5",
    [BBOX6] = "/sys/bus/event_source/devices/amd_umc_6",
    [BBOX7] = "/sys/bus/event_source/devices/amd_umc_7",
    [BBOX8] = "/sys/bus/event_source/devices/amd_umc_8",
    [BBOX9] = "/sys/bus/event_source/devices/amd_umc_9",
    [BBOX10] = "/sys/bus/event_source/devices/amd_umc_10",
    [BBOX11] = "/sys/bus/event_source/devices/amd_umc_11",
    /* UMC Performance counters for socket 1 */
    /* The selection is done at runtime in perfmon_perfevent.h */
    [BBOX12] = "/sys/bus/event_source/devices/amd_umc_12",
    [BBOX13] = "/sys/bus/event_source/devices/amd_umc_13",
    [BBOX14] = "/sys/bus/event_source/devices/amd_umc_14",
    [BBOX15] = "/sys/bus/event_source/devices/amd_umc_15",
    [BBOX16] = "/sys/bus/event_source/devices/amd_umc_16",
    [BBOX17] = "/sys/bus/event_source/devices/amd_umc_17",
    [BBOX18] = "/sys/bus/event_source/devices/amd_umc_18",
    [BBOX19] = "/sys/bus/event_source/devices/amd_umc_19",
    [BBOX20] = "/sys/bus/event_source/devices/amd_umc_20",
    [BBOX21] = "/sys/bus/event_source/devices/amd_umc_21",
    [BBOX22] = "/sys/bus/event_source/devices/amd_umc_22",
    [BBOX23] = "/sys/bus/event_source/devices/amd_umc_23",
};

static char* registerTypeNamesZen5[MAX_UNITS] = {
    [POWER] = "AMD RAPL",
    [CBOX0] = "L3 Cache",
    [MBOX0] = "Data Fabric",
    [BBOX0 ... BBOX23] = "Unified Memory Controller",
};

#endif //PERFMON_ZEN5_COUNTERS_H
