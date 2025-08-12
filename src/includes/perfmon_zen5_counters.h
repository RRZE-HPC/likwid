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

#define NUM_COUNTERS_ZEN5 98
#define NUM_COUNTERS_CORE_ZEN5 9

#define AMD_K1A_UMC_EVSEL_MASK 0xFF
#define AMD_K1A_UMC_EVSEL_SHIFT 0x0
#define AMD_K1A_UMC_RWMASK_MASK 0x3
#define AMD_K1A_UMC_RWMASK_SHIFT 0x8
#define AMD_K1A_UMC_ENABLE_BIT 31

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
    /* Energy counters */
    {"PWR0", PMC15, POWER, 0, MSR_AMD1A_RAPL_CORE_STATUS, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PWR2", PMC16, POWER, 0, MSR_AMD1A_RAPL_L3_STATUS, 0, 0, EVENT_OPTION_NONE_MASK},
    /* Data fabric counters */
    {"DFC0",PMC17, MBOX0, MSR_AMD19_DF_PERFEVTSEL0, MSR_AMD19_DF_PMC0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC1",PMC18, MBOX0, MSR_AMD19_DF_PERFEVTSEL1, MSR_AMD19_DF_PMC1, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC2",PMC19, MBOX0, MSR_AMD19_DF_PERFEVTSEL2, MSR_AMD19_DF_PMC2, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC3",PMC20, MBOX0, MSR_AMD19_DF_PERFEVTSEL3, MSR_AMD19_DF_PMC3, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC4",PMC21, MBOX0, MSR_AMD19_DF_PERFEVTSEL4, MSR_AMD19_DF_PMC4, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC5",PMC22, MBOX0, MSR_AMD19_DF_PERFEVTSEL5, MSR_AMD19_DF_PMC5, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC6",PMC23, MBOX0, MSR_AMD19_DF_PERFEVTSEL6, MSR_AMD19_DF_PMC6, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC7",PMC24, MBOX0, MSR_AMD19_DF_PERFEVTSEL7, MSR_AMD19_DF_PMC7, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC8",PMC25, MBOX0, MSR_AMD19_DF_PERFEVTSEL8, MSR_AMD19_DF_PMC8, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC9",PMC26, MBOX0, MSR_AMD19_DF_PERFEVTSEL9, MSR_AMD19_DF_PMC9, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC10",PMC27, MBOX0, MSR_AMD19_DF_PERFEVTSEL10, MSR_AMD19_DF_PMC10, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC11",PMC28, MBOX0, MSR_AMD19_DF_PERFEVTSEL11, MSR_AMD19_DF_PMC11, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC12",PMC29, MBOX0, MSR_AMD19_DF_PERFEVTSEL12, MSR_AMD19_DF_PMC12, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC13",PMC30, MBOX0, MSR_AMD19_DF_PERFEVTSEL13, MSR_AMD19_DF_PMC13, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC14",PMC31, MBOX0, MSR_AMD19_DF_PERFEVTSEL14, MSR_AMD19_DF_PMC14, 0, 0, EVENT_OPTION_NONE_MASK},
    {"DFC15",PMC32, MBOX0, MSR_AMD19_DF_PERFEVTSEL15, MSR_AMD19_DF_PMC15, 0, 0, EVENT_OPTION_NONE_MASK},
    /* UMC Performance counters */
    /* Each register is an own device (third column) but there are not enough MBOX* devices defined, */
    /* so we reuse the BBOX* devices here */
    {"UMC0",PMC33, BBOX0, MSR_AMD1A_UMC_PERFEVTSEL0, MSR_AMD1A_UMC_PMC0, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC1",PMC34, BBOX1, MSR_AMD1A_UMC_PERFEVTSEL1, MSR_AMD1A_UMC_PMC1, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC2",PMC35, BBOX2, MSR_AMD1A_UMC_PERFEVTSEL2, MSR_AMD1A_UMC_PMC2, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC3",PMC36, BBOX3, MSR_AMD1A_UMC_PERFEVTSEL3, MSR_AMD1A_UMC_PMC3, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC4",PMC37, BBOX4, MSR_AMD1A_UMC_PERFEVTSEL4, MSR_AMD1A_UMC_PMC4, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC5",PMC38, BBOX5, MSR_AMD1A_UMC_PERFEVTSEL5, MSR_AMD1A_UMC_PMC5, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC6",PMC39, BBOX6, MSR_AMD1A_UMC_PERFEVTSEL6, MSR_AMD1A_UMC_PMC6, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC7",PMC40, BBOX7, MSR_AMD1A_UMC_PERFEVTSEL7, MSR_AMD1A_UMC_PMC7, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC8",PMC41, BBOX8, MSR_AMD1A_UMC_PERFEVTSEL8, MSR_AMD1A_UMC_PMC8, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC9",PMC42, BBOX9, MSR_AMD1A_UMC_PERFEVTSEL9, MSR_AMD1A_UMC_PMC9, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC10",PMC43, BBOX10, MSR_AMD1A_UMC_PERFEVTSEL10, MSR_AMD1A_UMC_PMC10, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC11",PMC44, BBOX11, MSR_AMD1A_UMC_PERFEVTSEL11, MSR_AMD1A_UMC_PMC11, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC12",PMC45, BBOX12, MSR_AMD1A_UMC_PERFEVTSEL12, MSR_AMD1A_UMC_PMC12, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC13",PMC46, BBOX13, MSR_AMD1A_UMC_PERFEVTSEL13, MSR_AMD1A_UMC_PMC13, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC14",PMC47, BBOX14, MSR_AMD1A_UMC_PERFEVTSEL14, MSR_AMD1A_UMC_PMC14, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC15",PMC48, BBOX15, MSR_AMD1A_UMC_PERFEVTSEL15, MSR_AMD1A_UMC_PMC15, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC16",PMC49, BBOX16, MSR_AMD1A_UMC_PERFEVTSEL16, MSR_AMD1A_UMC_PMC16, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC17",PMC50, BBOX17, MSR_AMD1A_UMC_PERFEVTSEL17, MSR_AMD1A_UMC_PMC17, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC18",PMC51, BBOX18, MSR_AMD1A_UMC_PERFEVTSEL18, MSR_AMD1A_UMC_PMC18, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC19",PMC52, BBOX19, MSR_AMD1A_UMC_PERFEVTSEL19, MSR_AMD1A_UMC_PMC19, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC20",PMC53, BBOX20, MSR_AMD1A_UMC_PERFEVTSEL20, MSR_AMD1A_UMC_PMC20, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC21",PMC54, BBOX21, MSR_AMD1A_UMC_PERFEVTSEL21, MSR_AMD1A_UMC_PMC21, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC22",PMC55, BBOX22, MSR_AMD1A_UMC_PERFEVTSEL22, MSR_AMD1A_UMC_PMC22, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC23",PMC56, BBOX23, MSR_AMD1A_UMC_PERFEVTSEL23, MSR_AMD1A_UMC_PMC23, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC24",PMC57, BBOX24, MSR_AMD1A_UMC_PERFEVTSEL24, MSR_AMD1A_UMC_PMC24, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC25",PMC58, BBOX25, MSR_AMD1A_UMC_PERFEVTSEL25, MSR_AMD1A_UMC_PMC25, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC26",PMC59, BBOX26, MSR_AMD1A_UMC_PERFEVTSEL26, MSR_AMD1A_UMC_PMC26, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC27",PMC60, BBOX27, MSR_AMD1A_UMC_PERFEVTSEL27, MSR_AMD1A_UMC_PMC27, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC28",PMC61, BBOX28, MSR_AMD1A_UMC_PERFEVTSEL28, MSR_AMD1A_UMC_PMC28, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC29",PMC62, BBOX29, MSR_AMD1A_UMC_PERFEVTSEL29, MSR_AMD1A_UMC_PMC29, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC30",PMC63, BBOX30, MSR_AMD1A_UMC_PERFEVTSEL30, MSR_AMD1A_UMC_PMC30, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC31",PMC64, BBOX31, MSR_AMD1A_UMC_PERFEVTSEL31, MSR_AMD1A_UMC_PMC31, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC32",PMC65, BBOX32, MSR_AMD1A_UMC_PERFEVTSEL32, MSR_AMD1A_UMC_PMC32, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC33",PMC66, BBOX33, MSR_AMD1A_UMC_PERFEVTSEL33, MSR_AMD1A_UMC_PMC33, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC34",PMC67, BBOX34, MSR_AMD1A_UMC_PERFEVTSEL34, MSR_AMD1A_UMC_PMC34, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC35",PMC68, BBOX35, MSR_AMD1A_UMC_PERFEVTSEL35, MSR_AMD1A_UMC_PMC35, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC36",PMC69, BBOX36, MSR_AMD1A_UMC_PERFEVTSEL36, MSR_AMD1A_UMC_PMC36, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC37",PMC70, BBOX37, MSR_AMD1A_UMC_PERFEVTSEL37, MSR_AMD1A_UMC_PMC37, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC38",PMC71, BBOX38, MSR_AMD1A_UMC_PERFEVTSEL38, MSR_AMD1A_UMC_PMC38, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC39",PMC72, BBOX39, MSR_AMD1A_UMC_PERFEVTSEL39, MSR_AMD1A_UMC_PMC39, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC40",PMC73, BBOX40, MSR_AMD1A_UMC_PERFEVTSEL40, MSR_AMD1A_UMC_PMC40, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC41",PMC74, BBOX41, MSR_AMD1A_UMC_PERFEVTSEL41, MSR_AMD1A_UMC_PMC41, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC42",PMC75, BBOX42, MSR_AMD1A_UMC_PERFEVTSEL42, MSR_AMD1A_UMC_PMC42, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC43",PMC76, BBOX43, MSR_AMD1A_UMC_PERFEVTSEL43, MSR_AMD1A_UMC_PMC43, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC44",PMC77, BBOX44, MSR_AMD1A_UMC_PERFEVTSEL44, MSR_AMD1A_UMC_PMC44, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC45",PMC78, BBOX45, MSR_AMD1A_UMC_PERFEVTSEL45, MSR_AMD1A_UMC_PMC45, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC46",PMC79, BBOX46, MSR_AMD1A_UMC_PERFEVTSEL46, MSR_AMD1A_UMC_PMC46, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC47",PMC80, BBOX47, MSR_AMD1A_UMC_PERFEVTSEL47, MSR_AMD1A_UMC_PMC47, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC48",PMC81, BBOX48, MSR_AMD1A_UMC_PERFEVTSEL48, MSR_AMD1A_UMC_PMC48, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC49",PMC82, BBOX49, MSR_AMD1A_UMC_PERFEVTSEL49, MSR_AMD1A_UMC_PMC49, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC50",PMC83, BBOX50, MSR_AMD1A_UMC_PERFEVTSEL50, MSR_AMD1A_UMC_PMC50, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC51",PMC84, BBOX51, MSR_AMD1A_UMC_PERFEVTSEL51, MSR_AMD1A_UMC_PMC51, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC52",PMC85, BBOX52, MSR_AMD1A_UMC_PERFEVTSEL52, MSR_AMD1A_UMC_PMC52, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC53",PMC86, BBOX53, MSR_AMD1A_UMC_PERFEVTSEL53, MSR_AMD1A_UMC_PMC53, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC54",PMC87, BBOX54, MSR_AMD1A_UMC_PERFEVTSEL54, MSR_AMD1A_UMC_PMC54, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC55",PMC88, BBOX55, MSR_AMD1A_UMC_PERFEVTSEL55, MSR_AMD1A_UMC_PMC55, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC56",PMC89, BBOX56, MSR_AMD1A_UMC_PERFEVTSEL56, MSR_AMD1A_UMC_PMC56, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC57",PMC90, BBOX57, MSR_AMD1A_UMC_PERFEVTSEL57, MSR_AMD1A_UMC_PMC57, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC58",PMC91, BBOX58, MSR_AMD1A_UMC_PERFEVTSEL58, MSR_AMD1A_UMC_PMC58, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC59",PMC92, BBOX59, MSR_AMD1A_UMC_PERFEVTSEL59, MSR_AMD1A_UMC_PMC59, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC60",PMC93, BBOX60, MSR_AMD1A_UMC_PERFEVTSEL60, MSR_AMD1A_UMC_PMC60, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC61",PMC94, BBOX61, MSR_AMD1A_UMC_PERFEVTSEL61, MSR_AMD1A_UMC_PMC61, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC62",PMC95, BBOX62, MSR_AMD1A_UMC_PERFEVTSEL62, MSR_AMD1A_UMC_PMC62, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"UMC63",PMC96, BBOX63, MSR_AMD1A_UMC_PERFEVTSEL63, MSR_AMD1A_UMC_PMC63, 0, 0, ZEN5_VALID_OPTIONS_UMC},
    {"PWR1", PMC97, POWER, 0x0, 0x0, 0x0, 0x0, EVENT_OPTION_NONE_MASK},
};

static BoxMap zen5_box_map[NUM_UNITS] = {
    [FIXED] = {0, 0, 0, 0, 0, 0, 64, 0, 0},
    [PMC] = {0, 0, 0, 0, 0, 0, 48, 0, 0},
    [CBOX0] = {0, 0, 0, 0, 0, 0, 48, 0, 0},
    [MBOX0] = {0, 0, 0, 0, 0, 0, 48, 0, 0},
    [POWER] = {0, 0, 0, 0, 0, 0, 64, 0, 0},
    [BBOX0 ... BBOX63] = {0, 0, 0, 0, 0, 0, 48, 0, 0},
};

static char* zen5_translate_types[NUM_UNITS] = {
    [FIXED] = "/sys/bus/event_source/devices/msr",
    [PMC] = "/sys/bus/event_source/devices/cpu",
    [POWER] = "/sys/bus/event_source/devices/power",
    [CBOX0] = "/sys/bus/event_source/devices/amd_l3",
    [MBOX0] = "/sys/bus/event_source/devices/amd_df",
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
    [BBOX24] = "/sys/bus/event_source/devices/amd_umc_24",
    [BBOX25] = "/sys/bus/event_source/devices/amd_umc_25",
    [BBOX26] = "/sys/bus/event_source/devices/amd_umc_26",
    [BBOX27] = "/sys/bus/event_source/devices/amd_umc_27",
    [BBOX28] = "/sys/bus/event_source/devices/amd_umc_28",
    [BBOX29] = "/sys/bus/event_source/devices/amd_umc_29",
    [BBOX30] = "/sys/bus/event_source/devices/amd_umc_30",
    [BBOX31] = "/sys/bus/event_source/devices/amd_umc_31",
    [BBOX32] = "/sys/bus/event_source/devices/amd_umc_32",
    [BBOX33] = "/sys/bus/event_source/devices/amd_umc_33",
    [BBOX34] = "/sys/bus/event_source/devices/amd_umc_34",
    [BBOX35] = "/sys/bus/event_source/devices/amd_umc_35",
    [BBOX36] = "/sys/bus/event_source/devices/amd_umc_36",
    [BBOX37] = "/sys/bus/event_source/devices/amd_umc_37",
    [BBOX38] = "/sys/bus/event_source/devices/amd_umc_38",
    [BBOX39] = "/sys/bus/event_source/devices/amd_umc_39",
    [BBOX40] = "/sys/bus/event_source/devices/amd_umc_40",
    [BBOX41] = "/sys/bus/event_source/devices/amd_umc_41",
    [BBOX42] = "/sys/bus/event_source/devices/amd_umc_42",
    [BBOX43] = "/sys/bus/event_source/devices/amd_umc_43",
    [BBOX44] = "/sys/bus/event_source/devices/amd_umc_44",
    [BBOX45] = "/sys/bus/event_source/devices/amd_umc_45",
    [BBOX46] = "/sys/bus/event_source/devices/amd_umc_46",
    [BBOX47] = "/sys/bus/event_source/devices/amd_umc_47",
    [BBOX48] = "/sys/bus/event_source/devices/amd_umc_48",
    [BBOX49] = "/sys/bus/event_source/devices/amd_umc_49",
    [BBOX50] = "/sys/bus/event_source/devices/amd_umc_50",
    [BBOX51] = "/sys/bus/event_source/devices/amd_umc_51",
    [BBOX52] = "/sys/bus/event_source/devices/amd_umc_52",
    [BBOX53] = "/sys/bus/event_source/devices/amd_umc_53",
    [BBOX54] = "/sys/bus/event_source/devices/amd_umc_54",
    [BBOX55] = "/sys/bus/event_source/devices/amd_umc_55",
    [BBOX56] = "/sys/bus/event_source/devices/amd_umc_56",
    [BBOX57] = "/sys/bus/event_source/devices/amd_umc_57",
    [BBOX58] = "/sys/bus/event_source/devices/amd_umc_58",
    [BBOX59] = "/sys/bus/event_source/devices/amd_umc_59",
    [BBOX60] = "/sys/bus/event_source/devices/amd_umc_60",
    [BBOX61] = "/sys/bus/event_source/devices/amd_umc_61",
    [BBOX62] = "/sys/bus/event_source/devices/amd_umc_62",
    [BBOX63] = "/sys/bus/event_source/devices/amd_umc_63",

};

static char* registerTypeNamesZen5[MAX_UNITS] = {
    [POWER] = "AMD RAPL",
    [CBOX0] = "L3 Cache",
    [MBOX0] = "Data Fabric",
    [BBOX0 ... BBOX63] = "Unified Memory Controller",
};

#endif //PERFMON_ZEN5_COUNTERS_H
