/*
 * =======================================================================================
 *
 *      Filename:  perfmon_skylake_counters.h
 *
 *      Description:  Counter Header File of perfmon module for Intel Skylake.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
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

#define NUM_COUNTERS_SKYLAKE 37
#define NUM_COUNTERS_CORE_SKYLAKE 16
#define NUM_COUNTERS_UNCORE_SKYLAKE 21

#define SKL_VALID_OPTIONS_FIXED EVENT_OPTION_ANYTHREAD_MASK|EVENT_OPTION_COUNT_KERNEL_MASK
#define SKL_VALID_OPTIONS_PMC EVENT_OPTION_EDGE_MASK|EVENT_OPTION_COUNT_KERNEL_MASK|EVENT_OPTION_INVERT_MASK| \
            EVENT_OPTION_ANYTHREAD_MASK|EVENT_OPTION_IN_TRANS_MASK|EVENT_OPTION_THRESHOLD_MASK
#define SKL_VALID_OPTIONS_CBOX EVENT_OPTION_EDGE_MASK|EVENT_OPTION_INVERT_MASK|EVENT_OPTION_THRESHOLD_MASK
#define SKL_VALID_OPTIONS_UBOX EVENT_OPTION_THRESHOLD_MASK|EVENT_OPTION_EDGE_MASK|EVENT_OPTION_INVERT_MASK

static RegisterMap skylake_counter_map[NUM_COUNTERS_SKYLAKE] = {
    /* Fixed Counters: instructions retired, cycles unhalted core */
    {"FIXC0", PMC0, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR0, 0, 0, SKL_VALID_OPTIONS_FIXED},
    {"FIXC1", PMC1, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR1, 0, 0, SKL_VALID_OPTIONS_FIXED},
    {"FIXC2", PMC2, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR2, 0, 0, SKL_VALID_OPTIONS_FIXED},
    /* PMC Counters: 4 48bit wide */
    {"PMC0", PMC3, PMC, MSR_PERFEVTSEL0, MSR_PMC0, 0, 0, SKL_VALID_OPTIONS_PMC},
    {"PMC1", PMC4, PMC, MSR_PERFEVTSEL1, MSR_PMC1, 0, 0, SKL_VALID_OPTIONS_PMC},
    {"PMC2", PMC5, PMC, MSR_PERFEVTSEL2, MSR_PMC2, 0, 0, SKL_VALID_OPTIONS_PMC|EVENT_OPTION_IN_TRANS_ABORT_MASK},
    {"PMC3", PMC6, PMC, MSR_PERFEVTSEL3, MSR_PMC3, 0, 0, SKL_VALID_OPTIONS_PMC},
    /* Additional PMC Counters if HyperThreading is not active: 4 48bit wide */
    {"PMC4", PMC7, PMC, MSR_PERFEVTSEL4, MSR_PMC4, 0, 0, SKL_VALID_OPTIONS_PMC},
    {"PMC5", PMC8, PMC, MSR_PERFEVTSEL5, MSR_PMC5, 0, 0, SKL_VALID_OPTIONS_PMC},
    {"PMC6", PMC9, PMC, MSR_PERFEVTSEL6, MSR_PMC6, 0, 0, SKL_VALID_OPTIONS_PMC|EVENT_OPTION_IN_TRANS_ABORT_MASK},
    {"PMC7", PMC10, PMC, MSR_PERFEVTSEL7, MSR_PMC7, 0, 0, SKL_VALID_OPTIONS_PMC},
    /* Temperature Sensor*/
    {"TMP0", PMC11, THERMAL, 0, IA32_THERM_STATUS, 0, 0, EVENT_OPTION_NONE_MASK},
    /* Vcore Status*/
    {"VTG0", PMC12, VOLTAGE, 0, MSR_PERF_STATUS, 0, 0, EVENT_OPTION_NONE_MASK},
    /* RAPL counters */
    {"PWR0", PMC13, POWER, 0, MSR_PKG_ENERGY_STATUS, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PWR1", PMC14, POWER, 0, MSR_PP0_ENERGY_STATUS, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PWR2", PMC15, POWER, 0, MSR_PP1_ENERGY_STATUS,  0, 0, EVENT_OPTION_NONE_MASK},
    {"PWR3", PMC16, POWER, 0, MSR_DRAM_ENERGY_STATUS,  0, 0, EVENT_OPTION_NONE_MASK},
    {"PWR4", PMC17, POWER, 0, MSR_PLATFORM_ENERGY_STATUS,  0, 0, EVENT_OPTION_NONE_MASK},
    /* Test */
    {"UBOXFIX", PMC18, UBOXFIX, MSR_UNC_PERF_FIXED_CTRL, MSR_UNC_PERF_FIXED_CTR, 0, 0, EVENT_OPTION_NONE_MASK},
    {"UBOX0", PMC19, UBOX, MSR_V4_ARB_PERF_CTRL0, MSR_V4_ARB_PERF_CTR0, 0, 0, SKL_VALID_OPTIONS_UBOX},
    {"UBOX1", PMC20, UBOX, MSR_V4_ARB_PERF_CTRL1, MSR_V4_ARB_PERF_CTR1, 0, 0, SKL_VALID_OPTIONS_UBOX},
    {"CBOX0C0", PMC21, CBOX0, MSR_V4_C0_PERF_CTRL0, MSR_V4_C0_PERF_CTR0, 0, 0, SKL_VALID_OPTIONS_CBOX},
    {"CBOX0C1", PMC22, CBOX0, MSR_V4_C0_PERF_CTRL1, MSR_V4_C0_PERF_CTR1, 0, 0, SKL_VALID_OPTIONS_CBOX},
    {"CBOX1C0", PMC23, CBOX1, MSR_V4_C1_PERF_CTRL0, MSR_V4_C1_PERF_CTR0, 0, 0, SKL_VALID_OPTIONS_CBOX},
    {"CBOX1C1", PMC24, CBOX1, MSR_V4_C1_PERF_CTRL1, MSR_V4_C1_PERF_CTR1, 0, 0, SKL_VALID_OPTIONS_CBOX},
    {"CBOX2C0", PMC25, CBOX2, MSR_V4_C2_PERF_CTRL0, MSR_V4_C2_PERF_CTR0, 0, 0, SKL_VALID_OPTIONS_CBOX},
    {"CBOX2C1", PMC26, CBOX2, MSR_V4_C2_PERF_CTRL1, MSR_V4_C2_PERF_CTR1, 0, 0, SKL_VALID_OPTIONS_CBOX},
    {"CBOX3C0", PMC27, CBOX3, MSR_V4_C3_PERF_CTRL0, MSR_V4_C3_PERF_CTR0, 0, 0, SKL_VALID_OPTIONS_CBOX},
    {"CBOX3C1", PMC28, CBOX3, MSR_V4_C3_PERF_CTRL1, MSR_V4_C3_PERF_CTR1, 0, 0, SKL_VALID_OPTIONS_CBOX},
    {"MBOX0C0", PMC29, MBOX0, 0x0, 0x0, 0, PCI_IMC_DEVICE_0_CH_0},
    {"MBOX0C1", PMC30, MBOX0, 0x0, 0x1, 0, PCI_IMC_DEVICE_0_CH_0},
    {"MBOX0C2", PMC31, MBOX0, 0x0, 0x2, 0, PCI_IMC_DEVICE_0_CH_0},
    {"MBOX0TMP0", PMC32, MBOX0TMP, 0x0, 0x3, 0, PCI_IMC_DEVICE_0_CH_0},
    {"MBOX0TMP1", PMC33, MBOX0TMP, 0x0, 0x4, 0, PCI_IMC_DEVICE_0_CH_0},
    /* PERF */
    {"MPERF", PMC34, PERF, 0, MSR_MPERF, 0, 0, EVENT_OPTION_NONE_MASK},
    {"APERF", PMC35, PERF, 0, MSR_APERF, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PPERF", PMC36, PERF, 0, MSR_PPERF, 0, 0, EVENT_OPTION_NONE_MASK},
};


static BoxMap skylake_box_map[NUM_UNITS] = {
    [PMC] = {MSR_PERF_GLOBAL_CTRL, MSR_V4_PERF_GLOBAL_STATUS, MSR_V4_PERF_GLOBAL_STATUS_RESET, 0, 0, 0, 48},
    [FIXED] =  {MSR_PERF_GLOBAL_CTRL, MSR_V4_PERF_GLOBAL_STATUS, MSR_V4_PERF_GLOBAL_STATUS_RESET, 0, 0, 0, 48},
    [PERF]    = {0, 0, 0, 0, 0, 0, 64},
    [THERMAL] = {0, 0, 0, 0, 0, 0, 8},
    [VOLTAGE] = {0, 0, 0, 0, 0, 0, 16},
    [POWER] = {0, 0, 0, 0, 0, 0, 32},
    [UBOXFIX] = {MSR_V4_UNC_PERF_GLOBAL_CTRL, MSR_V4_UNC_PERF_GLOBAL_STATUS, MSR_V4_UNC_PERF_GLOBAL_STATUS, 0, 0, 0, 44},
    [UBOX] = {MSR_V4_UNC_PERF_GLOBAL_CTRL, MSR_V4_UNC_PERF_GLOBAL_STATUS, MSR_V4_UNC_PERF_GLOBAL_STATUS, 1, 0, 0, 44},
    [CBOX0] = {MSR_V4_UNC_PERF_GLOBAL_CTRL, MSR_V4_UNC_PERF_GLOBAL_STATUS, MSR_V4_UNC_PERF_GLOBAL_STATUS, 3, 0, 0, 44},
    [CBOX1] = {MSR_V4_UNC_PERF_GLOBAL_CTRL, MSR_V4_UNC_PERF_GLOBAL_STATUS, MSR_V4_UNC_PERF_GLOBAL_STATUS, 3, 0, 0, 44},
    [CBOX2] = {MSR_V4_UNC_PERF_GLOBAL_CTRL, MSR_V4_UNC_PERF_GLOBAL_STATUS, MSR_V4_UNC_PERF_GLOBAL_STATUS, 3, 0, 0, 44},
    [CBOX3] = {MSR_V4_UNC_PERF_GLOBAL_CTRL, MSR_V4_UNC_PERF_GLOBAL_STATUS, MSR_V4_UNC_PERF_GLOBAL_STATUS, 3, 0, 0, 44},
    [MBOX0] = {0, 0, 0, 0, 1, PCI_IMC_DEVICE_0_CH_0, 32,},
};

static PciDevice skylake_pci_devices[MAX_NUM_PCI_DEVICES] = {
 [MSR_DEV] = {NODEVTYPE, "", "MSR", ""},
 [PCI_IMC_DEVICE_0_CH_0] = {IMC, "00.0", "MMAP_IMC_DEVICE", "MBOX0", 0x0},
};

static char* skylake_translate_types[NUM_UNITS] = {
    [FIXED] = "/sys/bus/event_source/devices/cpu",
    [PERF] = "/sys/bus/event_source/devices/msr",
    [PMC] = "/sys/bus/event_source/devices/cpu",
    [MBOX0] = "/sys/bus/event_source/devices/uncore_imc",
    [CBOX0] = "/sys/bus/event_source/devices/uncore_cbox_0",
    [CBOX1] = "/sys/bus/event_source/devices/uncore_cbox_1",
    [CBOX2] = "/sys/bus/event_source/devices/uncore_cbox_2",
    [CBOX3] = "/sys/bus/event_source/devices/uncore_cbox_3",
    [UBOX] = "/sys/bus/event_source/devices/uncore_arb",
    [UBOXFIX] = "/sys/bus/event_source/devices/uncore_arb",
    [POWER] = "/sys/bus/event_source/devices/power",
};
