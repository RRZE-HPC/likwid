/*
 * =======================================================================================
 *
 *      Filename:  perfmon_sierraforrest_counters.h
 *
 *      Description:  Counter Header File of perfmon module for Intel Granite Rapids.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
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

#include <intel_perfmon_uncore_discovery.h>

#define NUM_COUNTERS_SIERRAFORREST 27
#define NUM_COUNTERS_CORE_SIERRAFORREST 18
#define NUM_COUNTERS_UNCORE_SIERRAFORREST 5


#define SRF_VALID_OPTIONS_FIXED EVENT_OPTION_ANYTHREAD_MASK|EVENT_OPTION_COUNT_KERNEL_MASK
#define SRF_VALID_OPTIONS_PMC EVENT_OPTION_EDGE_MASK|EVENT_OPTION_COUNT_KERNEL_MASK|EVENT_OPTION_INVERT_MASK| \
            EVENT_OPTION_ANYTHREAD_MASK|EVENT_OPTION_IN_TRANS_MASK|EVENT_OPTION_THRESHOLD_MASK

#define SRF_VALID_OPTIONS_CBOX  EVENT_OPTION_EDGE_MASK|EVENT_OPTION_INVERT_MASK|\
                                EVENT_OPTION_THRESHOLD_MASK|EVENT_OPTION_TID_MASK
#define SRF_VALID_OPTIONS_UNCORE EVENT_OPTION_EDGE_MASK|EVENT_OPTION_INVERT_MASK|\
                                 EVENT_OPTION_THRESHOLD_MASK


static RegisterMap sierraforrest_counter_map[NUM_COUNTERS_SIERRAFORREST] = {
    /* Fixed Counters: instructions retired, cycles unhalted core */
    {"FIXC0", PMC0, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR0, 0, MSR_DEV, SRF_VALID_OPTIONS_FIXED},
    {"FIXC1", PMC1, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR1, 0, MSR_DEV, SRF_VALID_OPTIONS_FIXED},
    {"FIXC2", PMC2, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR2, 0, MSR_DEV, SRF_VALID_OPTIONS_FIXED},
    {"FIXC3", PMC3, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR3, 0, MSR_DEV, SRF_VALID_OPTIONS_FIXED},
    /* PMC Counters: 4 48bit wide */
    {"PMC0", PMC4, PMC, MSR_PERFEVTSEL0, MSR_PMC0, 0, MSR_DEV, SRF_VALID_OPTIONS_PMC},
    {"PMC1", PMC5, PMC, MSR_PERFEVTSEL1, MSR_PMC1, 0, MSR_DEV, SRF_VALID_OPTIONS_PMC},
    {"PMC2", PMC6, PMC, MSR_PERFEVTSEL2, MSR_PMC2, 0, MSR_DEV, SRF_VALID_OPTIONS_PMC|EVENT_OPTION_IN_TRANS_ABORT_MASK},
    {"PMC3", PMC7, PMC, MSR_PERFEVTSEL3, MSR_PMC3, 0, MSR_DEV, SRF_VALID_OPTIONS_PMC},
    /* Additional PMC Counters if HyperThreading is not active: 4 48bit wide */
    {"PMC4", PMC8, PMC, MSR_PERFEVTSEL4, MSR_PMC4, 0, MSR_DEV, SRF_VALID_OPTIONS_PMC},
    {"PMC5", PMC9, PMC, MSR_PERFEVTSEL5, MSR_PMC5, 0, MSR_DEV, SRF_VALID_OPTIONS_PMC},
    {"PMC6", PMC10, PMC, MSR_PERFEVTSEL6, MSR_PMC6, 0, MSR_DEV, SRF_VALID_OPTIONS_PMC|EVENT_OPTION_IN_TRANS_ABORT_MASK},
    {"PMC7", PMC11, PMC, MSR_PERFEVTSEL7, MSR_PMC7, 0, MSR_DEV, SRF_VALID_OPTIONS_PMC},
    /* Temperature Sensor*/
    {"TMP0", PMC12, THERMAL, 0, IA32_THERM_STATUS, 0, MSR_DEV, EVENT_OPTION_NONE_MASK},
    /* Vcore Status*/
    {"VTG0", PMC13, VOLTAGE, 0, MSR_PERF_STATUS, 0, MSR_DEV, EVENT_OPTION_NONE_MASK},
    /* Intel Performance Metrics */
    {"TMA0", PMC14, METRICS, 0, MSR_PERF_METRICS, 0, MSR_DEV, EVENT_OPTION_NONE_MASK}, 
    {"TMA1", PMC15, METRICS, 0, MSR_PERF_METRICS, 0, MSR_DEV, EVENT_OPTION_NONE_MASK}, 
    {"TMA2", PMC16, METRICS, 0, MSR_PERF_METRICS, 0, MSR_DEV, EVENT_OPTION_NONE_MASK}, 
    {"TMA3", PMC17, METRICS, 0, MSR_PERF_METRICS, 0, MSR_DEV, EVENT_OPTION_NONE_MASK}, 
    {"TMA4", PMC18, METRICS, 0, MSR_PERF_METRICS, 0, MSR_DEV, EVENT_OPTION_NONE_MASK}, 
    {"TMA5", PMC19, METRICS, 0, MSR_PERF_METRICS, 0, MSR_DEV, EVENT_OPTION_NONE_MASK}, 
    {"TMA6", PMC20, METRICS, 0, MSR_PERF_METRICS, 0, MSR_DEV, EVENT_OPTION_NONE_MASK}, 
    {"TMA7", PMC21, METRICS, 0, MSR_PERF_METRICS, 0, MSR_DEV, EVENT_OPTION_NONE_MASK}, 
    /* RAPL counters */
    {"PWR0", PMC22, POWER, 0, MSR_PKG_ENERGY_STATUS, 0, MSR_DEV, EVENT_OPTION_NONE_MASK},
    {"PWR1", PMC23, POWER, 0, MSR_PP0_ENERGY_STATUS, 0, MSR_DEV, EVENT_OPTION_NONE_MASK},
    {"PWR2", PMC24, POWER, 0, MSR_PP1_ENERGY_STATUS,  0, MSR_DEV, EVENT_OPTION_NONE_MASK},
    {"PWR3", PMC25, POWER, 0, MSR_DRAM_ENERGY_STATUS,  0, MSR_DEV, EVENT_OPTION_NONE_MASK},
    {"PWR4", PMC26, POWER, 0, MSR_PLATFORM_ENERGY_STATUS,  0, MSR_DEV, EVENT_OPTION_NONE_MASK},
};

static BoxMap sierraforrest_box_map[NUM_UNITS] = {
    [FIXED] =  {MSR_PERF_GLOBAL_CTRL, MSR_V4_PERF_GLOBAL_STATUS, MSR_V4_PERF_GLOBAL_STATUS_RESET, 0, 0, 0, 48},
    [PMC] = {MSR_PERF_GLOBAL_CTRL, MSR_V4_PERF_GLOBAL_STATUS, MSR_V4_PERF_GLOBAL_STATUS_RESET, 0, 0, 0, 48},
    [THERMAL] = {0, 0, 0, 0, 0, 0, 8},
    [POWER] = {0, 0, 0, 0, 0, 0, 32},
    [VOLTAGE] = {0, 0, 0, 0, 0, 0, 16},
    [METRICS] = {0, MSR_V4_PERF_GLOBAL_STATUS, MSR_V4_PERF_GLOBAL_STATUS, 48, 0, 0, 8},
};

static PciDevice sierraforrest_pci_devices[MAX_NUM_PCI_DEVICES] = {
    [MSR_DEV] = {NODEVTYPE, "", "MSR", ""},
};

static char* sierraforrest_translate_types[NUM_UNITS] = {
    [FIXED] = "/sys/bus/event_source/devices/cpu",
    [PMC] = "/sys/bus/event_source/devices/cpu",
    // I don't get it to work, so no TMA metrics with perf_event at the moment
    //[METRICS] = "/sys/bus/event_source/devices/cpu",
    [POWER] = "/sys/bus/event_source/devices/power",
};


static char* registerTypeNamesSierraForrest[MAX_UNITS] = {
};

#define     SRF_DEVICE_ID_CHA 0
#define     SRF_DEVICE_ID_IIO 1
#define     SRF_DEVICE_ID_IRP 2
#define     SRF_DEVICE_ID_iMC 6
#define     SRF_DEVICE_ID_UPI 8
#define     SRF_DEVICE_ID_PCU 4
#define     SRF_DEVICE_ID_UBOX 5
#define     SRF_DEVICE_ID_B2UPI 18
#define     SRF_DEVICE_ID_B2CMI 16
#define     SRF_DEVICE_ID_B2CXL 17
#define     SRF_DEVICE_ID_MDF 20

static PerfmonUncoreDiscovery sierraforrest_uncore_discovery_map[] = {
    {"CBOX", SRF_DEVICE_ID_CHA, 127, MSR_CBOX_DEVICE_C0},
    {"MBOX", SRF_DEVICE_ID_iMC, 15, MMIO_IMC_DEVICE_0_CH_0},
    {"UBOX", SRF_DEVICE_ID_UBOX, 1, MSR_UBOX_DEVICE},
    {"WBOX", SRF_DEVICE_ID_PCU, 3, MSR_PCU_DEVICE_0},
    {"IRP", SRF_DEVICE_ID_IRP, 15, MSR_IRP_DEVICE_0},
    {"IIO", SRF_DEVICE_ID_IIO, 15, MSR_IIO_DEVICE_0},
    {"QBOX", SRF_DEVICE_ID_UPI, 3, PCI_QPI_DEVICE_PORT_0},
    {"MDF", SRF_DEVICE_ID_MDF, 49, MSR_MDF_DEVICE_0},
    {"M2M", SRF_DEVICE_ID_B2CMI, 31, PCI_HA_DEVICE_0},
    {"RBOX", SRF_DEVICE_ID_B2UPI, 3, PCI_R3QPI_DEVICE_LINK_0},
    {"PBOX", SRF_DEVICE_ID_B2CXL, 31, PCI_R2PCIE_DEVICE0},
    {"INVALID", -1, 0, MSR_DEV}
};
