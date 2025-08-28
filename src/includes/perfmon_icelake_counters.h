/*
 * =======================================================================================
 *
 *      Filename:  perfmon_icelake_counters.h
 *
 *      Description:  Counter Header File of perfmon module for Intel Icelake.
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
#ifndef PERFMON_ICELAKE_COUNTERS_H
#define PERFMON_ICELAKE_COUNTERS_H

#define NUM_COUNTERS_ICELAKE 47
#define NUM_COUNTERS_CORE_ICELAKE 18
#define NUM_COUNTERS_UNCORE_ICELAKE 21

#define ICL_VALID_OPTIONS_FIXED EVENT_OPTION_ANYTHREAD_MASK | EVENT_OPTION_COUNT_KERNEL_MASK
#define ICL_VALID_OPTIONS_PMC                                                                                                                                                      \
    EVENT_OPTION_EDGE_MASK | EVENT_OPTION_COUNT_KERNEL_MASK | EVENT_OPTION_INVERT_MASK | EVENT_OPTION_ANYTHREAD_MASK | EVENT_OPTION_IN_TRANS_MASK | EVENT_OPTION_THRESHOLD_MASK
#define ICL_VALID_OPTIONS_CBOX EVENT_OPTION_EDGE_MASK | EVENT_OPTION_INVERT_MASK | EVENT_OPTION_THRESHOLD_MASK
#define ICL_VALID_OPTIONS_UBOX EVENT_OPTION_EDGE_MASK | EVENT_OPTION_INVERT_MASK | EVENT_OPTION_THRESHOLD_MASK

static RegisterMap icelake_counter_map[NUM_COUNTERS_ICELAKE] = {
    /* Fixed Counters: instructions retired, cycles unhalted core */
    { "FIXC0",     PMC0,  FIXED,    MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR0,        0, 0,                     ICL_VALID_OPTIONS_FIXED                                  },
    { "FIXC1",     PMC1,  FIXED,    MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR1,        0, 0,                     ICL_VALID_OPTIONS_FIXED                                  },
    { "FIXC2",     PMC2,  FIXED,    MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR2,        0, 0,                     ICL_VALID_OPTIONS_FIXED                                  },
    { "FIXC3",     PMC3,  FIXED,    MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR3,        0, 0,                     ICL_VALID_OPTIONS_FIXED                                  },
    /* PMC Counters: 4 48bit wide */
    { "PMC0",      PMC4,  PMC,      MSR_PERFEVTSEL0,         MSR_PMC0,                   0, 0,                     ICL_VALID_OPTIONS_PMC                                    },
    { "PMC1",      PMC5,  PMC,      MSR_PERFEVTSEL1,         MSR_PMC1,                   0, 0,                     ICL_VALID_OPTIONS_PMC                                    },
    { "PMC2",      PMC6,  PMC,      MSR_PERFEVTSEL2,         MSR_PMC2,                   0, 0,                     ICL_VALID_OPTIONS_PMC | EVENT_OPTION_IN_TRANS_ABORT_MASK },
    { "PMC3",      PMC7,  PMC,      MSR_PERFEVTSEL3,         MSR_PMC3,                   0, 0,                     ICL_VALID_OPTIONS_PMC                                    },
    /* Additional PMC Counters if HyperThreading is not active: 4 48bit wide */
    { "PMC4",      PMC8,  PMC,      MSR_PERFEVTSEL4,         MSR_PMC4,                   0, 0,                     ICL_VALID_OPTIONS_PMC                                    },
    { "PMC5",      PMC9,  PMC,      MSR_PERFEVTSEL5,         MSR_PMC5,                   0, 0,                     ICL_VALID_OPTIONS_PMC                                    },
    { "PMC6",      PMC10, PMC,      MSR_PERFEVTSEL6,         MSR_PMC6,                   0, 0,                     ICL_VALID_OPTIONS_PMC | EVENT_OPTION_IN_TRANS_ABORT_MASK },
    { "PMC7",      PMC11, PMC,      MSR_PERFEVTSEL7,         MSR_PMC7,                   0, 0,                     ICL_VALID_OPTIONS_PMC                                    },
    /* Temperature Sensor*/
    { "TMP0",      PMC12, THERMAL,  0,                       IA32_THERM_STATUS,          0, 0,                     EVENT_OPTION_NONE_MASK                                   },
    /* Vcore Status*/
    { "VTG0",      PMC13, VOLTAGE,  0,                       MSR_PERF_STATUS,            0, 0,                     EVENT_OPTION_NONE_MASK                                   },
    /* Intel Performance Metrics (first level of TMA tree) */
    { "TMA0",      PMC14, METRICS,  0,                       MSR_PERF_METRICS,           0, 0,                     EVENT_OPTION_NONE_MASK                                   },
    { "TMA1",      PMC15, METRICS,  0,                       MSR_PERF_METRICS,           0, 0,                     EVENT_OPTION_NONE_MASK                                   },
    { "TMA2",      PMC16, METRICS,  0,                       MSR_PERF_METRICS,           0, 0,                     EVENT_OPTION_NONE_MASK                                   },
    { "TMA3",      PMC17, METRICS,  0,                       MSR_PERF_METRICS,           0, 0,                     EVENT_OPTION_NONE_MASK                                   },
    /* RAPL counters */
    { "PWR0",      PMC18, POWER,    0,                       MSR_PKG_ENERGY_STATUS,      0, 0,                     EVENT_OPTION_NONE_MASK                                   },
    { "PWR1",      PMC19, POWER,    0,                       MSR_PP0_ENERGY_STATUS,      0, 0,                     EVENT_OPTION_NONE_MASK                                   },
    { "PWR2",      PMC20, POWER,    0,                       MSR_PP1_ENERGY_STATUS,      0, 0,                     EVENT_OPTION_NONE_MASK                                   },
    { "PWR3",      PMC21, POWER,    0,                       MSR_DRAM_ENERGY_STATUS,     0, 0,                     EVENT_OPTION_NONE_MASK                                   },
    { "PWR4",      PMC22, POWER,    0,                       MSR_PLATFORM_ENERGY_STATUS, 0, 0,                     EVENT_OPTION_NONE_MASK                                   },
    /* Uncore (general) counters */
    { "UBOXFIX",   PMC23, UBOXFIX,  MSR_UNC_PERF_FIXED_CTRL, MSR_UNC_PERF_FIXED_CTR,     0, 0,                     EVENT_OPTION_NONE_MASK                                   },
    { "UBOX0",     PMC24, UBOX,     MSR_V4_ARB_PERF_CTRL0,   MSR_V4_ARB_PERF_CTR0,       0, 0,                     ICL_VALID_OPTIONS_UBOX                                   },
    { "UBOX1",     PMC25, UBOX,     MSR_V4_ARB_PERF_CTRL1,   MSR_V4_ARB_PERF_CTR1,       0, 0,                     ICL_VALID_OPTIONS_UBOX                                   },
    /* Memory controller counters (MMIO) */
    { "MBOX0C0",   PMC26, MBOX0,    0x0,                     0x0,                        0, PCI_IMC_DEVICE_0_CH_0, EVENT_OPTION_NONE_MASK                                   },
    { "MBOX0C1",   PMC27, MBOX0,    0x0,                     0x1,                        0, PCI_IMC_DEVICE_0_CH_0, EVENT_OPTION_NONE_MASK                                   },
    { "MBOX0C2",   PMC28, MBOX0,    0x0,                     0x2,                        0, PCI_IMC_DEVICE_0_CH_0, EVENT_OPTION_NONE_MASK                                   },
    { "MBOX0TMP0", PMC29, MBOX0TMP, 0x0,                     0x3,                        0, PCI_IMC_DEVICE_0_CH_0, EVENT_OPTION_NONE_MASK                                   },
    { "MBOX0TMP1", PMC30, MBOX0TMP, 0x0,                     0x4,                        0, PCI_IMC_DEVICE_0_CH_0, EVENT_OPTION_NONE_MASK                                   },
    { "CBOX0C0",   PMC31, CBOX0,    MSR_V5_C0_PERF_CTRL0,    MSR_V5_C0_PERF_CTR0,        0, 0,                     ICL_VALID_OPTIONS_CBOX                                   },
    { "CBOX0C1",   PMC32, CBOX0,    MSR_V5_C0_PERF_CTRL1,    MSR_V5_C0_PERF_CTR1,        0, 0,                     ICL_VALID_OPTIONS_CBOX                                   },
    { "CBOX1C0",   PMC33, CBOX1,    MSR_V5_C1_PERF_CTRL0,    MSR_V5_C1_PERF_CTR0,        0, 0,                     ICL_VALID_OPTIONS_CBOX                                   },
    { "CBOX1C1",   PMC34, CBOX1,    MSR_V5_C1_PERF_CTRL1,    MSR_V5_C1_PERF_CTR1,        0, 0,                     ICL_VALID_OPTIONS_CBOX                                   },
    { "CBOX2C0",   PMC35, CBOX2,    MSR_V5_C2_PERF_CTRL0,    MSR_V5_C2_PERF_CTR0,        0, 0,                     ICL_VALID_OPTIONS_CBOX                                   },
    { "CBOX2C1",   PMC36, CBOX2,    MSR_V5_C2_PERF_CTRL1,    MSR_V5_C2_PERF_CTR1,        0, 0,                     ICL_VALID_OPTIONS_CBOX                                   },
    { "CBOX3C0",   PMC37, CBOX3,    MSR_V5_C3_PERF_CTRL0,    MSR_V5_C3_PERF_CTR0,        0, 0,                     ICL_VALID_OPTIONS_CBOX                                   },
    { "CBOX3C1",   PMC38, CBOX3,    MSR_V5_C3_PERF_CTRL1,    MSR_V5_C3_PERF_CTR1,        0, 0,                     ICL_VALID_OPTIONS_CBOX                                   },
    { "CBOX4C0",   PMC39, CBOX4,    MSR_V5_C4_PERF_CTRL0,    MSR_V5_C4_PERF_CTR0,        0, 0,                     ICL_VALID_OPTIONS_CBOX                                   },
    { "CBOX4C1",   PMC40, CBOX4,    MSR_V5_C4_PERF_CTRL1,    MSR_V5_C4_PERF_CTR1,        0, 0,                     ICL_VALID_OPTIONS_CBOX                                   },
    { "CBOX5C0",   PMC41, CBOX5,    MSR_V5_C5_PERF_CTRL0,    MSR_V5_C5_PERF_CTR0,        0, 0,                     ICL_VALID_OPTIONS_CBOX                                   },
    { "CBOX5C1",   PMC42, CBOX5,    MSR_V5_C5_PERF_CTRL1,    MSR_V5_C5_PERF_CTR1,        0, 0,                     ICL_VALID_OPTIONS_CBOX                                   },
    { "CBOX6C0",   PMC43, CBOX6,    MSR_V5_C6_PERF_CTRL0,    MSR_V5_C6_PERF_CTR0,        0, 0,                     ICL_VALID_OPTIONS_CBOX                                   },
    { "CBOX6C1",   PMC44, CBOX6,    MSR_V5_C6_PERF_CTRL1,    MSR_V5_C6_PERF_CTR1,        0, 0,                     ICL_VALID_OPTIONS_CBOX                                   },
    { "CBOX7C0",   PMC45, CBOX7,    MSR_V5_C7_PERF_CTRL0,    MSR_V5_C7_PERF_CTR0,        0, 0,                     ICL_VALID_OPTIONS_CBOX                                   },
    { "CBOX7C1",   PMC46, CBOX7,    MSR_V5_C7_PERF_CTRL1,    MSR_V5_C7_PERF_CTR1,        0, 0,                     ICL_VALID_OPTIONS_CBOX                                   },
};

static BoxMap icelake_box_map[NUM_UNITS] = {
    [PMC]     = { MSR_PERF_GLOBAL_CTRL,        MSR_V4_PERF_GLOBAL_STATUS,     MSR_V4_PERF_GLOBAL_STATUS_RESET, 0,  0, 0,                     48, 0, 0 },
    [THERMAL] = { 0,                           0,                             0,                               0,  0, 0,                     8,  0, 0 },
    [VOLTAGE] = { 0,                           0,                             0,                               0,  0, 0,                     16, 0, 0 },
    [FIXED]   = { MSR_PERF_GLOBAL_CTRL,        MSR_V4_PERF_GLOBAL_STATUS,     MSR_V4_PERF_GLOBAL_STATUS_RESET, 0,  0, 0,                     48, 0, 0 },
    [POWER]   = { 0,                           0,                             0,                               0,  0, 0,                     32, 0, 0 },
    [METRICS] = { 0,                           MSR_V4_PERF_GLOBAL_STATUS,     MSR_V4_PERF_GLOBAL_STATUS,       48, 0, 0,                     8,  0, 0 },
    [UBOXFIX] = { MSR_V4_UNC_PERF_GLOBAL_CTRL, MSR_V4_UNC_PERF_GLOBAL_STATUS, MSR_V4_UNC_PERF_GLOBAL_STATUS,   0,  0, 0,                     44, 0, 0 },
    [UBOX]    = { MSR_V4_UNC_PERF_GLOBAL_CTRL, MSR_V4_UNC_PERF_GLOBAL_STATUS, MSR_V4_UNC_PERF_GLOBAL_STATUS,   1,  0, 0,                     44, 0, 0 },
    [CBOX0]   = { MSR_V4_UNC_PERF_GLOBAL_CTRL, MSR_V4_UNC_PERF_GLOBAL_STATUS, MSR_V4_UNC_PERF_GLOBAL_STATUS,   3,  0, 0,                     44, 0, 0 },
    [CBOX1]   = { MSR_V4_UNC_PERF_GLOBAL_CTRL, MSR_V4_UNC_PERF_GLOBAL_STATUS, MSR_V4_UNC_PERF_GLOBAL_STATUS,   3,  0, 0,                     44, 0, 0 },
    [CBOX2]   = { MSR_V4_UNC_PERF_GLOBAL_CTRL, MSR_V4_UNC_PERF_GLOBAL_STATUS, MSR_V4_UNC_PERF_GLOBAL_STATUS,   3,  0, 0,                     44, 0, 0 },
    [CBOX3]   = { MSR_V4_UNC_PERF_GLOBAL_CTRL, MSR_V4_UNC_PERF_GLOBAL_STATUS, MSR_V4_UNC_PERF_GLOBAL_STATUS,   3,  0, 0,                     44, 0, 0 },
    [CBOX4]   = { MSR_V4_UNC_PERF_GLOBAL_CTRL, MSR_V4_UNC_PERF_GLOBAL_STATUS, MSR_V4_UNC_PERF_GLOBAL_STATUS,   3,  0, 0,                     44, 0, 0 },
    [CBOX5]   = { MSR_V4_UNC_PERF_GLOBAL_CTRL, MSR_V4_UNC_PERF_GLOBAL_STATUS, MSR_V4_UNC_PERF_GLOBAL_STATUS,   3,  0, 0,                     44, 0, 0 },
    [CBOX6]   = { MSR_V4_UNC_PERF_GLOBAL_CTRL, MSR_V4_UNC_PERF_GLOBAL_STATUS, MSR_V4_UNC_PERF_GLOBAL_STATUS,   3,  0, 0,                     44, 0, 0 },
    [CBOX7]   = { MSR_V4_UNC_PERF_GLOBAL_CTRL, MSR_V4_UNC_PERF_GLOBAL_STATUS, MSR_V4_UNC_PERF_GLOBAL_STATUS,   3,  0, 0,                     44, 0, 0 },
    [MBOX0]   = { 0,                           0,                             0,                               0,  1, PCI_IMC_DEVICE_0_CH_0, 32, 0, 0 },
};

static PciDevice icelake_pci_devices[MAX_NUM_PCI_DEVICES] = {
    [MSR_DEV]               = { NODEVTYPE, "",     "MSR",             "",      0,   0 },
    [PCI_IMC_DEVICE_0_CH_0] = { IMC,       "00.0", "MMAP_IMC_DEVICE", "MBOX0", 0x0, 0 },
};

#endif //PERFMON_ICELAKE_COUNTERS_H
