/*
 * =======================================================================================
 *
 *      Filename:  perfmon_sandybridgeEP_counters.h
 *
 *      Description: Counter header file of perfmon module for Intel Sandy Bridge EP.
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
#ifndef PERFMON_SANDYBRIDGEEP_COUNTERS_H
#define PERFMON_SANDYBRIDGEEP_COUNTERS_H

#define NUM_COUNTERS_SANDYBRIDGEEP 102
#define NUM_COUNTERS_UNCORE_SANDYBRIDGEEP 53
#define NUM_COUNTERS_CORE_SANDYBRIDGEEP 13

#define SNBEP_VALID_OPTIONS_FIXED EVENT_OPTION_COUNT_KERNEL_MASK | EVENT_OPTION_ANYTHREAD_MASK
#define SNBEP_VALID_OPTIONS_PMC EVENT_OPTION_EDGE_MASK | EVENT_OPTION_COUNT_KERNEL_MASK | EVENT_OPTION_ANYTHREAD_MASK | EVENT_OPTION_INVERT_MASK | EVENT_OPTION_THRESHOLD_MASK
#define SNBEP_VALID_OPTIONS_CBOX                                                                                                                                                   \
    EVENT_OPTION_THRESHOLD_MASK | EVENT_OPTION_EDGE_MASK | EVENT_OPTION_TID_MASK | EVENT_OPTION_INVERT_MASK | EVENT_OPTION_OPCODE_MASK | EVENT_OPTION_NID_MASK |                   \
        EVENT_OPTION_STATE_MASK
#define SNBEP_VALID_OPTIONS_WBOX                                                                                                                                                   \
    EVENT_OPTION_EDGE_MASK | EVENT_OPTION_INVERT_MASK | EVENT_OPTION_THRESHOLD_MASK | EVENT_OPTION_OCCUPANCY_MASK | EVENT_OPTION_OCCUPANCY_EDGE_MASK |                             \
        EVENT_OPTION_OCCUPANCY_INVERT_MASK | EVENT_OPTION_MATCH0_MASK
#define SNBEP_VALID_OPTIONS_UBOX EVENT_OPTION_THRESHOLD_MASK | EVENT_OPTION_EDGE_MASK | EVENT_OPTION_INVERT_MASK
#define SNBEP_VALID_OPTIONS_BBOX EVENT_OPTION_THRESHOLD_MASK | EVENT_OPTION_EDGE_MASK | EVENT_OPTION_INVERT_MASK | EVENT_OPTION_OPCODE_MASK | EVENT_OPTION_MATCH0_MASK
#define SNBEP_VALID_OPTIONS_MBOX EVENT_OPTION_THRESHOLD_MASK | EVENT_OPTION_EDGE_MASK | EVENT_OPTION_INVERT_MASK
#define SNBEP_VALID_OPTIONS_SBOX                                                                                                                                                   \
    EVENT_OPTION_THRESHOLD_MASK | EVENT_OPTION_EDGE_MASK | EVENT_OPTION_INVERT_MASK | EVENT_OPTION_MATCH0_MASK | EVENT_OPTION_MATCH1_MASK | EVENT_OPTION_MASK0_MASK |              \
        EVENT_OPTION_MASK1_MASK
#define SNBEP_VALID_OPTIONS_RBOX EVENT_OPTION_THRESHOLD_MASK | EVENT_OPTION_EDGE_MASK | EVENT_OPTION_INVERT_MASK
#define SNBEP_VALID_OPTIONS_PBOX EVENT_OPTION_THRESHOLD_MASK | EVENT_OPTION_EDGE_MASK | EVENT_OPTION_INVERT_MASK

static RegisterMap sandybridgeEP_counter_map[NUM_COUNTERS_SANDYBRIDGEEP] = {
    /* Fixed Counters: instructions retired, cycles unhalted core */
    { "FIXC0",    PMC0,   FIXED,    MSR_PERF_FIXED_CTR_CTRL,   MSR_PERF_FIXED_CTR0,         0,                           0,                          SNBEP_VALID_OPTIONS_FIXED },
    { "FIXC1",    PMC1,   FIXED,    MSR_PERF_FIXED_CTR_CTRL,   MSR_PERF_FIXED_CTR1,         0,                           0,                          SNBEP_VALID_OPTIONS_FIXED },
    { "FIXC2",    PMC2,   FIXED,    MSR_PERF_FIXED_CTR_CTRL,   MSR_PERF_FIXED_CTR2,         0,                           0,                          SNBEP_VALID_OPTIONS_FIXED },
    /* PMC Counters: 4 48bit wide */
    { "PMC0",     PMC3,   PMC,      MSR_PERFEVTSEL0,           MSR_PMC0,                    0,                           0,                          SNBEP_VALID_OPTIONS_PMC   },
    { "PMC1",     PMC4,   PMC,      MSR_PERFEVTSEL1,           MSR_PMC1,                    0,                           0,                          SNBEP_VALID_OPTIONS_PMC   },
    { "PMC2",     PMC5,   PMC,      MSR_PERFEVTSEL2,           MSR_PMC2,                    0,                           0,                          SNBEP_VALID_OPTIONS_PMC   },
    { "PMC3",     PMC6,   PMC,      MSR_PERFEVTSEL3,           MSR_PMC3,                    0,                           0,                          SNBEP_VALID_OPTIONS_PMC   },
    /* Additional PMC Counters: 4 48bit wide if HyperThreading is disabled*/
    { "PMC4",     PMC7,   PMC,      MSR_PERFEVTSEL4,           MSR_PMC4,                    0,                           0,                          SNBEP_VALID_OPTIONS_PMC   },
    { "PMC5",     PMC8,   PMC,      MSR_PERFEVTSEL5,           MSR_PMC5,                    0,                           0,                          SNBEP_VALID_OPTIONS_PMC   },
    { "PMC6",     PMC9,   PMC,      MSR_PERFEVTSEL6,           MSR_PMC6,                    0,                           0,                          SNBEP_VALID_OPTIONS_PMC   },
    { "PMC7",     PMC10,  PMC,      MSR_PERFEVTSEL7,           MSR_PMC7,                    0,                           0,                          SNBEP_VALID_OPTIONS_PMC   },
    /* Temperature Sensor*/
    { "TMP0",     PMC11,  THERMAL,  0,                         IA32_THERM_STATUS,           0,                           0,                          EVENT_OPTION_NONE_MASK    },
    /* Vcore Status*/
    { "VTG0",     PMC12,  VOLTAGE,  0,                         MSR_PERF_STATUS,             0,                           0,                          EVENT_OPTION_NONE_MASK    },
    /* RAPL counters */
    { "PWR0",     PMC13,  POWER,    0,                         MSR_PKG_ENERGY_STATUS,       0,                           0,                          EVENT_OPTION_NONE_MASK    },
    { "PWR1",     PMC14,  POWER,    0,                         MSR_PP0_ENERGY_STATUS,       0,                           0,                          EVENT_OPTION_NONE_MASK    },
    { "PWR2",     PMC15,  POWER,    0,                         MSR_PP1_ENERGY_STATUS,       0,                           0,                          EVENT_OPTION_NONE_MASK    },
    { "PWR3",     PMC16,  POWER,    0,                         MSR_DRAM_ENERGY_STATUS,      0,                           0,                          EVENT_OPTION_NONE_MASK    },
    /* CBOX counters */
    { "CBOX0C0",  PMC17,  CBOX0,    MSR_UNC_C0_PMON_CTL0,      MSR_UNC_C0_PMON_CTR0,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX0C1",  PMC18,  CBOX0,    MSR_UNC_C0_PMON_CTL1,      MSR_UNC_C0_PMON_CTR1,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX0C2",  PMC19,  CBOX0,    MSR_UNC_C0_PMON_CTL2,      MSR_UNC_C0_PMON_CTR2,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX0C3",  PMC20,  CBOX0,    MSR_UNC_C0_PMON_CTL3,      MSR_UNC_C0_PMON_CTR3,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX1C0",  PMC21,  CBOX1,    MSR_UNC_C1_PMON_CTL0,      MSR_UNC_C1_PMON_CTR0,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX1C1",  PMC22,  CBOX1,    MSR_UNC_C1_PMON_CTL1,      MSR_UNC_C1_PMON_CTR1,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX1C2",  PMC23,  CBOX1,    MSR_UNC_C1_PMON_CTL2,      MSR_UNC_C1_PMON_CTR2,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX1C3",  PMC24,  CBOX1,    MSR_UNC_C1_PMON_CTL3,      MSR_UNC_C1_PMON_CTR3,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX2C0",  PMC25,  CBOX2,    MSR_UNC_C2_PMON_CTL0,      MSR_UNC_C2_PMON_CTR0,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX2C1",  PMC26,  CBOX2,    MSR_UNC_C2_PMON_CTL1,      MSR_UNC_C2_PMON_CTR1,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX2C2",  PMC27,  CBOX2,    MSR_UNC_C2_PMON_CTL2,      MSR_UNC_C2_PMON_CTR2,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX2C3",  PMC28,  CBOX2,    MSR_UNC_C2_PMON_CTL3,      MSR_UNC_C2_PMON_CTR3,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX3C0",  PMC29,  CBOX3,    MSR_UNC_C3_PMON_CTL0,      MSR_UNC_C3_PMON_CTR0,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX3C1",  PMC30,  CBOX3,    MSR_UNC_C3_PMON_CTL1,      MSR_UNC_C3_PMON_CTR1,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX3C2",  PMC31,  CBOX3,    MSR_UNC_C3_PMON_CTL2,      MSR_UNC_C3_PMON_CTR2,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX3C3",  PMC32,  CBOX3,    MSR_UNC_C3_PMON_CTL3,      MSR_UNC_C3_PMON_CTR3,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX4C0",  PMC33,  CBOX4,    MSR_UNC_C4_PMON_CTL0,      MSR_UNC_C4_PMON_CTR0,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX4C1",  PMC34,  CBOX4,    MSR_UNC_C4_PMON_CTL1,      MSR_UNC_C4_PMON_CTR1,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX4C2",  PMC35,  CBOX4,    MSR_UNC_C4_PMON_CTL2,      MSR_UNC_C4_PMON_CTR2,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX4C3",  PMC36,  CBOX4,    MSR_UNC_C4_PMON_CTL3,      MSR_UNC_C4_PMON_CTR3,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX5C0",  PMC37,  CBOX5,    MSR_UNC_C5_PMON_CTL0,      MSR_UNC_C5_PMON_CTR0,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX5C1",  PMC38,  CBOX5,    MSR_UNC_C5_PMON_CTL1,      MSR_UNC_C5_PMON_CTR1,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX5C2",  PMC39,  CBOX5,    MSR_UNC_C5_PMON_CTL2,      MSR_UNC_C5_PMON_CTR2,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX5C3",  PMC40,  CBOX5,    MSR_UNC_C5_PMON_CTL3,      MSR_UNC_C5_PMON_CTR3,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX6C0",  PMC41,  CBOX6,    MSR_UNC_C6_PMON_CTL0,      MSR_UNC_C6_PMON_CTR0,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX6C1",  PMC42,  CBOX6,    MSR_UNC_C6_PMON_CTL1,      MSR_UNC_C6_PMON_CTR1,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX6C2",  PMC43,  CBOX6,    MSR_UNC_C6_PMON_CTL2,      MSR_UNC_C6_PMON_CTR2,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX6C3",  PMC44,  CBOX6,    MSR_UNC_C6_PMON_CTL3,      MSR_UNC_C6_PMON_CTR3,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX7C0",  PMC45,  CBOX7,    MSR_UNC_C7_PMON_CTL0,      MSR_UNC_C7_PMON_CTR0,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX7C1",  PMC46,  CBOX7,    MSR_UNC_C7_PMON_CTL1,      MSR_UNC_C7_PMON_CTR1,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX7C2",  PMC47,  CBOX7,    MSR_UNC_C7_PMON_CTL2,      MSR_UNC_C7_PMON_CTR2,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    { "CBOX7C3",  PMC48,  CBOX7,    MSR_UNC_C7_PMON_CTL3,      MSR_UNC_C7_PMON_CTR3,        0,                           0,                          SNBEP_VALID_OPTIONS_CBOX  },
    /* UBOX counters */
    { "UBOX0",    PMC49,  UBOX,     MSR_UNC_U_PMON_CTL0,       MSR_UNC_U_PMON_CTR0,         0,                           0,                          SNBEP_VALID_OPTIONS_UBOX  },
    { "UBOX1",    PMC50,  UBOX,     MSR_UNC_U_PMON_CTL1,       MSR_UNC_U_PMON_CTR0,         0,                           0,                          SNBEP_VALID_OPTIONS_UBOX  },
    { "UBOXFIX",  PMC51,  UBOXFIX,  MSR_UNC_U_UCLK_FIXED_CTL,  MSR_UNC_U_UCLK_FIXED_CTR,    0,                           0,                          EVENT_OPTION_NONE_MASK    },
    { "WBOX0",    PMC52,  WBOX,     MSR_UNC_PCU_PMON_CTL0,     MSR_UNC_PCU_PMON_CTR0,       0,                           0,                          SNBEP_VALID_OPTIONS_WBOX  },
    { "WBOX1",    PMC53,  WBOX,     MSR_UNC_PCU_PMON_CTL1,     MSR_UNC_PCU_PMON_CTR1,       0,                           0,                          SNBEP_VALID_OPTIONS_WBOX  },
    { "WBOX2",    PMC54,  WBOX,     MSR_UNC_PCU_PMON_CTL2,     MSR_UNC_PCU_PMON_CTR2,       0,                           0,                          SNBEP_VALID_OPTIONS_WBOX  },
    { "WBOX3",    PMC55,  WBOX,     MSR_UNC_PCU_PMON_CTL3,     MSR_UNC_PCU_PMON_CTR3,       0,                           0,                          SNBEP_VALID_OPTIONS_WBOX  },
    { "WBOXFIX0", PMC56,  WBOX0FIX, 0,                         MSR_UNC_PCU_PMON_FIXED_CTR0, 0,                           0,                          EVENT_OPTION_NONE_MASK    },
    { "WBOXFIX1", PMC57,  WBOX0FIX, 0,                         MSR_UNC_PCU_PMON_FIXED_CTR1, 0,                           0,                          EVENT_OPTION_NONE_MASK    },
    /* IMC Counters: 4 48bit wide per memory channel, split in two reads */
    { "MBOX0C0",  PMC58,  MBOX0,    PCI_UNC_MC_PMON_CTL_0,     PCI_UNC_MC_PMON_CTR_0_A,     PCI_UNC_MC_PMON_CTR_0_B,     PCI_IMC_DEVICE_0_CH_0,      SNBEP_VALID_OPTIONS_MBOX  },
    { "MBOX0C1",  PMC59,  MBOX0,    PCI_UNC_MC_PMON_CTL_1,     PCI_UNC_MC_PMON_CTR_1_A,     PCI_UNC_MC_PMON_CTR_1_B,     PCI_IMC_DEVICE_0_CH_0,      SNBEP_VALID_OPTIONS_MBOX  },
    { "MBOX0C2",  PMC60,  MBOX0,    PCI_UNC_MC_PMON_CTL_2,     PCI_UNC_MC_PMON_CTR_2_A,     PCI_UNC_MC_PMON_CTR_2_B,     PCI_IMC_DEVICE_0_CH_0,      SNBEP_VALID_OPTIONS_MBOX  },
    { "MBOX0C3",  PMC61,  MBOX0,    PCI_UNC_MC_PMON_CTL_3,     PCI_UNC_MC_PMON_CTR_3_A,     PCI_UNC_MC_PMON_CTR_3_B,     PCI_IMC_DEVICE_0_CH_0,      SNBEP_VALID_OPTIONS_MBOX  },
    { "MBOX0FIX", PMC62,  MBOX0FIX, PCI_UNC_MC_PMON_FIXED_CTL, PCI_UNC_MC_PMON_FIXED_CTR_A, PCI_UNC_MC_PMON_FIXED_CTR_B, PCI_IMC_DEVICE_0_CH_0,      EVENT_OPTION_NONE_MASK    },
    { "MBOX1C0",  PMC63,  MBOX1,    PCI_UNC_MC_PMON_CTL_0,     PCI_UNC_MC_PMON_CTR_0_A,     PCI_UNC_MC_PMON_CTR_0_B,     PCI_IMC_DEVICE_0_CH_1,      SNBEP_VALID_OPTIONS_MBOX  },
    { "MBOX1C1",  PMC64,  MBOX1,    PCI_UNC_MC_PMON_CTL_1,     PCI_UNC_MC_PMON_CTR_1_A,     PCI_UNC_MC_PMON_CTR_1_B,     PCI_IMC_DEVICE_0_CH_1,      SNBEP_VALID_OPTIONS_MBOX  },
    { "MBOX1C2",  PMC65,  MBOX1,    PCI_UNC_MC_PMON_CTL_2,     PCI_UNC_MC_PMON_CTR_2_A,     PCI_UNC_MC_PMON_CTR_2_B,     PCI_IMC_DEVICE_0_CH_1,      SNBEP_VALID_OPTIONS_MBOX  },
    { "MBOX1C3",  PMC66,  MBOX1,    PCI_UNC_MC_PMON_CTL_3,     PCI_UNC_MC_PMON_CTR_3_A,     PCI_UNC_MC_PMON_CTR_3_B,     PCI_IMC_DEVICE_0_CH_1,      SNBEP_VALID_OPTIONS_MBOX  },
    { "MBOX1FIX", PMC67,  MBOX1FIX, PCI_UNC_MC_PMON_FIXED_CTL, PCI_UNC_MC_PMON_FIXED_CTR_A, PCI_UNC_MC_PMON_FIXED_CTR_B, PCI_IMC_DEVICE_0_CH_1,      EVENT_OPTION_NONE_MASK    },
    { "MBOX2C0",  PMC68,  MBOX2,    PCI_UNC_MC_PMON_CTL_0,     PCI_UNC_MC_PMON_CTR_0_A,     PCI_UNC_MC_PMON_CTR_0_B,     PCI_IMC_DEVICE_0_CH_2,      SNBEP_VALID_OPTIONS_MBOX  },
    { "MBOX2C1",  PMC69,  MBOX2,    PCI_UNC_MC_PMON_CTL_1,     PCI_UNC_MC_PMON_CTR_1_A,     PCI_UNC_MC_PMON_CTR_1_B,     PCI_IMC_DEVICE_0_CH_2,      SNBEP_VALID_OPTIONS_MBOX  },
    { "MBOX2C2",  PMC70,  MBOX2,    PCI_UNC_MC_PMON_CTL_2,     PCI_UNC_MC_PMON_CTR_2_A,     PCI_UNC_MC_PMON_CTR_2_B,     PCI_IMC_DEVICE_0_CH_2,      SNBEP_VALID_OPTIONS_MBOX  },
    { "MBOX2C3",  PMC71,  MBOX2,    PCI_UNC_MC_PMON_CTL_3,     PCI_UNC_MC_PMON_CTR_3_A,     PCI_UNC_MC_PMON_CTR_3_B,     PCI_IMC_DEVICE_0_CH_2,      SNBEP_VALID_OPTIONS_MBOX  },
    { "MBOX2FIX", PMC72,  MBOX2FIX, PCI_UNC_MC_PMON_FIXED_CTL, PCI_UNC_MC_PMON_FIXED_CTR_A, PCI_UNC_MC_PMON_FIXED_CTR_B, PCI_IMC_DEVICE_0_CH_2,      EVENT_OPTION_NONE_MASK    },
    { "MBOX3C0",  PMC73,  MBOX3,    PCI_UNC_MC_PMON_CTL_0,     PCI_UNC_MC_PMON_CTR_0_A,     PCI_UNC_MC_PMON_CTR_0_B,     PCI_IMC_DEVICE_0_CH_3,      SNBEP_VALID_OPTIONS_MBOX  },
    { "MBOX3C1",  PMC74,  MBOX3,    PCI_UNC_MC_PMON_CTL_1,     PCI_UNC_MC_PMON_CTR_1_A,     PCI_UNC_MC_PMON_CTR_1_B,     PCI_IMC_DEVICE_0_CH_3,      SNBEP_VALID_OPTIONS_MBOX  },
    { "MBOX3C2",  PMC75,  MBOX3,    PCI_UNC_MC_PMON_CTL_2,     PCI_UNC_MC_PMON_CTR_2_A,     PCI_UNC_MC_PMON_CTR_2_B,     PCI_IMC_DEVICE_0_CH_3,      SNBEP_VALID_OPTIONS_MBOX  },
    { "MBOX3C3",  PMC76,  MBOX3,    PCI_UNC_MC_PMON_CTL_3,     PCI_UNC_MC_PMON_CTR_3_A,     PCI_UNC_MC_PMON_CTR_3_B,     PCI_IMC_DEVICE_0_CH_3,      SNBEP_VALID_OPTIONS_MBOX  },
    { "MBOX3FIX", PMC77,  MBOX3FIX, PCI_UNC_MC_PMON_FIXED_CTL, PCI_UNC_MC_PMON_FIXED_CTR_A, PCI_UNC_MC_PMON_FIXED_CTR_B, PCI_IMC_DEVICE_0_CH_3,      EVENT_OPTION_NONE_MASK    },
    /* QPI counters four 48bit  wide per port, split in two reads */
    { "SBOX0C0",  PMC78,  SBOX0,    PCI_UNC_QPI_PMON_CTL_0,    PCI_UNC_QPI_PMON_CTR_0_A,    PCI_UNC_QPI_PMON_CTR_0_B,    PCI_QPI_DEVICE_PORT_0,      SNBEP_VALID_OPTIONS_SBOX  },
    { "SBOX0C1",  PMC79,  SBOX0,    PCI_UNC_QPI_PMON_CTL_1,    PCI_UNC_QPI_PMON_CTR_1_A,    PCI_UNC_QPI_PMON_CTR_1_B,    PCI_QPI_DEVICE_PORT_0,      SNBEP_VALID_OPTIONS_SBOX  },
    { "SBOX0C2",  PMC80,  SBOX0,    PCI_UNC_QPI_PMON_CTL_2,    PCI_UNC_QPI_PMON_CTR_2_A,    PCI_UNC_QPI_PMON_CTR_2_B,    PCI_QPI_DEVICE_PORT_0,      SNBEP_VALID_OPTIONS_SBOX  },
    { "SBOX0C3",  PMC81,  SBOX0,    PCI_UNC_QPI_PMON_CTL_3,    PCI_UNC_QPI_PMON_CTR_3_A,    PCI_UNC_QPI_PMON_CTR_3_B,    PCI_QPI_DEVICE_PORT_0,      SNBEP_VALID_OPTIONS_SBOX  },
    { "SBOX0FIX", PMC82,  SBOX0FIX, 0,                         PCI_UNC_QPI_RATE_STATUS,     0,                           PCI_QPI_MISC_DEVICE_PORT_0, EVENT_OPTION_NONE_MASK    },
    { "SBOX1C0",  PMC83,  SBOX1,    PCI_UNC_QPI_PMON_CTL_0,    PCI_UNC_QPI_PMON_CTR_0_A,    PCI_UNC_QPI_PMON_CTR_0_B,    PCI_QPI_DEVICE_PORT_1,      SNBEP_VALID_OPTIONS_SBOX  },
    { "SBOX1C1",  PMC84,  SBOX1,    PCI_UNC_QPI_PMON_CTL_1,    PCI_UNC_QPI_PMON_CTR_1_A,    PCI_UNC_QPI_PMON_CTR_1_B,    PCI_QPI_DEVICE_PORT_1,      SNBEP_VALID_OPTIONS_SBOX  },
    { "SBOX1C2",  PMC85,  SBOX1,    PCI_UNC_QPI_PMON_CTL_2,    PCI_UNC_QPI_PMON_CTR_2_A,    PCI_UNC_QPI_PMON_CTR_2_B,    PCI_QPI_DEVICE_PORT_1,      SNBEP_VALID_OPTIONS_SBOX  },
    { "SBOX1C3",  PMC86,  SBOX1,    PCI_UNC_QPI_PMON_CTL_3,    PCI_UNC_QPI_PMON_CTR_3_A,    PCI_UNC_QPI_PMON_CTR_3_B,    PCI_QPI_DEVICE_PORT_1,      SNBEP_VALID_OPTIONS_SBOX  },
    { "SBOX1FIX", PMC87,  SBOX1FIX, 0,                         PCI_UNC_QPI_RATE_STATUS,     0,                           PCI_QPI_MISC_DEVICE_PORT_1, EVENT_OPTION_NONE_MASK    },
    /* BBOX or better known as Home Agent (HA) */
    { "BBOX0",    PMC88,  BBOX0,    PCI_UNC_HA_PMON_CTL_0,     PCI_UNC_HA_PMON_CTR_0_A,     PCI_UNC_HA_PMON_CTR_0_B,     PCI_HA_DEVICE_0,            SNBEP_VALID_OPTIONS_BBOX  },
    { "BBOX1",    PMC89,  BBOX0,    PCI_UNC_HA_PMON_CTL_1,     PCI_UNC_HA_PMON_CTR_1_A,     PCI_UNC_HA_PMON_CTR_1_B,     PCI_HA_DEVICE_0,            SNBEP_VALID_OPTIONS_BBOX  },
    { "BBOX2",    PMC90,  BBOX0,    PCI_UNC_HA_PMON_CTL_2,     PCI_UNC_HA_PMON_CTR_2_A,     PCI_UNC_HA_PMON_CTR_2_B,     PCI_HA_DEVICE_0,            SNBEP_VALID_OPTIONS_BBOX  },
    { "BBOX3",    PMC91,  BBOX0,    PCI_UNC_HA_PMON_CTL_3,     PCI_UNC_HA_PMON_CTR_3_A,     PCI_UNC_HA_PMON_CTR_3_B,     PCI_HA_DEVICE_0,            SNBEP_VALID_OPTIONS_BBOX  },
    { "RBOX0C0",  PMC92,  RBOX0,    PCI_UNC_R3QPI_PMON_CTL_0,  PCI_UNC_R3QPI_PMON_CTR_0_A,  PCI_UNC_R3QPI_PMON_CTR_0_B,  PCI_R3QPI_DEVICE_LINK_0,    SNBEP_VALID_OPTIONS_RBOX  },
    { "RBOX0C1",  PMC93,  RBOX0,    PCI_UNC_R3QPI_PMON_CTL_1,  PCI_UNC_R3QPI_PMON_CTR_1_A,  PCI_UNC_R3QPI_PMON_CTR_1_B,  PCI_R3QPI_DEVICE_LINK_0,    SNBEP_VALID_OPTIONS_RBOX  },
    { "RBOX0C2",  PMC94,  RBOX0,    PCI_UNC_R3QPI_PMON_CTL_2,  PCI_UNC_R3QPI_PMON_CTR_2_A,  PCI_UNC_R3QPI_PMON_CTR_2_B,  PCI_R3QPI_DEVICE_LINK_0,    SNBEP_VALID_OPTIONS_RBOX  },
    { "RBOX1C0",  PMC95,  RBOX1,    PCI_UNC_R3QPI_PMON_CTL_0,  PCI_UNC_R3QPI_PMON_CTR_0_A,  PCI_UNC_R3QPI_PMON_CTR_0_B,  PCI_R3QPI_DEVICE_LINK_1,    SNBEP_VALID_OPTIONS_RBOX  },
    { "RBOX1C1",  PMC96,  RBOX1,    PCI_UNC_R3QPI_PMON_CTL_1,  PCI_UNC_R3QPI_PMON_CTR_1_A,  PCI_UNC_R3QPI_PMON_CTR_1_B,  PCI_R3QPI_DEVICE_LINK_1,    SNBEP_VALID_OPTIONS_RBOX  },
    { "RBOX1C2",  PMC97,  RBOX1,    PCI_UNC_R3QPI_PMON_CTL_2,  PCI_UNC_R3QPI_PMON_CTR_2_A,  PCI_UNC_R3QPI_PMON_CTR_2_B,  PCI_R3QPI_DEVICE_LINK_1,    SNBEP_VALID_OPTIONS_RBOX  },
    { "PBOX0",    PMC98,  PBOX,     PCI_UNC_R2PCIE_PMON_CTL_0, PCI_UNC_R2PCIE_PMON_CTR_0_A, PCI_UNC_R2PCIE_PMON_CTR_0_B, PCI_R2PCIE_DEVICE,          SNBEP_VALID_OPTIONS_PBOX  },
    { "PBOX1",    PMC99,  PBOX,     PCI_UNC_R2PCIE_PMON_CTL_1, PCI_UNC_R2PCIE_PMON_CTR_1_A, PCI_UNC_R2PCIE_PMON_CTR_1_B, PCI_R2PCIE_DEVICE,          SNBEP_VALID_OPTIONS_PBOX  },
    { "PBOX2",    PMC100, PBOX,     PCI_UNC_R2PCIE_PMON_CTL_2, PCI_UNC_R2PCIE_PMON_CTR_2_A, PCI_UNC_R2PCIE_PMON_CTR_2_B, PCI_R2PCIE_DEVICE,          SNBEP_VALID_OPTIONS_PBOX  },
    { "PBOX3",    PMC101, PBOX,     PCI_UNC_R2PCIE_PMON_CTL_3, PCI_UNC_R2PCIE_PMON_CTR_3_A, PCI_UNC_R2PCIE_PMON_CTR_3_B, PCI_R2PCIE_DEVICE,          SNBEP_VALID_OPTIONS_PBOX  },
};

static BoxMap sandybridgeEP_box_map[NUM_UNITS] = {
    [PMC]      = { MSR_PERF_GLOBAL_CTRL,        MSR_PERF_GLOBAL_STATUS,    MSR_PERF_GLOBAL_OVF_CTRL, 0, 0, MSR_DEV,                    48, 0,                          0 },
    [FIXED]    = { MSR_PERF_GLOBAL_CTRL,        MSR_PERF_GLOBAL_STATUS,    MSR_PERF_GLOBAL_OVF_CTRL, 0, 0, MSR_DEV,                    48, 0,                          0 },
    [THERMAL]  = { 0,                           0,                         0,                        0, 0, MSR_DEV,                    8,  0,                          0 },
    [VOLTAGE]  = { 0,                           0,                         0,                        0, 0, 0,                          16, 0,                          0 },
    [POWER]    = { 0,                           0,                         0,                        0, 0, MSR_DEV,                    32, 0,                          0 },
    [WBOX]     = { MSR_UNC_PCU_PMON_BOX_CTL,    0,                         0,                        0, 0, MSR_DEV,                    48, 0,                          0 },
    [WBOX0FIX] = { 0,                           0,                         0,                        0, 0, MSR_DEV,                    48, 0,                          0 },
    [UBOX]     = { 0,                           MSR_UNC_U_PMON_BOX_STATUS, 0,                        0, 0, MSR_DEV,                    44, 0,                          0 },
    [UBOXFIX]  = { 0,                           0,                         0,                        0, 0, MSR_DEV,                    44, 0,                          0 },
    [CBOX0]    = { MSR_UNC_C0_PMON_BOX_CTL,     0,                         0,                        0, 0, MSR_DEV,                    44, MSR_UNC_C0_PMON_BOX_FILTER, 0 },
    [CBOX1]    = { MSR_UNC_C1_PMON_BOX_CTL,     0,                         0,                        0, 0, MSR_DEV,                    44, MSR_UNC_C1_PMON_BOX_FILTER, 0 },
    [CBOX2]    = { MSR_UNC_C2_PMON_BOX_CTL,     0,                         0,                        0, 0, MSR_DEV,                    44, MSR_UNC_C2_PMON_BOX_FILTER, 0 },
    [CBOX3]    = { MSR_UNC_C3_PMON_BOX_CTL,     0,                         0,                        0, 0, MSR_DEV,                    44, MSR_UNC_C3_PMON_BOX_FILTER, 0 },
    [CBOX4]    = { MSR_UNC_C4_PMON_BOX_CTL,     0,                         0,                        0, 0, MSR_DEV,                    44, MSR_UNC_C4_PMON_BOX_FILTER, 0 },
    [CBOX5]    = { MSR_UNC_C5_PMON_BOX_CTL,     0,                         0,                        0, 0, MSR_DEV,                    44, MSR_UNC_C5_PMON_BOX_FILTER, 0 },
    [CBOX6]    = { MSR_UNC_C6_PMON_BOX_CTL,     0,                         0,                        0, 0, MSR_DEV,                    44, MSR_UNC_C6_PMON_BOX_FILTER, 0 },
    [CBOX7]    = { MSR_UNC_C7_PMON_BOX_CTL,     0,                         0,                        0, 0, MSR_DEV,                    44, MSR_UNC_C7_PMON_BOX_FILTER, 0 },
    [MBOX0]    = { PCI_UNC_MC_PMON_BOX_CTL,     0,                         0,                        0, 1, PCI_IMC_DEVICE_0_CH_0,      48, 0,                          0 },
    [MBOX1]    = { PCI_UNC_MC_PMON_BOX_CTL,     0,                         0,                        0, 1, PCI_IMC_DEVICE_0_CH_1,      48, 0,                          0 },
    [MBOX2]    = { PCI_UNC_MC_PMON_BOX_CTL,     0,                         0,                        0, 1, PCI_IMC_DEVICE_0_CH_2,      48, 0,                          0 },
    [MBOX3]    = { PCI_UNC_MC_PMON_BOX_CTL,     0,                         0,                        0, 1, PCI_IMC_DEVICE_0_CH_3,      48, 0,                          0 },
    [MBOX0FIX] = { 0,                           0,                         0,                        0, 1, PCI_IMC_DEVICE_0_CH_0,      48, 0,                          0 },
    [MBOX1FIX] = { 0,                           0,                         0,                        0, 1, PCI_IMC_DEVICE_0_CH_1,      48, 0,                          0 },
    [MBOX2FIX] = { 0,                           0,                         0,                        0, 1, PCI_IMC_DEVICE_0_CH_2,      48, 0,                          0 },
    [MBOX3FIX] = { 0,                           0,                         0,                        0, 1, PCI_IMC_DEVICE_0_CH_3,      48, 0,                          0 },
    [BBOX0]    = { PCI_UNC_HA_PMON_BOX_CTL,     0,                         0,                        0, 1, PCI_HA_DEVICE_0,            48, 0,                          0 },
    [SBOX0]    = { PCI_UNC_QPI_PMON_BOX_CTL,    0,                         0,                        0, 1, PCI_QPI_DEVICE_PORT_0,      48, 0,                          0 },
    [SBOX1]    = { PCI_UNC_QPI_PMON_BOX_CTL,    0,                         0,                        0, 1, PCI_QPI_DEVICE_PORT_1,      48, 0,                          0 },
    [SBOX0FIX] = { 0,                           0,                         0,                        0, 1, PCI_QPI_MISC_DEVICE_PORT_0, 64, 0,                          0 },
    [SBOX1FIX] = { 0,                           0,                         0,                        0, 1, PCI_QPI_MISC_DEVICE_PORT_1, 64, 0,                          0 },
    [RBOX0]    = { PCI_UNC_R3QPI_PMON_BOX_CTL,  0,                         0,                        0, 1, PCI_R3QPI_DEVICE_LINK_0,    44, 0,                          0 },
    [RBOX1]    = { PCI_UNC_R3QPI_PMON_BOX_CTL,  0,                         0,                        0, 1, PCI_R3QPI_DEVICE_LINK_1,    44, 0,                          0 },
    [PBOX]     = { PCI_UNC_R2PCIE_PMON_BOX_CTL, 0,                         0,                        0, 1, PCI_R2PCIE_DEVICE,          44, 0,                          0 },
};

static PciDevice sandybridgeEP_pci_devices[MAX_NUM_PCI_DEVICES] = {
    [MSR_DEV]                    = { NODEVTYPE, "",     "",                           "",         0,      0 },
    [PCI_R3QPI_DEVICE_LINK_0]    = { R3QPI,     "13.5", "PCI_R3QPI_DEVICE_LINK_0",    "RBOX0",    0x3c44, 0 },
    [PCI_R3QPI_DEVICE_LINK_1]    = { R3QPI,     "13.6", "PCI_R3QPI_DEVICE_LINK_1",    "RBOX1",    0x3c45, 0 },
    [PCI_R2PCIE_DEVICE]          = { R2PCIE,    "13.1", "PCI_R2PCIE_DEVICE",          "PBOX0",    0x3c43, 0 },
    [PCI_IMC_DEVICE_0_CH_0]      = { IMC,       "10.0", "PCI_IMC_DEVICE_CH_0",        "MBOX0",    0x3cb0, 0 },
    [PCI_IMC_DEVICE_0_CH_1]      = { IMC,       "10.1", "PCI_IMC_DEVICE_CH_1",        "MBOX1",    0x3cb1, 0 },
    [PCI_IMC_DEVICE_0_CH_2]      = { IMC,       "10.4", "PCI_IMC_DEVICE_CH_2",        "MBOX2",    0x3cb4, 0 },
    [PCI_IMC_DEVICE_0_CH_3]      = { IMC,       "10.5", "PCI_IMC_DEVICE_CH_3",        "MBOX3",    0x3cb5, 0 },
    [PCI_HA_DEVICE_0]            = { HA,        "0e.1", "PCI_HA_DEVICE",              "BBOX",     0x3c46, 0 },
    [PCI_QPI_DEVICE_PORT_0]      = { QPI,       "08.2", "PCI_QPI_DEVICE_PORT_0",      "SBOX0",    0x3c41, 0 },
    [PCI_QPI_DEVICE_PORT_1]      = { QPI,       "09.2", "PCI_QPI_DEVICE_PORT_1",      "SBOX1",    0x3c42, 0 },
    [PCI_QPI_MASK_DEVICE_PORT_0] = { QPI,       "08.6", "PCI_QPI_MASK_DEVICE_PORT_0", NULL,       0x3c86, 0 },
    [PCI_QPI_MASK_DEVICE_PORT_1] = { QPI,       "09.6", "PCI_QPI_MASK_DEVICE_PORT_1", NULL,       0x3c96, 0 },
    [PCI_QPI_MISC_DEVICE_PORT_0] = { QPI,       "08.0", "PCI_QPI_MISC_DEVICE_PORT_0", "SBOX0FIX", 0x3c80, 0 },
    [PCI_QPI_MISC_DEVICE_PORT_1] = { QPI,       "09.0", "PCI_QPI_MISC_DEVICE_PORT_1", "SBOX1FIX", 0x3c91, 0 },
};

static char *sandybridgeEP_translate_types[NUM_UNITS] = {
    [FIXED]   = "/sys/bus/event_source/devices/cpu",
    [PMC]     = "/sys/bus/event_source/devices/cpu",
    [POWER]   = "/sys/bus/event_source/devices/power",
    [MBOX0]   = "/sys/bus/event_source/devices/uncore_imc_0",
    [MBOX1]   = "/sys/bus/event_source/devices/uncore_imc_1",
    [MBOX2]   = "/sys/bus/event_source/devices/uncore_imc_2",
    [MBOX3]   = "/sys/bus/event_source/devices/uncore_imc_3",
    [MBOX4]   = "/sys/bus/event_source/devices/uncore_imc_4",
    [MBOX5]   = "/sys/bus/event_source/devices/uncore_imc_5",
    [MBOX6]   = "/sys/bus/event_source/devices/uncore_imc_6",
    [MBOX7]   = "/sys/bus/event_source/devices/uncore_imc_7",
    [CBOX0]   = "/sys/bus/event_source/devices/uncore_cbox_0",
    [CBOX1]   = "/sys/bus/event_source/devices/uncore_cbox_1",
    [CBOX2]   = "/sys/bus/event_source/devices/uncore_cbox_2",
    [CBOX3]   = "/sys/bus/event_source/devices/uncore_cbox_3",
    [CBOX4]   = "/sys/bus/event_source/devices/uncore_cbox_4",
    [CBOX5]   = "/sys/bus/event_source/devices/uncore_cbox_5",
    [CBOX6]   = "/sys/bus/event_source/devices/uncore_cbox_6",
    [CBOX7]   = "/sys/bus/event_source/devices/uncore_cbox_7",
    [BBOX0]   = "/sys/bus/event_source/devices/uncore_ha",
    [WBOX]    = "/sys/bus/event_source/devices/uncore_pcu",
    [SBOX0]   = "/sys/bus/event_source/devices/uncore_qpi_0",
    [SBOX1]   = "/sys/bus/event_source/devices/uncore_qpi_1",
    [PBOX]    = "/sys/bus/event_source/devices/uncore_r2pcie",
    [RBOX0]   = "/sys/bus/event_source/devices/uncore_r3qpi_0",
    [RBOX1]   = "/sys/bus/event_source/devices/uncore_r3qpi_1",
    [UBOX]    = "/sys/bus/event_source/devices/uncore_ubox",
    [UBOXFIX] = "/sys/bus/event_source/devices/uncore_ubox",
};

#endif //PERFMON_SANDYBRIDGEEP_COUNTERS_H
