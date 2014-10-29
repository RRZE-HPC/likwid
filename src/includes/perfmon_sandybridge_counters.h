/*
 * =======================================================================================
 *
 *      Filename:  perfmon_sandybridge_counters.h
 *
 *      Description: Counter header file of perfmon module for Sandy Bridge.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
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

#define NUM_COUNTERS_SANDYBRIDGE 29
#define NUM_COUNTERS_UNCORE_SANDYBRIDGE 21
#define NUM_COUNTERS_CORE_SANDYBRIDGE 8

static PerfmonCounterMap sandybridge_counter_map[NUM_COUNTERS_SANDYBRIDGE] = {
    /* Fixed Counters: instructions retired, cycles unhalted core */
    {"FIXC0", PMC0, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR0, 0, 0},
    {"FIXC1", PMC1, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR1, 0, 0},
    {"FIXC2", PMC2, FIXED, MSR_PERF_FIXED_CTR_CTRL, MSR_PERF_FIXED_CTR2, 0, 0},
    /* PMC Counters: 4 48bit wide */
    {"PMC0", PMC3, PMC, MSR_PERFEVTSEL0, MSR_PMC0, 0, 0},
    {"PMC1", PMC4, PMC, MSR_PERFEVTSEL1, MSR_PMC1, 0, 0},
    {"PMC2", PMC5, PMC, MSR_PERFEVTSEL2, MSR_PMC2, 0, 0},
    {"PMC3", PMC6, PMC, MSR_PERFEVTSEL3, MSR_PMC3, 0, 0},
    /* Temperature Sensor*/
    {"TMP0", PMC7, THERMAL, 0, 0, 0, 0},
    /* RAPL counters */
    {"PWR0", PMC8, POWER, 0, MSR_PKG_ENERGY_STATUS, 0, 0},
    {"PWR1", PMC9, POWER, 0, MSR_PP0_ENERGY_STATUS, 0, 0},
    {"PWR2", PMC10, POWER, 0, MSR_PP1_ENERGY_STATUS,  0, 0},
    {"PWR3", PMC11, POWER, 0, MSR_DRAM_ENERGY_STATUS,  0, 0},
    /* IMC Counters: 4 48bit wide per memory channel, split in two reads */
    {"MBOX0C0",PMC12, MBOX0, PCI_UNC_MC_PMON_CTL_0, PCI_UNC_MC_PMON_CTR_0_A, PCI_UNC_MC_PMON_CTR_0_B, PCI_IMC_DEVICE_CH_0},
    {"MBOX0C1",PMC13, MBOX0, PCI_UNC_MC_PMON_CTL_1, PCI_UNC_MC_PMON_CTR_1_A, PCI_UNC_MC_PMON_CTR_1_B, PCI_IMC_DEVICE_CH_0},
    {"MBOX0C2",PMC14, MBOX0, PCI_UNC_MC_PMON_CTL_2, PCI_UNC_MC_PMON_CTR_2_A, PCI_UNC_MC_PMON_CTR_2_B, PCI_IMC_DEVICE_CH_0},
    {"MBOX0C3",PMC15, MBOX0, PCI_UNC_MC_PMON_CTL_3, PCI_UNC_MC_PMON_CTR_3_A, PCI_UNC_MC_PMON_CTR_3_B, PCI_IMC_DEVICE_CH_0},
    {"MBOX1C0",PMC16, MBOX1, PCI_UNC_MC_PMON_CTL_0, PCI_UNC_MC_PMON_CTR_0_A, PCI_UNC_MC_PMON_CTR_0_B, PCI_IMC_DEVICE_CH_1},
    {"MBOX1C1",PMC17, MBOX1, PCI_UNC_MC_PMON_CTL_1, PCI_UNC_MC_PMON_CTR_1_A, PCI_UNC_MC_PMON_CTR_1_B, PCI_IMC_DEVICE_CH_1},
    {"MBOX1C2",PMC18, MBOX1, PCI_UNC_MC_PMON_CTL_2, PCI_UNC_MC_PMON_CTR_2_A, PCI_UNC_MC_PMON_CTR_2_B, PCI_IMC_DEVICE_CH_1},
    {"MBOX1C3",PMC19, MBOX1, PCI_UNC_MC_PMON_CTL_3, PCI_UNC_MC_PMON_CTR_3_A, PCI_UNC_MC_PMON_CTR_3_B, PCI_IMC_DEVICE_CH_1},
    {"MBOX2C0",PMC20, MBOX2, PCI_UNC_MC_PMON_CTL_0, PCI_UNC_MC_PMON_CTR_0_A, PCI_UNC_MC_PMON_CTR_0_B, PCI_IMC_DEVICE_CH_2},
    {"MBOX2C1",PMC21, MBOX2, PCI_UNC_MC_PMON_CTL_1, PCI_UNC_MC_PMON_CTR_1_A, PCI_UNC_MC_PMON_CTR_1_B, PCI_IMC_DEVICE_CH_2},
    {"MBOX2C2",PMC22, MBOX2, PCI_UNC_MC_PMON_CTL_2, PCI_UNC_MC_PMON_CTR_2_A, PCI_UNC_MC_PMON_CTR_2_B, PCI_IMC_DEVICE_CH_2},
    {"MBOX2C3",PMC23, MBOX2, PCI_UNC_MC_PMON_CTL_3, PCI_UNC_MC_PMON_CTR_3_A, PCI_UNC_MC_PMON_CTR_3_B, PCI_IMC_DEVICE_CH_2},
    {"MBOX3C0",PMC24, MBOX3, PCI_UNC_MC_PMON_CTL_0, PCI_UNC_MC_PMON_CTR_0_A, PCI_UNC_MC_PMON_CTR_0_B, PCI_IMC_DEVICE_CH_3},
    {"MBOX3C1",PMC25, MBOX3, PCI_UNC_MC_PMON_CTL_1, PCI_UNC_MC_PMON_CTR_1_A, PCI_UNC_MC_PMON_CTR_1_B, PCI_IMC_DEVICE_CH_3},
    {"MBOX3C2",PMC26, MBOX3, PCI_UNC_MC_PMON_CTL_2, PCI_UNC_MC_PMON_CTR_2_A, PCI_UNC_MC_PMON_CTR_2_B, PCI_IMC_DEVICE_CH_3},
    {"MBOX3C3",PMC27, MBOX3, PCI_UNC_MC_PMON_CTL_3, PCI_UNC_MC_PMON_CTR_3_A, PCI_UNC_MC_PMON_CTR_3_B, PCI_IMC_DEVICE_CH_3},
    {"MBOXFIX",PMC28, MBOXFIX, 0, PCI_UNC_MC_PMON_FIXED_CTR_A, PCI_UNC_MC_PMON_FIXED_CTR_B, PCI_UNC_MC_PMON_FIXED_CTL},
};


