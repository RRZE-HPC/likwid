/*
 * =======================================================================================
 *
 *      Filename:  perfmon_nvidiagrace_counters.h
 *
 *      Description:  Counter Header File of perfmon module for Nvidia Grace CPU.
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


#define NUM_COUNTERS_NVIDIAGRACE 41

static RegisterMap nvidiagrace_counter_map[NUM_COUNTERS_NVIDIAGRACE] = {
    {"PMC0", PMC0, PMC, A57_PERFEVTSEL0, A57_PMC0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC1", PMC1, PMC, A57_PERFEVTSEL1, A57_PMC1, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC2", PMC2, PMC, A57_PERFEVTSEL2, A57_PMC2, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC3", PMC3, PMC, A57_PERFEVTSEL3, A57_PMC3, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC4", PMC4, PMC, A57_PERFEVTSEL4, A57_PMC4, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC5", PMC5, PMC, A57_PERFEVTSEL5, A57_PMC5, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SCFFIX", PMC6,MBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SCF0", PMC7, MBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SCF1", PMC8, MBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SCF2", PMC9, MBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SCF3", PMC10, MBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SCF4", PMC11, MBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SCF5", PMC12, MBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CNV0", PMC13, BBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CNV1", PMC14, BBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CNV2", PMC15, BBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CNV3", PMC16, BBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CNV4", PMC17, BBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CNV5", PMC18, BBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CNVFIX", PMC19, BBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"NV0C0", PMC20, QBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"NV0C1", PMC21, QBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"NV0C2", PMC22, QBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"NV0C3", PMC23, QBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"NV0C4", PMC24, QBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"NV0C5", PMC25, QBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"NV0FIX", PMC26, QBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"NV1C0", PMC27, SBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"NV1C1", PMC28, SBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"NV1C2", PMC29, SBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"NV1C3", PMC30, SBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"NV1C4", PMC31, SBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"NV1C5", PMC32, SBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"NV1FIX", PMC33, SBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PCIE0", PMC34, PBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PCIE1", PMC35, PBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PCIE2", PMC36, PBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PCIE3", PMC37, PBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PCIE4", PMC38, PBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PCIE5", PMC39, PBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PCIEFIX", PMC40, PBOX0, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
};

static BoxMap nvidiagrace_box_map[NUM_UNITS] = {
    [PMC] = {A57_PERF_CONTROL_CTRL, A57_OVERFLOW_STATUS, A57_OVERFLOW_FLAGS, 0, 0, 0, 32},
    [MBOX0] = {0, 0, 0, 0, 0, 0, 32},
    [BBOX0] = {0, 0, 0, 0, 0, 0, 32},
    [QBOX0] = {0, 0, 0, 0, 0, 0, 32},
    [SBOX0] = {0, 0, 0, 0, 0, 0, 32},
    [PBOX0] = {0, 0, 0, 0, 0, 0, 32},
    [MBOX1] = {0, 0, 0, 0, 0, 0, 32},
    [BBOX1] = {0, 0, 0, 0, 0, 0, 32},
    [QBOX1] = {0, 0, 0, 0, 0, 0, 32},
    [SBOX1] = {0, 0, 0, 0, 0, 0, 32},
    [PBOX1] = {0, 0, 0, 0, 0, 0, 32},
};

static char* nvidiagrace_translate_types[NUM_UNITS] = {
    [PMC] = "/sys/bus/event_source/devices/armv8_pmuv3_0",
    [BBOX0] = "/sys/bus/event_source/devices/nvidia_cnvlink_pmu_0",
    [BBOX1] = "/sys/bus/event_source/devices/nvidia_cnvlink_pmu_1",
    [QBOX0] = "/sys/bus/event_source/devices/nvidia_nvlink_c2c0_pmu_0",
    [QBOX1] = "/sys/bus/event_source/devices/nvidia_nvlink_c2c0_pmu_1",
    [SBOX0] = "/sys/bus/event_source/devices/nvidia_nvlink_c2c1_pmu_0",
    [SBOX1] = "/sys/bus/event_source/devices/nvidia_nvlink_c2c1_pmu_1",
    [PBOX0] = "/sys/bus/event_source/devices/nvidia_pcie_pmu_0",
    [PBOX1] = "/sys/bus/event_source/devices/nvidia_pcie_pmu_1",
    [MBOX0] = "/sys/bus/event_source/devices/nvidia_scf_pmu_0",
    [MBOX1] = "/sys/bus/event_source/devices/nvidia_scf_pmu_1",
};

