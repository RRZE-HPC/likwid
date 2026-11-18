/*
 * =======================================================================================
 *
 *      Filename:  perfmon_nvidiavera_counters.h
 *
 *      Description:  Counter Header File of perfmon module for Nvidia Vera CPU.
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
#ifndef PERFMON_NVIDIAVERA_COUNTERS_H
#define PERFMON_NVIDIAVERA_COUNTERS_H

#define NUM_COUNTERS_NVIDIAVERA 109

#define EVENT_OPTIONS_PMC EVENT_OPTION_THRESHOLD_MASK|EVENT_OPTION_THRESHOLD_COMPARE_MASK|EVENT_OPTION_THRESHOLD_COUNT_MASK
#define EVENT_OPTIONS_UCF EVENT_OPTION_DESTINATION_CMEM_MASK|EVENT_OPTION_DESTINATION_GMEM_MASK|EVENT_OPTION_DESTINATION_OTHER_MASK|EVENT_OPTION_DESTINATION_REMOTE_MASK|EVENT_OPTION_SOURCE_CPU_MASK|EVENT_OPTION_SOURCE_NONCPU_MASK|EVENT_OPTION_SOURCE_REMOTE_MASK
#define EVENT_OPTIONS_PCIE EVENT_OPTION_DESTINATION_CMEM_MASK|EVENT_OPTION_DESTINATION_GMEM_MASK|EVENT_OPTION_DESTINATION_REMOTE_MASK|EVENT_OPTION_DESTINATION_PCIE_CXL_MASK|EVENT_OPTION_DESTINATION_PCIE_P2P_MASK|EVENT_OPTION_SOURCE_BDF_MASK|EVENT_OPTION_SOURCE_BDF_EN_MASK|EVENT_OPTION_SOURCE_RP_MASK_MASK
#define EVENT_OPTIONS_PCIETARGET EVENT_OPTION_DESTINATION_ADDR_BASE_MASK|EVENT_OPTION_DESTINATION_ADDR_MASK_MASK|EVENT_OPTION_DESTINATION_ADDR_EN_MASK|EVENT_OPTION_DESTINATION_RP_MASK_MASK
#define EVENT_OPTIONS_NVCLINK EVENT_OPTION_REMOTE_SOCKET_MASK
#define EVENT_OPTIONS_C2C EVENT_OPTION_GPU_INDEX_MASK

static RegisterMap nvidiavera_counter_map[NUM_COUNTERS_NVIDIAVERA] = {
    {"PMC0", PMC0, PMC, A57_PERFEVTSEL0, A57_PMC0, 0, 0, EVENT_OPTIONS_PMC},
    {"PMC1", PMC1, PMC, A57_PERFEVTSEL1, A57_PMC1, 0, 0, EVENT_OPTIONS_PMC},
    {"PMC2", PMC2, PMC, A57_PERFEVTSEL2, A57_PMC2, 0, 0, EVENT_OPTIONS_PMC},
    {"PMC3", PMC3, PMC, A57_PERFEVTSEL3, A57_PMC3, 0, 0, EVENT_OPTIONS_PMC},
    {"PMC4", PMC4, PMC, A57_PERFEVTSEL4, A57_PMC4, 0, 0, EVENT_OPTIONS_PMC},
    {"PMC5", PMC5, PMC, A57_PERFEVTSEL5, A57_PMC5, 0, 0, EVENT_OPTIONS_PMC},
    {"UCF0", PMC6, MBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_UCF},
    {"UCF1", PMC7, MBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_UCF},
    {"UCF2", PMC8, MBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_UCF},
    {"UCF3", PMC9, MBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_UCF},
    {"UCF4", PMC10, MBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_UCF},
    {"UCF5", PMC11, MBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_UCF},
    {"PCIE0C0", PMC12, PBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE0C1", PMC13, PBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE0C2", PMC14, PBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE0C3", PMC15, PBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE0C4", PMC16, PBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE0C5", PMC17, PBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE1C0", PMC18, PBOX2, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE1C1", PMC19, PBOX2, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE1C2", PMC20, PBOX2, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE1C3", PMC21, PBOX2, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE1C4", PMC22, PBOX2, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE1C5", PMC23, PBOX2, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE2C0", PMC24, PBOX4, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE2C1", PMC25, PBOX4, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE2C2", PMC26, PBOX4, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE2C3", PMC27, PBOX4, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE2C4", PMC28, PBOX4, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE2C5", PMC29, PBOX4, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE3C0", PMC30, PBOX6, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE3C1", PMC31, PBOX6, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE3C2", PMC32, PBOX6, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE3C3", PMC33, PBOX6, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE3C4", PMC34, PBOX6, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE3C5", PMC35, PBOX6, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE4C0", PMC36, PBOX8, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE4C1", PMC37, PBOX8, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE4C2", PMC38, PBOX8, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE4C3", PMC39, PBOX8, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE4C4", PMC40, PBOX8, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE4C5", PMC41, PBOX8, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE5C0", PMC42, PBOX10, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE5C1", PMC43, PBOX10, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE5C2", PMC44, PBOX10, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE5C3", PMC45, PBOX10, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE5C4", PMC46, PBOX10, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"PCIE5C5", PMC47, PBOX10, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIE},
    {"CXL0", PMC48, QBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CXL1", PMC49, QBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CXL2", PMC50, QBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CXL3", PMC51, QBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CXL4", PMC52, QBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CXL5", PMC53, QBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PCIET0C0", PMC54, BBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET0C1", PMC55, BBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET0C2", PMC56, BBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET0C3", PMC57, BBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET0C4", PMC58, BBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET0C5", PMC59, BBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET1C0", PMC60, BBOX2, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET1C1", PMC61, BBOX2, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET1C2", PMC62, BBOX2, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET1C3", PMC63, BBOX2, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET1C4", PMC64, BBOX2, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET1C5", PMC65, BBOX2, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET2C0", PMC66, BBOX4, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET2C1", PMC67, BBOX4, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET2C2", PMC68, BBOX4, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET2C3", PMC69, BBOX4, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET2C4", PMC70, BBOX4, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET2C5", PMC71, BBOX4, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET3C0", PMC72, BBOX6, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET3C1", PMC73, BBOX6, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET3C2", PMC74, BBOX6, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET3C3", PMC75, BBOX6, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET3C4", PMC76, BBOX6, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET3C5", PMC77, BBOX6, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET4C0", PMC78, BBOX8, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET4C1", PMC79, BBOX8, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET4C2", PMC80, BBOX8, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET4C3", PMC81, BBOX8, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET4C4", PMC82, BBOX8, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET4C5", PMC83, BBOX8, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET5C0", PMC84, BBOX10, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET5C1", PMC85, BBOX10, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET5C2", PMC86, BBOX10, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET5C3", PMC87, BBOX10, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET5C4", PMC88, BBOX10, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"PCIET5C5", PMC89, BBOX10, 0x0, 0x0, 0, 0, EVENT_OPTIONS_PCIETARGET},
    {"NVC0", PMC90, SBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_NVCLINK},
    {"NVC1", PMC91, SBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_NVCLINK},
    {"NVC2", PMC92, SBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_NVCLINK},
    {"NVC3", PMC93, SBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_NVCLINK},
    {"NVC4", PMC94, SBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_NVCLINK},
    {"NVC5", PMC95, SBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_NVCLINK},
    {"CLAT0", PMC96, WBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CLAT1", PMC97, WBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CLAT2", PMC98, WBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CLAT3", PMC99, WBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CLAT4", PMC100, WBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CLAT5", PMC101, WBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"C2C0", PMC102, CBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_C2C},
    {"C2C1", PMC103, CBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_C2C},
    {"C2C2", PMC104, CBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_C2C},
    {"C2C3", PMC105, CBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_C2C},
    {"C2C4", PMC106, CBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_C2C},
    {"C2C5", PMC107, CBOX0, 0x0, 0x0, 0, 0, EVENT_OPTIONS_C2C},
    {"FIXC0", PMC108, FIXED, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
};

static BoxMap nvidiavera_box_map[NUM_UNITS] = {
    [PMC] = {A57_PERF_CONTROL_CTRL, A57_OVERFLOW_STATUS, A57_OVERFLOW_FLAGS, 0, 0, 0, 32, 0, 0},
    [FIXED] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [MBOX0] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [MBOX1] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [PBOX0] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [PBOX1] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [PBOX2] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [PBOX3] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [PBOX4] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [PBOX5] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [PBOX6] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [PBOX7] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [PBOX8] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [PBOX9] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [PBOX10] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [PBOX11] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [QBOX0] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [QBOX1] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [BBOX0] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [BBOX1] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [BBOX2] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [BBOX3] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [BBOX4] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [BBOX5] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [BBOX6] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [BBOX7] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [BBOX8] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [BBOX9] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [BBOX10] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [BBOX11] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [SBOX0] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [SBOX1] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [WBOX0] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [WBOX1] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [CBOX0] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
    [CBOX1] = {0, 0, 0, 0, 0, 0, 32, 0, 0},
/*    [BBOX0] = {0, 0, 0, 0, 0, 0, 32, 0, 0},*/
/*    [QBOX0] = {0, 0, 0, 0, 0, 0, 32, 0, 0},*/
/*    [SBOX0] = {0, 0, 0, 0, 0, 0, 32, 0, 0},*/
/*    [PBOX0] = {0, 0, 0, 0, 0, 0, 32, 0, 0},*/
/*    [MBOX1] = {0, 0, 0, 0, 0, 0, 32, 0, 0},*/
/*    [BBOX1] = {0, 0, 0, 0, 0, 0, 32, 0, 0},*/
/*    [QBOX1] = {0, 0, 0, 0, 0, 0, 32, 0, 0},*/
/*    [SBOX1] = {0, 0, 0, 0, 0, 0, 32, 0, 0},*/
/*    [PBOX1] = {0, 0, 0, 0, 0, 0, 32, 0, 0},*/
};

static char* nvidiavera_translate_types[NUM_UNITS] = {
    [PMC] = "/sys/bus/event_source/devices/armv8_pmuv3_0",
    [FIXED] = "/sys/bus/event_source/devices/armv8_pmuv3_0",
    [PBOX0] = "/sys/bus/event_source/devices/nvidia_pcie_pmu_0_rc_0",
    [PBOX1] = "/sys/bus/event_source/devices/nvidia_pcie_pmu_1_rc_0",
    [PBOX2] = "/sys/bus/event_source/devices/nvidia_pcie_pmu_0_rc_1",
    [PBOX3] = "/sys/bus/event_source/devices/nvidia_pcie_pmu_1_rc_1",
    [PBOX4] = "/sys/bus/event_source/devices/nvidia_pcie_pmu_0_rc_2",
    [PBOX5] = "/sys/bus/event_source/devices/nvidia_pcie_pmu_1_rc_2",
    [PBOX6] = "/sys/bus/event_source/devices/nvidia_pcie_pmu_0_rc_3",
    [PBOX7] = "/sys/bus/event_source/devices/nvidia_pcie_pmu_1_rc_3",
    [PBOX8] = "/sys/bus/event_source/devices/nvidia_pcie_pmu_0_rc_4",
    [PBOX9] = "/sys/bus/event_source/devices/nvidia_pcie_pmu_1_rc_4",
    [PBOX10] = "/sys/bus/event_source/devices/nvidia_pcie_pmu_0_rc5",
    [PBOX11] = "/sys/bus/event_source/devices/nvidia_pcie_pmu_1_rc5",
    [MBOX0] = "/sys/bus/event_source/devices/nvidia_ucf_pmu_0",
    [MBOX1] = "/sys/bus/event_source/devices/nvidia_ucf_pmu_1",
    [QBOX0] = "/sys/bus/event_source/devices/nvidia_nvdlink_pmu_0",
    [QBOX1] = "/sys/bus/event_source/devices/nvidia_nvdlink_pmu_1",
    [BBOX0] = "/sys/bus/event_source/devices/nvidia_pcie_tgt_pmu_0_rc_0",
    [BBOX1] = "/sys/bus/event_source/devices/nvidia_pcie_tgt_pmu_1_rc_0",
    [BBOX2] = "/sys/bus/event_source/devices/nvidia_pcie_tgt_pmu_0_rc_1",
    [BBOX3] = "/sys/bus/event_source/devices/nvidia_pcie_tgt_pmu_1_rc_1",
    [BBOX4] = "/sys/bus/event_source/devices/nvidia_pcie_tgt_pmu_0_rc_2",
    [BBOX5] = "/sys/bus/event_source/devices/nvidia_pcie_tgt_pmu_1_rc_2",
    [BBOX6] = "/sys/bus/event_source/devices/nvidia_pcie_tgt_pmu_0_rc_3",
    [BBOX7] = "/sys/bus/event_source/devices/nvidia_pcie_tgt_pmu_1_rc_3",
    [BBOX8] = "/sys/bus/event_source/devices/nvidia_pcie_tgt_pmu_0_rc_4",
    [BBOX9] = "/sys/bus/event_source/devices/nvidia_pcie_tgt_pmu_1_rc_4",
    [BBOX10] = "/sys/bus/event_source/devices/nvidia_pcie_tgt_pmu_0_rc5",
    [BBOX11] = "/sys/bus/event_source/devices/nvidia_pcie_tgt_pmu_1_rc5",
    [SBOX0] = "/sys/bus/event_source/devices/nvidia_cnvlink_pmu_0",
    [SBOX1] = "/sys/bus/event_source/devices/nvidia_cnvlink_pmu_1",
    [WBOX0] = "/sys/bus/event_source/devices/nvidia_cmem_latency_pmu_0",
    [WBOX1] = "/sys/bus/event_source/devices/nvidia_cmem_latency_pmu_1",
    [CBOX0] = "/sys/bus/event_source/devices/nvidia_nvlink_c2c_pmu_0",
    [CBOX1] = "/sys/bus/event_source/devices/nvidia_nvlink_c2c_pmu_1",
};

static char* registerTypeNamesNvidiaVera[MAX_UNITS] = {
    [PMC] = "ARM core",
    [FIXED] = "ARM core (fixed counter)",
    [MBOX0 ... MBOX1] = "Unified Coherency Fabric",
    [PBOX0 ... PBOX11] = "Traffic initiated by PCIe root ports or bus-device-function sources",
    [BBOX0 ... BBOX11] = "Traffic targeting PCIe root ports or address ranges",
    [QBOX0 ... QBOX1] = "NV-DLink (CXL memory)",
    [SBOX0 ... SBOX1] = "NV-CLink",
    [WBOX0 ... WBOX1] = "CPU-memory latency",
    [CBOX0 ... CBOX1] = "NVLink-C2C",
};

#endif //PERFMON_NVIDIAVERA_COUNTERS_H
