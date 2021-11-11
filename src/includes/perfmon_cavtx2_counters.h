/*
 * =======================================================================================
 *
 *      Filename:  perfmon_cavtx2_counters.h
 *
 *      Description:  Counter Header File of perfmon module for Marvell/Cavium
 *                    Thunder X2.
 *
 *      Version:   5.2.1
 *      Released:  11.11.2021
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 NHR@FAU, University Erlangen-Nuremberg
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

#define NUM_COUNTERS_CAV_TX2 38

static RegisterMap cav_tx2_counter_map[NUM_COUNTERS_CAV_TX2] = {
    {"PMC0", PMC0, PMC, A57_PERFEVTSEL0, A57_PMC0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC1", PMC1, PMC, A57_PERFEVTSEL1, A57_PMC1, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC2", PMC2, PMC, A57_PERFEVTSEL2, A57_PMC2, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC3", PMC3, PMC, A57_PERFEVTSEL3, A57_PMC3, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC4", PMC4, PMC, A57_PERFEVTSEL4, A57_PMC4, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC5", PMC5, PMC, A57_PERFEVTSEL5, A57_PMC5, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CBOX0C0", PMC6, CBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CBOX0C1", PMC7, CBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CBOX0C2", PMC8, CBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CBOX0C3", PMC9, CBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CBOX1C0", PMC10, CBOX1, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CBOX1C1", PMC11, CBOX1, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CBOX1C2", PMC12, CBOX1, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"CBOX1C3", PMC13, CBOX1, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"MBOX0C0", PMC14, MBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"MBOX0C1", PMC15, MBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"MBOX0C2", PMC16, MBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"MBOX0C3", PMC17, MBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"MBOX1C0", PMC18, MBOX1, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"MBOX1C1", PMC19, MBOX1, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"MBOX1C2", PMC20, MBOX1, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"MBOX1C3", PMC21, MBOX1, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SBOX0C0", PMC22, SBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SBOX0C1", PMC23, SBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SBOX0C2", PMC24, SBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SBOX0C3", PMC25, SBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SBOX0C4", PMC26, SBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SBOX0C5", PMC27, SBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SBOX0C6", PMC28, SBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SBOX0C7", PMC29, SBOX0, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SBOX1C0", PMC30, SBOX1, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SBOX1C1", PMC31, SBOX1, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SBOX1C2", PMC32, SBOX1, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SBOX1C3", PMC33, SBOX1, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SBOX1C4", PMC34, SBOX1, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SBOX1C5", PMC35, SBOX1, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SBOX1C6", PMC36, SBOX1, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"SBOX1C7", PMC37, SBOX1},
};

static BoxMap cav_tx2_box_map[NUM_UNITS] = {
    [PMC] = {A57_PERF_CONTROL_CTRL, A57_OVERFLOW_STATUS, A57_OVERFLOW_FLAGS, 0, 0, 0, 32},
    [CBOX0] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [CBOX1] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX0] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX1] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [SBOX0] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [SBOX1] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
};

static char* cav_tx2_translate_types[NUM_UNITS] = {
    [PMC] = "/sys/bus/event_source/devices/armv8_pmuv3_0",
    [CBOX0] = "/sys/bus/event_source/devices/uncore_l3c_0",
    [CBOX1] = "/sys/bus/event_source/devices/uncore_l3c_1",
    [MBOX0] = "/sys/bus/event_source/devices/uncore_dmc_0",
    [MBOX1] = "/sys/bus/event_source/devices/uncore_dmc_1",
    [SBOX0] = "/sys/bus/event_source/devices/uncore_ccpi2_0",
    [SBOX1] = "/sys/bus/event_source/devices/uncore_ccpi2_1",
};
