/*
 * =======================================================================================
 *
 *      Filename:  perfmon_power9_counters.h
 *
 *      Description:  Counter header File of perfmon module for IBM POWER9.
 *
 *      Version:   5.2.1
 *      Released:  03.12.2021
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

#define NUM_COUNTERS_POWER9 48

static RegisterMap power9_counter_map[NUM_COUNTERS_POWER9] = {
    {"PMC0", PMC0, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC1", PMC1, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC2", PMC2, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC3", PMC3, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC4", PMC4, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC5", PMC5, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"MBOX0C0", PMC6, MBOX0},
    {"MBOX0C1", PMC7, MBOX0},
    {"MBOX0C2", PMC8, MBOX0},
    {"MBOX1C0", PMC9, MBOX1},
    {"MBOX1C1", PMC10, MBOX1},
    {"MBOX1C2", PMC11, MBOX1},
    {"MBOX2C0", PMC12, MBOX2},
    {"MBOX2C1", PMC13, MBOX2},
    {"MBOX2C2", PMC14, MBOX2},
    {"MBOX3C0", PMC15, MBOX3},
    {"MBOX3C1", PMC16, MBOX3},
    {"MBOX3C2", PMC17, MBOX3},
    {"MBOX4C0", PMC18, MBOX4},
    {"MBOX4C1", PMC19, MBOX4},
    {"MBOX4C2", PMC20, MBOX4},
    {"MBOX5C0", PMC21, MBOX5},
    {"MBOX5C1", PMC22, MBOX5},
    {"MBOX5C2", PMC23, MBOX5},
    {"MBOX6C0", PMC24, MBOX6},
    {"MBOX6C1", PMC25, MBOX6},
    {"MBOX6C2", PMC26, MBOX6},
    {"MBOX7C0", PMC27, MBOX7},
    {"MBOX7C1", PMC28, MBOX7},
    {"MBOX7C2", PMC29, MBOX7},
    {"QBOX0C0", PMC30, QBOX0},
    {"QBOX0C1", PMC31, QBOX0},
    {"QBOX0C2", PMC32, QBOX0},
    {"QBOX1C0", PMC33, QBOX1},
    {"QBOX1C1", PMC34, QBOX1},
    {"QBOX1C2", PMC35, QBOX1},
    {"QBOX2C0", PMC36, QBOX2},
    {"QBOX2C1", PMC37, QBOX2},
    {"QBOX2C2", PMC38, QBOX2},
    {"SBOX0C0", PMC39, SBOX0},
    {"SBOX0C1", PMC40, SBOX0},
    {"SBOX0C2", PMC41, SBOX0},
    {"BBOX0C0", PMC42, BBOX0},
    {"BBOX0C1", PMC43, BBOX0},
    {"BBOX0C2", PMC44, BBOX0},
    {"BBOX1C0", PMC45, BBOX1},
    {"BBOX1C1", PMC46, BBOX1},
    {"BBOX1C2", PMC47, BBOX1},
};

static BoxMap power9_box_map[NUM_UNITS] = {
    [PMC] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX0] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX1] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX2] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX3] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX4] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX5] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX6] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [MBOX7] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [QBOX0] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [QBOX1] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [QBOX2] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [SBOX0] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [BBOX0] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
    [BBOX1] = {0x0, 0x0, 0x0, 0, 0, 0, 64},
};

static char* power9_translate_types[NUM_UNITS] = {
    [PMC] = "/sys/bus/event_source/devices/cpu",
    [MBOX0] = "/sys/bus/event_source/devices/nest_mba0_imc",
    [MBOX1] = "/sys/bus/event_source/devices/nest_mba1_imc",
    [MBOX2] = "/sys/bus/event_source/devices/nest_mba2_imc",
    [MBOX3] = "/sys/bus/event_source/devices/nest_mba3_imc",
    [MBOX4] = "/sys/bus/event_source/devices/nest_mba4_imc",
    [MBOX5] = "/sys/bus/event_source/devices/nest_mba5_imc",
    [MBOX6] = "/sys/bus/event_source/devices/nest_mba6_imc",
    [MBOX7] = "/sys/bus/event_source/devices/nest_mba7_imc",
    [QBOX0] = "/sys/bus/event_source/devices/nest_xlink0_imc",
    [QBOX1] = "/sys/bus/event_source/devices/nest_xlink1_imc",
    [QBOX2] = "/sys/bus/event_source/devices/nest_xlink2_imc",
    [SBOX0] = "/sys/bus/event_source/devices/nest_powerbus0_imc",
    [BBOX0] = "/sys/bus/event_source/devices/nest_mcs01_imc",
    [BBOX1] = "/sys/bus/event_source/devices/nest_mcs23_imc",
};
