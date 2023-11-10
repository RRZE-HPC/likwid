/*
 * =======================================================================================
 *
 *      Filename:  perfmon_a64fx_counters.h
 *
 *      Description:  Counter Header File of perfmon module for Fujitsu A64FX
 *
 *      Version:   5.3
 *      Released:  10.11.2023
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2023 RRZE, University Erlangen-Nuremberg
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

#define NUM_COUNTERS_APPLEM1 10

static RegisterMap applem1_counter_map[NUM_COUNTERS_APPLEM1] = {
    {"PMC0", PMC0, PMC, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC1", PMC1, PMC, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC2", PMC2, PMC, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC3", PMC3, PMC, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC4", PMC4, PMC, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC5", PMC5, PMC, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC6", PMC6, PMC, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC7", PMC7, PMC, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC8", PMC8, PMC, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC9", PMC9, PMC, 0, 0, 0, 0, EVENT_OPTION_NONE_MASK},
};

static BoxMap applem1_box_map[NUM_UNITS] = {
    [PMC] = {0, 0, 0, 0, 0, 0, 32},
};

static char* applem1_translate_types[NUM_UNITS] = {
    [IPMC] = "/sys/bus/event_source/devices/apple_icestorm_pmu",
    [FPMC] = "/sys/bus/event_source/devices/apple_firestorm_pmu",
};
