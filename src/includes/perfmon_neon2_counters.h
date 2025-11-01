/*
 * =======================================================================================
 *
 *      Filename:  perfmon_neon2_counters.h
 *
 *      Description:  Counter Header File of perfmon module for ARM Neoverse N2.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 RRZE, University Erlangen-Nuremberg
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


#define NUM_COUNTERS_NEON2 6

static RegisterMap neon2_counter_map[NUM_COUNTERS_NEON2] = {
    {"PMC0", PMC0, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC1", PMC1, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC2", PMC2, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC3", PMC3, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC4", PMC4, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC5", PMC5, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK},
};

static BoxMap neon2_box_map[NUM_UNITS] = {
    [PMC] = {0x0, 0x0, 0x0, 0, 0, 0, 32},
};

static char* neon2_translate_types[NUM_UNITS] = {
    [PMC] = "/sys/bus/event_source/devices/armv8_neoverse_n2",
};

