/*
 * =======================================================================================
 *
 *      Filename:  perfmon_power8_counters.h
 *
 *      Description:  Counter header File of perfmon module for IBM POWER8.
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
#ifndef PERFMON_POWER8_COUNTERS_H
#define PERFMON_POWER8_COUNTERS_H

#define NUM_COUNTERS_POWER8 7

static RegisterMap power8_counter_map[NUM_COUNTERS_POWER8] = {
    { "PMC0", PMC0, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK },
    { "PMC1", PMC1, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK },
    { "PMC2", PMC2, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK },
    { "PMC3", PMC3, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK },
    { "PMC4", PMC4, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK },
    { "PMC5", PMC5, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK },
    { "PURR", PMC6, PMC, 0x0, 0x0, 0, 0, EVENT_OPTION_NONE_MASK },
};

static BoxMap power8_box_map[NUM_UNITS] = {
    [PMC] = { 0x0, 0x0, 0x0, 0, 0, 0, 64 },
};

static char *power8_translate_types[NUM_UNITS] = {
    [PMC] = "/sys/bus/event_source/devices/cpu",
};

#endif //PERFMON_POWER8_COUNTERS_H
