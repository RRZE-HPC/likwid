/*
 * =======================================================================================
 *
 *      Filename:  perfmon_a15_counters.h
 *
 *      Description:  Counter Header File of perfmon module for ARM A15.
 *
 *      Version:   5.2.2
 *      Released:  26.07.2022
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg
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


#define NUM_COUNTERS_A15 6

static RegisterMap a15_counter_map[NUM_COUNTERS_A15] = {
    {"PMC0", PMC0, PMC, A15_PERFEVTSEL0, A15_PMC0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC1", PMC1, PMC, A15_PERFEVTSEL1, A15_PMC1, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC2", PMC2, PMC, A15_PERFEVTSEL2, A15_PMC2, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC3", PMC3, PMC, A15_PERFEVTSEL3, A15_PMC3, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC4", PMC4, PMC, A15_PERFEVTSEL4, A15_PMC4, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC5", PMC5, PMC, A15_PERFEVTSEL5, A15_PMC5, 0, 0, EVENT_OPTION_NONE_MASK},
};

static BoxMap a15_box_map[NUM_UNITS] = {
    [PMC] = {A15_PERF_CONTROL_CTRL, A15_OVERFLOW_STATUS, A15_OVERFLOW_FLAGS, 0, 0, 0, 32},
};

static char* a15_translate_types[NUM_UNITS] = {
    [PMC] = "/sys/bus/event_source/devices/cpu",
};
