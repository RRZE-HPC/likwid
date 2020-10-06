/*
 * =======================================================================================
 *
 *      Filename:  perfmon_a57_counters.h
 *
 *      Description:  Counter Header File of perfmon module for ARM A57.
 *
 *      Version:   5.0.2
 *      Released:  06.10.2020
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


#define NUM_COUNTERS_A57 6

static RegisterMap a57_counter_map[NUM_COUNTERS_A57] = {
    {"PMC0", PMC0, PMC, A57_PERFEVTSEL0, A57_PMC0, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC1", PMC1, PMC, A57_PERFEVTSEL1, A57_PMC1, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC2", PMC2, PMC, A57_PERFEVTSEL2, A57_PMC2, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC3", PMC3, PMC, A57_PERFEVTSEL3, A57_PMC3, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC4", PMC4, PMC, A57_PERFEVTSEL4, A57_PMC4, 0, 0, EVENT_OPTION_NONE_MASK},
    {"PMC5", PMC5, PMC, A57_PERFEVTSEL5, A57_PMC5, 0, 0, EVENT_OPTION_NONE_MASK},
};

static BoxMap a57_box_map[NUM_UNITS] = {
    [PMC] = {A57_PERF_CONTROL_CTRL, A57_OVERFLOW_STATUS, A57_OVERFLOW_FLAGS, 0, 0, 0, 32},
};

static char* a57_translate_types[NUM_UNITS] = {
    [PMC] = "/sys/bus/event_source/devices/armv8_pmuv3",
};

static char* a53_translate_types[NUM_UNITS] = {
    [PMC] = "/sys/bus/event_source/devices/cpu",
};
