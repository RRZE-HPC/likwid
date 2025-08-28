/*
 * =======================================================================================
 *
 *      Filename:  perfmon_kabini_counters.h
 *
 *      Description:  Counter Header File of perfmon module for AMD Family 16
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
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
#ifndef PERFMON_KABINI_COUNTERS_H
#define PERFMON_KABINI_COUNTERS_H

#define NUM_COUNTERS_KABINI 12
#define NUM_COUNTERS_CORE_KABINI 8

#define KAB_VALID_OPTIONS_PMC EVENT_OPTION_EDGE_MASK | EVENT_OPTION_COUNT_KERNEL_MASK | EVENT_OPTION_INVERT_MASK | EVENT_OPTION_THRESHOLD
#define KAB_VALID_OPTIONS_CBOX EVENT_OPTION_INVERT_MASK | EVENT_OPTION_THRESHOLD | EVENT_OPTION_TID_MASK | EVENT_OPTION_NID_MASK

static RegisterMap kabini_counter_map[NUM_COUNTERS_KABINI] = {
    /* Core counters */
    { "PMC0",  PMC0,  PMC,    MSR_AMD16_PERFEVTSEL0,    MSR_AMD16_PMC0,    0, 0, EVENT_OPTION_NONE_MASK },
    { "PMC1",  PMC1,  PMC,    MSR_AMD16_PERFEVTSEL1,    MSR_AMD16_PMC1,    0, 0, EVENT_OPTION_NONE_MASK },
    { "PMC2",  PMC2,  PMC,    MSR_AMD16_PERFEVTSEL2,    MSR_AMD16_PMC2,    0, 0, EVENT_OPTION_NONE_MASK },
    { "PMC3",  PMC3,  PMC,    MSR_AMD16_PERFEVTSEL3,    MSR_AMD16_PMC3,    0, 0, EVENT_OPTION_NONE_MASK },
    /* L2 cache counters */
    { "CPMC0", PMC4,  CBOX0,  MSR_AMD16_L2_PERFEVTSEL0, MSR_AMD16_L2_PMC0, 0, 0, EVENT_OPTION_NONE_MASK },
    { "CPMC1", PMC5,  CBOX0,  MSR_AMD16_L2_PERFEVTSEL1, MSR_AMD16_L2_PMC1, 0, 0, EVENT_OPTION_NONE_MASK },
    { "CPMC2", PMC6,  CBOX0,  MSR_AMD16_L2_PERFEVTSEL2, MSR_AMD16_L2_PMC2, 0, 0, EVENT_OPTION_NONE_MASK },
    { "CPMC3", PMC7,  CBOX0,  MSR_AMD16_L2_PERFEVTSEL3, MSR_AMD16_L2_PMC3, 0, 0, EVENT_OPTION_NONE_MASK },
    /* Northbridge counters */
    { "UPMC0", PMC8,  UNCORE, MSR_AMD16_NB_PERFEVTSEL0, MSR_AMD16_NB_PMC0, 0, 0, EVENT_OPTION_NONE_MASK },
    { "UPMC1", PMC9,  UNCORE, MSR_AMD16_NB_PERFEVTSEL1, MSR_AMD16_NB_PMC1, 0, 0, EVENT_OPTION_NONE_MASK },
    { "UPMC2", PMC10, UNCORE, MSR_AMD16_NB_PERFEVTSEL2, MSR_AMD16_NB_PMC2, 0, 0, EVENT_OPTION_NONE_MASK },
    { "UPMC3", PMC11, UNCORE, MSR_AMD16_NB_PERFEVTSEL3, MSR_AMD16_NB_PMC3, 0, 0, EVENT_OPTION_NONE_MASK }
};

static BoxMap kabini_box_map[NUM_UNITS] = {
    [PMC]    = { 0, 0, 0, 0, 0, 0, 48, 0, 0 },
    [UNCORE] = { 0, 0, 0, 0, 0, 0, 48, 0, 0 },
    [CBOX0]  = { 0, 0, 0, 0, 0, 0, 48, 0, 0 },
};

#endif //PERFMON_KABINI_COUNTERS_H
