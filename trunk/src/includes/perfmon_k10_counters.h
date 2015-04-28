/*
 * =======================================================================================
 *
 *      Filename:  perfmon_k10_counters.h
 *
 *      Description:  AMD K10 performance counter definition. Also used for AMD K8.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *               Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig and Thomas Roehl
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

#define NUM_COUNTERS_K10 4
#define NUM_COUNTERS_CORE_K10 4

#define K10_VALID_OPTIONS_PMC EVENT_OPTION_EDGE_MASK|EVENT_OPTION_COUNT_KERNEL_MASK|EVENT_OPTION_INVERT_MASK|EVENT_OPTION_THRESHOLD

static RegisterMap k10_counter_map[NUM_COUNTERS_K10] = {
    {"PMC0",PMC0, PMC, MSR_AMD_PERFEVTSEL0, MSR_AMD_PMC0, 0, 0, K10_VALID_OPTIONS_PMC},
    {"PMC1",PMC1, PMC, MSR_AMD_PERFEVTSEL1, MSR_AMD_PMC1, 0, 0, K10_VALID_OPTIONS_PMC},
    {"PMC2",PMC2, PMC, MSR_AMD_PERFEVTSEL2, MSR_AMD_PMC2, 0, 0, K10_VALID_OPTIONS_PMC},
    {"PMC3",PMC3, PMC, MSR_AMD_PERFEVTSEL3, MSR_AMD_PMC3, 0, 0, K10_VALID_OPTIONS_PMC}
};

static BoxMap k10_box_map[NUM_UNITS] = {
    [PMC] = {0, 0, 0, 0, 0, 0, 48}
};
