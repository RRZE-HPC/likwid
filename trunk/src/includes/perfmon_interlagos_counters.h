/*
 * =======================================================================================
 *
 *      Filename:  perfmon_interlagos_counters.h
 *
 *      Description:  Counter Header File of perfmon module for AMD Interlagos
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *               Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2013 Jan Treibig and Thomas Roehl
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

#define NUM_COUNTERS_INTERLAGOS 10
#define NUM_COUNTERS_CORE_INTERLAGOS 6

#define ILG_VALID_OPTIONS_PMC EVENT_OPTION_EDGE_MASK|EVENT_OPTION_COUNT_KERNEL_MASK|EVENT_OPTION_INVERT_MASK|EVENT_OPTION_THRESHOLD

static RegisterMap interlagos_counter_map[NUM_COUNTERS_INTERLAGOS] = {
    /* Core counters */
    {"PMC0",PMC0, PMC, MSR_AMD15_PERFEVTSEL0, MSR_AMD15_PMC0, 0, 0, ILG_VALID_OPTIONS_PMC},
    {"PMC1",PMC1, PMC, MSR_AMD15_PERFEVTSEL1, MSR_AMD15_PMC1, 0, 0, ILG_VALID_OPTIONS_PMC},
    {"PMC2",PMC2, PMC, MSR_AMD15_PERFEVTSEL2, MSR_AMD15_PMC2, 0, 0, ILG_VALID_OPTIONS_PMC},
    {"PMC3",PMC3, PMC, MSR_AMD15_PERFEVTSEL3, MSR_AMD15_PMC3, 0, 0, ILG_VALID_OPTIONS_PMC},
    {"PMC4",PMC4, PMC, MSR_AMD15_PERFEVTSEL4, MSR_AMD15_PMC4, 0, 0, ILG_VALID_OPTIONS_PMC},
    {"PMC5",PMC5, PMC, MSR_AMD15_PERFEVTSEL5, MSR_AMD15_PMC5, 0, 0, ILG_VALID_OPTIONS_PMC},
    /* Northbridge counters */
    {"UPMC0",PMC6, UNCORE, MSR_AMD15_NB_PERFEVTSEL0, MSR_AMD15_NB_PMC0, 0, 0},
    {"UPMC1",PMC7, UNCORE, MSR_AMD15_NB_PERFEVTSEL1, MSR_AMD15_NB_PMC1, 0, 0},
    {"UPMC2",PMC8, UNCORE, MSR_AMD15_NB_PERFEVTSEL2, MSR_AMD15_NB_PMC2, 0, 0},
    {"UPMC3",PMC9, UNCORE, MSR_AMD15_NB_PERFEVTSEL3, MSR_AMD15_NB_PMC3, 0, 0}
};

static BoxMap interlagos_box_map[NUM_UNITS] = {
    [PMC] = {0, 0, 0, 0, 0, 0, 48},
    [UNCORE] = {0, 0, 0, 0, 0, 0, 48},
};
