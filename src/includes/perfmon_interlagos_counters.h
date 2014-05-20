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
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
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

static PerfmonCounterMap interlagos_counter_map[NUM_COUNTERS_INTERLAGOS] = {
    /* Core counters */
    {"PMC0",PMC0, PMC, MSR_AMD15_PERFEVTSEL0, MSR_AMD15_PMC0, 0, 0},
    {"PMC1",PMC1, PMC, MSR_AMD15_PERFEVTSEL1, MSR_AMD15_PMC1, 0, 0},
    {"PMC2",PMC2, PMC, MSR_AMD15_PERFEVTSEL2, MSR_AMD15_PMC2, 0, 0},
    {"PMC3",PMC3, PMC, MSR_AMD15_PERFEVTSEL3, MSR_AMD15_PMC3, 0, 0},
    {"PMC4",PMC4, PMC, MSR_AMD15_PERFEVTSEL4, MSR_AMD15_PMC4, 0, 0},
    {"PMC5",PMC5, PMC, MSR_AMD15_PERFEVTSEL5, MSR_AMD15_PMC5, 0, 0},
    /* Northbridge counters */
    {"UPMC0",PMC6, UNCORE, MSR_AMD15_NB_PERFEVTSEL0, MSR_AMD15_NB_PMC0, 0, 0},
    {"UPMC1",PMC7, UNCORE, MSR_AMD15_NB_PERFEVTSEL0, MSR_AMD15_NB_PMC0, 0, 0},
    {"UPMC2",PMC8, UNCORE, MSR_AMD15_NB_PERFEVTSEL0, MSR_AMD15_NB_PMC0, 0, 0},
    {"UPMC3",PMC9, UNCORE, MSR_AMD15_NB_PERFEVTSEL0, MSR_AMD15_NB_PMC0, 0, 0}
};

