/*
 * =======================================================================================
 *
 *      Filename:  perfmon_kabini_counters.h
 *
 *      Description:  Counter Header File of perfmon module for AMD Family16
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

#define NUM_COUNTERS_KABINI 8 
#define NUM_COUNTERS_CORE_KABINI 4

static PerfmonCounterMap kabini_counter_map[NUM_COUNTERS_KABINI] = {
    /* Core counters */
    {"PMC0",PMC0, PMC, MSR_AMD16_PERFEVTSEL0, MSR_AMD16_PMC0, 0, 0},
    {"PMC1",PMC1, PMC, MSR_AMD16_PERFEVTSEL1, MSR_AMD16_PMC1, 0, 0},
    {"PMC2",PMC2, PMC, MSR_AMD16_PERFEVTSEL2, MSR_AMD16_PMC2, 0, 0},
    {"PMC3",PMC3, PMC, MSR_AMD16_PERFEVTSEL3, MSR_AMD16_PMC3, 0, 0},
    /* Northbridge counters */
    {"UPMC0",PMC4, UNCORE, MSR_AMD16_NB_PERFEVTSEL0, MSR_AMD16_NB_PMC0, 0, 0},
    {"UPMC1",PMC5, UNCORE, MSR_AMD16_NB_PERFEVTSEL1, MSR_AMD16_NB_PMC1, 0, 0},
    {"UPMC2",PMC6, UNCORE, MSR_AMD16_NB_PERFEVTSEL2, MSR_AMD16_NB_PMC2, 0, 0},
    {"UPMC3",PMC7, UNCORE, MSR_AMD16_NB_PERFEVTSEL3, MSR_AMD16_NB_PMC3, 0, 0}
};

