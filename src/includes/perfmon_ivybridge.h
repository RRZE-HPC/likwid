/*
 * =======================================================================================
 *
 *      Filename:  perfmon_ivybridge.h
 *
 *      Description:  Header File of perfmon module for Ivy Bridge.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2012 Jan Treibig 
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

#include <perfmon_ivybridge_events.h>
#include <perfmon_ivybridge_groups.h>

#define NUM_COUNTERS_IVYBRIDGE 11

static int perfmon_numCountersIvybridge = NUM_COUNTERS_IVYBRIDGE;
static int perfmon_numGroupsIvybridge = NUM_GROUPS_IVYBRIDGE;
static int perfmon_numArchEventsIvybridge = NUM_ARCH_EVENTS_IVYBRIDGE;

static PerfmonCounterMap ivybridge_counter_map[NUM_COUNTERS_IVYBRIDGE] = {
    {"FIXC0",PMC0},
    {"FIXC1",PMC1},
    {"FIXC2",PMC2},
    {"PMC0",PMC3},
    {"PMC1",PMC4},
    {"PMC2",PMC5},
    {"PMC3",PMC6},
    {"PWR0",PMC7},
    {"PWR1",PMC8},
    {"PWR2",PMC9},
    {"PWR3",PMC10},
};



