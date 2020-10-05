/*
 * =======================================================================================
 *
 *      Filename:  perfmon_skylake.h
 *
 *      Description:  Header File of perfmon module for Intel Skylake.
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

#include <perfmon_icelake_counters.h>
#include <perfmon_icelake_events.h>

static int perfmon_numCountersIcelake = NUM_COUNTERS_ICELAKE;
static int perfmon_numCoreCountersIcelake = NUM_COUNTERS_CORE_ICELAKE;
static int perfmon_numArchEventsIcelake = NUM_ARCH_EVENTS_ICELAKE;
