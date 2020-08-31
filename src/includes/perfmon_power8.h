/*
 * =======================================================================================
 *
 *      Filename:  perfmon_power8.h
 *
 *      Description:  Header File of perfmon module for IBM POWER8.
 *
 *      Version:   5.0.2
 *      Released:  31.08.2020
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


#include <perfmon_power8_counters.h>
#include <perfmon_power8_events.h>

static int perfmon_numCountersPower8 = NUM_COUNTERS_POWER8;
static int perfmon_numCoreCountersPower8 = NUM_COUNTERS_POWER8;
static int perfmon_numArchEventsPower8 = NUM_ARCH_EVENTS_POWER8;
