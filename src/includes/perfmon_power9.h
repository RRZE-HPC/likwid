/*
 * =======================================================================================
 *
 *      Filename:  perfmon_power9.h
 *
 *      Description:  Header File of perfmon module for IBM POWER9.
 *
 *      Version:   5.2
 *      Released:  17.6.2021
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 NHR@FAU, University Erlangen-Nuremberg
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


#include <perfmon_power9_counters.h>
#include <perfmon_power9_events.h>

static int perfmon_numCountersPower9 = NUM_COUNTERS_POWER9;
static int perfmon_numCoreCountersPower9 = NUM_COUNTERS_POWER9;
static int perfmon_numArchEventsPower9 = NUM_ARCH_EVENTS_POWER9;
