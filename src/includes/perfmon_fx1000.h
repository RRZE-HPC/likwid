/*
 * =======================================================================================
 *
 *      Filename:  perfmon_fx1000.h
 *
 *      Description:  Header File of perfmon module for Fujitsu A64FX FX1000
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

#include <perfmon_fx1000_events.h>
#include <perfmon_fx1000_counters.h>

static int perfmon_numCountersFX1000 = NUM_COUNTERS_FX1000;
static int perfmon_numArchEventsFX1000 = NUM_ARCH_EVENTS_FX1000;
