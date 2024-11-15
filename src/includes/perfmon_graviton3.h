/*
 * =======================================================================================
 *
 *      Filename:  perfmon_graviton3.h
 *
 *      Description:  Header File of perfmon module for ARM AWS Graviton3
 *
 *      Version:   5.4.0
 *      Released:  15.11.2024
 *
 *      Author:   Carlos Falquez, c.falquez@fz-juelich.de
 *      Project:  likwid
 *
 *      Copyright (C) 2024 RRZE, University Erlangen-Nuremberg
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

#include <perfmon_graviton3_events.h>
#include <perfmon_graviton3_counters.h>

static int perfmon_numCountersGraviton3   = NUM_COUNTERS_GRAVITON3;
static int perfmon_numArchEventsGraviton3 = NUM_ARCH_EVENTS_GRAVITON3;
