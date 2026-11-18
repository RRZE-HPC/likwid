/*
 * =======================================================================================
 *
 *      Filename:  perfmon_nvidiavera.h
 *
 *      Description:  Header File of perfmon module for Nvidia Vera CPU
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
#ifndef PERFMON_NVIDIAVERA_H
#define PERFMON_NVIDIAVERA_H

#include <perfmon_nvidiavera_events.h>
#include <perfmon_nvidiavera_counters.h>

#include <error.h>
#include <affinity.h>
#include <limits.h>
#include <topology.h>
#include <access.h>

static int perfmon_numCountersNvidiaVera = NUM_COUNTERS_NVIDIAVERA;
static int perfmon_numArchEventsNvidiaVera = NUM_ARCH_EVENTS_NVIDIAVERA;

#endif //PERFMON_NVIDIAVERA_H
