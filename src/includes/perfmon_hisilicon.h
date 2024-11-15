/*
 * =======================================================================================
 *
 *      Filename:  perfmon_hisilicon.h
 *
 *      Description:  Header File of perfmon module for HiSilicon chips.
 *
 *      Version:   5.4.0
 *      Released:  15.11.2024
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
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
#ifndef NUM_ARCH_EVENTS_A57
#include <perfmon_a57_events.h>
#endif
#include <perfmon_hisilicon_counters.h>


static int perfmon_numCountersHiSiliconTsv110 = NUM_COUNTERS_HISILICON_TSV110;
static int perfmon_numArchEventsHiSiliconTsv110 = NUM_ARCH_EVENTS_A57;
