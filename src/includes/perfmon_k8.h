/*
 * =======================================================================================
 *
 *      Filename:  perfmon_k8.h
 *
 *      Description:  Header File of perfmon module for AMD K8 support.
 *                    Configures and reads out performance counters
 *                    on x86 based architectures. Supports multi threading.
 *                    The setup routines are similar to AMD K10
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *               Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig and Thomas Roehl
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

#include <perfmon_k8_events.h>
#include <error.h>


static int perfmon_numArchEventsK8 = NUM_ARCH_EVENTS_K8;


