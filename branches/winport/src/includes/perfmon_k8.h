/*
 * ===========================================================================
 *
 *      Filename:  perfmon_k8.h
 *
 *      Description:  Header File of perfmon module for K8 support.
 *                    Configures and reads out performance counters
 *                    on x86 based architectures. Supports multi threading.
 *
 *      Version:  <VERSION>
 *      Created:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Company:  RRZE Erlangen
 *      Project:  likwid
 *      Copyright:  Copyright (c) 2010, Jan Treibig
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License, v2, as
 *      published by the Free Software Foundation
 *     
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *     
 *      You should have received a copy of the GNU General Public License
 *      along with this program; if not, write to the Free Software
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * ===========================================================================
 */

#include <stdlib.h>
#include <stdio.h>

#include <bstrlib.h>
#include <types.h>
#include <registers.h>
#include <perfmon_k8_events.h>
#include <perfmon_k8_groups.h>


static int perfmon_numGroupsK8 = NUM_GROUPS_K8;
static int perfmon_numArchEventsK8 = NUM_ARCH_EVENTS_K8;


