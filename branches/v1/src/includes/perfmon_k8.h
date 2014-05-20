/*
 * ===========================================================================
 *
 *      Filename:  perfmon_k8.h
 *
 *      Description:  Header File of perfmon module.
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

#define NUM_GROUPS_K8 5

static int perfmon_numGroupsK8 = NUM_GROUPS_K8;
static int perfmon_numArchEventsK8 = NUM_ARCH_EVENTS_K8;

static PerfmonGroupMap k8_group_map[NUM_GROUPS_K8] = {
    {"L2",L2,"L2 cache bandwidth in MBytes/s","DATA_CACHE_REFILLS_L2_ALL:PMC0,DATA_CACHE_EVICTED_ALL:PMC1,CPU_CLOCKS_UNHALTED:PMC2"},
    {"CACHE",CACHE,"Data cache miss rate/ratio","INSTRUCTIONS_RETIRED:PMC0,DATA_CACHE_ACCESSES:PMC1,DATA_CACHE_REFILLS_L2_ALL:PMC2,DATA_CACHE_REFILLS_NORTHBRIDGE_ALL:PMC3"},
    {"ICACHE",ICACHE,"Instruction cache miss rate/ratio","INSTRUCTIONS_RETIRED:PMC0,ICACHE_FETCHES:PMC1,ICACHE_REFILLS_L2:PMC2,ICACHE_REFILLS_MEM:PMC3"},
    {"BRANCH",BRANCH,"Branch prediction miss rate/ratio","INSTRUCTIONS_RETIRED:PMC0,BRANCH_RETIRED:PMC1,BRANCH_MISPREDICT_RETIRED:PMC2,BRANCH_TAKEN_RETIRED:PMC3"},
    {"CPI",CPI,"cycles per instruction","INSTRUCTIONS_RETIRED:PMC0,CPU_CLOCKS_UNHALTED:PMC1,UOPS_RETIRED:PMC2"}
};


