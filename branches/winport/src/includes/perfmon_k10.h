/*
 * ===========================================================================
 *
 *      Filename:  perfmon_k10.h
 *
 *      Description:  AMD K10 specific subroutines
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
#include <perfmon_k10_events.h>
#include <perfmon_k10_groups.h>

#define NUM_COUNTERS_K10 4
#define NUM_SETS_K10 8

static int perfmon_numCountersK10 = NUM_COUNTERS_K10;
static int perfmon_numGroupsK10 = NUM_GROUPS_K10;
static int perfmon_numArchEventsK10 = NUM_ARCH_EVENTS_K10;

static PerfmonCounterMap k10_counter_map[NUM_COUNTERS_K10] = {
    {"PMC0",PMC0},
    {"PMC1",PMC1},
    {"PMC2",PMC2},
    {"PMC3",PMC3}
};

void perfmon_init_k10(PerfmonThread *thread)
{
    uint64_t flags = 0x0ULL;
    int cpu_id = thread->processorId;

    thread->counters[PMC0].configRegister = MSR_AMD_PERFEVTSEL0;
    thread->counters[PMC0].counterRegister = MSR_AMD_PMC0;
    thread->counters[PMC0].type = PMC;
    thread->counters[PMC1].configRegister = MSR_AMD_PERFEVTSEL1;
    thread->counters[PMC1].counterRegister = MSR_AMD_PMC1;
    thread->counters[PMC0].type = PMC;
    thread->counters[PMC2].configRegister = MSR_AMD_PERFEVTSEL2;
    thread->counters[PMC2].counterRegister = MSR_AMD_PMC2;
    thread->counters[PMC0].type = PMC;
    thread->counters[PMC3].configRegister = MSR_AMD_PERFEVTSEL3;
    thread->counters[PMC3].counterRegister = MSR_AMD_PMC3;
    thread->counters[PMC0].type = PMC;

    msr_write(cpu_id, MSR_AMD_PERFEVTSEL0, 0x0ULL);
    msr_write(cpu_id, MSR_AMD_PERFEVTSEL1, 0x0ULL);
    msr_write(cpu_id, MSR_AMD_PERFEVTSEL2, 0x0ULL);
    msr_write(cpu_id, MSR_AMD_PERFEVTSEL3, 0x0ULL);

    flags |= (1<<16);  /* user mode flag */

    msr_write(cpu_id, MSR_AMD_PERFEVTSEL0, flags);
    msr_write(cpu_id, MSR_AMD_PERFEVTSEL1, flags);
    msr_write(cpu_id, MSR_AMD_PERFEVTSEL2, flags);
    msr_write(cpu_id, MSR_AMD_PERFEVTSEL3, flags);
}


void
perfmon_setupCounterThread_k10(int thread_id,
        uint32_t event, uint32_t umask,
        PerfmonCounterIndex index)
{
    uint64_t flags;
    uint64_t reg = perfmon_threadData[thread_id].counters[index].configRegister;
    int cpu_id = perfmon_threadData[thread_id].processorId;
    perfmon_threadData[thread_id].counters[index].init = TRUE;

    flags = msr_read(cpu_id,reg);
    flags &= ~(0xFFFFU); 

	/* AMD uses a 12 bit Event mask: [35:32][7:0] */
	flags |= ((uint64_t)(event>>8)<<32) + (umask<<8) + (event & ~(0xF00U));

    if (perfmon_verbose)
    {
        printf("[%d] perfmon_setup_counter: Write Register 0x%llX , Flags: 0x%llX \n",
               cpu_id,
               LLU_CAST reg,
               LLU_CAST flags);
    }
    msr_write(cpu_id, reg , flags);
}


void
perfmon_startCountersThread_k10(int thread_id)
{
    int i;
    uint64_t flags;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    for (i=0;i<NUM_COUNTERS_K10;i++) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister , 0x0ULL);
            flags = msr_read(cpu_id,perfmon_threadData[thread_id].counters[i].configRegister);
            flags |= (1<<22);  /* enable flag */
            if (perfmon_verbose) 
            {
                printf("perfmon_start_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                        LLU_CAST perfmon_threadData[thread_id].counters[i].configRegister,
                        LLU_CAST flags);
            }

            msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].configRegister , flags);
        }
    }
}

void 
perfmon_stopCountersThread_k10(int thread_id)
{
    uint64_t flags;
    int i;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    for (i=0;i<NUM_COUNTERS_K10;i++) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            flags = msr_read(cpu_id,perfmon_threadData[thread_id].counters[i].configRegister);
            flags &= ~(1<<22);  /* clear enable flag */
            msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].configRegister , flags);
            if (perfmon_verbose)
            {
                printf("perfmon_stop_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                        LLU_CAST perfmon_threadData[thread_id].counters[i].configRegister,
                        LLU_CAST flags);
            }
            perfmon_threadData[thread_id].counters[i].counterData =
                msr_read(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister);
        }
    }
}


void 
perfmon_readCountersThread_k10(int thread_id)
{
    int i;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    for (i=0;i<NUM_COUNTERS_K10;i++) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            perfmon_threadData[thread_id].counters[i].counterData =
                msr_read(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister);
        }
    }
}


