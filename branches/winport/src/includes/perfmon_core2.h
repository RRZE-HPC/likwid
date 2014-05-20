/*
 * ===========================================================================
 *
 *      Filename:  perfmon_core2.h
 *
 *      Description:  Core 2 specific subroutines
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
#include <perfmon_core2_events.h>
#include <perfmon_core2_groups.h>

#define NUM_COUNTERS_CORE2 4
#define NUM_SETS_CORE2 8

static int perfmon_numCountersCore2 = NUM_COUNTERS_CORE2;
static int perfmon_numGroupsCore2 = NUM_GROUPS_CORE2;
static int perfmon_numArchEventsCore2 = NUM_ARCH_EVENTS_CORE2;

static PerfmonCounterMap core2_counter_map[NUM_COUNTERS_CORE2] = {
    {"FIXC0",PMC0},
    {"FIXC1",PMC1},
    {"PMC0",PMC2},
    {"PMC1",PMC3}
};


void 
perfmon_init_core2(PerfmonThread *thread)
{
    uint64_t flags = 0x0ULL;
    int cpu_id = thread->processorId;

    /* Fixed Counters: instructions retired, cycles unhalted core */
    thread->counters[PMC0].configRegister = MSR_PERF_FIXED_CTR_CTRL;
    thread->counters[PMC0].counterRegister = MSR_PERF_FIXED_CTR0;
    thread->counters[PMC0].type = FIXED;
    thread->counters[PMC1].configRegister = MSR_PERF_FIXED_CTR_CTRL;
    thread->counters[PMC1].counterRegister = MSR_PERF_FIXED_CTR1;
    thread->counters[PMC1].type = FIXED;

    /* PMC Counters: 2 40bit wide */
    thread->counters[PMC2].configRegister = MSR_PERFEVTSEL0;
    thread->counters[PMC2].counterRegister = MSR_PMC0;
    thread->counters[PMC2].type = PMC;
    thread->counters[PMC3].configRegister = MSR_PERFEVTSEL1;
    thread->counters[PMC3].counterRegister = MSR_PMC1;
    thread->counters[PMC3].type = PMC;

    /* Initialize registers */
    msr_write(cpu_id, MSR_PERFEVTSEL0, 0x0ULL);
    msr_write(cpu_id, MSR_PERFEVTSEL1, 0x0ULL);

    msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x0ULL);
    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);
    msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x0ULL);
    msr_write(cpu_id, MSR_PEBS_ENABLE, 0x0ULL);

    /* always initialize fixed counters
     * FIXED 0: Instructions retired
     * FIXED 1: Clocks unhalted */
    msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x22ULL);

    /* Preinit of PMC counters */
    flags |= (1<<16);  /* user mode flag */
    flags |= (1<<19);  /* pin control flag */
    flags |= (1<<22);  /* enable flag */

    msr_write(cpu_id, MSR_PERFEVTSEL0, flags);
    msr_write(cpu_id, MSR_PERFEVTSEL1, flags);
}


void
perfmon_setupCounterThread_core2(int thread_id,
        uint32_t event, uint32_t umask,
        PerfmonCounterIndex index)
{
    uint64_t flags;
    uint64_t reg = perfmon_threadData[thread_id].counters[index].configRegister;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if (perfmon_threadData[thread_id].counters[index].type == PMC)
    {

        perfmon_threadData[thread_id].counters[index].init = TRUE;
        flags = msr_read(cpu_id,reg);
        flags &= ~(0xFFFFU); 

        /* Intel with standard 8 bit event mask: [7:0] */
        flags |= (umask<<8) + event;

        msr_write(cpu_id, reg , flags);

        if (perfmon_verbose)
        {
            printf("[%d] perfmon_setup_counter: Write Register 0x%llX , Flags: 0x%llX \n",
                    cpu_id,
                    LLU_CAST reg,
                    LLU_CAST flags);
        }
    }
    else if (perfmon_threadData[thread_id].counters[index].type == FIXED)
    {
        perfmon_threadData[thread_id].counters[index].init = TRUE;
    }
}

void
perfmon_startCountersThread_core2(int thread_id)
{
    int i;
    uint64_t flags = 0ULL;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);

    for (i=0;i<NUM_COUNTERS_CORE2;i++) {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) {
            msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister , 0x0ULL);

            if (perfmon_threadData[thread_id].counters[i].type == PMC)
            {
                flags |= (1<<(i-2));  /* enable counter */
            }
            else if (perfmon_threadData[thread_id].counters[i].type == FIXED)
            {
                flags |= (1ULL<<(i+32));  /* enable fixed counter */
            }
        }
    }

    if (perfmon_verbose)
    {
        printf("perfmon_start_counters: Write Register 0x%X , Flags: 0x%llX \n",
                MSR_PERF_GLOBAL_CTRL, LLU_CAST flags);
    }

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, flags);
    msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x300000003ULL);
}

void 
perfmon_stopCountersThread_core2(int thread_id)
{
    uint64_t flags;
    int i;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    /* stop counters */
    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);

    /* read out counter results */
    for (i=0;i<NUM_COUNTERS_CORE2;i++) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            perfmon_threadData[thread_id].counters[i].counterData =
                msr_read(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister);
        }
    }

    /* check overflow status */
    flags = msr_read(cpu_id,MSR_PERF_GLOBAL_STATUS);
    if((flags & 0x3) || (flags & (0x3ULL<<32)) ) 
    {
        printf ("Overflow occured \n");
        printf ("Status: 0x%llX \n", LLU_CAST flags);
    }
}

void 
perfmon_readCountersThread_core2(int thread_id)
{
    int i;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    /* read out counter results */
    for (i=0;i<NUM_COUNTERS_CORE2;i++) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            perfmon_threadData[thread_id].counters[i].counterData =
                msr_read(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister);
        }
    }
}

