/*
 * =======================================================================================
 *
 *      Filename:  perfmon_nehalem.h
 *
 *      Description:  Header File of perfmon module for Nehalem.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
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

#include <perfmon_nehalem_events.h>
#include <perfmon_nehalem_groups.h>
#include <perfmon_nehalem_counters.h>

static int perfmon_numCountersNehalem = NUM_COUNTERS_NEHALEM;
static int perfmon_numGroupsNehalem = NUM_GROUPS_NEHALEM;
static int perfmon_numArchEventsNehalem = NUM_ARCH_EVENTS_NEHALEM;

#define OFFSET_PMC 3
#define OFFSET_UPMC 7

void perfmon_init_nehalem(PerfmonThread *thread)
{
    uint64_t flags = 0x0ULL;
    int cpu_id = thread->processorId;

    msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x0ULL);
    msr_write(cpu_id, MSR_PERFEVTSEL0, 0x0ULL);
    msr_write(cpu_id, MSR_PERFEVTSEL1, 0x0ULL);
    msr_write(cpu_id, MSR_PERFEVTSEL2, 0x0ULL);
    msr_write(cpu_id, MSR_PERFEVTSEL3, 0x0ULL);
    msr_write(cpu_id, MSR_PMC0, 0x0ULL);
    msr_write(cpu_id, MSR_PMC1, 0x0ULL);
    msr_write(cpu_id, MSR_PMC2, 0x0ULL);
    msr_write(cpu_id, MSR_PMC3, 0x0ULL);
    msr_write(cpu_id, MSR_PERF_FIXED_CTR0, 0x0ULL);
    msr_write(cpu_id, MSR_PERF_FIXED_CTR1, 0x0ULL);
    msr_write(cpu_id, MSR_PERF_FIXED_CTR2, 0x0ULL);
    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);
    msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x0ULL);
    msr_write(cpu_id, MSR_PEBS_ENABLE, 0x0ULL);

    /* initialize fixed counters
     * FIXED 0: Instructions retired
     * FIXED 1: Clocks unhalted core
     * FIXED 2: Clocks unhalted ref */
    msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x222ULL);

    //    flags |= (1<<22);  /* enable flag */
    //    flags |= (1<<16);  /* user mode flag */
    setBit(flags,16); /* set user mode flag */
    setBit(flags,22); /* set enable flag */

    msr_write(cpu_id, MSR_PERFEVTSEL0, flags);
    msr_write(cpu_id, MSR_PERFEVTSEL1, flags);
    msr_write(cpu_id, MSR_PERFEVTSEL2, flags);
    msr_write(cpu_id, MSR_PERFEVTSEL3, flags);


    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id) ||
            lock_acquire(
                (int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id)
       )
    {
        /* UNCORE FIXED 0: Uncore cycles */
        msr_write(cpu_id, MSR_UNCORE_FIXED_CTR_CTRL, 0x01ULL);
        msr_write(cpu_id, MSR_UNCORE_FIXED_CTR_CTRL, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_FIXED_CTR0, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PERFEVTSEL0, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PERFEVTSEL1, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PERFEVTSEL2, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PERFEVTSEL3, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PERFEVTSEL4, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PERFEVTSEL5, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PERFEVTSEL6, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PERFEVTSEL7, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PMC0, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PMC1, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PMC2, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PMC3, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PMC4, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PMC5, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PMC6, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PMC7, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_PERF_GLOBAL_OVF_CTRL, 0x0ULL);
        msr_write(cpu_id, MSR_UNCORE_ADDR_OPCODE_MATCH, 0x0ULL);
        msr_write(cpu_id, MSR_OFFCORE_RESP0, 0x0ULL);

        /* Preinit of PERFEVSEL registers */
        clearBit(flags,16); /* set enable flag */

        msr_write(cpu_id, MSR_UNCORE_PERFEVTSEL0, flags);
        msr_write(cpu_id, MSR_UNCORE_PERFEVTSEL1, flags);
        msr_write(cpu_id, MSR_UNCORE_PERFEVTSEL2, flags);
        msr_write(cpu_id, MSR_UNCORE_PERFEVTSEL3, flags);
        msr_write(cpu_id, MSR_UNCORE_PERFEVTSEL4, flags);
        msr_write(cpu_id, MSR_UNCORE_PERFEVTSEL5, flags);
        msr_write(cpu_id, MSR_UNCORE_PERFEVTSEL6, flags);
        msr_write(cpu_id, MSR_UNCORE_PERFEVTSEL7, flags);
    }
}


void perfmon_setupCounterThread_nehalem(
        int thread_id,
        PerfmonEvent* event,
        PerfmonCounterIndex index)
{
    int haveLock = 0;
    uint64_t flags;
    uint64_t reg = nehalem_counter_map[index].configRegister;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    if ( nehalem_counter_map[index].type == PMC )
    {
        perfmon_threadData[thread_id].counters[index].init = TRUE;
        flags = msr_read(cpu_id,reg);
        flags &= ~(0xFFFFU);  /* clear lower 16bits */

        /* Intel with standard 8 bit event mask: [7:0] */
        flags |= (event->umask<<8) + event->eventId;

        if (event->cfgBits != 0) /* set custom cfg and cmask */
        {
            flags &= ~(0xFFFFU<<16);  /* clear upper 16bits */
            flags |= ((event->cmask<<8) + event->cfgBits)<<16;
        }

        msr_write(cpu_id, reg , flags);

        if (perfmon_verbose)
        {
            printf("[%d] perfmon_setup_counter: Write Register 0x%llX , Flags: 0x%llX \n",
                    cpu_id,
                    LLU_CAST reg,
                    LLU_CAST flags);
        }
    }
    else if ( nehalem_counter_map[index].type == UNCORE )
    {
        if(haveLock)
        {
            perfmon_threadData[thread_id].counters[index].init = TRUE;
            flags = msr_read(cpu_id,reg);
            flags &= ~(0xFFFFU);  /* clear lower 16bits */

            /* Intel with standard 8 bit event mask: [7:0] */
            flags |= (event->umask<<8) + event->eventId;

            if (event->cfgBits != 0) /* set custom cfg and cmask */
            {
                flags &= ~(0xFFFFU<<16);  /* clear upper 16bits */
                flags |= ((event->cmask<<8) + event->cfgBits)<<16;
            }

            msr_write(cpu_id, reg , flags);

            if (perfmon_verbose)
            {
                printf("[%d] perfmon_setup_counter: Write Register 0x%llX , Flags: 0x%llX \n",
                        cpu_id,
                        LLU_CAST reg,
                        LLU_CAST flags);
            }

        }
    }
    else if (nehalem_counter_map[index].type == FIXED)
    {
        perfmon_threadData[thread_id].counters[index].init = TRUE;
    }
}

void perfmon_startCountersThread_nehalem(int thread_id)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t uflags = 0x0ULL;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
        msr_write(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL);
        /* Fixed Uncore counter */
        uflags = 0x100000000ULL;
    }

    for ( int i=0; i<NUM_PMC; i++ )
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE)
        {
            if (nehalem_counter_map[i].type == PMC)
            {
                msr_write(cpu_id, nehalem_counter_map[i].counterRegister , 0x0ULL);
                flags |= (1<<(i-OFFSET_PMC));  /* enable counter */
            }
            else if (nehalem_counter_map[i].type == FIXED)
            {
                msr_write(cpu_id, nehalem_counter_map[i].counterRegister , 0x0ULL);
                flags |= (1ULL<<(i+32));  /* enable fixed counter */
            }
            else if (nehalem_counter_map[i].type == UNCORE)
            {
                if(haveLock)
                {
                    msr_write(cpu_id, nehalem_counter_map[i].counterRegister , 0x0ULL);
                    uflags |= (1<<(i-OFFSET_UPMC));  /* enable uncore counter */
                }
            }
        }
    }

    if (perfmon_verbose)
    {
        printf("perfmon_start_counters: Write Register 0x%X , \
                Flags: 0x%llX \n",MSR_PERF_GLOBAL_CTRL, LLU_CAST flags);
        printf("perfmon_start_counters: Write Register 0x%X , \
                Flags: 0x%llX \n",MSR_UNCORE_PERF_GLOBAL_CTRL, LLU_CAST uflags);
    }

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, flags);
    if (haveLock) msr_write(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, uflags);
    msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x30000000FULL);
}

void perfmon_stopCountersThread_nehalem(int thread_id)
{
    uint64_t flags;
    int haveLock = 0;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
        msr_write(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL);
    }

    for ( int i=0; i<NUM_COUNTERS_NEHALEM; i++ )
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE)
        {
            if (nehalem_counter_map[i].type == UNCORE)
            {
                if(haveLock)
                {
                    perfmon_threadData[thread_id].counters[i].counterData =
                        msr_read(cpu_id, nehalem_counter_map[i].counterRegister);
                }
            }
            else
            {
                perfmon_threadData[thread_id].counters[i].counterData =
                    msr_read(cpu_id, nehalem_counter_map[i].counterRegister);
            }
        }
    }

    flags = msr_read(cpu_id,MSR_PERF_GLOBAL_STATUS);
    printf ("Status: 0x%llX \n", LLU_CAST flags);

    if((flags & 0x3) || (flags & (0x3ULL<<32)) )
    {
        printf ("Overflow occured \n");
    }
}

void perfmon_readCountersThread_nehalem(int thread_id)
{
    int haveLock = 0;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    for ( int i=0; i<NUM_COUNTERS_NEHALEM; i++ )
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE)
        {
            if (nehalem_counter_map[i].type == UNCORE)
            {
                if(haveLock)
                {
                    perfmon_threadData[thread_id].counters[i].counterData =
                        msr_read(cpu_id, nehalem_counter_map[i].counterRegister);
                }
            }
            else
            {
                perfmon_threadData[thread_id].counters[i].counterData =
                    msr_read(cpu_id, nehalem_counter_map[i].counterRegister);
            }
        }
    }
}

