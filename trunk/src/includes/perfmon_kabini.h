/*
 * =======================================================================================
 *
 *      Filename:  perfmon_kabini.h
 *
 *      Description:  Header file of perfmon module for AMD Family16
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2012 Jan Treibig 
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

#include <perfmon_kabini_events.h>
#include <perfmon_kabini_groups.h>
#include <perfmon_kabini_counters.h>

static int perfmon_numCountersKabini = NUM_COUNTERS_KABINI;
static int perfmon_numGroupsKabini = NUM_GROUPS_KABINI;
static int perfmon_numArchEventsKabini = NUM_ARCH_EVENTS_KABINI;

void perfmon_init_kabini(PerfmonThread *thread)
{
    uint64_t flags = 0x0ULL;
    int cpu_id = thread->processorId;

    msr_write(cpu_id, MSR_AMD16_PERFEVTSEL0, 0x0ULL);
    msr_write(cpu_id, MSR_AMD16_PERFEVTSEL1, 0x0ULL);
    msr_write(cpu_id, MSR_AMD16_PERFEVTSEL2, 0x0ULL);
    msr_write(cpu_id, MSR_AMD16_PERFEVTSEL3, 0x0ULL);

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id) ||
            lock_acquire(
                (int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id)
       )
    {
        msr_write(cpu_id, MSR_AMD16_NB_PERFEVTSEL0, 0x0ULL);
        msr_write(cpu_id, MSR_AMD16_NB_PERFEVTSEL1, 0x0ULL);
        msr_write(cpu_id, MSR_AMD16_NB_PERFEVTSEL2, 0x0ULL);
        msr_write(cpu_id, MSR_AMD16_NB_PERFEVTSEL3, 0x0ULL);
    }

    flags |= (1<<16);  /* user mode flag */
    msr_write(cpu_id, MSR_AMD16_PERFEVTSEL0, flags);
    msr_write(cpu_id, MSR_AMD16_PERFEVTSEL1, flags);
    msr_write(cpu_id, MSR_AMD16_PERFEVTSEL2, flags);
    msr_write(cpu_id, MSR_AMD16_PERFEVTSEL3, flags);
}


void perfmon_setupCounterThread_kabini(
        int thread_id,
        PerfmonEvent* event,
        PerfmonCounterIndex index)
{
    uint64_t flags;
    uint64_t reg = kabini_counter_map[index].configRegister;
    int cpu_id = perfmon_threadData[thread_id].processorId;
    perfmon_threadData[thread_id].counters[index].init = TRUE;

    /* only one thread accesses Uncore */
    if ( (kabini_counter_map[index].type == UNCORE) &&
            !(socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id) )
    {
        return;
    }

    flags = msr_read(cpu_id,reg);
    flags &= ~(0xFFFFU); 

    /* AMD uses a 12 bit Event mask: [35:32][7:0] */
    flags |= ((uint64_t)(event->eventId>>8)<<32) + (event->umask<<8) + (event->eventId & ~(0xF00U));

    if (perfmon_verbose)
    {
        printf("[%d] perfmon_setup_counter: Write Register 0x%llX , Flags: 0x%llX \n",
                cpu_id,
                LLU_CAST reg,
                LLU_CAST flags);
    }

    msr_write(cpu_id, reg , flags);
}


void perfmon_startCountersThread_kabini(int thread_id)
{
    int haveLock = 0;
    uint64_t flags;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    for ( int i=0; i<NUM_COUNTERS_KABINI; i++ )
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE)
        {
            if (kabini_counter_map[i].type == PMC)
            {
                msr_write(cpu_id, kabini_counter_map[i].counterRegister , 0x0ULL);
                flags = msr_read(cpu_id, kabini_counter_map[i].configRegister);
                flags |= (1<<22);  /* enable flag */

                if (perfmon_verbose) 
                {
                    printf("perfmon_start_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                            LLU_CAST kabini_counter_map[i].configRegister,
                            LLU_CAST flags);
                }

                msr_write(cpu_id, kabini_counter_map[i].configRegister , flags);

            }
            else if ( kabini_counter_map[i].type == UNCORE )
            {
                if(haveLock)
                {
                    msr_write(cpu_id, kabini_counter_map[i].counterRegister , 0x0ULL);
                    flags = msr_read(cpu_id, kabini_counter_map[i].configRegister);
                    flags |= (1<<22);  /* enable flag */

                    if (perfmon_verbose)
                    {
                        printf("perfmon_start_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                                LLU_CAST kabini_counter_map[i].configRegister,
                                LLU_CAST flags);
                    }

                    msr_write(cpu_id, kabini_counter_map[i].configRegister , flags);
                }
            }
        }
    }
}

void perfmon_stopCountersThread_kabini(int thread_id)
{
    uint64_t flags;
    int haveLock = 0;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    for ( int i=0; i<NUM_COUNTERS_KABINI; i++ )
    {
        if ( perfmon_threadData[thread_id].counters[i].init == TRUE )
        {
            if ( kabini_counter_map[i].type == PMC )
            {
                flags = msr_read(cpu_id,kabini_counter_map[i].configRegister);
                flags &= ~(1<<22);  /* clear enable flag */
                msr_write(cpu_id, kabini_counter_map[i].configRegister , flags);
                perfmon_threadData[thread_id].counters[i].counterData =
                    msr_read(cpu_id, kabini_counter_map[i].counterRegister);

                if (perfmon_verbose)
                {
                    printf("perfmon_stop_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                            LLU_CAST kabini_counter_map[i].configRegister,
                            LLU_CAST flags);
                    printf("perfmon_stop_counters: Read Register 0x%llX , Flags: 0x%llX \n",
                            LLU_CAST kabini_counter_map[i].counterRegister,
                            LLU_CAST perfmon_threadData[thread_id].counters[i].counterData);
                }

            }
            else if (kabini_counter_map[i].type == UNCORE)
            {
                if(haveLock)
                {
                    flags = msr_read(cpu_id, kabini_counter_map[i].configRegister);
                    flags &= ~(1<<22);  /* clear enable flag */
                    msr_write(cpu_id, kabini_counter_map[i].configRegister , flags);

                    if (perfmon_verbose)
                    {
                        printf("perfmon_stop_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                                LLU_CAST kabini_counter_map[i].configRegister,
                                LLU_CAST flags);
                    }
                    perfmon_threadData[thread_id].counters[i].counterData =
                        msr_read(cpu_id, kabini_counter_map[i].counterRegister);
                }
            }
        }
    }
}


void perfmon_readCountersThread_kabini(int thread_id)
{
    int haveLock = 0;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }


    for (int i=0;i<NUM_COUNTERS_KABINI;i++)
    {
        if ( perfmon_threadData[thread_id].counters[i].init == TRUE )
        {
            if ( kabini_counter_map[i].type == UNCORE )
            {
                if ( haveLock )
                {
                    perfmon_threadData[thread_id].counters[i].counterData =
                        msr_read(cpu_id, kabini_counter_map[i].counterRegister);
                }
            }
            else
            {
                perfmon_threadData[thread_id].counters[i].counterData =
                    msr_read(cpu_id, kabini_counter_map[i].counterRegister);
            }
        }
    }
}

