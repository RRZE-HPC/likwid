/*
 * =======================================================================================
 *
 *      Filename:  perfmon_silvermont.h
 *
 *      Description:  Header file of perfmon module for Intel Atom Silvermont
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
 
#include <perfmon_silvermont_events.h>
#include <perfmon_silvermont_groups.h>
#include <perfmon_silvermont_counters.h>

static int perfmon_numCountersSilvermont = NUM_COUNTERS_SILVERMONT;
static int perfmon_numGroupsSilvermont = NUM_GROUPS_SILVERMONT;
static int perfmon_numArchEventsSilvermont = NUM_ARCH_EVENTS_SILVERMONT;


void perfmon_init_silvermont(PerfmonThread *thread)
{
    uint64_t flags = 0x0ULL;
    int cpu_id = thread->processorId;
    lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);

    /* Initialize registers */
    msr_write(cpu_id, MSR_PERFEVTSEL0, 0x0ULL);
    msr_write(cpu_id, MSR_PERFEVTSEL1, 0x0ULL);
    msr_write(cpu_id, MSR_OFFCORE_RESP0, 0x0ULL);
    msr_write(cpu_id, MSR_OFFCORE_RESP1, 0x0ULL);

    msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x0ULL);
    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);
    msr_write(cpu_id, MSR_PEBS_ENABLE, 0x0ULL);
}

void perfmon_setupCounterThread_silvermont(
        int thread_id,
        PerfmonEvent* event,
        PerfmonCounterIndex index)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint32_t uflags;
    uint64_t reg = silvermont_counter_map[index].configRegister;
    int cpu_id = perfmon_threadData[thread_id].processorId;
    uint64_t fixed_flags = msr_read(cpu_id, MSR_PERF_FIXED_CTR_CTRL);
    uint64_t orig_fixed_flags = fixed_flags;
    perfmon_threadData[thread_id].counters[index].init = TRUE;

    switch (silvermont_counter_map[index].type)
    {
        case PMC:

            flags = (1<<16)|(1<<22);
            flags &= ~(0xFFFFU);   /* clear lower 16bits */

            /* Intel with standard 8 bit event mask: [7:0] */
            flags |= (event->umask<<8) + event->eventId;



            if (perfmon_verbose)
            {
                printf("[%d] perfmon_setup_counter PMC: Write Register 0x%llX , Flags: 0x%llX \n",
                        cpu_id,
                        LLU_CAST reg,
                        LLU_CAST flags);
            }
            msr_write(cpu_id, reg , flags);

            // Offcore event with additional configuration register
            // We included the additional register as counterRegister2
            // to avoid creating a new data structure
            // cfgBits contain offset of "request type" bit
            // cmask contain offset of "response type" bit
            if (event->eventId == 0xB7) 
            {
                if (event->umask == 0x01)
                {
                    reg = MSR_OFFCORE_RESP0;
                }
                else if (event->umask == 0x02)
                {
                    reg = MSR_OFFCORE_RESP1;
                }
                flags = 0x0ULL;
                flags = (1<<event->cfgBits)|(1<<event->cmask);
                msr_write(cpu_id, reg , flags);
            }

            break;

        case FIXED:
            fixed_flags |= (2ULL<<(index*4));
            break;

        case POWER:
            break;

        default:
            /* should never be reached */
            break;
    }
    if (fixed_flags != orig_fixed_flags)
    {
        msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, fixed_flags);
    }
}


void perfmon_startCountersThread_silvermont(int thread_id)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint32_t uflags = 0x10000UL; /* Clear freeze bit */
    uint64_t fixed_flags = 0x0ULL;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);

    for ( int i=0; i<perfmon_numCountersSilvermont; i++ )
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE)
        {
            switch (silvermont_counter_map[i].type)
            {
                case PMC:
                    msr_write(cpu_id, silvermont_counter_map[i].counterRegister, 0x0ULL);
                    flags |= (1<<(i-OFFSET_PMC));  /* enable counter */
                    break;

                case FIXED:
                    msr_write(cpu_id, silvermont_counter_map[i].counterRegister, 0x0ULL);
                    flags |= (1ULL<<(i+32));  /* enable fixed counter */
                    break;

                case POWER:
                    if(haveLock)
                    {
                        perfmon_threadData[thread_id].counters[i].counterData =
                            power_read(cpu_id, silvermont_counter_map[i].counterRegister);
                    }

                    break;

                default:
                    /* should never be reached */
                    break;
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
    if (flags != 0x0ULL)
    {
        msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, flags);
    }
}


void perfmon_stopCountersThread_silvermont(int thread_id)
{
    uint64_t flags;
    uint32_t uflags = 0x10100UL; /* Set freeze bit */
    uint64_t counter_result = 0x0ULL;
    int haveLock = 0;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);

    for ( int i=0; i < perfmon_numCountersSilvermont; i++ ) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            switch (silvermont_counter_map[i].type)
            {
                case PMC:

                case FIXED:
                    perfmon_threadData[thread_id].counters[i].counterData =
                        (double)msr_read(cpu_id, silvermont_counter_map[i].counterRegister);
                    break;

                case POWER:
                    if(haveLock)
                    {
                        perfmon_threadData[thread_id].counters[i].counterData =
                            power_info.energyUnit *
                            ( power_read(cpu_id, silvermont_counter_map[i].counterRegister) -
                              perfmon_threadData[thread_id].counters[i].counterData);
                    }
                    break;

                case THERMAL:
                        perfmon_threadData[thread_id].counters[i].counterData =
                             thermal_read(cpu_id);
                    break;

                default:
                    /* should never be reached */
                    break;
            }
        }
    }

    flags = msr_read(cpu_id,MSR_PERF_GLOBAL_STATUS);
    //    printf ("Status: 0x%llX \n", LLU_CAST flags);
    if ( (flags & 0x3) || (flags & (0x3ULL<<32)) ) 
    {
        printf ("Overflow occured \n");
    }
}

void perfmon_readCountersThread_silvermont(int thread_id)
{
    uint64_t counter_result = 0x0ULL;
    int haveLock = 0;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    for ( int i=0; i<perfmon_numCountersSilvermont; i++ )
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE)
        {
            if ((silvermont_counter_map[i].type == PMC) ||
                    (silvermont_counter_map[i].type == FIXED))
            {
                perfmon_threadData[thread_id].counters[i].counterData =
                    msr_read(cpu_id, silvermont_counter_map[i].counterRegister);
            }
            else
            {
                if(haveLock)
                {
                    switch (silvermont_counter_map[i].type)
                    {
                        case POWER:
                            perfmon_threadData[thread_id].counters[i].counterData =
                                power_info.energyUnit *
                                power_read(cpu_id, silvermont_counter_map[i].counterRegister);
                            break;

                        default:
                            /* should never be reached */
                            break;
                    }
                }
            }
        }
    }
}
