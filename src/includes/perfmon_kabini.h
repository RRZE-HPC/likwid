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
#include <perfmon_kabini_counters.h>
#include <error.h>
#include <affinity.h>

static int perfmon_numCountersKabini = NUM_COUNTERS_KABINI;
static int perfmon_numArchEventsKabini = NUM_ARCH_EVENTS_KABINI;

int perfmon_init_kabini(int cpu_id)
{
    uint64_t flags = 0x0ULL;

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD16_PERFEVTSEL0, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD16_PERFEVTSEL1, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD16_PERFEVTSEL2, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD16_PERFEVTSEL3, 0x0ULL));

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id) ||
            lock_acquire(
                (int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id)
       )
    {
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD16_NB_PERFEVTSEL0, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD16_NB_PERFEVTSEL1, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD16_NB_PERFEVTSEL2, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD16_NB_PERFEVTSEL3, 0x0ULL));
    }

    flags |= (1<<16);  /* user mode flag */
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD16_PERFEVTSEL0, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD16_PERFEVTSEL1, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD16_PERFEVTSEL2, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD16_PERFEVTSEL3, flags));
    return 0;
}


int perfmon_setupCounterThread_kabini(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags;
    int cpu_id = groupSet->threads[thread_id].processorId;
    
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        PerfmonCounterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = kabini_counter_map[index].configRegister;

        /* only one thread accesses Uncore */
        if ( (kabini_counter_map[index].type == UNCORE) &&
                !(socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id) )
        {
            continue;
        }

        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        CHECK_MSR_READ_ERROR(msr_read(cpu_id,reg, &flags));
        flags &= ~(0xFFFFU); 

        /* AMD uses a 12 bit Event mask: [35:32][7:0] */
        flags |= ((uint64_t)(event->eventId>>8)<<32) + (event->umask<<8) + (event->eventId & ~(0xF00U));

        /*if (perfmon_verbose)
        {
            printf("[%d] perfmon_setup_counter: Write Register 0x%llX , Flags: 0x%llX \n",
                    cpu_id,
                    LLU_CAST reg,
                    LLU_CAST flags);
        }*/

        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags));
    }
    return 0;
}


int perfmon_startCountersThread_kabini(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            PerfmonCounterIndex index = eventSet->events[i].index;
            
            if (kabini_counter_map[index].type == PMC)
            {
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, kabini_counter_map[index].counterRegister , 0x0ULL));
                CHECK_MSR_READ_ERROR(msr_read(cpu_id, kabini_counter_map[index].configRegister, &flags));
                flags |= (1<<22);  /* enable flag */

                /*if (perfmon_verbose) 
                {
                    printf("perfmon_start_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                            LLU_CAST kabini_counter_map[index].configRegister,
                            LLU_CAST flags);
                }*/

                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, kabini_counter_map[index].configRegister , flags));

            }
            else if ( kabini_counter_map[index].type == UNCORE )
            {
                if(haveLock)
                {
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, kabini_counter_map[index].counterRegister , 0x0ULL));
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, kabini_counter_map[index].configRegister, &flags));
                    flags |= (1<<22);  /* enable flag */

                    /*if (perfmon_verbose)
                    {
                        printf("perfmon_start_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                                LLU_CAST kabini_counter_map[index].configRegister,
                                LLU_CAST flags);
                    }*/

                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, kabini_counter_map[index].configRegister , flags));
                }
            }
        }
    }
    return 0;
}

int perfmon_stopCountersThread_kabini(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags;
    int haveLock = 0;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            PerfmonCounterIndex index = eventSet->events[i].index;
            if ( kabini_counter_map[index].type == PMC )
            {
                CHECK_MSR_READ_ERROR(msr_read(cpu_id,kabini_counter_map[index].configRegister, &flags));
                flags &= ~(1<<22);  /* clear enable flag */
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, kabini_counter_map[index].configRegister , flags));

                /*if (perfmon_verbose)
                {
                    printf("perfmon_stop_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                            LLU_CAST kabini_counter_map[index].configRegister,
                            LLU_CAST flags);
                    printf("perfmon_stop_counters: Read Register 0x%llX , Flags: 0x%llX \n",
                            LLU_CAST kabini_counter_map[index].counterRegister,
                            LLU_CAST perfmon_threadData[thread_id].counters[i].counterData);
                }*/

                CHECK_MSR_READ_ERROR(msr_read(cpu_id, kabini_counter_map[index].counterRegister, &counter_result));
                eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
            }
            else if (kabini_counter_map[index].type == UNCORE)
            {
                if(haveLock)
                {
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id,kabini_counter_map[index].configRegister, &flags));
                    flags &= ~(1<<22);  /* clear enable flag */
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, kabini_counter_map[index].configRegister , flags));

                    /*if (perfmon_verbose)
                    {
                        printf("perfmon_stop_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                                LLU_CAST kabini_counter_map[index].configRegister,
                                LLU_CAST flags);
                    }*/

                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, kabini_counter_map[index].counterRegister, &counter_result));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                }
            }
        }
    }
    return 0;
}


int perfmon_readCountersThread_kabini(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }


    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            PerfmonCounterIndex index = eventSet->events[i].index;
            if ( kabini_counter_map[index].type == UNCORE )
            {
                if ( haveLock )
                {
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, kabini_counter_map[index].counterRegister, &counter_result));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                }
            }
            else
            {
                CHECK_MSR_READ_ERROR(msr_read(cpu_id, kabini_counter_map[index].counterRegister, &counter_result));
                eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
            }
        }
    }
    return 0;
}

