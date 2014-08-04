/*
 * =======================================================================================
 *
 *      Filename:  perfmon_k10.h
 *
 *      Description:  Header file of perfmon module for K10
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2013 Jan Treibig 
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

#include <perfmon_k10_events.h>
#include <perfmon_k10_counters.h>
#include <error.h>

static int perfmon_numCountersK10 = NUM_COUNTERS_K10;
static int perfmon_numArchEventsK10 = NUM_ARCH_EVENTS_K10;

int perfmon_init_k10(int cpu_id)
{
    uint64_t flags = 0x0ULL;

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD_PERFEVTSEL0, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD_PERFEVTSEL1, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD_PERFEVTSEL2, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD_PERFEVTSEL3, 0x0ULL));

    flags |= (1<<16);  /* user mode flag */

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD_PERFEVTSEL0, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD_PERFEVTSEL1, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD_PERFEVTSEL2, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_AMD_PERFEVTSEL3, flags));
    return 0;
}


int perfmon_setupCounterThread_k10(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    uint64_t flags;
    int cpu_id = groupSet->threads[thread_id].processorId;
    
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterIndex index = eventSet->events[i].index;
        uint64_t reg = k10_counter_map[index].configRegister;
        PerfmonEvent *event = &(eventSet->events[i].event);
        
        eventSet->events[i].threadCounter[thread_id].init = TRUE;

        CHECK_MSR_READ_ERROR(msr_read(cpu_id, reg, &flags));
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

int perfmon_startCountersThread_k10(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags;
    int cpu_id = groupSet->threads[thread_id].processorId;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, k10_counter_map[index].counterRegister , 0x0ULL));
            CHECK_MSR_READ_ERROR(msr_read(cpu_id, k10_counter_map[index].configRegister, &flags));
            flags |= (1<<22);  /* enable flag */

            /*if (perfmon_verbose)
            {
                printf("perfmon_start_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                        LLU_CAST k10_counter_map[index].configRegister,
                        LLU_CAST flags);
            }*/

            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, k10_counter_map[index].configRegister, flags));
        }
    }
    return 0;
}

int perfmon_stopCountersThread_k10(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags;
    uint64_t tmp;
    int cpu_id = groupSet->threads[thread_id].processorId;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            CHECK_MSR_READ_ERROR(msr_read(cpu_id, k10_counter_map[index].configRegister, &flags));
            flags &= ~(1<<22);  /* clear enable flag */
            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, k10_counter_map[index].configRegister , flags));

            /*if (perfmon_verbose)
            {
                printf("perfmon_stop_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                        LLU_CAST k10_counter_map[i].configRegister,
                        LLU_CAST flags);
            }*/

            CHECK_MSR_READ_ERROR(msr_read(cpu_id, k10_counter_map[index].counterRegister, &tmp));
            eventSet->events[i].threadCounter[thread_id].counterData = tmp;
        }
    }
    return 0;
}

int perfmon_readCountersThread_k10(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t tmp;
    int cpu_id = groupSet->threads[thread_id].processorId;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            CHECK_MSR_READ_ERROR(msr_read(cpu_id, k10_counter_map[i].counterRegister, &tmp));
            eventSet->events[i].threadCounter[thread_id].counterData = tmp;
        }
    }
    return 0;
}

