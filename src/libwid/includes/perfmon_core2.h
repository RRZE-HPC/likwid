/*
 * =======================================================================================
 *
 *      Filename:  perfmon_core2.h
 *
 *      Description:  Header file of perfmon module for Core 2
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

#include <perfmon_core2_events.h>
#include <perfmon_core2_groups.h>
#include <perfmon_core2_counters.h>
#include <error.h>

static int perfmon_numCountersCore2 = NUM_COUNTERS_CORE2;
static int perfmon_numGroupsCore2 = NUM_GROUPS_CORE2;
static int perfmon_numArchEventsCore2 = NUM_ARCH_EVENTS_CORE2;

int perfmon_init_core2(int cpu_id)
{
    uint64_t flags = 0x0ULL;

    /* Initialize registers */
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL0, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL1, 0x0ULL));

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PEBS_ENABLE, 0x0ULL));

    /* always initialize fixed counters
     * FIXED 0: Instructions retired
     * FIXED 1: Clocks unhalted */
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x22ULL));

    /* Preinit of PMC counters */
    flags |= (1<<16);  /* user mode flag */
    flags |= (1<<19);  /* pin control flag */
    flags |= (1<<22);  /* enable flag */

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL0, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL1, flags));
    return 0;
}


int perfmon_setupCounterThread_core2(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    uint64_t flags;
    int cpu_id = groupSet->threads[thread_id].processorId;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        PerfmonCounterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = core2_counter_map[index].configRegister;
        
        switch (core2_counter_map[index].type)
        {
            case PMC:
                eventSet->events[i].threadCounter[thread_id].init = TRUE;
                CHECK_MSR_READ_ERROR(msr_read(cpu_id, reg, &flags));
                flags &= ~(0xFFFFU); 

                /* Intel with standard 8 bit event mask: [7:0] */
                flags |= (event->umask<<8) + event->eventId;

                if ( event->cfgBits != 0 ) /* set custom cfg and cmask */
                {
                    flags &= ~(0xFFFFU<<16);  /* clear upper 16bits */
                    flags |= ((event->cmask<<8) + event->cfgBits)<<16;
                }

                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags));

                /*if (perfmon_verbose)
                {
                    printf("[%d] perfmon_setup_counter: Write Register 0x%llX , Flags: 0x%llX \n",
                            cpu_id,
                            LLU_CAST reg,
                            LLU_CAST flags);
                }*/
                break;
            case FIXED:
                eventSet->events[i].threadCounter[thread_id].init = TRUE;
                break;
        }
    }
    return 0;
}

int perfmon_startCountersThread_core2(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags = 0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            PerfmonCounterIndex index = eventSet->events[i].index;
            
            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, core2_counter_map[index].counterRegister , 0x0ULL));

            if (core2_counter_map[i].type == PMC)
            {
                flags |= (1<<(i-2));  /* enable counter */
            }
            else if (core2_counter_map[i].type == FIXED)
            {
                flags |= (1ULL<<(i+32));  /* enable fixed counter */
            }
        }
    }

    /*if (perfmon_verbose)
    {
        printf("perfmon_start_counters: Write Register 0x%X , Flags: 0x%llX \n",
                MSR_PERF_GLOBAL_CTRL, LLU_CAST flags);
    }*/

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x300000003ULL));
    return 0;
}

int perfmon_stopCountersThread_core2(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags;
    uint64_t tmp;
    int cpu_id = groupSet->threads[thread_id].processorId;

    /* stop counters */
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));

    /* read out counter results */
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            PerfmonCounterIndex index = eventSet->events[i].index;
            CHECK_MSR_READ_ERROR(msr_read(cpu_id, core2_counter_map[index].counterRegister,
                                    (uint64_t*)&tmp));
            eventSet->events[i].threadCounter[thread_id].counterData = tmp;
        }
    }

    /* check overflow status */
    CHECK_MSR_READ_ERROR(msr_read(cpu_id,MSR_PERF_GLOBAL_STATUS, &flags));
    if ( (flags & 0x3) || (flags & (0x3ULL<<32)) )
    {
        printf ("Overflow occured \n");
        printf ("Status: 0x%llX \n", LLU_CAST flags);
    }
    return 0;
}

int perfmon_readCountersThread_core2(int thread_id, PerfmonEventSet* eventSet)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t tmp;
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            CHECK_MSR_READ_ERROR(msr_read(cpu_id, core2_counter_map[i].counterRegister, &tmp));
            eventSet->events[i].threadCounter[thread_id].counterData = tmp;
        }
    }
    return 0;
}

