/*
 * =======================================================================================
 *
 *      Filename:  perfmon_pm.h
 *
 *      Description:  Header File of perfmon module Pentium M.
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

#include <perfmon_pm_events.h>
#include <perfmon_pm_counters.h>
#include <error.h>
#include <affinity.h>


static int perfmon_numCounters_pm = NUM_COUNTERS_PM;
static int perfmon_numArchEvents_pm = NUM_ARCH_EVENTS_PM;


int perfmon_init_pm(int cpu_id)
{
    uint64_t flags = 0x0ULL;

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL0, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL1, 0x0ULL));

    /* Preinit of two PMC counters */
    flags |= (1<<16);  /* user mode flag */
    flags |= (1<<19);  /* pin control flag */
    //    flags |= (1<<22);  /* enable flag */

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL0, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL1, flags));
    return 0;
}

int perfmon_setupCounterThread_pm(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags;
    int cpu_id = groupSet->threads[thread_id].processorId;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        PerfmonCounterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = pm_counter_map[index].configRegister;
        
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, reg, &flags));
        flags &= ~(0xFFFFU); 

        /* Intel with standard 8 bit event mask: [7:0] */
        flags |= (event->umask<<8) + event->eventId;

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


int perfmon_startCountersThread_pm(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags = 0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, pm_counter_map[0].counterRegister , 0x0ULL));
            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, pm_counter_map[1].counterRegister , 0x0ULL));

            /* on p6 only MSR_PERFEVTSEL0 has the enable bit
             * it enables both counters as long MSR_PERFEVTSEL1 
             * has a valid configuration */
            CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_PERFEVTSEL0, &flags));
            flags |= (1<<22);  /* enable flag */

            /*if (perfmon_verbose)
            {
                printf("perfmon_start_counters: Write Register 0x%X , \
                        Flags: 0x%llX \n",MSR_PERFEVTSEL0, LLU_CAST flags);
            }*/

            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL0, flags));
        }
    }
    return 0;
}

int perfmon_stopCountersThread_pm(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL0, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL1, 0x0ULL));

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE) 
        {
            PerfmonCounterIndex index = eventSet->events[i].index;
            CHECK_MSR_READ_ERROR(msr_read(cpu_id, pm_counter_map[index].counterRegister, &counter_result));
            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
        }
    }
    return 0;
}

/*void perfmon_printDerivedMetrics_pm(PerfmonGroup group)
{

    switch ( group )
    {
        case FLOPS_DP:

        case FLOPS_SP:

        case L2:

        case BRANCH:

        case _NOGROUP:
            fprintf (stderr, "The Pentium M supports only two counters. Therefore derived metrics are not computed due to missing runtime!\n" );
            break;

        default:
            fprintf (stderr, "perfmon_printDerivedMetricsCore2: Unknown group! Exiting!\n" );
            exit (EXIT_FAILURE);
            break;
    }
}*/


