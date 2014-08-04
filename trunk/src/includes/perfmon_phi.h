/*
 * =======================================================================================
 *
 *      Filename:  perfmon_phi.h
 *
 *      Description:  Header File of perfmon module for Xeon Phi.
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

#include <perfmon_phi_events.h>
#include <perfmon_phi_counters.h>
#include <error.h>
#include <affinity.h>

static int perfmon_numCountersPhi = NUM_COUNTERS_PHI;
static int perfmon_numArchEventsPhi = NUM_ARCH_EVENTS_PHI;

int perfmon_init_phi(int cpu_id)
{
    uint32_t flags = 0x0UL;

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_MIC_PERFEVTSEL0, 0x0UL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_MIC_PERFEVTSEL1, 0x0UL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_MIC_PMC0, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_MIC_PMC1, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_MIC_SPFLT_CONTROL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_MIC_PERF_GLOBAL_CTRL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_MIC_PERF_GLOBAL_OVF_CTRL, 0x0ULL));

    flags |= (1<<16);  /* user mode flag */
    flags |= (1<<22);  /* enable flag */

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_MIC_PERFEVTSEL0, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_MIC_PERFEVTSEL1, flags));
    return 0;
}

int perfmon_setupCounterThread_phi(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags;
    int cpu_id = groupSet->threads[thread_id].processorId;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = phi_counter_map[index].configRegister;
        
        if (phi_counter_map[index].type == PMC)
        {
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
    }
    return 0;
}

int perfmon_startCountersThread_phi(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags = 0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_MIC_PERF_GLOBAL_CTRL, 0x0ULL));

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE) 
        {
            RegisterIndex index = eventSet->events[i].index;
            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, phi_counter_map[index].counterRegister , 0x0ULL));
            flags |= (1<<(index));  /* enable counter */
        }
    }

    /*if (perfmon_verbose)
    {
        printf("perfmon_start_counters: Write Register 0x%X , \
                Flags: 0x%llX \n",MSR_MIC_PERF_GLOBAL_CTRL, LLU_CAST flags);
    }*/

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_MIC_PERF_GLOBAL_CTRL, flags));
    flags |= (1ULL<<63);
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_MIC_SPFLT_CONTROL, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_MIC_PERF_GLOBAL_OVF_CTRL, 0x000000003ULL));
    return 0;
}

int perfmon_stopCountersThread_phi(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags;
    uint64_t counter_result = 0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_MIC_SPFLT_CONTROL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_MIC_PERF_GLOBAL_CTRL, 0x0ULL));

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            CHECK_MSR_WRITE_ERROR(msr_read(cpu_id, phi_counter_map[index].counterRegister, &counter_result));
            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
        }
    }

    CHECK_MSR_READ_ERROR(msr_read(cpu_id,MSR_MIC_PERF_GLOBAL_STATUS, &flags));
    //printf ("Status: 0x%llX \n", LLU_CAST flags);
    if((flags & 0x3))
    {
        printf ("Overflow occured \n");
    }
    return 0;
}

int perfmon_readCountersThread_phi(int thread_id, PerfmonEventSet* eventSet)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter_result = 0ULL;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            CHECK_MSR_WRITE_ERROR(msr_read(cpu_id, phi_counter_map[i].counterRegister, &counter_result));
            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
        }
    }
    return 0;
}

