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
    return 0;
}

int pm_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x0ULL;

    flags = (1ULL<<16)|(1ULL<<19);
    flags |= (event->umask<<8) + event->eventId;

    if (event->numberOfOptions > 0)
    {
        for(int j=0;j<event->numberOfOptions;j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                case EVENT_OPTION_COUNT_KERNEL:
                    flags |= (1ULL<<17);
                    break;
                case EVENT_OPTION_INVERT:
                    flags |= (1ULL<<23);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= (event->options[j].value & 0xFFULL)<<24;
                    break;
                default:
                    break;
            }
        }
    }
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_PMC);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    return 0;
}

int perfmon_setupCounterThread_pm(int thread_id, PerfmonEventSet* eventSet)
{
    int cpu_id = groupSet->threads[thread_id].processorId;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        pm_pmc_setup(cpu_id, index, event);
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
            RegisterType type = eventSet->events[i].type;
            if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
            {
                continue;
            }
            RegisterIndex index = eventSet->events[i].index;
            VERBOSEPRINTREG(cpu_id, counter_map[index].counterRegister, LLU_CAST 0x0ULL, SETUP_PMC_CTR);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].counterRegister , 0x0ULL));
        }
    }
    if (eventSet->numberOfEvents > 0)
    {
        /* on p6 only MSR_PERFEVTSEL0 has the enable bit
         * it enables both counters as long MSR_PERFEVTSEL1 
         * has a valid configuration */
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERFEVTSEL0, &flags));
        flags |= (1<<22);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERFEVTSEL0, flags));
        VERBOSEPRINTREG(cpu_id, MSR_PERFEVTSEL0, LLU_CAST flags, UNFREEZE_PMC);
    }
    return 0;
}

int perfmon_stopCountersThread_pm(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERFEVTSEL0, &counter_result));
    counter_result &= ~(1<<22);
    VERBOSEPRINTREG(cpu_id, MSR_PERFEVTSEL0, counter_result, FREEZE_PMC);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERFEVTSEL0, counter_result));

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE) 
        {
            RegisterType type = eventSet->events[i].type;
            if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
            {
                continue;
            }
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter_map[index].counterRegister, &counter_result));
            VERBOSEPRINTREG(cpu_id, counter_map[index].counterRegister, counter_result, READ_PMC);
            if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
            {
                eventSet->events[i].threadCounter[thread_id].overflows++;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
        }
    }
    return 0;
}

int perfmon_readCountersThread_pm(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t counter_result = 0x0ULL;
    uint64_t pmc_flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERFEVTSEL0, &pmc_flags));
    pmc_flags &= ~(1<<22);
    VERBOSEPRINTREG(cpu_id, MSR_PERFEVTSEL0, pmc_flags & ~(1<<22), FREEZE_PMC);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERFEVTSEL0, pmc_flags & ~(1<<22)));

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE) 
        {
            RegisterType type = eventSet->events[i].type;
            if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
            {
                continue;
            }
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint32_t counter = counter_map[index].counterRegister;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
            if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
            {
                eventSet->events[i].threadCounter[thread_id].overflows++;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
        }
    }

    VERBOSEPRINTREG(cpu_id, MSR_PERFEVTSEL0, pmc_flags, UNFREEZE_PMC);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERFEVTSEL0, pmc_flags));
    return 0;
}

int perfmon_finalizeCountersThread_pm(int thread_id, PerfmonEventSet* eventSet)
{
    int cpu_id = groupSet->threads[thread_id].processorId;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        uint32_t reg = counter_map[index].configRegister;
        if ((reg) && ((type == PMC)||(type == FIXED)))
        {
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, 0x0ULL));
            VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, reg, 0x0ULL, CLEAR_CTL);
        }
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
    }
    return 0;
}

