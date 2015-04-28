/*
 * =======================================================================================
 *
 *      Filename:  perfmon_k10.h
 *
 *      Description:  Header file of perfmon module for AMD K10
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *               Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig and Thomas Roehl
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
    return 0;
}

int k10_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent* event)
{
    uint64_t flags = 0x0ULL;

    flags |= (1ULL<<16);
    flags |= ((uint64_t)(event->eventId>>8)<<32) + (event->umask<<8) + (event->eventId & ~(0xF00U));

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
                    if ((event->options[j].value & 0xFFULL) < 0x04ULL)
                    {
                        flags |= (event->options[j].value & 0xFFULL) << 24;
                    }
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

int perfmon_setupCounterThread_k10(int thread_id, PerfmonEventSet* eventSet)
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
        if (type == PMC)
        {
            k10_pmc_setup(cpu_id, index, event);
            eventSet->events[i].threadCounter[thread_id].init = TRUE;
        }
    }
    return 0;
}

int perfmon_startCountersThread_k10(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags = 0x0ULL;
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
            uint32_t reg = counter_map[index].configRegister;
            uint32_t counter = counter_map[index].counterRegister;
            VERBOSEPRINTREG(cpu_id, counter, 0x0ULL, CLEAR_PMC);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter, 0x0ULL));
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, reg, &flags));
            VERBOSEPRINTREG(cpu_id, reg, flags, READ_PMC_CTRL);
            flags |= (1ULL<<22);  /* enable flag */
            VERBOSEPRINTREG(cpu_id, reg, flags, START_PMC);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
        }
    }
    return 0;
}

int perfmon_stopCountersThread_k10(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags = 0x0ULL;
    uint64_t tmp;
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
            tmp = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint32_t reg = counter_map[index].configRegister;
            uint32_t counter = counter_map[index].counterRegister;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, reg, &flags));
            VERBOSEPRINTREG(cpu_id, reg, flags, READ_PMC_CTRL);
            flags &= ~(1ULL<<22);  /* clear enable flag */
            VERBOSEPRINTREG(cpu_id, reg, flags, STOP_PMC);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &tmp));
            VERBOSEPRINTREG(cpu_id, counter, tmp, READ_PMC);
            if (tmp < eventSet->events[i].threadCounter[thread_id].counterData)
            {
                eventSet->events[i].threadCounter[thread_id].overflows++;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(tmp, 0, box_map[type].regWidth);
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
            RegisterType type = eventSet->events[i].type;
            if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
            {
                continue;
            }
            tmp = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint32_t counter = counter_map[index].counterRegister;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &tmp));
            VERBOSEPRINTREG(cpu_id, counter, tmp, READ_PMC);
            if (tmp < eventSet->events[i].threadCounter[thread_id].counterData)
            {
                eventSet->events[i].threadCounter[thread_id].overflows++;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(tmp, 0, box_map[type].regWidth);
        }
    }
    return 0;
}


int perfmon_finalizeCountersThread_k10(int thread_id, PerfmonEventSet* eventSet)
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
        if (reg)
        {
            VERBOSEPRINTREG(cpu_id, reg, 0x0ULL, CLEAR_CTRL);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, 0x0ULL));
        }
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
    }
    return 0;
}
