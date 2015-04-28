/*
 * =======================================================================================
 *
 *      Filename:  perfmon_interlagos.h
 *
 *      Description:  Header file of perfmon module for AMD Interlagos
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *               Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2013 Jan Treibig and Thomas Roehl
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

#include <perfmon_interlagos_events.h>
#include <perfmon_interlagos_counters.h>
#include <error.h>

static int perfmon_numCountersInterlagos = NUM_COUNTERS_INTERLAGOS;
static int perfmon_numArchEventsInterlagos = NUM_ARCH_EVENTS_INTERLAGOS;


int perfmon_init_interlagos(int cpu_id)
{
    lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    return 0;
}

int ilg_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent* event)
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
                    if ((event->options[j].value & 0xFFULL) < 0x20)
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

int ilg_uncore_setup(int cpu_id, RegisterIndex index, PerfmonEvent* event)
{
    uint64_t flags = 0x0ULL;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags |= ((uint64_t)(event->eventId>>8)<<32) + (event->umask<<8) + (event->eventId & ~(0xF00U));

    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_UNCORE);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    return 0;
}


int perfmon_setupCounterThread_interlagos(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

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
        switch(type)
        {
            case PMC:
                ilg_pmc_setup(cpu_id, index, event);
                break;
            case UNCORE:
                ilg_uncore_setup(cpu_id, index, event);
                break;
            default:
                break;
        }
    }
    return 0;
}


int perfmon_startCountersThread_interlagos(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

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
            uint32_t counter = counter_map[index].counterRegister;
            uint32_t reg = counter_map[index].configRegister;
            if (type == PMC || ((type == UNCORE) && (haveLock)))
            {
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter, 0x0ULL));
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, reg, &flags));
                flags |= (1<<22);  /* enable flag */
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
            }
        }
    }
    return 0;
}

int perfmon_stopCountersThread_interlagos(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags = 0x0ULL;
    int haveLock = 0;
    uint64_t tmp;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

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
            uint32_t reg = counter_map[index].configRegister;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, reg, &flags));
                    flags &= ~(1<<22);  /* clear enable flag */
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &tmp));
                    break;
                case UNCORE:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, reg, &flags));
                        flags &= ~(1<<22);  /* clear enable flag */
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &tmp));
                    }
                    break;
                default:
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(tmp, 0, box_map[type].regWidth);
        }
    }
    return 0;
}


int perfmon_readCountersThread_interlagos(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t tmp;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

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
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &tmp));
                    VERBOSEPRINTREG(cpu_id, counter, LLU_CAST tmp, READ_PMC);
                    break;
                case UNCORE:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &tmp));
                    VERBOSEPRINTREG(cpu_id, counter, LLU_CAST tmp, READ_UNCORE);
                    break;
                default:
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(tmp, 0, box_map[type].regWidth);
        }
    }
    return 0;
}


int perfmon_finalizeCountersThread_interlagos(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        uint32_t reg = counter_map[index].configRegister;
        if ((reg) && (((type == PMC)||(type == FIXED))||((type >= UNCORE) && (haveLock))))
        {
            VERBOSEPRINTREG(cpu_id, reg, LLU_CAST 0x0ULL, CLEAR_CTRL);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, 0x0ULL));
        }
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
    }
    return 0;
}
