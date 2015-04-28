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
 *               Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2012 Jan Treibig and Thomas Roehl
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
    lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &tile_lock[affinity_thread2tile_lookup[cpu_id]], cpu_id);
    return 0;
}


int k16_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent* event)
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
                    if ((event->options[j].value & 0xFFULL) < 0x04)
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

int k16_uncore_setup(int cpu_id, RegisterIndex index, PerfmonEvent* event)
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

int k16_cache_setup(int cpu_id, RegisterIndex index, PerfmonEvent* event)
{
    uint64_t flags = 0x0ULL;

    if (tile_lock[affinity_thread2tile_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags |= ((uint64_t)(event->eventId>>8)<<32) + (event->umask<<8) + (event->eventId & ~(0xF00U));
    if (event->numberOfOptions > 0)
    {
        for(int j=0;j<event->numberOfOptions;j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_INVERT:
                    flags |= (1ULL<<23);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    if ((event->options[j].value & 0xFFULL) < 0x04)
                    {
                        flags |= (event->options[j].value & 0xFFULL) << 24;
                    }
                    break;
                case EVENT_OPTION_TID:
                    flags |= (~((uint64_t)(event->options[j].value & 0xFULL))) << 56;
                    break;
                case EVENT_OPTION_NID:
                    flags |= (~((uint64_t)(event->options[j].value & 0xFULL))) << 48;
                    break;
                default:
                    break;
            }
        }
    }
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_CBOX);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    return 0;
}

int perfmon_setupCounterThread_kabini(int thread_id, PerfmonEventSet* eventSet)
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
        switch (type)
        {
            case PMC:
                k16_pmc_setup(cpu_id, index, event);
                break;
            case UNCORE:
                k16_uncore_setup(cpu_id, index, event);
                break;
            case CBOX0:
                k16_cache_setup(cpu_id, index, event);
                break;
            default:
                break;
        }
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
    }
    return 0;
}


int perfmon_startCountersThread_kabini(int thread_id, PerfmonEventSet* eventSet)
{
    int haveSLock = 0;
    int haveTLock = 0;
    uint64_t flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveSLock = 1;
    }
    if (tile_lock[affinity_thread2tile_lookup[cpu_id]] == cpu_id)
    {
        haveTLock = 1;
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
            uint32_t reg = counter_map[index].configRegister;
            uint32_t counter = counter_map[index].counterRegister;

            if ((type == PMC) ||
                ((type == UNCORE) && (haveSLock)) ||
                ((type == CBOX0) && (haveTLock)))
            {
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter, 0x0ULL));
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, reg, &flags));
                flags |= (1ULL<<22);  /* enable flag */
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
            }
        }
    }
    return 0;
}

int perfmon_stopCountersThread_kabini(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags = 0x0ULL;
    int haveSLock = 0;
    int haveTLock = 0;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveSLock = 1;
    }
    if (tile_lock[affinity_thread2tile_lookup[cpu_id]] == cpu_id)
    {
        haveTLock = 1;
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
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint32_t reg = counter_map[index].configRegister;
            uint32_t counter = counter_map[index].counterRegister;
            if ((type == PMC) ||
                ((type == UNCORE) && (haveSLock)) ||
                ((type == CBOX0) && (haveTLock)))
            {
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, reg, &flags));
                flags &= ~(1ULL<<22);  /* clear enable flag */
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                {
                    eventSet->events[i].threadCounter[thread_id].overflows++;
                }
                eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
            }
        }
    }
    return 0;
}


int perfmon_readCountersThread_kabini(int thread_id, PerfmonEventSet* eventSet)
{
    int haveSLock = 0;
    int haveTLock = 0;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveSLock = 1;
    }
    if (tile_lock[affinity_thread2tile_lookup[cpu_id]] == cpu_id)
    {
        haveTLock = 1;
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
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint32_t counter = counter_map[index].counterRegister;

            if ((type == PMC) ||
                ((type == UNCORE) && (haveSLock)) ||
                ((type == CBOX0) && (haveTLock)))
            {
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                VERBOSEPRINTREG(cpu_id, counter, counter_result, CLEAR_CTRL);
                if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                {
                    eventSet->events[i].threadCounter[thread_id].overflows++;
                }
                eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
            }
        }
    }
    return 0;
}


int perfmon_finalizeCountersThread_kabini(int thread_id, PerfmonEventSet* eventSet)
{
    int haveSLock = 0;
    int haveTLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveSLock = 1;
    }
    if (tile_lock[affinity_thread2tile_lookup[cpu_id]] == cpu_id)
    {
        haveTLock = 1;
    }
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        if ((type == PMC) ||
            ((type == UNCORE) && (haveSLock)) ||
            ((type == CBOX0) && (haveTLock)))
        {
            VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, 0x0ULL, CLEAR_CTRL);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, 0x0ULL));
            eventSet->events[i].threadCounter[thread_id].init = FALSE;
        }
    }
    return 0;
}
