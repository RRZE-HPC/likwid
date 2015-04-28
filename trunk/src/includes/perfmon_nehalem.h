/*
 * =======================================================================================
 *
 *      Filename:  perfmon_nehalem.h
 *
 *      Description:  Header File of perfmon module for Nehalem.
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

#include <perfmon_nehalem_events.h>
#include <perfmon_nehalem_counters.h>
#include <error.h>
#include <affinity.h>


static int perfmon_numCountersNehalem = NUM_COUNTERS_NEHALEM;
static int perfmon_numArchEventsNehalem = NUM_ARCH_EVENTS_NEHALEM;


int perfmon_init_nehalem(int cpu_id)
{
    lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &tile_lock[affinity_thread2tile_lookup[cpu_id]], cpu_id);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL));
    return 0;
}

uint32_t neh_fixed_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t flags = (1ULL<<(1+(index*4)));
    for(j = 0; j < event->numberOfOptions; j++)
    {
        switch (event->options[j].type)
        {
            case EVENT_OPTION_COUNT_KERNEL:
                flags |= (1ULL<<(index*4));
                break;
            case EVENT_OPTION_ANYTHREAD:
                flags |= (1ULL<<(2+(index*4)));
            default:
                break;
        }
    }
    return flags;
}

int neh_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    int haveLock = 0;
    uint64_t offcore_flags = 0x0ULL;

    if (tile_lock[affinity_thread2tile_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    flags = (1ULL<<22)|(1ULL<<16);
    flags |= (event->umask<<8) + event->eventId;
    /* set custom cfg and cmask */
    if ((event->cfgBits != 0) &&
        (event->eventId != 0xB7) &&
        (event->eventId != 0xBB))
    {
        flags |= ((event->cmask<<8) + event->cfgBits)<<16;
    }
    if (event->numberOfOptions > 0)
    {
        for(j = 0; j < event->numberOfOptions; j++)
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
                case EVENT_OPTION_ANYTHREAD:
                    flags |= (1ULL<<21);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= (event->options[j].value & 0xFFULL)<<24;
                    break;
                case EVENT_OPTION_MATCH0:
                    offcore_flags |= (event->options[j].value & 0xFF);
                    break;
                case EVENT_OPTION_MATCH1:
                    offcore_flags |= (event->options[j].value & 0xF7)<<7;
                    break;
                default:
                    break;
            }
        }
    }
    // Offcore event with additional configuration register
    // cfgBits contain offset of "request type" bit
    // cmask contain offset of "response type" bit
    if ((haveLock) && (event->eventId == 0xB7))
    {
        if ((event->cfgBits != 0xFF) && (event->cmask != 0xFF))
        {
            offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);
        }
        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, offcore_flags));
    }
    if ((haveLock) && (event->eventId == 0xBB) &&
        ((cpuid_info.model == NEHALEM_WESTMERE) || (cpuid_info.model == NEHALEM_WESTMERE_M)))
    {
        if ((event->cfgBits != 0xFF) && (event->cmask != 0xFF))
        {
            offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);
        }
        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP1, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP1, offcore_flags));
    }
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_PMC);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    return 0;
}

int neh_uncore_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    uint64_t mask_flags = 0x0ULL;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags = (1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;
    if (event->cfgBits != 0x0 && event->eventId != 0x35) /* set custom cfg and cmask */
    {
        flags |= ((event->cmask<<8) + event->cfgBits)<<16;
    }
    else if (event->cfgBits != 0x0 && event->eventId == 0x35) /* set custom cfg and cmask */
    {
        mask_flags |= ((uint64_t)event->cfgBits)<<61;
        if (event->cmask != 0x0)
        {
            mask_flags |= ((uint64_t)event->cmask)<<40;
        }
    }
    if (event->numberOfOptions > 0)
    {
        for(j = 0; j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                case EVENT_OPTION_INVERT:
                    flags |= (1ULL<<23);
                    break;
                case EVENT_OPTION_ANYTHREAD:
                    flags |= (1ULL<<21);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= (event->options[j].value & 0xFFULL)<<24;
                    break;
                case EVENT_OPTION_MATCH0:
                    mask_flags |= field64(event->options[j].value,3,37)<<2;
                    break;
                case EVENT_OPTION_OPCODE:
                    mask_flags |= field64(event->options[j].value,0,8)<<40;
                    break;
                default:
                    break;
            }
        }
    }
    if ((mask_flags != 0x0ULL) && (event->eventId == 0x35))
    {
        if ((cpuid_info.model == NEHALEM_BLOOMFIELD) ||
             (cpuid_info.model == NEHALEM_LYNNFIELD) ||
             (cpuid_info.model == NEHALEM_LYNNFIELD_M))
        {
            DEBUG_PLAIN_PRINT(DEBUGLEV_ONLY_ERROR, Register documented in SDM but ADDR_OPCODE_MATCH event not documented for Nehalem architectures);
        }
        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_ADDR_OPCODE_MATCH, LLU_CAST mask_flags, SETUP_UNCORE_MATCH);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNCORE_ADDR_OPCODE_MATCH, mask_flags));
    }
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_UNCORE);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    return 0;
}

int perfmon_setupCounterThread_nehalem(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t fixed_flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, 0x0ULL));
    }
    if (haveLock && (eventSet->regTypeMask & ~(0xF)))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL));
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

        switch (type)
        {
            case PMC:
                neh_pmc_setup(cpu_id, index, event);
                break;
            case FIXED:
                fixed_flags |= neh_fixed_setup(cpu_id, index, event);
                break;
            case UNCORE:
                if (haveLock)
                {
                    if (index < NUM_COUNTERS_UNCORE_NEHALEM-1)
                    {
                        neh_uncore_setup(cpu_id, index, event);
                    }
                    else
                    {
                        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_FIXED_CTR_CTRL, LLU_CAST 0x1ULL, SETUP_UPMCFIX);
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNCORE_FIXED_CTR_CTRL, 0x1ULL));
                    }
                }
                break;
            default:
                break;
        }
    }
    if (fixed_flags != 0x0ULL)
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_FIXED_CTR_CTRL, LLU_CAST fixed_flags, SETUP_FIXED);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_FIXED_CTR_CTRL, fixed_flags));
    }
    return 0;
}

int perfmon_startCountersThread_nehalem(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t uflags = 0x0ULL;
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
            uint64_t counter = counter_map[index].counterRegister;
            switch(type)
            {
                case PMC:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter, 0x0ULL));
                    flags |= (1ULL<<(index - cpuid_info.perf_num_fixed_ctr));  /* enable counter */
                    break;
                case FIXED:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter, 0x0ULL));
                    flags |= (1ULL<<(index+32));  /* enable fixed counter */
                    break;
                case UNCORE:
                    if(haveLock)
                    {
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter, 0x0ULL));
                        if (index < NUM_COUNTERS_UNCORE_NEHALEM-1)
                        {
                            uflags |= (1ULL<<(index-NUM_COUNTERS_CORE_NEHALEM));  /* enable uncore counter */
                        }
                        else
                        {
                            uflags |= (1ULL<<32);
                        }
                    }
                    break;
                default:
                    break;
            }
        }
    }

    if (haveLock && (uflags != 0x0ULL) && (eventSet->regTypeMask & ~(0xF)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, LLU_CAST uflags, UNFREEZE_UNCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNCORE_PERF_GLOBAL_CTRL, uflags));
    }

    if ((flags != 0x0ULL) && (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED))))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, UNFREEZE_PMC_AND_FIXED);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, flags));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<63)|(1ULL<<62)|flags));
    }
    return 0;
}

#define NEH_CHECK_OVERFLOW(offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_STATUS, &tmp)); \
        if (tmp & (1ULL<<offset)) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (tmp & (1ULL<<offset)))); \
        } \
    }

#define NEH_CHECK_UNCORE_OVERFLOW(offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_UNCORE_PERF_GLOBAL_STATUS, &tmp)); \
        if (tmp & (1ULL<<offset)) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNCORE_PERF_GLOBAL_OVF_CTRL, (tmp & (1ULL<<offset)))); \
        } \
    }

int perfmon_stopCountersThread_nehalem(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    if (haveLock && (eventSet->regTypeMask & ~(0xF)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_UNCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL));
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
            uint64_t counter = counter_map[index].counterRegister;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                    VERBOSEPRINTREG(cpu_id, counter, counter_result, READ_PMC);
                    NEH_CHECK_OVERFLOW(index - cpuid_info.perf_num_fixed_ctr);
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                    VERBOSEPRINTREG(cpu_id, counter, counter_result, READ_FIXED);
                    NEH_CHECK_OVERFLOW(index + 32);
                    break;
                case UNCORE:
                    if(haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                        VERBOSEPRINTREG(cpu_id, counter, counter_result, READ_UNCORE);
                        if (index < NUM_COUNTERS_UNCORE_NEHALEM-1)
                        {
                            NEH_CHECK_UNCORE_OVERFLOW(index - NUM_COUNTERS_CORE_NEHALEM);
                        }
                        else
                        {
                            NEH_CHECK_UNCORE_OVERFLOW(32);
                        }
                    }
                    break;
                default:
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
        }
    }
    return 0;
}

int perfmon_readCountersThread_nehalem(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t pmc_flags = 0x0ULL;
    uint64_t uncore_flags = 0x0ULL;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, &pmc_flags));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    if (haveLock && (eventSet->regTypeMask & ~(0xF)))
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_UNCORE_PERF_GLOBAL_CTRL, &uncore_flags));
        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_UNCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL));
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
            uint64_t counter = counter_map[index].counterRegister;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                    VERBOSEPRINTREG(cpu_id, counter, counter_result, READ_PMC);
                    NEH_CHECK_OVERFLOW(index - cpuid_info.perf_num_fixed_ctr);
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                    VERBOSEPRINTREG(cpu_id, counter, counter_result, READ_FIXED);
                    NEH_CHECK_OVERFLOW(index + 32);
                    break;
                case UNCORE:
                    if(haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                        VERBOSEPRINTREG(cpu_id, counter, counter_result, READ_UNCORE);
                        if (index < NUM_COUNTERS_UNCORE_NEHALEM-1)
                        {
                            NEH_CHECK_UNCORE_OVERFLOW(index - NUM_COUNTERS_CORE_NEHALEM);
                        }
                        else
                        {
                            NEH_CHECK_UNCORE_OVERFLOW(32);
                        }
                    }
                    break;
                default:
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
        }
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, pmc_flags, UNFREEZE_PMC_AND_FIXED);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, pmc_flags));
    }
    if (haveLock && (eventSet->regTypeMask & ~(0xF)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, uncore_flags, UNFREEZE_UNCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNCORE_PERF_GLOBAL_CTRL, uncore_flags));
    }
    return 0;
}

int perfmon_finalizeCountersThread_nehalem(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int haveTileLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t ovf_values_core = (1ULL<<63)|(1ULL<<62);

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }
    if (tile_lock[affinity_thread2tile_lookup[cpu_id]] == cpu_id)
    {
        haveTileLock = 1;
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterType type = eventSet->events[i].type;
            if (type == NOTYPE)
            {
                continue;
            }
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = counter_map[index].configRegister;
            PciDeviceIndex dev = counter_map[index].device;
            switch (type)
            {
                case PMC:
                    ovf_values_core |= (1ULL<<(index-cpuid_info.perf_num_fixed_ctr));
                    if ((haveTileLock) && (eventSet->events[i].event.eventId == 0xB7))
                    {
                        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, 0x0ULL, CLEAR_OFFCORE_RESP0);
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, 0x0ULL));
                    }
                    else if ((haveTileLock) && (eventSet->events[i].event.eventId == 0xBB) &&
                             ((cpuid_info.model == NEHALEM_WESTMERE) || (cpuid_info.model == NEHALEM_WESTMERE_M)))
                    {
                        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP1, 0x0ULL, CLEAR_OFFCORE_RESP1);
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP1, 0x0ULL));
                    }
                    else if ((haveTileLock) && (eventSet->events[i].event.eventId == 0x35) &&
                             ((cpuid_info.model == NEHALEM_WESTMERE) || (cpuid_info.model == NEHALEM_WESTMERE_M)))
                    {
                        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_ADDR_OPCODE_MATCH, 0x0ULL, CLEAR_UNCORE_MATCH);
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNCORE_ADDR_OPCODE_MATCH, 0x0ULL));
                    }
                    break;
                case FIXED:
                    ovf_values_core |= (1ULL<<(index+32));
                    break;
                default:
                    break;
            }
            if ((reg) && (((type == PMC)||(type == FIXED))||((type >= UNCORE) && (haveLock))))
            {
                VERBOSEPRINTPCIREG(cpu_id, dev, reg, 0x0ULL, CLEAR_CTL);
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, 0x0ULL));
            }
            eventSet->events[i].threadCounter[thread_id].init = FALSE;
        }
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, ovf_values_core, CLEAR_OVF_CTRL);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, ovf_values_core));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, CLEAR_PMC_AND_FIXED_CTRL);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    if (haveLock && eventSet->regTypeMask & ~(0xFULL))
    {
        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_PERF_GLOBAL_OVF_CTRL, 0x0ULL, CLEAR_UNCORE_OVF);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNCORE_PERF_GLOBAL_OVF_CTRL, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL, CLEAR_UNCORE_CTRL);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    return 0;
}

