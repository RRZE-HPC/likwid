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

#include <perfmon_nehalem_events.h>
#include <perfmon_nehalem_counters.h>
#include <error.h>
#include <affinity.h>

#define GET_READFD(cpu_id) \
    int read_fd; \
    if (accessClient_mode != DAEMON_AM_DIRECT) \
    { \
        read_fd = socket_fd; \
        if (socket_fd == -1 || thread_sockets[cpu_id] != -1) \
        { \
            read_fd = thread_sockets[cpu_id]; \
        } \
        if (read_fd == -1) \
        { \
            return -ENOENT; \
        } \
    }

static int perfmon_numCountersNehalem = NUM_COUNTERS_NEHALEM;
static int perfmon_numArchEventsNehalem = NUM_ARCH_EVENTS_NEHALEM;


int perfmon_init_nehalem(int cpu_id)
{
    GET_READFD(cpu_id);
    lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &tile_lock[affinity_core2tile_lookup[cpu_id]], cpu_id);
    CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PEBS_ENABLE, 0x0ULL));
    return 0;
}

uint32_t neh_fixed_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t flags = (0x2 << (4*index));
    for(j=0;j<event->numberOfOptions;j++)
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
    GET_READFD(cpu_id);

    if ((tile_lock[affinity_core2tile_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    flags = (1ULL<<22)|(1ULL<<16);
    flags |= (event->umask<<8) + event->eventId;
    if (event->cfgBits != 0) /* set custom cfg and cmask */
    {
        flags |= ((event->cmask<<8) + event->cfgBits)<<16;
    }
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
                case EVENT_OPTION_ANYTHREAD:
                    flags |= (1ULL<<21);
                    break;
                default:
                    break;
            }
        }
    }
    // Offcore event with additional configuration register
    // cfgBits contain offset of "request type" bit
    // cmask contain offset of "response type" bit
    if (haveLock && event->eventId == 0xB7)
    {
        uint64_t offcore_flags = 0x0ULL;
        offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);
        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE);
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_OFFCORE_RESP0, offcore_flags));
    }
    if ((haveLock) && (event->eventId == 0xBB) &&
        ((cpuid_info.model == NEHALEM_WESTMERE) || (cpuid_info.model == NEHALEM_WESTMERE_M)))
    {
        uint64_t offcore_flags = 0x0ULL;
        offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);
        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP1, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE);
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_OFFCORE_RESP1, offcore_flags));
    }
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_PMC);
    CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, counter_map[index].configRegister, flags));
    return 0;
}

int neh_uncore_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    uint64_t mask_flags = 0x0ULL;
    GET_READFD(cpu_id);

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id))
    {
        return 0;
    }

    flags = (1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;
    if (event->cfgBits != 0 && event->eventId != 0x35) /* set custom cfg and cmask */
    {
        flags |= ((event->cmask<<8) + event->cfgBits)<<16;
    }
    if (event->cfgBits != 0 && event->eventId == 0x35) /* set custom cfg and cmask */
    {
        mask_flags |= ((uint64_t)event->cfgBits)<<60;
        mask_flags |= ((uint64_t)event->cmask)<<40;
    }
    if (event->numberOfOptions > 0)
    {
        for(int j=0;j<event->numberOfOptions;j++)
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
                case EVENT_OPTION_MATCH0:
                    mask_flags |= extractBitField(event->options[j].value,40,0)<<2;
                    break;
                case EVENT_OPTION_OPCODE:
                    mask_flags |= ((uint64_t)extractBitField(event->options[j].value,8,0))<<40;
                    break;
                default:
                    break;
            }
        }
    }
    if ((mask_flags != 0x0ULL) && (event->eventId == 0x35) &&
        ((cpuid_info.model == NEHALEM_WESTMERE) || (cpuid_info.model == NEHALEM_WESTMERE)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_ADDR_OPCODE_MATCH, LLU_CAST mask_flags, SETUP_UNCORE_MATCH);
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_UNCORE_ADDR_OPCODE_MATCH, mask_flags));
    }
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_UNCORE);
    CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, counter_map[index].configRegister, flags));
    return 0;
}

int perfmon_setupCounterThread_nehalem(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags;
    uint64_t fixed_flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    GET_READFD(cpu_id);

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)))
    {
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x0ULL));
    }
    if (haveLock && (eventSet->regTypeMask & ~(0xF)))
    {
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        uint64_t reg = nehalem_counter_map[index].configRegister;

        switch (counter_map[index].type)
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
                        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_UNCORE_FIXED_CTR_CTRL, 0x1ULL));
                    }
                }
                break;
        }
    }
    if (fixed_flags != 0x0ULL)
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_FIXED_CTR_CTRL, LLU_CAST fixed_flags, SETUP_FIXED);
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_FIXED_CTR_CTRL, fixed_flags));
    }
    return 0;
}

int perfmon_startCountersThread_nehalem(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t uflags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    GET_READFD(cpu_id);


    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            uint64_t counter = counter_map[index].counterRegister;
            switch(counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, counter, 0x0ULL));
                    flags |= (1ULL<<(index - cpuid_info.perf_num_fixed_ctr));  /* enable counter */
                    break;
                case FIXED:
                    CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, counter, 0x0ULL));
                    flags |= (1ULL<<(index+32));  /* enable fixed counter */
                    break;
                case UNCORE:
                    if(haveLock)
                    {
                        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, counter, 0x0ULL));
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
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, uflags));
    }

    if ((flags != 0x0ULL) && (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED))))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, UNFREEZE_PMC_AND_FIXED);
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_CTRL, flags));
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x30000000FULL));
    }
    return 0;
}

#define NEH_CHECK_OVERFLOW(offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, MSR_PERF_GLOBAL_STATUS, &tmp)); \
        if (tmp & (1ULL<<offset)) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
            CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, (tmp & (1ULL<<offset)))); \
        } \
    }

#define NEH_CHECK_UNCORE_OVERFLOW(offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, MSR_UNCORE_PERF_GLOBAL_STATUS, &tmp)); \
        if (tmp & (1ULL<<offset)) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
            CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_UNCORE_PERF_GLOBAL_OVF_CTRL, (tmp & (1ULL<<offset)))); \
        } \
    }

int perfmon_stopCountersThread_nehalem(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags;
    int haveLock = 0;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    GET_READFD(cpu_id);

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED);
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    if (haveLock && (eventSet->regTypeMask & ~(0xF)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_UNCORE);
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter = counter_map[index].counterRegister;
            switch (counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, counter, &counter_result));
                    VERBOSEPRINTREG(cpu_id, counter, counter_result, READ_PMC);
                    NEH_CHECK_OVERFLOW(index - cpuid_info.perf_num_fixed_ctr);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, counter, &counter_result));
                    VERBOSEPRINTREG(cpu_id, counter, counter_result, READ_FIXED);
                    NEH_CHECK_OVERFLOW(index + 32);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case UNCORE:
                    if(haveLock)
                    {
                        CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, counter, &counter_result));
                        VERBOSEPRINTREG(cpu_id, counter, counter_result, READ_UNCORE);
                        if (index < NUM_COUNTERS_UNCORE_NEHALEM-1)
                        {
                            NEH_CHECK_UNCORE_OVERFLOW(index - NUM_COUNTERS_CORE_NEHALEM);
                        }
                        else
                        {
                            NEH_CHECK_UNCORE_OVERFLOW(32);
                        }
                        eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    }
                    break;
            }
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
    GET_READFD(cpu_id);

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, MSR_PERF_GLOBAL_CTRL, &pmc_flags));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED);
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    if (haveLock && (eventSet->regTypeMask & ~(0xF)))
    {
        CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, &uncore_flags));
        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_UNCORE);
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter = counter_map[index].counterRegister;
            switch (counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, counter, &counter_result));
                    VERBOSEPRINTREG(cpu_id, counter, counter_result, READ_PMC);
                    NEH_CHECK_OVERFLOW(index - cpuid_info.perf_num_fixed_ctr);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, counter, &counter_result));
                    VERBOSEPRINTREG(cpu_id, counter, counter_result, READ_FIXED);
                    NEH_CHECK_OVERFLOW(index + 32);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case UNCORE:
                    if(haveLock)
                    {
                        CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, counter, &counter_result));
                        VERBOSEPRINTREG(cpu_id, counter, counter_result, READ_UNCORE);
                        if (index < NUM_COUNTERS_UNCORE_NEHALEM-1)
                        {
                            NEH_CHECK_UNCORE_OVERFLOW(index - NUM_COUNTERS_CORE_NEHALEM);
                        }
                        else
                        {
                            NEH_CHECK_UNCORE_OVERFLOW(32);
                        }
                        eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    }
                    break;
            }
        }
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, pmc_flags, UNFREEZE_PMC_AND_FIXED);
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_CTRL, pmc_flags));
    }
    if (haveLock && (eventSet->regTypeMask & ~(0xF)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, uncore_flags, UNFREEZE_UNCORE);
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, uncore_flags));
    }
    return 0;
}

int perfmon_finalizeCountersThread_nehalem(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;
    GET_READFD(cpu_id);
    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = counter_map[index].configRegister;
            VERBOSEPRINTREG(cpu_id, reg, 0x0ULL, CLEAR_CTRL);
            CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, reg, 0x0ULL));
        }
    }

    VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, 0x0ULL, CLEAR_OFFCORE_RESP0);
    CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_OFFCORE_RESP0, 0x0ULL));
    if ((cpuid_info.model == NEHALEM_WESTMERE) || (cpuid_info.model == NEHALEM_WESTMERE_M))
    {
        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP1, 0x0ULL, CLEAR_OFFCORE_RESP1);
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_OFFCORE_RESP1, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_ADDR_OPCODE_MATCH, 0x0ULL, CLEAR_UNCORE_MATCH);
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_UNCORE_ADDR_OPCODE_MATCH, 0x0ULL));
    }

    VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, ~(0x0ULL), CLEAR_OVF_CTRL);
    CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, ~(0x0ULL)));
    VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, CLEAR_PMC_AND_FIXED_CTRL);
    CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));

    if (haveLock)
    {
        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_PERF_GLOBAL_OVF_CTRL, ~(0x0ULL), CLEAR_UNCORE_OVF);
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_UNCORE_PERF_GLOBAL_OVF_CTRL, ~(0x0ULL)));
        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL, CLEAR_UNCORE_CTRL);
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    return 0;
}

