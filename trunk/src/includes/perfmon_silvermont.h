/*
 * =======================================================================================
 *
 *      Filename:  perfmon_silvermont.h
 *
 *      Description:  Header file of perfmon module for Intel Atom Silvermont
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
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
 
#include <perfmon_silvermont_events.h>
#include <perfmon_silvermont_counters.h>

static int perfmon_numCountersSilvermont = NUM_COUNTERS_SILVERMONT;
static int perfmon_numCoreCountersSilvermont = NUM_COUNTERS_SILVERMONT;
static int perfmon_numArchEventsSilvermont = NUM_ARCH_EVENTS_SILVERMONT;


int perfmon_init_silvermont(int cpu_id)
{
    GET_READFD(cpu_id);
    if ( cpuid_info.model == ATOM_SILVERMONT )
    {
        lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    }
    CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PEBS_ENABLE, 0x0ULL));
    return 0;
}

uint32_t svm_fixed_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint32_t flags = (0x2 << (4*index));
    if (event->numberOfOptions > 0)
    {
        for(int i=0;i<event->numberOfOptions;i++)
        {
            switch(event->options[i].type)
            {
                case EVENT_OPTION_ANYTHREAD:
                    flags |= (0x4 << (4*index));
                    break;
                case EVENT_OPTION_COUNT_KERNEL:
                    flags |= (0x1 << (4*index));
                    break;
            }
        }
    }
    return flags;
}

int svm_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags;
    GET_READFD(cpu_id);

    flags |= (1<<16)|(1<<22);
    flags |= (event->umask<<8) + event->eventId;
    /* For event id 0xB7 the cmask must be written in an extra register */
    if ((event->cmask) && (event->eventId != 0xB7))
    {
        flags |= (event->cmask << 24);
    }

    if (event->numberOfOptions > 0)
    {
        for(int i=0;i<event->numberOfOptions;i++)
        {
            switch(event->options[i].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                case EVENT_OPTION_ANYTHREAD:
                    flags |= (1ULL<<21);
                    break;
                case EVENT_OPTION_INVERT:
                    flags |= (1ULL<<23);
                    break;
                case EVENT_OPTION_COUNT_KERNEL:
                    flags |= (1ULL<<17);
                    break;
            }
        }
    }

    // Offcore event with additional configuration register
    // cfgBits contain offset of "request type" bit
    // cmask contain offset of "response type" bit
    if (event->eventId == 0xB7)
    {
        uint32_t reg = 0x0;
        if (event->umask == 0x01)
        {
            reg = MSR_OFFCORE_RESP0;
        }
        else if (event->umask == 0x02)
        {
            reg = MSR_OFFCORE_RESP1;
        }
        if (reg)
        {
            CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, reg, &flags));
            flags = (1<<event->cfgBits)|(1<<event->cmask);
            CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, reg , flags));
        }
    }
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_PMC)
    CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, counter_map[index].configRegister, flags));
    return 0;
}

int perfmon_setupCountersThread_silvermont(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags;
    uint32_t uflags;
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
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x0ULL));
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        flags = 0x0ULL;
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = counter_map[index].configRegister;
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        switch (silvermont_counter_map[index].type)
        {
            case PMC:
                svm_pmc_setup(cpu_id, index, event);
                break;

            case FIXED:
                fixed_flags |= svm_fixed_setup(cpu_id, index, event);
                break;

            case POWER:
                break;

            default:
                /* should never be reached */
                break;
        }
    }
    if (fixed_flags > 0x0)
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_FIXED_CTR_CTRL, LLU_CAST fixed_flags, SETUP_FIXED)
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_FIXED_CTR_CTRL, fixed_flags));
    }
    return 0;
}




int perfmon_startCountersThread_silvermont(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t tmp;
    uint64_t flags = 0x0ULL;
    uint32_t uflags = 0x10000UL; /* Clear freeze bit */
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
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;
            switch (counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, counter1, 0x0ULL));
                    flags |= (1<<(index-OFFSET_PMC));  /* enable counter */
                    break;

                case FIXED:
                    CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, counter1, 0x0ULL));
                    flags |= (1ULL<<(index+32));  /* enable fixed counter */
                    break;

                case POWER:
                    if(haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&tmp));
                        eventSet->events[i].threadCounter[thread_id].startData = tmp;
                    }

                    break;

                default:
                    /* should never be reached */
                    break;
            }
        }
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, UNFREEZE_PMC_OR_FIXED)
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_CTRL, flags));
    }

    return 0;
}


int perfmon_stopCountersThread_silvermont(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags;
    uint32_t uflags = 0x10100UL; /* Set freeze bit */
    uint64_t counter_result = 0x0ULL;
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;
    GET_READFD(cpu_id);

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_OR_FIXED)
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE) 
        {
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;
            switch (counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, counter1, &counter_result));
                    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                    {
                        uint64_t ovf_values = 0x0ULL;
                        CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, MSR_MIC_PERF_GLOBAL_STATUS, &ovf_values));
                        if (ovf_values & (1ULL<<(index - cpuid_info.perf_num_fixed_ctr)))
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                            CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_MIC_PERF_GLOBAL_OVF_CTRL,
                                                    (1ULL<<(index - cpuid_info.perf_num_fixed_ctr))));
                        }
                    }
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, counter1, &counter_result));
                    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                    {
                        uint64_t ovf_values = 0x0ULL;
                        CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, MSR_MIC_PERF_GLOBAL_STATUS, &ovf_values));
                        if (ovf_values & (1ULL<<(index + 32)))
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                            CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_MIC_PERF_GLOBAL_OVF_CTRL, (1ULL<<(index + 32))));
                        }
                    }
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case POWER:
                    if(haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&counter_result));
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                        eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    }
                    break;

                case THERMAL:
                        CHECK_TEMP_READ_ERROR(thermal_read(cpu_id, (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                default:
                    /* should never be reached */
                    break;
            }
        }
    }
    return 0;
}

int perfmon_readCountersThread_silvermont(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t counter_result = 0x0ULL;
    uint64_t pmc_flags = 0x0ULL;
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;
    GET_READFD(cpu_id);

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, MSR_PERF_GLOBAL_CTRL, &pmc_flags));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_OR_FIXED)
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;
            switch (counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, counter1, &counter_result));
                    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                    {
                        uint64_t ovf_values = 0x0ULL;
                        CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, MSR_MIC_PERF_GLOBAL_STATUS, &ovf_values));
                        if (ovf_values & (1ULL<<(index - cpuid_info.perf_num_fixed_ctr)))
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                            CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_MIC_PERF_GLOBAL_OVF_CTRL,
                                                    (1ULL<<(index - cpuid_info.perf_num_fixed_ctr))));
                        }
                    }
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, counter1, &counter_result));
                    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                    {
                        uint64_t ovf_values = 0x0ULL;
                        CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, MSR_MIC_PERF_GLOBAL_STATUS, &ovf_values));
                        if (ovf_values & (1ULL<<(index + 32)))
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                            CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_MIC_PERF_GLOBAL_OVF_CTRL, (1ULL<<(index + 32))));
                        }
                    }
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case POWER:
                    if(haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&counter_result));
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                        eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    }
                    break;

                case THERMAL:
                        CHECK_TEMP_READ_ERROR(thermal_read(cpu_id, (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                default:
                    /* should never be reached */
                    break;
            }
        }
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_CTRL, pmc_flags));
    }
    return 0;
}


int perfmon_finalizeCountersThread_silvermont(int thread_id, PerfmonEventSet* eventSet)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    GET_READFD(cpu_id);
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = counter_map[index].configRegister;
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, reg, 0x0ULL));
        if (event->eventId == 0xB7)
        {
            if (event->umask == 0x1)
            {
                CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_OFFCORE_RESP0, 0x0ULL));
            }
            else if (event->umask == 0x2)
            {
                CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_OFFCORE_RESP1, 0x0ULL));
            }
        }
    }
    CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_MIC_PERF_GLOBAL_OVF_CTRL, ~(0x0ULL)));
    return 0;
}
