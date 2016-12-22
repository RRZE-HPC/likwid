/*
 * =======================================================================================
 *
 *      Filename:  perfmon_goldmont.h
 *
 *      Description:  Header File of perfmon module for Intel Goldmont.
 *
 *      Version:   4.2
 *      Released:  22.12.2016
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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

#include <perfmon_goldmont_events.h>
#include <perfmon_goldmont_counters.h>
#include <error.h>
#include <affinity.h>
#include <limits.h>
#include <topology.h>
#include <access.h>

static int perfmon_numCountersGoldmont = NUM_COUNTERS_GOLDMONT;
static int perfmon_numCoreCountersGoldmont = NUM_COUNTERS_CORE_GOLDMONT;
static int perfmon_numArchEventsGoldmont = NUM_ARCH_EVENTS_GOLDMONT;

int perfmon_init_goldmont(int cpu_id)
{
    lock_acquire((int*) &tile_lock[affinity_thread2tile_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL));
    return 0;
}

uint32_t glm_fixed_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t flags = (1ULL<<(1+(index*4)));
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

int glm_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    uint64_t offcore_flags = 0x0ULL;
    uint64_t latency_flags = 0x0ULL;

    flags = (1ULL<<22)|(1ULL<<16);
    /* Intel with standard 8 bit event mask: [7:0] */
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
                    flags |= (event->options[j].value & 0xFFULL) << 24;
                    break;
                case EVENT_OPTION_IN_TRANS:
                    flags |= (1ULL<<32);
                    break;
                case EVENT_OPTION_IN_TRANS_ABORT:
                    flags |= (1ULL<<33);
                    break;
                case EVENT_OPTION_MATCH0:
                    offcore_flags |= (event->options[j].value & 0x8FFFULL);
                    break;
                case EVENT_OPTION_MATCH1:
                    offcore_flags |= (event->options[j].value<< 16);
                    break;
                default:
                    break;
            }
        }
    }

    if (event->eventId == 0xB7)
    {
        if ((event->cfgBits != 0xFF) && (event->cmask != 0xFF))
        {
            offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);
        }
        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, offcore_flags));
    }
    else if (event->eventId == 0xBB)
    {
        if ((event->cfgBits != 0xFF) && (event->cmask != 0xFF))
        {
            offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);
        }
        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP1, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP1, offcore_flags));
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_PMC)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister , flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int perfmon_setupCounterThread_goldmont(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags;
    uint64_t fixed_flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, 0xC00000070000000F));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL));
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!TESTTYPE(eventSet, type))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = counter_map[index].configRegister;
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        flags = 0x0ULL;
        switch (type)
        {
            case PMC:
                glm_pmc_setup(cpu_id, index, event);
                break;

            case FIXED:
                fixed_flags |= glm_fixed_setup(cpu_id, index, event);
                break;

            case POWER:
                break;
                break;
            default:
                break;
        }
    }
    if ((fixed_flags > 0x0ULL))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_FIXED_CTR_CTRL, LLU_CAST fixed_flags, SETUP_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_FIXED_CTR_CTRL, fixed_flags));
    }
    return 0;
}

int perfmon_startCountersThread_goldmont(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t uflags = 0x0ULL;
    uint64_t tmp = 0x0ULL;
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
            if (!TESTTYPE(eventSet, type))
            {
                continue;
            }
            tmp = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint64_t counter1 = counter_map[index].counterRegister;

            PciDeviceIndex dev = counter_map[index].device;
            eventSet->events[i].threadCounter[thread_id].startData = 0;
            eventSet->events[i].threadCounter[thread_id].counterData = 0;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    flags |= (1ULL<<(index-cpuid_info.perf_num_fixed_ctr));  /* enable counter */
                    break;

                case FIXED:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    flags |= (1ULL<<(index+32));  /* enable fixed counter */
                    break;

                case POWER:
                    if (haveLock)
                    {
                        tmp = 0x0ULL;
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1,(uint32_t*)&tmp));
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST tmp, START_POWER)
                        eventSet->events[i].threadCounter[thread_id].startData = field64(tmp, 0, box_map[type].regWidth);
                    }
                    break;
                default:
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = eventSet->events[i].threadCounter[thread_id].startData;
        }
    }

    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, LLU_CAST (1ULL<<63)|(1ULL<<62)|flags, CLEAR_PMC_AND_FIXED_OVERFLOW)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<63)|(1ULL<<62)|flags));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, UNFREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, flags));
    }

    return 0;
}


#define GLM_CHECK_CORE_OVERFLOW(offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t ovf_values = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_STATUS, &ovf_values)); \
        if (ovf_values & (1ULL<<offset)) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
        } \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<offset))); \
    }

#define GLM_CHECK_LOCAL_OVERFLOW \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t ovf_values = 0x0ULL; \
        uint64_t offset = getCounterTypeOffset(eventSet->events[i].index); \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, box_map[eventSet->events[i].type].statusRegister, &ovf_values)); \
        if (ovf_values & (1ULL<<offset)) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[eventSet->events[i].type].statusRegister, (1ULL<<offset))); \
        } \
    }

int perfmon_stopCountersThread_goldmont(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterType type = eventSet->events[i].type;
            if (!TESTTYPE(eventSet, type))
            {
                continue;
            }
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            PerfmonEvent *event = &(eventSet->events[i].event);
            PciDeviceIndex dev = counter_map[index].device;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t* current = &(eventSet->events[i].threadCounter[thread_id].counterData);
            int* overflows = &(eventSet->events[i].threadCounter[thread_id].overflows);
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    GLM_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
                    *current = field64(counter_result, 0, box_map[type].regWidth);
                    break;

                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    GLM_CHECK_CORE_OVERFLOW(index+32);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED)
                    *current = field64(counter_result, 0, box_map[type].regWidth);
                    break;

                case POWER:
                    if (haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&counter_result));
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, STOP_POWER)
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                        *current = field64(counter_result, 0, box_map[type].regWidth);
                    }
                    break;

                case THERMAL:
                    CHECK_TEMP_READ_ERROR(thermal_read(cpu_id,(uint32_t*)&counter_result));
                    *current = field64(counter_result, 0, box_map[type].regWidth);
                    break;

                default:
                    break;
            }
        }
    }

    return 0;
}


int perfmon_readCountersThread_goldmont(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags = 0x0ULL;
    uint64_t uflags = 0x0ULL;
    int haveLock = 0;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (MEASURE_CORE(eventSet))
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, &flags));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, SAFE_PMC_FLAGS)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, RESET_PMC_FLAGS)
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            counter_result= 0x0ULL;
            RegisterType type = eventSet->events[i].type;
            if (!TESTTYPE(eventSet, type))
            {
                continue;
            }
            RegisterIndex index = eventSet->events[i].index;
            PerfmonEvent *event = &(eventSet->events[i].event);
            PciDeviceIndex dev = counter_map[index].device;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t* current = &(eventSet->events[i].threadCounter[thread_id].counterData);
            int* overflows = &(eventSet->events[i].threadCounter[thread_id].overflows);
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    GLM_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
                    eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
                    break;

                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    GLM_CHECK_CORE_OVERFLOW(index+32);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED)
                    eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
                    break;

                case POWER:
                    if (haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&counter_result));
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, STOP_POWER)
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                        eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
                    }
                    break;


                default:
                    break;
            }
        }
    }

    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, RESTORE_PMC_FLAGS)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, flags));
    }

    return 0;
}

int perfmon_finalizeCountersThread_goldmont(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int haveTileLock = 0;
    int clearPBS = 0;
    uint64_t ovf_values_core = (1ULL<<63)|(1ULL<<62);
    int cpu_id = groupSet->threads[thread_id].processorId;

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
        RegisterType type = eventSet->events[i].type;
        if (!TESTTYPE(eventSet, type))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        PciDeviceIndex dev = counter_map[index].device;
        uint64_t reg = counter_map[index].configRegister;
        switch (type)
        {
            case PMC:
                ovf_values_core |= (1ULL<<(index-cpuid_info.perf_num_fixed_ctr));
                if ((haveTileLock) && (eventSet->events[i].event.eventId == 0xB7))
                {
                    VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, 0x0ULL, CLEAR_OFFCORE_RESP0);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, 0x0ULL));
                }
                else if ((haveTileLock) && (eventSet->events[i].event.eventId == 0xBB))
                {
                    VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP1, 0x0ULL, CLEAR_OFFCORE_RESP1);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP1, 0x0ULL));
                }
                break;
            case FIXED:
                ovf_values_core |= (1ULL<<(index+32));
                break;
            default:
                break;
        }
        if ((reg) && (((type == PMC)||(type == FIXED))||((type >= UBOXFIX) && (haveLock))))
        {
            VERBOSEPRINTPCIREG(cpu_id, dev, reg, 0x0ULL, CLEAR_CTL);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, 0x0ULL));
            VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL, CLEAR_CTR);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL));
            if (counter_map[index].counterRegister2 != 0x0)
            {
                VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].counterRegister2, 0x0ULL, CLEAR_CTR);
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister2, 0x0ULL));
            }
        }
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
    }

    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, LLU_CAST ovf_values_core, CLEAR_GLOBAL_OVF)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, ovf_values_core));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST 0x0ULL, CLEAR_GLOBAL_CTRL)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    return 0;
}
