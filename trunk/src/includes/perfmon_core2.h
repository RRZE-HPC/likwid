/*
 * =======================================================================================
 *
 *      Filename:  perfmon_core2.h
 *
 *      Description:  Header file of perfmon module for Core 2
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

#include <perfmon_core2_events.h>
#include <perfmon_core2_counters.h>
#include <error.h>


static int perfmon_numCountersCore2 = NUM_COUNTERS_CORE2;
static int perfmon_numArchEventsCore2 = NUM_ARCH_EVENTS_CORE2;

int perfmon_init_core2(int cpu_id)
{
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL));
    return 0;
}


uint32_t core2_fixed_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
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
            default:
                break;
        }
    }
    return flags;
}

int core2_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;

    flags = (1ULL<<22)|(1ULL<<16)|(1ULL<<19);
    flags |= (event->umask<<8) + event->eventId;
    if ( event->cfgBits != 0 ) /* set custom cfg and cmask */
    {
        flags |= ((event->cmask<<8) + event->cfgBits)<<16;
    }
    if (event->numberOfOptions > 0)
    {
        for(j=0;j<event->numberOfOptions;j++)
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
                    flags |= ((event->options[j].value & 0xFFULL)<<24);
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

int perfmon_setupCounterThread_core2( int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t fixed_flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
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
                core2_pmc_setup(cpu_id, index, event);
                break;
            case FIXED:
                fixed_flags |= core2_fixed_setup(cpu_id, index, event);
                break;
            default:
                break;
        }
    }
    if (fixed_flags > 0x0ULL)
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_FIXED_CTR_CTRL, LLU_CAST fixed_flags, SETUP_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_FIXED_CTR_CTRL, fixed_flags));
    }
    return 0;
}

int perfmon_startCountersThread_core2(int thread_id, PerfmonEventSet* eventSet)
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
            uint64_t counter = counter_map[index].counterRegister;
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter, 0x0ULL));

            if (type == PMC)
            {
                flags |= (1ULL<<(index - cpuid_info.perf_num_fixed_ctr));  /* enable counter */
            }
            else if (type == FIXED)
            {
                flags |= (1ULL<<(index + 32));  /* enable fixed counter */
            }
        }
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, UNFREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, flags));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<63)|(1ULL<<62)|flags));
    }
    return 0;
}

#define CORE2_CHECK_OVERFLOW(offset) \
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

int perfmon_stopCountersThread_core2(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t counter_result;
    int cpu_id = groupSet->threads[thread_id].processorId;

    /* stop counters */
    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    /* read out counter results */
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
                    CORE2_CHECK_OVERFLOW(index - cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, READ_PMC)
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                    CORE2_CHECK_OVERFLOW(index + 32);
                    VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, READ_FIXED)
                    break;
                default:
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
        }
    }

    return 0;
}

int perfmon_readCountersThread_core2(int thread_id, PerfmonEventSet* eventSet)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter_result;
    uint64_t flags;

    if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)))
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, &flags));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, SAFE_PMC_FLAGS)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, RESET_PMC_FLAGS)
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        counter_result = 0x0ULL;
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterType type = eventSet->events[i].type;
            if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
            {
                continue;
            }
            RegisterIndex index = eventSet->events[i].index;
            uint64_t counter = counter_map[index].counterRegister;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                    CORE2_CHECK_OVERFLOW(index - cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, READ_PMC)
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                    CORE2_CHECK_OVERFLOW(index - 32);
                    VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, READ_FIXED)
                    break;
                default:
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
        }
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, RESTORE_PMC_FLAGS)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, flags));
    }

    return 0;
}


int perfmon_finalizeCountersThread_core2(int thread_id, PerfmonEventSet* eventSet)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t ovf_values_core = (1ULL<<63)|(1ULL<<62);

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        uint64_t reg = counter_map[index].configRegister;
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
        if (type == PMC)
        {
            ovf_values_core |= (1ULL<<(index - cpuid_info.perf_num_fixed_ctr));
        }
        else if (type == FIXED)
        {
            ovf_values_core |= (1ULL<<(index + 32));
        }
        if ((reg) && ((type == PMC)||(type == FIXED)))
        {
            VERBOSEPRINTREG(cpu_id, reg, LLU_CAST 0x0ULL, CLEAR_CTRL);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, 0x0ULL));
        }
    }
    VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, LLU_CAST ovf_values_core, CLEAR_GLOBAL_OVF)
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, ovf_values_core));
    VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST 0x0ULL, CLEAR_GLOBAL_CTRL)
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    return 0;
}
