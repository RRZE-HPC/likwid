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
    return 0;
}

int phi_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x0ULL;

    flags |= (1ULL<<16)|(1ULL<<22);
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
                case EVENT_OPTION_ANYTHREAD:
                    flags |= (1ULL<<21);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= (event->options[j].value & 0xFFULL) <<24;
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

int perfmon_setupCounterThread_phi(int thread_id, PerfmonEventSet* eventSet)
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
            phi_pmc_setup(cpu_id, index, event);
            eventSet->events[i].threadCounter[thread_id].init = TRUE;
        }
    }
    return 0;
}

int perfmon_startCountersThread_phi(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC_PERF_GLOBAL_CTRL, 0x0ULL));

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
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].counterRegister , 0x0ULL));
            flags |= (1ULL<<(index));  /* enable counter */
        }
    }

    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC_PERF_GLOBAL_CTRL, flags));
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC_SPFLT_CONTROL, flags|(1ULL<<63)));
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC_PERF_GLOBAL_OVF_CTRL, flags));
    return 0;
}

int perfmon_stopCountersThread_phi(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t counter_result = 0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC_SPFLT_CONTROL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC_PERF_GLOBAL_CTRL, 0x0ULL));

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
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, phi_counter_map[index].counterRegister, &counter_result));
            if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
            {
                uint64_t ovf_values = 0x0ULL;
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_MIC_PERF_GLOBAL_STATUS, &ovf_values));
                if (ovf_values & (1ULL<<index))
                {
                    eventSet->events[i].threadCounter[thread_id].overflows++;
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC_PERF_GLOBAL_OVF_CTRL, (1ULL<<index)));
                }
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
        }
    }
    return 0;
}

int perfmon_readCountersThread_phi(int thread_id, PerfmonEventSet* eventSet)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter_result = 0x0ULL;
    uint64_t core_flags = 0x0ULL;

    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_MIC_PERF_GLOBAL_CTRL, &core_flags));
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC_PERF_GLOBAL_CTRL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC_SPFLT_CONTROL, 0x0ULL));

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
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter_map[i].counterRegister, &counter_result));
            if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
            {
                uint64_t ovf_values = 0x0ULL;
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_MIC_PERF_GLOBAL_STATUS, &ovf_values));
                if (ovf_values & (1ULL<<index))
                {
                    eventSet->events[i].threadCounter[thread_id].overflows++;
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC_PERF_GLOBAL_OVF_CTRL, (1ULL<<index)));
                }
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
        }
    }

    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC_SPFLT_CONTROL, core_flags|(1ULL<<63)));
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC_PERF_GLOBAL_CTRL, core_flags));
    return 0;
}


int perfmon_finalizeCountersThread_phi(int thread_id, PerfmonEventSet* eventSet)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t ovf_values_core = 0x0ULL;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        ovf_values_core |= (1ULL<<(index));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[i].configRegister, 0x0ULL));
    }
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC_PERF_GLOBAL_CTRL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC_SPFLT_CONTROL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC_PERF_GLOBAL_OVF_CTRL, ovf_values_core));
    return 0;
}
