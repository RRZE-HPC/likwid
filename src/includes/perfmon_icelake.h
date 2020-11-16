/*
 * =======================================================================================
 *
 *      Filename:  perfmon_icelake.h
 *
 *      Description:  Header File of perfmon module for Intel Icelake.
 *
 *      Version:   5.1
 *      Released:  16.11.2020
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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

#include <perfmon_icelake_counters.h>
#include <perfmon_icelake_events.h>
#include <perfmon_icelakeX_counters.h>
#include <perfmon_icelakeX_events.h>

static int perfmon_numCountersIcelake = NUM_COUNTERS_ICELAKE;
static int perfmon_numCoreCountersIcelake = NUM_COUNTERS_CORE_ICELAKE;
static int perfmon_numArchEventsIcelake = NUM_ARCH_EVENTS_ICELAKE;

static int perfmon_numCountersIcelakeX = NUM_COUNTERS_ICELAKEX;
static int perfmon_numCoreCountersIcelakeX = NUM_COUNTERS_CORE_ICELAKEX;
static int perfmon_numArchEventsIcelakeX = NUM_ARCH_EVENTS_ICELAKEX;

#define MEASURE_METRICS(eventset) ((eventset)->regTypeMask1 & (REG_TYPE_MASK(METRICS))

static int icl_did_cbox_check = 0;
int icl_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event);
int icx_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event);
int (*icelake_cbox_setup)(int, RegisterIndex, PerfmonEvent *) = NULL;

int icl_cbox_nosetup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    cpu_id++;
    index++;
    event++;
    return 0;
}


int perfmon_init_icelake(int cpu_id)
{
    int ret = 0;
    lock_acquire((int*) &tile_lock[affinity_thread2core_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &socket_lock[affinity_thread2socket_lookup[cpu_id]], cpu_id);
    ret = HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL);
    if (ret != 0)
    {
        ERROR_PRINT(Cannot zero MSR_PEBS_ENABLE (0x%X), MSR_PEBS_ENABLE);
    }
    ret = HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_FRONTEND, 0x0ULL);
    if (ret != 0)
    {
        ERROR_PRINT(Cannot zero MSR_PEBS_FRONTEND (0x%X), MSR_PEBS_FRONTEND);
    }
    if (!icl_did_cbox_check)
    {
        switch(cpuid_info.model)
        {
            case ICELAKE1:
            case ICELAKE2:
                icelake_cbox_setup = icl_cbox_setup;
                break;
            case ICELAKEX1:
            case ICELAKEX2:
            default:
                icelake_cbox_setup = icl_cbox_nosetup;
                break;
        }
        icl_did_cbox_check = 1;
    }
    return 0;
}

uint32_t icl_fixed_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t flags = (1ULL<<(1+(index*4)));
    cpu_id++;
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

int icl_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
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
        (event->eventId != 0xBB) &&
        (event->eventId != 0xCD))
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

    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_PMC)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister , flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int icl_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    flags = (1ULL<<22)|(1ULL<<20);
    flags |= (event->umask<<8) + event->eventId;
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
                case EVENT_OPTION_THRESHOLD:
                    flags |= (event->options[j].value & 0x1FULL) << 24;
                    break;
                default:
                    break;
            }
        }
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_CBOX);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int icl_uboxfix_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x0ULL;
    event++;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    flags = (1ULL<<22)|(1ULL<<20);
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_UBOXFIX);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int icl_ubox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags = (1ULL<<22)|(1ULL<<20);
    flags |= (event->umask<<8) + event->eventId;
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
                case EVENT_OPTION_THRESHOLD:
                    flags |= (event->options[j].value & 0x1FULL) << 24;
                    break;
                default:
                    break;
            }
        }
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_UBOX);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int perfmon_setupCounterThread_icelake(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t fixed_flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, 0xC00000070000000F));
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
        switch (type)
        {
            case FIXED:
                fixed_flags |= icl_fixed_setup(cpu_id, index, event);
                break;
            case PMC:
                icl_pmc_setup(cpu_id, index, event);
                break;
            case POWER:
            case THERMAL:
            case VOLTAGE:
            case METRICS:
                break;
            case MBOX0:
                break;
            case UBOXFIX:
                if (haveLock)
                {
                    icl_uboxfix_setup(cpu_id, index, event);
                }
                break;
            case UBOX:
                if (haveLock)
                {
                    icl_ubox_setup(cpu_id, index, event);
                }
                break;
            case CBOX0:
            case CBOX1:
            case CBOX2:
            case CBOX3:
            case CBOX4:
            case CBOX5:
            case CBOX6:
            case CBOX7:
                if (haveLock)
                {
                    icelake_cbox_setup(cpu_id, index, event);
                }
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


int perfmon_startCountersThread_icelake(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t uflags = 0x0ULL;
    uint64_t tmp = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
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
                case FIXED:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    flags |= (1ULL<<(index+32));  /* enable fixed counter */
                    break;
                case PMC:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    flags |= (1ULL<<(index-cpuid_info.perf_num_fixed_ctr));  /* enable counter */
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
                case METRICS:
                    flags |= (1ULL << 48);
                    break;
                case THERMAL:
                case VOLTAGE:
                    break;
                case MBOX0:
                case MBOX0TMP:
                    if (haveLock && (cpuid_info.model == ICELAKE1 || cpuid_info.model == ICELAKE2))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                        eventSet->events[i].threadCounter[thread_id].startData = field64(tmp, 0, box_map[type].regWidth);
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST tmp, READ_MBOX)
                    }
                    break;


                case UBOXFIX:
                    if (haveLock && ((cpuid_info.model == ICELAKE1) || (cpuid_info.model == ICELAKE2)))
                    {
                        VERBOSEPRINTREG(cpu_id, counter1, 0x0ULL, CLEAR_UBOXFIX)
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    }
                    break;
                case UBOX:
                    if (haveLock && ((cpuid_info.model == ICELAKE1) || (cpuid_info.model == ICELAKE2)))
                    {
                        VERBOSEPRINTREG(cpu_id, counter1, 0x0ULL, CLEAR_UBOX)
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    }
                    break;
                case CBOX0:
                case CBOX1:
                case CBOX2:
                case CBOX3:
                case CBOX4:
                case CBOX5:
                case CBOX6:
                case CBOX7:
                    if (haveLock && ((cpuid_info.model == ICELAKE1) || (cpuid_info.model == ICELAKE2)))
                    {
                        VERBOSEPRINTREG(cpu_id, counter1, 0x0ULL, CLEAR_CBOX)
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    }
                    break;
                default:
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = eventSet->events[i].threadCounter[thread_id].startData;
        }
    }
    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        if ((cpuid_info.model == ICELAKE1) || (cpuid_info.model == ICELAKE2))
        {
            VERBOSEPRINTREG(cpu_id, MSR_V4_UNC_PERF_GLOBAL_CTRL, uflags|(1ULL<<29), UNFREEZE_UNCORE);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_CTRL, uflags|(1ULL<<29)));
        }
    }
    if (MEASURE_CORE(eventSet))
    {
        if (flags & (1ULL << 48))
        {
            VERBOSEPRINTREG(cpu_id, MSR_PERF_METRICS, 0x0ULL, CLEAR_METRICS)
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_METRICS, 0x0ULL));
        }
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, LLU_CAST (1ULL<<63)|(1ULL<<62)|flags, CLEAR_PMC_AND_FIXED_OVERFLOW)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<63)|(1ULL<<62)|flags));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, UNFREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, flags));
    }
    return 0;
}


#define ICL_CHECK_CORE_OVERFLOW(offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t ovf_values = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_STATUS, &ovf_values)); \
        if (ovf_values & (1ULL<<(offset))) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
        } \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<(offset)))); \
    }

#define ICX_CHECK_UNCORE_OVERFLOW(offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t ovf_values = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_STATUS, &ovf_values)); \
        if (ovf_values & (1ULL<<(offset))) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
        } \
    }



int perfmon_stopCountersThread_icelake(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int coffset = 0;
    uint64_t counter_result = 0x0ULL;
    uint64_t tmp = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        if (cpuid_info.model == ICELAKE1 || cpuid_info.model == ICELAKE2)
        {
            VERBOSEPRINTREG(cpu_id, MSR_V4_UNC_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_UNCORE);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_CTRL, 0x0ULL));
        }
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
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            PerfmonEvent *event = &(eventSet->events[i].event);
            PciDeviceIndex dev = counter_map[index].device;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t* current = &(eventSet->events[i].threadCounter[thread_id].counterData);
            int* overflows = &(eventSet->events[i].threadCounter[thread_id].overflows);
            int ovf_offset = box_map[type].ovflOffset;
            switch (type)
            {
                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    ICL_CHECK_CORE_OVERFLOW(index+32);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, STOP_FIXED)
                    break;
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    ICL_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, STOP_PMC)
                    break;
                case POWER:
                    if (haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&counter_result));
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, STOP_POWER)
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, OVERFLOW_POWER)
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                    }
                    break;

                case THERMAL:
                    CHECK_TEMP_READ_ERROR(thermal_read(cpu_id,(uint32_t*)&counter_result));
                    break;

                case VOLTAGE:
                    CHECK_TEMP_READ_ERROR(voltage_read(cpu_id, &counter_result));
                    break;

                case METRICS:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    counter_result= field64(counter_result, getCounterTypeOffset(index)*box_map[type].regWidth, box_map[type].regWidth);
                    break;
                case MBOX0:
                case MBOX0TMP:
                    if (haveLock && (cpuid_info.model == ICELAKE1 || cpuid_info.model == ICELAKE2))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, PCI_IMC_DEVICE_0_CH_0, counter1, &counter_result));
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, OVERFLOW_CLIENTMEM)
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                    }
                    break;
                case UBOXFIX:
                    if (haveLock && (cpuid_info.model == ICELAKE1 || cpuid_info.model == ICELAKE2))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        ICX_CHECK_UNCORE_OVERFLOW(box_map[type].ovflOffset);
                    }
                    break;
                case UBOX:
                    if (haveLock && (cpuid_info.model == ICELAKE1 || cpuid_info.model == ICELAKE2))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        ICX_CHECK_UNCORE_OVERFLOW(box_map[type].ovflOffset);
                    }
                    break;
                case CBOX0:
                case CBOX1:
                case CBOX2:
                case CBOX3:
                case CBOX4:
                case CBOX5:
                case CBOX6:
                    if (haveLock && (cpuid_info.model == ICELAKE1 || cpuid_info.model == ICELAKE2))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        ICX_CHECK_UNCORE_OVERFLOW(box_map[type].ovflOffset);
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_CBOX)
                    }
                    break;
                default:
                    break;
            }
            *current = field64(counter_result, 0, box_map[type].regWidth);
        }
    }
    return 0;
}

int perfmon_readCountersThread_icelake(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags = 0x0ULL;
    uint64_t uflags = 0x0ULL;
    int haveLock = 0;
    uint64_t counter_result = 0x0ULL;
    uint64_t tmp = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
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
    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        if (cpuid_info.model == ICELAKE1 && cpuid_info.model == ICELAKE2)
        {
            VERBOSEPRINTREG(cpu_id, MSR_V4_UNC_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_UNCORE);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_CTRL, 0x0ULL));
        }
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            counter_result = 0x0ULL;
            tmp = 0x0ULL;
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
            int ovf_offset = box_map[type].ovflOffset;
            switch (type)
            {
                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    ICL_CHECK_CORE_OVERFLOW(index+32);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED)
                    break;
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    ICL_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
                    break;
                case POWER:
                    if (haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&counter_result));
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_POWER)
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, OVERFLOW_POWER)
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                    }
                    break;

                case THERMAL:
                    CHECK_TEMP_READ_ERROR(thermal_read(cpu_id,(uint32_t*)&counter_result));
                    eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
                    break;

                case VOLTAGE:
                    CHECK_TEMP_READ_ERROR(voltage_read(cpu_id, &counter_result));
                    break;

                case METRICS:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    counter_result= field64(counter_result, getCounterTypeOffset(index)*box_map[type].regWidth, box_map[type].regWidth);
                    break;

                case UBOXFIX:
                    if (haveLock && (cpuid_info.model == ICELAKE1 || cpuid_info.model == ICELAKE2))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        ICX_CHECK_UNCORE_OVERFLOW(box_map[type].ovflOffset);
                    }
                    break;
                case UBOX:
                    if (haveLock && (cpuid_info.model == ICELAKE1 || cpuid_info.model == ICELAKE2))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        ICX_CHECK_UNCORE_OVERFLOW(box_map[type].ovflOffset);
                    }
                    break;
                case MBOX0:
                case MBOX0TMP:
                    if (haveLock && (cpuid_info.model == ICELAKE1 || cpuid_info.model == ICELAKE2))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_MBOX);
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, OVERFLOW_CLIENTMEM)
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                    }
                    break;

                case CBOX0:
                case CBOX1:
                case CBOX2:
                case CBOX3:
                case CBOX4:
                case CBOX5:
                case CBOX6:
                case CBOX7:
                    if (haveLock && (cpuid_info.model == ICELAKE1 || cpuid_info.model == ICELAKE2))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        ICX_CHECK_UNCORE_OVERFLOW(box_map[type].ovflOffset);
                        *current = field64(counter_result, 0, box_map[type].regWidth);
                        uflags |= (1ULL<<(type-CBOX0));
                    }
                    break;

                default:
                    break;
            }
            *current = field64(counter_result, 0, box_map[type].regWidth);
        }
    }
    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        if (cpuid_info.model == ICELAKE1 || cpuid_info.model == ICELAKE2)
        {
            VERBOSEPRINTREG(cpu_id, MSR_V4_UNC_PERF_GLOBAL_CTRL, uflags|(1ULL<<29), UNFREEZE_UNCORE);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_CTRL, uflags|(1ULL<<29)));
        }
    }
    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, RESTORE_PMC_FLAGS)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, flags));
    }
    return 0;
}

int perfmon_finalizeCountersThread_icelake(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int haveTileLock = 0;
    int clearPBS = 0;
    uint64_t ovf_values_core = (1ULL<<63)|(1ULL<<62);
    uint64_t ovf_values_uncore = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }
    if (tile_lock[affinity_thread2core_lookup[cpu_id]] == cpu_id)
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
            case FIXED:
                ovf_values_core |= (1ULL<<(index+32));
                break;
            default:
                break;
        }
        if ((reg) && (((type == PMC)||(type == FIXED))||(type == METRICS)|| ((type >= UNCORE) && (haveLock))))
        {
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, reg, &ovf_values_uncore));
            VERBOSEPRINTPCIREG(cpu_id, dev, reg, ovf_values_uncore, SHOW_CTL);
            ovf_values_uncore = 0x0ULL;
            VERBOSEPRINTPCIREG(cpu_id, dev, reg, 0x0ULL, CLEAR_CTL);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, 0x0ULL));
            if ((type >= SBOX0) && (type <= SBOX3))
            {
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, 0x0ULL));
            }
            VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL, CLEAR_CTR);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL));
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
