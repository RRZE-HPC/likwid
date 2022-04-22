/*
 * =======================================================================================
 *
 *      Filename:  perfmon_sapphirerapids.h
 *
 *      Description:  Header File of perfmon module for Intel Sapphire Rapids.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
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


#include <perfmon_sapphirerapids_counters.h>
#include <perfmon_sapphirerapids_events.h>


static int perfmon_numCountersSapphireRapids = NUM_COUNTERS_SAPPHIRERAPIDS;
static int perfmon_numCoreCountersSapphireRapids = NUM_COUNTERS_CORE_SAPPHIRERAPIDS;
static int perfmon_numArchEventsSapphireRapids = NUM_ARCH_EVENTS_SAPPHIRERAPIDS;

#define SPR_CHECK_CORE_OVERFLOW(offset) \
    if (counter_result < data->counterData) \
    { \
        uint64_t ovf_values = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_STATUS, &ovf_values)); \
        if (ovf_values & (1ULL<<(offset))) \
        { \
            data->overflows++; \
        } \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<(offset)))); \
    }


int perfmon_init_sapphirerapids(int cpu_id)
{
    int ret = 0;
    lock_acquire((int*) &tile_lock[affinity_thread2core_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &socket_lock[affinity_thread2socket_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &die_lock[affinity_thread2die_lookup[cpu_id]], cpu_id);

    uint64_t misc_enable = 0x0;
    ret = HPMread(cpu_id, MSR_DEV, MSR_IA32_MISC_ENABLE, &misc_enable);
    if (ret == 0 && (testBit(misc_enable, 7) == 1) && (testBit(misc_enable, 12) == 0))
    {
        ret = HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL);
        if (ret != 0)
        {
            ERROR_PRINT(Cannot zero %s (0x%X), str(MSR_PEBS_ENABLE), MSR_PEBS_ENABLE);
        }
        ret = HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_FRONTEND, 0x0ULL);
        if (ret != 0)
        {
            ERROR_PRINT(Cannot zero %s (0x%X), str(MSR_PEBS_FRONTEND), MSR_PEBS_FRONTEND);
        }
    }
    return 0;
}


uint64_t spr_fixed_setup(int thread_id, RegisterIndex index, PerfmonEvent *event, PerfmonCounter* data)
{
    int j;
    uint32_t flags = (1ULL<<(1+(index*4)));
    int cpu_id = groupSet->threads[thread_id].processorId;
    cpu_id++; // to avoid warnings
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

uint64_t spr_fixed_start(int thread_id, RegisterIndex index, PerfmonEvent *event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter1 = counter_map[index].counterRegister;
    PciDeviceIndex dev = counter_map[index].device;
    VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_FIXED);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
    data->startData = 0x0ULL;
    data->counterData = 0x0ULL;
    return (1ULL<<(index+32));  /* enable fixed counter */
}

uint64_t spr_pmc_setup(int thread_id, RegisterIndex index, PerfmonEvent *event, PerfmonCounter* data)
{
    int j;
    uint64_t flags = 0x0ULL;
    uint64_t offcore_flags = 0x0ULL;
    uint64_t latency_flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

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
/*                case EVENT_OPTION_MATCH0:*/
/*                    offcore_flags |= (event->options[j].value & 0xAFB7ULL);*/
/*                    break;*/
/*                case EVENT_OPTION_MATCH1:*/
/*                    offcore_flags |= ((event->options[j].value & 0x3FFFDDULL) << 16);*/
/*                    break;*/
                default:
                    break;
            }
        }
    }

/*    if (event->eventId == 0xB7)*/
/*    {*/
/*        if ((event->cfgBits != 0xFF) && (event->cmask != 0xFF))*/
/*        {*/
/*            offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);*/
/*        }*/
/*        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE0);*/
/*        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, offcore_flags));*/
/*    }*/
/*    else if (event->eventId == 0xBB)*/
/*    {*/
/*        if ((event->cfgBits != 0xFF) && (event->cmask != 0xFF))*/
/*        {*/
/*            offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);*/
/*        }*/
/*        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP1, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE1);*/
/*        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP1, offcore_flags));*/
/*    }*/

    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_PMC)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister , flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

uint64_t spr_pmc_start(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter1 = counter_map[index].counterRegister;
    PciDeviceIndex dev = counter_map[index].device;
    VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_PMC);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
    data->startData = 0x0ULL;
    data->counterData = 0x0ULL;
    return (1ULL<<(index-cpuid_info.perf_num_fixed_ctr));  /* enable counter */
}


uint64_t spr_power_start(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        uint64_t counter1 = counter_map[index].counterRegister;
        uint64_t tmp = 0x0ULL;
        RegisterType type = counter_map[index].type;
        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1,(uint32_t*)&tmp));
        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST tmp, START_POWER)
        data->startData = field64(tmp, 0, box_map[type].regWidth);
    }
    return 0;
}

uint64_t spr_metrics_start(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return (1ULL << 48);
}

int perfmon_setupCounterThread_sapphirerapids(
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
        PerfmonCounter* data = eventSet->events[i].threadCounter;
        uint64_t reg = counter_map[index].configRegister;
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        switch (type)
        {
            case FIXED:
                fixed_flags |= spr_fixed_setup(thread_id, index, event, data);
                break;
            case PMC:
                spr_pmc_setup(thread_id, index, event, data);
                break;
            case POWER:
            case THERMAL:
            case VOLTAGE:
            case METRICS:
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


int perfmon_startCountersThread_sapphirerapids(int thread_id, PerfmonEventSet* eventSet)
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
            PerfmonEvent *event = &(eventSet->events[i].event);
            PerfmonCounter* data = eventSet->events[i].threadCounter;
            uint64_t counter1 = counter_map[index].counterRegister;

            PciDeviceIndex dev = counter_map[index].device;
            eventSet->events[i].threadCounter[thread_id].startData = 0;
            eventSet->events[i].threadCounter[thread_id].counterData = 0;
            switch (type)
            {
                case FIXED:
                    flags |= spr_fixed_start(thread_id, index, event, data);
                    break;
                case PMC:
                    flags |= spr_pmc_start(thread_id, index, event, data);
                    break;
                case POWER:
                    spr_power_start(thread_id, index, event, data);
                    break;
                case METRICS:
                    spr_metrics_start(thread_id, index, event, data);
                    break;
                case THERMAL:
                case VOLTAGE:
                    break;
                default:
                    break;
            }
            data->counterData = data->startData;
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


uint32_t spr_fixed_stop(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    uint64_t counter_result = 0x0ULL;
    uint64_t counter1 = counter_map[index].counterRegister;
    int cpu_id = groupSet->threads[thread_id].processorId;
    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
    SPR_CHECK_CORE_OVERFLOW(index+32);
    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, STOP_FIXED)
    data->counterData = counter_result;
    return 0;
}

uint32_t spr_pmc_stop(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter1 = counter_map[index].counterRegister;
    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
    SPR_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, STOP_PMC)
    data->counterData = counter_result;
    return 0;
}

uint32_t spr_power_stop(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        uint64_t counter_result = 0x0ULL;
        uint64_t counter1 = counter_map[index].counterRegister;
        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&counter_result));
        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, STOP_POWER)
        if (counter_result < data->counterData)
        {
            VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, OVERFLOW_POWER)
            data->overflows++;
        }
        data->counterData = counter_result;
    }
    return 0;
}

uint32_t spr_thermal_stop(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter_result = 0x0ULL;
    uint64_t counter1 = counter_map[index].counterRegister;
    CHECK_TEMP_READ_ERROR(thermal_read(cpu_id,(uint32_t*)&counter_result));
    VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, counter1, LLU_CAST counter_result, STOP_THERMAL);
    data->counterData = counter_result;
    return 0;
}

uint32_t spr_voltage_stop(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter_result = 0x0ULL;
    uint64_t counter1 = counter_map[index].counterRegister;
    CHECK_TEMP_READ_ERROR(voltage_read(cpu_id, &counter_result));
    VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, counter1, LLU_CAST counter_result, STOP_VOLTAGE);
    data->counterData = counter_result;
    return 0;
}

uint32_t spr_metrics_stop(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter_result = 0x0ULL;
    uint64_t counter1 = counter_map[index].counterRegister;
    RegisterType type = counter_map[index].type;
    uint64_t offset = getCounterTypeOffset(index)*box_map[type].regWidth;
    uint64_t width = box_map[type].regWidth;
    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
    counter_result= field64(counter_result, offset, width);
    VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, counter1, LLU_CAST counter_result, STOP_METRICS);
    data->counterData = counter_result;
    return 0;
}



int perfmon_stopCountersThread_sapphirerapids(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int coffset = 0;
    uint64_t counter_result = 0x0ULL;
    uint64_t tmp = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

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
            tmp = 0x0ULL;
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            PerfmonEvent *event = &(eventSet->events[i].event);
            PciDeviceIndex dev = counter_map[index].device;
            PerfmonCounter* data = eventSet->events[i].threadCounter;
            
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t* current = &(eventSet->events[i].threadCounter[thread_id].counterData);
            int* overflows = &(eventSet->events[i].threadCounter[thread_id].overflows);
            int ovf_offset = box_map[type].ovflOffset;
            switch (type)
            {
                case FIXED:
                    spr_fixed_stop(thread_id, index, event, data);
                    break;
                case PMC:
                    spr_pmc_stop(thread_id, index, event, data);
                    break;
                case POWER:
                    spr_power_stop(thread_id, index, event, data);
                    break;

                case THERMAL:
                    spr_thermal_stop(thread_id, index, event, data);
                    break;

                case VOLTAGE:
                    spr_voltage_stop(thread_id, index, event, data);
                    break;

                case METRICS:
                    spr_metrics_stop(thread_id, index, event, data);
                    break;
            }
        }
    }
    return 0;
}


uint32_t spr_fixed_read(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    uint64_t counter_result = 0x0ULL;
    uint64_t counter1 = counter_map[index].counterRegister;
    int cpu_id = groupSet->threads[thread_id].processorId;
    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
    SPR_CHECK_CORE_OVERFLOW(index+32);
    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED)
    data->counterData = counter_result;
    return 0;
}

uint32_t spr_pmc_read(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter1 = counter_map[index].counterRegister;
    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
    SPR_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
    data->counterData = counter_result;
    return 0;
}

uint32_t spr_power_read(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        uint64_t counter_result = 0x0ULL;
        uint64_t counter1 = counter_map[index].counterRegister;
        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&counter_result));
        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_POWER)
        if (counter_result < data->counterData)
        {
            VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, OVERFLOW_POWER)
            data->overflows++;
        }
        data->counterData = counter_result;
    }
    return 0;
}

uint32_t spr_thermal_read(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter_result = 0x0ULL;
    uint64_t counter1 = counter_map[index].counterRegister;
    CHECK_TEMP_READ_ERROR(thermal_read(cpu_id,(uint32_t*)&counter_result));
    VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, counter1, LLU_CAST counter_result, READ_THERMAL);
    data->counterData = counter_result;
    return 0;
}

uint32_t spr_voltage_read(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter_result = 0x0ULL;
    uint64_t counter1 = counter_map[index].counterRegister;
    CHECK_TEMP_READ_ERROR(voltage_read(cpu_id, &counter_result));
    VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, counter1, LLU_CAST counter_result, READ_VOLTAGE);
    data->counterData = counter_result;
    return 0;
}

uint32_t spr_metrics_read(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter_result = 0x0ULL;
    uint64_t counter1 = counter_map[index].counterRegister;
    RegisterType type = counter_map[index].type;
    uint64_t offset = getCounterTypeOffset(index)*box_map[type].regWidth;
    uint64_t width = box_map[type].regWidth;
    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
    counter_result= field64(counter_result, offset, width);
    VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, counter1, LLU_CAST counter_result, READ_METRICS);
    data->counterData = counter_result;
    return 0;
}


int perfmon_readCountersThread_sapphirerapids(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int coffset = 0;
    uint64_t flags = 0x0ULL;
    uint64_t counter_result = 0x0ULL;
    uint64_t tmp = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

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
            PerfmonCounter* data = eventSet->events[i].threadCounter;
            
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t* current = &(eventSet->events[i].threadCounter[thread_id].counterData);
            int* overflows = &(eventSet->events[i].threadCounter[thread_id].overflows);
            int ovf_offset = box_map[type].ovflOffset;
            switch (type)
            {
                case FIXED:
                    spr_fixed_read(thread_id, index, event, data);
                    break;
                case PMC:
                    spr_pmc_read(thread_id, index, event, data);
                    break;
                case POWER:
                    spr_power_read(thread_id, index, event, data);
                    break;

                case THERMAL:
                    spr_thermal_read(thread_id, index, event, data);
                    break;

                case VOLTAGE:
                    spr_voltage_read(thread_id, index, event, data);
                    break;

                case METRICS:
                    spr_metrics_read(thread_id, index, event, data);
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


int perfmon_finalizeCountersThread_sapphirerapids(int thread_id, PerfmonEventSet* eventSet)
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
