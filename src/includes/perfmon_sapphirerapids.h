/*
 * =======================================================================================
 *
 *      Filename:  perfmon_sapphirerapids.h
 *
 *      Description:  Header File of perfmon module for Intel Sapphire Rapids.
 *
 *      Version:   5.3
 *      Released:  10.11.2023
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2023 RRZE, University Erlangen-Nuremberg
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
    data[thread_id].startData = 0x0ULL;
    data[thread_id].counterData = 0x0ULL;
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
    data[thread_id].startData = 0x0ULL;
    data[thread_id].counterData = 0x0ULL;
    return (1ULL<<(index-cpuid_info.perf_num_fixed_ctr));  /* enable counter */
}


uint64_t spr_power_start(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    data[thread_id].startData = 0x0ULL;
    data[thread_id].counterData = 0x0ULL;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        uint64_t counter1 = counter_map[index].counterRegister;
        uint64_t tmp = 0x0ULL;
        RegisterType type = counter_map[index].type;
        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1,(uint32_t*)&tmp));
        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST tmp, START_POWER)
        data[thread_id].startData = field64(tmp, 0, box_map[type].regWidth);
    }
    return 0;
}

uint64_t spr_metrics_start(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return (1ULL << 48);
}

int spr_setup_uncore(int thread_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    PciDeviceIndex dev = counter_map[index].device;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!HPMcheck(dev, cpu_id))
    {
        return -ENODEV;
    }
    
    flags = (1ULL<<20);
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
                case EVENT_OPTION_TID:
                    if (counter_map[index].type >= CBOX0 && counter_map[index].index <= CBOX55)
                    {
                        uint64_t reg = box_map[counter_map[index].type].filterRegister1;
                        uint64_t val = event->options[j].value & 0x3FF;
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, val));
                        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_CBOX_FILTER);
                        flags |= (1ULL << 16);
                    }
                case EVENT_OPTION_MATCH0:
                    flags |= (event->options[j].value & 0xFFFFFF) << 32;
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= (event->options[j].value & 0xFFULL) << 24;
                    break;
                default:
                    break;
            }
        }
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_UNCORE);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
        HPMread(cpu_id, dev, counter_map[index].configRegister, &flags);
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, VALIDATE_UNCORE);
    }
    return 0;
}



int spr_start_uncore(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter1 = counter_map[index].counterRegister;
    PciDeviceIndex dev = counter_map[index].device;
    VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_UNCORE);
    data[thread_id].startData = 0x0ULL;
    data[thread_id].counterData = 0x0ULL;
    return HPMwrite(cpu_id, dev, counter1, 0x0ULL);
}

int spr_stop_uncore(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    uint64_t counter_result = 0x0;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter1 = counter_map[index].counterRegister;
    PciDeviceIndex dev = counter_map[index].device;
    int err = HPMread(cpu_id, dev, counter1, &counter_result);
    //counter_result = field64(counter_result, 0, box_map[type].regWidth);
    VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_UNCORE);
    data[thread_id].counterData = counter_result;
    return err;
}

int spr_read_uncore(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    uint64_t counter_result = 0x0;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter1 = counter_map[index].counterRegister;
    PciDeviceIndex dev = counter_map[index].device;
    int err = HPMread(cpu_id, dev, counter1, &counter_result);
    VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_UNCORE);
    data[thread_id].counterData = counter_result;
    return err;
}

int spr_setup_uncore_fixed(int thread_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    PciDeviceIndex dev = counter_map[index].device;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!HPMcheck(dev, cpu_id))
    {
        return -ENODEV;
    }
    
    flags = (1ULL<<20)|(1ULL<<22);
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_UNCORE_FIXED);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
        HPMread(cpu_id, dev, counter_map[index].configRegister, &flags);
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, VALIDATE_UNCORE_FIXED);
    }
    return 0;
}

int spr_start_uncore_fixed(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter1 = counter_map[index].counterRegister;
    PciDeviceIndex dev = counter_map[index].device;
    VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_UNCORE_FIXED);
    data[thread_id].startData = data[thread_id].counterData = 0x0;
    return HPMwrite(cpu_id, dev, counter1, 0x0ULL);
}

int spr_stop_uncore_fixed(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter1 = counter_map[index].counterRegister;
    PciDeviceIndex dev = counter_map[index].device;
    uint64_t flags = 0x0;
    int err = HPMread(cpu_id, dev, counter1, &flags);
    if (err != 0)
    {
        return err;
    }
    VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, STOP_UNCORE_FIXED);
    data[thread_id].counterData = flags;
    return 0;
}

int spr_read_uncore_fixed(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter1 = counter_map[index].counterRegister;
    PciDeviceIndex dev = counter_map[index].device;
    uint64_t flags = 0x0;
    int err = HPMread(cpu_id, dev, counter1, &flags);
    if (err != 0)
    {
        return err;
    }
    VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, READ_UNCORE_FIXED);
    data[thread_id].counterData = flags;
    return 0;
}

int spr_setup_uncore_freerun(int thread_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    PciDeviceIndex dev = counter_map[index].device;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!HPMcheck(dev, cpu_id))
    {
        return -ENODEV;
    }
    return 0;
}

int spr_start_uncore_freerun(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    uint64_t flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter1 = counter_map[index].counterRegister;
    PciDeviceIndex dev = counter_map[index].device;
    data->startData = data->counterData = 0x0;
    VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, START_UNCORE_FREERUN);
    int err = HPMread(cpu_id, dev, counter1, &flags);
    if (err == 0)
    {
        data[thread_id].startData = field64(flags, 0, 48);
    }
    return err;
}

int spr_stop_uncore_freerun(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter1 = counter_map[index].counterRegister;
    PciDeviceIndex dev = counter_map[index].device;
    uint64_t flags = 0x0;
    VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, STOP_UNCORE_FREERUN);
    int err = HPMread(cpu_id, dev, counter1, &flags);
    if (err == 0)
    {
        data[thread_id].counterData = field64(flags, 0, 48);
    }
    return 0;
}



int perfmon_setupCounterThread_sapphirerapids(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int err = 0;
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
    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        VERBOSEPRINTPCIREG(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, LLU_CAST (1ULL<<0), FREEZE_UNCORE);
        HPMwrite(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, (1ULL<<0));
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
            case MBOX0FIX:
            case MBOX1FIX:
            case MBOX2FIX:
            case MBOX3FIX:
            case MBOX4FIX:
            case MBOX5FIX:
            case MBOX6FIX:
            case MBOX7FIX:
            case MBOX8FIX:
            case MBOX9FIX:
            case MBOX10FIX:
            case MBOX11FIX:
            case MBOX12FIX:
            case MBOX13FIX:
            case MBOX14FIX:
            case MBOX15FIX:
            case HBM0FIX:
            case HBM1FIX:
            case HBM2FIX:
            case HBM3FIX:
            case HBM4FIX:
            case HBM5FIX:
            case HBM6FIX:
            case HBM7FIX:
            case HBM8FIX:
            case HBM9FIX:
            case HBM10FIX:
            case HBM11FIX:
            case HBM12FIX:
            case HBM13FIX:
            case HBM14FIX:
            case HBM15FIX:
            case HBM16FIX:
            case HBM17FIX:
            case HBM18FIX:
            case HBM19FIX:
            case HBM20FIX:
            case HBM21FIX:
            case HBM22FIX:
            case HBM23FIX:
            case HBM24FIX:
            case HBM25FIX:
            case HBM26FIX:
            case HBM27FIX:
            case HBM28FIX:
            case HBM29FIX:
            case HBM30FIX:
            case HBM31FIX:
                if (haveLock)
                {
                    spr_setup_uncore_fixed(thread_id, index, event);
                }
                break;
/*            case MDEV0:*/
/*                spr_setup_uncore_freerun(thread_id, index, event);*/
/*                break;*/

            case MBOX0:
            case MBOX1:
            case MBOX2:
            case MBOX3:
            case MBOX4:
            case MBOX5:
            case MBOX6:
            case MBOX7:
            case MBOX8:
            case MBOX9:
            case MBOX10:
            case MBOX11:
            case MBOX12:
            case MBOX13:
            case MBOX14:
            case MBOX15:
            case MDF0:
            case MDF1:
            case MDF2:
            case MDF3:
            case MDF4:
            case MDF5:
            case MDF6:
            case MDF7:
            case MDF8:
            case MDF9:
            case MDF10:
            case MDF11:
            case MDF12:
            case MDF13:
            case MDF14:
            case MDF15:
            case MDF16:
            case MDF17:
            case MDF18:
            case MDF19:
            case MDF20:
            case MDF21:
            case MDF22:
            case MDF23:
            case MDF24:
            case MDF25:
            case MDF26:
            case MDF27:
            case MDF28:
            case MDF29:
            case MDF30:
            case MDF31:
            case MDF32:
            case MDF33:
            case MDF34:
            case MDF35:
            case MDF36:
            case MDF37:
            case MDF38:
            case MDF39:
            case MDF40:
            case MDF41:
            case MDF42:
            case MDF43:
            case MDF44:
            case MDF45:
            case MDF46:
            case MDF47:
            case MDF48:
            case MDF49:
            case CBOX0:
            case CBOX1:
            case CBOX2:
            case CBOX3:
            case CBOX4:
            case CBOX5:
            case CBOX6:
            case CBOX7:
            case CBOX8:
            case CBOX9:
            case CBOX10:
            case CBOX11:
            case CBOX12:
            case CBOX13:
            case CBOX14:
            case CBOX15:
            case CBOX16:
            case CBOX17:
            case CBOX18:
            case CBOX19:
            case CBOX20:
            case CBOX21:
            case CBOX22:
            case CBOX23:
            case CBOX24:
            case CBOX25:
            case CBOX26:
            case CBOX27:
            case CBOX28:
            case CBOX29:
            case CBOX30:
            case CBOX31:
            case CBOX32:
            case CBOX33:
            case CBOX34:
            case CBOX35:
            case CBOX36:
            case CBOX37:
            case CBOX38:
            case CBOX39:
            case CBOX40:
            case CBOX41:
            case CBOX42:
            case CBOX43:
            case CBOX44:
            case CBOX45:
            case CBOX46:
            case CBOX47:
            case CBOX48:
            case CBOX49:
            case CBOX50:
            case CBOX51:
            case CBOX52:
            case CBOX53:
            case CBOX54:
            case CBOX55:
            case CBOX56:
            case CBOX57:
            case CBOX58:
            case CBOX59:
            case BBOX0:
            case BBOX1:
            case BBOX2:
            case BBOX3:
            case BBOX4:
            case BBOX5:
            case BBOX6:
            case BBOX7:
            case BBOX8:
            case BBOX9:
            case BBOX10:
            case BBOX11:
            case BBOX12:
            case BBOX13:
            case BBOX14:
            case BBOX15:
            case BBOX16:
            case BBOX17:
            case BBOX18:
            case BBOX19:
            case BBOX20:
            case BBOX21:
            case BBOX22:
            case BBOX23:
            case BBOX24:
            case BBOX25:
            case BBOX26:
            case BBOX27:
            case BBOX28:
            case BBOX29:
            case BBOX30:
            case BBOX31:
            case QBOX0:
            case QBOX1:
            case QBOX2:
            case RBOX0:
            case RBOX1:
            case RBOX2:
            case RBOX3:
            case WBOX:
            case IBOX0:
            case IBOX1:
            case IBOX2:
            case IBOX3:
            case IBOX4:
            case IBOX5:
            case IBOX6:
            case IBOX7:
            case IBOX8:
            case IBOX9:
            case IBOX10:
            case IBOX11:
            case IBOX12:
            case IRP0:
            case IRP1:
            case IRP2:
            case IRP3:
            case IRP4:
            case IRP5:
            case IRP6:
            case IRP7:
            case IRP8:
            case IRP9:
            case IRP10:
            case IRP11:
            case IRP12:
            case HBM0:
            case HBM1:
            case HBM2:
            case HBM3:
            case HBM4:
            case HBM5:
            case HBM6:
            case HBM7:
            case HBM8:
            case HBM9:
            case HBM10:
            case HBM11:
            case HBM12:
            case HBM13:
            case HBM14:
            case HBM15:
            case HBM16:
            case HBM17:
            case HBM18:
            case HBM19:
            case HBM20:
            case HBM21:
            case HBM22:
            case HBM23:
            case HBM24:
            case HBM25:
            case HBM26:
            case HBM27:
            case HBM28:
            case HBM29:
            case HBM30:
            case HBM31:
            case PBOX0:
            case PBOX1:
            case PBOX2:
            case PBOX3:
            case PBOX4:
            case PBOX5:
            case PBOX6:
            case PBOX7:
            case PBOX8:
            case PBOX9:
            case PBOX10:
            case PBOX11:
            case PBOX12:
            case PBOX13:
            case PBOX14:
            case PBOX15:
                if (haveLock) {
                    err = spr_setup_uncore(thread_id, index, event);
                    if (err < 0)
                    {
                        ERROR_PRINT(Failed to setup register 0x%X, reg);
                    }
                }
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
            data->startData = 0;
            data->counterData = 0;
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
                case MBOX0FIX:
                case MBOX1FIX:
                case MBOX2FIX:
                case MBOX3FIX:
                case MBOX4FIX:
                case MBOX5FIX:
                case MBOX6FIX:
                case MBOX7FIX:
                case MBOX8FIX:
                case MBOX9FIX:
                case MBOX10FIX:
                case MBOX11FIX:
                case MBOX12FIX:
                case MBOX13FIX:
                case MBOX14FIX:
                case MBOX15FIX:
                case HBM0FIX:
                case HBM1FIX:
                case HBM2FIX:
                case HBM3FIX:
                case HBM4FIX:
                case HBM5FIX:
                case HBM6FIX:
                case HBM7FIX:
                case HBM8FIX:
                case HBM9FIX:
                case HBM10FIX:
                case HBM11FIX:
                case HBM12FIX:
                case HBM13FIX:
                case HBM14FIX:
                case HBM15FIX:
                case HBM16FIX:
                case HBM17FIX:
                case HBM18FIX:
                case HBM19FIX:
                case HBM20FIX:
                case HBM21FIX:
                case HBM22FIX:
                case HBM23FIX:
                case HBM24FIX:
                case HBM25FIX:
                case HBM26FIX:
                case HBM27FIX:
                case HBM28FIX:
                case HBM29FIX:
                case HBM30FIX:
                case HBM31FIX:
                    if (haveLock)
                    {
                        spr_start_uncore_fixed(thread_id, index, event, data);
                    }
                    break;
/*                case MDEV0:*/
/*                    spr_start_uncore_freerun(thread_id, index, event, data);*/
/*                    break;*/
                case MBOX0:
                case MBOX1:
                case MBOX2:
                case MBOX3:
                case MBOX4:
                case MBOX5:
                case MBOX6:
                case MBOX7:
                case MBOX8:
                case MBOX9:
                case MBOX10:
                case MBOX11:
                case MBOX12:
                case MBOX13:
                case MBOX14:
                case MBOX15:
                case MDF0:
                case MDF1:
                case MDF2:
                case MDF3:
                case MDF4:
                case MDF5:
                case MDF6:
                case MDF7:
                case MDF8:
                case MDF9:
                case MDF10:
                case MDF11:
                case MDF12:
                case MDF13:
                case MDF14:
                case MDF15:
                case MDF16:
                case MDF17:
                case MDF18:
                case MDF19:
                case MDF20:
                case MDF21:
                case MDF22:
                case MDF23:
                case MDF24:
                case MDF25:
                case MDF26:
                case MDF27:
                case MDF28:
                case MDF29:
                case MDF30:
                case MDF31:
                case MDF32:
                case MDF33:
                case MDF34:
                case MDF35:
                case MDF36:
                case MDF37:
                case MDF38:
                case MDF39:
                case MDF40:
                case MDF41:
                case MDF42:
                case MDF43:
                case MDF44:
                case MDF45:
                case MDF46:
                case MDF47:
                case MDF48:
                case MDF49:
                case CBOX0:
                case CBOX1:
                case CBOX2:
                case CBOX3:
                case CBOX4:
                case CBOX5:
                case CBOX6:
                case CBOX7:
                case CBOX8:
                case CBOX9:
                case CBOX10:
                case CBOX11:
                case CBOX12:
                case CBOX13:
                case CBOX14:
                case CBOX15:
                case CBOX16:
                case CBOX17:
                case CBOX18:
                case CBOX19:
                case CBOX20:
                case CBOX21:
                case CBOX22:
                case CBOX23:
                case CBOX24:
                case CBOX25:
                case CBOX26:
                case CBOX27:
                case CBOX28:
                case CBOX29:
                case CBOX30:
                case CBOX31:
                case CBOX32:
                case CBOX33:
                case CBOX34:
                case CBOX35:
                case CBOX36:
                case CBOX37:
                case CBOX38:
                case CBOX39:
                case CBOX40:
                case CBOX41:
                case CBOX42:
                case CBOX43:
                case CBOX44:
                case CBOX45:
                case CBOX46:
                case CBOX47:
                case CBOX48:
                case CBOX49:
                case CBOX50:
                case CBOX51:
                case CBOX52:
                case CBOX53:
                case CBOX54:
                case CBOX55:
                case CBOX56:
                case CBOX57:
                case CBOX58:
                case CBOX59:
                case BBOX0:
                case BBOX1:
                case BBOX2:
                case BBOX3:
                case BBOX4:
                case BBOX5:
                case BBOX6:
                case BBOX7:
                case BBOX8:
                case BBOX9:
                case BBOX10:
                case BBOX11:
                case BBOX12:
                case BBOX13:
                case BBOX14:
                case BBOX15:
                case BBOX16:
                case BBOX17:
                case BBOX18:
                case BBOX19:
                case BBOX20:
                case BBOX21:
                case BBOX22:
                case BBOX23:
                case BBOX24:
                case BBOX25:
                case BBOX26:
                case BBOX27:
                case BBOX28:
                case BBOX29:
                case BBOX30:
                case BBOX31:
                case QBOX0:
                case QBOX1:
                case QBOX2:
                case RBOX0:
                case RBOX1:
                case RBOX2:
                case RBOX3:
                case WBOX:
                case IBOX0:
                case IBOX1:
                case IBOX2:
                case IBOX3:
                case IBOX4:
                case IBOX5:
                case IBOX6:
                case IBOX7:
                case IBOX8:
                case IBOX9:
                case IBOX10:
                case IBOX11:
                case IBOX12:
                case IRP0:
                case IRP1:
                case IRP2:
                case IRP3:
                case IRP4:
                case IRP5:
                case IRP6:
                case IRP7:
                case IRP8:
                case IRP9:
                case IRP10:
                case IRP11:
                case IRP12:
                case HBM0:
                case HBM1:
                case HBM2:
                case HBM3:
                case HBM4:
                case HBM5:
                case HBM6:
                case HBM7:
                case HBM8:
                case HBM9:
                case HBM10:
                case HBM11:
                case HBM12:
                case HBM13:
                case HBM14:
                case HBM15:
                case HBM16:
                case HBM17:
                case HBM18:
                case HBM19:
                case HBM20:
                case HBM21:
                case HBM22:
                case HBM23:
                case HBM24:
                case HBM25:
                case HBM26:
                case HBM27:
                case HBM28:
                case HBM29:
                case HBM30:
                case HBM31:
                case PBOX0:
                case PBOX1:
                case PBOX2:
                case PBOX3:
                case PBOX4:
                case PBOX5:
                case PBOX6:
                case PBOX7:
                case PBOX8:
                case PBOX9:
                case PBOX10:
                case PBOX11:
                case PBOX12:
                case PBOX13:
                case PBOX14:
                case PBOX15:
                    if (haveLock)
                    {
                        spr_start_uncore(thread_id, index, event, data);
                    }
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
    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        for (int i = MSR_DEV + 1; i < MAX_NUM_PCI_DEVICES; i++)
        {
            if (TESTTYPE(eventSet, i) && box_map[i].device != MSR_DEV)
            {
                VERBOSEPRINTPCIREG(cpu_id, box_map[i].device, box_map[i].ctrlRegister, LLU_CAST 0x0ULL, UNFREEZE_UNIT);
                HPMwrite(cpu_id, box_map[i].device, box_map[i].ctrlRegister, 0x0ULL);
            }
        }
        VERBOSEPRINTPCIREG(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, LLU_CAST 0x0ULL, UNFREEZE_UNCORE);
        HPMwrite(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, 0x0ULL);
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
    data[thread_id].counterData = counter_result;
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
    data[thread_id].counterData = counter_result;
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
            data[thread_id].overflows++;
        }
        data[thread_id].counterData = counter_result;
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
    data[thread_id].counterData = counter_result;
    return 0;
}

uint32_t spr_voltage_stop(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter_result = 0x0ULL;
    uint64_t counter1 = counter_map[index].counterRegister;
    CHECK_TEMP_READ_ERROR(voltage_read(cpu_id, &counter_result));
    VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, counter1, LLU_CAST counter_result, STOP_VOLTAGE);
    data[thread_id].counterData = counter_result;
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
    data[thread_id].counterData = counter_result;
    return 0;
}

uint32_t spr_mboxfix_stop(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter1 = counter_map[index].counterRegister;
    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
    SPR_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, STOP_MBOXFIX);
    data[thread_id].counterData = counter_result;
    return 0;
}


int perfmon_stopCountersThread_sapphirerapids(int thread_id, PerfmonEventSet* eventSet)
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
        for (int i = MSR_DEV + 1; i < MAX_NUM_PCI_DEVICES; i++)
        {
            if (TESTTYPE(eventSet, i) && box_map[i].device != MSR_DEV)
            {
                VERBOSEPRINTPCIREG(cpu_id, box_map[i].device, box_map[i].ctrlRegister, LLU_CAST (1ULL<<0), FREEZE_UNIT);
                HPMwrite(cpu_id, box_map[i].device, box_map[i].ctrlRegister, (1ULL<<0));
            }
        }
        VERBOSEPRINTPCIREG(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, LLU_CAST (1ULL<<0), FREEZE_UNCORE);
        HPMwrite(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, (1ULL<<0));
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
                case MBOX0FIX:
                case MBOX1FIX:
                case MBOX2FIX:
                case MBOX3FIX:
                case MBOX4FIX:
                case MBOX5FIX:
                case MBOX6FIX:
                case MBOX7FIX:
                case MBOX8FIX:
                case MBOX9FIX:
                case MBOX10FIX:
                case MBOX11FIX:
                case MBOX12FIX:
                case MBOX13FIX:
                case MBOX14FIX:
                case MBOX15FIX:
                case HBM0FIX:
                case HBM1FIX:
                case HBM2FIX:
                case HBM3FIX:
                case HBM4FIX:
                case HBM5FIX:
                case HBM6FIX:
                case HBM7FIX:
                case HBM8FIX:
                case HBM9FIX:
                case HBM10FIX:
                case HBM11FIX:
                case HBM12FIX:
                case HBM13FIX:
                case HBM14FIX:
                case HBM15FIX:
                case HBM16FIX:
                case HBM17FIX:
                case HBM18FIX:
                case HBM19FIX:
                case HBM20FIX:
                case HBM21FIX:
                case HBM22FIX:
                case HBM23FIX:
                case HBM24FIX:
                case HBM25FIX:
                case HBM26FIX:
                case HBM27FIX:
                case HBM28FIX:
                case HBM29FIX:
                case HBM30FIX:
                case HBM31FIX:
                    if (haveLock)
                    {
                        spr_stop_uncore_fixed(thread_id, index, event, data);
                    }
                    break;
/*                case MDEV0:*/
/*                    spr_stop_uncore_freerun(thread_id, index, event, data);*/
/*                    break;*/
                case MBOX0:
                case MBOX1:
                case MBOX2:
                case MBOX3:
                case MBOX4:
                case MBOX5:
                case MBOX6:
                case MBOX7:
                case MBOX8:
                case MBOX9:
                case MBOX10:
                case MBOX11:
                case MBOX12:
                case MBOX13:
                case MBOX14:
                case MBOX15:
                case MDF0:
                case MDF1:
                case MDF2:
                case MDF3:
                case MDF4:
                case MDF5:
                case MDF6:
                case MDF7:
                case MDF8:
                case MDF9:
                case MDF10:
                case MDF11:
                case MDF12:
                case MDF13:
                case MDF14:
                case MDF15:
                case MDF16:
                case MDF17:
                case MDF18:
                case MDF19:
                case MDF20:
                case MDF21:
                case MDF22:
                case MDF23:
                case MDF24:
                case MDF25:
                case MDF26:
                case MDF27:
                case MDF28:
                case MDF29:
                case MDF30:
                case MDF31:
                case MDF32:
                case MDF33:
                case MDF34:
                case MDF35:
                case MDF36:
                case MDF37:
                case MDF38:
                case MDF39:
                case MDF40:
                case MDF41:
                case MDF42:
                case MDF43:
                case MDF44:
                case MDF45:
                case MDF46:
                case MDF47:
                case MDF48:
                case MDF49:
                case CBOX0:
                case CBOX1:
                case CBOX2:
                case CBOX3:
                case CBOX4:
                case CBOX5:
                case CBOX6:
                case CBOX7:
                case CBOX8:
                case CBOX9:
                case CBOX10:
                case CBOX11:
                case CBOX12:
                case CBOX13:
                case CBOX14:
                case CBOX15:
                case CBOX16:
                case CBOX17:
                case CBOX18:
                case CBOX19:
                case CBOX20:
                case CBOX21:
                case CBOX22:
                case CBOX23:
                case CBOX24:
                case CBOX25:
                case CBOX26:
                case CBOX27:
                case CBOX28:
                case CBOX29:
                case CBOX30:
                case CBOX31:
                case CBOX32:
                case CBOX33:
                case CBOX34:
                case CBOX35:
                case CBOX36:
                case CBOX37:
                case CBOX38:
                case CBOX39:
                case CBOX40:
                case CBOX41:
                case CBOX42:
                case CBOX43:
                case CBOX44:
                case CBOX45:
                case CBOX46:
                case CBOX47:
                case CBOX48:
                case CBOX49:
                case CBOX50:
                case CBOX51:
                case CBOX52:
                case CBOX53:
                case CBOX54:
                case CBOX55:
                case CBOX56:
                case CBOX57:
                case CBOX58:
                case CBOX59:
                case BBOX0:
                case BBOX1:
                case BBOX2:
                case BBOX3:
                case BBOX4:
                case BBOX5:
                case BBOX6:
                case BBOX7:
                case BBOX8:
                case BBOX9:
                case BBOX10:
                case BBOX11:
                case BBOX12:
                case BBOX13:
                case BBOX14:
                case BBOX15:
                case BBOX16:
                case BBOX17:
                case BBOX18:
                case BBOX19:
                case BBOX20:
                case BBOX21:
                case BBOX22:
                case BBOX23:
                case BBOX24:
                case BBOX25:
                case BBOX26:
                case BBOX27:
                case BBOX28:
                case BBOX29:
                case BBOX30:
                case BBOX31:
                case QBOX0:
                case QBOX1:
                case QBOX2:
                case RBOX0:
                case RBOX1:
                case RBOX2:
                case RBOX3:
                case WBOX:
                case IBOX0:
                case IBOX1:
                case IBOX2:
                case IBOX3:
                case IBOX4:
                case IBOX5:
                case IBOX6:
                case IBOX7:
                case IBOX8:
                case IBOX9:
                case IBOX10:
                case IBOX11:
                case IBOX12:
                case IBOX13:
                case IBOX14:
                case IBOX15:
                case IRP0:
                case IRP1:
                case IRP2:
                case IRP3:
                case IRP4:
                case IRP5:
                case IRP6:
                case IRP7:
                case IRP8:
                case IRP9:
                case IRP10:
                case IRP11:
                case IRP12:
                case IRP13:
                case IRP14:
                case IRP15:
                case HBM0:
                case HBM1:
                case HBM2:
                case HBM3:
                case HBM4:
                case HBM5:
                case HBM6:
                case HBM7:
                case HBM8:
                case HBM9:
                case HBM10:
                case HBM11:
                case HBM12:
                case HBM13:
                case HBM14:
                case HBM15:
                case HBM16:
                case HBM17:
                case HBM18:
                case HBM19:
                case HBM20:
                case HBM21:
                case HBM22:
                case HBM23:
                case HBM24:
                case HBM25:
                case HBM26:
                case HBM27:
                case HBM28:
                case HBM29:
                case HBM30:
                case HBM31:
                case PBOX0:
                case PBOX1:
                case PBOX2:
                case PBOX3:
                case PBOX4:
                case PBOX5:
                case PBOX6:
                case PBOX7:
                case PBOX8:
                case PBOX9:
                case PBOX10:
                case PBOX11:
                case PBOX12:
                case PBOX13:
                case PBOX14:
                case PBOX15:
                    if (haveLock)
                    {
                        spr_stop_uncore(thread_id, index, event, data);
                    }
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
    data[thread_id].counterData = counter_result;
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
    data[thread_id].counterData = counter_result;
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
            data[thread_id].overflows++;
        }
        data[thread_id].counterData = counter_result;
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
    data[thread_id].counterData = counter_result;
    return 0;
}

uint32_t spr_voltage_read(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter_result = 0x0ULL;
    uint64_t counter1 = counter_map[index].counterRegister;
    CHECK_TEMP_READ_ERROR(voltage_read(cpu_id, &counter_result));
    VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, counter1, LLU_CAST counter_result, READ_VOLTAGE);
    data[thread_id].counterData = counter_result;
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
    data[thread_id].counterData = counter_result;
    return 0;
}

uint32_t spr_mboxfix_read(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter1 = counter_map[index].counterRegister;
    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
    SPR_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, STOP_MBOXFIX);
    data[thread_id].counterData = counter_result;
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
        for (int i = MSR_DEV + 1; i < MAX_NUM_PCI_DEVICES; i++)
        {
            if (TESTTYPE(eventSet, i) && box_map[i].device != MSR_DEV)
            {
                VERBOSEPRINTPCIREG(cpu_id, box_map[i].device, box_map[i].ctrlRegister, LLU_CAST (1ULL<<0), FREEZE_UNIT);
                HPMwrite(cpu_id, box_map[i].device, box_map[i].ctrlRegister, (1ULL<<0));
            }
        }
        VERBOSEPRINTPCIREG(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, LLU_CAST (1ULL<<0), FREEZE_UNCORE);
        HPMwrite(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, (1ULL<<0));
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

                case MBOX0FIX:
                case MBOX1FIX:
                case MBOX2FIX:
                case MBOX3FIX:
                case MBOX4FIX:
                case MBOX5FIX:
                case MBOX6FIX:
                case MBOX7FIX:
                case MBOX8FIX:
                case MBOX9FIX:
                case MBOX10FIX:
                case MBOX11FIX:
                case MBOX12FIX:
                case MBOX13FIX:
                case MBOX14FIX:
                case MBOX15FIX:
                case HBM0FIX:
                case HBM1FIX:
                case HBM2FIX:
                case HBM3FIX:
                case HBM4FIX:
                case HBM5FIX:
                case HBM6FIX:
                case HBM7FIX:
                case HBM8FIX:
                case HBM9FIX:
                case HBM10FIX:
                case HBM11FIX:
                case HBM12FIX:
                case HBM13FIX:
                case HBM14FIX:
                case HBM15FIX:
                case HBM16FIX:
                case HBM17FIX:
                case HBM18FIX:
                case HBM19FIX:
                case HBM20FIX:
                case HBM21FIX:
                case HBM22FIX:
                case HBM23FIX:
                case HBM24FIX:
                case HBM25FIX:
                case HBM26FIX:
                case HBM27FIX:
                case HBM28FIX:
                case HBM29FIX:
                case HBM30FIX:
                case HBM31FIX:
                    if (haveLock)
                    {
                        spr_read_uncore_fixed(thread_id, index, event, data);
                    }
                    break;

                case MBOX0:
                case MBOX1:
                case MBOX2:
                case MBOX3:
                case MBOX4:
                case MBOX5:
                case MBOX6:
                case MBOX7:
                case MBOX8:
                case MBOX9:
                case MBOX10:
                case MBOX11:
                case MBOX12:
                case MBOX13:
                case MBOX14:
                case MBOX15:
                case MDF0:
                case MDF1:
                case MDF2:
                case MDF3:
                case MDF4:
                case MDF5:
                case MDF6:
                case MDF7:
                case MDF8:
                case MDF9:
                case MDF10:
                case MDF11:
                case MDF12:
                case MDF13:
                case MDF14:
                case MDF15:
                case MDF16:
                case MDF17:
                case MDF18:
                case MDF19:
                case MDF20:
                case MDF21:
                case MDF22:
                case MDF23:
                case MDF24:
                case MDF25:
                case MDF26:
                case MDF27:
                case MDF28:
                case MDF29:
                case MDF30:
                case MDF31:
                case MDF32:
                case MDF33:
                case MDF34:
                case MDF35:
                case MDF36:
                case MDF37:
                case MDF38:
                case MDF39:
                case MDF40:
                case MDF41:
                case MDF42:
                case MDF43:
                case MDF44:
                case MDF45:
                case MDF46:
                case MDF47:
                case MDF48:
                case MDF49:
                case CBOX0:
                case CBOX1:
                case CBOX2:
                case CBOX3:
                case CBOX4:
                case CBOX5:
                case CBOX6:
                case CBOX7:
                case CBOX8:
                case CBOX9:
                case CBOX10:
                case CBOX11:
                case CBOX12:
                case CBOX13:
                case CBOX14:
                case CBOX15:
                case CBOX16:
                case CBOX17:
                case CBOX18:
                case CBOX19:
                case CBOX20:
                case CBOX21:
                case CBOX22:
                case CBOX23:
                case CBOX24:
                case CBOX25:
                case CBOX26:
                case CBOX27:
                case CBOX28:
                case CBOX29:
                case CBOX30:
                case CBOX31:
                case CBOX32:
                case CBOX33:
                case CBOX34:
                case CBOX35:
                case CBOX36:
                case CBOX37:
                case CBOX38:
                case CBOX39:
                case CBOX40:
                case CBOX41:
                case CBOX42:
                case CBOX43:
                case CBOX44:
                case CBOX45:
                case CBOX46:
                case CBOX47:
                case CBOX48:
                case CBOX49:
                case CBOX50:
                case CBOX51:
                case CBOX52:
                case CBOX53:
                case CBOX54:
                case CBOX55:
                case CBOX56:
                case CBOX57:
                case CBOX58:
                case CBOX59:
                case BBOX0:
                case BBOX1:
                case BBOX2:
                case BBOX3:
                case BBOX4:
                case BBOX5:
                case BBOX6:
                case BBOX7:
                case BBOX8:
                case BBOX9:
                case BBOX10:
                case BBOX11:
                case BBOX12:
                case BBOX13:
                case BBOX14:
                case BBOX15:
                case BBOX16:
                case BBOX17:
                case BBOX18:
                case BBOX19:
                case BBOX20:
                case BBOX21:
                case BBOX22:
                case BBOX23:
                case BBOX24:
                case BBOX25:
                case BBOX26:
                case BBOX27:
                case BBOX28:
                case BBOX29:
                case BBOX30:
                case BBOX31:
                case QBOX0:
                case QBOX1:
                case QBOX2:
                case QBOX3:
                case RBOX0:
                case RBOX1:
                case RBOX2:
                case RBOX3:
                case WBOX:
                case IBOX0:
                case IBOX1:
                case IBOX2:
                case IBOX3:
                case IBOX4:
                case IBOX5:
                case IBOX6:
                case IBOX7:
                case IBOX8:
                case IBOX9:
                case IBOX10:
                case IBOX11:
                case IBOX12:
                case IBOX13:
                case IBOX14:
                case IBOX15:
                case IRP0:
                case IRP1:
                case IRP2:
                case IRP3:
                case IRP4:
                case IRP5:
                case IRP6:
                case IRP7:
                case IRP8:
                case IRP9:
                case IRP10:
                case IRP11:
                case IRP12:
                case IRP13:
                case IRP14:
                case IRP15:
                case HBM0:
                case HBM1:
                case HBM2:
                case HBM3:
                case HBM4:
                case HBM5:
                case HBM6:
                case HBM7:
                case HBM8:
                case HBM9:
                case HBM10:
                case HBM11:
                case HBM12:
                case HBM13:
                case HBM14:
                case HBM15:
                case HBM16:
                case HBM17:
                case HBM18:
                case HBM19:
                case HBM20:
                case HBM21:
                case HBM22:
                case HBM23:
                case HBM24:
                case HBM25:
                case HBM26:
                case HBM27:
                case HBM28:
                case HBM29:
                case HBM30:
                case HBM31:
                case PBOX0:
                case PBOX1:
                case PBOX2:
                case PBOX3:
                case PBOX4:
                case PBOX5:
                case PBOX6:
                case PBOX7:
                case PBOX8:
                case PBOX9:
                case PBOX10:
                case PBOX11:
                case PBOX12:
                case PBOX13:
                case PBOX14:
                case PBOX15:
                    if (haveLock)
                    {
                        spr_read_uncore(thread_id, index, event, data);
                    }
                    break;
            }
        }
    }
    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        for (int i = MSR_DEV + 1; i < MAX_NUM_PCI_DEVICES; i++)
        {
            if (TESTTYPE(eventSet, i) && box_map[i].device != MSR_DEV)
            {
                VERBOSEPRINTPCIREG(cpu_id, box_map[i].device, box_map[i].ctrlRegister, LLU_CAST 0x0ULL, UNFREEZE_UNIT);
                HPMwrite(cpu_id, box_map[i].device, box_map[i].ctrlRegister, 0x0ULL);
            }
        }
        VERBOSEPRINTPCIREG(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, LLU_CAST 0x0ULL, UNFREEZE_UNCORE);
        HPMwrite(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, 0x0ULL);
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
            if (box_map[type].filterRegister1 != 0x0)
            {
                VERBOSEPRINTPCIREG(cpu_id, dev, box_map[type].filterRegister1, 0x0ULL, CLEAR_FILTER);
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, box_map[type].filterRegister1, 0x0ULL));
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
