/*
 * =======================================================================================
 *
 *      Filename:  perfmon_icelake.h
 *
 *      Description:  Header File of perfmon module for Intel Icelake.
 *
 *      Version:   5.2
 *      Released:  17.6.2021
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 NHR@FAU, University Erlangen-Nuremberg
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
    lock_acquire((int*) &die_lock[affinity_thread2die_lookup[cpu_id]], cpu_id);

    uint64_t misc_enable = 0x0;
    ret = HPMread(cpu_id, MSR_DEV, MSR_IA32_MISC_ENABLE, &misc_enable);
    if (ret == 0 && testBit(misc_enable, 7) && (testBit(misc_enable, 12) == 0))
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
                icelake_cbox_setup = icx_cbox_setup;
                break;
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

int icx_setup_mbox(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
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
                    flags |= (event->options[j].value & 0xFFULL) << 24;
                    break;
                default:
                    break;
            }
        }
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_MBOX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int icx_setup_mboxfix(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
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
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_MBOXFIX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}
    
int icx_ubox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
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

int icx_uboxfix_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
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

int icx_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    uint64_t filter_flags0 = 0x0ULL;
    uint32_t filter0 = box_map[counter_map[index].type].filterRegister1;
    uint64_t umask_ext_mask = 0x1FFFFF;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags = (1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;
    // TODO: Check for perf_event whether there is a separate field umask_ext
    //       or a multi-slice umask
    switch (event->eventId)
    {
        case 0x36:
        case 0x35:
            umask_ext_mask = 0x1FFFFF;
            break;
        case 0x34:
            umask_ext_mask = 0x1FFF;
            break;
        case 0x58:
        case 0x5A:
            umask_ext_mask = 0x3F;
            break;
        case 0x37:
            umask_ext_mask = 0xA0;
            break;
        default:
            umask_ext_mask = 0x0;
            break;
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
                case EVENT_OPTION_THRESHOLD:
                    flags |= (event->options[j].value & 0xFFULL) << 24;
                    break;
                case EVENT_OPTION_MATCH0:
                    flags |= (event->options[j].value & umask_ext_mask) << 32;
                    break;
                case EVENT_OPTION_STATE:
                    flags |= (event->options[j].value & 0xFFULL) << 8;
                    break;
                case EVENT_OPTION_TID:
                    flags |= (1ULL<<19);
                    filter_flags0 |= (event->options[j].value & 0x1FFULL);
                    break;
                    break;
                default:
                    break;
            }
        }
    }

    if (filter_flags0 != 0x0ULL)
    {
        VERBOSEPRINTREG(cpu_id, filter0, filter_flags0, SETUP_CBOX_FILTER0);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, filter0, filter_flags0));
    }
    else
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, filter0, 0x0ULL));
    }

    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_CBOX);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int icx_wbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
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
                case EVENT_OPTION_OCCUPANCY_EDGE:
                    flags |= (1ULL<<31);
                    break;
                case EVENT_OPTION_OCCUPANCY_INVERT:
                    flags |= (1ULL<<30);
                    break;
                default:
                    break;
            }
        }
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_UNCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int icx_uncore_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    PciDeviceIndex dev = counter_map[index].device;

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
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_UNCORE);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}


int icx_upi_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    PciDeviceIndex dev = counter_map[index].device;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags = (1ULL<<22)|(1ULL<<20);
    flags |= (event->umask<<8) + event->eventId;
    if (event->cfgBits != 0x0)
    {
        flags |= (event->cfgBits & 0xFFFFFFULL) << 32;
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
                case EVENT_OPTION_THRESHOLD:
                    flags |= (event->options[j].value & 0x1FULL) << 24;
                    break;
                case EVENT_OPTION_MATCH0:
                    flags |= (event->options[j].value & 0xFFFFFFULL) << 32;
                    break;
                case EVENT_OPTION_OPCODE:
                    flags |= (event->options[j].value & 0xFULL) << 12;
                    flags |= (1ULL<<32);
                    break;
                default:
                    break;
            }
        }
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_QBOX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}


int icx_irp_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
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
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_IRP);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int icx_tc_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
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
                    flags |= (event->options[j].value & 0xFFFULL) << 24;
                    break;
                case EVENT_OPTION_MASK0:
                    flags |= (event->options[j].value & 0xFFFULL) << 36;
                    break;
                case EVENT_OPTION_MASK1:
                    flags |= (event->options[j].value & 0x7ULL) << 48;
                    break;
                default:
                    break;
            }
        }
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_IIO);
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
            case WBOX0FIX:
                break;
            case MDEV0:
            case MDEV1:
            case MDEV2:
            case MDEV3:
                break;
            case MBOX0FIX:
            case MBOX1FIX:
            case MBOX2FIX:
            case MBOX3FIX:
            case MBOX4FIX:
            case MBOX5FIX:
            case MBOX6FIX:
            case MBOX7FIX:
		if (haveLock && ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2)))
                {
                    icx_setup_mboxfix(cpu_id, index, event);
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
                if (haveLock && ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2)))
                {
                    icx_setup_mbox(cpu_id, index, event);
                }
                break;
            case UBOXFIX:
                if (haveLock)
                {
                    icx_uboxfix_setup(cpu_id, index, event);
                }
                break;
            case UBOX:
                if (haveLock)
                {
                    icx_ubox_setup(cpu_id, index, event);
                }
                break;
            case WBOX:
                if (haveLock)
                {
                    icx_wbox_setup(cpu_id, index, event);
                }
                break;
            case BBOX0:
            case BBOX1:
            case BBOX2:
            case BBOX3:
                if (haveLock)
                {
                    icx_uncore_setup(cpu_id, index, event);
                }
                break;
            case SBOX0:
            case SBOX1:
            case SBOX2:
                if (haveLock)
                {
                    icx_uncore_setup(cpu_id, index, event);
                }
                break;
            case QBOX0:
            case QBOX1:
            case QBOX2:
                if (haveLock)
                {
                    icx_upi_setup(cpu_id, index, event);
                }
                break;
            case PBOX:
            case PBOX1:
            case PBOX2:
            case PBOX3:
            case PBOX4:
            case PBOX5:
                if (haveLock)
                {
                    icx_uncore_setup(cpu_id, index, event);
                }
                break;
            case IBOX0:
            case IBOX1:
            case IBOX2:
            case IBOX3:
            case IBOX4:
            case IBOX5:
                if (haveLock)
                {
                    icx_irp_setup(cpu_id, index, event);
                }
                break;
            case EUBOX0:
            case EUBOX1:
            case EUBOX2:
            case EUBOX3:
            case EUBOX4:
            case EUBOX5:
                if (haveLock)
                {
                    icx_tc_setup(cpu_id, index, event);
                }
                break;
            case EUBOX0FIX:
            case EUBOX1FIX:
            case EUBOX2FIX:
            case EUBOX3FIX:
            case EUBOX4FIX:
            case EUBOX5FIX:
            case IBOX0FIX:
            case IBOX1FIX:
            case IBOX2FIX:
            case IBOX3FIX:
            case IBOX4FIX:
            case IBOX5FIX:
                break;
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
                    VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_FIXED);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    flags |= (1ULL<<(index+32));  /* enable fixed counter */
                    break;
                case PMC:
                    VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_PMC);
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
                case MDEV0:
                case MDEV1:
                case MDEV2:
                case MDEV3:
                    if (haveLock && ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2)))
                    {
                        CHECK_MMIO_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST tmp, START_MDEV_RAW);
                        eventSet->events[i].threadCounter[thread_id].startData = tmp;//field64(tmp, 0, box_map[type].regWidth);
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST eventSet->events[i].threadCounter[thread_id].startData, START_MDEV);
                    }
                    break;
                case MBOX0FIX:
                case MBOX1FIX:
                case MBOX2FIX:
                case MBOX3FIX:
                case MBOX4FIX:
                case MBOX5FIX:
                case MBOX6FIX:
                case MBOX7FIX:
		    if (haveLock && ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2)))
                    {
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_MBOXFIX);
                        CHECK_MMIO_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
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
                    if (haveLock)
                    {
                        if ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2))
                        {
                            VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_MBOX);
                            CHECK_MMIO_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                        }
                        else if (cpuid_info.model == ICELAKE1 || cpuid_info.model == ICELAKE2)
                        {
                            CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                            eventSet->events[i].threadCounter[thread_id].startData = field64(tmp, 0, box_map[type].regWidth);
                            VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST eventSet->events[i].threadCounter[thread_id].startData, READ_MBOX);
                        }
                    }
                    break;
                case MBOX0TMP:
                    if (haveLock && (cpuid_info.model == ICELAKE1 || cpuid_info.model == ICELAKE2))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                        eventSet->events[i].threadCounter[thread_id].startData = field64(tmp, 0, box_map[type].regWidth);
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST eventSet->events[i].threadCounter[thread_id].startData, READ_MBOX);
                    }
                    break;
                case BBOX0:
                case BBOX1:
                case BBOX2:
                case BBOX3:
                    if (haveLock)
                    {
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_BBOX);
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    }
                    break;
                case SBOX0:
                case SBOX1:
                case SBOX2:
                    if (haveLock)
                    {
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_SBOX);
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    }
                    break;
                case QBOX0:
                case QBOX1:
                case QBOX2:
                    if (haveLock)
                    {
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_QBOX);
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    }
                    break;
                case UBOXFIX:
                    if (haveLock)
                    {
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_UBOXFIX);
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    }
                    break;
                case UBOX:
                    if (haveLock)
                    {
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_UBOX);
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    }
                    break;
                case WBOX:
                    if (haveLock)
                    {
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_WBOX);
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    }
                    break;
                case WBOX0FIX:
                    if (haveLock)
                    {
                        tmp = 0x0ULL;
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                        eventSet->events[i].threadCounter[thread_id].startData = field64(tmp, 0, box_map[type].regWidth);
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST eventSet->events[i].threadCounter[thread_id].startData, START_WBOXFIX);
                    }
                    break;
                case EUBOX0:
                case EUBOX1:
                case EUBOX2:
                case EUBOX3:
                case EUBOX4:
                case EUBOX5:
                    if (haveLock)
                    {
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_IIO);
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    }
                    break;
                case PBOX:
                case PBOX1:
                case PBOX2:
                case PBOX3:
                case PBOX4:
                case PBOX5:
                    if (haveLock)
                    {
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_PBOX);
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    }
                    break;
                case EUBOX0FIX:
                case EUBOX1FIX:
                case EUBOX2FIX:
                case EUBOX3FIX:
                case EUBOX4FIX:
                case EUBOX5FIX:
                case IBOX0FIX:
                case IBOX1FIX:
                case IBOX2FIX:
                case IBOX3FIX:
                case IBOX4FIX:
                case IBOX5FIX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                        eventSet->events[i].threadCounter[thread_id].startData = field64(tmp, 0, box_map[type].regWidth);
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST eventSet->events[i].threadCounter[thread_id].startData, READ_IIOPORT);
                    }
                    break;
                case IBOX0:
                case IBOX1:
                case IBOX2:
                case IBOX3:
                case IBOX4:
                case IBOX5:
                    if (haveLock)
                    {
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_IBOX);
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
                    if (haveLock && ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2)))
                    {
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0ULL, CLEAR_CBOX);
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
        else if ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2))
        {
            VERBOSEPRINTREG(cpu_id, MSR_UNC_ICX_U_PMON_GLOBAL_CTRL, (1ULL<<61), UNFREEZE_UNCORE);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_ICX_U_PMON_GLOBAL_CTRL, (1ULL<<61)));
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

#define ICX_CHECK_UNCORE_OVERFLOW(offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t ovf_values = 0x0ULL; \
        if ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2)) \
        { \
            int my_off = offset; \
            if (my_off <= 63) \
            { \
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_UNC_ICX_U_PMON_GLOBAL_STATUS1, &ovf_values)); \
            } \
            else \
            { \
                my_off -= 64; \
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_UNC_ICX_U_PMON_GLOBAL_STATUS2, &ovf_values)); \
            } \
            if (ovf_values & (1ULL<<(my_off))) \
            { \
                eventSet->events[i].threadCounter[thread_id].overflows++; \
            } \
        } \
        else if ((cpuid_info.model == ICELAKE1) || (cpuid_info.model == ICELAKE2)) \
        { \
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_STATUS, &ovf_values)); \
            if (ovf_values & (1ULL<<(offset))) \
            { \
                eventSet->events[i].threadCounter[thread_id].overflows++; \
            } \
        } \
    }

int icx_uncore_read(int cpu_id, RegisterIndex index, PerfmonEvent *event,
                     uint64_t* cur_result, int* overflows, int flags,
                     int global_offset, int box_offset)
{
    uint64_t result = 0x0ULL;
    uint64_t tmp = 0x0ULL;
    RegisterType type = counter_map[index].type;
    PciDeviceIndex dev = counter_map[index].device;
    uint64_t counter1 = counter_map[index].counterRegister;
    event++;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter1, &result));
    VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST result, READ_REG_1);
    if (flags & FREEZE_FLAG_CLEAR_CTR)
    {
        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0U, CLEAR_REG_1);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0U));
    }
    result = field64(result, 0, box_map[type].regWidth);
    if (result < *cur_result)
    {
        uint64_t ovf_values = 0x0ULL;
        int test_local = 0;
        uint32_t global_status_reg = MSR_UNC_ICX_U_PMON_GLOBAL_STATUS1;
        if ((cpuid_info.model == ICELAKE1) || (cpuid_info.model == ICELAKE2))
        {
            global_status_reg = MSR_V4_UNC_PERF_GLOBAL_STATUS;
        }
        else if (global_offset >= 64)
        {
            global_status_reg = MSR_UNC_ICX_U_PMON_GLOBAL_STATUS2;
            global_offset -= 64;
        }
        
        if (global_offset != -1)
        {
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV,
                                           global_status_reg,
                                           &ovf_values));
            VERBOSEPRINTREG(cpu_id, global_status_reg, LLU_CAST ovf_values, READ_GLOBAL_OVFL);
            if (ovf_values & (1ULL<<global_offset))
            {
                VERBOSEPRINTREG(cpu_id, global_status_reg, LLU_CAST (1<<global_offset), CLEAR_GLOBAL_OVFL);
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV,
                                                 global_status_reg,
                                                 (1<<global_offset)));
                test_local = 1;
            }
        }
        else
        {
            test_local = 1;
        }

        if (test_local && box_offset >= 0)
        {
            ovf_values = 0x0ULL;
            CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev,
                                              box_map[type].statusRegister,
                                              &ovf_values));
            VERBOSEPRINTPCIREG(cpu_id, dev, box_map[type].statusRegister, LLU_CAST ovf_values, READ_BOX_OVFL);
            if (ovf_values & (1ULL<<box_offset))
            {
                (*overflows)++;
                VERBOSEPRINTPCIREG(cpu_id, dev, box_map[type].statusRegister, LLU_CAST (1<<box_offset), RESET_BOX_OVFL);
                CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev,
                                                    box_map[type].statusRegister,
                                                    (1<<box_offset)));
            }
        }
        else if (test_local)
        {
            (*overflows)++;
        }
    }
    *cur_result = result;
    return 0;
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
        if ((cpuid_info.model == ICELAKE1) || (cpuid_info.model == ICELAKE2))
        {
            VERBOSEPRINTREG(cpu_id, MSR_V4_UNC_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_UNCORE);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_CTRL, 0x0ULL));
        }
        else if ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2))
        {
            VERBOSEPRINTREG(cpu_id, MSR_UNC_ICX_U_PMON_GLOBAL_CTRL, (1ULL<<63), FREEZE_UNCORE);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_ICX_U_PMON_GLOBAL_CTRL, (1ULL<<63)));
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
                    SKL_CHECK_CORE_OVERFLOW(index+32);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, STOP_FIXED)
                    break;
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    SKL_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
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
                    VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, counter1, LLU_CAST counter_result, STOP_THERMAL);
                    break;

                case VOLTAGE:
                    CHECK_TEMP_READ_ERROR(voltage_read(cpu_id, &counter_result));
                    VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, counter1, LLU_CAST counter_result, STOP_VOLTAGE);
                    break;

                case METRICS:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    counter_result= field64(counter_result, getCounterTypeOffset(index)*box_map[type].regWidth, box_map[type].regWidth);
                    VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, counter1, LLU_CAST counter_result, STOP_METRICS);
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
                    if (haveLock)
                    {
                        if ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2))
                        {
                            CHECK_MMIO_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                            counter_result = field64(counter_result, 0, box_map[type].regWidth);
                            VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_MBOX);
                        }
                        else if ((cpuid_info.model == ICELAKE1) || (cpuid_info.model == ICELAKE2))
                        {
                            CHECK_MSR_READ_ERROR(HPMread(cpu_id, PCI_IMC_DEVICE_0_CH_0, counter1, &counter_result));
                            counter_result = field64(counter_result, 0, box_map[type].regWidth);
                            VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_MBOX);
                        }
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, OVERFLOW_MEM)
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                    }
                    break;
                case MBOX0FIX:
                case MBOX1FIX:
                case MBOX2FIX:
                case MBOX3FIX:
                case MBOX4FIX:
                case MBOX5FIX:
                case MBOX6FIX:
                case MBOX7FIX:
		    if (haveLock && (cpuid_info.model == ICELAKEX1 || cpuid_info.model == ICELAKEX2))
                    {
                        CHECK_MMIO_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                        counter_result = field64(counter_result, 0, box_map[type].regWidth);
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_MBOXFIX);
                    }
                    break;
                case MBOX0TMP:
                    if (haveLock && (cpuid_info.model == ICELAKE1 || cpuid_info.model == ICELAKE2))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, PCI_IMC_DEVICE_0_CH_0, counter1, &counter_result));
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, OVERFLOW_CLIENTMEM)
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_MBOX);
                    }
                    break;
                case MDEV0:
                case MDEV1:
                case MDEV2:
                case MDEV3:
                    if (haveLock && ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2)))
                    {
                        CHECK_MMIO_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                        counter_result = tmp;//field64(tmp, 0, box_map[type].regWidth);
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, OVERFLOW_MDEV)
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST eventSet->events[i].threadCounter[thread_id].startData, STOP_MDEV);
                    }
                    break;
                case UBOXFIX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        ICX_CHECK_UNCORE_OVERFLOW(box_map[type].ovflOffset);
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_UBOXFIX);
                    }
                    break;
                case UBOX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        ICX_CHECK_UNCORE_OVERFLOW(box_map[type].ovflOffset);
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_UBOX);
                    }
                    break;
                case EUBOX0:
                case EUBOX1:
                case EUBOX2:
                case EUBOX3:
                case EUBOX4:
                case EUBOX5:
                    if (haveLock && ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2)))
                    {
                        icx_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_IIO);
                    }
                    break;
                case EUBOX0FIX:
                case EUBOX1FIX:
                case EUBOX2FIX:
                case EUBOX3FIX:
                case EUBOX4FIX:
                case EUBOX5FIX:
                case IBOX0FIX:
                case IBOX1FIX:
                case IBOX2FIX:
                case IBOX3FIX:
                case IBOX4FIX:
                case IBOX5FIX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                        counter_result = field64(tmp, 0, box_map[type].regWidth);
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_IIOPORT);
                    }
                    break;
                case IBOX0:
                case IBOX1:
                case IBOX2:
                case IBOX3:
                case IBOX4:
                case IBOX5:
                    if (haveLock)
                    {
                        icx_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_IBOX);
                    }
                    break;
                case WBOX:
                    if (haveLock && ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2)))
                    {
                        icx_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_WBOX);
                    }
                    break;
                case WBOX0FIX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_WBOXFIX);
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
                    if (haveLock)
                    {
                        icx_uncore_read(cpu_id, index, event, current, overflows, 
                                        FREEZE_FLAG_CLEAR_CTR, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_CBOX);
                    }
                    break;
                case BBOX0:
                case BBOX1:
                case BBOX2:
                case BBOX3:
                    if (haveLock)
                    {
                        icx_uncore_read(cpu_id, index, event, current, overflows, 
                                        FREEZE_FLAG_CLEAR_CTR, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_BBOX);
                    }
                    break;
                case SBOX0:
                case SBOX1:
                case SBOX2:
                    if (haveLock)
                    {
                        icx_uncore_read(cpu_id, index, event, current, overflows, 
                                        FREEZE_FLAG_CLEAR_CTR, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_SBOX);
                    }
                    break;
                case QBOX0:
                case QBOX1:
                case QBOX2:
                    if (haveLock)
                    {
                        icx_uncore_read(cpu_id, index, event, current, overflows, 
                                        FREEZE_FLAG_CLEAR_CTR, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_QBOX);
                    }
                    break;
                case PBOX:
                case PBOX1:
                case PBOX2:
                case PBOX3:
                case PBOX4:
                case PBOX5:
                    if (haveLock)
                    {
                        icx_uncore_read(cpu_id, index, event, current, overflows, 
                                        FREEZE_FLAG_CLEAR_CTR, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_PBOX);
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
        if ((cpuid_info.model == ICELAKE1) && (cpuid_info.model == ICELAKE2))
        {
            VERBOSEPRINTREG(cpu_id, MSR_V4_UNC_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_UNCORE);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_CTRL, 0x0ULL));
        }
        else if (((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2)))
        {
            VERBOSEPRINTREG(cpu_id, MSR_UNC_ICX_U_PMON_GLOBAL_CTRL, (1ULL<<63), FREEZE_UNCORE);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_ICX_U_PMON_GLOBAL_CTRL, (1ULL<<63)));
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
                    SKL_CHECK_CORE_OVERFLOW(index+32);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED)
                    break;
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    SKL_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
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
                    VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, counter1, LLU_CAST counter_result, READ_THERMAL);
                    break;

                case VOLTAGE:
                    CHECK_TEMP_READ_ERROR(voltage_read(cpu_id, &counter_result));
                    VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, counter1, LLU_CAST counter_result, READ_VOLTAGE);
                    break;

                case METRICS:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    counter_result= field64(counter_result, getCounterTypeOffset(index)*box_map[type].regWidth, box_map[type].regWidth);
                    VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, counter1, LLU_CAST counter_result, READ_METRICS);
                    break;

                case UBOXFIX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        ICX_CHECK_UNCORE_OVERFLOW(box_map[type].ovflOffset);
                        VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, counter1, LLU_CAST counter_result, READ_UBOXFIX);
                    }
                    break;
                case UBOX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        ICX_CHECK_UNCORE_OVERFLOW(box_map[type].ovflOffset);
                        VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, counter1, LLU_CAST counter_result, READ_UBOX);
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
                    if (haveLock)
                    {
                        if ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2))
                        {
                            CHECK_MMIO_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                            counter_result = field64(counter_result, 0, box_map[type].regWidth);
                            VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_MBOX);
                        }
                        else if ((cpuid_info.model == ICELAKE1) && (cpuid_info.model == ICELAKE2))
                        {
                            CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                            counter_result = field64(counter_result, 0, box_map[type].regWidth);
                            VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_MBOX);
                        }
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, OVERFLOW_MEM)
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                    }
                    break;
                case MBOX0TMP:
                    if (haveLock && (cpuid_info.model == ICELAKE1 || cpuid_info.model == ICELAKE2))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_MBOX);
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, OVERFLOW_CLIENTMEM)
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                    }
                    break;
                case MDEV0:
                case MDEV1:
                case MDEV2:
                case MDEV3:
                    if (haveLock && ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2)))
                    {
                        CHECK_MMIO_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                        counter_result = tmp;//field64(tmp, 0, box_map[type].regWidth);
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, OVERFLOW_MDEV)
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST eventSet->events[i].threadCounter[thread_id].startData, READ_MDEV);
                    }
                    break;
                case MBOX0FIX:
                case MBOX1FIX:
                case MBOX2FIX:
                case MBOX3FIX:
                case MBOX4FIX:
                case MBOX5FIX:
                case MBOX6FIX:
                case MBOX7FIX:
		    if (haveLock && (cpuid_info.model == ICELAKEX1 || cpuid_info.model == ICELAKEX2))
                    {
                        CHECK_MMIO_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                        counter_result = field64(counter_result, 0, box_map[type].regWidth);
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, OVERFLOW_MBOXFIX)
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_MBOXFIX);
                    }
                    break;
                case EUBOX0:
                case EUBOX1:
                case EUBOX2:
                case EUBOX3:
                case EUBOX4:
                case EUBOX5:
                    if (haveLock && ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2)))
                    {
                        icx_uncore_read(cpu_id, index, event, current, overflows, 0, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_IIO);
                    }
                    break;
                case EUBOX0FIX:
                case EUBOX1FIX:
                case EUBOX2FIX:
                case EUBOX3FIX:
                case EUBOX4FIX:
                case EUBOX5FIX:
                case IBOX0FIX:
                case IBOX1FIX:
                case IBOX2FIX:
                case IBOX3FIX:
                case IBOX4FIX:
                case IBOX5FIX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                        counter_result = field64(tmp, 0, box_map[type].regWidth);
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_IIOPORT);
                    }
                    break;
                case IBOX0:
                case IBOX1:
                case IBOX2:
                case IBOX3:
                case IBOX4:
                case IBOX5:
                    if (haveLock)
                    {
                        icx_uncore_read(cpu_id, index, event, current, overflows, 0, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_IBOX);
                    }
                    break;
                case WBOX:
                    if (haveLock && ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2)))
                    {
                        icx_uncore_read(cpu_id, index, event, current, overflows, 0, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_WBOX);
                    }
                    break;
                case WBOX0FIX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_WBOXFIX);
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
                    if (haveLock && ((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2)))
                    {
                        icx_uncore_read(cpu_id, index, event, current, overflows, 0, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_CBOX);
                    }
                    break;
                case BBOX0:
                case BBOX1:
                case BBOX2:
                case BBOX3:
                    if (haveLock)
                    {
                        icx_uncore_read(cpu_id, index, event, current, overflows, 0, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_BBOX);
                    }
                    break;
                case SBOX0:
                case SBOX1:
                case SBOX2:
                    if (haveLock)
                    {
                        icx_uncore_read(cpu_id, index, event, current, overflows, 0, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_SBOX);
                    }
                    break;
                case QBOX0:
                case QBOX1:
                case QBOX2:
                    if (haveLock)
                    {
                        icx_uncore_read(cpu_id, index, event, current, overflows, 0, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_QBOX);
                    }
                    break;
                case PBOX:
                case PBOX1:
                case PBOX2:
                case PBOX3:
                case PBOX4:
                case PBOX5:
                    if (haveLock)
                    {
                        icx_uncore_read(cpu_id, index, event, current, overflows, 0, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_PBOX);
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
        if ((cpuid_info.model == ICELAKE1) || (cpuid_info.model == ICELAKE2))
        {
            VERBOSEPRINTREG(cpu_id, MSR_V4_UNC_PERF_GLOBAL_CTRL, uflags|(1ULL<<29), UNFREEZE_UNCORE);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_CTRL, uflags|(1ULL<<29)));
        }
        else if (((cpuid_info.model == ICELAKEX1) || (cpuid_info.model == ICELAKEX2)))
        {
            VERBOSEPRINTREG(cpu_id, MSR_UNC_ICX_U_PMON_GLOBAL_CTRL, (1ULL<<61), UNFREEZE_UNCORE);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_ICX_U_PMON_GLOBAL_CTRL, (1ULL<<61)));
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
