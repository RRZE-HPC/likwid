/*
 * =======================================================================================
 *
 *      Filename:  perfmon_skylake.h
 *
 *      Description:  Header File of perfmon module for Intel Skylake.
 *
 *      Version:   5.2
 *      Released:  17.6.2021
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
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

#include <perfmon_skylake_events.h>
#include <perfmon_skylake_counters.h>
#include <perfmon_skylakeX_events.h>
#include <perfmon_skylakeX_counters.h>
#include <error.h>
#include <affinity.h>
#include <limits.h>
#include <topology.h>
#include <access.h>
#include <voltage.h>
#include <linux/version.h>

#define IS_SKYLAKE_CLIENT(model) ((model) == SKYLAKE1 || \
                                  (model) == SKYLAKE2 || \
                                  (model) == KABYLAKE1 || \
                                  (model) == KABYLAKE2 || \
                                  (model) == CANNONLAKE || \
                                  (model) == COMETLAKE1 || \
                                  (model) == COMETLAKE2)

static int perfmon_numCountersSkylake = NUM_COUNTERS_SKYLAKE;
static int perfmon_numCoreCountersSkylake = NUM_COUNTERS_CORE_SKYLAKE;
static int perfmon_numArchEventsSkylake = NUM_ARCH_EVENTS_SKYLAKE;

static int perfmon_numCountersSkylakeX = NUM_COUNTERS_SKYLAKEX;
static int perfmon_numCoreCountersSkylakeX = NUM_COUNTERS_CORE_SKYLAKEX;
static int perfmon_numArchEventsSkylakeX = NUM_ARCH_EVENTS_SKYLAKEX;

static int skl_did_cbox_check = 0;
int skl_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event);
int skx_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event);
int (*skylake_cbox_setup)(int, RegisterIndex, PerfmonEvent *);

int skl_cbox_nosetup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    cpu_id++;
    index++;
    event++;
    return 0;
}

int has_uncore_lock(int cpu_id)
{
    if (IS_SKYLAKE_CLIENT(cpuid_info.model) &&
        socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        return 1;
    }
    else if (cpuid_info.model == SKYLAKEX)
    {
        if (cpuid_topology.numSockets == cpuid_topology.numDies)
        {
            if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
            {
                return 1;
            }
        }
        else
        {
            if (die_lock[affinity_thread2die_lookup[cpu_id]] == cpu_id)
            {
                return 1;
            }
        }
    }
    return 0;
}
int perfmon_init_skylake(int cpu_id)
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
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        if (skl_did_cbox_check == 0)
        {
            if (IS_SKYLAKE_CLIENT(cpuid_info.model))
            {
                uint64_t data = 0x0ULL;
                ret = HPMwrite(cpu_id, MSR_DEV, MSR_V4_C0_PERF_CTRL0, 0x0ULL);
                ret += HPMread(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_CTRL, &data);
                ret += HPMwrite(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_CTRL, 0x0ULL);
                ret += HPMread(cpu_id, MSR_DEV, MSR_V4_C0_PERF_CTRL0, &data);
                if ((ret == 0) && (data == 0x0ULL))
                    skylake_cbox_setup = skl_cbox_setup;
                else
                    skylake_cbox_setup = skl_cbox_nosetup;
            }
            else if (cpuid_info.model == SKYLAKEX)
            {
                skylake_cbox_setup = skx_cbox_setup;
            }
            else
            {
                skylake_cbox_setup = skl_cbox_nosetup;
            }
            skl_did_cbox_check = 1;
        }
    }
    return 0;
}

uint32_t skl_fixed_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
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
            case EVENT_OPTION_ANYTHREAD:
                flags |= (1ULL<<(2+(index*4)));
            default:
                break;
        }
    }
    return flags;
}

int skl_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
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
        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE0);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, offcore_flags));
    }
    else if (event->eventId == 0xBB)
    {
        if ((event->cfgBits != 0xFF) && (event->cmask != 0xFF))
        {
            offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);
        }
        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP1, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE1);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP1, offcore_flags));
    }
    if ((event->eventId == 0xC6) &&
        (event->cmask != 0))
    {
        VERBOSEPRINTREG(cpu_id, MSR_V4_PEBS_FRONTEND, LLU_CAST event->cmask, SETUP_PMC_FRONTEND);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_V4_PEBS_FRONTEND, event->cmask));
    }
    if ((event->eventId == 0xCD) &&
        (cpuid_info.model == SKYLAKEX) &&
        (event->cmask != 0))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PEBS_LD_LAT, LLU_CAST event->cmask, SETUP_PMC_LATENCY);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_LD_LAT, event->cmask));
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_PMC)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister , flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int skl_ubox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;

    if (!has_uncore_lock(cpu_id))
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

int skl_uboxfix_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x0ULL;
    event++;
    if (!has_uncore_lock(cpu_id))
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

int skl_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    if (!has_uncore_lock(cpu_id))
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

int skx_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    uint64_t filter_flags0 = 0x0ULL;
    uint64_t filter_flags1 = 0x0ULL;
    uint32_t filter0 = box_map[counter_map[index].type].filterRegister1;
    uint32_t filter1 = box_map[counter_map[index].type].filterRegister2;
    int set_state_all = 0;
    int opc_match = 0;
    int match1 = 0;

    if (!has_uncore_lock(cpu_id))
    {
        return 0;
    }

    flags = (1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;
    if (event->eventId == 0x34)
    {
        set_state_all = 1;
    }
    if ((event->eventId == 0x13 || event->eventId == 0x11) && (event->umask & 0x2ULL))
    {
        fprintf(stderr, "IRQ_REJECTED should not be Ored with the other umasks.");
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
                case EVENT_OPTION_OPCODE:
                    filter_flags1 |= (extractBitField(event->options[j].value,20,0) << 9);
                    filter_flags1 |= (0x3<<27);
                    filter_flags1 |= (0x3<<17);
                    opc_match = 1;
                    break;
                case EVENT_OPTION_STATE:
                    filter_flags0 |= (extractBitField(event->options[j].value,10,0) << 17);
                    set_state_all = 0;
                    break;
                case EVENT_OPTION_TID:
                    filter_flags0 |= (extractBitField(event->options[j].value,9,0));
                    flags |= (1ULL<<19);
                    break;
                case EVENT_OPTION_MATCH0:
                    filter_flags1 |= ((extractBitField(event->options[j].value,2,0) & 0x3ULL) << 30);
                    break;
                case EVENT_OPTION_MATCH1:
                    filter_flags1 |= (extractBitField(event->options[j].value,6,0) & 0x33);
                    match1 = 1;
                    break;
                default:
                    break;
            }
        }
    }
    if (opc_match && !match1)
    {
        filter_flags1 |= 0x33ULL;
        VERBOSEPRINTREG(cpu_id, filter1, filter_flags1, SETUP_CBOX_ADD_OPCODE_MATCH1);
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
    if (filter_flags1 != 0x0ULL)
    {
        VERBOSEPRINTREG(cpu_id, filter1, filter_flags1, SETUP_CBOX_FILTER1);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, filter1, filter_flags1));
    }
    else
    {
        VERBOSEPRINTREG(cpu_id, filter1, 0x3BULL, SETUP_CBOX_DEF_FILTER_STATE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, filter1, 0x3BULL));
    }

    if (set_state_all)
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, filter0, &filter_flags0));
        filter_flags0 |= (0x3FF << 17);
        VERBOSEPRINTREG(cpu_id, filter0, filter_flags0, SETUP_CBOX_DEF_FILTER_STATE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, filter0, filter_flags0));
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_CBOX);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int skx_mbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    PciDeviceIndex dev = counter_map[index].device;

    if (!has_uncore_lock(cpu_id))
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

int skx_mboxfix_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    PciDeviceIndex dev = counter_map[index].device;
    event++;
    if (!has_uncore_lock(cpu_id))
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
        VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].configRegister, flags, SETUP_MBOX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int skx_wbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j = 0;
    uint64_t flags = 0x0ULL;
    uint64_t filter = box_map[counter_map[index].type].filterRegister1;
    int clean_filter = 1;

    if (!has_uncore_lock(cpu_id))
    {
        return 0;
    }

    flags = (1ULL<<22)|(1ULL<<20);
    flags |= event->eventId;
    if ((event->umask > 0x00) && (event->umask <= 0x3))
    {
        flags |= (event->umask << 14);
    }
    else if (event->umask == 0xFF)
    {
        flags = (1ULL<<21);
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
                case EVENT_OPTION_OCCUPANCY:
                    flags |= ((event->options[j].value & 0x3ULL)<<14);
                    break;
                case EVENT_OPTION_OCCUPANCY_FILTER:
                    clean_filter = 0;
                    VERBOSEPRINTREG(cpu_id, filter, (event->options[j].value & 0xFFFFFFFFULL), SETUP_WBOX_FILTER);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, filter, (event->options[j].value & 0xFFFFFFFFULL)));
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

    if (clean_filter)
    {
        VERBOSEPRINTREG(cpu_id, filter, (event->options[j].value & 0xFFFFFFFFULL), CLEAN_WBOX_FILTER);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, filter, 0x0ULL));
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_WBOX);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int skx_sbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j = 0;
    uint64_t flags = 0x0ULL;
    PciDeviceIndex dev = counter_map[index].device;

    if (!has_uncore_lock(cpu_id))
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
                case EVENT_OPTION_NID:
                    flags |= (event->options[j].value & 0xFULL) << 40;
                    flags |= (1ULL<<45);
                    break;
                case EVENT_OPTION_MATCH0:
                    flags |= (event->options[j].value & 0xFFULL) << 32;
                    break;
                case EVENT_OPTION_MATCH1:
                    flags |= (event->options[j].value & 0x3FF) << 46;
                    break;
                default:
                    break;
            }
        }
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_SBOX_BOX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int skx_uncorebox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    PciDeviceIndex dev = counter_map[index].device;

    if (!has_uncore_lock(cpu_id))
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
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_UNCORE_BOX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int skx_ibox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    PciDeviceIndex dev = counter_map[index].device;

    if (!has_uncore_lock(cpu_id))
    {
        return 0;
    }
    if (!HPMcheck(counter_map[index].device, cpu_id))
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
                    flags |= (event->options[j].value & 0xFFFULL) << 24;
                    break;
                case EVENT_OPTION_MASK0:
                    flags |= (event->options[j].value & 0xFFULL) << 36;
                    break;
                case EVENT_OPTION_MASK1:
                    flags |= (event->options[j].value & 0x7ULL) << 44;
                    break;
                default:
                    break;
            }
        }
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_IBOX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}


#define SKL_UNCORE_FREEZE \
    if (haveLock && MEASURE_UNCORE(eventSet)) \
    { \
        if (IS_SKYLAKE_CLIENT(cpuid_info.model)) \
        { \
            VERBOSEPRINTREG(cpu_id, MSR_V4_UNC_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_UNCORE) \
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_CTRL, 0x0ULL)); \
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_STATUS, 0x0ULL)); \
        } \
        else if (cpuid_info.model == SKYLAKEX) \
        { \
            VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_CTL, (1ULL<<63), FREEZE_UNCORE) \
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_V3_U_PMON_GLOBAL_CTL, (1ULL<<63))); \
        } \
    }

#define SKL_UNCORE_UNFREEZE \
    if (haveLock && MEASURE_UNCORE(eventSet)) \
    { \
        if (IS_SKYLAKE_CLIENT(cpuid_info.model)) \
        { \
            VERBOSEPRINTREG(cpu_id, MSR_V4_UNC_PERF_GLOBAL_CTRL, uflags|(1ULL<<29), UNFREEZE_UNCORE) \
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_CTRL, uflags|(1ULL<<29))); \
        } \
        else if (cpuid_info.model == SKYLAKEX) \
        { \
            VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_CTL, (1ULL<<61), UNFREEZE_UNCORE) \
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_V3_U_PMON_GLOBAL_CTL, (1ULL<<61))); \
        } \
    }

int perfmon_setupCounterThread_skylake(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t fixed_flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    haveLock = has_uncore_lock(cpu_id);

    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, 0xC00000070000000F));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL));
    }
    SKL_UNCORE_FREEZE;

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
            case PMC:
                if ((getCounterTypeOffset(index) == 3) &&
                    (cpuid_info.featureFlags & (1ULL<<RTM)) &&
                    (cpuid_info.model == SKYLAKEX) &&
                    (cpuid_info.stepping < 5))
                {
                    uint64_t flags = 0x0;
                    HPMread(cpu_id, MSR_DEV, TSX_FORCE_ABORT, &flags);
                    if ((flags & 0x1) == 0)
                    {
                        fprintf(stderr, "Warning: Counter PMC3 cannot be used if Restricted Transactional Memory feature is enabled and\n");
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5,2,0)
                        fprintf(stderr, "         bit 0 of register TSX_FORCE_ABORT is 0. As workaround enable\n");
                        fprintf(stderr, "         allow_tsx_force_abort in /sys/devices/cpu/\n");
#else
                        fprintf(stderr, "         bit 0 of register TSX_FORCE_ABORT is 0. As workaround write 0x1 to TSX_FORCE_ABORT:\n");
                        fprintf(stderr, "         sudo wrmsr 0x10f 0x1\n");
#endif
                        eventSet->events[i].type = NOTYPE;
                        continue;
                    }
                    else
                    {
                        skl_pmc_setup(cpu_id, index, event);
                    }
                }
                else
                {
                    skl_pmc_setup(cpu_id, index, event);
                }
                break;

            case FIXED:
                fixed_flags |= skl_fixed_setup(cpu_id, index, event);
                break;

            case POWER:
            case IBOX0FIX:
            case IBOX1FIX:
            case IBOX2FIX:
            case IBOX3FIX:
            case IBOX4FIX:
            case IBOX5FIX:
                break;
            case UBOXFIX:
                if (haveLock)
                {
                    uint64_t uflags = 0x0ULL;
                    uflags |= (1ULL<<20)|(1ULL<<22);
                    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, uflags, SETUP_UBOXFIX)
                    HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, uflags);
                }
                break;
            case UBOX:
                skl_ubox_setup(cpu_id, index, event);
                break;
            case WBOX:
                skx_wbox_setup(cpu_id, index, event);
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
                skylake_cbox_setup(cpu_id, index, event);
                break;
            case MBOX0:
                if (haveLock && !cpuid_info.supportClientmem)
                {
                    skx_mbox_setup(cpu_id, index, event);
                }
                break;
            case MBOX1:
            case MBOX2:
            case MBOX3:
            case MBOX4:
            case MBOX5:
                if (haveLock)
                {
                    skx_mbox_setup(cpu_id, index, event);
                }
                break;
            case MBOX0FIX:
            case MBOX1FIX:
            case MBOX2FIX:
            case MBOX3FIX:
            case MBOX4FIX:
            case MBOX5FIX:
                skx_mboxfix_setup(cpu_id, index, event);
                break;
            case SBOX0:
            case SBOX1:
            case SBOX2:
                skx_sbox_setup(cpu_id, index, event);
                break;
            case BBOX0:
            case BBOX1:
            case RBOX0:
            case RBOX1:
            case RBOX2:
            case EUBOX0:
            case EUBOX1:
            case EUBOX2:
            case EUBOX3:
            case EUBOX4:
            case EUBOX5:
                skx_uncorebox_setup(cpu_id, index, event);
                break;
            case IBOX0:
            case IBOX1:
            case IBOX2:
            case IBOX3:
            case IBOX4:
            case IBOX5:
                skx_ibox_setup(cpu_id, index, event);
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

int perfmon_startCountersThread_skylake(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t uflags = 0x0ULL;
    uint64_t tmp = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    haveLock = has_uncore_lock(cpu_id);

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

                case PERF:
                    tmp = 0x0ULL;
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                    eventSet->events[i].threadCounter[thread_id].startData = field64(tmp, 0, box_map[type].regWidth);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST tmp, READ_PERF)
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
                case UBOXFIX:
                    if (haveLock)
                    {
                        VERBOSEPRINTREG(cpu_id, counter1, 0x0ULL, CLEAR_UBOXFIX)
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    }
                    break;
                case UBOX:
                    if (haveLock)
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
                    if (haveLock && IS_SKYLAKE_CLIENT(cpuid_info.model))
                    {
                        uflags |= (1ULL<<(type-CBOX0));
                    }
                    else if (haveLock && cpuid_info.model == SKYLAKEX)
                    {
                        VERBOSEPRINTREG(cpu_id, counter1, 0x0ULL, CLEAR_CBOX)
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    }
                    break;
                case MBOX0:
                    if (haveLock)
                    {
                        if (!cpuid_info.supportClientmem)
                        {
                            VERBOSEPRINTREG(cpu_id, counter1, 0x0ULL, CLEAR_BOX)
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                        }
                        else
                        {
                            CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                            eventSet->events[i].threadCounter[thread_id].startData = field64(tmp, 0, box_map[type].regWidth);
                            VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST tmp, READ_MBOX)
                        }
                    }
                    break;
                case WBOX:
                case MBOX1:
                case MBOX2:
                case MBOX3:
                case MBOX4:
                case MBOX5:
                case MBOX0FIX:
                case MBOX1FIX:
                case MBOX2FIX:
                case MBOX3FIX:
                case MBOX4FIX:
                case MBOX5FIX:
                case BBOX0:
                case BBOX1:
                case SBOX0:
                case SBOX1:
                case SBOX2:
                case RBOX0:
                case RBOX1:
                case RBOX2:
                case IBOX0:
                case IBOX1:
                case IBOX2:
                case IBOX3:
                case IBOX4:
                case IBOX5:
                case EUBOX0:
                case EUBOX1:
                case EUBOX2:
                case EUBOX3:
                case EUBOX4:
                case EUBOX5:
                case IBOX0FIX:
                case IBOX1FIX:
                case IBOX2FIX:
                case IBOX3FIX:
                case IBOX4FIX:
                case IBOX5FIX:
                    if (haveLock)
                    {
                        VERBOSEPRINTREG(cpu_id, counter1, 0x0ULL, CLEAR_BOX)
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    }
                    break;
                case WBOX0FIX:
                    if (haveLock)
                    {
                        tmp = 0x0ULL;
                        VERBOSEPRINTREG(cpu_id, counter1, 0x0ULL, START_WBOXFIX)
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                        eventSet->events[i].threadCounter[thread_id].startData = field64(tmp, 0, box_map[type].regWidth);
                    }
                    break;
                default:
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = eventSet->events[i].threadCounter[thread_id].startData;
        }
    }

    SKL_UNCORE_UNFREEZE;

    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, LLU_CAST (1ULL<<63)|(1ULL<<62)|flags, CLEAR_PMC_AND_FIXED_OVERFLOW)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<63)|(1ULL<<62)|flags));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, UNFREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, flags));
    }

    return 0;
}


#define SKL_CHECK_CORE_OVERFLOW(offset) \
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

#define SKL_CHECK_UNCORE_OVERFLOW(offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t ovf_values = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_STATUS, &ovf_values)); \
        if (ovf_values & (1ULL<<(offset))) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
        } \
    }

#define SKL_CHECK_LOCAL_OVERFLOW \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t ovf_values = 0x0ULL; \
        uint64_t offset = getCounterTypeOffset(eventSet->events[i].index); \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, box_map[eventSet->events[i].type].statusRegister, &ovf_values)); \
        if (ovf_values & (1ULL<<(offset))) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[eventSet->events[i].type].statusRegister, (1ULL<<(offset)))); \
        } \
    }

int skl_uncore_read(int cpu_id, RegisterIndex index, PerfmonEvent *event,
                     uint64_t* cur_result, int* overflows, int flags,
                     int global_offset, int box_offset)
{
    uint64_t result = 0x0ULL;
    uint64_t tmp = 0x0ULL;
    RegisterType type = counter_map[index].type;
    PciDeviceIndex dev = counter_map[index].device;
    uint64_t counter1 = counter_map[index].counterRegister;
    event++;
    int haveLock = has_uncore_lock(cpu_id);

    CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter1, &result));
    VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST result, READ_REG_1);
    if (flags & FREEZE_FLAG_CLEAR_CTR)
    {
        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0U, CLEAR_REG_1);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0U));
    }
    //printf("%s: %lu\n", counter_map[index].key, result);
    result = field64(result, 0, box_map[type].regWidth);
    //printf("%s: %lu\n", counter_map[index].key, result);
    if (result < *cur_result)
    {
        uint64_t ovf_values = 0x0ULL;
        //int global_offset = box_map[type].ovflOffset;
        int test_local = 0;
        uint32_t global_status_reg = MSR_UNC_V3_U_PMON_GLOBAL_STATUS;
        if (cpuid_info.model != SKYLAKEX)
        {
            global_status_reg = MSR_V4_UNC_PERF_GLOBAL_STATUS;
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

int perfmon_stopCountersThread_skylake(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    haveLock = has_uncore_lock(cpu_id);

    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    SKL_UNCORE_FREEZE;

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
            int ovf_offset = box_map[type].ovflOffset;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    SKL_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
                    break;

                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    SKL_CHECK_CORE_OVERFLOW(index+32);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED)
                    break;

                case PERF:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PERF)
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

                case UBOXFIX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        SKL_CHECK_UNCORE_OVERFLOW(box_map[type].ovflOffset);
                    }
                    break;
                case UBOX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        SKL_CHECK_UNCORE_OVERFLOW(box_map[type].ovflOffset);
                    }
                    break;
                case WBOX0FIX:
                case IBOX0FIX:
                case IBOX1FIX:
                case IBOX2FIX:
                case IBOX3FIX:
                case IBOX4FIX:
                case IBOX5FIX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_UNC_FIXED_CTR);
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
                case WBOX:
                    if (haveLock && IS_SKYLAKE_CLIENT(cpuid_info.model))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        SKL_CHECK_UNCORE_OVERFLOW(box_map[type].ovflOffset);
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_CBOX)
                    }
                    else if (haveLock && cpuid_info.model == SKYLAKEX)
                    {
                        skl_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_CBOX)
                    }
                    break;
                case MBOX0:
                case MBOX0TMP:
                    if (haveLock)
                    {
                        if (!cpuid_info.supportClientmem)
                        {
                            skl_uncore_read(cpu_id, index, event, current, overflows,
                                            0, ovf_offset, getCounterTypeOffset(index)+1);
                            counter_result = *current;
                        }
                        else
                        {
                            CHECK_MSR_READ_ERROR(HPMread(cpu_id, PCI_IMC_DEVICE_0_CH_0, counter1, &counter_result));
                            if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                            {
                                VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, OVERFLOW_CLIENTMEM)
                                eventSet->events[i].threadCounter[thread_id].overflows++;
                            }
                        }
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_MBOX)
                    }
                    break;
                case MBOX1:
                case MBOX2:
                case MBOX3:
                case MBOX4:
                case MBOX5:
                    if (haveLock)
                    {
                        skl_uncore_read(cpu_id, index, event, current, overflows,
                                        0, ovf_offset, getCounterTypeOffset(index)+1);
                        counter_result = *current;
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_MBOX)
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
                    if (haveLock)
                    {
                        skl_uncore_read(cpu_id, index, event, current, overflows,
                                        FREEZE_FLAG_CLEAR_CTR, ovf_offset, 0);
                        counter_result = *current;
                    }
                    break;
                case BBOX0:
                case BBOX1:
                case SBOX0:
                case SBOX1:
                case SBOX2:
                case RBOX0:
                case RBOX1:
                case RBOX2:
                case IBOX0:
                case IBOX1:
                case IBOX2:
                case IBOX3:
                case IBOX4:
                case IBOX5:
                case EUBOX0:
                case EUBOX1:
                case EUBOX2:
                case EUBOX3:
                case EUBOX4:
                case EUBOX5:
                    if (haveLock)
                    {
                        skl_uncore_read(cpu_id, index, event, current, overflows,
                                        FREEZE_FLAG_CLEAR_CTR, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_BBOX)
                    }
                    break;
                default:
                    break;
            }
            *current = field64(counter_result, 0, box_map[type].regWidth);
        }
    }
    if ((haveLock) && MEASURE_UNCORE(eventSet))
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_STATUS, &counter_result));
        if (counter_result != 0x0ULL)
        {
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_V4_UNC_PERF_GLOBAL_STATUS, counter_result));
        }
    }


    return 0;
}


int perfmon_readCountersThread_skylake(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags = 0x0ULL;
    uint64_t uflags = 0x0ULL;
    int haveLock = 0;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    haveLock = has_uncore_lock(cpu_id);

    if (MEASURE_CORE(eventSet))
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, &flags));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, SAFE_PMC_FLAGS)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, RESET_PMC_FLAGS)
    }

    SKL_UNCORE_FREEZE;

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
            int ovf_offset = box_map[type].ovflOffset;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    SKL_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
                    break;

                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    SKL_CHECK_CORE_OVERFLOW(index+32);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED)
                    break;

                case PERF:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PERF)
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
                    eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
                    break;

                case UBOXFIX:
                case WBOX0FIX:
                case IBOX0FIX:
                case IBOX1FIX:
                case IBOX2FIX:
                case IBOX3FIX:
                case IBOX4FIX:
                case IBOX5FIX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        SKL_CHECK_UNCORE_OVERFLOW(box_map[type].ovflOffset);
                    }
                    break;
                case UBOX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        SKL_CHECK_UNCORE_OVERFLOW(box_map[type].ovflOffset);
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
                    if (haveLock && IS_SKYLAKE_CLIENT(cpuid_info.model))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        SKL_CHECK_UNCORE_OVERFLOW(box_map[type].ovflOffset);
                        *current = field64(counter_result, 0, box_map[type].regWidth);
                        uflags |= (1ULL<<(type-CBOX0));
                    }
                    else if (haveLock && cpuid_info.model == SKYLAKEX)
                    {
                        skl_uncore_read(cpu_id, index, event, current, overflows,
                                    0, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                    }
                    break;
                case MBOX0:
                case MBOX0TMP:
                    if (haveLock)
                    {
                        if (!cpuid_info.supportClientmem)
                        {
                            skl_uncore_read(cpu_id, index, event, current, overflows,
                                            0, ovf_offset, getCounterTypeOffset(index)+1);
                            counter_result = *current;
                        }
                        else
                        {
                            CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                            if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                            {
                                VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, OVERFLOW_CLIENTMEM)
                                eventSet->events[i].threadCounter[thread_id].overflows++;
                            }
                        }
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_MBOX)
                    }
                    break;
                case MBOX1:
                case MBOX2:
                case MBOX3:
                case MBOX4:
                case MBOX5:
                    if (haveLock)
                    {
                        skl_uncore_read(cpu_id, index, event, current, overflows,
                                        0, ovf_offset, getCounterTypeOffset(index)+1);
                        counter_result = *current;
                    }
                    break;
                case MBOX0FIX:
                case MBOX1FIX:
                case MBOX2FIX:
                case MBOX3FIX:
                case MBOX4FIX:
                case MBOX5FIX:
                    if (haveLock)
                    {
                        skl_uncore_read(cpu_id, index, event, current, overflows,
                                        0, ovf_offset, 0);
                        counter_result = *current;
                    }
                    break;
                case BBOX0:
                case BBOX1:
                case SBOX0:
                case SBOX1:
                case SBOX2:
                case RBOX0:
                case RBOX1:
                case RBOX2:
                case IBOX0:
                case IBOX1:
                case IBOX2:
                case IBOX3:
                case IBOX4:
                case IBOX5:
                case EUBOX0:
                case EUBOX1:
                case EUBOX2:
                case EUBOX3:
                case EUBOX4:
                case EUBOX5:
                case WBOX:
                    if (haveLock)
                    {
                        skl_uncore_read(cpu_id, index, event, current, overflows,
                                        0, ovf_offset, getCounterTypeOffset(index));
                        counter_result = *current;
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_BBOX)
                    }
                    break;
                default:
                    break;
            }
            *current = field64(counter_result, 0, box_map[type].regWidth);
        }
    }

    SKL_UNCORE_UNFREEZE;

    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, RESTORE_PMC_FLAGS)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, flags));
    }

    return 0;
}

int perfmon_finalizeCountersThread_skylake(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int haveTileLock = 0;
    int clearPBS = 0;
    uint64_t ovf_values_core = (1ULL<<63)|(1ULL<<62);
    uint64_t ovf_values_uncore = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    haveLock = has_uncore_lock(cpu_id);
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
            case PMC:
                ovf_values_core |= (1ULL<<(index-cpuid_info.perf_num_fixed_ctr));
                if ((haveTileLock) && (eventSet->events[i].event.eventId == 0xB7))
                {
                    VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, 0x0ULL, CLEAR_PMC_OFFCORE0);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, 0x0ULL));
                }
                else if ((haveTileLock) && (eventSet->events[i].event.eventId == 0xBB))
                {
                    VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP1, 0x0ULL, CLEAR_PMC_OFFCORE1);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP1, 0x0ULL));
                }
                else if (eventSet->events[i].event.eventId == 0xCD)
                {
                    VERBOSEPRINTREG(cpu_id, MSR_PEBS_LD_LAT, 0x0ULL, CLEAR_PMC_LATENCY);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_LD_LAT, 0x0ULL));
                }
                else if (eventSet->events[i].event.eventId == 0xC6)
                {
                    VERBOSEPRINTREG(cpu_id, MSR_V4_PEBS_FRONTEND, 0x0ULL, CLEAR_PMC_FRONTEND);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_V4_PEBS_FRONTEND, 0x0ULL));
                }
                break;
            case FIXED:
                ovf_values_core |= (1ULL<<(index+32));
                break;
            default:
                break;
        }
        if ((reg) && (((type == PMC)||(type == FIXED))|| ((type >= UNCORE) && (haveLock))))
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
    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        uint32_t status_reg = MSR_V4_UNC_PERF_GLOBAL_STATUS;
        uint32_t ctrl_reg = MSR_V4_UNC_PERF_GLOBAL_CTRL;
        if (cpuid_info.model == SKYLAKEX)
        {
            status_reg = MSR_UNC_V3_U_PMON_GLOBAL_STATUS;
            ctrl_reg = MSR_UNC_V3_U_PMON_GLOBAL_CTL;
        }
        VERBOSEPRINTREG(cpu_id, status_reg, LLU_CAST 0x0ULL, CLEAR_UNCORE_STATUS)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, status_reg, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, ctrl_reg, LLU_CAST 0x0ULL, CLEAR_UNCORE_CTRL)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, ctrl_reg, 0x0ULL));
        for (int i=UNCORE;i<NUM_UNITS;i++)
        {
            if (TESTTYPE(eventSet, i) && box_map[i].ctrlRegister != 0x0)
            {
                VERBOSEPRINTPCIREG(cpu_id, box_map[i].device, box_map[i].ctrlRegister, 0x0ULL, CLEAR_UNCORE_BOX_CTRL);
                HPMwrite(cpu_id, box_map[i].device, box_map[i].ctrlRegister, 0x0ULL);
                if (box_map[i].filterRegister1)
                {
                    VERBOSEPRINTPCIREG(cpu_id, box_map[i].device, box_map[i].filterRegister1, 0x0ULL, CLEAR_FILTER);
                    HPMwrite(cpu_id, box_map[i].device, box_map[i].filterRegister1, 0x0ULL);
                }
                if (box_map[i].filterRegister2)
                {
                    VERBOSEPRINTPCIREG(cpu_id, box_map[i].device, box_map[i].filterRegister2, 0x0ULL, CLEAR_FILTER);
                    HPMwrite(cpu_id, box_map[i].device, box_map[i].filterRegister2, 0x0ULL);
                }
            }
        }
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
