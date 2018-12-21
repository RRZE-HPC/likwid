/*
 * =======================================================================================
 *
 *      Filename:  perfmon_broadwell.h
 *
 *      Description:  Header File of perfmon module for Intel Broadwell.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Roehl (tr), thomas.roehl@googlemail.com
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

#include <perfmon_broadwell_events.h>
#include <perfmon_broadwell_counters.h>
#include <perfmon_broadwelld_events.h>
#include <perfmon_broadwelld_counters.h>
#include <perfmon_broadwellEP_events.h>
#include <perfmon_broadwellEP_counters.h>
#include <error.h>
#include <affinity.h>
#include <limits.h>
#include <topology.h>
#include <access.h>


static int perfmon_numCountersBroadwell = NUM_COUNTERS_BROADWELL;
static int perfmon_numCoreCountersBroadwell = NUM_COUNTERS_CORE_BROADWELL;
static int perfmon_numArchEventsBroadwell = NUM_ARCH_EVENTS_BROADWELL;

static int perfmon_numCountersBroadwellD = NUM_COUNTERS_BROADWELLD;
static int perfmon_numCoreCountersBroadwellD = NUM_COUNTERS_CORE_BROADWELLD;
static int perfmon_numArchEventsBroadwellD = NUM_ARCH_EVENTS_BROADWELLD;

static int perfmon_numCountersBroadwellEP = NUM_COUNTERS_BROADWELLEP;
static int perfmon_numCoreCountersBroadwellEP = NUM_COUNTERS_CORE_BROADWELLEP;
static int perfmon_numArchEventsBroadwellEP = NUM_ARCH_EVENTS_BROADWELLEP;

static int bdw_did_cbox_check = 0;
int bdw_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event);
int bdwep_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event);
int (*broadwell_cbox_setup)(int, RegisterIndex, PerfmonEvent *);
int print_ht_warn_once = 1;

int bdw_cbox_nosetup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    cpu_id++;
    index++;
    event++;
    return 0;
}

int perfmon_init_broadwell(int cpu_id)
{
    int ret;
    uint64_t data;
    lock_acquire((int*) &tile_lock[affinity_thread2core_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &socket_lock[affinity_thread2socket_lookup[cpu_id]], cpu_id);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL));
    if ((cpuid_info.model == BROADWELL_E) || (cpuid_info.model == BROADWELL_D))
    {
        broadwell_cbox_setup = bdwep_cbox_setup;
        bdw_did_cbox_check = 1;
    }
    else if ((cpuid_info.model == BROADWELL || cpuid_info.model == BROADWELL_E3) &&
             socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id &&
             bdw_did_cbox_check == 0)
    {
        ret = HPMwrite(cpu_id, MSR_DEV, MSR_UNC_CBO_0_PERFEVTSEL0, 0x0ULL);
        ret += HPMread(cpu_id, MSR_DEV, MSR_UNCORE_PERF_GLOBAL_CTRL, &data);
        ret += HPMwrite(cpu_id, MSR_DEV, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL);
        ret += HPMread(cpu_id, MSR_DEV, MSR_UNC_CBO_0_PERFEVTSEL0, &data);
        if ((ret == 0) && (data == 0x0ULL))
            broadwell_cbox_setup = bdw_cbox_setup;
        else
            broadwell_cbox_setup = bdw_cbox_nosetup;
        bdw_did_cbox_check = 1;
    }
    return 0;
}


uint32_t bdw_fixed_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
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

int bdw_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    uint64_t offcore_flags = 0x0ULL;

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

    if (getCounterTypeOffset(index) >= 4)
    {
        if (print_ht_warn_once)
        {
            fprintf(stderr, "WARNING: PMC4-7 on Intel Broadwell systems requires KERNEL option to work\n");
            fprintf(stderr, "         properly. The KERNEL option is added automatically for PMC4-7.\n");
            fprintf(stderr, "         Be aware that the events also count during kernel execution and may\n");
            fprintf(stderr, "         be higher than expected.\n");
            print_ht_warn_once = 0;
        }
        flags |= (1ULL<<17);
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
                    offcore_flags |= (event->options[j].value<<16);
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

int bdw_ubox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
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

int bdw_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
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

int bdwep_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    uint64_t filter_flags0 = 0x0ULL;
    uint64_t filter_flags1 = 0x0ULL;
    uint32_t filter0 = box_map[counter_map[index].type].filterRegister1;
    uint32_t filter1 = box_map[counter_map[index].type].filterRegister2;
    int set_state_all = 0;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
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
                    filter_flags1 |= (0x3<<27);
                    filter_flags1 |= (extractBitField(event->options[j].value,5,0) << 20);
                    break;
                case EVENT_OPTION_NID:
                    filter_flags1 |= (extractBitField(event->options[j].value,16,0));
                    break;
                case EVENT_OPTION_STATE:
                    filter_flags0 |= (extractBitField(event->options[j].value,7,0) << 17);
                    set_state_all = 0;
                    break;
                case EVENT_OPTION_TID:
                    filter_flags0 |= (extractBitField(event->options[j].value,6,0));
                    flags |= (1ULL<<19);
                    break;
                case EVENT_OPTION_MATCH0:
                    filter_flags1 |= (extractBitField(event->options[j].value,2,0) << 30);
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
    if (filter_flags1 != 0x0ULL)
    {
        VERBOSEPRINTREG(cpu_id, filter1, filter_flags1, SETUP_CBOX_FILTER1);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, filter1, filter_flags1));
    }
    else
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, filter1, 0x0ULL));
    }

    if (set_state_all)
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, filter0, &filter_flags0));
        filter_flags0 |= (0x1F << 17);
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

int bdw_wbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j = 0;
    uint64_t flags = 0x0ULL;
    uint64_t filter = box_map[counter_map[index].type].filterRegister1;
    int clean_filter = 1;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
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

int bdw_bbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    uint64_t filter = 0x0ULL;
    int opcode_flag = 0;
    int match_flag = 0;
    PciDeviceIndex dev = counter_map[index].device;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!HPMcheck(dev, cpu_id))
    {
        return -ENODEV;
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
                    flags |= (event->options[j].value & 0xFFULL) << 24;
                    break;
                case EVENT_OPTION_OPCODE:
                    VERBOSEPRINTPCIREG(cpu_id, dev, PCI_UNC_HA_PMON_OPCODEMATCH,
                                        (event->options[j].value & 0x3FULL), SETUP_BBOX_OPCODE);
                    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, PCI_UNC_HA_PMON_OPCODEMATCH,
                                        (event->options[j].value & 0x3FULL)));
                    opcode_flag = 1;
                    break;
                case EVENT_OPTION_MATCH0:
                    filter = ((event->options[j].value & 0xFFFFFFC0ULL));
                    VERBOSEPRINTPCIREG(cpu_id, dev, PCI_UNC_HA_PMON_ADDRMATCH0, filter, SETUP_ADDR0_FILTER);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, PCI_UNC_HA_PMON_ADDRMATCH0, filter));
                    filter = (((event->options[j].value>>32) & 0x3FFFULL));
                    VERBOSEPRINTPCIREG(cpu_id, dev, PCI_UNC_HA_PMON_ADDRMATCH1, filter, SETUP_ADDR1_FILTER);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, PCI_UNC_HA_PMON_ADDRMATCH1, filter));
                    match_flag = 1;
                    break;
                default:
                    break;
            }
        }
    }
    if (!opcode_flag)
    {
        VERBOSEPRINTPCIREG(cpu_id, dev, PCI_UNC_HA_PMON_OPCODEMATCH, 0x0ULL, CLEAR_BBOX_OPCODE);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, PCI_UNC_HA_PMON_OPCODEMATCH, 0x0ULL));
    }
    if (!match_flag)
    {
        VERBOSEPRINTPCIREG(cpu_id, dev, PCI_UNC_HA_PMON_ADDRMATCH0, 0x0ULL, CLEAR_BBOX_MATCH0);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, PCI_UNC_HA_PMON_ADDRMATCH0, 0x0ULL));
        VERBOSEPRINTPCIREG(cpu_id, dev, PCI_UNC_HA_PMON_ADDRMATCH1, 0x0ULL, CLEAR_BBOX_MATCH1);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, PCI_UNC_HA_PMON_ADDRMATCH1, 0x0ULL));
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_BBOX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int bdw_mbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
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

int bdw_mboxfix_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
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
    if (event->numberOfOptions > 0)
    {
        for(j = 0; j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_INVERT:
                    flags |= (1ULL<<23);
                    break;
                default:
                    break;
            }
        }
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].configRegister, flags, SETUP_MBOX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int bdw_ibox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    PciDeviceIndex dev = counter_map[index].device;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
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
                    flags |= (event->options[j].value & 0xFFULL) << 24;
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

int bdw_pbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    PciDeviceIndex dev = counter_map[index].device;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!HPMcheck(counter_map[index].device, cpu_id))
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
                case EVENT_OPTION_THRESHOLD:
                    flags |= (event->options[j].value & 0xFFULL) << 24;
                    break;
                default:
                    break;
            }
        }
    }
    if ((flags|(1ULL<<22)) != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].configRegister, flags, SETUP_PBOX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        /* Intel notes the registers must be written twice to hold, once without enable and again with enable */
        flags |= (1ULL<<22);
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_PBOX_TWICE);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int bdw_rbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    PciDeviceIndex dev = counter_map[index].device;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!HPMcheck(counter_map[index].device, cpu_id))
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
                case EVENT_OPTION_THRESHOLD:
                    flags |= (event->options[j].value & 0xFFULL) << 24;
                    break;
                default:
                    break;
            }
        }
    }
    if ((flags|(1ULL<<22)) != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].configRegister, flags, SETUP_PBOX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        /* Intel notes the registers must be written twice to hold, once without enable and again with enable */
        flags |= (1ULL<<22);
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_PBOX_TWICE);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int bdw_sbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!HPMcheck(counter_map[index].device, cpu_id))
    {
        return -ENODEV;
    }

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
                    flags |= (1ULL<<19);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= ((event->options[j].value & 0xFFULL)<<24);
                    break;

                default:
                    break;
            }
        }
    }
    if ((flags|(1ULL<<22)) != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_SBOX);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, counter_map[index].device, counter_map[index].configRegister, flags));
        flags |= (1ULL<<22);
        /* Due to an issue found with the Intel® Xeon® Processor E5 and E7 v4 Product Families
         * hardware, it will be necessary to write each control register twice in a row in order for
         * the Event Select field to take hold. It is recommended that SW perform the first write
         * with the enable bit set to 0 followed by a write of the same control register value but
         * with the enable bit set to 1.*/
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_SBOX_TWICE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, counter_map[index].device, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int bdw_qbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event, PciDeviceIndex filterdev)
{
    int j;
    uint64_t flags = 0x0ULL;
    uint64_t filterreg;
    uint64_t filterval = 0x0ULL;
    PciDeviceIndex dev = counter_map[index].device;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!HPMcheck(counter_map[index].device, cpu_id))
    {
        return -ENODEV;
    }

    flags = (1ULL<<20)|(1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;
    if (event->cfgBits == 0x01)
    {
        flags |= (1ULL<<21);
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
                    if (HPMcheck(filterdev, cpu_id))
                    {
                        filterreg = PCI_UNC_V3_QPI_PMON_RX_MATCH_0;
                        filterval = event->options[j].value & 0x8003FFF8ULL;
                        VERBOSEPRINTPCIREG(cpu_id, filterdev, filterreg, filterval, SETUP_QBOX_RX_MATCH0);
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, filterreg, filterval));
                    }
                    else
                    {
                        DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, Filtering for counter %s cannot be applied. PCI device not available, counter_map[index].key);
                    }
                    break;
                case EVENT_OPTION_MATCH1:
                    if (HPMcheck(filterdev, cpu_id))
                    {
                        filterreg = PCI_UNC_V3_QPI_PMON_RX_MATCH_1;
                        filterval = event->options[j].value & 0x000F000FULL;
                        VERBOSEPRINTPCIREG(cpu_id, filterdev, filterreg, filterval, SETUP_QBOX_RX_MATCH1);
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, filterreg, filterval));
                    }
                    else
                    {
                        DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, Filtering for counter %s cannot be applied. PCI device not available, counter_map[index].key);
                    }
                    break;
                case EVENT_OPTION_MATCH2:
                    if (HPMcheck(filterdev, cpu_id))
                    {
                        filterreg = PCI_UNC_V3_QPI_PMON_TX_MATCH_0;
                        filterval = event->options[j].value & 0x8003FFF8ULL;
                        VERBOSEPRINTPCIREG(cpu_id, filterdev, filterreg, filterval, SETUP_QBOX_TX_MATCH0);
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, filterreg, filterval));
                    }
                    else
                    {
                        DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, Filtering for counter %s cannot be applied. PCI device not available, counter_map[index].key);
                    }
                    break;
                case EVENT_OPTION_MATCH3:
                    if (HPMcheck(filterdev, cpu_id))
                    {
                        filterreg = PCI_UNC_V3_QPI_PMON_TX_MATCH_1;
                        filterval = event->options[j].value & 0x000F000FULL;
                        VERBOSEPRINTPCIREG(cpu_id, filterdev, filterreg, filterval, SETUP_QBOX_TX_MATCH1);
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, filterreg, filterval));
                    }
                    else
                    {
                        DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, Filtering for counter %s cannot be applied. PCI device not available, counter_map[index].key);
                    }
                    break;
                case EVENT_OPTION_MASK0:
                    if (HPMcheck(filterdev, cpu_id))
                    {
                        filterreg = PCI_UNC_V3_QPI_PMON_RX_MASK_0;
                        filterval = event->options[j].value & 0x8003FFF8ULL;
                        VERBOSEPRINTPCIREG(cpu_id, filterdev, filterreg, filterval, SETUP_QBOX_RX_MASK0);
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, filterreg, filterval));
                    }
                    else
                    {
                        DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, Filtering for counter %s cannot be applied. PCI device not available, counter_map[index].key);
                    }
                    break;
                case EVENT_OPTION_MASK1:
                    if (HPMcheck(filterdev, cpu_id))
                    {
                        filterreg = PCI_UNC_V3_QPI_PMON_RX_MASK_1;
                        filterval = event->options[j].value & 0x000F000FULL;
                        VERBOSEPRINTPCIREG(cpu_id, filterdev, filterreg, filterval, SETUP_QBOX_RX_MASK1);
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, filterreg, filterval));
                    }
                    else
                    {
                        DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, Filtering for counter %s cannot be applied. PCI device not available, counter_map[index].key);
                    }
                    break;
                case EVENT_OPTION_MASK2:
                    if (HPMcheck(filterdev, cpu_id))
                    {
                        filterreg = PCI_UNC_V3_QPI_PMON_TX_MASK_0;
                        filterval = event->options[j].value & 0x8003FFF8ULL;
                        VERBOSEPRINTPCIREG(cpu_id, filterdev, filterreg, filterval, SETUP_QBOX_TX_MASK0);
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, filterreg, filterval));
                    }
                    else
                    {
                        DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, Filtering for counter %s cannot be applied. PCI device not available, counter_map[index].key);
                    }
                    break;
                case EVENT_OPTION_MASK3:
                    if (HPMcheck(filterdev, cpu_id))
                    {
                        filterreg = PCI_UNC_V3_QPI_PMON_TX_MASK_0;
                        filterval = event->options[j].value & 0x000F000FULL;
                        VERBOSEPRINTPCIREG(cpu_id, filterdev, filterreg, filterval, SETUP_QBOX_TX_MASK1);
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, filterreg, filterval));
                    }
                    else
                    {
                        DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, Filtering for counter %s cannot be applied. PCI device not available, counter_map[index].key);
                    }
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

#define BDW_FREEZE_UNCORE \
    if (haveLock && MEASURE_UNCORE(eventSet) && (cpuid_info.model == BROADWELL_E || cpuid_info.model == BROADWELL_D)) \
    { \
        VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<31), FREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_V3_U_PMON_GLOBAL_CTL, (1ULL<<31))); \
    } \
    else if (haveLock && MEASURE_UNCORE(eventSet) && (cpuid_info.model == BROADWELL || cpuid_info.model == BROADWELL_E3)) \
    { \
        uint64_t data = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_UNCORE_PERF_GLOBAL_CTRL, &data)); \
        if (!(data & (1ULL<<29))) \
        { \
            data &= ~(1ULL<<29); \
            VERBOSEPRINTREG(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, data, FREEZE_UNCORE); \
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNCORE_PERF_GLOBAL_CTRL, data)); \
        } \
    } \

#define BDW_UNFREEZE_UNCORE \
    if (haveLock && MEASURE_UNCORE(eventSet) && (cpuid_info.model == BROADWELL_E || cpuid_info.model == BROADWELL_D)) \
    { \
        VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<29), UNFREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_V3_U_PMON_GLOBAL_CTL, (1ULL<<29))); \
    } \
    else if (haveLock && MEASURE_UNCORE(eventSet) && (cpuid_info.model == BROADWELL || cpuid_info.model == BROADWELL_E3)) \
    { \
        uint64_t data = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_UNCORE_PERF_GLOBAL_CTRL, &data)); \
        data |= (1ULL<<29);\
        VERBOSEPRINTREG(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, data, FREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNCORE_PERF_GLOBAL_CTRL, data)); \
    } \

#define BDW_UNFREEZE_UNCORE_AND_RESET_CTR \
    if (haveLock && MEASURE_UNCORE(eventSet)) \
    { \
        for (int i=0;i < eventSet->numberOfEvents;i++) \
        { \
            RegisterIndex index = eventSet->events[i].index; \
            RegisterType type = eventSet->events[i].type; \
            if ((type < UNCORE) || (type == WBOX0FIX) || (type == NOTYPE)) \
            { \
                continue; \
            } \
            PciDeviceIndex dev = counter_map[index].device; \
            if (HPMcheck(dev, cpu_id)) { \
                VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL, CLEAR_CTR_MANUAL); \
                CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL)); \
                if (counter_map[index].counterRegister2 != 0x0) \
                { \
                    VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].counterRegister2, 0x0ULL, CLEAR_CTR_MANUAL); \
                    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister2, 0x0ULL)); \
                } \
            } \
        } \
        BDW_UNFREEZE_UNCORE; \
    }

int perfmon_setupCounterThread_broadwell(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags;
    uint64_t fixed_flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    print_ht_warn_once = 1;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }
    BDW_FREEZE_UNCORE;
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
        PciDeviceIndex dev = counter_map[index].device;
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        flags = 0x0ULL;
        switch (type)
        {
            case PMC:
                bdw_pmc_setup(cpu_id, index, event);
                break;

            case FIXED:
                fixed_flags |= bdw_fixed_setup(cpu_id, index, event);
                break;

            case POWER:
            case THERMAL:
                break;

            case UBOX:
                bdw_ubox_setup(cpu_id, index, event);
                break;
            case UBOXFIX:
                if (haveLock)
                {
                    flags = (1ULL<<22)|(1ULL<<20);
                    VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_UBOXFIX);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
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
                broadwell_cbox_setup(cpu_id, index, event);
                break;

            case BBOX0:
            case BBOX1:
                bdw_bbox_setup(cpu_id, index, event);
                break;

            case WBOX:
                bdw_wbox_setup(cpu_id, index, event);
                break;
            case WBOX0FIX:
                break;

            case MBOX0:
                if (!cpuid_info.supportClientmem)
                {
                    bdw_mbox_setup(cpu_id, index, event);
                }
                break;
            case MBOX1:
            case MBOX2:
            case MBOX3:
            case MBOX4:
            case MBOX5:
            case MBOX6:
            case MBOX7:
                bdw_mbox_setup(cpu_id, index, event);
                break;
            case MBOX0FIX:
            case MBOX1FIX:
            case MBOX2FIX:
            case MBOX3FIX:
            case MBOX4FIX:
            case MBOX5FIX:
            case MBOX6FIX:
            case MBOX7FIX:
                bdw_mboxfix_setup(cpu_id, index, event);
                break;

            case PBOX:
                bdw_pbox_setup(cpu_id, index, event);
                break;

            case IBOX0:
            case IBOX1:
                bdw_ibox_setup(cpu_id, index, event);
                break;

            case RBOX0:
            case RBOX1:
                bdw_rbox_setup(cpu_id, index, event);
                break;

            case SBOX0:
            case SBOX1:
            case SBOX2:
            case SBOX3:
                bdw_sbox_setup(cpu_id, index, event);
                break;

            case QBOX0:
                bdw_qbox_setup(cpu_id, index, event, PCI_QPI_MASK_DEVICE_PORT_0);
                break;
            case QBOX1:
                bdw_qbox_setup(cpu_id, index, event, PCI_QPI_MASK_DEVICE_PORT_1);
                break;
            case QBOX2:
                bdw_qbox_setup(cpu_id, index, event, PCI_QPI_MASK_DEVICE_PORT_2);
                break;

            default:
                break;
        }
    }

    for (int i=UNCORE;i<NUM_UNITS;i++)
    {
        if (haveLock && TESTTYPE(eventSet, i) && box_map[i].ctrlRegister != 0x0)
        {
            VERBOSEPRINTPCIREG(cpu_id, box_map[i].device, box_map[i].ctrlRegister, 0x0ULL, CLEAR_UNCORE_BOX_CTRL);
            HPMwrite(cpu_id, box_map[i].device, box_map[i].ctrlRegister, 0x0ULL);
        }
    }
    if (fixed_flags > 0x0ULL)
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_FIXED_CTR_CTRL, LLU_CAST fixed_flags, SETUP_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_FIXED_CTR_CTRL, fixed_flags));
    }
    return 0;
}

int perfmon_startCountersThread_broadwell(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
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

                case WBOX0FIX:
                    if (haveLock)
                    {
                        tmp = 0x0ULL;
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST tmp, START_WBOXFIX);
                        eventSet->events[i].threadCounter[thread_id].startData = field64(tmp, 0, box_map[type].regWidth);
                    }
                    break;
                case QBOX0FIX:
                case QBOX1FIX:
                    if (haveLock && HPMcheck(dev, cpu_id))
                    {
                        if (eventSet->events[i].event.eventId != 0x00)
                        {
                            CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                            VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST tmp, START_QBOXFIX);
                            eventSet->events[i].threadCounter[thread_id].startData = field64(tmp, 0, box_map[type].regWidth);
                        }
                    }
                    break;

                default:
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = eventSet->events[i].threadCounter[thread_id].startData;
        }
    }

    BDW_UNFREEZE_UNCORE_AND_RESET_CTR;

    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, LLU_CAST (1ULL<<63)|(1ULL<<62)|flags, CLEAR_PMC_AND_FIXED_OVERFLOW)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<63)|(1ULL<<62)|flags));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, UNFREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, flags));
    }

    return 0;
}

int bdw_uncore_read(int cpu_id, RegisterIndex index, PerfmonEvent *event,
                     uint64_t* cur_result, int* overflows, int flags,
                     int global_offset, int box_offset)
{
    uint64_t result = 0x0ULL;
    uint64_t tmp = 0x0ULL;
    RegisterType type = counter_map[index].type;
    PciDeviceIndex dev = counter_map[index].device;
    uint64_t counter1 = counter_map[index].counterRegister;
    uint64_t counter2 = counter_map[index].counterRegister2;
    event++;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter1, &result));
    VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST result, READ_REG_1);
    if (flags & FREEZE_FLAG_CLEAR_CTR)
    {
        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST 0x0U, CLEAR_PCI_REG_1);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0U));
    }
    if (counter2 != 0x0)
    {
        result <<= 32;
        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter2, &tmp));
        VERBOSEPRINTPCIREG(cpu_id, dev, counter2, LLU_CAST tmp, READ_REG_2);
        result += tmp;
        if (flags & FREEZE_FLAG_CLEAR_CTR)
        {
            VERBOSEPRINTPCIREG(cpu_id, dev, counter2, LLU_CAST 0x0U, CLEAR_PCI_REG_2);
            CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter2, 0x0U));
        }
    }
    result = field64(result, 0, box_map[type].regWidth);
    if (result < *cur_result)
    {
        uint64_t ovf_values = 0x0ULL;
        //int global_offset = box_map[type].ovflOffset;
        int test_local = 0;
        uint32_t global_status_reg = MSR_UNC_V3_U_PMON_GLOBAL_STATUS;
        if (cpuid_info.model == BROADWELL)
            global_status_reg = MSR_UNC_PERF_GLOBAL_STATUS;
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

        if (test_local)
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
    }
    *cur_result = result;
    return 0;
}


#define BDW_CHECK_CORE_OVERFLOW(offset) \
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

#define BDW_CHECK_LOCAL_OVERFLOW \
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

int perfmon_stopCountersThread_broadwell(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t counter_result = 0x0ULL;
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
    BDW_FREEZE_UNCORE;

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
                    BDW_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
                    break;

                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    BDW_CHECK_CORE_OVERFLOW(index+32);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED)
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

                case BBOX0:
                case BBOX1:
                    bdw_uncore_read(cpu_id, index, event, &counter_result, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, ovf_offset, getCounterTypeOffset(index));
                    break;

                case MBOX0:
                    if (!cpuid_info.supportClientmem)
                    {
                        bdw_uncore_read(cpu_id, index, event, &counter_result, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, ovf_offset, getCounterTypeOffset(index)+1);
                    }
                    else
                    {
                        if (haveLock)
                        {
                            uint64_t tmp = 0x0;
                            CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                            eventSet->events[i].threadCounter[thread_id].startData = field64(tmp, 0, box_map[type].regWidth);
                            VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST tmp, READ_MBOX)
                        }
                    }
                    break;
                case MBOX1:
                case MBOX2:
                case MBOX3:
                case MBOX4:
                case MBOX5:
                case MBOX6:
                case MBOX7:
                    bdw_uncore_read(cpu_id, index, event, &counter_result, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, ovf_offset, getCounterTypeOffset(index)+1);
                    break;

                case MBOX0FIX:
                case MBOX1FIX:
                case MBOX2FIX:
                case MBOX3FIX:
                case MBOX4FIX:
                case MBOX5FIX:
                case MBOX6FIX:
                case MBOX7FIX:
                    bdw_uncore_read(cpu_id, index, event, &counter_result, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, ovf_offset, 0);
                    break;

                case IBOX1:
                    bdw_uncore_read(cpu_id, index, event, &counter_result, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, ovf_offset, getCounterTypeOffset(index)+2);
                    break;

                case PBOX:
                case IBOX0:
                case WBOX:
                case UBOX:
                case UBOXFIX:
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
                case RBOX0:
                case RBOX1:
                case SBOX0:
                case SBOX1:
                case SBOX2:
                case SBOX3:
                case QBOX0:
                case QBOX1:
                case QBOX2:
                    bdw_uncore_read(cpu_id, index, event, &counter_result, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, ovf_offset, getCounterTypeOffset(index));
                    break;

                case WBOX0FIX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, STOP_WBOXFIX);
                    }
                    break;

                case QBOX0FIX:
                case QBOX1FIX:
                    if (haveLock)
                    {
                        HPMread(cpu_id, dev, counter1, &counter_result);
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_QBOXFIX)
                        eventSet->events[i].threadCounter[thread_id].startData = 0;
                        if (eventSet->events[i].event.eventId == 0x00)
                        {
                            switch(extractBitField(counter_result, 3, 0))
                            {
                                case 0x2:
                                    counter_result = 5.6E9;
                                    break;
                                case 0x3:
                                    counter_result = 6.4E9;
                                    break;
                                case 0x4:
                                    counter_result = 7.2E9;
                                    break;
                                case 0x5:
                                    counter_result = 8.0E9;
                                    break;
                                case 0x6:
                                    counter_result = 8.8E9;
                                    break;
                                case 0x7:
                                    counter_result = 9.6E9;
                                    break;
                                default:
                                    counter_result = 0;
                                    break;
                            }
                            
                        }
                        else if ((eventSet->events[i].event.eventId == 0x01) ||
                                 (eventSet->events[i].event.eventId == 0x02))
                        {
                            counter_result = field64(counter_result, 0, box_map[type].regWidth);
                        }
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


int perfmon_readCountersThread_broadwell(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags = 0x0ULL;
    int haveLock = 0;
    uint64_t counter_result = 0x0ULL;
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
    BDW_FREEZE_UNCORE;

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
                    BDW_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
                    *current = field64(counter_result, 0, box_map[type].regWidth);
                    break;

                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    BDW_CHECK_CORE_OVERFLOW(index+32);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED)
                    *current = field64(counter_result, 0, box_map[type].regWidth);
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
                        *current = field64(counter_result, 0, box_map[type].regWidth);
                    }
                    break;

                case THERMAL:
                    CHECK_TEMP_READ_ERROR(thermal_read(cpu_id,(uint32_t*)&counter_result));
                    *current = field64(counter_result, 0, box_map[type].regWidth);
                    break;

                case MBOX0:
                    if (!cpuid_info.supportClientmem)
                    {
                        bdw_uncore_read(cpu_id, index, event, current, overflows,
                                0, ovf_offset, getCounterTypeOffset(index)+1);
                    }
                    else if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, OVERFLOW_CLIENTMEM)
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                        *current = counter_result;
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_MBOX)
                    }
                    
                case MBOX1:
                case MBOX2:
                case MBOX3:
                case MBOX4:
                case MBOX5:
                case MBOX6:
                case MBOX7:
                    bdw_uncore_read(cpu_id, index, event, current, overflows,
                                    0, ovf_offset, getCounterTypeOffset(index)+1);
                    break;

                case MBOX0FIX:
                case MBOX1FIX:
                case MBOX2FIX:
                case MBOX3FIX:
                case MBOX4FIX:
                case MBOX5FIX:
                case MBOX6FIX:
                case MBOX7FIX:
                    bdw_uncore_read(cpu_id, index, event, current, overflows,
                                    0, ovf_offset, 0);
                    break;

                case IBOX1:
                    bdw_uncore_read(cpu_id, index, event, current, overflows,
                                    0, ovf_offset, getCounterTypeOffset(index)+2);
                    break;

                case BBOX0:
                case BBOX1:
                case PBOX:
                case IBOX0:
                case WBOX:
                case UBOX:
                case UBOXFIX:
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
                case RBOX0:
                case RBOX1:
                case SBOX0:
                case SBOX1:
                case SBOX2:
                case SBOX3:
                case QBOX0:
                case QBOX1:
                case QBOX2:
                    bdw_uncore_read(cpu_id, index, event, current, overflows,
                                    0, ovf_offset, getCounterTypeOffset(index));
                    break;

                case WBOX0FIX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_WBOXFIX);
                    }
                    break;

                case QBOX0FIX:
                case QBOX1FIX:
                    if (haveLock)
                    {
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_QBOXFIX)
                        if (eventSet->events[i].event.eventId == 0x00)
                        {
                            HPMread(cpu_id, dev, counter1, &counter_result);
                            switch(extractBitField(counter_result, 3, 0))
                            {
                                case 0x2:
                                    counter_result = 5.6E9;
                                    break;
                                case 0x3:
                                    counter_result = 6.4E9;
                                    break;
                                case 0x4:
                                    counter_result = 7.2E9;
                                    break;
                                case 0x5:
                                    counter_result = 8.0E9;
                                    break;
                                case 0x6:
                                    counter_result = 8.8E9;
                                    break;
                                case 0x7:
                                    counter_result = 9.6E9;
                                    break;
                                default:
                                    counter_result = 0;
                                    break;
                            }
                            
                        }
                        else if ((eventSet->events[i].event.eventId == 0x01) ||
                                 (eventSet->events[i].event.eventId == 0x02))
                        {
                            HPMread(cpu_id, dev, counter1, &counter_result);
                            counter_result = field64(counter_result, 0, box_map[type].regWidth);
                        }
                        *current = counter_result;
                    }
                    break;

                default:
                    break;
            }
        }
    }
    BDW_UNFREEZE_UNCORE;
    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, RESTORE_PMC_FLAGS)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, flags));
    }

    return 0;
}

int perfmon_finalizeCountersThread_broadwell(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int haveTileLock = 0;
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
        if ((reg) && (((type == PMC)||(type == FIXED))||((type >= UNCORE) && (haveLock))))
        {
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, reg, &ovf_values_uncore));
            VERBOSEPRINTPCIREG(cpu_id, dev, reg, ovf_values_uncore, SHOW_CTL);
            ovf_values_uncore = 0x0ULL;
            VERBOSEPRINTPCIREG(cpu_id, dev, reg, 0x0ULL, CLEAR_CTL);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, 0x0ULL));
            if (type >= SBOX0 && type <= SBOX3)
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, 0x0ULL));
            VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL, CLEAR_CTR);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL));
            if (type >= SBOX0 && type <= SBOX3)
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL));
            if (counter_map[index].counterRegister2 != 0x0)
            {
                VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].counterRegister2, 0x0ULL, CLEAR_CTR);
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister2, 0x0ULL));
                if (type >= SBOX0 && type <= SBOX3)
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister2, 0x0ULL));
            }
        }
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
    }

    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_STATUS, LLU_CAST ovf_values_uncore, CLEAR_UNCORE_OVF)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_V3_U_PMON_GLOBAL_STATUS, ovf_values_uncore));
        VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_CTL, LLU_CAST 0x0ULL, CLEAR_UNCORE_CTRL)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_V3_U_PMON_GLOBAL_CTL, 0x0ULL));
        for (int i=UNCORE;i<NUM_UNITS;i++)
        {
            if (TESTTYPE(eventSet, i) && box_map[i].ctrlRegister != 0x0)
            {
                VERBOSEPRINTPCIREG(cpu_id, box_map[i].device, box_map[i].ctrlRegister, 0x0ULL, CLEAR_UNCORE_BOX_CTRL);
                HPMwrite(cpu_id, box_map[i].device, box_map[i].ctrlRegister, 0x0ULL);
                if (i >= SBOX0 && i <= SBOX3)
                    HPMwrite(cpu_id, box_map[i].device, box_map[i].ctrlRegister, 0x0ULL);
                if (box_map[i].filterRegister1 != 0x0)
                {
                    VERBOSEPRINTPCIREG(cpu_id, box_map[i].device, box_map[i].filterRegister1, 0x0ULL, CLEAR_FILTER);
                    HPMwrite(cpu_id, box_map[i].device, box_map[i].filterRegister1, 0x0ULL);
                }
                if (box_map[i].filterRegister2 != 0x0)
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
