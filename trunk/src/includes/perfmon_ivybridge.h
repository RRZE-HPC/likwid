/*
 * =======================================================================================
 *
 *      Filename:  perfmon_ivybridge.h
 *
 *      Description:  Header File of perfmon module for Ivy Bridge.
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


#include <perfmon_ivybridge_events.h>
#include <perfmon_ivybridge_counters.h>
#include <error.h>
#include <affinity.h>
#include <limits.h>

static int perfmon_numCountersIvybridge = NUM_COUNTERS_IVYBRIDGE;
static int perfmon_numCoreCountersIvybridge = NUM_COUNTERS_CORE_IVYBRIDGE;
static int perfmon_numArchEventsIvybridge = NUM_ARCH_EVENTS_IVYBRIDGE;


int perfmon_init_ivybridge(int cpu_id)
{
    if ( cpuid_info.model == IVYBRIDGE_EP )
    {
        lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    }
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL));
    return 0;
}


uint32_t ivb_fixed_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint32_t flags = 0x0UL;
    flags |= (0x2 << (4*index));
    for(int j=0;j<event->numberOfOptions;j++)
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


int ivb_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint32_t flags = 0x0UL;
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
                case EVENT_OPTION_MATCH0:
                    offcore_flags |= event->options[j].value;
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
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_PMC)
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    return 0;
}

int ivb_bbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x0UL;
    uint64_t filter = 0x0UL;
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id))
    {
        return 0;
    }
    if (!pci_checkDevice(counter_map[index].device, cpu_id))
    {
        return -ENODEV;
    }
    flags = (1ULL<<22)|(1ULL<<20);
    flags |= (event->umask<<8) + event->eventId;
    if (event->numberOfOptions > 0)
    {
        for (int j=0;j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= ((event->options[j].value & 0x1FULL) << 24);
                    break;
                case EVENT_OPTION_OPCODE:
                    filter = (event->options[j].value & 0x3FULL);
                    VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device,
                            PCI_UNC_HA_PMON_OPCODEMATCH, flags, SETUP_OPCODE_FILTER);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, counter_map[index].device,
                                                PCI_UNC_HA_PMON_OPCODEMATCH, filter));
                    break;
                case EVENT_OPTION_MATCH0:
                    filter = ((event->options[j].value & 0xFFFFFFC0ULL));
                    VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device,
                            PCI_UNC_HA_PMON_ADDRMATCH0, flags, SETUP_ADDR0_FILTER);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, counter_map[index].device,
                                                PCI_UNC_HA_PMON_ADDRMATCH0, filter));
                    filter = (((event->options[j].value>>32) & 0x3FFFULL));
                    VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device,
                            PCI_UNC_HA_PMON_ADDRMATCH1, flags, SETUP_ADDR1_FILTER);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, counter_map[index].device,
                                                PCI_UNC_HA_PMON_ADDRMATCH1, filter));
                    break;
                default:
                    break;
            }
        }
    }
    VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device, counter_map[index].configRegister,
                        flags, SETUP_BBOX);
    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, counter_map[index].device,
                                     counter_map[index].configRegister, flags));
    return 0;
}

int ivb_pci_box_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x0UL;
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id))
    {
        return 0;
    }
    if (!pci_checkDevice(counter_map[index].device, cpu_id))
    {
        return -ENODEV;
    }
    flags = (1ULL<<22)|(1ULL<<20);
    flags |= (event->umask<<8) + event->eventId;
    if (event->numberOfOptions > 0)
    {
        for (int j=0;j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= ((event->options[j].value & 0x1FULL) << 24);
                    break;
                default:
                    break;
            }
        }
    }
    VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device, counter_map[index].configRegister,
                        flags, SETUP_BOX);
    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, counter_map[index].device,
                                     counter_map[index].configRegister, flags));
    return 0;
}

int ivb_mboxfix_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x0ULL;
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id))
    {
        return 0;
    }
    if (!pci_checkDevice(counter_map[index].device, cpu_id))
    {
        return -ENODEV;
    }
    flags = (1ULL<<22);
    VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device,
        counter_map[index].configRegister, flags, SETUP_MBOXFIX);
    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, counter_map[index].device,
        counter_map[index].configRegister, flags));
    return 0;
}

int ivb_sbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x0UL;
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id))
    {
        return 0;
    }
    if (!pci_checkDevice(counter_map[index].device, cpu_id))
    {
        return -ENODEV;
    }
    PciDeviceIndex dev = counter_map[index].device;
    flags = (1ULL<<22)|(1ULL<<20);
    flags |= (event->umask<<8) + event->eventId;
    if (event->cfgBits != 0x0)
    {
        flags = (1ULL<<21);
    }
    if (event->numberOfOptions > 0)
    {
        for (int j=0;j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= ((event->options[j].value & 0x1FULL) << 24);
                    break;
                case EVENT_OPTION_MATCH0:
                    VERBOSEPRINTPCIREG(cpu_id, dev,
                            PCI_UNC_QPI_PMON_MATCH_0, flags, SETUP_SBOX_MATCH0);
                    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev,
                            PCI_UNC_QPI_PMON_MATCH_0, event->options[j].value));
                case EVENT_OPTION_MATCH1:
                    VERBOSEPRINTPCIREG(cpu_id, dev,
                            PCI_UNC_QPI_PMON_MATCH_1, flags, SETUP_SBOX_MATCH1);
                    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev,
                            PCI_UNC_QPI_PMON_MATCH_1, event->options[j].value));
                case EVENT_OPTION_MASK0:
                    VERBOSEPRINTPCIREG(cpu_id, dev,
                            PCI_UNC_QPI_PMON_MASK_0, flags, SETUP_SBOX_MASK0);
                    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev,
                            PCI_UNC_QPI_PMON_MASK_0, event->options[j].value));
                case EVENT_OPTION_MASK1:
                    VERBOSEPRINTPCIREG(cpu_id, dev,
                            PCI_UNC_QPI_PMON_MASK_1, flags, SETUP_SBOX_MASK1);
                    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev,
                            PCI_UNC_QPI_PMON_MASK_1, event->options[j].value));
                default:
                    break;
            }
        }
    }
    VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].configRegister, flags, SETUP_SBOX);
    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
    flags = 0x0ULL;
    CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter_map[index].configRegister, &flags));
    VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].configRegister, flags, SETUP_SBOX_CHECK);
    return 0;
}

int ivb_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint32_t flags = 0x0UL;
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id))
    {
        return 0;
    }
    flags = (1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;
    if (event->numberOfOptions > 0)
    {
        uint64_t filter0 = 0x0ULL;
        uint64_t filter1 = 0x0ULL;
        int state_set = 0;
        for (int j=0;j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= ((event->options[j].value & 0x1FULL) << 24);
                    break;
                case EVENT_OPTION_TID:
                    flags |= (1<<19);
                    filter0 |= (event->options[j].value & 0x1FULL);
                    break;
                case EVENT_OPTION_STATE:
                    filter0 |= ((event->options[j].value & 0x3FULL) << 17);
                    state_set = 1;
                    break;
                case EVENT_OPTION_NID:
                    if (event->options[j].value >= 0x1 && 
                        event->options[j].value <= (affinityDomains.numberOfNumaDomains+1<<1))
                    {
                        filter1 |= (event->options[j].value & 0xFFFFULL);
                    }
                    break;
                case EVENT_OPTION_OPCODE:
                    filter1 |= ((event->options[j].value & 0x1FFULL) << 20);
                    break;
                case EVENT_OPTION_MATCH0:
                    filter1 |= ((event->options[j].value & 0x3) << 30);
                    break;
                default:
                    break;
            }
        }
        if (state_set == 0 && event->eventId == 0x34)
        {
            filter0 |= (0x1FULL<<17);
        }
        if (filter0 != 0x0ULL)
        {
            VERBOSEPRINTREG(cpu_id, box_map[counter_map[index].type].filterRegister1,
                            filter0, SETUP_CBOX_FILTER0);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV,
                                             box_map[counter_map[index].type].filterRegister1,
                                             filter0));
        }
        if (filter1 != 0x0ULL)
        {
            VERBOSEPRINTREG(cpu_id, box_map[counter_map[index].type].filterRegister2, 
                            filter1, SETUP_CBOX_FILTER1);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV,
                                             box_map[counter_map[index].type].filterRegister2,
                                             filter1));
        }
    }
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_CBOX);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    return 0;
}

int ivb_ubox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint32_t flags = 0x0UL;
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id))
    {
        return 0;
    }
    flags = (1ULL<<22)|(1ULL<<20)|(1ULL<<17);
    flags |= (event->umask<<8) + event->eventId;
    if (event->numberOfOptions > 0)
    {
        for (int j=0;j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= ((event->options[j].value & 0x1F) << 24);
                    break;
                default:
                    break;
            }
        }
    }
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_UBOX);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    return 0;
}

int ivb_uboxfix_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint32_t flags = 0x0UL;
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id))
    {
        return 0;
    }
    flags = (1ULL<<22)|(1ULL<<20);
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_UBOXFIX)
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    return 0;
}

int ivb_wbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint32_t flags = 0x0UL;
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id))
    {
        return 0;
    }
    flags = (1ULL<<22)|(1ULL<<20);
    flags |= event->eventId;
    if (event->cfgBits != 0x0)
    {
        flags |= ((event->cfgBits & 0x1) << 21);
    }
    if (event->numberOfOptions > 0)
    {
        for (int j=0;j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= ((event->options[j].value & 0x1F) << 24);
                    break;
                case EVENT_OPTION_OCCUPANCY:
                    flags |= ((event->options[j].value & 0x3) << 14);
                    break;
                case EVENT_OPTION_OCCUPANCY_INVERT:
                    flags |= (1ULL<<30);
                    break;
                case EVENT_OPTION_OCCUPANCY_EDGE:
                    flags |= (1ULL<<31);
                    break;
                case EVENT_OPTION_MATCH0:
                    VERBOSEPRINTREG(cpu_id, box_map[counter_map[index].type].filterRegister1,
                                    event->options[j].value, SETUP_WBOX_FILTER);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV,
                                    box_map[counter_map[index].type].filterRegister1,
                                    event->options[j].value));
                    break;
                default:
                    break;
            }
        }
    }
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_WBOX);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    return 0;
}

int ivb_ibox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint32_t flags = 0x0UL;
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id))
    {
        return 0;
    }
    if (!pci_checkDevice(counter_map[index].device, cpu_id))
    {
        return -ENODEV;
    }
    flags = (1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;
    if (event->numberOfOptions > 0)
    {
        for (int j=0;j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= ((event->options[j].value & 0xFULL) << 24);
                    break;
                default:
                    break;
            }
        }
    }
    VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device, counter_map[index].configRegister,
                        flags, SETUP_BOX);
    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, counter_map[index].device,
                                     counter_map[index].configRegister, flags));
    return 0;
}


int ivb_uncore_freeze(int cpu_id, PerfmonEventSet* eventSet, int flags)
{
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id))
    {
        return 0;
    }
    if (eventSet->regTypeMask & ~(0xF))
    {
        VERBOSEPRINTREG(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<31), FREEZE_UNCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_U_PMON_GLOBAL_CTL, (1ULL<<31)));
    }
    if ((flags != FREEZE_FLAG_ONLYFREEZE) && (eventSet->regTypeMask & ~(0xF)))
    {
        for (int j=UNCORE; j<NUM_UNITS; j++)
        {
            if (eventSet->regTypeMask & REG_TYPE_MASK(j))
            {
                if ((box_map[j].ctrlRegister != 0x0) && (box_map[j].isPci))
                {
                    DEBUG_PRINT(DEBUGLEV_DETAIL, Clearing %s registers of %s, (flags == FREEZE_FLAG_CLEAR_CTL ? "control" : "counter"), RegisterTypeNames[j]);
                    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, box_map[j].device,
                                                    box_map[j].ctrlRegister, flags));
                }
                else if (box_map[j].ctrlRegister != 0x0)
                {
                    DEBUG_PRINT(DEBUGLEV_DETAIL, Clearing %s registers of %s, (flags == FREEZE_FLAG_CLEAR_CTL ? "control" : "counter"), RegisterTypeNames[j]);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV,
                                                     box_map[j].ctrlRegister, flags));
                }
            }
        }
    }
    return 0;
}

int ivb_uncore_unfreeze(int cpu_id, PerfmonEventSet* eventSet, int flags)
{
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id))
    {
        return 0;
    }
    if ((flags != FREEZE_FLAG_ONLYFREEZE) && (eventSet->regTypeMask & ~(0xF)))
    {
        for (int j=UNCORE; j<NUM_UNITS; j++)
        {
            if (eventSet->regTypeMask & REG_TYPE_MASK(j))
            {
                if ((box_map[j].ctrlRegister != 0x0) && (box_map[j].isPci))
                {
                    DEBUG_PRINT(DEBUGLEV_DETAIL, Clearing %s registers of %s, (flags == FREEZE_FLAG_CLEAR_CTL ? "control" : "counter"), RegisterTypeNames[j]);
                    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, box_map[j].device,
                                                    box_map[j].ctrlRegister, flags));
                }
                else if (box_map[j].ctrlRegister != 0x0)
                {
                    DEBUG_PRINT(DEBUGLEV_DETAIL, Clearing %s registers of %s, (flags == FREEZE_FLAG_CLEAR_CTL ? "control" : "counter"), RegisterTypeNames[j]);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV,
                                                     box_map[j].ctrlRegister, flags));
                }
            }
        }
    }
    if (eventSet->regTypeMask & ~(0xF))
    {
        VERBOSEPRINTREG(cpu_id, MSR_UNC_U_PMON_GLOBAL_STATUS, LLU_CAST 0x0ULL, CLEAR_UNCORE_OVF)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_U_PMON_GLOBAL_STATUS, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<29), UNFREEZE_UNCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_U_PMON_GLOBAL_CTL, (1ULL<<29)));
    }
    return 0;
}


int perfmon_setupCounterThread_ivybridge(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t fixed_flags = 0x0ULL;
    uint32_t uflags = 0x0;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL));
    }

    ivb_uncore_freeze(cpu_id, eventSet, FREEZE_FLAG_ONLYFREEZE);

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        uint64_t reg = counter_map[index].configRegister;
        PciDeviceIndex dev = counter_map[index].device;
        PerfmonEvent *event = &(eventSet->events[i].event);
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        switch (eventSet->events[i].type)
        {
            case PMC:
                ivb_pmc_setup(cpu_id, index, event);
                break;

            case FIXED:
                fixed_flags |= ivb_fixed_setup(cpu_id, index, event);
                break;

            case POWER:
                break;

            case MBOX0:
            case MBOX1:
            case MBOX2:
            case MBOX3:
            case PBOX:
            case RBOX0:
            case RBOX1:
                ivb_pci_box_setup(cpu_id, index, event);
                break;

            case BBOX0:
            case BBOX1:
                ivb_bbox_setup(cpu_id, index, event);
                break;

            case MBOX0FIX:
            case MBOX1FIX:
            case MBOX2FIX:
            case MBOX3FIX:
                ivb_mboxfix_setup(cpu_id, index, event);
                break;

            case SBOX0:
            case SBOX1:
                ivb_sbox_setup(cpu_id, index, event);
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
                ivb_cbox_setup(cpu_id, index, event);
                break;

            case UBOX:
                ivb_ubox_setup(cpu_id, index, event);
                break;
            case UBOXFIX:
                ivb_uboxfix_setup(cpu_id, index, event);
                break;

            case WBOX:
                ivb_wbox_setup(cpu_id, index, event);
                break;

            case IBOX0:
            case IBOX1:
                ivb_ibox_setup(cpu_id, index, event);
                break;

            default:
                break;
        }
    }
    if (fixed_flags > 0x0)
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_FIXED_CTR_CTRL, LLU_CAST fixed_flags, SETUP_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_FIXED_CTR_CTRL, fixed_flags));
    }
    return 0;
}


int perfmon_startCountersThread_ivybridge(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t power = 0x0ULL;
    uint64_t fixed_flags = 0x0ULL;
    uint32_t uflags = 0x10000UL; /* Clear freeze bit */
    int cpu_id = groupSet->threads[thread_id].processorId;

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
            PciDeviceIndex dev = counter_map[index].device;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;

            switch (eventSet->events[i].type)
            {
                case PMC:
                    if (eventSet->regTypeMask & REG_TYPE_MASK(PMC))
                    {
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter1, 0x0ULL));
                        fixed_flags |= (1ULL<<(index-cpuid_info.perf_num_fixed_ctr));  /* enable counter */
                    }
                    break;

                case FIXED:
                    if (eventSet->regTypeMask & REG_TYPE_MASK(FIXED))
                    {
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter1, 0x0ULL));
                        fixed_flags |= (1ULL<<(index+32));  /* enable fixed counter */
                    }
                    break;

                case POWER:
                    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(POWER)))
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1,
                                        (uint32_t*)&eventSet->events[i].threadCounter[thread_id].startData));
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST eventSet->events[i].threadCounter[thread_id].startData, START_POWER)
                    }
                    break;
                /*case MBOX0FIX:
                case MBOX1FIX:
                case MBOX2FIX:
                case MBOX3FIX:
                    if (haveLock)
                    {
                        uint64_t tmp_1 = 0x0ULL;
                        uint64_t tmp_2 = 0x0ULL;
                        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp_1));
                        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter2, &tmp_2));
                        eventSet->events[i].threadCounter[thread_id].startData = (tmp_1<<32)|(tmp_2 & 0xFFFFFFFF);
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1,
                            LLU_CAST eventSet->events[i].threadCounter[thread_id].startData, START_MBOXFIX);
                    }
                    break;
                case SBOX0:
                case SBOX1:
                case SBOX2:
                    if (haveLock)
                    {
                        uint64_t tmp_1 = 0x0ULL;
                        uint64_t tmp_2 = 0x0ULL;
                        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp_1));
                        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter2, &tmp_2));
                        eventSet->events[i].threadCounter[thread_id].startData = (tmp_1<<32)|(tmp_2 & 0xFFFFFFFF);
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1,
                            LLU_CAST eventSet->events[i].threadCounter[thread_id].startData, START_SBOX);
                    }
                    break;
                case BBOX0:
                case BBOX1:
                    if (haveLock)
                    {
                        uint64_t tmp_1 = 0x0ULL;
                        uint64_t tmp_2 = 0x0ULL;
                        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp_1));
                        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter2, &tmp_2));
                        eventSet->events[i].threadCounter[thread_id].startData = (tmp_1<<32)|(tmp_2 & 0xFFFFFFFF);
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1,
                            LLU_CAST eventSet->events[i].threadCounter[thread_id].startData, START_SBOX);
                    }
                    break;*/
                default:
                    /*if (haveLock && (eventSet->events[i].type >= UNCORE))
                    {
                        uint64_t tmp_1 = 0x0ULL;
                        uint64_t tmp_2 = 0x0ULL;
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                        if (counter2 != 0x0)
                        {
                            CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter2, 0x0ULL));
                        }
                        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp_1));
                        if (counter2 != 0x0)
                        {
                            CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter2, &tmp_2));
                            eventSet->events[i].threadCounter[thread_id].startData =
                                                    (tmp_1<<32)|(tmp_2 & 0xFFFFFFFF);
                        }
                        else
                        {
                            eventSet->events[i].threadCounter[thread_id].startData = tmp_1;
                        }
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1<<16|counter2,
                            LLU_CAST eventSet->events[i].threadCounter[thread_id].startData, SAFE_START_VALUE);
                    }*/
                    break;
            }
        }
    }

    ivb_uncore_unfreeze(cpu_id, eventSet, FREEZE_FLAG_CLEAR_CTR);

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST fixed_flags, UNFREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, fixed_flags));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<63)|(1ULL<<62)|fixed_flags));
    }
    return 0;
}



int ivb_uncore_read(int cpu_id, RegisterIndex index, PerfmonEvent *event,
                     uint64_t* cur_result, int* overflows, int flags,
                     int global_offset, int box_offset)
{
    uint64_t result = 0x0ULL;
    uint64_t tmp = 0x0ULL;
    uint64_t reg = counter_map[index].configRegister;
    RegisterType type = counter_map[index].type;
    PciDeviceIndex dev = counter_map[index].device;
    uint64_t counter1 = counter_map[index].counterRegister;
    uint64_t counter2 = counter_map[index].counterRegister2;
    
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id))
    {
        *cur_result = 0;
        return 0;
    }
    if (box_map[type].isPci && !pci_checkDevice(dev, cpu_id))
    {
        *cur_result = 0;
        return 0;
    }

    CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));

    if (flags & FREEZE_FLAG_CLEAR_CTR)
    {
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0U));
    }
    if (counter2 != 0x0)
    {
        result = (tmp<<32);
        tmp = 0x0ULL;
        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter2, &tmp));
        result |= (tmp & 0xFFFFFFFF);
        if (flags & FREEZE_FLAG_CLEAR_CTR)
        {
            CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter2, 0x0U));
        }
    }
    else
    {
        result = tmp;
    }

    if (result < *cur_result)
    {
        uint64_t ovf_values = 0x0ULL;
        int test_local = 0;
        if (global_offset != -1)
        {
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV,
                                           MSR_UNC_U_PMON_GLOBAL_STATUS,
                                           &ovf_values));
            if (ovf_values & (1<<global_offset))
            {
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV,
                                                 MSR_UNC_U_PMON_GLOBAL_STATUS,
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
            if (ivybridge_box_map[type].isPci)
            {
                CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev,
                                              box_map[type].statusRegister,
                                              &ovf_values));
            }
            else
            {
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV,
                                              box_map[type].statusRegister,
                                              &ovf_values));
            }
            if (ovf_values & (1<<box_offset))
            {
                (*overflows)++;
                if (ivybridge_box_map[type].isPci)
                {
                    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev,
                                                    box_map[type].statusRegister,
                                                    (1<<box_offset)));
                }
                else
                {
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, 
                                                     box_map[type].statusRegister,
                                                     (1<<box_offset)));
                }
            }
        }
    }
    *cur_result = result;
    return 0;
}

int perfmon_stopCountersThread_ivybridge(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags;
    uint32_t uflags = 0x10100UL; /* Set freeze bit */
    uint64_t counter_result = 0x0ULL;
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    ivb_uncore_freeze(cpu_id, eventSet, FREEZE_FLAG_CLEAR_CTL);

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        counter_result= 0x0ULL;
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            PerfmonEvent *event = &(eventSet->events[i].event);
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;
            uint64_t* current = &(eventSet->events[i].threadCounter[thread_id].counterData);
            int* overflows = &(eventSet->events[i].threadCounter[thread_id].overflows);
            switch (eventSet->events[i].type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    if (counter_result < *current)
                    {
                        uint64_t ovf_values = 0x0ULL;
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_STATUS, &ovf_values));
                        if (ovf_values & (1<<index-cpuid_info.perf_num_fixed_ctr))
                        {
                            (*overflows)++;
                        }
                    }
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
                    *current = counter_result;
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    if (counter_result < *current)
                    {
                        uint64_t ovf_values = 0x0ULL;
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_STATUS, &ovf_values));
                        if (ovf_values & (1<<(index+32)))
                        {
                            (*overflows)++;
                        }
                    }
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED)
                    *current = counter_result;
                    break;

                case POWER:
                    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(POWER)))
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&counter_result));
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, STOP_POWER)
                        if (counter_result < *current)
                        {
                            (*overflows)++;
                        }
                        *current = counter_result;
                    }
                    break;

                case THERMAL:
                    CHECK_TEMP_READ_ERROR(thermal_read(cpu_id, (uint32_t*)&counter_result));
                    *current = counter_result;
                    break;

                case SBOX0FIX:
                case SBOX1FIX:
                case SBOX2FIX:
                    CHECK_PCI_READ_ERROR(HPMread(cpu_id, counter_map[index].device, counter1, &counter_result));
                    VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device, counter1, LLU_CAST counter_result, READ_SBOX_FIXED)
                    switch (extractBitField(counter_result,3,0))
                    {
                        case 0x2:
                            counter_result = 5600000000ULL;
                            break;
                        case 0x3:
                            counter_result = 6400000000ULL;
                            break;
                        case 0x4:
                            counter_result = 7200000000ULL;
                            break;
                        case 0x5:
                            counter_result = 8000000000ULL;
                            break;
                        case 0x6:
                            counter_result = 8800000000ULL;
                            break;
                        case 0x7:
                            counter_result = 9600000000ULL;
                            break;
                        default:
                            counter_result = 0x0ULL;
                            break;
                    }
                    VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device, counter1, LLU_CAST counter_result, READ_SBOX_FIXED_REAL)
                    *current = counter_result;
                    break;

                case MBOX0:
                case MBOX1:
                case MBOX2:
                case MBOX3:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 20, getCounterTypeOffset(index)+1);
                    break;


                case MBOX0FIX:
                case MBOX1FIX:
                case MBOX2FIX:
                case MBOX3FIX:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 20, 0);
                    break;

                case SBOX0:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 22, getCounterTypeOffset(index));
                    break;

                case SBOX1:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 23, getCounterTypeOffset(index));
                    break;

                
                case CBOX0:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 3, getCounterTypeOffset(index));
                    break;
                case CBOX1:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 4, getCounterTypeOffset(index));
                    break;
                case CBOX2:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 5, getCounterTypeOffset(index));
                    break;
                case CBOX3:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 6, getCounterTypeOffset(index));
                    break;
                case CBOX4:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 7, getCounterTypeOffset(index));
                    break;
                case CBOX5:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 8, getCounterTypeOffset(index));
                    break;
                case CBOX6:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 9, getCounterTypeOffset(index));
                    break;
                case CBOX7:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 10, getCounterTypeOffset(index));
                    break;
                case CBOX8:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 11, getCounterTypeOffset(index));
                    break;
                case CBOX9:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 12, getCounterTypeOffset(index));
                    break;
                case CBOX10:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 13, getCounterTypeOffset(index));
                    break;
                case CBOX11:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 14, getCounterTypeOffset(index));
                    break;
                case CBOX12:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 15, getCounterTypeOffset(index));
                    break;
                case CBOX13:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 16, getCounterTypeOffset(index));
                    break;
                case CBOX14:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 17, getCounterTypeOffset(index));
                    break;

                case UBOX:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 1, getCounterTypeOffset(index));
                    break;
                case UBOXFIX:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 0, getCounterTypeOffset(index));
                    break;

                case BBOX0:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 18, getCounterTypeOffset(index));
                    break;
                case BBOX1:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 19, getCounterTypeOffset(index));
                    break;

                case WBOX:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 2, getCounterTypeOffset(index));
                    break;

                case PBOX:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 26, getCounterTypeOffset(index));
                    break;

                case RBOX0:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 14, getCounterTypeOffset(index));
                    break;
                case RBOX1:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, 25, getCounterTypeOffset(index));
                    break;

                case IBOX0:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, -1, getCounterTypeOffset(index));
                    break;
                case IBOX1:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, -1, getCounterTypeOffset(index)+2);
                    break;

                default:
                    /* should never be reached */
                    break;
            }
        }
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
    }

    return 0;
}

int perfmon_readCountersThread_ivybridge(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t counter_result = 0x0ULL;
    uint64_t pmc_flags = 0x0ULL;
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, &pmc_flags));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    ivb_uncore_freeze(cpu_id, eventSet, FREEZE_FLAG_ONLYFREEZE);

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        counter_result = 0x0ULL;
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            PerfmonEvent *event = &(eventSet->events[i].event);
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;
            uint64_t* current = &(eventSet->events[i].threadCounter[thread_id].counterData);
            int* overflows = &(eventSet->events[i].threadCounter[thread_id].overflows);
            switch (eventSet->events[i].type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    if (counter_result < *current)
                    {
                        uint64_t ovf_values = 0x0ULL;
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_STATUS, &ovf_values));
                        if (ovf_values & (1<<index-cpuid_info.perf_num_fixed_ctr))
                        {
                            (*overflows)++;
                        }
                    }
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
                    *current = counter_result;
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    if (counter_result < *current)
                    {
                        uint64_t ovf_values = 0x0ULL;
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_STATUS, &ovf_values));
                        if (ovf_values & (1<<(index+32)))
                        {
                            (*overflows)++;
                        }
                    }
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED)
                    *current = counter_result;
                    break;

                case POWER:
                    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(POWER)))
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id,
                                                           counter1,
                                                           (uint32_t*)&counter_result));
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, STOP_POWER)
                        if (counter_result < *current)
                        {
                            (*overflows)++;
                        }
                        *current = counter_result;
                    }
                    break;

                case THERMAL:
                    CHECK_TEMP_READ_ERROR(thermal_read(cpu_id, (uint32_t*)&counter_result));
                    *current = counter_result;
                    break;

                case SBOX0FIX:
                case SBOX1FIX:
                case SBOX2FIX:
                    CHECK_PCI_READ_ERROR(HPMread(cpu_id, counter_map[index].device, counter1, &counter_result));
                    VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device, counter1, LLU_CAST counter_result, READ_SBOX_FIXED)
                    switch (extractBitField(counter_result,3,0))
                    {
                        case 0x2:
                            counter_result = 5600000000ULL;
                            break;
                        case 0x3:
                            counter_result = 6400000000ULL;
                            break;
                        case 0x4:
                            counter_result = 7200000000ULL;
                            break;
                        case 0x5:
                            counter_result = 8000000000ULL;
                            break;
                        case 0x6:
                            counter_result = 8800000000ULL;
                            break;
                        case 0x7:
                            counter_result = 9600000000ULL;
                            break;
                        default:
                            counter_result = 0x0ULL;
                            break;
                    }
                    VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device, counter1, LLU_CAST counter_result, READ_SBOX_FIXED_REAL)
                    *current = counter_result;
                    break;

                case MBOX0:
                case MBOX1:
                case MBOX2:
                case MBOX3:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 20, getCounterTypeOffset(index)+1);
                    break;


                case MBOX0FIX:
                case MBOX1FIX:
                case MBOX2FIX:
                case MBOX3FIX:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 20, 0);
                    break;

                case SBOX0:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 22, getCounterTypeOffset(index));
                    break;

                case SBOX1:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 23, getCounterTypeOffset(index));
                    break;

                
                case CBOX0:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 3, getCounterTypeOffset(index));
                    break;
                case CBOX1:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 4, getCounterTypeOffset(index));
                    break;
                case CBOX2:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 5, getCounterTypeOffset(index));
                    break;
                case CBOX3:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 6, getCounterTypeOffset(index));
                    break;
                case CBOX4:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 7, getCounterTypeOffset(index));
                    break;
                case CBOX5:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 8, getCounterTypeOffset(index));
                    break;
                case CBOX6:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 9, getCounterTypeOffset(index));
                    break;
                case CBOX7:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 10, getCounterTypeOffset(index));
                    break;
                case CBOX8:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 11, getCounterTypeOffset(index));
                    break;
                case CBOX9:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 12, getCounterTypeOffset(index));
                    break;
                case CBOX10:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 13, getCounterTypeOffset(index));
                    break;
                case CBOX11:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 14, getCounterTypeOffset(index));
                    break;
                case CBOX12:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 15, getCounterTypeOffset(index));
                    break;
                case CBOX13:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 16, getCounterTypeOffset(index));
                    break;
                case CBOX14:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 17, getCounterTypeOffset(index));
                    break;

                case UBOX:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 1, getCounterTypeOffset(index));
                    break;
                case UBOXFIX:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 0, getCounterTypeOffset(index));
                    break;

                case BBOX0:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 18, getCounterTypeOffset(index));
                    break;
                case BBOX1:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 19, getCounterTypeOffset(index));
                    break;

                case WBOX:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 2, getCounterTypeOffset(index));
                    break;

                case PBOX:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 26, getCounterTypeOffset(index));
                    break;

                case RBOX0:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 14, getCounterTypeOffset(index));
                    break;
                case RBOX1:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, 25, getCounterTypeOffset(index));
                    break;

                case IBOX0:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, -1, getCounterTypeOffset(index));
                    break;
                case IBOX1:
                    ivb_uncore_read(cpu_id, index, event, current, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, -1, getCounterTypeOffset(index)+2);
                    break;

                default:
                    /* should never be reached */
                    break;
            }
        }
    }

    ivb_uncore_unfreeze(cpu_id, eventSet, FREEZE_FLAG_ONLYFREEZE);
    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, pmc_flags));
    }
    return 0;
}


int perfmon_finalizeCountersThread_ivybridge(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t ovf_values_core = (1ULL<<63)|(1ULL<<62);
    uint64_t ovf_values_uncore = 0x0ULL;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterIndex index = eventSet->events[i].index;
        PciDeviceIndex dev = counter_map[index].device;
        uint64_t reg = counter_map[index].configRegister;
        RegisterType type = counter_map[index].type;
        if (eventSet->events[i].type == NOTYPE)
        {
            continue;
        }

        switch(type)
        {
            case PMC:
                ovf_values_core |= (1ULL<<(index-cpuid_info.perf_num_fixed_ctr));
                if (eventSet->events[i].event.eventId == 0xB7)
                {
                    VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, 0x0ULL, CLEAR_OFFCORE_RESP0);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, 0x0ULL));
                }
                else if (eventSet->events[i].event.eventId == 0xBB)
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
            VERBOSEPRINTPCIREG(cpu_id, dev, reg, 0x0ULL, CLEAR_CTL);
            CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, 0x0ULL));
            VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL, CLEAR_CTR);
            CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL));
        }
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
    }
    if (haveLock && eventSet->regTypeMask & ~(0xFULL))
    {
        VERBOSEPRINTREG(cpu_id, MSR_UNC_U_PMON_GLOBAL_STATUS, LLU_CAST 0x0ULL, CLEAR_UNCORE_OVF)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_U_PMON_GLOBAL_STATUS, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, LLU_CAST 0x0ULL, CLEAR_UNCORE_CTRL)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_U_PMON_GLOBAL_CTL, 0x0ULL));
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, LLU_CAST ovf_values_core, CLEAR_GLOBAL_OVF)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, ovf_values_core));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST 0x0ULL, CLEAR_GLOBAL_CTRL)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    return 0;
}
