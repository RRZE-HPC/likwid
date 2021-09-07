/*
 * =======================================================================================
 *
 *      Filename:  perfmon_ivybridge.h
 *
 *      Description:  Header File of perfmon module for Intel Ivy Bridge.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
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


#include <perfmon_ivybridge_events.h>
#include <perfmon_ivybridge_counters.h>
#include <perfmon_ivybridgeEP_events.h>
#include <perfmon_ivybridgeEP_counters.h>
#include <error.h>
#include <affinity.h>
#include <limits.h>
#include <topology.h>

static int perfmon_numCountersIvybridgeEP = NUM_COUNTERS_IVYBRIDGEEP;
static int perfmon_numCoreCountersIvybridgeEP = NUM_COUNTERS_CORE_IVYBRIDGEEP;
static int perfmon_numArchEventsIvybridgeEP = NUM_ARCH_EVENTS_IVYBRIDGEEP;
static int perfmon_numCountersIvybridge = NUM_COUNTERS_IVYBRIDGE;
static int perfmon_numCoreCountersIvybridge = NUM_COUNTERS_CORE_IVYBRIDGE;
static int perfmon_numArchEventsIvybridge = NUM_ARCH_EVENTS_IVYBRIDGE;

int ivb_did_cbox_test = 0;
int ivb_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event);
int ivbep_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event);
int (*ivy_cbox_setup)(int, RegisterIndex, PerfmonEvent*);

int ivb_cbox_nosetup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    cpu_id++;
    index++;
    event++;
    return 0;
}

int perfmon_init_ivybridge(int cpu_id)
{
    int ret;
    uint64_t data = 0x0ULL;
    lock_acquire((int*) &socket_lock[affinity_thread2socket_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &tile_lock[affinity_thread2core_lookup[cpu_id]], cpu_id);
    HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL);
    if (cpuid_info.model == IVYBRIDGE_EP)
    {
        ivy_cbox_setup = ivbep_cbox_setup;
        ivb_did_cbox_test = 1;
    }
    else if (cpuid_info.model == IVYBRIDGE &&
             socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id &&
             ivb_did_cbox_test == 0)
    {
        ret = HPMwrite(cpu_id, MSR_DEV, MSR_UNC_CBO_0_PERFEVTSEL0, 0x0ULL);
        ret += HPMread(cpu_id, MSR_DEV, MSR_UNC_PERF_GLOBAL_CTRL, &data);
        ret += HPMwrite(cpu_id, MSR_DEV, MSR_UNC_PERF_GLOBAL_CTRL, 0x0ULL);
        ret += HPMread(cpu_id, MSR_DEV, MSR_UNC_CBO_0_PERFEVTSEL0, &data);
        if ((ret == 0) && (data == 0x0ULL))
            ivy_cbox_setup = ivb_cbox_setup;
        else
            ivy_cbox_setup = ivb_cbox_nosetup;
        ivb_did_cbox_test = 1;
    }
    return 0;
}


uint32_t ivb_fixed_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint32_t flags = 0x0UL;
    flags |= (1ULL<<(1+(index*4)));
    cpu_id++;
    for(int j=0;j<event->numberOfOptions;j++)
    {
        switch (event->options[j].type)
        {
            case EVENT_OPTION_COUNT_KERNEL:
                flags |= (1ULL<<(index*4));
                break;
            case EVENT_OPTION_ANYTHREAD:
                flags |= (1ULL<<(2+(index*4)));
                break;
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
                case EVENT_OPTION_THRESHOLD:
                    flags |= (event->options[j].value & 0xFFULL) << 24;
                    break;
                case EVENT_OPTION_MATCH0:
                    offcore_flags |= (event->options[j].value & 0x8FFF);
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
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int ivb_bbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x0UL;
    uint64_t filter = 0x0UL;
    uint32_t reg = counter_map[index].configRegister;
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
                    VERBOSEPRINTPCIREG(cpu_id, dev, PCI_UNC_HA_PMON_OPCODEMATCH, flags, SETUP_OPCODE_FILTER);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, PCI_UNC_HA_PMON_OPCODEMATCH, filter));
                    break;
                case EVENT_OPTION_MATCH0:
                    filter = ((event->options[j].value & 0xFFFFFFC0ULL));
                    VERBOSEPRINTPCIREG(cpu_id, dev, PCI_UNC_HA_PMON_ADDRMATCH0, filter, SETUP_ADDR0_FILTER);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, PCI_UNC_HA_PMON_ADDRMATCH0, filter));
                    filter = (((event->options[j].value>>32) & 0x3FFFULL));
                    VERBOSEPRINTPCIREG(cpu_id, dev, PCI_UNC_HA_PMON_ADDRMATCH1, filter, SETUP_ADDR1_FILTER);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, PCI_UNC_HA_PMON_ADDRMATCH1, filter));
                    break;
                default:
                    break;
            }
        }
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTPCIREG(cpu_id, dev, reg, flags, SETUP_BBOX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int ivb_pci_box_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x0UL;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!HPMcheck(counter_map[index].device, cpu_id))
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
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device, counter_map[index].configRegister,
                            flags, SETUP_BOX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, counter_map[index].device,
                                         counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int ivb_mboxfix_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x0ULL;
    event++;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!HPMcheck(counter_map[index].device, cpu_id))
    {
        return -ENODEV;
    }
    flags = (1ULL<<22);
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device,
            counter_map[index].configRegister, flags, SETUP_MBOXFIX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, counter_map[index].device,
            counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int ivb_sbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event, PciDeviceIndex filterdev)
{
    uint64_t flags = 0x0UL;
    uint32_t filterreg = 0x0U;
    uint64_t filterval = 0x0ULL;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!HPMcheck(counter_map[index].device, cpu_id))
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
                    if (HPMcheck(filterdev, cpu_id))
                    {
                        filterreg = PCI_UNC_QPI_PMON_MATCH_0;
                        filterval = event->options[j].value & 0x8003FFF8ULL;
                        VERBOSEPRINTPCIREG(cpu_id, filterdev, filterreg, filterval, SETUP_SBOX_MATCH0);
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
                        filterreg = PCI_UNC_QPI_PMON_MATCH_1;
                        filterval = event->options[j].value & 0x000F000FULL;
                        VERBOSEPRINTPCIREG(cpu_id, filterdev, filterreg, filterval, SETUP_SBOX_MATCH1);
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
                        filterreg = PCI_UNC_QPI_PMON_MASK_0;
                        filterval = event->options[j].value & 0x8003FFF8ULL;
                        VERBOSEPRINTPCIREG(cpu_id, filterdev, filterreg, filterval, SETUP_SBOX_MASK0);
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
                        filterreg = PCI_UNC_QPI_PMON_MASK_1;
                        filterval = event->options[j].value & 0x000F000FULL;
                        VERBOSEPRINTPCIREG(cpu_id, filterdev, filterreg, filterval, SETUP_SBOX_MASK1);
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
        VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].configRegister, flags, SETUP_SBOX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int ivb_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint32_t flags = 0x0UL;
    uint64_t mask = 0x0ULL;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
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
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_CBOX);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int ivbep_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint32_t flags = 0x0UL;
    uint64_t mask = 0x0ULL;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    flags = (1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;
    if (event->numberOfOptions > 0)
    {
        RegisterType type = counter_map[index].type;
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
                    mask = 0x0ULL;
                    for (uint32_t i=0; i < affinityDomains.numberOfNumaDomains; i++)
                    {
                        mask |= (1ULL<<i);
                    }
                    if (event->options[j].value & mask)
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
            VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister1, filter0, SETUP_CBOX_FILTER0);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister1, filter0));
        }
        if (filter1 != 0x0ULL)
        {
            VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister2, filter1, SETUP_CBOX_FILTER1);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister2, filter1));
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

int ivb_ubox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint32_t flags = 0x0UL;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    flags = (1ULL<<22)|(1ULL<<20);
    if (cpuid_info.model == IVYBRIDGE_EP)
    {
        flags |= (1ULL<<17);
    }
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
                case EVENT_OPTION_INVERT:
                    if (cpuid_info.model == IVYBRIDGE)
                    {
                        flags |= (1ULL<<23);
                    }
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

int ivb_uboxfix_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint32_t flags = 0x0UL;
    event++;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    flags = (1ULL<<22)|(1ULL<<20);
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_UBOXFIX)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int ivb_wbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint32_t flags = 0x0UL;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
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
        RegisterType type = counter_map[index].type;
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
                    VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister1,
                                    event->options[j].value & 0xFFFFFFFFULL, SETUP_WBOX_FILTER);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV,
                                    box_map[type].filterRegister1,
                                    event->options[j].value & 0xFFFFFFFFULL));
                    break;
                default:
                    break;
            }
        }
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_WBOX);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int ivb_ibox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint32_t flags = 0x0UL;
    PciDeviceIndex dev = counter_map[index].device;
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!HPMcheck(dev, cpu_id))
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
                    flags |= ((event->options[j].value & 0xFFULL) << 24);
                    break;
                default:
                    break;
            }
        }
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].configRegister, flags, SETUP_IBOX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}


int ivb_uncore_freeze(int cpu_id, PerfmonEventSet* eventSet)
{
    uint32_t freeze_reg = 0x0;
    if (cpuid_info.model == IVYBRIDGE_EP)
    {
        freeze_reg = MSR_UNC_U_PMON_GLOBAL_CTL;
    }
    else if (cpuid_info.model == IVYBRIDGE && ivy_cbox_setup == ivb_cbox_setup)
    {
        freeze_reg = MSR_UNC_PERF_GLOBAL_CTRL;
    }
    else
    {
        return 0;
    }
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (MEASURE_UNCORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, freeze_reg, LLU_CAST (1ULL<<31), FREEZE_UNCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, freeze_reg, (1ULL<<31)));
    }
    return 0;
}

int ivb_uncore_unfreeze(int cpu_id, PerfmonEventSet* eventSet)
{
    uint32_t unfreeze_reg = 0x0;
    uint32_t ovf_reg = 0x0;
    if (cpuid_info.model == IVYBRIDGE_EP)
    {
        unfreeze_reg = MSR_UNC_U_PMON_GLOBAL_CTL;
        ovf_reg = MSR_UNC_U_PMON_GLOBAL_STATUS;
    }
    else if (cpuid_info.model == IVYBRIDGE && ivy_cbox_setup == ivb_cbox_setup)
    {
        unfreeze_reg = MSR_UNC_PERF_GLOBAL_CTRL;
        ovf_reg = MSR_UNC_PERF_GLOBAL_OVF_CTRL;
    }
    else
    {
        return 0;
    }
    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (MEASURE_UNCORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, ovf_reg, LLU_CAST 0x0ULL, CLEAR_UNCORE_OVF)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, ovf_reg, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, unfreeze_reg, LLU_CAST (1ULL<<29), UNFREEZE_UNCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, unfreeze_reg, (1ULL<<29)));
    }
    return 0;
}


int perfmon_setupCounterThread_ivybridge(
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
    }
    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        ivb_uncore_freeze(cpu_id, eventSet);
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
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        switch (type)
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
                if (!cpuid_info.supportClientmem)
                {
                    ivb_pci_box_setup(cpu_id, index, event);
                }
                break;
            case MBOX1:
            case MBOX2:
            case MBOX3:
            case MBOX4:
            case MBOX5:
            case MBOX6:
            case MBOX7:
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
            case MBOX4FIX:
            case MBOX5FIX:
            case MBOX6FIX:
            case MBOX7FIX:
                ivb_mboxfix_setup(cpu_id, index, event);
                break;

            case SBOX0:
                ivb_sbox_setup(cpu_id, index, event, PCI_QPI_MASK_DEVICE_PORT_0);
                break;
            case SBOX1:
                ivb_sbox_setup(cpu_id, index, event, PCI_QPI_MASK_DEVICE_PORT_1);
                break;
            case SBOX2:
                ivb_sbox_setup(cpu_id, index, event, PCI_QPI_MASK_DEVICE_PORT_2);
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
                ivy_cbox_setup(cpu_id, index, event);
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
    for (int i=UNCORE;i<NUM_UNITS;i++)
    {
        if (haveLock && TESTTYPE(eventSet, i) && box_map[i].ctrlRegister != 0x0)
        {
            VERBOSEPRINTPCIREG(cpu_id, box_map[i].device, box_map[i].ctrlRegister, 0x0ULL, CLEAR_UNCORE_BOX_CTRL);
            HPMwrite(cpu_id, box_map[i].device, box_map[i].ctrlRegister, 0x0ULL);
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
    uint64_t fixed_flags = 0x0ULL;
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
            uint64_t counter2 = counter_map[index].counterRegister2;
            eventSet->events[i].threadCounter[thread_id].startData = 0;
            eventSet->events[i].threadCounter[thread_id].counterData = 0;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter1, 0x0ULL));
                    fixed_flags |= (1ULL<<(index-cpuid_info.perf_num_fixed_ctr));  /* enable counter */
                    break;

                case FIXED:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter1, 0x0ULL));
                    fixed_flags |= (1ULL<<(index+32));  /* enable fixed counter */
                    break;

                case POWER:
                    if (haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&tmp));
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST field64(tmp, 0, box_map[type].regWidth), START_POWER)
                        eventSet->events[i].threadCounter[thread_id].startData = field64(tmp, 0, box_map[type].regWidth);
                    }
                    break;
                case WBOX0FIX:
                case WBOX1FIX:
                    if (haveLock)
                    {
                        CHECK_PCI_READ_ERROR(HPMread(cpu_id, counter_map[index].device, counter1, &tmp));
                        eventSet->events[i].threadCounter[thread_id].startData = field64(tmp, 0, box_map[type].regWidth);
                    }
                    break;

                default:
                    if (type >= UNCORE && haveLock)
                    {
                        if (counter1 != 0x0)
                        {
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, box_map[type].device, counter1, 0x0ULL));
                            if (counter2 != 0x0)
                                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, box_map[type].device, counter2, 0x0ULL));
                        }
                    }
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = eventSet->events[i].threadCounter[thread_id].startData;
        }
    }

    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        ivb_uncore_unfreeze(cpu_id, eventSet);
    }

    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST fixed_flags, UNFREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, fixed_flags));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<63)|(1ULL<<62)|fixed_flags));
    }
    return 0;
}



uint64_t ivb_uncore_read(int cpu_id, RegisterIndex index, PerfmonEvent *event, int flags)
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
        return result;
    }
    if (box_map[type].isPci && !HPMcheck(dev, cpu_id))
    {
        return result;
    }

    CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
    VERBOSEPRINTPCIREG(cpu_id, dev, counter1, tmp, UNCORE_READ);

    if (flags & FREEZE_FLAG_CLEAR_CTR)
    {
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0U));
    }
    if (counter2 != 0x0)
    {
        result = (tmp<<32);
        tmp = 0x0ULL;
        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter2, &tmp));
        VERBOSEPRINTPCIREG(cpu_id, dev, counter2, tmp, UNCORE_READ);
        result += (tmp & 0xFFFFFFFF);
        if (flags & FREEZE_FLAG_CLEAR_CTR)
        {
            CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter2, 0x0U));
        }
    }
    else
    {
        result = tmp;
    }
    return result;
}

int ivb_uncore_overflow(int cpu_id, RegisterIndex index, PerfmonEvent *event,
                         int* overflows, uint64_t result, uint64_t cur_result,
                         int global_offset, int box_offset)
{
    int test_local = 0;
    uint64_t ovf_values = 0x0ULL;
    RegisterType type = counter_map[index].type;
    PciDeviceIndex dev = counter_map[index].device;
    event++;
    if (result < cur_result)
    {
        if (global_offset != -1)
        {
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV,
                                           MSR_UNC_U_PMON_GLOBAL_STATUS,
                                           &ovf_values));
            if (ovf_values & (1ULL<<global_offset))
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
            if (ovf_values & (1ULL<<box_offset))
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
    return 0;
}

int perfmon_stopCountersThread_ivybridge(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t counter_result = 0x0ULL;
    int haveLock = 0;
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
        ivb_uncore_freeze(cpu_id, eventSet);
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
            counter_result= 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            PerfmonEvent *event = &(eventSet->events[i].event);
            PciDeviceIndex dev = counter_map[index].device;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t* current = &(eventSet->events[i].threadCounter[thread_id].counterData);
            int* overflows = &(eventSet->events[i].threadCounter[thread_id].overflows);
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                    if (counter_result < *current)
                    {
                        uint64_t ovf_values = 0x0ULL;
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, MSR_PERF_GLOBAL_STATUS, &ovf_values));
                        if (ovf_values & (1<<(index-cpuid_info.perf_num_fixed_ctr)))
                        {
                            (*overflows)++;
                        }
                    }
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                    if (counter_result < *current)
                    {
                        uint64_t ovf_values = 0x0ULL;
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, MSR_PERF_GLOBAL_STATUS, &ovf_values));
                        if (ovf_values & (1<<(index+32)))
                        {
                            (*overflows)++;
                        }
                    }
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
                        *current = field64(counter_result, 0, box_map[type].regWidth);
                    }
                    break;

                case THERMAL:
                    CHECK_TEMP_READ_ERROR(thermal_read(cpu_id, (uint32_t*)&counter_result));
                    break;

                case SBOX0FIX:
                case SBOX1FIX:
                case SBOX2FIX:
                    if (haveLock && HPMcheck(dev, cpu_id))
                    {
                        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_SBOX_FIXED)
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
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_SBOX_FIXED_REAL)
                        eventSet->events[i].threadCounter[thread_id].startData = 0;
                    }
                    break;

                case MBOX0:
                    if (haveLock)
                    {
                        if (!cpuid_info.supportClientmem)
                        {
                            counter_result = ivb_uncore_read(cpu_id, index, event, FREEZE_FLAG_CLEAR_CTR);
                            ivb_uncore_overflow(cpu_id, index, event, overflows, counter_result,
                                                *current, box_map[type].ovflOffset, getCounterTypeOffset(index)+1);
                        }
                        else
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
                    if (haveLock)
                    {
                        counter_result = ivb_uncore_read(cpu_id, index, event, FREEZE_FLAG_CLEAR_CTR);
                        ivb_uncore_overflow(cpu_id, index, event, overflows, counter_result,
                                            *current, box_map[type].ovflOffset, getCounterTypeOffset(index)+1);
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
                        counter_result = ivb_uncore_read(cpu_id, index, event, FREEZE_FLAG_CLEAR_CTR);
                        ivb_uncore_overflow(cpu_id, index, event, overflows, counter_result,
                                            *current, box_map[type].ovflOffset, 0);
                    }
                    break;
                case WBOX0FIX:
                case WBOX1FIX:
                    if (haveLock)
                    {
                        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                        if (counter_result < *current)
                        {
                            (*overflows)++;
                        }
                    }
                    break;

                case IBOX1:
                    counter_result = ivb_uncore_read(cpu_id, index, event, FREEZE_FLAG_CLEAR_CTR);
                    ivb_uncore_overflow(cpu_id, index, event, overflows, counter_result,
                                        *current, -1, getCounterTypeOffset(index)+2);
                    break;

                case SBOX0:
                case SBOX1:
                case SBOX2:
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
                case UBOX:
                case UBOXFIX:
                case BBOX0:
                case BBOX1:
                case WBOX:
                case PBOX:
                case RBOX0:
                case RBOX1:
                case RBOX2:
                case IBOX0:
                    if (haveLock)
                    {
                        counter_result = ivb_uncore_read(cpu_id, index, event, FREEZE_FLAG_CLEAR_CTR);
                        ivb_uncore_overflow(cpu_id, index, event, overflows, counter_result,
                                            *current, box_map[type].ovflOffset, getCounterTypeOffset(index));
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

int perfmon_readCountersThread_ivybridge(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t counter_result = 0x0ULL;
    uint64_t pmc_flags = 0x0ULL;
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (MEASURE_CORE(eventSet))
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, &pmc_flags));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        ivb_uncore_freeze(cpu_id, eventSet);
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
                    if (counter_result < *current)
                    {
                        uint64_t ovf_values = 0x0ULL;
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_STATUS, &ovf_values));
                        if (ovf_values & (1<<(index-cpuid_info.perf_num_fixed_ctr)))
                        {
                            (*overflows)++;
                        }
                    }
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
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
                    CHECK_TEMP_READ_ERROR(thermal_read(cpu_id, (uint32_t*)&counter_result));
                    break;

                case SBOX0FIX:
                case SBOX1FIX:
                case SBOX2FIX:
                    if (haveLock && HPMcheck(dev, cpu_id))
                    {
                        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_SBOX_FIXED)
                        if (eventSet->events[i].event.eventId == 0x00)
                        {
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
                        }
                        else if (eventSet->events[i].event.eventId == 0x01)
                        {
                            counter_result = extractBitField(counter_result,1,4);
                        }
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_SBOX_FIXED_REAL)
                        eventSet->events[i].threadCounter[thread_id].startData = 0;
                    }
                    break;

                case MBOX0:
                    if (haveLock)
                    {
                        if (!cpuid_info.supportClientmem)
                        {
                            counter_result = ivb_uncore_read(cpu_id, index, event, FREEZE_FLAG_ONLYFREEZE);
                            ivb_uncore_overflow(cpu_id, index, event, overflows, counter_result,
                                                *current, box_map[type].ovflOffset, getCounterTypeOffset(index)+1);
                        }
                        else
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
                    }
                    break;
                case MBOX1:
                case MBOX2:
                case MBOX3:
                case MBOX4:
                case MBOX5:
                case MBOX6:
                case MBOX7:
                    if (haveLock)
                    {
                        counter_result = ivb_uncore_read(cpu_id, index, event, FREEZE_FLAG_ONLYFREEZE);
                        ivb_uncore_overflow(cpu_id, index, event, overflows, counter_result,
                                            *current, box_map[type].ovflOffset, getCounterTypeOffset(index)+1);
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
                        counter_result = ivb_uncore_read(cpu_id, index, event, FREEZE_FLAG_ONLYFREEZE);
                        ivb_uncore_overflow(cpu_id, index, event, overflows, counter_result,
                                            *current, box_map[type].ovflOffset, 0);
                    }
                    break;
                case WBOX0FIX:
                case WBOX1FIX:
                    if (haveLock)
                    {
                        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                        if (counter_result < *current)
                        {
                            (*overflows)++;
                        }
                    }
                    break;

                case IBOX1:
                    if (haveLock)
                    {
                        counter_result = ivb_uncore_read(cpu_id, index, event, FREEZE_FLAG_ONLYFREEZE);
                        ivb_uncore_overflow(cpu_id, index, event, overflows, counter_result,
                                            *current, -1, getCounterTypeOffset(index)+2);
                    }
                    break;

                case SBOX0:
                case SBOX1:
                case SBOX2:
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
                case UBOX:
                case UBOXFIX:
                case BBOX0:
                case BBOX1:
                case WBOX:
                case PBOX:
                case RBOX0:
                case RBOX1:
                case RBOX2:
                case IBOX0:
                    if (haveLock)
                    {
                        counter_result = ivb_uncore_read(cpu_id, index, event, FREEZE_FLAG_ONLYFREEZE);
                        ivb_uncore_overflow(cpu_id, index, event, overflows, counter_result,
                                            *current, box_map[type].ovflOffset, getCounterTypeOffset(index));
                    }
                    break;

                default:
                    /* should never be reached */
                    break;
            }
            *current = field64(counter_result, 0, box_map[type].regWidth);
        }
    }

    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        ivb_uncore_unfreeze(cpu_id, eventSet);
    }

    if (MEASURE_CORE(eventSet))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, pmc_flags));
    }
    return 0;
}


int perfmon_finalizeCountersThread_ivybridge(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int haveTileLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t ovf_values_core = (1ULL<<63)|(1ULL<<62);

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

        switch(type)
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
            VERBOSEPRINTPCIREG(cpu_id, dev, reg, 0x0ULL, CLEAR_CTL);
            CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, 0x0ULL));
            if (type >= SBOX0 && type <= SBOX3)
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, 0x0ULL));
            VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL, CLEAR_CTR);
            CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL));
            if (type >= SBOX0 && type <= SBOX3)
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL));
            if (counter_map[index].counterRegister2 != 0x0)
            {
                VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].counterRegister2, 0x0ULL, CLEAR_CTR);
                CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister2, 0x0ULL));
                if (type >= SBOX0 && type <= SBOX3)
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL));
            }
        }
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
    }
    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        if (cpuid_info.model == IVYBRIDGE_EP)
        {
            VERBOSEPRINTREG(cpu_id, MSR_UNC_U_PMON_GLOBAL_STATUS, LLU_CAST 0x0ULL, CLEAR_UNCORE_OVF)
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_U_PMON_GLOBAL_STATUS, 0x0ULL));
            VERBOSEPRINTREG(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, LLU_CAST 0x0ULL, CLEAR_UNCORE_CTRL)
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_U_PMON_GLOBAL_CTL, 0x0ULL));
        }
        else if (cpuid_info.model == IVYBRIDGE)
        {
            VERBOSEPRINTREG(cpu_id, MSR_UNC_PERF_GLOBAL_OVF_CTRL, LLU_CAST 0x0ULL, CLEAR_UNCORE_OVF)
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_PERF_GLOBAL_OVF_CTRL, 0x0ULL));
            VERBOSEPRINTREG(cpu_id, MSR_UNC_PERF_GLOBAL_CTRL, LLU_CAST 0x0ULL, CLEAR_UNCORE_CTRL)
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_PERF_GLOBAL_CTRL, 0x0ULL));
        }
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
