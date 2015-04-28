/*
 * =======================================================================================
 *
 *      Filename:  perfmon_sandybridge.h
 *
 *      Description:  Header File of perfmon module for Sandy Bridge.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *               Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2013 Jan Treibig and Thomas Roehl
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

#include <perfmon_sandybridge_events.h>
#include <perfmon_sandybridgeEP_events.h>
#include <perfmon_sandybridge_counters.h>
#include <error.h>
#include <affinity.h>

static int perfmon_numCountersSandybridgeEP = NUM_COUNTERS_SANDYBRIDGEEP;
static int perfmon_numCoreCountersSandybridgeEP = NUM_COUNTERS_CORE_SANDYBRIDGEEP;
static int perfmon_numArchEventsSandybridgeEP = NUM_ARCH_EVENTS_SANDYBRIDGEEP;
static int perfmon_numCountersSandybridge = NUM_COUNTERS_SANDYBRIDGE;
static int perfmon_numCoreCountersSandybridge = NUM_COUNTERS_CORE_SANDYBRIDGE;
static int perfmon_numArchEventsSandybridge = NUM_ARCH_EVENTS_SANDYBRIDGE;


int perfmon_init_sandybridge(int cpu_id)
{
    lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &tile_lock[affinity_thread2tile_lookup[cpu_id]], cpu_id);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL));
    return 0;
}

uint32_t snb_fixed_setup(RegisterIndex index, PerfmonEvent *event)
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
            case EVENT_OPTION_ANYTHREAD:
                flags |= (1ULL<<(2+(index*4)));
                break;
            default:
                break;
        }
    }
    return flags;
}

int snb_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    int haveTileLock = 0;
    uint32_t flags = 0x0U;
    uint64_t offcore_flags = 0x0ULL;

    flags |= (1ULL<<22);  /* enable flag */
    flags |= (1ULL<<16);  /* user mode flag */

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
                case EVENT_OPTION_ANYTHREAD:
                    flags |= (1ULL<<21);
                    break;
                case EVENT_OPTION_INVERT:
                    flags |= (1ULL<<23);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= ((event->options[j].value & 0xFFULL)<<24);
                    break;
                case EVENT_OPTION_MATCH0:
                    offcore_flags |= (event->options[j].value & 0x8FFFULL);
                    break;
                case EVENT_OPTION_MATCH1:
                    offcore_flags |= ((event->options[j].value & 0x3F807FULL)<<16);
                    break;
                default:
                    break;
            }
        }
    }
    if (tile_lock[affinity_thread2tile_lookup[cpu_id]] == cpu_id)
    {
        haveTileLock = 1;
    }
    if ((haveTileLock) && (event->eventId == 0xB7))
    {
        if ((event->cfgBits != 0xFF) && (event->cmask != 0xFF))
        {
            offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);
        }
        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, offcore_flags));
    }
    else if ((haveTileLock) && (event->eventId == 0xBB))
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

int snb_mbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t flags = 0x0U;
    PciDeviceIndex dev = counter_map[index].device;
    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!pci_checkDevice(dev, cpu_id))
    {
        return -ENODEV;
    }

    flags = (1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;
    for(j=0;j<event->numberOfOptions;j++)
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
                flags |= ((event->options[j].value & 0xFFULL)<<24);
                break;
            default:
                break;
        }
    }
    VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].configRegister, LLU_CAST flags, SETUP_MBOX);
    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
    return 0;
}


uint32_t snb_cbox_filter(PerfmonEvent *event)
{
    int j;
    uint32_t ret = 0x0;
    uint64_t mask = 0x0ULL;
    int set_state = 0;

    for(j=0;j<event->numberOfOptions;j++)
    {
        switch (event->options[j].type)
        {
            case EVENT_OPTION_OPCODE:
                if ((event->options[j].value == 0x180) ||
                    (event->options[j].value == 0x181) ||
                    (event->options[j].value == 0x182) ||
                    (event->options[j].value == 0x187) ||
                    (event->options[j].value == 0x18C) ||
                    (event->options[j].value == 0x18D) ||
                    (event->options[j].value == 0x190) ||
                    (event->options[j].value == 0x191) ||
                    (event->options[j].value == 0x192) ||
                    (event->options[j].value == 0x194) ||
                    (event->options[j].value == 0x195) ||
                    (event->options[j].value == 0x19C) ||
                    (event->options[j].value == 0x19E) ||
                    (event->options[j].value == 0x1C4) ||
                    (event->options[j].value == 0x1C5) ||
                    (event->options[j].value == 0x1C8) ||
                    (event->options[j].value == 0x1E4) ||
                    (event->options[j].value == 0x1E5) ||
                    (event->options[j].value == 0x1E6))
                {
                    ret |= ((event->options[j].value & 0x1FFULL) << 23);
                }
                else
                {
                    ERROR_PRINT(Invalid value 0x%llx for opcode option, LLU_CAST event->options[j].value);
                }
                break;
            case EVENT_OPTION_STATE:
                if (event->options[j].value & 0x1F)
                {
                    ret |= ((event->options[j].value & 0x1FULL) << 18);
                    set_state = 1;
                }
                else
                {
                    ERROR_PRINT(Invalid value 0x%llx for state option, LLU_CAST event->options[j].value);
                }
                break;
            case EVENT_OPTION_NID:
                mask = 0x0ULL;
                for (int i=0; i<affinityDomains.numberOfNumaDomains;i++)
                    mask |= (1ULL<<i);
                if (event->options[j].value & mask)
                {
                    ret |= ((event->options[j].value & 0xFFULL) << 10);
                }
                else
                {
                    ERROR_PRINT(Invalid value 0x%llx for node id option, LLU_CAST event->options[j].value);
                }
                break;
            case EVENT_OPTION_TID:
                if (event->options[j].value <= 0xF)
                {
                    ret |= (event->options[j].value & 0x1FULL);
                }
                else
                {
                    ERROR_PRINT(Invalid value 0x%llx for thread id option, LLU_CAST event->options[j].value);
                }
                break;
            default:
                break;
        }
    }
    if ((event->eventId == 0x34) && (set_state == 0))
    {
        ret |= (0x1FULL << 18);
    }
    return ret;
}

int snb_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t flags = 0x0U;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags |= (1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;

    if (event->numberOfOptions > 0)
    {
        uint32_t optflags = snb_cbox_filter(event);
        uint32_t filter_reg = box_map[counter_map[index].type].filterRegister1;
        if (optflags != 0x0U)
        {
            VERBOSEPRINTREG(cpu_id, filter_reg, LLU_CAST optflags, SETUP_CBOX_FILTER);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, filter_reg, optflags));
        }
    }

    for(j=0;j<event->numberOfOptions;j++)
    {
        switch (event->options[j].type)
        {
            case EVENT_OPTION_TID:
                flags |= (1ULL<<19);
                break;
            case EVENT_OPTION_EDGE:
                flags |= (1ULL<<18);
                break;
            case EVENT_OPTION_INVERT:
                flags |= (1ULL<<23);
                break;
            case EVENT_OPTION_THRESHOLD:
                flags |= (event->options[j].value & 0xFFULL)<<24;
                break;
            default:
                break;
        }
    }
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_CBOX);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    return 0;
}


int snb_ubox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t flags = 0x0U;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags |= (1ULL<<17);
    flags |= (event->umask<<8) + event->eventId;
    for(j=0;j<event->numberOfOptions;j++)
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
                flags |= ((event->options[j].value & 0x1FULL) << 24);
                break;
            default:
                break;
        }
    }
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_UBOX)
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    return 0;
}

int snb_bbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t flags = 0x0U;
    uint64_t match = 0x0ULL;
    PciDeviceIndex dev = counter_map[index].device;
    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!pci_checkDevice(dev, cpu_id))
    {
        return -ENODEV;
    }

    flags = (1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;
    for(j=0;j<event->numberOfOptions;j++)
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
                flags |= ((event->options[j].value & 0xFFULL) << 24);
                break;
            case EVENT_OPTION_OPCODE:
                VERBOSEPRINTPCIREG(cpu_id, dev, PCI_UNC_HA_PMON_OPCODEMATCH,
                                    LLU_CAST (event->options[j].value & 0x3FULL), SETUP_BBOX_OPCODE);
                CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, PCI_UNC_HA_PMON_OPCODEMATCH,
                                    (event->options[j].value & 0x3FULL)));
                break;
            case EVENT_OPTION_MATCH0:
                match = event->options[j].value & 0xFFFFFFC0ULL;
                VERBOSEPRINTPCIREG(cpu_id, dev, PCI_UNC_HA_PMON_ADDRMATCH0, LLU_CAST match, SETUP_BBOX_MATCH0);
                CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, PCI_UNC_HA_PMON_ADDRMATCH0, match));
                match = (event->options[j].value >> 32) & 0x3FFFULL;
                VERBOSEPRINTPCIREG(cpu_id, dev, PCI_UNC_HA_PMON_ADDRMATCH1, LLU_CAST match, SETUP_BBOX_MATCH1);
                CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, PCI_UNC_HA_PMON_ADDRMATCH1, match));
                break;
            default:
                break;
        }
    }
    VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].configRegister, LLU_CAST flags, SETUP_BBOX);
    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev,  counter_map[index].configRegister, flags));
    return 0;
}


int snb_wbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t flags = 0x0U;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags = (1ULL<<22);
    flags |= event->eventId & 0xFF;
    for(j=0;j<event->numberOfOptions;j++)
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
                flags |= ((event->options[j].value & 0x1FULL) << 24);
                break;
            case EVENT_OPTION_OCCUPANCY:
                flags |= ((event->options[j].value & 0x3ULL) << 14);
                break;
            case EVENT_OPTION_OCCUPANCY_EDGE:
                flags |= (1ULL<<31);
                break;
            case EVENT_OPTION_OCCUPANCY_INVERT:
                flags |= (1ULL<<30);
                break;
            case EVENT_OPTION_OCCUPANCY_FILTER:
                VERBOSEPRINTREG(cpu_id, MSR_UNC_PCU_PMON_BOX_FILTER, LLU_CAST event->options[j].value & 0xFFFFFFFFULL, SETUP_WBOX_FILTER);
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_PCU_PMON_BOX_FILTER, event->options[j].value & 0xFFFFFFFFULL));
            default:
                break;
        }
    }
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_WBOX);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    return 0;
}

int snb_sbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event, PciDeviceIndex filterdev)
{
    int j;
    uint32_t flags = 0x0U;
    PciDeviceIndex dev = counter_map[index].device;
    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!pci_checkDevice(dev, cpu_id))
    {
        return -ENODEV;
    }

    flags = (1ULL<<22);
    flags |= event->cfgBits;
    flags |= (event->umask<<8) + event->eventId;
    for(j=0;j<event->numberOfOptions;j++)
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
                flags |= ((event->options[j].value & 0xFFULL) << 24);
                break;
            case EVENT_OPTION_MATCH0:
                if (pci_checkDevice(filterdev, cpu_id))
                {
                    VERBOSEPRINTPCIREG(cpu_id, filterdev, PCI_UNC_QPI_PMON_MATCH_0,
                                    event->options[j].value & 0x8003FFF8ULL, SETUP_SBOX_MATCH0);
                    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, filterdev, PCI_UNC_QPI_PMON_MATCH_0,
                                    event->options[j].value & 0x8003FFF8ULL));
                }
                else
                {
                    DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, Filtering for counter %s cannot be applied. PCI device not available, counter_map[index].key);
                }
                break;
            case EVENT_OPTION_MATCH1:
                if (pci_checkDevice(filterdev, cpu_id))
                {
                    VERBOSEPRINTPCIREG(cpu_id, filterdev, PCI_UNC_QPI_PMON_MATCH_1,
                                    event->options[j].value & 0x000F000FULL, SETUP_SBOX_MATCH1);
                    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, filterdev, PCI_UNC_QPI_PMON_MATCH_1,
                                    event->options[j].value & 0x000F000FULL));
                }
                else
                {
                    DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, Filtering for counter %s cannot be applied. PCI device not available, counter_map[index].key);
                }
                break;
            case EVENT_OPTION_MASK0:
                if (pci_checkDevice(filterdev, cpu_id))
                {
                    VERBOSEPRINTPCIREG(cpu_id, filterdev, PCI_UNC_QPI_PMON_MASK_0,
                                    event->options[j].value & 0x8003FFF8ULL, SETUP_SBOX_MASK0);
                    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, filterdev, PCI_UNC_QPI_PMON_MASK_0,
                                    event->options[j].value & 0x8003FFF8ULL));
                }
                else
                {
                    DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, Filtering for counter %s cannot be applied. PCI device not available, counter_map[index].key);
                }
                break;
            case EVENT_OPTION_MASK1:
                if (pci_checkDevice(filterdev, cpu_id))
                {
                    VERBOSEPRINTPCIREG(cpu_id, filterdev, PCI_UNC_QPI_PMON_MASK_1,
                                    event->options[j].value & 0x000F000FULL, SETUP_SBOX_MASK1);
                    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, filterdev, PCI_UNC_QPI_PMON_MASK_1,
                                    event->options[j].value & 0x000F000FULL));
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
    VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].configRegister, LLU_CAST flags, SETUP_SBOX);
    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev,  counter_map[index].configRegister, flags));
    return 0;
}



int snb_rbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t flags = 0x0U;
    PciDeviceIndex dev = counter_map[index].device;
    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!pci_checkDevice(dev, cpu_id))
    {
        return -ENODEV;
    }

    flags = (1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;
    for(j=0;j<event->numberOfOptions;j++)
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
                flags |= ((event->options[j].value & 0xFFULL) << 24);
                break;
            default:
                break;
        }
    }
    VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].configRegister, LLU_CAST flags, SETUP_RBOX)
    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
    return 0;
}

int snb_pbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t flags = 0x0U;
    PciDeviceIndex dev = counter_map[index].device;
    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (!pci_checkDevice(dev, cpu_id))
    {
        return -ENODEV;
    }

    flags = (1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;
    for(j=0;j<event->numberOfOptions;j++)
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
                flags |= ((event->options[j].value & 0xFFULL) << 24);
                break;
            default:
                break;
        }
    }
    VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].configRegister, LLU_CAST flags, SETUP_PBOX)
    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
    return 0;
}

// Macros to stop counting and reset control registers
// FREEZE(_AND_RESET_CTL) uses central box register to freeze (bit 8 + 16) and bit 1 to reset control registers
#define SNB_FREEZE_AND_RESET_CTL_BOX(id) \
    if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(id))) \
    { \
        VERBOSEPRINTREG(cpu_id, box_map[id].ctrlRegister, 0x10101U, FREEZE_AND_RESET_CTL_BOX_##id) \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[id].ctrlRegister, 0x10101ULL)); \
    }

#define SNB_FREEZE_BOX(id) \
    if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(id))) \
    { \
        VERBOSEPRINTREG(cpu_id, box_map[id].ctrlRegister, 0x10100U, FREEZE_AND_RESET_CTL_BOX_##id) \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[id].ctrlRegister, 0x10100ULL)); \
    }

// FREEZE(_AND_RESET_CTL)_PCI uses central box register to freeze (bit 8 + 16) and bit 1 to reset control registers
// Checks whether PCI device exists, because this is the first operation we do on the devices
#define SNB_FREEZE_AND_RESET_CTL_PCI_BOX(id) \
    if (haveLock && \
        (eventSet->regTypeMask & (REG_TYPE_MASK(id))) && \
        (pci_checkDevice(box_map[id].device, cpu_id) == 0)) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, box_map[id].device, box_map[id].ctrlRegister, 0x10101ULL, FREEZE_AND_RESET_CTL_PCI_BOX_##id); \
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, box_map[id].device, box_map[id].ctrlRegister, 0x10101ULL)); \
    }

#define SNB_FREEZE_PCI_BOX(id) \
    if (haveLock && \
        (eventSet->regTypeMask & (REG_TYPE_MASK(id))) && \
        (pci_checkDevice(box_map[id].device, cpu_id) == 0)) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, box_map[id].device, box_map[id].ctrlRegister, 0x10100ULL, FREEZE_PCI_BOX_##id) \
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, box_map[id].device, box_map[id].ctrlRegister, 0x10100ULL)); \
    }

// MBOX*FIX have a slightly different scheme, setting the whole register to 0 freeze the counter
#define SNB_FREEZE_MBOXFIX(number) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX##number##FIX))) && \
                    (pci_checkDevice(PCI_IMC_DEVICE_0_CH_##number, cpu_id))) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, PCI_IMC_DEVICE_0_CH_##number, PCI_UNC_MC_PMON_FIXED_CTL, 0x0ULL, FREEZE_MBOXFIX##number) \
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, PCI_IMC_DEVICE_0_CH_##number,  PCI_UNC_MC_PMON_FIXED_CTL, 0x0ULL)); \
    }



int perfmon_setupCounterThread_sandybridge(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int i;
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t fixed_flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, 0x0ULL));
    }
    SNB_FREEZE_BOX(CBOX0);
    SNB_FREEZE_BOX(CBOX1);
    SNB_FREEZE_BOX(CBOX2);
    SNB_FREEZE_BOX(CBOX3);
    SNB_FREEZE_BOX(CBOX4);
    SNB_FREEZE_BOX(CBOX5);
    SNB_FREEZE_BOX(CBOX6);
    SNB_FREEZE_BOX(CBOX7);

    SNB_FREEZE_PCI_BOX(MBOX0);
    SNB_FREEZE_PCI_BOX(MBOX1);
    SNB_FREEZE_PCI_BOX(MBOX2);
    SNB_FREEZE_PCI_BOX(MBOX3);

    SNB_FREEZE_MBOXFIX(0);
    SNB_FREEZE_MBOXFIX(1);
    SNB_FREEZE_MBOXFIX(2);
    SNB_FREEZE_MBOXFIX(3);

    SNB_FREEZE_PCI_BOX(SBOX0);
    SNB_FREEZE_PCI_BOX(SBOX1);

    SNB_FREEZE_PCI_BOX(RBOX0);
    SNB_FREEZE_PCI_BOX(RBOX1);

    SNB_FREEZE_PCI_BOX(PBOX);

    SNB_FREEZE_PCI_BOX(BBOX0);
    SNB_FREEZE_BOX(WBOX);

    for (i=0;i < eventSet->numberOfEvents;i++)
    {
        flags = 0x0ULL;
        RegisterType type = eventSet->events[i].type;
        if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
        {
            continue;
        }
        PerfmonEvent *event = &(eventSet->events[i].event);
        RegisterIndex index = eventSet->events[i].index;
        PciDeviceIndex dev = counter_map[index].device;
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        switch (type)
        {
            case PMC:
                snb_pmc_setup(cpu_id, index, event);
                break;

            case FIXED:
                /* initialize fixed counters
                 * FIXED 0: Instructions retired
                 * FIXED 1: Clocks unhalted core
                 * FIXED 2: Clocks unhalted ref */
                fixed_flags |= snb_fixed_setup(index,event);
                /* Written in the end of function for all fixed purpose registers */
                break;

            case POWER:
                break;

            case MBOX0:
            case MBOX1:
            case MBOX2:
            case MBOX3:
                snb_mbox_setup(cpu_id, index, event);
                break;

            case MBOX0FIX:
                break;
            case MBOX1FIX:
                break;
            case MBOX2FIX:
                break;
            case MBOX3FIX:
                break;

            case CBOX0:
            case CBOX1:
            case CBOX2:
            case CBOX3:
            case CBOX4:
            case CBOX5:
            case CBOX6:
            case CBOX7:
                snb_cbox_setup(cpu_id, index, event);
                break;

            case UBOX:
                snb_ubox_setup(cpu_id, index, event);
                break;
                
            case UBOXFIX:
                VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST 0x0ULL, SETUP_UBOXFIX)
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, 0x0ULL));
                break;

            case SBOX0:
                snb_sbox_setup(cpu_id, index, event, PCI_QPI_MASK_DEVICE_PORT_0);
                break;
            case SBOX1:
                snb_sbox_setup(cpu_id, index, event, PCI_QPI_MASK_DEVICE_PORT_1);
                break;

            case SBOX0FIX:
            case SBOX1FIX:
                break;

            case BBOX0:
                snb_bbox_setup(cpu_id, index, event);
                break;

            case WBOX:
                snb_wbox_setup(cpu_id, index, event);
                break;

            case RBOX0:
            case RBOX1:
                snb_rbox_setup(cpu_id, index, event);
                break;

            case PBOX:
                snb_pbox_setup(cpu_id, index, event);
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


// Macros for MSR HPM counters
// UNFREEZE(_AND_RESET_CTR) uses the central box registers to unfreeze and reset the counter registers
#define SNB_UNFREEZE_BOX(id) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) { \
        VERBOSEPRINTREG(cpu_id, box_map[id].ctrlRegister, LLU_CAST 0x0ULL, UNFREEZE_BOX_##id) \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[id].ctrlRegister, 0x0ULL)); \
    }

#define SNB_UNFREEZE_AND_RESET_CTR_BOX(id) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) { \
        VERBOSEPRINTREG(cpu_id, box_map[id].ctrlRegister, LLU_CAST 0x2ULL, UNFREEZE_BOX_##id) \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[id].ctrlRegister, 0x2ULL)); \
    }

// ENABLE(_AND_RESET_CTR) uses the control registers to enable (bit 22) and reset the counter registers (bit 19)
#define SNB_ENABLE_BOX(id, reg) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, reg, &tmp)); \
        tmp |= (1ULL<<22); \
        VERBOSEPRINTREG(cpu_id, reg, LLU_CAST tmp, ENABLE_BOX_##id) \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, tmp)); \
    }

#define SNB_ENABLE_AND_RESET_CTR_BOX(id) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, box_map[id].ctrlRegister, &tmp)); \
        tmp |= (1ULL<<22)|(1ULL<<17); \
        VERBOSEPRINTREG(cpu_id, box_map[id].ctrlRegister, LLU_CAST tmp, ENABLE_AND_RESET_CTR_BOX_##id) \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[id].ctrlRegister, tmp)); \
    }

// UNFREEZE(_AND_RESET_CTR)_PCI is similar to MSR UNFREEZE but for PCI devices
#define SNB_UNFREEZE_PCI_BOX(id) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id))) \
                && (pci_checkDevice(box_map[id].device, cpu_id))) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, box_map[id].device, box_map[id].ctrlRegister, LLU_CAST 0x0ULL, UNFREEZE_PCI_BOX_##id) \
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, box_map[id].device, box_map[id].ctrlRegister, 0x0ULL)); \
    }
#define SNB_UNFREEZE_AND_RESET_CTR_PCI_BOX(id) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id))) \
                && (pci_checkDevice(box_map[id].device, cpu_id))) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, box_map[id].device, box_map[id].ctrlRegister, LLU_CAST 0x2ULL, UNFREEZE_AND_RESET_CTR_PCI_BOX_##id) \
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, box_map[id].device, box_map[id].ctrlRegister, 0x2ULL)); \
    }

// UNFREEZE(_AND_RESET_CTR)_MBOXFIX is kind of ENABLE for PCI but uses bit 19 for reset
#define SNB_UNFREEZE_AND_RESET_CTR_MBOXFIX(number) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX##number##FIX))) && \
                    (pci_checkDevice(PCI_IMC_DEVICE_0_CH_##number, cpu_id))) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, PCI_IMC_DEVICE_0_CH_##number, \
                PCI_UNC_MC_PMON_FIXED_CTL, LLU_CAST (1ULL<<22)|(1ULL<<19), UNFREEZE_AND_RESET_CTR_MBOX##number##FIX) \
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, PCI_IMC_DEVICE_0_CH_##number, PCI_UNC_MC_PMON_FIXED_CTL, (1ULL<<22)|(1ULL<<19))); \
    }
#define SNB_UNFREEZE_MBOXFIX(number) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX##number##FIX))) && \
                    (pci_checkDevice(PCI_IMC_DEVICE_0_CH_##number, cpu_id))) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, PCI_IMC_DEVICE_0_CH_##number, \
                PCI_UNC_MC_PMON_FIXED_CTL, LLU_CAST (1ULL<<22), UNFREEZE_MBOXFIX##id) \
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, PCI_IMC_DEVICE_0_CH_##number,  PCI_UNC_MC_PMON_FIXED_CTL, (1ULL<<22))); \
    }

int perfmon_startCountersThread_sandybridge(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t tmp = 0x0ULL;
    uint64_t flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterType type = eventSet->events[i].type;
            if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
            {
                continue;
            }
            tmp = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;
            PciDeviceIndex dev = counter_map[index].device;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter1, 0x0ULL));
                    flags |= (1<<(index-cpuid_info.perf_num_fixed_ctr));  /* enable counter */
                    break;

                case FIXED:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter1, 0x0ULL));
                    flags |= (1ULL<<(index+32));  /* enable fixed counter */
                    break;

                case POWER:
                    if(haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&tmp));
                        eventSet->events[i].threadCounter[thread_id].startData = tmp;
                    }
                    break;

                case MBOX0:
                case MBOX1:
                case MBOX2:
                case MBOX3:
                    if (haveLock && pci_checkDevice(dev, cpu_id))
                    {
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter2, 0x0ULL));
                    }
                    break;

                case MBOX0FIX:
                case MBOX1FIX:
                case MBOX2FIX:
                case MBOX3FIX:
                    /*if (haveLock && pci_checkDevice(dev, cpu_id))
                    {
                        tmp = 0x0ULL;
                        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, counter1, &tmp));
                        eventSet->events[i].threadCounter[thread_id].startData = tmp;
                    }*/
                    break;


                case SBOX0:
                case SBOX1:
                case SBOX0FIX:
                case SBOX1FIX:
                case CBOX0:
                case CBOX1:
                case CBOX2:
                case CBOX3:
                case CBOX4:
                case CBOX5:
                case CBOX6:
                case CBOX7:
                    break;

                case UBOX:
                    //SNB_ENABLE_AND_RESET_CTR_BOX(UBOX);
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, reg, &tmp));
                        tmp |= (1ULL<<22)|(1ULL<<17);
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, tmp));
                    }
                    break;
                case UBOXFIX:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter1, 0x0ULL));
                    SNB_ENABLE_BOX(UBOXFIX, reg);
                    break;

                case BBOX0:
                    if (haveLock && pci_checkDevice(dev, cpu_id))
                    {
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter2, 0x0ULL));
                    }
                    break;

                case WBOX:
                    if (haveLock)
                    {
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_PCU_PMON_BOX_FILTER, 0x0U));
                    }
                    break;
                case WBOX0FIX:
                case WBOX1FIX:
                    if(haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &tmp));
                        eventSet->events[i].threadCounter[thread_id].startData = field64(tmp, 0, box_map[WBOX0FIX].regWidth);
                    }
                    break;
                default:
                    break;
            }
        }
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, UNFREEZE_PMC_OR_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, flags));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, 0x300000000ULL|flags));
    }
    SNB_UNFREEZE_AND_RESET_CTR_BOX(CBOX0);
    SNB_UNFREEZE_AND_RESET_CTR_BOX(CBOX1);
    SNB_UNFREEZE_AND_RESET_CTR_BOX(CBOX2);
    SNB_UNFREEZE_AND_RESET_CTR_BOX(CBOX3);
    SNB_UNFREEZE_AND_RESET_CTR_BOX(CBOX4);
    SNB_UNFREEZE_AND_RESET_CTR_BOX(CBOX5);
    SNB_UNFREEZE_AND_RESET_CTR_BOX(CBOX6);
    SNB_UNFREEZE_AND_RESET_CTR_BOX(CBOX7);
    SNB_UNFREEZE_AND_RESET_CTR_PCI_BOX(SBOX0);
    SNB_UNFREEZE_AND_RESET_CTR_PCI_BOX(SBOX1);
    SNB_UNFREEZE_PCI_BOX(MBOX0);
    SNB_UNFREEZE_PCI_BOX(MBOX1);
    SNB_UNFREEZE_PCI_BOX(MBOX2);
    SNB_UNFREEZE_PCI_BOX(MBOX3);
    SNB_UNFREEZE_AND_RESET_CTR_MBOXFIX(0);
    SNB_UNFREEZE_AND_RESET_CTR_MBOXFIX(1);
    SNB_UNFREEZE_AND_RESET_CTR_MBOXFIX(2);
    SNB_UNFREEZE_AND_RESET_CTR_MBOXFIX(3);
    SNB_UNFREEZE_PCI_BOX(BBOX0);
    SNB_UNFREEZE_AND_RESET_CTR_BOX(WBOX);
    SNB_UNFREEZE_AND_RESET_CTR_PCI_BOX(RBOX0);
    SNB_UNFREEZE_AND_RESET_CTR_PCI_BOX(RBOX1);
    SNB_UNFREEZE_AND_RESET_CTR_PCI_BOX(PBOX);
    return 0;
}

// Read MSR counter register
#define SNB_READ_BOX(id, reg1) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) \
    { \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, reg1, &counter_result)); \
        VERBOSEPRINTREG(cpu_id, reg1, LLU_CAST counter_result, READ_BOX_##id) \
    }

// Read PCI counter registers and combine them to a single value
#define SNB_READ_PCI_BOX(id, dev, reg1, reg2) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id))) && pci_checkDevice(dev, cpu_id)) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, reg1, &tmp)); \
        counter_result = (tmp<<32); \
        CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev, reg2, &tmp)); \
        counter_result += tmp; \
        VERBOSEPRINTPCIREG(cpu_id, dev, reg1, LLU_CAST counter_result, READ_PCI_BOX_##id) \
    }

// Check counter result for overflows. We do not handle overflows directly, that is done in the getResults function in perfmon.c
// SandyBridge has no bits indicating that overflows occured, therefore we use this simple check
#define SNB_CHECK_OVERFLOW \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        eventSet->events[i].threadCounter[thread_id].overflows++; \
    }


int perfmon_stopCountersThread_sandybridge(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t counter_result = 0x0ULL;
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    SNB_FREEZE_BOX(CBOX0);
    SNB_FREEZE_BOX(CBOX1);
    SNB_FREEZE_BOX(CBOX2);
    SNB_FREEZE_BOX(CBOX3);
    SNB_FREEZE_BOX(CBOX4);
    SNB_FREEZE_BOX(CBOX5);
    SNB_FREEZE_BOX(CBOX6);
    SNB_FREEZE_BOX(CBOX7);

    SNB_FREEZE_PCI_BOX(MBOX0);
    SNB_FREEZE_PCI_BOX(MBOX1);
    SNB_FREEZE_PCI_BOX(MBOX2);
    SNB_FREEZE_PCI_BOX(MBOX3);

    SNB_FREEZE_AND_RESET_CTL_PCI_BOX(SBOX0);
    SNB_FREEZE_AND_RESET_CTL_PCI_BOX(SBOX1);

    SNB_FREEZE_AND_RESET_CTL_PCI_BOX(RBOX0);
    SNB_FREEZE_AND_RESET_CTL_PCI_BOX(RBOX1);

    SNB_FREEZE_AND_RESET_CTL_PCI_BOX(PBOX);

    SNB_FREEZE_PCI_BOX(BBOX0);
    SNB_FREEZE_AND_RESET_CTL_BOX(WBOX);

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
            PciDeviceIndex dev = counter_map[index].device;
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                    {
                        uint64_t ovf_values = 0x0ULL;
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_STATUS, &ovf_values));
                        if (ovf_values & (1ULL<<(index - cpuid_info.perf_num_fixed_ctr)))
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL,
                                                        (1ULL<<(index - cpuid_info.perf_num_fixed_ctr))));
                        }
                    }
                    break;

                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                    {
                        uint64_t ovf_values = 0x0ULL;
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_STATUS, &ovf_values));
                        if (ovf_values & (1ULL<<(index+32)))
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<(index+32))));
                        }
                    }
                    break;

                case POWER:
                    if (haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&counter_result));
                        SNB_CHECK_OVERFLOW;
                    }
                    break;

                case THERMAL:
                    CHECK_MSR_READ_ERROR(thermal_read(cpu_id, (uint32_t*)&counter_result));
                    break;

                case MBOX0:
                    
                    SNB_READ_PCI_BOX(MBOX0, dev, counter1, counter2);
                    VERBOSEPRINTPCIREG(cpu_id, dev, reg,  LLU_CAST counter_result, READ_MBOX0);
                    SNB_CHECK_OVERFLOW;
                    break;

                case MBOX1:
                    SNB_READ_PCI_BOX(MBOX1, dev, counter1, counter2);
                    VERBOSEPRINTPCIREG(cpu_id, dev, reg,  LLU_CAST counter_result, READ_MBOX1);
                    SNB_CHECK_OVERFLOW;
                    break;

                case MBOX2:
                    SNB_READ_PCI_BOX(MBOX2, dev, counter1, counter2);
                    VERBOSEPRINTPCIREG(cpu_id, dev, reg,  LLU_CAST counter_result, READ_MBOX2);
                    SNB_CHECK_OVERFLOW;
                    break;

                case MBOX3:
                    SNB_READ_PCI_BOX(MBOX3, dev, counter1, counter2);
                    VERBOSEPRINTPCIREG(cpu_id, dev, counter1,  LLU_CAST counter_result, READ_MBOX3);
                    SNB_CHECK_OVERFLOW;
                    break;

                case MBOX0FIX:
                    SNB_READ_PCI_BOX(MBOX0FIX, dev, counter1, counter2);
                    VERBOSEPRINTPCIREG(cpu_id, dev, counter1,  LLU_CAST counter_result, READ_MBOX0FIX);
                    SNB_CHECK_OVERFLOW;
                    break;
                case MBOX1FIX:
                    SNB_READ_PCI_BOX(MBOX1FIX, dev, counter1, counter2);
                    VERBOSEPRINTPCIREG(cpu_id, dev, counter1,  LLU_CAST counter_result, READ_MBOX1FIX);
                    SNB_CHECK_OVERFLOW;
                    break;
                case MBOX2FIX:
                    SNB_READ_PCI_BOX(MBOX2FIX, dev, counter1, counter2);
                    VERBOSEPRINTPCIREG(cpu_id, dev, counter1,  LLU_CAST counter_result, READ_MBOX2FIX);
                    SNB_CHECK_OVERFLOW;
                    break;
                case MBOX3FIX:
                    SNB_READ_PCI_BOX(MBOX3FIX, dev, counter1, counter2);
                    VERBOSEPRINTPCIREG(cpu_id, dev, counter1,  LLU_CAST counter_result, READ_MBOX3FIX);
                    SNB_CHECK_OVERFLOW;
                    break;

                case SBOX0:
                    SNB_READ_PCI_BOX(SBOX0, dev, counter1, counter2);
                    VERBOSEPRINTPCIREG(cpu_id, dev, counter1,  LLU_CAST counter_result, READ_SBOX0);
                    SNB_CHECK_OVERFLOW;
                    break;

                case SBOX1:
                    SNB_READ_PCI_BOX(SBOX1, dev, counter1, counter2);
                    VERBOSEPRINTPCIREG(cpu_id, dev, counter1,  LLU_CAST counter_result, READ_SBOX1);
                    SNB_CHECK_OVERFLOW;
                    break;

                case SBOX0FIX:
                case SBOX1FIX:
                    if (haveLock && pci_checkDevice(dev, cpu_id))
                    {
                        HPMread(cpu_id, dev, counter1, &counter_result);
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
                        else if (eventSet->events[i].event.eventId == 0x01)
                        {
                            counter_result = extractBitField(counter_result, 1, 4);
                        }
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1,  LLU_CAST counter_result, READ_SBOXFIX);
                    }
                    break;

                case CBOX0:
                    SNB_READ_BOX(CBOX0, counter1);
                    VERBOSEPRINTPCIREG(cpu_id, dev, counter1,  LLU_CAST counter_result, READ_CBOX0);
                    SNB_CHECK_OVERFLOW;
                    break;
                case CBOX1:
                    SNB_READ_BOX(CBOX1, counter1);
                    VERBOSEPRINTPCIREG(cpu_id, dev, counter1,  LLU_CAST counter_result, READ_CBOX1);
                    SNB_CHECK_OVERFLOW;
                    break;
                case CBOX2:
                    SNB_READ_BOX(CBOX2, counter1);
                    SNB_CHECK_OVERFLOW;
                    break;
                case CBOX3:
                    SNB_READ_BOX(CBOX3, counter1);
                    SNB_CHECK_OVERFLOW;
                    break;
                case CBOX4:
                    SNB_READ_BOX(CBOX4, counter1);
                    SNB_CHECK_OVERFLOW;
                    break;
                case CBOX5:
                    SNB_READ_BOX(CBOX5, counter1);
                    SNB_CHECK_OVERFLOW;
                    break;
                case CBOX6:
                    SNB_READ_BOX(CBOX6, counter1);
                    SNB_CHECK_OVERFLOW;
                    break;
                case CBOX7:
                    SNB_READ_BOX(CBOX7, counter1);
                    SNB_CHECK_OVERFLOW;
                    break;

                case UBOX:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                case UBOXFIX:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    SNB_CHECK_OVERFLOW;
                    break;

                case BBOX0:
                    SNB_READ_PCI_BOX(BBOX0, dev, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    break;

                case WBOX:
                    SNB_READ_BOX(WBOX, counter1);
                    SNB_CHECK_OVERFLOW;
                    break;
                case WBOX0FIX:
                    SNB_READ_BOX(WBOX0FIX, counter1);
                    break;
                case WBOX1FIX:
                    SNB_READ_BOX(WBOX1FIX, counter1);
                    break;

                case RBOX0:
                    SNB_READ_PCI_BOX(RBOX0, dev, counter1, counter2);
                    break;
                case RBOX1:
                    SNB_READ_PCI_BOX(RBOX1, dev, counter1, counter2);
                    break;

                case PBOX:
                    SNB_READ_PCI_BOX(PBOX, dev, counter1, counter2);
                    break;
                default:
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData =
                    field64(counter_result, 0, box_map[type].regWidth);
        }
    }

    return 0;
}

int perfmon_readCountersThread_sandybridge(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t counter_result = 0x0ULL;
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t pmc_flags = 0x0ULL;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, &pmc_flags));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    SNB_FREEZE_BOX(CBOX0);
    SNB_FREEZE_BOX(CBOX1);
    SNB_FREEZE_BOX(CBOX2);
    SNB_FREEZE_BOX(CBOX3);
    SNB_FREEZE_BOX(CBOX4);
    SNB_FREEZE_BOX(CBOX5);
    SNB_FREEZE_BOX(CBOX6);
    SNB_FREEZE_BOX(CBOX7);

    SNB_FREEZE_PCI_BOX(MBOX0);
    SNB_FREEZE_PCI_BOX(MBOX1);
    SNB_FREEZE_PCI_BOX(MBOX2);
    SNB_FREEZE_PCI_BOX(MBOX3);

    SNB_FREEZE_MBOXFIX(0);
    SNB_FREEZE_MBOXFIX(1);
    SNB_FREEZE_MBOXFIX(2);
    SNB_FREEZE_MBOXFIX(3);

    SNB_FREEZE_PCI_BOX(SBOX0);
    SNB_FREEZE_PCI_BOX(SBOX1);

    SNB_FREEZE_PCI_BOX(RBOX0);
    SNB_FREEZE_PCI_BOX(RBOX1);

    SNB_FREEZE_PCI_BOX(PBOX);

    SNB_FREEZE_PCI_BOX(BBOX0);
    SNB_FREEZE_BOX(WBOX);

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
            PciDeviceIndex dev = counter_map[index].device;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                    {
                        uint64_t ovf_values = 0x0ULL;
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_STATUS, &ovf_values));
                        if (ovf_values & (1ULL<<(index - cpuid_info.perf_num_fixed_ctr)))
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL,
                                                        (1ULL<<(index - cpuid_info.perf_num_fixed_ctr))));
                        }
                    }
                    break;

                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                    {
                        uint64_t ovf_values = 0x0ULL;
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_STATUS, &ovf_values));
                        if (ovf_values & (1ULL<<(index+32)))
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<(index+32))));
                        }
                    }
                    break;

                case THERMAL:
                    CHECK_MSR_READ_ERROR(thermal_read(cpu_id, (uint32_t*)&counter_result));
                    break;

                case POWER:
                    if (haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&counter_result));
                        SNB_CHECK_OVERFLOW;
                    }
                    break;

                case MBOX0:
                    SNB_READ_PCI_BOX(MBOX0, dev, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    break;

                case MBOX1:
                    SNB_READ_PCI_BOX(MBOX1, dev, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    break;

                case MBOX2:
                    SNB_READ_PCI_BOX(MBOX2, dev, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    break;

                case MBOX3:
                    SNB_READ_PCI_BOX(MBOX3, dev, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    break;

                case UBOX:
                case UBOXFIX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        SNB_CHECK_OVERFLOW;
                    }

                case CBOX0:
                    SNB_READ_BOX(CBOX0, counter1);
                    SNB_CHECK_OVERFLOW;
                    break;
                case CBOX1:
                    SNB_READ_BOX(CBOX1, counter1);
                    SNB_CHECK_OVERFLOW;
                    break;
                case CBOX2:
                    SNB_READ_BOX(CBOX2, counter1);
                    SNB_CHECK_OVERFLOW;
                    break;
                case CBOX3:
                    SNB_READ_BOX(CBOX3, counter1);
                    SNB_CHECK_OVERFLOW;
                    break;
                case CBOX4:
                    SNB_READ_BOX(CBOX4, counter1);
                    SNB_CHECK_OVERFLOW;
                    break;
                case CBOX5:
                    SNB_READ_BOX(CBOX5, counter1);
                    SNB_CHECK_OVERFLOW;
                    break;
                case CBOX6:
                    SNB_READ_BOX(CBOX6, counter1);
                    SNB_CHECK_OVERFLOW;
                    break;
                case CBOX7:
                    SNB_READ_BOX(CBOX7, counter1);
                    SNB_CHECK_OVERFLOW;
                    break;

                case BBOX0:
                    SNB_READ_PCI_BOX(BBOX0, dev, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    break;

                case SBOX0:
                    SNB_READ_PCI_BOX(SBOX0, dev, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    break;

                case SBOX1:
                    SNB_READ_PCI_BOX(SBOX1, dev, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    break;

                case SBOX0FIX:
                case SBOX1FIX:
                    HPMread(cpu_id, dev, counter1, &counter_result);
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
                    else if (eventSet->events[i].event.eventId == 0x01)
                    {
                        counter_result = extractBitField(counter_result, 1, 4);
                    }
                    eventSet->events[i].threadCounter[thread_id].startData = 0x0ULL;
                    break;

                case WBOX:
                    SNB_READ_BOX(WBOX, counter1);
                    SNB_CHECK_OVERFLOW;
                    break;
                case WBOX0FIX:
                    SNB_READ_BOX(WBOX0FIX, counter1);
                    break;
                case WBOX1FIX:
                    SNB_READ_BOX(WBOX1FIX, counter1);
                    break;

                default:
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData =
                    field64(counter_result, 0, box_map[type].regWidth);
        }
    }

    SNB_UNFREEZE_BOX(CBOX0);
    SNB_UNFREEZE_BOX(CBOX1);
    SNB_UNFREEZE_BOX(CBOX2);
    SNB_UNFREEZE_BOX(CBOX3);
    SNB_UNFREEZE_BOX(CBOX4);
    SNB_UNFREEZE_BOX(CBOX5);
    SNB_UNFREEZE_BOX(CBOX6);
    SNB_UNFREEZE_BOX(CBOX7);

    SNB_UNFREEZE_PCI_BOX(MBOX0);
    SNB_UNFREEZE_PCI_BOX(MBOX1);
    SNB_UNFREEZE_PCI_BOX(MBOX2);
    SNB_UNFREEZE_PCI_BOX(MBOX3);

    SNB_UNFREEZE_MBOXFIX(0);
    SNB_UNFREEZE_MBOXFIX(1);
    SNB_UNFREEZE_MBOXFIX(2);
    SNB_UNFREEZE_MBOXFIX(3);

    SNB_UNFREEZE_PCI_BOX(SBOX0);
    SNB_UNFREEZE_PCI_BOX(SBOX1);

    SNB_UNFREEZE_PCI_BOX(RBOX0);
    SNB_UNFREEZE_PCI_BOX(RBOX1);

    SNB_UNFREEZE_PCI_BOX(PBOX);

    SNB_UNFREEZE_PCI_BOX(BBOX0);
    SNB_UNFREEZE_BOX(WBOX);

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, pmc_flags));
    }

    return 0;
}

int perfmon_finalizeCountersThread_sandybridge(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int haveTileLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t ovf_values_core = (1ULL<<63)|(1ULL<<62);

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }
    if (tile_lock[affinity_thread2tile_lookup[cpu_id]] == cpu_id)
    {
        haveTileLock = 1;
    }

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
            if ((reg) &&
                (((type == PMC)||(type == FIXED)) || ((type >= UNCORE) && (haveLock) && (pci_checkDevice(dev, cpu_id)))))
            {
                VERBOSEPRINTPCIREG(cpu_id, dev, reg, 0x0ULL, CLEAR_CTL);
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, 0x0ULL));
            }
            eventSet->events[i].threadCounter[thread_id].init = FALSE;
        }
    }


    if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, LLU_CAST ovf_values_core, CLEAR_GLOBAL_OVF)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, ovf_values_core));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST 0x0ULL, CLEAR_GLOBAL_CTRL)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    return 0;
}
