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

#include <perfmon_sandybridge_events.h>
#include <perfmon_sandybridge_counters.h>
#include <error.h>
#include <affinity.h>

static int perfmon_numCountersSandybridge = NUM_COUNTERS_SANDYBRIDGE;
static int perfmon_numArchEventsSandybridge = NUM_ARCH_EVENTS_SANDYBRIDGE;


int perfmon_init_sandybridge(int cpu_id)
{
    uint64_t flags = 0x0ULL;
    if ( cpuid_info.model == SANDYBRIDGE_EP )
    {
        lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    }
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PEBS_ENABLE, 0x0ULL));
    return 0;
}

#define BOX_GATE_IVYB(channel,label) \
    if(haveLock) { \
        uflags = (1<<22);  \
        uflags |= (event->umask<<8) + event->eventId;  \
        if (event->numberOfOptions > 0) \
        { \
            for(int j=0;j<event->numberOfOptions;j++) \
            { \
                switch(event->options[j].type) \
                { \
                    case EVENT_OPTION_THRESHOLD: \
                        uflags |= ((event->options[j].value << 24) & 0xFF000000); \
                        break; \
                    case EVENT_OPTION_INVERT: \
                        uflags |= (1<<23); \
                        break; \
                    case EVENT_OPTION_EDGE: \
                        uflags |= (1<<18); \
                        break; \
                } \
            } \
        } \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, channel,  reg, uflags));  \
    }

uint32_t snb_fixed_config(RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t ret = 0x0U;
    for(j=0;j<event->numberOfOptions;j++)
    {
        switch (event->options[j].type)
        {
            case EVENT_OPTION_COUNT_KERNEL:
                if (index == 0)
                {
                    ret |= (1<<0);
                }
                else if (index == 1)
                {
                    ret |= (1<<4);
                }
                else if (index == 2)
                {
                    ret |= (1<<8);
                }
                break;
            default:
                break;
        }
    }
    return ret;
}

uint32_t snb_pmc_config(RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t ret = 0x0U;
    for(j=0;j<event->numberOfOptions;j++)
    {
        switch (event->options[j].type)
        {
            case EVENT_OPTION_EDGE:
                ret |= (1<<18);
                break;
            case EVENT_OPTION_COUNT_KERNEL:
                ret |= (1<<17);
                break;
            default:
                break;
        }
    }
    return ret;
}

uint32_t snb_mbox_config(RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t ret = 0x0U;
    for(j=0;j<event->numberOfOptions;j++)
    {
        switch (event->options[j].type)
        {
            case EVENT_OPTION_EDGE:
                ret |= (1<<18);
                break;
            case EVENT_OPTION_INVERT:
                ret |= (1<<23);
                break;
            case EVENT_OPTION_THRESHOLD:
                ret |= ((event->options[j].value << 24) & 0xFF000000);
                break;
            default:
                break;
        }
    }
}

uint32_t snb_cbox_filter(PerfmonEvent *event)
{
    int j;
    uint32_t ret = 0x0;
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
                    ret |= ((event->options[j].value << 23) & 0xFF800000);
                }
                else
                {
                    ERROR_PRINT(Invalid value 0x%x for opcode option, event->options[j].value);
                }
                break;
            case EVENT_OPTION_STATE:
                if (event->options[j].value & 0x1F)
                {
                    ret |= ((event->options[j].value << 18) & 0x7C0000);
                }
                else
                {
                    ERROR_PRINT(Invalid value 0x%x for state option, event->options[j].value);
                }
                break;
            case EVENT_OPTION_NID:
                if (event->options[j].value >= 0x1 && 
                    event->options[j].value <= (affinityDomains.numberOfNumaDomains+1<<1))
                {
                    ret |= ((event->options[j].value << 10) & 0x3FC00);
                }
                else
                {
                    ERROR_PRINT(Invalid value 0x%x for node id option, event->options[j].value);
                }
                break;
            case EVENT_OPTION_TID:
                if (event->options[j].value <= 0xF)
                {
                    ret |= ((event->options[j].value << 0) & 0x1F);
                }
                else
                {
                    ERROR_PRINT(Invalid value 0x%x for thread id option, event->options[j].value);
                }
                break;
            default:
                break;
        }
    }
    return ret;
}

uint32_t snb_cbox_config(RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t ret = 0x0U;
    for(j=0;j<event->numberOfOptions;j++)
    {
        switch (event->options[j].type)
        {
            case EVENT_OPTION_TID:
                ret |= (1<<19);
                break;
            case EVENT_OPTION_EDGE:
                ret |= (1<<18);
                break;
            case EVENT_OPTION_THRESHOLD:
                ret |= ((event->options[j].value << 24) & 0xFF000000);
                break;
            default:
                break;
        }
    }
    return ret;
}

uint32_t snb_ubox_config(RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t ret = 0x0U;
    for(j=0;j<event->numberOfOptions;j++)
    {
        switch (event->options[j].type)
        {
            case EVENT_OPTION_EDGE:
                ret |= (1<<18);
                break;
            case EVENT_OPTION_THRESHOLD:
                ret |= ((event->options[j].value << 24) & 0x1F000000);
                break;
            default:
                break;
        }
    }
    return ret;
}

uint32_t snb_bbox_config(   RegisterIndex index, 
                            PerfmonEvent *event, 
                            uint32_t* opcodematch, 
                            uint32_t* addr0match, 
                            uint32_t* addr1match)
{
    int j;
    uint32_t ret = 0x0U;
    for(j=0;j<event->numberOfOptions;j++)
    {
        switch (event->options[j].type)
        {
            case EVENT_OPTION_EDGE:
                ret |= (1<<18);
                break;
            case EVENT_OPTION_INVERT:
                ret |= (1<<23);
                break;
            case EVENT_OPTION_THRESHOLD:
                ret |= ((event->options[j].value << 24) & 0xFF000000);
                break;
            case EVENT_OPTION_OPCODE:
                *opcodematch = (event->options[j].value & 0x3F);
                break;
            case EVENT_OPTION_MATCH0:
                *addr0match = (extractBitField(event->options[j].value,0,26))<<5;
                break;
            case EVENT_OPTION_MATCH1:
                *addr1match = extractBitField(event->options[j].value,32,14) & 0x3FFF;
                break;
            default:
                break;
        }
    }
    return ret;
}


uint32_t snb_wbox_config(RegisterIndex index, PerfmonEvent *event, uint64_t* filter)
{
    int j;
    uint32_t ret = 0x0U;
    for(j=0;j<event->numberOfOptions;j++)
    {
        switch (event->options[j].type)
        {
            case EVENT_OPTION_EDGE:
                ret |= (1<<18);
                break;
            case EVENT_OPTION_INVERT:
                ret |= (1<<23);
                break;
            case EVENT_OPTION_THRESHOLD:
                ret |= ((event->options[j].value & 0x1F) << 24);
                break;
            case EVENT_OPTION_OCCUPANCY:
                ret |= ((event->options[j].value & 0x3) << 14);
                break;
            case EVENT_OPTION_OCCUPANCY_EDGE:
                ret |= (1<<31);
                break;
            case EVENT_OPTION_OCCUPANCY_INVERT:
                ret |= (1<<30);
                break;
            case EVENT_OPTION_OCCUPANCY_FILTER:
                *filter = event->options[j].value;
            default:
                break;
        }
    }
    return ret;
}

uint32_t snb_sbox_config(RegisterIndex index, PerfmonEvent *event,
                            uint32_t* match0, uint32_t* match1,
                            uint32_t* mask0, uint32_t* mask1)
{
    int j;
    uint32_t ret = 0x0U;
    for(j=0;j<event->numberOfOptions;j++)
    {
        switch (event->options[j].type)
        {
            case EVENT_OPTION_EDGE:
                ret |= (1<<18);
                break;
            case EVENT_OPTION_INVERT:
                ret |= (1<<23);
                break;
            case EVENT_OPTION_THRESHOLD:
                ret |= ((event->options[j].value & 0x1F) << 24);
                break;
            case EVENT_OPTION_MATCH0:
                *match0 = event->options[j].value;
            case EVENT_OPTION_MATCH1:
                *match1 = event->options[j].value;
            case EVENT_OPTION_MASK0:
                *mask0 = event->options[j].value;
            case EVENT_OPTION_MASK1:
                *mask1 = event->options[j].value;
            default:
                break;
        }
    }
    return ret;
}

// Macros to stop counting and reset control registers
// FREEZE(_AND_RESET_CTL) uses central box register to freeze (bit 8 + 16) and bit 1 to reset control registers
#define SNB_FREEZE_AND_RESET_CTL_BOX(id, reg) \
    if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(id))) \
    { \
        VERBOSEPRINTREG(cpu_id, reg, 0x10101U, FREEZE_AND_RESET_CTL_BOX_##id) \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, 0x10101U)); \
    }

#define SNB_FREEZE_BOX(id, reg) \
    if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(id))) \
    { \
        VERBOSEPRINTREG(cpu_id, reg, 0x10100U, FREEZE_AND_RESET_CTL_BOX_##id) \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, 0x10100U)); \
    }

// FREEZE(_AND_RESET_CTL)_PCI uses central box register to freeze (bit 8 + 16) and bit 1 to reset control registers
// Checks whether PCI device exists, because this is the first operation we do on the devices
#define SNB_FREEZE_AND_RESET_CTL_PCI_BOX(id, dev, reg) \
    if (pci_checkDevice(dev, cpu_id) == 0) \
    { \
        fprintf(stderr, "PCI device with index %d does not exist. Skipping all operations on device\n", dev); \
    } \
    else if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, dev, reg, 0x10101U, FREEZE_AND_RESET_CTL_PCI_BOX_##id); \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, dev, reg, 0x10101U)); \
    }

#define SNB_FREEZE_PCI_BOX(id, dev, reg) \
    if (pci_checkDevice(dev, cpu_id) == 0) \
    { \
        fprintf(stderr, "PCI device with index %d does not exist. Skipping all operations on device\n", dev); \
    } \
    else if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, dev, reg, 0x10100U, FREEZE_PCI_BOX_##id) \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, dev, reg, 0x10100U)); \
    }

// MBOX control registers must be reset to 0 manually
#define SNB_RESET_MBOX_CTL(number, ctl) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX##number))) && \
                    (pci_checkDevice(PCI_IMC_DEVICE_CH_##number, cpu_id))) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, IMC_DEVICE_CH_##number,ctl,  0x0U, RESET_MBOX##number##_CTL) \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_##number, ctl, 0x0U)); \
    }

// MBOX*FIX have a slightly different scheme, setting the whole register to 0 freeze the counter
#define SNB_FREEZE_MBOXFIX(number) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX##number##FIX))) && \
                    (pci_checkDevice(PCI_IMC_DEVICE_CH_##number, cpu_id))) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, PCI_IMC_DEVICE_CH_##number, PCI_UNC_MC_PMON_FIXED_CTL, 0x0U, FREEZE_MBOXFIX##number) \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_##number,  PCI_UNC_MC_PMON_FIXED_CTL, 0x0U)); \
    }




// Some setup macros to avoid polluting the functions
#define SNB_SETUP_MBOX(number) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX##number))) && \
                    (pci_checkDevice(PCI_IMC_DEVICE_CH_##number, cpu_id))) { \
        uflags = (1<<22);  \
        uflags |= (event->umask<<8) + event->eventId;  \
        if (event->numberOfOptions > 0) \
        { \
            uflags |= snb_mbox_config(index, event); \
        } \
        VERBOSEPRINTPCIREG(cpu_id, PCI_IMC_DEVICE_CH_##number, reg, LLU_CAST uflags, SETUP_MBOX##number) \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_##number, reg, uflags));  \
    }

#define SNB_SETUP_CBOX(number) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(CBOX##number)))) { \
        uflags = (1<<22);  \
        uflags |= (event->umask<<8) + event->eventId;  \
        if (event->numberOfOptions > 0) \
        { \
            uint32_t optflags = snb_cbox_filter(event); \
            if (optflags != 0x0U) \
            { \
                VERBOSEPRINTREG(cpu_id, MSR_UNC_C##number##_PMON_BOX_FILTER, LLU_CAST optflags, SETUP_CBOX##number##_FILTER) \
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C##number##_PMON_BOX_FILTER, optflags)); \
            } \
            uflags |= snb_cbox_config(index, event); \
        } \
        VERBOSEPRINTREG(cpu_id, reg, LLU_CAST uflags, SETUP_CBOX##number) \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, uflags));  \
    }

#define SNB_SETUP_SBOX(number) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(SBOX##number))) && \
                    (pci_checkDevice(PCI_QPI_DEVICE_PORT_##number, cpu_id))) { \
        uint32_t match0 = 0x0U; \
        uint32_t match1 = 0x0U; \
        uint32_t mask0 = 0x0U; \
        uint32_t mask1 = 0x0U; \
        uint64_t counter1 = sandybridge_counter_map[index].counterRegister; \
        uint64_t counter2 = sandybridge_counter_map[index].counterRegister2; \
        uflags = (1<<22);  \
        uflags |= event->cfgBits; \
        uflags |= (event->umask<<8) + event->eventId;  \
        if (event->numberOfOptions > 0) \
        { \
            uflags |= snb_sbox_config(index, event, &match0, &match1, &mask0, &mask1); \
            if (match0 != 0x0) \
            { \
                VERBOSEPRINTPCIREG(cpu_id, PCI_QPI_DEVICE_PORT_##number, PCI_UNC_QPI_PMON_MATCH_0, LLU_CAST match0, SETUP_SBOX##number##_MATCH0) \
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_MASK_DEVICE_PORT_##number, \
                                                PCI_UNC_QPI_PMON_MATCH_0, match0)); \
            } \
            if (match1 != 0x0) \
            { \
                VERBOSEPRINTPCIREG(cpu_id, PCI_QPI_DEVICE_PORT_##number, PCI_UNC_QPI_PMON_MATCH_1, LLU_CAST match1, SETUP_SBOX##number##_MATCH1) \
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_MASK_DEVICE_PORT_##number, \
                                                PCI_UNC_QPI_PMON_MATCH_1, match1)); \
            } \
            if (mask0 != 0x0) \
            { \
                VERBOSEPRINTPCIREG(cpu_id, PCI_QPI_DEVICE_PORT_##number, PCI_UNC_QPI_PMON_MASK_0, LLU_CAST mask0, SETUP_SBOX##number##_MASK0) \
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_MASK_DEVICE_PORT_##number, \
                                                PCI_UNC_QPI_PMON_MASK_0, uflags)); \
            } \
            if (mask1 != 0x0) \
            { \
                VERBOSEPRINTPCIREG(cpu_id, PCI_QPI_DEVICE_PORT_##number, PCI_UNC_QPI_PMON_MASK_1, LLU_CAST mask1, SETUP_SBOX##number##_MASK1) \
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_MASK_DEVICE_PORT_##number, \
                                                PCI_UNC_QPI_PMON_MASK_1, uflags)); \
            } \
        } \
        VERBOSEPRINTPCIREG(cpu_id, PCI_QPI_DEVICE_PORT_##number, reg, LLU_CAST uflags, SETUP_SBOX##number) \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_##number,  reg, uflags));  \
    }

int perfmon_setupCounterThread_sandybridge(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int i, j;
    int haveLock = 0;
    uint64_t flags;
    uint64_t fixed_flags = 0x0ULL;
    uint32_t uflags;

    int cpu_id = groupSet->threads[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)))
    {
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PEBS_ENABLE, 0x0ULL));
    }
    SNB_FREEZE_BOX(CBOX0, MSR_UNC_C0_PMON_BOX_CTL);
    SNB_FREEZE_BOX(CBOX1, MSR_UNC_C1_PMON_BOX_CTL);
    SNB_FREEZE_BOX(CBOX2, MSR_UNC_C2_PMON_BOX_CTL);
    SNB_FREEZE_BOX(CBOX3, MSR_UNC_C3_PMON_BOX_CTL);
    SNB_FREEZE_BOX(CBOX4, MSR_UNC_C4_PMON_BOX_CTL);
    SNB_FREEZE_BOX(CBOX5, MSR_UNC_C5_PMON_BOX_CTL);
    SNB_FREEZE_BOX(CBOX6, MSR_UNC_C6_PMON_BOX_CTL);
    SNB_FREEZE_BOX(CBOX7, MSR_UNC_C7_PMON_BOX_CTL);

    SNB_FREEZE_PCI_BOX(MBOX0, PCI_IMC_DEVICE_CH_0, PCI_UNC_MC_PMON_BOX_CTL);
    SNB_FREEZE_PCI_BOX(MBOX1, PCI_IMC_DEVICE_CH_1, PCI_UNC_MC_PMON_BOX_CTL);
    SNB_FREEZE_PCI_BOX(MBOX2, PCI_IMC_DEVICE_CH_2, PCI_UNC_MC_PMON_BOX_CTL);
    SNB_FREEZE_PCI_BOX(MBOX3, PCI_IMC_DEVICE_CH_3, PCI_UNC_MC_PMON_BOX_CTL);

    SNB_FREEZE_MBOXFIX(0);
    SNB_FREEZE_MBOXFIX(1);
    SNB_FREEZE_MBOXFIX(2);
    SNB_FREEZE_MBOXFIX(3);

    SNB_FREEZE_PCI_BOX(SBOX0, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_BOX_CTL);
    SNB_FREEZE_PCI_BOX(SBOX1, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_BOX_CTL);

    SNB_FREEZE_PCI_BOX(BBOX0, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_BOX_CTL);
    SNB_FREEZE_BOX(WBOX, MSR_UNC_PCU_PMON_BOX_CTL1);

    for (i=0;i < eventSet->numberOfEvents;i++)
    {
        PerfmonEvent *event = &(eventSet->events[i].event);
        RegisterIndex index = eventSet->events[i].index;
        uint64_t reg = sandybridge_counter_map[index].configRegister;
        uint64_t filter_reg;
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        switch (sandybridge_counter_map[index].type)
        {
            case PMC:
                flags = 0x0ULL;
                flags |= (1<<22);  /* enable flag */
                flags |= (1<<16);  /* user mode flag */

                /* Intel with standard 8 bit event mask: [7:0] */
                flags |= (event->umask<<8) + event->eventId;

                if (event->cfgBits != 0) /* set custom cfg and cmask */
                {
                    flags |= ((event->cmask<<8) + event->cfgBits)<<16;
                }

                if (event->numberOfOptions > 0)
                {
                    flags |= snb_pmc_config(index, event);
                }
                VERBOSEPRINTREG(cpu_id, reg, LLU_CAST flags, SETUP_PMC)
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, flags));
                break;

            case FIXED:
                /* initialize fixed counters
                 * FIXED 0: Instructions retired
                 * FIXED 1: Clocks unhalted core
                 * FIXED 2: Clocks unhalted ref */
                fixed_flags |= (0x2 << (4*index));
                if (event->numberOfOptions > 0)
                {
                    fixed_flags |= snb_fixed_config(index,event);
                }
                /* Written in the end of function for all fixed purpose registers */
                break;

            case POWER:
                break;

            case MBOX0:
                SNB_SETUP_MBOX(0);
                break;
            case MBOX1:
                SNB_SETUP_MBOX(1);
                break;
            case MBOX2:
                SNB_SETUP_MBOX(2);
                break;
            case MBOX3:
                SNB_SETUP_MBOX(3);
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
                SNB_SETUP_CBOX(0);
                break;
            case CBOX1:
                SNB_SETUP_CBOX(1);
                break;
            case CBOX2:
                SNB_SETUP_CBOX(2);
                break;
            case CBOX3:
                SNB_SETUP_CBOX(3);
                break;
            case CBOX4:
                SNB_SETUP_CBOX(4);
                break;
            case CBOX5:
                SNB_SETUP_CBOX(5);
                break;
            case CBOX6:
                SNB_SETUP_CBOX(6);
                break;
            case CBOX7:
                SNB_SETUP_CBOX(7);
                break;

            case UBOX:
                if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(UBOX))))
                {
                    uflags = (1<<17);
                    uflags |= (event->umask<<8) + event->eventId;

                    if (event->numberOfOptions > 0)
                    {
                        uflags |= snb_ubox_config(index, event);
                    }
                    VERBOSEPRINTREG(cpu_id, reg, LLU_CAST uflags, SETUP_UBOX)
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, uflags));
                }
                break;
                
            case UBOXFIX:
                if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(UBOXFIX))))
                {
                    VERBOSEPRINTREG(cpu_id, reg, LLU_CAST 0x0U, SETUP_UBOXFIX)
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, 0x0U));
                }
                break;

            case SBOX0:
                SNB_SETUP_SBOX(0);
                break;
            case SBOX1:
                SNB_SETUP_SBOX(1);
                break;

            case BBOX0:
                if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(BBOX0))) && 
                                (pci_checkDevice(PCI_HA_DEVICE, cpu_id))) 
                {
                    uint32_t opcode = 0x0U;
                    uint32_t match0 = 0x0U;
                    uint32_t match1 = 0x0U;
                    uint64_t counter1 = sandybridge_counter_map[index].counterRegister;
                    uint64_t counter2 = sandybridge_counter_map[index].counterRegister2;
                    uflags = (1<<22);
                    uflags |= (event->umask<<8) + event->eventId;
                    if (event->numberOfOptions > 0)
                    {
                        uflags |= snb_bbox_config(index, event, &opcode, &match0, &match1);
                        if (opcode != 0x0)
                        {
                            VERBOSEPRINTPCIREG(cpu_id, PCI_HA_DEVICE, PCI_UNC_HA_PMON_OPCODEMATCH, LLU_CAST opcode, SETUP_BBOX_OPCODE)
                            CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,
                                            PCI_UNC_HA_PMON_OPCODEMATCH, opcode));
                        }
                        if (match0 != 0x0)
                        {
                            VERBOSEPRINTPCIREG(cpu_id, PCI_HA_DEVICE, PCI_UNC_HA_PMON_ADDRMATCH0, LLU_CAST match0, SETUP_BBOX_MATCH0)
                            CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,
                                            PCI_UNC_HA_PMON_ADDRMATCH0, match0));
                        }
                        if (match1 != 0x0)
                        {
                            VERBOSEPRINTPCIREG(cpu_id, PCI_HA_DEVICE, PCI_UNC_HA_PMON_ADDRMATCH1, LLU_CAST match1, SETUP_BBOX_MATCH1)
                            CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,
                                            PCI_UNC_HA_PMON_ADDRMATCH1, match1));
                        }
                    }
                    VERBOSEPRINTPCIREG(cpu_id, PCI_HA_DEVICE, reg, LLU_CAST uflags, SETUP_BBOX)
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,  reg, uflags));
                }
                break;

            case WBOX:
                if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(WBOX))))
                {
                    flags = (1<<22);
                    flags |= event->eventId & 0xFF;
                    if (event->numberOfOptions > 0)
                    {
                        uint64_t filter = 0x0U;
                        flags |= snb_wbox_config(index, event, &filter);
                        if (filter != 0x0)
                        {
                            VERBOSEPRINTREG(cpu_id, MSR_UNC_PCU_PMON_BOX_FILTER, LLU_CAST filter, SETUP_WBOX_FILTER)
                            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_PCU_PMON_BOX_FILTER, filter));
                        }
                    }
                    VERBOSEPRINTREG(cpu_id, reg, LLU_CAST flags, SETUP_WBOX)
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, flags));
                }

            default:
                /* should never be reached */
                break;
        }
    }
    
    if (fixed_flags > 0x0)
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_FIXED_CTR_CTRL, LLU_CAST fixed_flags, SETUP_FIXED)
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, fixed_flags));
    }
    return 0;
}


// Macros for MSR HPM counters
// UNFREEZE(_AND_RESET_CTR) uses the central box registers to unfreeze and reset the counter registers
#define SNB_UNFREEZE_BOX(id, reg) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) { \
        VERBOSEPRINTREG(cpu_id, reg, LLU_CAST 0x0ULL, UNFREEZE_BOX_##id) \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, 0x0ULL)); \
    }

#define SNB_UNFREEZE_AND_RESET_CTR_BOX(id, reg) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) { \
        VERBOSEPRINTREG(cpu_id, reg, LLU_CAST 0x2ULL, UNFREEZE_BOX_##id) \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, 0x2ULL)); \
    }

// ENABLE(_AND_RESET_CTR) uses the control registers to enable (bit 22) and reset the counter registers (bit 19)
#define SNB_ENABLE_BOX(id, reg) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, reg, &tmp)); \
        tmp |= (1<<22); \
        VERBOSEPRINTREG(cpu_id, reg, LLU_CAST tmp, ENABLE_BOX_##id) \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, tmp)); \
    }

#define SNB_ENABLE_AND_RESET_CTR_BOX(id, reg) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, reg, &tmp)); \
        tmp |= (1<<22)|(1<<17); \
        VERBOSEPRINTREG(cpu_id, reg, LLU_CAST tmp, ENABLE_AND_RESET_CTR_BOX_##id) \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, tmp)); \
    }

// UNFREEZE(_AND_RESET_CTR)_PCI is similar to MSR UNFREEZE but for PCI devices
#define SNB_UNFREEZE_PCI_BOX(id, dev, reg) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id))) \
                && (pci_checkDevice(dev, cpu_id))) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, dev, reg, LLU_CAST 0x0ULL, UNFREEZE_PCI_BOX_##id) \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, dev, reg, 0x0ULL)); \
    }
#define SNB_UNFREEZE_AND_RESET_CTR_PCI_BOX(id, dev, reg) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id))) \
                && (pci_checkDevice(dev, cpu_id))) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, dev, reg, LLU_CAST 0x2ULL, UNFREEZE_AND_RESET_CTR_PCI_BOX_##id) \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, dev, reg, 0x2ULL)); \
    }

// UNFREEZE(_AND_RESET_CTR)_MBOXFIX is kind of ENABLE for PCI but uses bit 19 for reset
#define SNB_UNFREEZE_AND_RESET_CTR_MBOXFIX(number) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX##number##FIX))) && \
                    (pci_checkDevice(PCI_IMC_DEVICE_CH_##number, cpu_id))) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, PCI_IMC_DEVICE_CH_##number, \
                PCI_UNC_MC_PMON_FIXED_CTL, LLU_CAST (1<<22)|(1<<19), UNFREEZE_AND_RESET_CTR_MBOXFIX##id) \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_##number,  PCI_UNC_MC_PMON_FIXED_CTL, (1<<22)|(1<<19))); \
    }
#define SNB_UNFREEZE_MBOXFIX(number) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX##number##FIX))) && \
                    (pci_checkDevice(PCI_IMC_DEVICE_CH_##number, cpu_id))) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, PCI_IMC_DEVICE_CH_##number, \
                PCI_UNC_MC_PMON_FIXED_CTL, LLU_CAST (1<<22), UNFREEZE_MBOXFIX##id) \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_##number,  PCI_UNC_MC_PMON_FIXED_CTL, (1<<22))); \
    }


int perfmon_startCountersThread_sandybridge(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t tmp;
    uint64_t flags = 0x0ULL;
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
            uint64_t reg = sandybridge_counter_map[index].configRegister;
            uint64_t counter1 = sandybridge_counter_map[index].counterRegister;
            uint64_t counter2 = sandybridge_counter_map[index].counterRegister2;
            switch (sandybridge_counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, counter1, 0x0ULL));
                    flags |= (1<<(index-OFFSET_PMC));  /* enable counter */
                    break;

                case FIXED:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, counter1, 0x0ULL));
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
                    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX0))) && \
                            pci_checkDevice(cpu_id, PCI_IMC_DEVICE_CH_0))
                    {
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0, counter1, 0x0ULL));
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0, counter2, 0x0ULL));
                    }
                    break;
                case MBOX1:
                    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX1))) && \
                            pci_checkDevice(cpu_id, PCI_IMC_DEVICE_CH_1))
                    {
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1, counter1, 0x0ULL));
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1, counter2, 0x0ULL));
                    }
                    break;

                case MBOX2:
                    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX2))) && \
                            pci_checkDevice(cpu_id, PCI_IMC_DEVICE_CH_2))
                    {
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2, counter1, 0x0ULL));
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2, counter2, 0x0ULL));
                    }
                    break;

                case MBOX3:
                    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX3))) && \
                            pci_checkDevice(cpu_id, PCI_IMC_DEVICE_CH_3))
                    {
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3, counter1, 0x0ULL));
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3, counter2, 0x0ULL));
                    }
                    break;

                case MBOX0FIX:
                    break;
                case MBOX1FIX:
                    break;
                case MBOX2FIX:
                    break;
                case MBOX3FIX:
                    break;


                case SBOX0:
                    break;

                case SBOX1:
                    break;

                case CBOX0:
                    break;
                case CBOX1:
                    break;
                case CBOX2:
                    break;
                case CBOX3:
                    break;
                case CBOX4:
                    break;
                case CBOX5:
                    break;
                case CBOX6:
                    break;
                case CBOX7:
                    break;

                case UBOX:
                    SNB_ENABLE_AND_RESET_CTR_BOX(UBOX, reg);
                    break;
                case UBOXFIX:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, counter1, 0x0ULL));
                    SNB_ENABLE_BOX(UBOXFIX, reg);
                    break;

                case BBOX0:
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE, counter1, 0x0ULL));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE, counter2, 0x0ULL));
                    break;

                case WBOX:
                    break;
                case WBOXFIX0:
                case WBOXFIX1:
                    if(haveLock)
                    {
                        CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter1, &tmp));
                        eventSet->events[i].threadCounter[thread_id].startData = tmp;
                    }
                    break;
                default:
                    /* should never be reached */
                    break;
            }
        }
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        DEBUG_PLAIN_PRINT(DEBUGLEV_DETAIL, Start thread-local MSR counters);
        DEBUG_PRINT(DEBUGLEV_DETAIL, Write flags 0x%llX to register 0x%X ,
                        MSR_PERF_GLOBAL_CTRL, LLU_CAST flags);
        DEBUG_PRINT(DEBUGLEV_DETAIL, Write flags 0x%llX to register 0x%X ,
                        MSR_UNCORE_PERF_GLOBAL_CTRL, LLU_CAST uflags);

        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, UNFREEZE_PMC_OR_FIXED)
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, flags));
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x30000000FULL));
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(CBOX0)))
    {
        SNB_UNFREEZE_AND_RESET_CTR_BOX(CBOX0, MSR_UNC_C0_PMON_BOX_CTL);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(CBOX1)))
    {
        SNB_UNFREEZE_AND_RESET_CTR_BOX(CBOX1, MSR_UNC_C1_PMON_BOX_CTL);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(CBOX2)))
    {
        SNB_UNFREEZE_AND_RESET_CTR_BOX(CBOX2, MSR_UNC_C2_PMON_BOX_CTL);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(CBOX3)))
    {
        SNB_UNFREEZE_AND_RESET_CTR_BOX(CBOX3, MSR_UNC_C3_PMON_BOX_CTL);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(CBOX4)))
    {
        SNB_UNFREEZE_AND_RESET_CTR_BOX(CBOX4, MSR_UNC_C4_PMON_BOX_CTL);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(CBOX5)))
    {
        SNB_UNFREEZE_AND_RESET_CTR_BOX(CBOX5, MSR_UNC_C5_PMON_BOX_CTL);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(CBOX6)))
    {
        SNB_UNFREEZE_AND_RESET_CTR_BOX(CBOX6, MSR_UNC_C6_PMON_BOX_CTL);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(CBOX7)))
    {
        SNB_UNFREEZE_AND_RESET_CTR_BOX(CBOX7, MSR_UNC_C7_PMON_BOX_CTL);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(SBOX0)))
    {
        SNB_UNFREEZE_AND_RESET_CTR_PCI_BOX(SBOX0, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_BOX_CTL);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(SBOX1)))
    {
        SNB_UNFREEZE_AND_RESET_CTR_PCI_BOX(SBOX1, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_BOX_CTL);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX0)))
    {
        SNB_UNFREEZE_PCI_BOX(MBOX0, PCI_IMC_DEVICE_CH_0, PCI_UNC_MC_PMON_BOX_CTL);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX1)))
    {
        SNB_UNFREEZE_PCI_BOX(MBOX1, PCI_IMC_DEVICE_CH_1, PCI_UNC_MC_PMON_BOX_CTL);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX2)))
    {
        SNB_UNFREEZE_PCI_BOX(MBOX2, PCI_IMC_DEVICE_CH_2, PCI_UNC_MC_PMON_BOX_CTL);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX3)))
    {
        SNB_UNFREEZE_PCI_BOX(MBOX3, PCI_IMC_DEVICE_CH_3, PCI_UNC_MC_PMON_BOX_CTL);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX0FIX)))
    {
        SNB_UNFREEZE_AND_RESET_CTR_MBOXFIX(0);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX1FIX)))
    {
        SNB_UNFREEZE_AND_RESET_CTR_MBOXFIX(1);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX2FIX)))
    {
        SNB_UNFREEZE_AND_RESET_CTR_MBOXFIX(2);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX3FIX)))
    {
        SNB_UNFREEZE_AND_RESET_CTR_MBOXFIX(3);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(BBOX0)))
    {
        SNB_UNFREEZE_PCI_BOX(BBOX0, PCI_HA_DEVICE, PCI_UNC_HA_PMON_BOX_CTL);
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(WBOX)))
    {
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_PCU_PMON_BOX_FILTER, 0x0U));
        SNB_UNFREEZE_AND_RESET_CTR_BOX(WBOX, MSR_UNC_PCU_PMON_BOX_CTL1);
    }
    return 0;
}

// Read MSR counter register
#define SNB_READ_BOX(id, reg1) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) \
    { \
        VERBOSEPRINTREG(cpu_id, reg1, LLU_CAST counter_result, READ_BOX_##id) \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, reg1, &counter_result)); \
    }

// Read PCI counter registers and combine them to a single value
#define SNB_READ_PCI_BOX(id, dev, reg1, reg2) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id))) && pci_checkDevice(dev, cpu_id)) \
    { \
        uint64_t tmp = 0x0U; \
        CHECK_PCI_READ_ERROR(pci_read(cpu_id, dev, reg1, (uint32_t*)&tmp)); \
        counter_result = (tmp<<32); \
        CHECK_PCI_READ_ERROR(pci_read(cpu_id, dev, reg2, (uint32_t*)&tmp)); \
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
    uint64_t flags;
    uint32_t uflags = 0x10100UL; /* Set freeze bit */
    uint64_t counter_result = 0x0ULL;
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    SNB_FREEZE_AND_RESET_CTL_BOX(CBOX0, MSR_UNC_C0_PMON_BOX_CTL);
    SNB_FREEZE_AND_RESET_CTL_BOX(CBOX1, MSR_UNC_C1_PMON_BOX_CTL);
    SNB_FREEZE_AND_RESET_CTL_BOX(CBOX2, MSR_UNC_C2_PMON_BOX_CTL);
    SNB_FREEZE_AND_RESET_CTL_BOX(CBOX3, MSR_UNC_C3_PMON_BOX_CTL);
    SNB_FREEZE_AND_RESET_CTL_BOX(CBOX4, MSR_UNC_C4_PMON_BOX_CTL);
    SNB_FREEZE_AND_RESET_CTL_BOX(CBOX5, MSR_UNC_C5_PMON_BOX_CTL);
    SNB_FREEZE_AND_RESET_CTL_BOX(CBOX6, MSR_UNC_C6_PMON_BOX_CTL);
    SNB_FREEZE_AND_RESET_CTL_BOX(CBOX7, MSR_UNC_C7_PMON_BOX_CTL);

    SNB_FREEZE_PCI_BOX(MBOX0, PCI_IMC_DEVICE_CH_0, PCI_UNC_MC_PMON_BOX_CTL);
    SNB_FREEZE_PCI_BOX(MBOX1, PCI_IMC_DEVICE_CH_1, PCI_UNC_MC_PMON_BOX_CTL);
    SNB_FREEZE_PCI_BOX(MBOX2, PCI_IMC_DEVICE_CH_2, PCI_UNC_MC_PMON_BOX_CTL);
    SNB_FREEZE_PCI_BOX(MBOX3, PCI_IMC_DEVICE_CH_3, PCI_UNC_MC_PMON_BOX_CTL);

    SNB_FREEZE_AND_RESET_CTL_PCI_BOX(SBOX0, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_BOX_CTL);
    SNB_FREEZE_AND_RESET_CTL_PCI_BOX(SBOX1, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_BOX_CTL);

    SNB_FREEZE_PCI_BOX(BBOX0, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_BOX_CTL);
    SNB_FREEZE_AND_RESET_CTL_BOX(WBOX, MSR_UNC_PCU_PMON_BOX_CTL1);

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE) 
        {
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = sandybridge_counter_map[index].configRegister;
            uint64_t counter1 = sandybridge_counter_map[index].counterRegister;
            uint64_t counter2 = sandybridge_counter_map[index].counterRegister2;
            switch (sandybridge_counter_map[index].type)
            {
                case PMC:

                case FIXED:
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter1, &counter_result));
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case POWER:
                    if(haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&counter_result));
                        SNB_CHECK_OVERFLOW;
                        eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    }
                    break;

                case THERMAL:
                        CHECK_TEMP_READ_ERROR(thermal_read(cpu_id, (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case MBOX0:
                    SNB_RESET_MBOX_CTL(0, reg);
                    SNB_READ_PCI_BOX(MBOX0, PCI_IMC_DEVICE_CH_0, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case MBOX1:
                    SNB_RESET_MBOX_CTL(1, reg);
                    SNB_READ_PCI_BOX(MBOX1, PCI_IMC_DEVICE_CH_1, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case MBOX2:
                    SNB_RESET_MBOX_CTL(2, reg);
                    SNB_READ_PCI_BOX(MBOX2, PCI_IMC_DEVICE_CH_2, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case MBOX3:
                    SNB_RESET_MBOX_CTL(3, reg);
                    SNB_READ_PCI_BOX(MBOX3, PCI_IMC_DEVICE_CH_3, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case MBOX0FIX:
                    SNB_READ_PCI_BOX(MBOX0FIX, PCI_IMC_DEVICE_CH_0, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case MBOX1FIX:
                    SNB_READ_PCI_BOX(MBOX1FIX, PCI_IMC_DEVICE_CH_1, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case MBOX2FIX:
                    SNB_READ_PCI_BOX(MBOX2FIX, PCI_IMC_DEVICE_CH_2, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case MBOX3FIX:
                    SNB_READ_PCI_BOX(MBOX3FIX, PCI_IMC_DEVICE_CH_3, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case SBOX0:
                    SNB_READ_PCI_BOX(SBOX0, PCI_QPI_DEVICE_PORT_0, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case SBOX1:
                    SNB_READ_PCI_BOX(SBOX1, PCI_QPI_DEVICE_PORT_1, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case CBOX0:
                    SNB_READ_BOX(CBOX0, counter1);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX1:
                    SNB_READ_BOX(CBOX1, counter1);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX2:
                    SNB_READ_BOX(CBOX2, counter1);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX3:
                    SNB_READ_BOX(CBOX3, counter1);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX4:
                    SNB_READ_BOX(CBOX4, counter1);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX5:
                    SNB_READ_BOX(CBOX5, counter1);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX6:
                    SNB_READ_BOX(CBOX6, counter1);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX7:
                    SNB_READ_BOX(CBOX7, counter1);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case UBOX:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, 0x0U));
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter1, &counter_result));
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case UBOXFIX:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, 0x0U));
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter1, &counter_result));
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case BBOX0:
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE, reg, 0x0U));
                    SNB_READ_PCI_BOX(BBOX0, PCI_HA_DEVICE, counter1, counter2);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case WBOX:
                    SNB_READ_BOX(WBOX, counter1);
                    SNB_CHECK_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case WBOXFIX0:
                    SNB_READ_BOX(WBOXFIX0, counter1);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case WBOXFIX1:
                    SNB_READ_BOX(WBOXFIX1, counter1);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                default:
                    /* should never be reached */
                    break;
            }
        }
    }

    CHECK_MSR_READ_ERROR(msr_read(cpu_id,MSR_PERF_GLOBAL_STATUS, &flags));
    //    printf ("Status: 0x%llX \n", LLU_CAST flags);
    if ( (flags & 0x3) || (flags & (0x3ULL<<32)) ) 
    {
        printf ("Overflow occured \n");
    }
    return 0;
}

int perfmon_readCountersThread_sandybridge(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t counter_result = 0x0ULL;
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    SNB_FREEZE_BOX(CBOX0, MSR_UNC_C0_PMON_BOX_CTL);
    SNB_FREEZE_BOX(CBOX1, MSR_UNC_C1_PMON_BOX_CTL);
    SNB_FREEZE_BOX(CBOX2, MSR_UNC_C2_PMON_BOX_CTL);
    SNB_FREEZE_BOX(CBOX3, MSR_UNC_C3_PMON_BOX_CTL);
    SNB_FREEZE_BOX(CBOX4, MSR_UNC_C4_PMON_BOX_CTL);
    SNB_FREEZE_BOX(CBOX5, MSR_UNC_C5_PMON_BOX_CTL);
    SNB_FREEZE_BOX(CBOX6, MSR_UNC_C6_PMON_BOX_CTL);
    SNB_FREEZE_BOX(CBOX7, MSR_UNC_C7_PMON_BOX_CTL);

    SNB_FREEZE_PCI_BOX(MBOX0, PCI_IMC_DEVICE_CH_0, PCI_UNC_MC_PMON_BOX_CTL);
    SNB_FREEZE_PCI_BOX(MBOX1, PCI_IMC_DEVICE_CH_1, PCI_UNC_MC_PMON_BOX_CTL);
    SNB_FREEZE_PCI_BOX(MBOX2, PCI_IMC_DEVICE_CH_2, PCI_UNC_MC_PMON_BOX_CTL);
    SNB_FREEZE_PCI_BOX(MBOX3, PCI_IMC_DEVICE_CH_3, PCI_UNC_MC_PMON_BOX_CTL);

    SNB_FREEZE_MBOXFIX(0);
    SNB_FREEZE_MBOXFIX(1);
    SNB_FREEZE_MBOXFIX(2);
    SNB_FREEZE_MBOXFIX(3);

    SNB_FREEZE_PCI_BOX(SBOX0, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_BOX_CTL);
    SNB_FREEZE_PCI_BOX(SBOX1, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_BOX_CTL);

    SNB_FREEZE_PCI_BOX(BBOX0, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_BOX_CTL);
    SNB_FREEZE_BOX(WBOX, MSR_UNC_PCU_PMON_BOX_CTL1);

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = sandybridge_counter_map[index].configRegister;
            uint64_t counter1 = sandybridge_counter_map[index].counterRegister;
            uint64_t counter2 = sandybridge_counter_map[index].counterRegister2;
            if ((sandybridge_counter_map[index].type == PMC) ||
                    (sandybridge_counter_map[index].type == FIXED))
            {
                CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter1, &counter_result));
                SNB_CHECK_OVERFLOW;
                eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
            }
            else
            {
                if(haveLock)
                {
                    switch (sandybridge_counter_map[index].type)
                    {
                        case POWER:
                            CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&counter_result));
                            SNB_CHECK_OVERFLOW;
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;

                        case MBOX0:
                            SNB_READ_PCI_BOX(MBOX0, PCI_IMC_DEVICE_CH_0, counter1, counter2);
                            SNB_CHECK_OVERFLOW;
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;

                        case MBOX1:
                            SNB_READ_PCI_BOX(MBOX1, PCI_IMC_DEVICE_CH_1, counter1, counter2);
                            SNB_CHECK_OVERFLOW;
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;

                        case MBOX2:
                            SNB_READ_PCI_BOX(MBOX2, PCI_IMC_DEVICE_CH_2, counter1, counter2);
                            SNB_CHECK_OVERFLOW;
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;

                        case MBOX3:
                            SNB_READ_PCI_BOX(MBOX3, PCI_IMC_DEVICE_CH_3, counter1, counter2);
                            SNB_CHECK_OVERFLOW;
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;

                        case UBOX:
                        case UBOXFIX:
                            CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter1, &counter_result));
                            SNB_CHECK_OVERFLOW;
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;

                        case CBOX0:
                            SNB_READ_BOX(CBOX0, counter1);
                            SNB_CHECK_OVERFLOW;
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;
                        case CBOX1:
                            SNB_READ_BOX(CBOX1, counter1);
                            SNB_CHECK_OVERFLOW;
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;
                        case CBOX2:
                            SNB_READ_BOX(CBOX2, counter1);
                            SNB_CHECK_OVERFLOW;
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;
                        case CBOX3:
                            SNB_READ_BOX(CBOX3, counter1);
                            SNB_CHECK_OVERFLOW;
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;
                        case CBOX4:
                            SNB_READ_BOX(CBOX4, counter1);
                            SNB_CHECK_OVERFLOW;
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;
                        case CBOX5:
                            SNB_READ_BOX(CBOX5, counter1);
                            SNB_CHECK_OVERFLOW;
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;
                        case CBOX6:
                            SNB_READ_BOX(CBOX6, counter1);
                            SNB_CHECK_OVERFLOW;
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;
                        case CBOX7:
                            SNB_READ_BOX(CBOX7, counter1);
                            SNB_CHECK_OVERFLOW;
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;

                        case BBOX0:
                            SNB_READ_PCI_BOX(BBOX0, PCI_HA_DEVICE, counter1, counter2);
                            SNB_CHECK_OVERFLOW;
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;

                        case WBOX:
                            SNB_READ_BOX(WBOX, counter1);
                            SNB_CHECK_OVERFLOW;
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;
                        case WBOXFIX0:
                            SNB_READ_BOX(WBOXFIX0, counter1);
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;
                        case WBOXFIX1:
                            SNB_READ_BOX(WBOXFIX1, counter1);
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                            break;

                        default:
                            /* should never be reached */
                            break;
                    }
                }
            }
        }
    }

    SNB_UNFREEZE_BOX(CBOX0, MSR_UNC_C0_PMON_BOX_CTL);
    SNB_UNFREEZE_BOX(CBOX1, MSR_UNC_C1_PMON_BOX_CTL);
    SNB_UNFREEZE_BOX(CBOX2, MSR_UNC_C2_PMON_BOX_CTL);
    SNB_UNFREEZE_BOX(CBOX3, MSR_UNC_C3_PMON_BOX_CTL);
    SNB_UNFREEZE_BOX(CBOX4, MSR_UNC_C4_PMON_BOX_CTL);
    SNB_UNFREEZE_BOX(CBOX5, MSR_UNC_C5_PMON_BOX_CTL);
    SNB_UNFREEZE_BOX(CBOX6, MSR_UNC_C6_PMON_BOX_CTL);
    SNB_UNFREEZE_BOX(CBOX7, MSR_UNC_C7_PMON_BOX_CTL);

    SNB_UNFREEZE_PCI_BOX(MBOX0, PCI_IMC_DEVICE_CH_0, PCI_UNC_MC_PMON_BOX_CTL);
    SNB_UNFREEZE_PCI_BOX(MBOX1, PCI_IMC_DEVICE_CH_1, PCI_UNC_MC_PMON_BOX_CTL);
    SNB_UNFREEZE_PCI_BOX(MBOX2, PCI_IMC_DEVICE_CH_2, PCI_UNC_MC_PMON_BOX_CTL);
    SNB_UNFREEZE_PCI_BOX(MBOX3, PCI_IMC_DEVICE_CH_3, PCI_UNC_MC_PMON_BOX_CTL);

    SNB_UNFREEZE_MBOXFIX(0);
    SNB_UNFREEZE_MBOXFIX(1);
    SNB_UNFREEZE_MBOXFIX(2);
    SNB_UNFREEZE_MBOXFIX(3);

    SNB_UNFREEZE_PCI_BOX(SBOX0, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_BOX_CTL);
    SNB_UNFREEZE_PCI_BOX(SBOX1, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_BOX_CTL);

    SNB_UNFREEZE_PCI_BOX(BBOX0, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_BOX_CTL);
    SNB_UNFREEZE_BOX(WBOX, MSR_UNC_PCU_PMON_BOX_CTL1);

    return 0;
}

