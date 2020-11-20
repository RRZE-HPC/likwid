/*
 * =======================================================================================
 *
 *      Filename:  perfmon_knl.h
 *
 *      Description:  Header file of perfmon module for Intel Xeon Phi (Knights Landing)
 *
 *      Version:   5.1.0
 *      Released:  20.11.2020
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2020 RRZE, University Erlangen-Nuremberg
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

#include <perfmon_knl_events.h>
#include <perfmon_knl_counters.h>

static int perfmon_numCountersKNL = NUM_COUNTERS_KNL;
static int perfmon_numCoreCountersKNL = NUM_COUNTERS_KNL;
static int perfmon_numArchEventsKNL = NUM_ARCH_EVENTS_KNL;


int perfmon_init_knl(int cpu_id)
{
    lock_acquire((int*) &socket_lock[affinity_thread2socket_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &tile_lock[affinity_thread2core_lookup[cpu_id]], cpu_id);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL));
    return 0;
}

uint32_t knl_fixed_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint32_t flags = (1ULL<<(1+(index*4)));
    cpu_id++;
    if (event->numberOfOptions > 0)
    {
        for(int i=0;i<event->numberOfOptions;i++)
        {
            switch(event->options[i].type)
            {
                case EVENT_OPTION_ANYTHREAD:
                    flags |= (1ULL<<(2+(index*4)));
                    break;
                case EVENT_OPTION_COUNT_KERNEL:
                    flags |= (1ULL<<(index*4));
                    break;
                default:
                    break;
            }
        }
    }
    return flags;
}

int knl_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x0ULL;
    uint64_t offcore_flags = 0x0ULL;

    if (event->eventId == 0xB7 && tile_lock[affinity_thread2core_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags |= (1ULL<<16)|(1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;
    /* For event id 0xB7 the cmask must be written in an extra register */
    if ((event->cmask != 0x00) && (event->eventId != 0xB7))
    {
        flags |= (event->cmask << 24);
    }
    /* set custom cfgbits */
    if ((event->cfgBits != 0x00) && (event->eventId != 0xB7))
    {
        flags |= (event->cfgBits << 16);
    }

    if (event->numberOfOptions > 0)
    {
        for(int i=0;i<event->numberOfOptions;i++)
        {
            switch(event->options[i].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                case EVENT_OPTION_ANYTHREAD:
                    flags |= (1ULL<<21);
                    break;
                case EVENT_OPTION_INVERT:
                    flags |= (1ULL<<23);
                    break;
                case EVENT_OPTION_COUNT_KERNEL:
                    flags |= (1ULL<<17);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= (event->options[i].value & 0xFFULL)<<24;
                    break;
                case EVENT_OPTION_MATCH0:
                    if (event->eventId == 0xB7)
                    {
                        offcore_flags |= (event->options[i].value & 0xFFFFULL);
                    }
                    break;
                case EVENT_OPTION_MATCH1:
                    if (event->eventId == 0xB7)
                    {
                        offcore_flags |= (event->options[i].value & 0x3FFFFFFFULL)<<16;
                    }
                    break;
                default:
                    break;
            }
        }
    }

    // Offcore event with additional configuration register
    // cfgBits contain offset of "request type" bit
    // cmask contain offset of "response type" bit
    if (event->eventId == 0xB7)
    {
        uint32_t reg = 0x0;
        if (event->umask == 0x01)
        {
            reg = MSR_OFFCORE_RESP0;
        }
        else if (event->umask == 0x02)
        {
            reg = MSR_OFFCORE_RESP1;
        }
        if (reg)
        {
            if ((event->cfgBits != 0xFF) && (event->cmask != 0xFF))
            {
                offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);
            }
            VERBOSEPRINTREG(cpu_id, reg, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , offcore_flags));
        }
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_PMC)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int knl_ubox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x0ULL;
    uint64_t offcore_flags = 0x0ULL;


    flags |= (1ULL<<16)|(1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;
    if (event->numberOfOptions > 0)
    {
        for(int i=0;i<event->numberOfOptions;i++)
        {
            switch(event->options[i].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                case EVENT_OPTION_ANYTHREAD:
                    flags |= (1ULL<<21);
                    break;
                case EVENT_OPTION_INVERT:
                    flags |= (1ULL<<23);
                    break;
                case EVENT_OPTION_TID:
                    flags |= (1ULL<<19);
                    break;
                default:
                    break;
            }
        }
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_UBOX)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int knl_wbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags = (1ULL<<22)|(1ULL<<20);
    flags |= event->eventId;
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
                    flags |= (1ULL<<7);
                    break;
                case EVENT_OPTION_OCCUPANCY_EDGE:
                    flags |= (1ULL<<31);
                    flags |= (1ULL<<7);
                    break;
                case EVENT_OPTION_OCCUPANCY_INVERT:
                    flags |= (1ULL<<30);
                    flags |= (1ULL<<7);
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

int knl_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    uint64_t filter_flags0 = 0x0ULL;
    uint64_t filter_flags1 = 0x0ULL;
    uint32_t filter0 = box_map[counter_map[index].type].filterRegister1;
    uint32_t filter1 = box_map[counter_map[index].type].filterRegister2;
    int set_state_all = 0;
    int set_opcode_all = 0;
    int set_match1_all = 1;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags = (1ULL<<22)|(1ULL<<20);
    flags |= (event->umask<<8) + event->eventId;
    if (event->eventId == 0x34)
    {
        set_state_all = 1;
    }
    if (event->eventId == 0x00 && event->cfgBits == 1)
    {
        filter_flags0 |= (1ULL<<12);
    }
    if (event->eventId == 0x00 || event->eventId == 0x00)
    {
        set_opcode_all = 1;
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
                    set_opcode_all = 0;
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
                    filter_flags1 |= (extractBitField(event->options[j].value,3,0) << 29);
                    break;
                case EVENT_OPTION_MATCH1:
                    filter_flags1 |= (extractBitField(event->options[j].value,2,0) << 4);
                    set_match1_all = 0;
                    break;
                case EVENT_OPTION_NID:
                    filter_flags1 |= extractBitField(event->options[j].value,2,0);
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
        filter_flags0 |= (1ULL<<18)|(1ULL<<19)|(1ULL<<20);
        VERBOSEPRINTREG(cpu_id, filter0, filter_flags0, SETUP_CBOX_DEF_FILTER_STATE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, filter0, filter_flags0));
    }
    if (set_match1_all)
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, filter1, &filter_flags1));
        filter_flags1 |= (1ULL<<4)|(1ULL<<5);
        VERBOSEPRINTREG(cpu_id, filter1, filter_flags1, SETUP_CBOX_COUNT_ALL_CACHE_EVENTS);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, filter1, filter_flags1));
    }
    if (set_opcode_all)
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, filter1, &filter_flags1));
        filter_flags1 |= (1ULL<<3);
        VERBOSEPRINTREG(cpu_id, filter1, filter_flags1, SETUP_CBOX_COUNT_ALL_OPCODES);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, filter1, filter_flags1));
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_CBOX);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int knl_mbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
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
        VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].configRegister, flags, SETUP_BOX);
        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}




int perfmon_setupCountersThread_knl(
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
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x0ULL, FREEZE_FIXED);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_FIXED_CTR_CTRL, 0x0ULL));
    }

    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_MIC2_U_GLOBAL_CTRL, 0x0ULL, FREEZE_UNCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC2_U_GLOBAL_CTRL, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC2_U_GLOBAL_CTRL, (1ULL<<63)));
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
        switch (type)
        {
            case PMC:
                knl_pmc_setup(cpu_id, index, event);
                break;

            case FIXED:
                fixed_flags |= knl_fixed_setup(cpu_id, index, event);
                break;

            case POWER:
                break;

            case UBOX:
                knl_ubox_setup(cpu_id, index, event);
                break;

            case UBOXFIX:
                if (haveLock)
                {
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, (1ULL<<22)|(1ULL<<20)));
                    VERBOSEPRINTREG(cpu_id, reg, (1ULL<<22)|(1ULL<<20), SETUP_UBOXFIX);
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
                knl_cbox_setup(cpu_id, index, event);
                break;

            case WBOX:
                knl_wbox_setup(cpu_id, index, event);
                break;

            case MBOX0:
            case MBOX1:
            case MBOX2:
            case MBOX3:
            case MBOX4:
            case MBOX5:
            case MBOX6:
            case MBOX7:
            case IBOX0:
            case EUBOX0:
            case EUBOX1:
            case EUBOX2:
            case EUBOX3:
            case EUBOX4:
            case EUBOX5:
            case EUBOX6:
            case EUBOX7:
            case EDBOX0:
            case EDBOX1:
            case EDBOX2:
            case EDBOX3:
            case EDBOX4:
            case EDBOX5:
            case EDBOX6:
            case EDBOX7:
            case PBOX:
                knl_mbox_setup(cpu_id, index, event);
                break;

            case MBOX0FIX:
            case MBOX1FIX:
            case MBOX2FIX:
            case MBOX3FIX:
            case MBOX4FIX:
            case MBOX5FIX:
            case MBOX6FIX:
            case MBOX7FIX:
            case EUBOX0FIX:
            case EUBOX1FIX:
            case EUBOX2FIX:
            case EUBOX3FIX:
            case EUBOX4FIX:
            case EUBOX5FIX:
            case EUBOX6FIX:
            case EUBOX7FIX:
            case EDBOX0FIX:
            case EDBOX1FIX:
            case EDBOX2FIX:
            case EDBOX3FIX:
            case EDBOX4FIX:
            case EDBOX5FIX:
            case EDBOX6FIX:
            case EDBOX7FIX:
                if (haveLock)
                {
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, 0x1ULL));
                    VERBOSEPRINTREG(cpu_id, reg, 0x1ULL, SETUP_MBOXFIX);
                }
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

#define KNL_FREEZE_UNCORE \
    if (haveLock && MEASURE_UNCORE(eventSet)) \
    { \
        VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<63), FREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_V3_U_PMON_GLOBAL_CTL, (1ULL<<63))); \
    }

#define KNL_UNFREEZE_UNCORE \
    if (haveLock && MEASURE_UNCORE(eventSet)) \
    { \
        VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<61), UNFREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_UNC_V3_U_PMON_GLOBAL_CTL, (1ULL<<61))); \
    }

#define KNL_UNFREEZE_UNCORE_AND_RESET_CTR \
    if (haveLock && MEASURE_UNCORE(eventSet)) \
    { \
        for (int i=0;i < eventSet->numberOfEvents;i++) \
        { \
            RegisterIndex index = eventSet->events[i].index; \
            RegisterType type = counter_map[index].type; \
            if (type < UNCORE) \
            { \
                continue; \
            } \
            PciDeviceIndex dev = counter_map[index].device; \
            if (HPMcheck(dev, cpu_id) && TESTTYPE(eventSet, type)) \
            { \
                VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL, CLEAR_CTR_MANUAL); \
                CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL)); \
                if (counter_map[index].counterRegister2 != 0x0) \
                { \
                    VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].counterRegister2, 0x0ULL, CLEAR_CTR_MANUAL); \
                    CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister2, 0x0ULL)); \
                } \
            } \
        } \
        VERBOSEPRINTREG(cpu_id, MSR_MIC2_U_GLOBAL_CTRL, LLU_CAST (1ULL<<61), UNFREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC2_U_GLOBAL_CTRL, (1ULL<<61))); \
    }


int perfmon_startCountersThread_knl(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t tmp;
    uint64_t flags = 0x0ULL;
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
            PciDeviceIndex dev = counter_map[index].device;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;
            eventSet->events[i].threadCounter[thread_id].startData = 0;
            eventSet->events[i].threadCounter[thread_id].counterData = 0;
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
                        eventSet->events[i].threadCounter[thread_id].startData = field64(tmp, 0, box_map[type].regWidth);
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
                case EUBOX0FIX:
                case EUBOX1FIX:
                case EUBOX2FIX:
                case EUBOX3FIX:
                case EUBOX4FIX:
                case EUBOX5FIX:
                case EUBOX6FIX:
                case EUBOX7FIX:
                case EDBOX0FIX:
                case EDBOX1FIX:
                case EDBOX2FIX:
                case EDBOX3FIX:
                case EDBOX4FIX:
                case EDBOX5FIX:
                case EDBOX6FIX:
                case EDBOX7FIX:
                    if (haveLock)
                    {
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                        CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev, counter2, 0x0ULL));
                    }
                    break;

                default:
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = eventSet->events[i].threadCounter[thread_id].startData;
        }
    }

    KNL_UNFREEZE_UNCORE_AND_RESET_CTR;

    if (MEASURE_CORE(eventSet))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, flags));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, UNFREEZE_PMC_OR_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, flags));
    }

    return 0;
}

int knl_uncore_read(int cpu_id, RegisterIndex index, PerfmonEvent *event,
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
        if (global_offset != -1)
        {
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV,
                                           MSR_PERF_GLOBAL_STATUS,
                                           &ovf_values));
            VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_STATUS, LLU_CAST ovf_values, READ_GLOBAL_OVFL);
            if (ovf_values & (1<<global_offset))
            {
                VERBOSEPRINTREG(cpu_id, MSR_MIC2_U_GLOBAL_STATUS, LLU_CAST (1<<global_offset), CLEAR_GLOBAL_OVFL);
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV,
                                                 MSR_MIC2_U_GLOBAL_STATUS,
                                                 (1<<global_offset)));
                test_local = 1;
            }
        }
        else
        {
            test_local = 1;
        }

        if (test_local && box_map[type].statusRegister != 0x0)
        {
            ovf_values = 0x0ULL;
            CHECK_PCI_READ_ERROR(HPMread(cpu_id, dev,
                                              box_map[type].statusRegister,
                                              &ovf_values));
            VERBOSEPRINTPCIREG(cpu_id, dev, box_map[type].statusRegister, LLU_CAST ovf_values, READ_BOX_OVFL);
            if (ovf_values & (1<<box_offset))
            {
                (*overflows)++;
                VERBOSEPRINTPCIREG(cpu_id, dev, box_map[type].statusRegister, LLU_CAST (1<<box_offset), RESET_BOX_OVFL);
                CHECK_PCI_WRITE_ERROR(HPMwrite(cpu_id, dev,
                                                    box_map[type].statusRegister,
                                                    (1<<box_offset)));
            }
        }
        else if ((ovf_values & (1<<global_offset)) && test_local)
        {
            (*overflows)++;
        }
    }
    *cur_result = result;
    return 0;
}

#define KNL_CHECK_CORE_OVERFLOW(offset) \
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

int perfmon_stopCountersThread_knl(int thread_id, PerfmonEventSet* eventSet)
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
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_OR_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    KNL_FREEZE_UNCORE;

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
            uint64_t counter2 = counter_map[index].counterRegister2;
            uint64_t* current = &(eventSet->events[i].threadCounter[thread_id].counterData);
            int* overflows = &(eventSet->events[i].threadCounter[thread_id].overflows);
            int ovf_offset = box_map[type].ovflOffset;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    KNL_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC);
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    KNL_CHECK_CORE_OVERFLOW(index+32);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED);
                    break;

                case POWER:
                    if(haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&counter_result));
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                    }
                    break;

                case THERMAL:
                    CHECK_TEMP_READ_ERROR(thermal_read(cpu_id, (uint32_t*)&counter_result));
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
                case MBOX0:
                case MBOX1:
                case MBOX2:
                case MBOX3:
                case MBOX4:
                case MBOX5:
                case MBOX6:
                case MBOX7:
                case IBOX0:
                case EUBOX0:
                case EUBOX1:
                case EUBOX2:
                case EUBOX3:
                case EUBOX4:
                case EUBOX5:
                case EUBOX6:
                case EUBOX7:
                case EDBOX0:
                case EDBOX1:
                case EDBOX2:
                case EDBOX3:
                case EDBOX4:
                case EDBOX5:
                case EDBOX6:
                case EDBOX7:
                case WBOX:
                case UBOX:
                case PBOX:
                    knl_uncore_read(cpu_id, index, event, &counter_result, overflows,
                                    FREEZE_FLAG_CLEAR_CTR, ovf_offset, getCounterTypeOffset(index));
                    break;

                case MBOX0FIX:
                case MBOX1FIX:
                case MBOX2FIX:
                case MBOX3FIX:
                case MBOX4FIX:
                case MBOX5FIX:
                case MBOX6FIX:
                case MBOX7FIX:
                case EUBOX0FIX:
                case EUBOX1FIX:
                case EUBOX2FIX:
                case EUBOX3FIX:
                case EUBOX4FIX:
                case EUBOX5FIX:
                case EUBOX6FIX:
                case EUBOX7FIX:
                case EDBOX0FIX:
                case EDBOX1FIX:
                case EDBOX2FIX:
                case EDBOX3FIX:
                case EDBOX4FIX:
                case EDBOX5FIX:
                case EDBOX6FIX:
                case EDBOX7FIX:
                    if (haveLock)
                    {
                        uint64_t tmp = 0x0ULL;
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter1, LLU_CAST counter_result, READ_FIXED_BOX_1);
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter2, &tmp));
                        VERBOSEPRINTPCIREG(cpu_id, dev, counter2, LLU_CAST tmp , READ_FIXED_BOX_1);
                        counter_result = (counter_result<<32)|tmp;
                        counter_result = field64(counter_result, 0, box_map[type].regWidth);
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                    }
                    break;

                default:
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
        }
    }
    return 0;
}

int perfmon_readCountersThread_knl(int thread_id, PerfmonEventSet* eventSet)
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
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_OR_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    KNL_FREEZE_UNCORE;

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
                    KNL_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC);
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    KNL_CHECK_CORE_OVERFLOW(index+32);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED);
                    break;

                case POWER:
                    if(haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&counter_result));
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                    }
                    break;

                case THERMAL:
                    CHECK_TEMP_READ_ERROR(thermal_read(cpu_id, (uint32_t*)&counter_result));
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
                case MBOX0:
                case MBOX1:
                case MBOX2:
                case MBOX3:
                case MBOX4:
                case MBOX5:
                case MBOX6:
                case MBOX7:
                case IBOX0:
                case EUBOX0:
                case EUBOX1:
                case EUBOX2:
                case EUBOX3:
                case EUBOX4:
                case EUBOX5:
                case EUBOX6:
                case EUBOX7:
                case EDBOX0:
                case EDBOX1:
                case EDBOX2:
                case EDBOX3:
                case EDBOX4:
                case EDBOX5:
                case EDBOX6:
                case EDBOX7:
                case WBOX:
                case UBOX:
                case PBOX:
                    knl_uncore_read(cpu_id, index, event, &counter_result, overflows,
                                    FREEZE_FLAG_ONLYFREEZE, ovf_offset, getCounterTypeOffset(index));
                    break;

                case MBOX0FIX:
                case MBOX1FIX:
                case MBOX2FIX:
                case MBOX3FIX:
                case MBOX4FIX:
                case MBOX5FIX:
                case MBOX6FIX:
                case MBOX7FIX:
                case EUBOX0FIX:
                case EUBOX1FIX:
                case EUBOX2FIX:
                case EUBOX3FIX:
                case EUBOX4FIX:
                case EUBOX5FIX:
                case EUBOX6FIX:
                case EUBOX7FIX:
                case EDBOX0FIX:
                case EDBOX1FIX:
                case EDBOX2FIX:
                case EDBOX3FIX:
                case EDBOX4FIX:
                case EDBOX5FIX:
                case EDBOX6FIX:
                case EDBOX7FIX:
                    if (haveLock)
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &counter_result));
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                    }
                    break;

                default:
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
        }
    }
    KNL_UNFREEZE_UNCORE;
    if (MEASURE_CORE(eventSet))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, pmc_flags));
    }
    return 0;
}


int perfmon_finalizeCountersThread_knl(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int haveTileLock = 1;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t ovf_values_core = (1ULL<<63)|(1ULL<<62);

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }
    /*if (tile_lock[affinity_thread2core_lookup[cpu_id]] == cpu_id)
    {
        haveTileLock = 1;
    }*/

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
        switch (type)
        {
            case PMC:
                ovf_values_core |= (1ULL<<(index-cpuid_info.perf_num_fixed_ctr));
                if ((haveTileLock) && (event->eventId == 0xB7))
                {
                    if (event->umask == 0x1)
                    {
                        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, 0x0ULL, CLEAR_OFFCORE_RESP0);
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, 0x0ULL));
                    }
                    else if (event->umask == 0x2)
                    {
                        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP1, 0x0ULL, CLEAR_OFFCORE_RESP1);
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP1, 0x0ULL));
                    }
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
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, 0x0ULL));
        }
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
    }

    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        uint64_t ovf_values_uncore = 0x0ULL;
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV,MSR_MIC2_U_GLOBAL_STATUS, &ovf_values_uncore));
        VERBOSEPRINTREG(cpu_id, MSR_MIC2_U_GLOBAL_STATUS, LLU_CAST ovf_values_uncore, CLEAR_UNCORE_OVF)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC2_U_GLOBAL_STATUS, ovf_values_uncore));
        VERBOSEPRINTREG(cpu_id, MSR_MIC2_U_GLOBAL_CTRL, LLU_CAST (1ULL<<59), CLEAR_UNCORE_CTRL)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_MIC2_U_GLOBAL_CTRL, (1ULL<<59)));
    }

    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST 0x0ULL, CLEAR_GLOBAL_CTRL)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, LLU_CAST ovf_values_core, CLEAR_GLOBAL_OVF)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, ovf_values_core));
    }
    return 0;
}
