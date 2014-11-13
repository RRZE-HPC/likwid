/*
 * =======================================================================================
 *
 *      Filename:  perfmon_haswell.h
 *
 *      Description:  Header File of perfmon module for Haswell.
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

#include <perfmon_haswellEP_events.h>
#include <perfmon_haswellEP_counters.h>
#include <error.h>
#include <affinity.h>
#include <limits.h>
#include <topology.h>


static int perfmon_numCountersHaswellEP = NUM_COUNTERS_HASWELL_EP;
static int perfmon_numCoreCountersHaswellEP = NUM_COUNTERS_CORE_HASWELL_EP;
static int perfmon_numArchEventsHaswellEP = NUM_ARCH_EVENTS_HASWELLEP;


int perfmon_init_haswellEP(int cpu_id)
{
    uint64_t flags = 0x0ULL;
    lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    return 0;
}

#define HASEP_FREEZE_UNCORE \
    if (haveLock && eventSet->regTypeMask & ~(0xFULL)) \
    { \
        VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<31), FREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_CTL, (1ULL<<31))); \
    }

#define HASEP_UNFREEZE_UNCORE \
    if (haveLock && eventSet->regTypeMask & ~(0xFULL)) \
    { \
        VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<29), UNFREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_CTL, (1ULL<<29))); \
    }

#define HASEP_UNFREEZE_UNCORE_AND_RESET_CTR \
    if (haveLock && (eventSet->regTypeMask & ~(REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)|REG_TYPE_MASK(THERMAL)))) \
    { \
        for (int j=0; j<NUM_UNITS; j++) \
        { \
            if (eventSet->regTypeMask & REG_TYPE_MASK(j)) \
            { \
                if (box_map[j].ctrlRegister != 0x0ULL) \
                { \
                    VERBOSEPRINTREG(cpu_id, box_map[j].ctrlRegister, LLU_CAST 0x2ULL, CLEAR_CTR); \
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, box_map[j].ctrlRegister, 0x2ULL)); \
                } \
                else \
                { \
                    for (int k=0;k<perfmon_numCounters;k++) \
                    { \
                        if (counter_map[k].type == j) \
                        { \
                            VERBOSEPRINTREG(cpu_id, counter_map[k].counterRegister, 0x0ULL, CLEAR_CTR_MANUAL); \
                            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, counter_map[k].counterRegister, 0x0ULL)); \
                            if (counter_map[k].counterRegister2 != 0x0) \
                            { \
                                VERBOSEPRINTREG(cpu_id, counter_map[k].counterRegister2, 0x0ULL, CLEAR_CTR_MANUAL); \
                                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, counter_map[k].counterRegister2, 0x0ULL)); \
                            } \
                        } \
                    } \
                } \
            } \
        } \
        VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<29), UNFREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_CTL, (1ULL<<29))); \
    }

#define HASEP_FREEZE_UNCORE_AND_RESET_CTL \
    if (haveLock && (eventSet->regTypeMask & ~(REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)|REG_TYPE_MASK(THERMAL)))) \
    { \
        VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<31), FREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_CTL, (1ULL<<31))); \
        for (int j=0; j<NUM_UNITS; j++) \
        { \
            if (eventSet->regTypeMask & REG_TYPE_MASK(j)) \
            { \
                if (box_map[j].ctrlRegister != 0x0ULL) \
                { \
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, box_map[j].ctrlRegister, 0x1ULL)); \
                    VERBOSEPRINTREG(cpu_id, box_map[j].ctrlRegister, LLU_CAST 0x1ULL, CLEAR_CTL); \
                } \
                else \
                { \
                    for (int k=0;k<perfmon_numCounters;k++) \
                    { \
                        if (counter_map[k].type == j) \
                        { \
                            VERBOSEPRINTREG(cpu_id, counter_map[k].configRegister, 0x0ULL, CLEAR_CTL_MANUAL); \
                            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, counter_map[k].configRegister, 0x0ULL)); \
                        } \
                    } \
                } \
            } \
        } \
    }

#define HASEP_SETUP_CBOX(id) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(CBOX##id))) \
    { \
        uint64_t filter_flags; \
        int set_state_all = 0; \
        flags = (1ULL<<22); \
        flags |= (event->umask<<8) + event->eventId; \
        if (event->eventId == 0x34) \
        { \
            set_state_all = 1; \
        } \
        if (event->numberOfOptions > 0) \
        { \
            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER0, 0x0ULL)); \
            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER1, 0x0ULL)); \
            for(int j=0;j<event->numberOfOptions;j++) \
            { \
                filter_flags = 0x0ULL; \
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_EDGE: \
                        flags |= (1ULL<<18); \
                        break; \
                    case EVENT_OPTION_INVERT: \
                        flags |= (1ULL<<23); \
                        break; \
                    case EVENT_OPTION_THRESHOLD: \
                        flags |= (event->options[j].value<<24); \
                        break; \
                    case EVENT_OPTION_OPCODE: \
                        CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER1, &filter_flags)); \
                        filter_flags |= (0x3<<27); \
                        filter_flags |= (extractBitField(event->options[j].value,5,0) << 20);\
                        VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER1, filter_flags, SETUP_CBOX##id##_FILTER_OPCODE); \
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER1, filter_flags)); \
                        break; \
                    case EVENT_OPTION_NID: \
                        CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER1, &filter_flags)); \
                        filter_flags |= (extractBitField(event->options[j].value,16,0));\
                        VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER1, filter_flags, SETUP_CBOX##id##_FILTER_NID); \
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER1, filter_flags)); \
                        break; \
                    case EVENT_OPTION_STATE: \
                        CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER0, &filter_flags)); \
                        filter_flags |= (extractBitField(event->options[j].value,6,0) << 17);\
                        VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER0, filter_flags, SETUP_CBOX##id##_FILTER_STATE); \
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER0, filter_flags)); \
                        set_state_all = 0; \
                        break; \
                    case EVENT_OPTION_TID: \
                        CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER0, &filter_flags)); \
                        filter_flags |= (extractBitField(event->options[j].value,6,0));\
                        VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER0, filter_flags, SETUP_CBOX##id##_FILTER_TID); \
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER0, filter_flags)); \
                        flags |= (1ULL<<19); \
                        break; \
                    default: \
                        break; \
                } \
            } \
        } \
        else \
        { \
            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER0, 0x0ULL)); \
            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER1, 0x0ULL)); \
        } \
        if (set_state_all) \
        { \
            CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER0, &filter_flags)); \
            filter_flags |= (0x1F << 17);\
            VERBOSEPRINTREG(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER0, filter_flags, SETUP_CBOX##id##_DEF_FILTER_STATE); \
            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_V3_C##id##_PMON_BOX_FILTER0, filter_flags)); \
        } \
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_CBOX##id); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, flags)); \
    }

#define HASEP_SETUP_SBOX(id) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(SBOX##id))) \
    { \
        flags = (1ULL<<22); \
        flags |= (event->umask<<8) + event->eventId; \
        if (event->numberOfOptions > 0) \
        { \
            for(int j=0;j<event->numberOfOptions;j++) \
            { \
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_EDGE: \
                        flags |= (1ULL<<18); \
                        break; \
                    case EVENT_OPTION_INVERT: \
                        flags |= (1ULL<<23); \
                        break; \
                    case EVENT_OPTION_THRESHOLD: \
                        flags |= (extractBitField(event->options[j].value,8,0)<<24); \
                        break; \
                    case EVENT_OPTION_TID: \
                        flags |= (1ULL<<23); \
                        break; \
                } \
            } \
        } \
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_SBOX##id); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, flags & ~(1ULL<<18))); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, flags)); \
    }

#define HASEP_SETUP_BOX(id) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(id))) \
    { \
        flags = (1ULL<<22)|(1ULL<<20); \
        flags |= (event->umask<<8) + event->eventId; \
        if (event->numberOfOptions > 0) \
        { \
            for(int j=0;j<event->numberOfOptions;j++) \
            { \
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_EDGE: \
                        flags |= (1ULL<<18); \
                        break; \
                    case EVENT_OPTION_INVERT: \
                        flags |= (1ULL<<23); \
                        break; \
                    case EVENT_OPTION_THRESHOLD: \
                        flags |= (extractBitField(event->options[j].value,5,0)<<24); \
                        break; \
                } \
            } \
        } \
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_##id); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, flags)); \
    }

#define HASEP_SETUP_PCI_BOX(id) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(id))) \
    { \
        flags = (1ULL<<22)|(1ULL<<20); \
        flags |= (event->umask<<8) + event->eventId; \
        if (event->numberOfOptions > 0) \
        { \
            for(int j=0;j<event->numberOfOptions;j++) \
            { \
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_EDGE: \
                        flags |= (1ULL<<18); \
                        break; \
                    case EVENT_OPTION_INVERT: \
                        flags |= (1ULL<<23); \
                        break; \
                    case EVENT_OPTION_THRESHOLD: \
                        flags |= (extractBitField(event->options[j].value,8,0)<<24); \
                        break; \
                } \
            } \
        } \
        VERBOSEPRINTPCIREG(cpu_id, dev, reg, flags, SETUP_##id); \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, dev, reg, flags)); \
    }

int perfmon_setupCounterThread_haswellEP(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int ret;
    uint64_t flags;
    uint32_t uflags;
    uint64_t fixed_flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PEBS_ENABLE, 0x0ULL));
    }
    HASEP_FREEZE_UNCORE;
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = counter_map[index].configRegister;
        PciDeviceIndex dev = counter_map[index].device;
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        flags = 0x0ULL;
        switch (eventSet->events[i].type)
        {
            case PMC:
                if (eventSet->regTypeMask & REG_TYPE_MASK(PMC))
                {
                    flags = (1ULL<<22)|(1ULL<<16);

                    /* Intel with standard 8 bit event mask: [7:0] */
                    flags |= (event->umask<<8) + event->eventId;

                    if (event->cfgBits != 0) /* set custom cfg and cmask */
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
                                case EVENT_OPTION_IN_TRANS:
                                    flags |= (1ULL<<32);
                                    break;
                                case EVENT_OPTION_IN_TRANS_ABORT:
                                    flags |= (1ULL<<33);
                                    break;
                                default:
                                    break;
                            }
                        }
                    }
                    VERBOSEPRINTREG(cpu_id, reg, LLU_CAST flags, SETUP_PMC)
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags));
                }
                break;

            case FIXED:
                if (eventSet->regTypeMask & REG_TYPE_MASK(FIXED))
                {
                    fixed_flags |= (0x2ULL << (4*index));
                    if (event->numberOfOptions > 0)
                    {
                        for(int j=0;j<event->numberOfOptions;j++)
                        {
                            switch (event->options[j].type)
                            {
                                case EVENT_OPTION_COUNT_KERNEL:
                                    fixed_flags |= (1ULL<<(index*4));
                                    break;
                                case EVENT_OPTION_ANYTHREAD:
                                    fixed_flags |= (1ULL<<(2+(index*4)));
                                    break;
                                default:
                                    break;
                            }
                        }
                    }
                }
                break;

            case POWER:
                break;

            case CBOX0:
                HASEP_SETUP_CBOX(0);
                break;
            case CBOX1:
                HASEP_SETUP_CBOX(1);
                break;
            case CBOX2:
                HASEP_SETUP_CBOX(2);
                break;
            case CBOX3:
                HASEP_SETUP_CBOX(3);
                break;
            case CBOX4:
                HASEP_SETUP_CBOX(4);
                break;
            case CBOX5:
                HASEP_SETUP_CBOX(5);
                break;
            case CBOX6:
                HASEP_SETUP_CBOX(6);
                break;
            case CBOX7:
                HASEP_SETUP_CBOX(7);
                break;
            case CBOX8:
                HASEP_SETUP_CBOX(8);
                break;
            case CBOX9:
                HASEP_SETUP_CBOX(9);
                break;
            case CBOX10:
                HASEP_SETUP_CBOX(10);
                break;
            case CBOX11:
                HASEP_SETUP_CBOX(11);
                break;
            case CBOX12:
                HASEP_SETUP_CBOX(12);
                break;
            case CBOX13:
                HASEP_SETUP_CBOX(13);
                break;
            case CBOX14:
                HASEP_SETUP_CBOX(14);
                break;
            case CBOX15:
                HASEP_SETUP_CBOX(15);
                break;
            case CBOX16:
                HASEP_SETUP_CBOX(16);
                break;
            case CBOX17:
                HASEP_SETUP_CBOX(17);
                break;

            case UBOX:
                HASEP_SETUP_BOX(UBOX);
                break;
            case UBOXFIX:
                flags = (1ULL<<22)|(1ULL<<20);
                VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_UBOXFIX);
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, flags));
                break;

            case SBOX0:
                HASEP_SETUP_SBOX(0);
                break;
            case SBOX1:
                HASEP_SETUP_SBOX(1);
                break;
            case SBOX2:
                HASEP_SETUP_SBOX(2);
                break;
            case SBOX3:
                HASEP_SETUP_SBOX(3);
                break;

            case BBOX0:
                HASEP_SETUP_PCI_BOX(BBOX0);
                break;
            case BBOX1:
                HASEP_SETUP_PCI_BOX(BBOX1);
                break;

            default:
                /* should never be reached */
                break;
        }
    }
    if (fixed_flags > 0x0ULL)
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_FIXED_CTR_CTRL, LLU_CAST fixed_flags, SETUP_FIXED)
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, fixed_flags));
    }
    return 0;
}

int perfmon_startCountersThread_haswellEP(int thread_id, PerfmonEventSet* eventSet)
{
    int ret;
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint32_t uflags = 0x10000UL; /* Clear freeze bit */
    int cpu_id = groupSet->threads[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    //CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;
            switch (eventSet->events[i].type)
            {
                case PMC:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, counter1, 0x0ULL));
                    flags |= (1ULL<<(index-cpuid_info.perf_num_fixed_ctr));  /* enable counter */
                    break;

                case FIXED:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, counter1, 0x0ULL));
                    flags |= (1ULL<<(index+32));  /* enable fixed counter */
                    break;

                case POWER:
                    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(POWER)))
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1,
                                        (uint32_t*)&eventSet->events[i].threadCounter[thread_id].startData));
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST eventSet->events[i].threadCounter[thread_id].startData, START_POWER)
                    }
                    break;

                default:
                    /* should never be reached */
                    break;
            }
        }
    }

    HASEP_UNFREEZE_UNCORE_AND_RESET_CTR;

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, UNFREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, flags));
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x30000000FULL));
    }

    return 0;
}

#define HASEP_CHECK_CORE_OVERFLOW(offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t ovf_values = 0x0ULL; \
        CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, MSR_PERF_GLOBAL_STATUS, &ovf_values)); \
        if (ovf_values & (1ULL<<offset)) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
        } \
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<offset))); \
    }

#define HASEP_CHECK_UNCORE_OVERFLOW(offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t ovf_values = 0x0ULL; \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_STATUS, &ovf_values)); \
        if (ovf_values & (1ULL<<offset)) \
        { \
            CHECK_MSR_READ_ERROR(msr_read(cpu_id, box_map[eventSet->events[i].type].statusRegister, &ovf_values)); \
            uint64_t looffset = getCounterTypeOffset(eventSet->events[i].index); \
            if (ovf_values & (1ULL<<looffset)) \
            { \
                eventSet->events[i].threadCounter[thread_id].overflows++; \
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, box_map[eventSet->events[i].type].statusRegister, (1ULL<<looffset))); \
            } \
        } \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_V3_U_PMON_GLOBAL_STATUS, (1ULL<<offset) ) ); \
    }

#define HASEP_CHECK_LOCAL_OVERFLOW \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t ovf_values = 0x0ULL; \
        uint64_t offset = getCounterTypeOffset(eventSet->events[i].index); \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, box_map[eventSet->events[i].type].statusRegister, &ovf_values)); \
        if (ovf_values & (1ULL<<offset)) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, box_map[eventSet->events[i].type].statusRegister, (1ULL<<offset))); \
        } \
    }


#define HASEP_READ_BOX(id, reg1) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) \
    { \
        VERBOSEPRINTREG(cpu_id, reg1, LLU_CAST counter_result, READ_BOX_##id) \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, reg1, &counter_result)); \
    }

#define HASEP_READ_BOX_SOCKET(socket, id, reg1) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) \
    { \
        VERBOSEPRINTREG(cpu_id, reg1, LLU_CAST counter_result, READ_BOX_##id) \
        CHECK_MSR_READ_ERROR(msr_tread(socket, cpu_id, reg1, &counter_result)); \
    }

#define HASEP_READ_PCI_BOX(id, reg1, reg2) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) \
    { \
        uint32_t tmp = 0; \
        CHECK_PCI_READ_ERROR(pci_read(cpu_id, box_map[id].device, reg1, (uint32_t*)&counter_result)); \
        CHECK_PCI_READ_ERROR(pci_read(cpu_id, box_map[id].device, reg2, &tmp)); \
        counter_result = (counter_result<<32) + tmp; \
        VERBOSEPRINTPCIREG(cpu_id, box_map[id].device, reg1, LLU_CAST counter_result, READ_PCI_BOX_##id); \
    }

#define HASEP_READ_PCI_BOX_SOCKET(socket, id, reg1, reg2) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) \
    { \
        uint32_t tmp = 0; \
        CHECK_PCI_READ_ERROR(pci_tread(socket, cpu_id, box_map[id].device, reg1, (uint32_t*)&counter_result)); \
        CHECK_PCI_READ_ERROR(pci_tread(socket, cpu_id, box_map[id].device, reg2, &tmp)); \
        counter_result = (counter_result<<32) + tmp; \
        VERBOSEPRINTPCIREG(cpu_id, box_map[id].device, reg1, LLU_CAST counter_result, READ_PCI_BOX_##id); \
    }

int perfmon_stopCountersThread_haswellEP(int thread_id, PerfmonEventSet* eventSet)
{
    int bit;
    int haveLock = 0;
    uint64_t flags;
    uint32_t uflags = 0x10100UL; /* Set freeze bit */
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    int read_fd = -1;
    read_fd = socket_fd;
    if (socket_fd == -1)
    {
        read_fd = thread_sockets[cpu_id];
    }
    if (read_fd == -1)
    {
        return -ENOENT;
    }

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    HASEP_FREEZE_UNCORE_AND_RESET_CTL;


    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            counter_result= 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            PciDeviceIndex dev = counter_map[index].device;
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;
            switch (eventSet->events[i].type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter1, &counter_result));
                    HASEP_CHECK_CORE_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case FIXED:
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter1, &counter_result));
                    HASEP_CHECK_CORE_OVERFLOW(index+32);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED)
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case POWER:
                    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(POWER)))
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, counter1, (uint32_t*)&counter_result));
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, STOP_POWER)
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            eventSet->events[i].threadCounter[thread_id].overflows++;
                        }
                        eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    }
                    break;

                case THERMAL:
                    CHECK_TEMP_READ_ERROR(thermal_read(cpu_id,(uint32_t*)&counter_result));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case CBOX0:
                    HASEP_READ_BOX(CBOX0, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX1:
                    HASEP_READ_BOX(CBOX1, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX2:
                    HASEP_READ_BOX(CBOX2, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX3:
                    HASEP_READ_BOX(CBOX3, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX4:
                    HASEP_READ_BOX(CBOX4, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX5:
                    HASEP_READ_BOX(CBOX5, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX6:
                    HASEP_READ_BOX(CBOX6, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX7:
                    HASEP_READ_BOX(CBOX7, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX8:
                    HASEP_READ_BOX(CBOX8, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX9:
                    HASEP_READ_BOX(CBOX9, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX10:
                    HASEP_READ_BOX(CBOX10, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX11:
                    HASEP_READ_BOX(CBOX11, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX12:
                    HASEP_READ_BOX(CBOX12, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX13:
                    HASEP_READ_BOX(CBOX13, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX14:
                    HASEP_READ_BOX(CBOX14, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX15:
                    HASEP_READ_BOX(CBOX15, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX16:
                    HASEP_READ_BOX(CBOX16, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX17:
                    HASEP_READ_BOX(CBOX17, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case UBOX:
                    HASEP_READ_BOX(UBOX, counter1);
                    HASEP_CHECK_UNCORE_OVERFLOW(1);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case UBOXFIX:
                    HASEP_READ_BOX(UBOXFIX, counter1);
                    HASEP_CHECK_UNCORE_OVERFLOW(0);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case SBOX0:
                    HASEP_READ_BOX(SBOX0, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case SBOX1:
                    HASEP_READ_BOX(SBOX1, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case SBOX2:
                    HASEP_READ_BOX(SBOX2, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case SBOX3:
                    HASEP_READ_BOX(SBOX3, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case BBOX0:
                    HASEP_READ_PCI_BOX(BBOX0, counter1, counter2);
                    HASEP_CHECK_UNCORE_OVERFLOW(21);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case BBOX1:
                    HASEP_READ_PCI_BOX(BBOX1, counter1, counter2);
                    HASEP_CHECK_UNCORE_OVERFLOW(22);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
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

#define START_READ_MASK 0x00070007
#define STOP_READ_MASK ~(START_READ_MASK)

int perfmon_readCountersThread_haswellEP(int thread_id, PerfmonEventSet* eventSet)
{
    int bit;
    uint64_t tmp = 0x0ULL;
    uint64_t flags;
    int haveLock = 0;
    uint64_t counter_result = 0x0ULL;
    int read_fd;

    int cpu_id = groupSet->threads[thread_id].processorId;
    read_fd = socket_fd;
    if (socket_fd == -1)
    {
        read_fd = thread_sockets[cpu_id];
    }
    if (read_fd == -1)
    {
        return -ENOENT;
    }

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)))
    {
        CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, MSR_PERF_GLOBAL_CTRL, &flags));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, SAFE_PMC_FLAGS)
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, RESET_PMC_FLAGS)
    }
    HASEP_FREEZE_UNCORE;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[cpu_id].init == TRUE)
        {
            counter_result= 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            PciDeviceIndex dev = counter_map[index].device;
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;
            switch (eventSet->events[i].type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, counter1, &counter_result));
                    HASEP_CHECK_CORE_OVERFLOW(index-3);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
                    eventSet->events[i].threadCounter[cpu_id].counterData = counter_result;
                    break;

                case FIXED:
                    CHECK_MSR_READ_ERROR(msr_tread(read_fd, cpu_id, counter1, &counter_result));
                    HASEP_CHECK_CORE_OVERFLOW(index+32);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED)
                    eventSet->events[i].threadCounter[cpu_id].counterData = counter_result;
                    break;

                case POWER:
                    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(POWER)))
                    {
                        CHECK_POWER_READ_ERROR(power_tread(read_fd, cpu_id, counter1, (uint32_t*)&counter_result));
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, STOP_POWER)
                        if (counter_result < eventSet->events[i].threadCounter[cpu_id].counterData)
                        {
                            eventSet->events[i].threadCounter[cpu_id].overflows++;
                        }
                        eventSet->events[i].threadCounter[cpu_id].counterData = counter_result;
                    }
                    break;

                case THERMAL:
                    CHECK_TEMP_READ_ERROR(thermal_tread(read_fd, cpu_id,(uint32_t*)&counter_result));
                    eventSet->events[i].threadCounter[cpu_id].counterData = counter_result;
                    break;

                case CBOX0:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX0, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX1:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX1, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX2:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX2, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX3:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX3, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX4:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX4, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX5:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX5, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX6:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX6, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX7:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX7, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX8:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX8, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX9:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX9, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX10:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX10, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX11:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX11, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX12:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX12, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX13:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX13, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX14:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX14, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX15:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX15, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX16:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX16, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX17:
                    HASEP_READ_BOX_SOCKET(read_fd, CBOX17, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case UBOX:
                    HASEP_READ_BOX_SOCKET(read_fd, UBOX, counter1);
                    HASEP_CHECK_UNCORE_OVERFLOW(1);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case UBOXFIX:
                    HASEP_READ_BOX_SOCKET(read_fd, UBOXFIX, counter1);
                    HASEP_CHECK_UNCORE_OVERFLOW(0);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case SBOX0:
                    HASEP_READ_BOX_SOCKET(read_fd, SBOX0, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case SBOX1:
                    HASEP_READ_BOX_SOCKET(read_fd, SBOX1, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case SBOX2:
                    HASEP_READ_BOX_SOCKET(read_fd, SBOX2, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case SBOX3:
                    HASEP_READ_BOX_SOCKET(read_fd, SBOX3, counter1);
                    HASEP_CHECK_LOCAL_OVERFLOW;
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case BBOX0:
                    HASEP_READ_PCI_BOX_SOCKET(read_fd, BBOX0, counter1, counter2);
                    HASEP_CHECK_UNCORE_OVERFLOW(21);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case BBOX1:
                    HASEP_READ_PCI_BOX_SOCKET(read_fd, BBOX1, counter1, counter2);
                    HASEP_CHECK_UNCORE_OVERFLOW(22);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                default:
                    /* should never be reached */
                    break;
            }
        }
    }

    HASEP_UNFREEZE_UNCORE;
    if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, RESTORE_PMC_FLAGS)
        CHECK_MSR_WRITE_ERROR(msr_twrite(read_fd, cpu_id, MSR_PERF_GLOBAL_CTRL, flags));
    }

    return 0;
}
