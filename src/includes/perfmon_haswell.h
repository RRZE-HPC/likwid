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

#include <perfmon_haswell_events.h>
#include <perfmon_haswell_counters.h>
#include <error.h>
#include <affinity.h>
#include <limits.h>
#include <topology.h>


static int perfmon_numCountersHaswell = NUM_COUNTERS_HASWELL;
static int perfmon_numCoreCountersHaswell = NUM_COUNTERS_CORE_HASWELL;
static int perfmon_numArchEventsHaswell = NUM_ARCH_EVENTS_HASWELL;
static int model_has_uncore = 0;

#define OFFSET_PMC 3


int perfmon_init_haswell(int cpu_id)
{
    uint64_t flags = 0x0ULL;
    lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    return 0;
    int ret;

    /* Initialize registers */
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL0, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL1, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL2, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL3, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PMC0, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PMC1, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PMC2, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PMC3, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_FIXED_CTR0, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_FIXED_CTR1, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_FIXED_CTR2, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PEBS_ENABLE, 0x0ULL));

    /* initialize fixed counters
     * FIXED 0: Instructions retired
     * FIXED 1: Clocks unhalted core
     * FIXED 2: Clocks unhalted ref */
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x222ULL));

    /* Preinit of PERFEVSEL registers */
    flags |= (1<<22);  /* enable flag */
    flags |= (1<<16);  /* user mode flag */

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL0, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL1, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL2, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL3, flags));


    lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    
    return 0;
}

#define HAS_FREEZE_UNCORE \
    if (haveLock && eventSet->regTypeMask & ~(0xFULL)) \
    { \
        VERBOSEPRINTREG(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<31), FREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, (1ULL<<31))); \
    }

#define HAS_UNFREEZE_UNCORE \
    if (haveLock && eventSet->regTypeMask & ~(0xFULL)) \
    { \
        VERBOSEPRINTREG(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<29), UNFREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, (1ULL<<29))); \
    }

#define HAS_UNFREEZE_UNCORE_AND_RESET_CTR \
    if (haveLock && eventSet->regTypeMask & ~(0xFULL)) \
    { \
        for (int j=0; j<NUM_COUNTERS_HASWELL; j++) \
        { \
            if (eventSet->regTypeMask & REG_TYPE_MASK(counter_map[j].type)) \
            { \
                if (counter_map[j].counterRegister != 0x0) \
                { \
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, counter_map[j].counterRegister, 0x0ULL)); \
                } \
                if (counter_map[j].counterRegister2 != 0x0) \
                { \
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, counter_map[j].counterRegister2, 0x0ULL)); \
                } \
            } \
        } \
        VERBOSEPRINTREG(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<29), UNFREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, (1ULL<<29))); \
    }

#define HAS_FREEZE_UNCORE_AND_RESET_CTL \
    if (haveLock && eventSet->regTypeMask & ~(0xF)) \
    { \
        VERBOSEPRINTREG(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<31), FREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, (1ULL<<31))); \
        for (int j=0; j<NUM_COUNTERS_HASWELL; j++) \
        { \
            if (eventSet->regTypeMask & REG_TYPE_MASK(counter_map[j].type)) \
            { \
                if (counter_map[j].configRegister != 0x0) \
                { \
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, counter_map[j].configRegister, 0x0ULL)); \
                } \
            } \
        } \
    }

#define HAS_SETUP_BOX(id) \
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
                    default: \
                        break; \
                } \
            } \
        } \
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_##id); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, flags)); \
    }

int perfmon_setupCounterThread_haswell(
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
    HAS_FREEZE_UNCORE;
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = counter_map[index].configRegister;
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
                HAS_SETUP_BOX(CBOX0);
                break;
            case CBOX1:
                HAS_SETUP_BOX(CBOX1);
                break;
            case CBOX2:
                HAS_SETUP_BOX(CBOX2);
                break;
            case CBOX3:
                HAS_SETUP_BOX(CBOX3);
                break;

            case UBOX:
                HAS_SETUP_BOX(UBOX);

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

int perfmon_startCountersThread_haswell(int thread_id, PerfmonEventSet* eventSet)
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
                    flags |= (1ULL<<(index-OFFSET_PMC));  /* enable counter */
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

    HAS_UNFREEZE_UNCORE_AND_RESET_CTR;

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, UNFREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, flags));
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x30000000FULL));
    }

    return 0;
}

#define HAS_CHECK_OVERFLOW(offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t ovf_values = 0x0ULL; \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_PERF_GLOBAL_STATUS, &ovf_values)); \
        if (ovf_values & (1ULL<<offset)) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
        } \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<offset))); \
    }

#define HAS_READ_BOX(id, reg1) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) \
    { \
        VERBOSEPRINTREG(cpu_id, reg1, LLU_CAST counter_result, READ_BOX_##id) \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, reg1, &counter_result)); \
    }

int perfmon_stopCountersThread_haswell(int thread_id, PerfmonEventSet* eventSet)
{
    int bit;
    int haveLock = 0;
    uint64_t flags;
    uint32_t uflags = 0x10100UL; /* Set freeze bit */
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    HAS_FREEZE_UNCORE_AND_RESET_CTL;


    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            counter_result= 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;
            switch (eventSet->events[i].type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter1, &counter_result));
                    HAS_CHECK_OVERFLOW(index-OFFSET_PMC);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case FIXED:
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter1, &counter_result));
                    HAS_CHECK_OVERFLOW(index+32);
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
                    HAS_READ_BOX(CBOX0, counter1);
                    HAS_CHECK_OVERFLOW(61);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX1:
                    HAS_READ_BOX(CBOX1, counter1);
                    HAS_CHECK_OVERFLOW(61);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX2:
                    HAS_READ_BOX(CBOX2, counter1);
                    HAS_CHECK_OVERFLOW(61);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX3:
                    HAS_READ_BOX(CBOX3, counter1);
                    HAS_CHECK_OVERFLOW(61);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case UBOX:
                    HAS_READ_BOX(UBOX, counter1);
                    HAS_CHECK_OVERFLOW(61);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                default:
                    /* should never be reached */
                    break;
            }
        }
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
    }

    //CHECK_MSR_READ_ERROR(msr_read(cpu_id,MSR_PERF_GLOBAL_STATUS, &flags));

    //    printf ("Status: 0x%llX \n", LLU_CAST flags);
    /*if ( (flags & 0x3) || (flags & (0x3ULL<<32)) ) 
    {
        printf ("Overflow occured \n");
    }*/
    return 0;
}

#define START_READ_MASK 0x00070007
#define STOP_READ_MASK ~(START_READ_MASK)

int perfmon_readCountersThread_haswell(int thread_id, PerfmonEventSet* eventSet)
{
    int bit;
    uint64_t tmp = 0x0ULL;
    uint64_t flags;
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }
    
    CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_PERF_GLOBAL_CTRL, &flags));
    flags &= STOP_READ_MASK;
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, flags));

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            
            if (haswell_counter_map[index].type == PMC)
            {
                CHECK_MSR_READ_ERROR(msr_read(cpu_id, haswell_counter_map[index].counterRegister, &tmp));
                
                eventSet->events[i].threadCounter[thread_id].counterData = tmp;
                
                if (eventSet->events[i].threadCounter[thread_id].startData == 0)
                {
                    eventSet->events[i].threadCounter[thread_id].startData = tmp;
                }
                
                bit = haswell_counter_map[index].index - cpuid_info.perf_num_fixed_ctr;
                CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_PERF_GLOBAL_STATUS, &tmp))
                if (extractBitField(tmp, 1, bit))
                {
                    fprintf(stderr,"Overflow occured for PMC%d\n",bit);
                    eventSet->events[i].threadCounter[thread_id].overflows++;
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, &flags))
                    flags |= (1<<bit);
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, flags));
                }
            }
            else if (haswell_counter_map[index].type == FIXED)
            {
                CHECK_MSR_READ_ERROR(msr_read(cpu_id, haswell_counter_map[index].counterRegister, &tmp));
                
                eventSet->events[i].threadCounter[thread_id].counterData = tmp;
                
                if (eventSet->events[i].threadCounter[thread_id].startData == 0)
                {
                    eventSet->events[i].threadCounter[thread_id].startData = tmp;
                }
                
                bit = 32 + haswell_counter_map[index].index;
                CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_PERF_GLOBAL_STATUS, &tmp))
                if (extractBitField(tmp, 1, bit))
                {
                    fprintf(stderr,"Overflow occured for FIXC%d\n",bit-32);
                    eventSet->events[i].threadCounter[thread_id].overflows++;
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, &flags))
                    flags |= (1<<bit);
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, flags));
                }
            }
            else if ((haswell_counter_map[index].type == POWER) && haveLock == 1 )
            {
                CHECK_POWER_READ_ERROR(power_read(cpu_id, 
                    haswell_counter_map[index].counterRegister,(uint32_t*) &tmp));
                if (eventSet->events[i].threadCounter[thread_id].startData = 0)
                {
                    eventSet->events[i].threadCounter[thread_id].startData = tmp-1;
                }
                
                eventSet->events[i].threadCounter[thread_id].counterData = tmp;
                
                if (tmp <= eventSet->events[i].threadCounter[thread_id].startData)
                {
                    fprintf(stderr,"Overflow in power status register 0x%x, assuming single overflow\n",
                                    haswell_counter_map[index].counterRegister);
                    eventSet->events[i].threadCounter[thread_id].overflows++;  
                }
            }
        }
    }
    
    CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_PERF_GLOBAL_CTRL, &flags));
    flags |= START_READ_MASK;
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, flags));
    
    return 0;
}
