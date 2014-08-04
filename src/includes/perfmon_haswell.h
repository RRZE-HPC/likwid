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
static int perfmon_numArchEventsHaswell = NUM_ARCH_EVENTS_HASWELL;


#define OFFSET_PMC 3


int perfmon_init_haswell(int cpu_id)
{
    uint64_t flags = 0x0ULL;
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

int perfmon_setupCounterThread_haswell(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int ret;
    uint64_t flags;
    uint32_t uflags;
    int cpu_id = groupSet->threads[thread_id].processorId;
    
    /*if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }*/
    
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = haswell_counter_map[index].configRegister;
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        switch (haswell_counter_map[index].type)
        {
            case PMC:

                CHECK_MSR_READ_ERROR(msr_read(cpu_id, reg, &flags));

                flags &= ~(0xFFFFU);   /* clear lower 16bits */

                /* Intel with standard 8 bit event mask: [7:0] */
                flags |= (event->umask<<8) + event->eventId;

                if (event->cfgBits != 0) /* set custom cfg and cmask */
                {
                    flags &= ~(0xFFFFU<<16);  /* clear upper 16bits */
                    flags |= ((event->cmask<<8) + event->cfgBits)<<16;
                }

                /*if (perfmon_verbose)
                {
                    printf("[%d] perfmon_setup_counter PMC: Write Register 0x%llX , Flags: 0x%llX \n",
                            cpu_id,
                            LLU_CAST reg,
                            LLU_CAST flags);
                }*/

                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags));
                break;

            case FIXED:
                break;

            case POWER:
                break;

            default:
                /* should never be reached */
                break;
        }
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

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));


    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            switch (haswell_counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, haswell_counter_map[index].counterRegister, 0x0ULL));
                    flags |= (1<<(index-OFFSET_PMC));  /* enable counter */
                    break;

                case FIXED:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, haswell_counter_map[index].counterRegister, 0x0ULL));
                    flags |= (1ULL<<(index+32));  /* enable fixed counter */
                    break;

                case POWER:
                    if(haveLock == 1)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, haswell_counter_map[index].counterRegister,
                                        (uint32_t*)&eventSet->events[i].threadCounter[thread_id].startData));
                    }
                    break;

                default:
                    /* should never be reached */
                    break;
            }
        }
    }

    /*if (perfmon_verbose)
    {
        printf("perfmon_start_counters: Write Register 0x%X , \
                Flags: 0x%llX \n",MSR_PERF_GLOBAL_CTRL, LLU_CAST flags);
        printf("perfmon_start_counters: Write Register 0x%X , \
                Flags: 0x%llX \n",MSR_UNCORE_PERF_GLOBAL_CTRL, LLU_CAST uflags);
    }*/

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x30000000FULL));

    return 0;
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

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));


    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            switch (haswell_counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, haswell_counter_map[index].counterRegister,
                                    (uint64_t*)&counter_result));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    
                    
                    bit = haswell_counter_map[index].index - cpuid_info.perf_num_fixed_ctr;
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_PERF_GLOBAL_STATUS, &flags))
                    if (extractBitField(flags, 1, bit))
                    {
                        fprintf(stderr,"Overflow occured for PMC%d\n",bit);
                        eventSet->events[i].threadCounter[thread_id].overflows++;
                        CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, &flags))
                        flags |= (1<<bit);
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, flags));
                    }
                case FIXED:
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, haswell_counter_map[index].counterRegister,
                                    (uint64_t*)&counter_result));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    
                    
                    bit = 32 + haswell_counter_map[index].index;
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_PERF_GLOBAL_STATUS, &flags))
                    if (extractBitField(flags, 1, bit))
                    {
                        fprintf(stderr,"Overflow occured for FIXC%d\n",bit-32);
                        eventSet->events[i].threadCounter[thread_id].overflows++;
                        CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, &flags))
                        flags |= (1<<bit);
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, flags));
                    }
                    break;

                case POWER:
                    if (haveLock == 1)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, haswell_counter_map[index].counterRegister, (uint32_t*)&counter_result));
                  
                        if (eventSet->events[i].threadCounter[thread_id].startData = 0)
                        {
                            eventSet->events[i].threadCounter[thread_id].startData = counter_result-1;
                        }
                        
                        eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                
                        if (counter_result <= eventSet->events[i].threadCounter[thread_id].startData)
                        {
                            fprintf(stderr,"Overflow in power status register 0x%x, assuming single overflow\n",
                                            haswell_counter_map[index].counterRegister);
                            eventSet->events[i].threadCounter[thread_id].overflows++;  
                        }
                    }
                    break;

                case THERMAL:
                        CHECK_TEMP_READ_ERROR(thermal_read(cpu_id,(uint32_t*)&counter_result));
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
