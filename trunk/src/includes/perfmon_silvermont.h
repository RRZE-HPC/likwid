/*
 * =======================================================================================
 *
 *      Filename:  perfmon_silvermont.h
 *
 *      Description:  Header file of perfmon module for Intel Atom Silvermont
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
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
 
#include <perfmon_silvermont_events.h>
#include <perfmon_silvermont_groups.h>
#include <perfmon_silvermont_counters.h>

static int perfmon_numCountersSilvermont = NUM_COUNTERS_SILVERMONT;
static int perfmon_numGroupsSilvermont = NUM_GROUPS_SILVERMONT;
static int perfmon_numArchEventsSilvermont = NUM_ARCH_EVENTS_SILVERMONT;


int perfmon_init_silvermont(int cpu_id)
{
    uint64_t flags = 0x0ULL;

    /* Initialize registers */
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL0, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL1, 0x0ULL));

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x0ULL));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PEBS_ENABLE, 0x0ULL));

    /* initialize fixed counters
     * FIXED 0: Instructions retired
     * FIXED 1: Clocks unhalted core
     * FIXED 2: Clocks unhalted ref */
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x222ULL));

    /* Preinit of PMC counters */
    flags |= (1<<16);  /* user mode flag */
    flags |= (1<<22);  /* enable flag */

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL0, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERFEVTSEL1, flags));
    return 0;
}

int perfmon_setupCountersThread_silvermont(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags;
    uint32_t uflags;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        PerfmonCounterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = silvermont_counter_map[index].configRegister;
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        switch (silvermont_counter_map[index].type)
        {
            case PMC:

                CHECK_MSR_READ_ERROR(msr_read(cpu_id, reg, &flags));
                flags &= ~(0xFFFFU);   /* clear lower 16bits */

                /* Intel with standard 8 bit event mask: [7:0] */
                flags |= (event->umask<<8) + event->eventId;

                

                /*if (perfmon_verbose)
                {
                    printf("[%d] perfmon_setup_counter PMC: Write Register 0x%llX , Flags: 0x%llX \n",
                            cpu_id,
                            LLU_CAST reg,
                            LLU_CAST flags);
                }*/

                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags));
                
                // Offcore event with additional configuration register
                // We included the additional register as counterRegister2
                // to avoid creating a new data structure
                // cfgBits contain offset of "request type" bit
                // cmask contain offset of "response type" bit
                if (event->eventId == 0xB7) 
                {
                    reg = silvermont_counter_map[index].counterRegister2;
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, reg, &flags));
                    flags = (1<<event->cfgBits)|(1<<event->cmask);
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags));
                }
                
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


int perfmon_startCountersThread_silvermont(int thread_id, PerfmonEventSet* eventSet)
{
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
            PerfmonCounterIndex index = eventSet->events[i].index;
            switch (silvermont_counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, silvermont_counter_map[index].counterRegister, 0x0ULL));
                    flags |= (1<<(i-OFFSET_PMC));  /* enable counter */
                    break;

                case FIXED:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, silvermont_counter_map[index].counterRegister, 0x0ULL));
                    flags |= (1ULL<<(i+32));  /* enable fixed counter */
                    break;

                case POWER:
                    if(haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, silvermont_counter_map[index].counterRegister,
                                (uint32_t*)eventSet->events[i].threadCounter[thread_id].counterData));
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


int perfmon_stopCountersThread_silvermont(int thread_id, PerfmonEventSet* eventSet)
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

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE) 
        {
            PerfmonCounterIndex index = eventSet->events[i].index;
            switch (silvermont_counter_map[index].type)
            {
                case PMC:

                case FIXED:
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, silvermont_counter_map[index].counterRegister, &counter_result));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case POWER:
                    if(haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, silvermont_counter_map[index].counterRegister, 
                                (uint32_t*)&counter_result));
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            fprintf(stderr,"Overflow in power status register 0x%x, assuming single overflow\n",
                                                silvermont_counter_map[index].counterRegister);
                            counter_result += (UINT_MAX - eventSet->events[i].threadCounter[thread_id].counterData);
                            eventSet->events[i].threadCounter[thread_id].counterData = power_info.energyUnit * counter_result;
                        }
                        else
                        {
                        eventSet->events[i].threadCounter[thread_id].counterData =
                            power_info.energyUnit *
                            ( counter_result - eventSet->events[i].threadCounter[thread_id].counterData);
                        }
                    }
                    break;

                case THERMAL:
                        CHECK_TEMP_READ_ERROR(thermal_read(cpu_id, (uint32_t*)&counter_result));
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

int perfmon_readCountersThread_silvermont(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t counter_result = 0x0ULL;
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            PerfmonCounterIndex index = eventSet->events[i].index;
            if ((silvermont_counter_map[index].type == PMC) ||
                    (silvermont_counter_map[index].type == FIXED))
            {
                CHECK_MSR_READ_ERROR(msr_read(cpu_id, silvermont_counter_map[index].counterRegister, &counter_result));
                eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
            }
            else
            {
                if(haveLock)
                {
                    switch (silvermont_counter_map[index].type)
                    {
                        case POWER:
                            CHECK_POWER_READ_ERROR(power_read(cpu_id, silvermont_counter_map[index].counterRegister, 
                                    (uint32_t*)&counter_result))
                            eventSet->events[i].threadCounter[thread_id].counterData =
                                power_info.energyUnit * counter_result;
                                ;
                            break;

                        default:
                            /* should never be reached */
                            break;
                    }
                }
            }
        }
    }
    return 0;
}

