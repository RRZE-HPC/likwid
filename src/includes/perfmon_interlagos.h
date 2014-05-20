/*
 * =======================================================================================
 *
 *      Filename:  perfmon_interlagos.h
 *
 *      Description:  AMD Interlagos specific subroutines
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2012 Jan Treibig 
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

#include <perfmon_interlagos_events.h>
#include <perfmon_interlagos_groups.h>

#define NUM_COUNTERS_INTERLAGOS 10

static int perfmon_numCountersInterlagos = NUM_COUNTERS_INTERLAGOS;
static int perfmon_numGroupsInterlagos = NUM_GROUPS_INTERLAGOS;
static int perfmon_numArchEventsInterlagos = NUM_ARCH_EVENTS_INTERLAGOS;

static PerfmonCounterMap interlagos_counter_map[NUM_COUNTERS_INTERLAGOS] = {
    {"PMC0",PMC0},
    {"PMC1",PMC1},
    {"PMC2",PMC2},
    {"PMC3",PMC3},
    {"PMC4",PMC4},
    {"PMC5",PMC5},
    {"UPMC0",PMC6},
    {"UPMC1",PMC7},
    {"UPMC2",PMC8},
    {"UPMC3",PMC9}
};

void perfmon_init_interlagos(PerfmonThread *thread)
{
    uint64_t flags = 0x0ULL;
    int cpu_id = thread->processorId;

    /* Core counters */
    thread->counters[PMC0].configRegister = MSR_AMD15_PERFEVTSEL0;
    thread->counters[PMC0].counterRegister = MSR_AMD15_PMC0;
    thread->counters[PMC0].type = PMC;
    thread->counters[PMC1].configRegister = MSR_AMD15_PERFEVTSEL1;
    thread->counters[PMC1].counterRegister = MSR_AMD15_PMC1;
    thread->counters[PMC1].type = PMC;
    thread->counters[PMC2].configRegister = MSR_AMD15_PERFEVTSEL2;
    thread->counters[PMC2].counterRegister = MSR_AMD15_PMC2;
    thread->counters[PMC2].type = PMC;
    thread->counters[PMC3].configRegister = MSR_AMD15_PERFEVTSEL3;
    thread->counters[PMC3].counterRegister = MSR_AMD15_PMC3;
    thread->counters[PMC3].type = PMC;
    thread->counters[PMC4].configRegister = MSR_AMD15_PERFEVTSEL4;
    thread->counters[PMC4].counterRegister = MSR_AMD15_PMC4;
    thread->counters[PMC4].type = PMC;
    thread->counters[PMC5].configRegister = MSR_AMD15_PERFEVTSEL5;
    thread->counters[PMC5].counterRegister = MSR_AMD15_PMC5;
    thread->counters[PMC5].type = PMC;

    /* Northbridge counters */
    thread->counters[PMC6].configRegister = MSR_AMD15_NB_PERFEVTSEL0;
    thread->counters[PMC6].counterRegister = MSR_AMD15_NB_PMC0;
    thread->counters[PMC6].type = UNCORE;
    thread->counters[PMC7].configRegister = MSR_AMD15_NB_PERFEVTSEL1;
    thread->counters[PMC7].counterRegister = MSR_AMD15_NB_PMC1;
    thread->counters[PMC7].type = UNCORE;
    thread->counters[PMC8].configRegister = MSR_AMD15_NB_PERFEVTSEL2;
    thread->counters[PMC8].counterRegister = MSR_AMD15_NB_PMC2;
    thread->counters[PMC8].type = UNCORE;
    thread->counters[PMC9].configRegister = MSR_AMD15_NB_PERFEVTSEL3;
    thread->counters[PMC9].counterRegister = MSR_AMD15_NB_PMC3;
    thread->counters[PMC9].type = UNCORE;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id) ||
            lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id))
    {
        msr_write(cpu_id, MSR_AMD15_PERFEVTSEL0, 0x0ULL);
        msr_write(cpu_id, MSR_AMD15_PERFEVTSEL1, 0x0ULL);
        msr_write(cpu_id, MSR_AMD15_PERFEVTSEL2, 0x0ULL);
        msr_write(cpu_id, MSR_AMD15_PERFEVTSEL3, 0x0ULL);
        msr_write(cpu_id, MSR_AMD15_PERFEVTSEL4, 0x0ULL);
        msr_write(cpu_id, MSR_AMD15_PERFEVTSEL5, 0x0ULL);
        msr_write(cpu_id, MSR_AMD15_NB_PERFEVTSEL0, 0x0ULL);
        msr_write(cpu_id, MSR_AMD15_NB_PERFEVTSEL1, 0x0ULL);
        msr_write(cpu_id, MSR_AMD15_NB_PERFEVTSEL2, 0x0ULL);
        msr_write(cpu_id, MSR_AMD15_NB_PERFEVTSEL3, 0x0ULL);
    }

#if 0
    flags |= (1<<16);  /* user mode flag */
    msr_write(cpu_id, MSR_AMD_PERFEVTSEL0, flags);
    msr_write(cpu_id, MSR_AMD_PERFEVTSEL1, flags);
    msr_write(cpu_id, MSR_AMD_PERFEVTSEL2, flags);
    msr_write(cpu_id, MSR_AMD_PERFEVTSEL3, flags);
#endif
}


void
perfmon_setupCounterThread_interlagos(int thread_id,
        PerfmonEvent* event,
        PerfmonCounterIndex index)
{
    uint64_t flags;
    uint64_t reg = perfmon_threadData[thread_id].counters[index].configRegister;
    int cpu_id = perfmon_threadData[thread_id].processorId;
    perfmon_threadData[thread_id].counters[index].init = TRUE;

    /* only one thread accesses Uncore */
    if ((perfmon_threadData[thread_id].counters[index].type == UNCORE) &&
            !(socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        return;
    }

    flags = msr_read(cpu_id,reg);
    flags &= ~(0xFFFFU); 

	/* AMD uses a 12 bit Event mask: [35:32][7:0] */
	flags |= ((uint64_t)(event->eventId>>8)<<32) + (event->umask<<8) + (event->eventId & ~(0xF00U));

    if (perfmon_verbose)
    {
        printf("[%d] perfmon_setup_counter: Write Register 0x%llX , Flags: 0x%llX \n",
               cpu_id,
               LLU_CAST reg,
               LLU_CAST flags);
    }
    msr_write(cpu_id, reg , flags);
}


void
perfmon_startCountersThread_interlagos(int thread_id)
{
    int i;
    int haveLock = 0;
    uint64_t flags;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    for ( i=0; i<NUM_COUNTERS_INTERLAGOS; i++ ) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            if (perfmon_threadData[thread_id].counters[i].type == PMC)
            {
                msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister , 0x0ULL);
                flags = msr_read(cpu_id,perfmon_threadData[thread_id].counters[i].configRegister);
                flags |= (1<<22);  /* enable flag */

                if (perfmon_verbose) 
                {
                    printf("perfmon_start_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                            LLU_CAST perfmon_threadData[thread_id].counters[i].configRegister,
                            LLU_CAST flags);
                }

                msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].configRegister , flags);

            }
            else if (perfmon_threadData[thread_id].counters[i].type == UNCORE)
            {
                if(haveLock)
                {
                    msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister , 0x0ULL);
                    flags = msr_read(cpu_id,perfmon_threadData[thread_id].counters[i].configRegister);
                    flags |= (1<<22);  /* enable flag */

                    if (perfmon_verbose) 
                    {
                        printf("perfmon_start_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                                LLU_CAST perfmon_threadData[thread_id].counters[i].configRegister,
                                LLU_CAST flags);
                    }

                    msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].configRegister , flags);
                }
            }
        }
    }
}

void 
perfmon_stopCountersThread_interlagos(int thread_id)
{
    int i;
    uint64_t flags;
    int haveLock = 0;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    for ( i=0; i<NUM_COUNTERS_INTERLAGOS; i++ ) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            if (perfmon_threadData[thread_id].counters[i].type == PMC)
            {
                flags = msr_read(cpu_id,perfmon_threadData[thread_id].counters[i].configRegister);
                flags &= ~(1<<22);  /* clear enable flag */
                msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].configRegister , flags);
                if (perfmon_verbose)
                {
                    printf("perfmon_stop_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                            LLU_CAST perfmon_threadData[thread_id].counters[i].configRegister,
                            LLU_CAST flags);
                }
                perfmon_threadData[thread_id].counters[i].counterData =
                    msr_read(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister);

            }
            else if (perfmon_threadData[thread_id].counters[i].type == UNCORE)
            {
                if(haveLock)
                {
                    flags = msr_read(cpu_id,perfmon_threadData[thread_id].counters[i].configRegister);
                    flags &= ~(1<<22);  /* clear enable flag */
                    msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].configRegister , flags);
                    if (perfmon_verbose)
                    {
                        printf("perfmon_stop_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                                LLU_CAST perfmon_threadData[thread_id].counters[i].configRegister,
                                LLU_CAST flags);
                    }
                    perfmon_threadData[thread_id].counters[i].counterData =
                        msr_read(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister);

                }
            }
        }
    }
}


void 
perfmon_readCountersThread_interlagos(int thread_id)
{
    int haveLock = 0;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }


    for (int i=0;i<NUM_COUNTERS_INTERLAGOS;i++) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        { 
            if (perfmon_threadData[thread_id].counters[i].type == UNCORE)
            {
                if(haveLock)
                {

                    perfmon_threadData[thread_id].counters[i].counterData =
                        msr_read(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister);
                }
            }
            else
            {
                perfmon_threadData[thread_id].counters[i].counterData =
                    msr_read(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister);
            }
        }
    }
}


