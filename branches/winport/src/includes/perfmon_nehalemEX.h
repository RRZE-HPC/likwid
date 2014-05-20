/*
 * ===========================================================================
 *
 *      Filename:  perfmon_nehalem.h
 *
 *      Description:  Header File of perfmon module for Nehalem EX.
 *
 *      Version:  <VERSION>
 *      Created:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Company:  RRZE Erlangen
 *      Project:  likwid
 *      Copyright:  Copyright (c) 2010, Jan Treibig
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License, v2, as
 *      published by the Free Software Foundation
 *     
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *     
 *      You should have received a copy of the GNU General Public License
 *      along with this program; if not, write to the Free Software
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * ===========================================================================
 */
#include <stdlib.h>
#include <stdio.h>

#include <cpuid.h>
#include <tree.h>
#include <bstrlib.h>
#include <types.h>
#include <registers.h>
#include <perfmon_nehalemEX_events.h>
#include <perfmon_nehalemEX_groups.h>

#define NUM_COUNTERS_NEHALEMEX 7

static int perfmon_numCountersNehalemEX = NUM_COUNTERS_NEHALEMEX;
static int perfmon_numGroupsNehalemEX = NUM_GROUPS_NEHALEMEX;
static int perfmon_numArchEventsNehalemEX = NUM_ARCH_EVENTS_NEHALEMEX;

//static volatile int nehalem_socket_lock[MAX_NUM_SOCKETS];
//static int nehalem_processor_lookup[MAX_NUM_THREADS];

static PerfmonCounterMap nehalemEX_counter_map[NUM_COUNTERS_NEHALEM] = {
    {"FIXC0",PMC0},
    {"FIXC1",PMC1},
    {"FIXC2",PMC2},
    {"PMC0",PMC3},
    {"PMC1",PMC4},
    {"PMC2",PMC5},
    {"PMC3",PMC6}
};

#define OFFSET_PMC 3


void
perfmon_init_nehalemEX(PerfmonThread *thread)
{
    uint64_t flags = 0x0ULL;
    int cpu_id = thread->processorId;

#if 0
    TreeNode* socketNode;
    TreeNode* coreNode;
    TreeNode* threadNode;
    int socketId;

    for(int i=0; i<MAX_NUM_SOCKETS; i++) nehalem_socket_lock[i] = 0;

    socketNode = tree_getChildNode(cpuid_topology.topologyTree);
    while (socketNode != NULL)
    {
        socketId = socketNode->id;
        coreNode = tree_getChildNode(socketNode);

        while (coreNode != NULL)
        {
            threadNode = tree_getChildNode(coreNode);

            while (threadNode != NULL)
            {
                nehalem_processor_lookup[threadNode->id] = socketId;
                threadNode = tree_getNextNode(threadNode);
            }
            coreNode = tree_getNextNode(coreNode);
        }
        socketNode = tree_getNextNode(socketNode);
    }
#endif

    /* Fixed Counters: instructions retired, cycles unhalted core */
    thread->counters[PMC0].configRegister = MSR_PERF_FIXED_CTR_CTRL;
    thread->counters[PMC0].counterRegister = MSR_PERF_FIXED_CTR0;
    thread->counters[PMC0].type = FIXED;
    thread->counters[PMC1].configRegister = MSR_PERF_FIXED_CTR_CTRL;
    thread->counters[PMC1].counterRegister = MSR_PERF_FIXED_CTR1;
    thread->counters[PMC1].type = FIXED;
    thread->counters[PMC2].configRegister = MSR_PERF_FIXED_CTR_CTRL;
    thread->counters[PMC2].counterRegister = MSR_PERF_FIXED_CTR2;
    thread->counters[PMC2].type = FIXED;

    /* PMC Counters: 4 48bit wide */
    thread->counters[PMC3].configRegister = MSR_PERFEVTSEL0;
    thread->counters[PMC3].counterRegister = MSR_PMC0;
    thread->counters[PMC3].type = PMC;
    thread->counters[PMC4].configRegister = MSR_PERFEVTSEL1;
    thread->counters[PMC4].counterRegister = MSR_PMC1;
    thread->counters[PMC4].type = PMC;
    thread->counters[PMC5].configRegister = MSR_PERFEVTSEL2;
    thread->counters[PMC5].counterRegister = MSR_PMC2;
    thread->counters[PMC5].type = PMC;
    thread->counters[PMC6].configRegister = MSR_PERFEVTSEL3;
    thread->counters[PMC6].counterRegister = MSR_PMC3;
    thread->counters[PMC6].type = PMC;

    msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x0ULL);
    msr_write(cpu_id, MSR_PERFEVTSEL0, 0x0ULL);
    msr_write(cpu_id, MSR_PERFEVTSEL1, 0x0ULL);
    msr_write(cpu_id, MSR_PERFEVTSEL2, 0x0ULL);
    msr_write(cpu_id, MSR_PERFEVTSEL3, 0x0ULL);
    msr_write(cpu_id, MSR_PMC0, 0x0ULL);
    msr_write(cpu_id, MSR_PMC1, 0x0ULL);
    msr_write(cpu_id, MSR_PMC2, 0x0ULL);
    msr_write(cpu_id, MSR_PMC3, 0x0ULL);
    msr_write(cpu_id, MSR_PERF_FIXED_CTR0, 0x0ULL);
    msr_write(cpu_id, MSR_PERF_FIXED_CTR1, 0x0ULL);
    msr_write(cpu_id, MSR_PERF_FIXED_CTR2, 0x0ULL);
    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);
    msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x0ULL);
    msr_write(cpu_id, MSR_PEBS_ENABLE, 0x0ULL);

    /* initialize fixed counters
     * FIXED 0: Instructions retired
     * FIXED 1: Clocks unhalted core
     * FIXED 2: Clocks unhalted ref */
    msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x222ULL);
    /* UNCORE FIXED 0: Uncore cycles */
    msr_write(cpu_id, MSR_UNCORE_FIXED_CTR_CTRL, 0x01ULL);

    /* Preinit of PERFEVSEL registers */
    flags |= (1<<22);  /* enable flag */
    flags |= (1<<16);  /* user mode flag */
#if 0
    flags |= (1<<19);  /* pin control flag */
#endif

    msr_write(cpu_id, MSR_PERFEVTSEL0, flags);
    msr_write(cpu_id, MSR_PERFEVTSEL1, flags);
    msr_write(cpu_id, MSR_PERFEVTSEL2, flags);
    msr_write(cpu_id, MSR_PERFEVTSEL3, flags);
}


void
perfmon_setupCounterThread_nehalemEX(int thread_id,
        uint32_t event, uint32_t umask,
        PerfmonCounterIndex index)
{
    uint64_t flags;
    uint64_t reg = perfmon_threadData[thread_id].counters[index].configRegister;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if ((perfmon_threadData[thread_id].counters[index].type == PMC) || perfmon_threadData[thread_id].counters[index].type == UNCORE)
    {

        perfmon_threadData[thread_id].counters[index].init = TRUE;
        flags = msr_read(cpu_id,reg);
        flags &= ~(0xFFFFU); 

        /* Intel with standard 8 bit event mask: [7:0] */
        flags |= (umask<<8) + event;

        msr_write(cpu_id, reg , flags);

        if (perfmon_verbose)
        {
            printf("[%d] perfmon_setup_counter: Write Register 0x%llX , Flags: 0x%llX \n",
                    cpu_id,
                    LLU_CAST reg,
                    LLU_CAST flags);
        }
    }
    else if (perfmon_threadData[thread_id].counters[index].type == FIXED)
    {
        perfmon_threadData[thread_id].counters[index].init = TRUE;
    }
}


void
perfmon_startCountersThread_nehalemEX(int thread_id)
{
    int i;
//    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t uflags = 0x0ULL;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);

#if 0
    if (!nehalem_socket_lock[nehalem_processor_lookup[cpu_id]])
    {
        nehalem_socket_lock[nehalem_processor_lookup[cpu_id]] = 1;
        haveLock = 1;
        msr_write(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL);
        /* Fixed Uncore counter */
        uflags = 0x100000000ULL;
    }
#endif

    for (i=0;i<NUM_PMC;i++) {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) {
            if (perfmon_threadData[thread_id].counters[i].type == PMC)
            {
                msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister , 0x0ULL);
                flags |= (1<<(i-OFFSET_PMC));  /* enable counter */
            }
            else if (perfmon_threadData[thread_id].counters[i].type == FIXED)
            {
                msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister , 0x0ULL);
                flags |= (1ULL<<(i+32));  /* enable fixed counter */
            }
            else if (perfmon_threadData[thread_id].counters[i].type == UNCORE)
            {
                //               if(haveLock)
                //              {
                //                 msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister , 0x0ULL);
                //                uflags |= (1<<(i-OFFSET_UPMC));  /* enable uncore counter */
                //           }
            }
        }
    }

    if (perfmon_verbose)
    {
        printf("perfmon_start_counters: Write Register 0x%X , Flags: 0x%llX \n",MSR_PERF_GLOBAL_CTRL, LLU_CAST flags);
        printf("perfmon_start_counters: Write Register 0x%X , Flags: 0x%llX \n",MSR_UNCORE_PERF_GLOBAL_CTRL, LLU_CAST uflags);
    }

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, flags);
//    if (haveLock) msr_write(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, uflags);
    msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x30000000FULL);

}

void 
perfmon_stopCountersThread_nehalemEX(int thread_id)
{
    uint64_t flags;
//    int haveLock = 0;
    int i;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);

#if 0
    if (nehalem_socket_lock[nehalem_processor_lookup[cpu_id]])
    {
        nehalem_socket_lock[nehalem_processor_lookup[cpu_id]] = 0;
        haveLock = 1;
        msr_write(cpu_id, MSR_UNCORE_PERF_GLOBAL_CTRL, 0x0ULL);
    }
#endif

    for (i=0; i<NUM_COUNTERS_NEHALEM; i++) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            if (perfmon_threadData[thread_id].counters[i].type == UNCORE)
            {
//                if(haveLock)
//                {
//                    perfmon_threadData[thread_id].counters[i].counterData = msr_read(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister);
//                }
            }
            else
            {
                perfmon_threadData[thread_id].counters[i].counterData = msr_read(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister);
            }
        }
    }

    flags = msr_read(cpu_id,MSR_PERF_GLOBAL_STATUS);
    printf ("Status: 0x%llX \n", LLU_CAST flags);
    if((flags & 0x3) || (flags & (0x3ULL<<32)) ) 
    {
        printf ("Overflow occured \n");
    }

}



void 
perfmon_readCountersThread_nehalemEX(int thread_id)
{
//    int haveLock = 0;
    int i;
    int cpu_id = perfmon_threadData[thread_id].processorId;

#if 0
    if (nehalem_socket_lock[nehalem_processor_lookup[cpu_id]])
    {
        nehalem_socket_lock[nehalem_processor_lookup[cpu_id]] = 0;
        haveLock = 1;
    }
#endif

    for (i=0; i<NUM_COUNTERS_NEHALEM; i++) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            if (perfmon_threadData[thread_id].counters[i].type == UNCORE)
            {
//                if(haveLock)
//                {
//                    perfmon_threadData[thread_id].counters[i].counterData =
//                        msr_read(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister);
//                }
            }
            else
            {
                perfmon_threadData[thread_id].counters[i].counterData =
                    msr_read(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister);
            }
        }
    }
}

