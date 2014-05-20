/*
 * =======================================================================================
 *
 *      Filename:  perfmon_nehalemEX.h
 *
 *      Description:  Header File of perfmon module for Nehalem EX.
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

#include <perfmon_nehalemEX_events.h>
#include <perfmon_nehalemEX_groups.h>

#define NUM_COUNTERS_NEHALEMEX 7

//static int perfmon_numCountersNehalemEX = NUM_COUNTERS_NEHALEMEX;
static int perfmon_numGroupsNehalemEX = NUM_GROUPS_NEHALEMEX;
static int perfmon_numArchEventsNehalemEX = NUM_ARCH_EVENTS_NEHALEMEX;

/* This SUCKS: There are only subtle difference between NehalemEX
 * and Westmere EX Uncore. Still one of them is that one field is 
 * 1 bit shifted. Thank you Intel for this mess!!! Do you want 
 * to change the register definitions for every architecture?*/

/* MBOX macros */

#define MBOX_GATE_NEHEX(NUM)  \
flags = 0x41ULL; \
switch (event->cfgBits)  \
{  \
    case 0x00:   /* primary Event */  \
        flags |= (event->eventId<<9);  \
        break;  \
    case 0x01: /* secondary Events */  \
        /* TODO fvid index is missing defaults to 0 */   \
        flags |= (1<<7); /* toggle flag mode */   \
        flags |= (event->eventId<<19);   \
        switch (event->eventId)   \
        {   \
            case 0x00: /* CYCLES_DSP_FILL: DSP */   \
                {   \
                    uint64_t dsp_flags = 0x0ULL;   \
                    dsp_flags |= (event->umask<<7);  \
                    msr_write(cpu_id, MSR_M##NUM##_PMON_DSP, dsp_flags);   \
                }   \
                break;   \
            case 0x01: /* CYCLES_SCHED_MODE: ISS */   \
                {   \
                    uint32_t iss_flags = 0x0UL;   \
                    iss_flags |= (event->umask<<4);   \
                    msr_write(cpu_id, MSR_M##NUM##_PMON_ISS, iss_flags);   \
                }    \
                break;   \
            case 0x05: /* CYCLES_PGT_STATE: PGT */   \
                {   \
                    uint32_t pgt_flags = 0x0UL;   \
                    pgt_flags |= (event->umask<<6);   \
                    msr_write(cpu_id, MSR_M##NUM##_PMON_PGT, pgt_flags);   \
                }    \
                break;   \
            case 0x06: /* BCMD_SCHEDQ_OCCUPANCY: MAP */   \
                {   \
                    uint32_t map_flags = 0x0UL;   \
                    map_flags |= (event->umask<<6);   \
                    msr_write(cpu_id, MSR_M##NUM##_PMON_MAP, map_flags);   \
                }   \
                break;   \
        }    \
    break;   \
    case 0x02: /* DRAM_CMD: PLD/ISS */   \
        flags |= (event->eventId<<9);   \
        {   \
            uint32_t pld_flags = 0x0UL;   \
            uint32_t iss_flags = 0x0UL;   \
            pld_flags |= (event->umask<<8);   \
            if (event->cmask != 0)   \
            {   \
                iss_flags |= (event->cmask<<7);   \
                pld_flags |= 1; /* toggle cmd flag */   \
            }   \
            msr_write(cpu_id, MSR_M##NUM##_PMON_PLD, pld_flags);   \
            msr_write(cpu_id, MSR_M##NUM##_PMON_ISS, iss_flags);   \
        }   \
        break;   \
    case 0x03: /* DSP_FILL: DSP */   \
        flags |= (event->eventId<<9);   \
        {   \
            uint64_t dsp_flags = 0x0ULL;   \
            dsp_flags |= (event->umask<<7);   \
            msr_write(cpu_id, MSR_M##NUM##_PMON_DSP, dsp_flags);   \
        }   \
        break;   \
    case 0x04: /* DRAM_MISC: PLD */   \
        flags |= (event->eventId<<9);   \
        {   \
            uint64_t pld_flags = 0x0ULL;   \
            switch (event->cmask)   \
            {   \
                case 0x0:   \
                    pld_flags |= (1<<16);   \
                    pld_flags |= (event->umask<<19);   \
                    break;   \
                case 0x1:   \
                    pld_flags |= (event->umask<<18);   \
                    break;   \
                case 0x2:   \
                    pld_flags |= (event->umask<<17);   \
                    break;   \
                case 0x3:   \
                    pld_flags |= (event->umask<<7);   \
                    break;   \
            }   \
            msr_write(cpu_id, MSR_M##NUM##_PMON_PLD, pld_flags);   \
        }   \
        break;   \
    case 0x05: /* FRM_TYPE: ISS */   \
        flags |= (event->eventId<<9);   \
        {   \
            uint32_t iss_flags = 0x0UL;   \
            iss_flags |= event->umask;   \
            msr_write(cpu_id, MSR_M##NUM##_PMON_ISS, iss_flags);   \
        }   \
    break;   \
    case 0x06: /* FVC_EV0: FVC */   \
        flags |= (event->eventId<<9);   \
        {   \
            uint32_t fvc_flags = 0x0UL;   \
            fvc_flags |= (event->umask<<11);   \
            if (event->umask == 0x5)   \
            {   \
                fvc_flags |= (event->cmask<<5);   \
            }   \
            else   \
            {   \
                fvc_flags |= (event->cmask<<8);   \
            }   \
            msr_write(cpu_id, MSR_M##NUM##_PMON_ZDP, fvc_flags);   \
            VERBOSEPRINTREG(cpu_id, MSR_M##NUM##_PMON_ZDP, fvc_flags, FVC_EV0) \
        }   \
        break;   \
    case 0x07: /* FVC_EV1: FVC */   \
        flags |= (event->eventId<<9);   \
        {   \
            uint32_t fvc_flags = 0x0UL;   \
            fvc_flags |= (event->umask<<14);   \
            if (event->umask == 0x5)   \
            {   \
                fvc_flags |= (event->cmask<<5);   \
            }   \
            else   \
            {   \
                fvc_flags |= (event->cmask<<8);   \
            }   \
            msr_write(cpu_id, MSR_M##NUM##_PMON_ZDP, fvc_flags);   \
            VERBOSEPRINTREG(cpu_id, MSR_M##NUM##_PMON_ZDP, fvc_flags, FVC_EV1) \
        }   \
        break;   \
    case 0x08: /* FVC_EV2: FVC */   \
        flags |= (event->eventId<<9);   \
        {   \
            uint32_t fvc_flags = 0x0UL;   \
            fvc_flags |= (event->umask<<17);   \
            if (event->umask == 0x5)   \
            {   \
                fvc_flags |= (event->cmask<<5);   \
            }   \
            else   \
            {   \
                fvc_flags |= (event->cmask<<8);   \
            }   \
            msr_write(cpu_id, MSR_M##NUM##_PMON_ZDP, fvc_flags);   \
            VERBOSEPRINTREG(cpu_id, MSR_M##NUM##_PMON_ZDP, fvc_flags, FVC_EV2) \
        }   \
        break;   \
    case 0x09: /* FVC_EV3: FVC(ZDP) */   \
    flags |= (event->eventId<<9);   \
    {   \
        uint32_t fvc_flags = 0x0UL;   \
        fvc_flags |= (event->umask<<20);   \
        if (event->umask == 0x5)   \
        {   \
            fvc_flags |= (event->cmask<<5);   \
        }   \
        else   \
        {   \
            fvc_flags |= (event->cmask<<8);   \
        }   \
        msr_write(cpu_id, MSR_M##NUM##_PMON_ZDP, fvc_flags);   \
    }   \
    break;   \
    case 0x0A: /* ISS_SCHED: ISS */   \
    flags |= (event->eventId<<9);   \
    {   \
        uint32_t iss_flags = 0x0UL;   \
        iss_flags |= (event->umask<<10);   \
        msr_write(cpu_id, MSR_M##NUM##_PMON_ISS, iss_flags);   \
    }   \
    break;   \
    case 0x0B: /* PGT_PAGE_EV: PGT */   \
    flags |= (event->eventId<<9);   \
    {   \
        uint32_t pgt_flags = 0x0UL;   \
        pgt_flags |= event->umask;   \
        msr_write(cpu_id, MSR_M##NUM##_PMON_PGT, pgt_flags);   \
    }   \
    break;   \
    case 0x0C: /* PGT_PAGE_EV2: PGT */   \
    flags |= (event->eventId<<9);   \
    {   \
        uint32_t pgt_flags = 0x0UL;   \
        pgt_flags |= (event->umask<<11);   \
        msr_write(cpu_id, MSR_M##NUM##_PMON_PGT, pgt_flags);   \
    }   \
    break;   \
    case 0x0D: /* THERM_TRP_DN: THR */   \
    flags |= (event->eventId<<9);   \
    {   \
        uint32_t thr_flags = 0x0UL;   \
        thr_flags |= (1<<3);   \
        thr_flags |= (event->umask<<9);   \
        msr_write(cpu_id, MSR_M##NUM##_PMON_PGT, thr_flags);   \
    }   \
    break;   \
}


void
perfmon_setupCounterThread_nehalemEX(int thread_id,
        PerfmonEvent* event,
        PerfmonCounterIndex index)
{
    uint64_t flags = 0x0ULL;;
    uint64_t reg = perfmon_threadData[thread_id].counters[index].configRegister;
    int cpu_id = perfmon_threadData[thread_id].processorId;
    perfmon_threadData[thread_id].counters[index].init = TRUE;

    switch (perfmon_threadData[thread_id].counters[index].type)
    {
        case PMC:
            flags = msr_read(cpu_id,reg);
            flags &= ~(0xFFFFU);   /* clear lower 16bits */

            /* Intel with standard 8 bit event mask: [7:0] */
            flags |= (event->umask<<8) + event->eventId;

            if (event->cfgBits != 0) /* set custom cfg and cmask */
            {
                flags &= ~(0xFFFFU<<16);  /* clear upper 16bits */
                flags |= ((event->cmask<<8) + event->cfgBits)<<16;
            }

            msr_write(cpu_id, reg , flags);
            VERBOSEPRINTREG(cpu_id, reg, flags, PMC_EV_SEL)
            break;

        case FIXED:
            break;

        case MBOX0:
            MBOX_GATE_NEHEX(0);
            msr_write(cpu_id, reg , flags);
            VERBOSEPRINTREG(cpu_id, reg, flags, MBOX0_CTRL)
            break;

        case MBOX1:
            MBOX_GATE_NEHEX(1);
            msr_write(cpu_id, reg , flags);
            VERBOSEPRINTREG(cpu_id, reg, flags, MBOX1_CTRL)
            break;

        case BBOX0:

        case BBOX1:
            flags = 0x1ULL; /* set enable bit */
            flags |=  (event->eventId<<1);
            msr_write(cpu_id, reg , flags);
            VERBOSEPRINTREG(cpu_id, reg, flags, BBOX_CTRL)
            break;

        case RBOX0:
            RBOX_GATE(0);
            msr_write(cpu_id, reg , flags);
            VERBOSEPRINTREG(cpu_id, reg, flags, RBOX0_CTRL)
            break;

        case RBOX1:
            RBOX_GATE(1);
            msr_write(cpu_id, reg , flags);
            VERBOSEPRINTREG(cpu_id, reg, flags, RBOX1_CTRL)
            break;

        case WBOX:
            if (event->eventId == 0xFF)  /* Fixed Counter */
            {
                flags = 0x1ULL; /* set enable bit */
            }
            else
            {
                flags |= (1<<22); /* set enable bit */
                flags |= (event->umask<<8) + event->eventId;
            }
            msr_write(cpu_id, reg , flags);
            VERBOSEPRINTREG(cpu_id, reg, flags, WBOX_CTRL)
            break;

        default:
            /* should never be reached */
            break;
    }
}


#if 0
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

    msr_write(cpu_id, MSR_PERFEVTSEL0, flags);
    msr_write(cpu_id, MSR_PERFEVTSEL1, flags);
    msr_write(cpu_id, MSR_PERFEVTSEL2, flags);
    msr_write(cpu_id, MSR_PERFEVTSEL3, flags);
}


void
perfmon_setupCounterThread_nehalemEX(int thread_id,
        PerfmonEvent* event,
        PerfmonCounterIndex index)
{
    uint64_t flags;
    uint64_t reg = perfmon_threadData[thread_id].counters[index].configRegister;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if ((perfmon_threadData[thread_id].counters[index].type == PMC) || perfmon_threadData[thread_id].counters[index].type == UNCORE)
    {

        perfmon_threadData[thread_id].counters[index].init = TRUE;
        flags = msr_read(cpu_id,reg);
        flags &= ~(0xFFFFU);   /* clear lower 16bits */

        /* Intel with standard 8 bit event mask: [7:0] */
        flags |= (event->umask<<8) + event->eventId;

        if (event->cfgBits != 0) /* set custom cfg and cmask */
        {
            flags &= ~(0xFFFFU<<16);  /* clear upper 16bits */
            flags |= ((event->cmask<<8) + event->cfgBits)<<16;
        }

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
//    printf ("Status: 0x%llX \n", LLU_CAST flags);
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

#endif

