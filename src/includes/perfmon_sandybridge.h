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

#include <perfmon_sandybridge_events.h>
#include <perfmon_sandybridge_groups.h>
#include <perfmon_sandybridge_counters.h>

static int perfmon_numCountersSandybridge = NUM_COUNTERS_SANDYBRIDGE;
static int perfmon_numGroupsSandybridge = NUM_GROUPS_SANDYBRIDGE;
static int perfmon_numArchEventsSandybridge = NUM_ARCH_EVENTS_SANDYBRIDGE;

#define OFFSET_PMC 3

void perfmon_init_sandybridge(PerfmonThread *thread)
{
    uint64_t flags = 0x0ULL;
    int cpu_id = thread->processorId;

    /* Initialize registers */
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

    /* Preinit of PERFEVSEL registers */
    flags |= (1<<22);  /* enable flag */
    flags |= (1<<16);  /* user mode flag */

    msr_write(cpu_id, MSR_PERFEVTSEL0, flags);
    msr_write(cpu_id, MSR_PERFEVTSEL1, flags);
    msr_write(cpu_id, MSR_PERFEVTSEL2, flags);
    msr_write(cpu_id, MSR_PERFEVTSEL3, flags);

    /* TODO Robust implementation which also works if stuff is not there */
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id) ||
            lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id))
    {
        if ( cpuid_info.model == SANDYBRIDGE_EP )
        {
            /* Only root can access pci address space in direct mode */
            if (accessClient_mode != DAEMON_AM_DIRECT)
            {
                uint32_t  uflags = 0x10100U; /* enable freeze (bit 16), freeze (bit 8) */
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_BOX_CTL, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_BOX_CTL, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_BOX_CTL, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_BOX_CTL, uflags);

                uflags = 0x0U;
                uflags |= (1<<22);  /* enable flag */
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTL_0, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTL_1, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTL_2, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTL_3, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTL_0, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTL_1, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTL_2, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTL_3, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTL_0, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTL_1, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTL_2, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTL_3, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTL_0, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTL_1, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTL_2, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTL_3, uflags);

                uflags |= (1<<19);  /* reset fixed counter */
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_FIXED_CTL, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_FIXED_CTL, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_FIXED_CTL, uflags);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_FIXED_CTL, uflags);

                /* iMC counters need to be manually reset to zero */
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_0_A, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_0_B, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_1_A, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_1_B, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_2_A, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_2_B, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_3_A, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_3_B, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_0_A, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_0_B, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_1_A, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_1_B, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_2_A, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_2_B, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_3_A, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_3_B, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_0_A, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_0_B, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_1_A, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_1_B, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_2_A, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_2_B, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_3_A, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_3_B, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_0_A, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_0_B, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_1_A, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_1_B, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_2_A, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_2_B, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_3_A, 0x0U);
                pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_3_B, 0x0U);

                /* FIXME: Not yet tested/ working due to BIOS issues on test
                 * machines */
#if 0
                /* QPI registers can be zeroed with single write */
                uflags = 0x0113UL; /*enable freeze (bit 16), freeze (bit 8), reset */
                pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_BOX_CTL, uflags);
                pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_BOX_CTL, uflags);
                uflags = 0x0UL;
                uflags |= (1UL<<22);  /* enable flag */
                pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_CTL_0, uflags);
                pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_CTL_1, uflags);
                pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_CTL_2, uflags);
                pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_CTL_3, uflags);
                pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_CTL_0, uflags);
                pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_CTL_1, uflags);
                pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_CTL_2, uflags);
                pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_CTL_3, uflags);
#endif
            }
        }
    }
//    lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
}

#define BOX_GATE_SNB(channel,label) \
    if (perfmon_verbose) { \
        printf("[%d] perfmon_setup_counter (label): Write Register 0x%llX , Flags: 0x%llX \n", \
                cpu_id, \
                LLU_CAST reg, \
                LLU_CAST flags); \
    } \
if(haveLock) { \
    uflags = pci_read(cpu_id, channel, reg);  \
    uflags &= ~(0xFFFFU);  \
    uflags |= (event->umask<<8) + event->eventId;  \
    pci_write(cpu_id, channel,  reg, uflags);  \
}


void perfmon_setupCounterThread_sandybridge(
        int thread_id,
        PerfmonEvent* event,
        PerfmonCounterIndex index)
{
    int haveLock = 0;
    uint64_t flags;
    uint32_t uflags;
    uint64_t reg = sandybridge_counter_map[index].configRegister;
    int cpu_id = perfmon_threadData[thread_id].processorId;
    perfmon_threadData[thread_id].counters[index].init = TRUE;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    switch (sandybridge_counter_map[index].type)
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

            if (perfmon_verbose)
            {
                printf("[%d] perfmon_setup_counter PMC: Write Register 0x%llX , Flags: 0x%llX \n",
                        cpu_id,
                        LLU_CAST reg,
                        LLU_CAST flags);
            }

            msr_write(cpu_id, reg , flags);
            break;

        case FIXED:
            break;

        case POWER:
            break;

        case MBOX0:
            BOX_GATE_SNB(PCI_IMC_DEVICE_CH_0,MBOX0);
            break;

        case MBOX1:
            BOX_GATE_SNB(PCI_IMC_DEVICE_CH_1,MBOX1);
            break;

        case MBOX2:
            BOX_GATE_SNB(PCI_IMC_DEVICE_CH_2,MBOX2);
            break;

        case MBOX3:
            BOX_GATE_SNB(PCI_IMC_DEVICE_CH_3,MBOX3);
            break;

        case SBOX0:

            /* CTO_COUNT event requires programming of MATCH/MASK registers */
            if (event->eventId == 0x38)
            {
                if(haveLock)
                {
                    uflags = pci_read(cpu_id, PCI_QPI_DEVICE_PORT_0, reg);
                    uflags &= ~(0xFFFFU);
                    uflags |= (1UL<<21) + event->eventId; /* Set extension bit */
                    printf("UFLAGS 0x%x \n",uflags);
                    pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  reg, uflags);

                    /* program MATCH0 */
                    uflags = 0x0UL;
                    uflags = (event->cmask<<13) + (event->umask<<8);
                    printf("MATCH UFLAGS 0x%x \n",uflags);
                    pci_write(cpu_id, PCI_QPI_MASK_DEVICE_PORT_0, PCI_UNC_QPI_PMON_MATCH_0, uflags);

                    /* program MASK0 */
                    uflags = 0x0UL;
                    uflags = (0x3F<<12) + (event->cfgBits<<4);
                    printf("MASK UFLAGS 0x%x \n",uflags);
                    pci_write(cpu_id, PCI_QPI_MASK_DEVICE_PORT_0, PCI_UNC_QPI_PMON_MASK_0, uflags);
                }
            }
            else
            {
                BOX_GATE_SNB(PCI_QPI_DEVICE_PORT_0,SBOX0);
            }

            break;

        case SBOX1:

            /* CTO_COUNT event requires programming of MATCH/MASK registers */
            if (event->eventId == 0x38)
            {
                if(haveLock)
                {
                    uflags = pci_read(cpu_id, PCI_QPI_DEVICE_PORT_1, reg);
                    uflags &= ~(0xFFFFU);
                    uflags |= (1UL<<21) + event->eventId; /* Set extension bit */
                    pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  reg, uflags);

                    /* program MATCH0 */
                    uflags = 0x0UL;
                    uflags = (event->cmask<<13) + (event->umask<<8);
                    pci_write(cpu_id, PCI_QPI_MASK_DEVICE_PORT_1, PCI_UNC_QPI_PMON_MATCH_0, uflags);

                    /* program MASK0 */
                    uflags = 0x0UL;
                    uflags = (0x3F<<12) + (event->cfgBits<<4);
                    pci_write(cpu_id, PCI_QPI_MASK_DEVICE_PORT_1, PCI_UNC_QPI_PMON_MASK_0, uflags);
                }
            }
            else
            {
                BOX_GATE_SNB(PCI_QPI_DEVICE_PORT_0,SBOX0);
            }
            break;

        default:
            /* should never be reached */
            break;
    }
}

void perfmon_startCountersThread_sandybridge(int thread_id)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint32_t uflags = 0x10000UL; /* Clear freeze bit */
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);

    for ( int i=0; i<perfmon_numCountersSandybridge; i++ )
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE)
        {
            switch (sandybridge_counter_map[i].type)
            {
                case PMC:
                    msr_write(cpu_id, sandybridge_counter_map[i].counterRegister, 0x0ULL);
                    flags |= (1<<(i-OFFSET_PMC));  /* enable counter */
                    break;

                case FIXED:
                    msr_write(cpu_id, sandybridge_counter_map[i].counterRegister, 0x0ULL);
                    flags |= (1ULL<<(i+32));  /* enable fixed counter */
                    break;

                case POWER:
                    if(haveLock)
                    {
                        perfmon_threadData[thread_id].counters[i].counterData =
                            power_read(cpu_id, sandybridge_counter_map[i].counterRegister);
                    }

                    break;

                case MBOX0:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_BOX_CTL, uflags);
                    }
                    break;

                case MBOX1:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_BOX_CTL, uflags);
                    }
                    break;

                case MBOX2:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_BOX_CTL, uflags);
                    }
                    break;

                case MBOX3:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_BOX_CTL, uflags);
                    }
                    break;

                case MBOXFIX:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_FIXED_CTL, 0x48000UL);
                    }
                    break;

                case SBOX0:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_BOX_CTL, uflags);
                    }
                    break;

                case SBOX1:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_BOX_CTL, uflags);
                    }
                    break;

                default:
                    /* should never be reached */
                    break;
            }
        }
    }

    if (perfmon_verbose)
    {
        printf("perfmon_start_counters: Write Register 0x%X , \
                Flags: 0x%llX \n",MSR_PERF_GLOBAL_CTRL, LLU_CAST flags);
        printf("perfmon_start_counters: Write Register 0x%X , \
                Flags: 0x%llX \n",MSR_UNCORE_PERF_GLOBAL_CTRL, LLU_CAST uflags);
    }

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, flags);
    msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x30000000FULL);
}

void perfmon_stopCountersThread_sandybridge(int thread_id)
{
    uint64_t flags;
    uint32_t uflags = 0x10100UL; /* Set freeze bit */
    uint64_t counter_result = 0x0ULL;
    int haveLock = 0;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);

    for ( int i=0; i < perfmon_numCountersSandybridge; i++ ) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            switch (sandybridge_counter_map[i].type)
            {
                case PMC:

                case FIXED:
                    perfmon_threadData[thread_id].counters[i].counterData =
                        msr_read(cpu_id, sandybridge_counter_map[i].counterRegister);
                    break;

                case POWER:
                    if(haveLock)
                    {
                        perfmon_threadData[thread_id].counters[i].counterData =
                            power_info.energyUnit *
                            ( power_read(cpu_id, sandybridge_counter_map[i].counterRegister) -
                              perfmon_threadData[thread_id].counters[i].counterData);
                    }
                    break;

                case THERMAL:
                        perfmon_threadData[thread_id].counters[i].counterData =
                             thermal_read(cpu_id);
                    break;

                case MBOX0:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_BOX_CTL, uflags);
                        counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_0,
                                sandybridge_counter_map[i].counterRegister);

                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_IMC_DEVICE_CH_0,
                                    sandybridge_counter_map[i].counterRegister2);

                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                    }
                    break;

                case MBOX1:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_BOX_CTL, uflags);
                        counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_1,
                                sandybridge_counter_map[i].counterRegister);

                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_IMC_DEVICE_CH_1,
                                    sandybridge_counter_map[i].counterRegister2);

                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                    }
                    break;

                case MBOX2:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_BOX_CTL, uflags);

                        counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_2,
                                sandybridge_counter_map[i].counterRegister);

                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_IMC_DEVICE_CH_2,
                                    sandybridge_counter_map[i].counterRegister2);

                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                    }
                    break;

                case MBOX3:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_BOX_CTL, uflags);

                        counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_3,
                                sandybridge_counter_map[i].counterRegister);

                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_IMC_DEVICE_CH_3,
                                    sandybridge_counter_map[i].counterRegister2);

                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                    }
                    break;

                case MBOXFIX:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_FIXED_CTL, uflags);

                        counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_0,
                                sandybridge_counter_map[i].counterRegister);

                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_IMC_DEVICE_CH_0,
                                    sandybridge_counter_map[i].counterRegister2);

                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                    }
                    break;

                case SBOX0:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_BOX_CTL, uflags);

                        counter_result = pci_read(cpu_id, PCI_QPI_DEVICE_PORT_0,
                                sandybridge_counter_map[i].counterRegister);

                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_QPI_DEVICE_PORT_0,
                                    sandybridge_counter_map[i].counterRegister2);

                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                    }
                    break;

                case SBOX1:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_BOX_CTL, uflags);
                        counter_result = pci_read(cpu_id, PCI_QPI_DEVICE_PORT_1,
                                sandybridge_counter_map[i].counterRegister);

                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_QPI_DEVICE_PORT_1,
                                    sandybridge_counter_map[i].counterRegister2);

                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                    }
                    break;

                default:
                    /* should never be reached */
                    break;
            }
        }
    }

    flags = msr_read(cpu_id,MSR_PERF_GLOBAL_STATUS);
    //    printf ("Status: 0x%llX \n", LLU_CAST flags);
    if ( (flags & 0x3) || (flags & (0x3ULL<<32)) ) 
    {
        printf ("Overflow occured \n");
    }
}

void perfmon_readCountersThread_sandybridge(int thread_id)
{
    uint64_t counter_result = 0x0ULL;
    int haveLock = 0;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    for ( int i=0; i<perfmon_numCountersSandybridge; i++ )
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE)
        {
            if ((sandybridge_counter_map[i].type == PMC) ||
                    (sandybridge_counter_map[i].type == FIXED))
            {
                perfmon_threadData[thread_id].counters[i].counterData =
                    msr_read(cpu_id, sandybridge_counter_map[i].counterRegister);
            }
            else
            {
                if(haveLock)
                {
                    switch (sandybridge_counter_map[i].type)
                    {
                        case POWER:
                            perfmon_threadData[thread_id].counters[i].counterData =
                                power_info.energyUnit *
                                power_read(cpu_id, sandybridge_counter_map[i].counterRegister);
                            break;

                        case MBOX0:
                            counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_0,
                                    sandybridge_counter_map[i].counterRegister);

                            counter_result = (counter_result<<32) +
                                pci_read(cpu_id, PCI_IMC_DEVICE_CH_0,
                                        sandybridge_counter_map[i].counterRegister2);

                            perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                            break;

                        case MBOX1:
                            counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_1,
                                    sandybridge_counter_map[i].counterRegister);

                            counter_result = (counter_result<<32) +
                                pci_read(cpu_id, PCI_IMC_DEVICE_CH_1,
                                        sandybridge_counter_map[i].counterRegister2);

                            perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                            break;

                        case MBOX2:
                            counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_2,
                                    sandybridge_counter_map[i].counterRegister);

                            counter_result = (counter_result<<32) +
                                pci_read(cpu_id, PCI_IMC_DEVICE_CH_2,
                                        sandybridge_counter_map[i].counterRegister2);

                            perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                            break;

                        case MBOX3:
                            counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_3,
                                    sandybridge_counter_map[i].counterRegister);

                            counter_result = (counter_result<<32) +
                                pci_read(cpu_id, PCI_IMC_DEVICE_CH_3,
                                        sandybridge_counter_map[i].counterRegister2);

                            perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                            break;

                        default:
                            /* should never be reached */
                            break;
                    }
                }
            }
        }
    }
}

