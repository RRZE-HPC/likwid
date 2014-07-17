/*
 * =======================================================================================
 *
 *      Filename:  perfmon_ivybridge.h
 *
 *      Description:  Header File of perfmon module for Ivy Bridge.
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

#include <perfmon_ivybridge_events.h>
#include <perfmon_ivybridge_groups.h>
#include <perfmon_ivybridge_counters.h>
#include <error.h>
#include <affinity.h>
#include <limits.h>

static int perfmon_numCountersIvybridge = NUM_COUNTERS_IVYBRIDGE;
static int perfmon_numGroupsIvybridge = NUM_GROUPS_IVYBRIDGE;
static int perfmon_numArchEventsIvybridge = NUM_ARCH_EVENTS_IVYBRIDGE;

#define OFFSET_PMC 3

int perfmon_init_ivybridge(int cpu_id)
{
    uint64_t flags = 0x0ULL;

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

    /* TODO Robust implementation which also works if stuff is not there */
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id) ||
            lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id))
    {
        if ( cpuid_info.model == IVYBRIDGE_EP )
        {
            /* Only root can access pci address space in direct mode */
            if (accessClient_mode != DAEMON_AM_DIRECT)
            {
                uint32_t  uflags = 0x10100U; /* enable freeze (bit 16), freeze (bit 8) */
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_BOX_CTL, uflags));

                uflags = 0x0U;
                uflags |= (1<<22);  /* enable flag */
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTL_0, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTL_1, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTL_2, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTL_3, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTL_0, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTL_1, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTL_2, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTL_3, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTL_0, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTL_1, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTL_2, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTL_3, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTL_0, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTL_1, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTL_2, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTL_3, uflags));

                uflags |= (1<<19);  /* reset fixed counter */
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_FIXED_CTL, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_FIXED_CTL, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_FIXED_CTL, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_FIXED_CTL, uflags));

                /* iMC counters need to be manually reset to zero */
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_0_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_0_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_1_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_1_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_2_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_2_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_3_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_3_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_0_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_0_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_1_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_1_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_2_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_2_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_3_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_3_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_0_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_0_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_1_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_1_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_2_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_2_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_3_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_3_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_0_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_0_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_1_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_1_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_2_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_2_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_3_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_3_B, 0x0U));

                /* FIXME: Not yet tested/ working due to BIOS issues on test
                 * machines */

                /* QPI registers can be zeroed with single write */
                uflags = 0x0113UL; /*enable freeze (bit 16), freeze (bit 8), reset */
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_BOX_CTL, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_BOX_CTL, uflags));
                uflags = 0x0UL;
                uflags |= (1UL<<22);  /* enable flag */
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_CTL_0, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_CTL_1, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_CTL_2, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_CTL_3, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_CTL_0, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_CTL_1, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_CTL_2, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_CTL_3, uflags));

#if 0
                /* Cbo counters */
                uflags = 0xF0103UL; /*enable freeze (bit 8), reset */
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C0_PMON_BOX_CTL, uflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C1_PMON_BOX_CTL, uflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C2_PMON_BOX_CTL, uflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C3_PMON_BOX_CTL, uflags));

                switch ( cpuid_topology.numCoresPerSocket )
                {
                    case 12:
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C11_PMON_BOX_CTL, uflags));
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C10_PMON_BOX_CTL, uflags));
                    case 10:
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C9_PMON_BOX_CTL, uflags));
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C8_PMON_BOX_CTL, uflags));
                    case 8:
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C7_PMON_BOX_CTL, uflags));
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C6_PMON_BOX_CTL, uflags));
                    case 6:
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C5_PMON_BOX_CTL, uflags));
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C4_PMON_BOX_CTL, uflags));
                }
#endif
            }
        }
    }
    return 0;
}

#define BOX_GATE_SNB(channel,label) \
    if(haveLock) { \
        CHECK_PCI_READ_ERROR(pci_read(cpu_id, channel, reg, &uflags));  \
        uflags &= ~(0xFFFFU);  \
        uflags |= (event->umask<<8) + event->eventId;  \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, channel,  reg, uflags));  \
    }
    /*if (perfmon_verbose) { \
        printf("[%d] perfmon_setup_counter (label): Write Register 0x%llX , Flags: 0x%llX \n", \
                cpu_id, \
                LLU_CAST reg, \
                LLU_CAST flags); \
    } \*/
    


int perfmon_setupCounterThread_ivybridge(
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
        uint64_t reg = ivybridge_counter_map[index].configRegister;
        PerfmonEvent *event = &(eventSet->events[i].event);
        
        
        switch (ivybridge_counter_map[index].type)
        {
            case PMC:
                eventSet->events[i].threadCounter[thread_id].init = TRUE;
                CHECK_MSR_READ_ERROR(msr_read(cpu_id,reg, &flags));
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
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_QPI_DEVICE_PORT_0, reg, &uflags));
                        uflags &= ~(0xFFFFU);
                        uflags |= (1UL<<21) + event->eventId; /* Set extension bit */
                        printf("UFLAGS 0x%x \n",uflags);
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  reg, uflags));

                        /* program MATCH0 */
                        uflags = 0x0UL;
                        uflags = (event->cmask<<13) + (event->umask<<8);
                        printf("MATCH UFLAGS 0x%x \n",uflags);
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_MASK_DEVICE_PORT_0, PCI_UNC_QPI_PMON_MATCH_0, uflags));

                        /* program MASK0 */
                        uflags = 0x0UL;
                        uflags = (0x3F<<12) + (event->cfgBits<<4);
                        printf("MASK UFLAGS 0x%x \n",uflags);
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_MASK_DEVICE_PORT_0, PCI_UNC_QPI_PMON_MASK_0, uflags));
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
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_QPI_DEVICE_PORT_1, reg, &uflags));
                        uflags &= ~(0xFFFFU);
                        uflags |= (1UL<<21) + event->eventId; /* Set extension bit */
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  reg, uflags));

                        /* program MATCH0 */
                        uflags = 0x0UL;
                        uflags = (event->cmask<<13) + (event->umask<<8);
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_MASK_DEVICE_PORT_1, PCI_UNC_QPI_PMON_MATCH_0, uflags));

                        /* program MASK0 */
                        uflags = 0x0UL;
                        uflags = (0x3F<<12) + (event->cfgBits<<4);
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_MASK_DEVICE_PORT_1, PCI_UNC_QPI_PMON_MASK_0, uflags));
                    }
                }
                else
                {
                    BOX_GATE_SNB(PCI_QPI_DEVICE_PORT_0,SBOX0);
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

                if(haveLock)
                {
                    eventSet->events[i].threadCounter[thread_id].init = TRUE;
                    uflags = 0x0U;

                    /* set local enable flag */
                    uflags |= 1<<22;
                    /* Intel with standard 8 bit event mask: [7:0] */
                    uflags |= (event->umask<<8) + event->eventId;
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , uflags));

                    /*if (perfmon_verbose)
                    {
                        printf("[%d] perfmon_setup_counter: Write Register 0x%llX , uFlags: 0x%lX \n",
                                cpu_id,
                                LLU_CAST reg,
                                (unsigned long) uflags);
                    }*/
                }
                break;

            default:
                /* should never be reached */
                break;
        }
    }
    return 0;
}

#define CBOX_START(NUM) \
if(haveLock) { \
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C##NUM##_PMON_BOX_CTL, uflags));  \
}

#define MBOX_START(NUM) \
if(haveLock) { \
    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_##NUM,  PCI_UNC_MC_PMON_BOX_CTL, uflags)); \
}



int perfmon_startCountersThread_ivybridge(int thread_id, PerfmonEventSet* eventSet)
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

    for ( int i=0; i<perfmon_numCountersIvybridge; i++ )
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE) 
        {
            PerfmonCounterIndex index = eventSet->events[i].index;
            switch (ivybridge_counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, ivybridge_counter_map[index].counterRegister, 0x0ULL));
                    flags |= (1<<(index-OFFSET_PMC));  /* enable counter */
                    break;

                case FIXED:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, ivybridge_counter_map[index].counterRegister, 0x0ULL));
                    flags |= (1ULL<<(index+32));  /* enable fixed counter */
                    break;

                case POWER:
                    if(haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, ivybridge_counter_map[index].counterRegister,
                                        (uint32_t*)&eventSet->events[i].threadCounter[thread_id].counterData));
                    }

                    break;

                case MBOX0:
                    MBOX_START(0);
                    break;

                case MBOX1:
                    MBOX_START(1);
                    break;

                case MBOX2:
                    MBOX_START(2);
                    break;

                case MBOX3:
                    MBOX_START(3);
                    break;

                case MBOXFIX:
                    break;

                case SBOX0:
                    if(haveLock)
                    {
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_BOX_CTL, uflags));
                    }
                    break;

                case SBOX1:
                    if(haveLock)
                    {
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_BOX_CTL, uflags));
                    }
                    break;

                case CBOX0:
                    CBOX_START(0);
                    break;

                case CBOX1:
                    CBOX_START(1);
                    break;

                case CBOX2:
                    CBOX_START(2);
                    break;

                case CBOX3:
                    CBOX_START(3);
                    break;

                case CBOX4:
                    CBOX_START(4);
                    break;

                case CBOX5:
                    CBOX_START(5);
                    break;

                case CBOX6:
                    CBOX_START(6);
                    break;

                case CBOX7:
                    CBOX_START(7);
                    break;

                case CBOX8:
                    CBOX_START(8);
                    break;

                case CBOX9:
                    CBOX_START(9);
                    break;

                case CBOX10:
                    CBOX_START(10);
                    break;

                case CBOX11:
                    CBOX_START(11);
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

#define CBOX_STOP(NUM) \
if(haveLock) { \
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C##NUM##_PMON_BOX_CTL, uflags));  \
    CHECK_MSR_READ_ERROR(msr_read(cpu_id, ivybridge_counter_map[index].counterRegister, &eventSet->events[i].threadCounter[thread_id].counterData));    \
}

#define MBOX_STOP(NUM) \
if(haveLock) { \
    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_##NUM ,  PCI_UNC_MC_PMON_BOX_CTL, uflags)); \
    CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_##NUM , ivybridge_counter_map[index].counterRegister, (uint32_t*)&counter_result)); \
    eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);  \
    CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_##NUM , ivybridge_counter_map[index].counterRegister2, (uint32_t*)&counter_result)); \
    eventSet->events[i].threadCounter[thread_id].counterData += counter_result; \
}

#define SBOX_STOP(NUM) \
if(haveLock) { \
    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_##NUM ,  PCI_UNC_QPI_PMON_BOX_CTL, uflags)); \
    pci_read(cpu_id, PCI_QPI_DEVICE_PORT_##NUM , ivybridge_counter_map[index].counterRegister, (uint32_t*)&counter_result); \
    eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);  \
    pci_read(cpu_id, PCI_QPI_DEVICE_PORT_##NUM , ivybridge_counter_map[index].counterRegister2, (uint32_t*)&counter_result);  \
    eventSet->events[i].threadCounter[thread_id].counterData += counter_result; \
}


int perfmon_stopCountersThread_ivybridge(int thread_id, PerfmonEventSet* eventSet)
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
            switch (ivybridge_counter_map[index].type)
            {
                case PMC:

                case FIXED:
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, ivybridge_counter_map[index].counterRegister, &counter_result));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case POWER:
                    if(haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, ivybridge_counter_map[index].counterRegister,
                            (uint32_t*)&counter_result));
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            fprintf(stderr,"Overflow in power status register 0x%x, assuming single overflow\n",
                                                ivybridge_counter_map[index].counterRegister);
                            counter_result += (UINT_MAX - perfmon_threadData[thread_id].counters[i].counterData);
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

                case MBOX0:
                    MBOX_STOP(0);
                    break;

                case MBOX1:
                    MBOX_STOP(1);
                    break;

                case MBOX2:
                    MBOX_STOP(2);
                    break;

                case MBOX3:
                    MBOX_STOP(3);
                    break;

                case MBOXFIX:
                    if(haveLock)
                    {
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_FIXED_CTL, uflags));

                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_0,
                                    ivybridge_counter_map[index].counterRegister, (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_0,
                                    ivybridge_counter_map[index].counterRegister2, (uint32_t*)&counter_result));

                        eventSet->events[i].threadCounter[thread_id].counterData += counter_result;
                    }
                    break;

                case SBOX0:
                    SBOX_STOP(0);
                    break;

                case SBOX1:
                    SBOX_STOP(1);
                    break;

                case CBOX0:
                    CBOX_STOP(0);
                    break;

                case CBOX1:
                    CBOX_STOP(1);
                    break;

                case CBOX2:
                    CBOX_STOP(2);
                    break;

                case CBOX3:
                    CBOX_STOP(3);
                    break;

                case CBOX4:
                    CBOX_STOP(4);
                    break;

                case CBOX5:
                    CBOX_STOP(5);
                    break;

                case CBOX6:
                    CBOX_STOP(6);
                    break;

                case CBOX7:
                    CBOX_STOP(7);
                    break;

                case CBOX8:
                    CBOX_STOP(8);
                    break;

                case CBOX9:
                    CBOX_STOP(9);
                    break;

                case CBOX10:
                    CBOX_STOP(10);
                    break;

                case CBOX11:
                    CBOX_STOP(11);
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

int perfmon_readCountersThread_ivybridge(int thread_id, PerfmonEventSet* eventSet)
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
            if ((ivybridge_counter_map[index].type == PMC) ||
                    (ivybridge_counter_map[index].type == FIXED))
            {
                CHECK_MSR_READ_ERROR(msr_read(cpu_id, ivybridge_counter_map[index].counterRegister, &counter_result));
                eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
            }
            else
            {
                if(haveLock)
                {
                    switch (ivybridge_counter_map[index].type)
                    {
                        case POWER:
                            CHECK_POWER_READ_ERROR(power_read(cpu_id, ivybridge_counter_map[index].counterRegister,
                                    (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData = power_info.energyUnit * counter_result;
                            break;

                        case MBOX0:
                            CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_0,
                                    ivybridge_counter_map[index].counterRegister, (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);
                            
                            CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_0,
                                        ivybridge_counter_map[index].counterRegister2, (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData += counter_result;

                            break;

                        case MBOX1:
                            CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_1,
                                    ivybridge_counter_map[index].counterRegister, (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);
                            
                            CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_1,
                                        ivybridge_counter_map[index].counterRegister2, (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData += counter_result;
                            break;

                        case MBOX2:
                            CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_2,
                                    ivybridge_counter_map[index].counterRegister, (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);
                            
                            CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_2,
                                        ivybridge_counter_map[index].counterRegister2, (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData += counter_result;
                            break;

                        case MBOX3:
                            CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_3,
                                    ivybridge_counter_map[index].counterRegister, (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);
                            
                            CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_3,
                                        ivybridge_counter_map[index].counterRegister2, (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData += counter_result;
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

