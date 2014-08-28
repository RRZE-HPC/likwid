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
#include <perfmon_ivybridge_counters.h>
#include <error.h>
#include <affinity.h>
#include <limits.h>

static int perfmon_numCountersIvybridge = NUM_COUNTERS_IVYBRIDGE;
static int perfmon_numCoreCountersIvybridge = NUM_COUNTERS_CORE_IVYBRIDGE;
static int perfmon_numArchEventsIvybridge = NUM_ARCH_EVENTS_IVYBRIDGE;

#define OFFSET_PMC 3

int perfmon_init_ivybridge(int cpu_id)
{
    uint64_t flags = 0x0ULL;
    if ( cpuid_info.model == IVYBRIDGE_EP )
    {
        lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    }
    return 0;

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
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_0,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_1,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_2,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_3,  PCI_UNC_MC_PMON_BOX_CTL, uflags));

                uflags = 0x0U;
                uflags |= (1<<22);  /* enable flag */
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_0,  PCI_UNC_MC_PMON_CTL_0, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_0,  PCI_UNC_MC_PMON_CTL_1, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_0,  PCI_UNC_MC_PMON_CTL_2, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_0,  PCI_UNC_MC_PMON_CTL_3, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_1,  PCI_UNC_MC_PMON_CTL_0, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_1,  PCI_UNC_MC_PMON_CTL_1, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_1,  PCI_UNC_MC_PMON_CTL_2, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_1,  PCI_UNC_MC_PMON_CTL_3, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_2,  PCI_UNC_MC_PMON_CTL_0, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_2,  PCI_UNC_MC_PMON_CTL_1, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_2,  PCI_UNC_MC_PMON_CTL_2, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_2,  PCI_UNC_MC_PMON_CTL_3, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_3,  PCI_UNC_MC_PMON_CTL_0, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_3,  PCI_UNC_MC_PMON_CTL_1, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_3,  PCI_UNC_MC_PMON_CTL_2, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_3,  PCI_UNC_MC_PMON_CTL_3, uflags));

                uflags |= (1<<19);  /* reset fixed counter */
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_0,  PCI_UNC_MC_PMON_FIXED_CTL, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_1,  PCI_UNC_MC_PMON_FIXED_CTL, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_2,  PCI_UNC_MC_PMON_FIXED_CTL, uflags));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_3,  PCI_UNC_MC_PMON_FIXED_CTL, uflags));

                /* iMC counters need to be manually reset to zero */
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_0,  PCI_UNC_MC_PMON_CTR_0_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_0,  PCI_UNC_MC_PMON_CTR_0_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_0,  PCI_UNC_MC_PMON_CTR_1_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_0,  PCI_UNC_MC_PMON_CTR_1_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_0,  PCI_UNC_MC_PMON_CTR_2_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_0,  PCI_UNC_MC_PMON_CTR_2_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_0,  PCI_UNC_MC_PMON_CTR_3_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_0,  PCI_UNC_MC_PMON_CTR_3_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_1,  PCI_UNC_MC_PMON_CTR_0_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_1,  PCI_UNC_MC_PMON_CTR_0_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_1,  PCI_UNC_MC_PMON_CTR_1_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_1,  PCI_UNC_MC_PMON_CTR_1_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_1,  PCI_UNC_MC_PMON_CTR_2_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_1,  PCI_UNC_MC_PMON_CTR_2_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_1,  PCI_UNC_MC_PMON_CTR_3_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_1,  PCI_UNC_MC_PMON_CTR_3_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_2,  PCI_UNC_MC_PMON_CTR_0_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_2,  PCI_UNC_MC_PMON_CTR_0_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_2,  PCI_UNC_MC_PMON_CTR_1_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_2,  PCI_UNC_MC_PMON_CTR_1_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_2,  PCI_UNC_MC_PMON_CTR_2_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_2,  PCI_UNC_MC_PMON_CTR_2_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_2,  PCI_UNC_MC_PMON_CTR_3_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_2,  PCI_UNC_MC_PMON_CTR_3_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_3,  PCI_UNC_MC_PMON_CTR_0_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_3,  PCI_UNC_MC_PMON_CTR_0_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_3,  PCI_UNC_MC_PMON_CTR_1_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_3,  PCI_UNC_MC_PMON_CTR_1_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_3,  PCI_UNC_MC_PMON_CTR_2_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_3,  PCI_UNC_MC_PMON_CTR_2_B, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_3,  PCI_UNC_MC_PMON_CTR_3_A, 0x0U));
                CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_3,  PCI_UNC_MC_PMON_CTR_3_B, 0x0U));

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
    

#define IVB_FREEZE_UNCORE \
    if (haveLock && eventSet->regTypeMask & ~(0xF)) \
    { \
        VERBOSEPRINTREG(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<31), FREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, (1ULL<<31))); \
    }

#define IVB_UNFREEZE_UNCORE \
    if (haveLock && eventSet->regTypeMask & ~(0xF)) \
    { \
        VERBOSEPRINTREG(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<29), UNFREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, (1ULL<<29))); \
    }

#define IVB_UNFREEZE_UNCORE_AND_RESET_CTR \
    if (haveLock && eventSet->regTypeMask & ~(0xF)) \
    { \
        for (int j=UNCORE; j<NUM_UNITS; j++) \
        { \
            if (eventSet->regTypeMask & REG_TYPE_MASK(j)) \
            { \
                if ((ivybridge_box_map[j].ctrlRegister != 0x0) && (ivybridge_box_map[j].isPci)) \
                { \
                    VERBOSEPRINTPCIREG(cpu_id, ivybridge_box_map[j].device, \
                                ivybridge_box_map[j].ctrlRegister, LLU_CAST (1ULL<<1), RESET_CTR); \
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, ivybridge_box_map[j].device, \
                            ivybridge_box_map[j].ctrlRegister, (1ULL<<1))); \
                } \
                else if (ivybridge_box_map[j].ctrlRegister != 0x0)\
                { \
                    VERBOSEPRINTREG(cpu_id, ivybridge_box_map[j].ctrlRegister, LLU_CAST (1ULL<<1), RESET_CTR); \
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, ivybridge_box_map[j].ctrlRegister, (1ULL<<1))); \
                } \
            } \
        } \
        VERBOSEPRINTREG(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<29), UNFREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, (1ULL<<29))); \
    }

#define IVB_FREEZE_UNCORE_AND_RESET_CTL \
    if (haveLock && eventSet->regTypeMask & ~(0xF)) \
    { \
        VERBOSEPRINTREG(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, LLU_CAST (1ULL<<31), FREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_U_PMON_GLOBAL_CTL, (1ULL<<31))); \
        for (int j=UNCORE; j< NUM_UNITS; j++) \
        { \
            if (eventSet->regTypeMask & REG_TYPE_MASK(j)) \
            { \
                if ((ivybridge_box_map[j].ctrlRegister != 0x0) && (ivybridge_box_map[j].isPci)) \
                { \
                    VERBOSEPRINTPCIREG(cpu_id, ivybridge_box_map[j].device, \
                            ivybridge_box_map[j].ctrlRegister, LLU_CAST (1ULL<<0), RESET_PCI_CTL); \
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, ivybridge_box_map[j].device, \
                            ivybridge_box_map[j].ctrlRegister, (1ULL<<0))); \
                } \
                else if (ivybridge_box_map[j].ctrlRegister != 0x0) \
                { \
                    VERBOSEPRINTREG(cpu_id, ivybridge_box_map[j].ctrlRegister, LLU_CAST (1ULL<<0), RESET_CTL); \
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, ivybridge_box_map[j].ctrlRegister, (1ULL<<0))); \
                } \
            } \
        } \
    }

#define IVB_FREEZE_BOX(id) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(id)) \
    { \
        VERBOSEPRINTREG(cpu_id, ivybridge_box_map[id].ctrlRegister, (1<<8), FREEZE_##id); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, ivybridge_box_map[id].ctrlRegister, (1ULL<<8))); \
    }

#define IVB_UNFREEZE_BOX(id) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(id)) \
    { \
        VERBOSEPRINTREG(cpu_id, ivybridge_box_map[id].ctrlRegister, 0x0ULL, UNFREEZE_##id); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, ivybridge_box_map[id].ctrlRegister, 0x0ULL)); \
    }

#define IVB_FREEZE_PCI_BOX(id) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(id)) && \
            pci_checkDevice(ivybridge_box_map[id].device, cpu_id)) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, ivybridge_box_map[id].device, \
            ivybridge_box_map[id].ctrlRegister, (1ULL<<8), FREEZE_PCI_##id); \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, ivybridge_box_map[id].device, \
            ivybridge_box_map[id].ctrlRegister, (1ULL<<8))); \
    }

#define IVB_UNFREEZE_PCI_BOX(id) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(id)) && \
            pci_checkDevice(ivybridge_box_map[id].device, cpu_id)) \
    { \
        VERBOSEPRINTPCIREG(cpu_id, ivybridge_box_map[id].device, \
            ivybridge_box_map[id].ctrlRegister, 0x0ULL, UNFREEZE_PCI_##id); \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, ivybridge_box_map[id].device, \
            ivybridge_box_map[id].ctrlRegister, 0x0ULL)); \
    }

#define IVB_SETUP_MBOX(number) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(MBOX##number)) && \
            pci_checkDevice(ivybridge_box_map[MBOX##number].device, cpu_id)) \
    { \
        flags = (1ULL<<22)|(1ULL<<20); \
        flags |= (event->umask<<8) + event->eventId; \
        if (event->numberOfOptions > 0) \
        { \
            for (int j=0;j < event->numberOfOptions; j++) \
            { \
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_EDGE: \
                        flags |= (1ULL<<18); \
                        break; \
                    case EVENT_OPTION_THRESHOLD: \
                        flags |= ((event->options[j].value & 0x1FULL) << 24); \
                        break; \
                    default: \
                        break; \
                } \
            } \
        } \
        VERBOSEPRINTPCIREG(cpu_id, ivybridge_box_map[MBOX##number].device, reg, flags, SETUP_MBOX##number); \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, ivybridge_box_map[MBOX##number].device, reg, flags)); \
    }

#define IVB_SETUP_MBOXFIX(number) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(MBOX##number##FIX)) && \
            pci_checkDevice(ivybridge_box_map[MBOX##number].device, cpu_id)) \
    { \
        flags = (1ULL<<22); \
        VERBOSEPRINTPCIREG(cpu_id, ivybridge_box_map[MBOX##number].device, \
            PCI_UNC_MC_PMON_FIXED_CTL, flags, SETUP_MBOX##number##FIX); \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, ivybridge_box_map[MBOX##number].device, \
            PCI_UNC_MC_PMON_FIXED_CTL, flags)); \
    }

#define IVB_SETUP_SBOX(number) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(SBOX##number)) && \
            pci_checkDevice(ivybridge_box_map[SBOX##number].device, cpu_id)) \
    { \
        flags = (1ULL<<22)|(1ULL<<20); \
        flags |= (event->umask<<8) + event->eventId; \
        if (event->cfgBits != 0x0) \
        { \
            flags = (1ULL<<21); \
        } \
        if (event->numberOfOptions > 0) \
        { \
            for (int j=0;j < event->numberOfOptions; j++) \
            { \
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_EDGE: \
                        flags |= (1ULL<<18); \
                        break; \
                    case EVENT_OPTION_THRESHOLD: \
                        flags |= ((event->options[j].value & 0x1FULL) << 24); \
                        break; \
                    case EVENT_OPTION_MATCH0: \
                        VERBOSEPRINTPCIREG(cpu_id, ivybridge_box_map[SBOX##number].device, \
                                PCI_UNC_QPI_PMON_MATCH_0, flags, SETUP_SBOX##number_MATCH0); \
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, ivybridge_box_map[SBOX##number].device, \
                                PCI_UNC_QPI_PMON_MATCH_0, event->options[j].value)); \
                    case EVENT_OPTION_MATCH1: \
                        VERBOSEPRINTPCIREG(cpu_id, ivybridge_box_map[SBOX##number].device, \
                                PCI_UNC_QPI_PMON_MATCH_1, flags, SETUP_SBOX##number_MATCH1); \
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, ivybridge_box_map[SBOX##number].device, \
                                PCI_UNC_QPI_PMON_MATCH_1, event->options[j].value)); \
                    case EVENT_OPTION_MASK0: \
                        VERBOSEPRINTPCIREG(cpu_id, ivybridge_box_map[SBOX##number].device, \
                                PCI_UNC_QPI_PMON_MASK_0, flags, SETUP_SBOX##number_MASK0); \
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, ivybridge_box_map[SBOX##number].device, \
                                PCI_UNC_QPI_PMON_MASK_0, event->options[j].value)); \
                    case EVENT_OPTION_MASK1: \
                        VERBOSEPRINTPCIREG(cpu_id, ivybridge_box_map[SBOX##number].device, \
                                PCI_UNC_QPI_PMON_MASK_1, flags, SETUP_SBOX##number_MASK1); \
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, ivybridge_box_map[SBOX##number].device, \
                                PCI_UNC_QPI_PMON_MASK_1, event->options[j].value)); \
                    default: \
                        break; \
                } \
            } \
        } \
        VERBOSEPRINTPCIREG(cpu_id, ivybridge_box_map[SBOX##number].device, reg, flags, SETUP_SBOX##number); \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, ivybridge_box_map[SBOX##number].device, reg, flags)); \
    }

#define IVB_SETUP_CBOX(number) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(CBOX##number))) \
    { \
        flags = (1ULL<<22); \
        flags |= (event->umask<<8) + event->eventId; \
        if (event->numberOfOptions > 0) \
        { \
            uint64_t filter0 = 0x0ULL; \
            uint64_t filter1 = 0x0ULL; \
            int state_set = 0; \
            for (int j=0;j < event->numberOfOptions; j++) \
            { \
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_EDGE: \
                        flags |= (1ULL<<18); \
                        break; \
                    case EVENT_OPTION_THRESHOLD: \
                        flags |= ((event->options[j].value & 0x1FULL) << 24); \
                        break; \
                    case EVENT_OPTION_TID: \
                        flags |= (1<<19); \
                        filter0 |= (event->options[j].value & 0x1FULL); \
                        break; \
                    case EVENT_OPTION_STATE: \
                        filter0 |= ((event->options[j].value & 0x3FULL) << 17); \
                        state_set = 1; \
                        break; \
                    case EVENT_OPTION_NID: \
                        if (event->options[j].value >= 0x1 && event->options[j].value <= (affinityDomains.numberOfNumaDomains+1<<1)) \
                        { \
                            filter1 |= (event->options[j].value & 0xFFFFULL); \
                        } \
                        break; \
                    case EVENT_OPTION_OPCODE: \
                        filter1 |= ((event->options[j].value & 0x1FFULL) << 20); \
                        break; \
                    case EVENT_OPTION_MATCH0: \
                        filter1 |= (1ULL<<30); \
                        break; \
                    case EVENT_OPTION_MATCH1: \
                        filter1 |= (1ULL<<31); \
                        break; \
                    default: \
                        break; \
                } \
            } \
            if (state_set == 0 && event->eventId == 0x34) \
            { \
                filter0 |= (0x1FULL<<17); \
            } \
            if (filter0 != 0x0ULL) \
            { \
                VERBOSEPRINTREG(cpu_id, MSR_UNC_C##number##_PMON_BOX_FILTER, filter0, SETUP_CBOX##number##_FILTER0); \
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C##number##_PMON_BOX_FILTER, filter0)); \
            } \
            if (filter1 != 0x0ULL) \
            { \
                VERBOSEPRINTREG(cpu_id, MSR_UNC_C##number##_PMON_BOX_FILTER1, filter1, SETUP_CBOX##number##_FILTER1); \
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C##number##_PMON_BOX_FILTER1, filter1)); \
            } \
        } \
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_CBOX##number); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, flags)); \
    }

#define IVB_SETUP_UBOX \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(UBOX))) \
    { \
        flags = (1ULL<<22)|(1ULL<<20)|(1ULL<<17); \
        flags |= (event->umask<<8) + event->eventId; \
        if (event->numberOfOptions > 0) \
        { \
            for (int j=0;j < event->numberOfOptions; j++) \
            { \
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_EDGE: \
                        flags |= (1ULL<<18); \
                        break; \
                    case EVENT_OPTION_THRESHOLD: \
                        flags |= ((event->options[j].value & 0x1F) << 24); \
                        break; \
                    default: \
                        break; \
                } \
            } \
        } \
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_UBOX); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, flags)); \
    }

#define IVB_SETUP_BBOX(number) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(BBOX##number)) && \
            pci_checkDevice(ivybridge_box_map[BBOX##number].device, cpu_id)) \
    { \
        flags = (1ULL<<22)|(1ULL<<20); \
        flags |= (event->umask<<8) + event->eventId; \
        if (event->numberOfOptions > 0) \
        { \
            for (int j=0;j < event->numberOfOptions; j++) \
            { \
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_EDGE: \
                        flags |= (1ULL<<18); \
                        break; \
                    case EVENT_OPTION_THRESHOLD: \
                        flags |= ((event->options[j].value & 0x1F) << 24); \
                        break; \
                    default: \
                        break; \
                } \
            } \
        } \
        VERBOSEPRINTPCIREG(cpu_id, ivybridge_box_map[BBOX##number].device, reg, flags, SETUP_BBOX##number); \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, ivybridge_box_map[BBOX##number].device, reg, flags)); \
    }

#define IVB_SETUP_WBOX \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(WBOX))) \
    { \
        flags = (1ULL<<22)|(1ULL<<20); \
        flags |= event->eventId; \
        if (event->cfgBits != 0x0) \
        { \
            flags |= ((event->cfgBits & 0x1) << 21); \
        } \
        if (event->numberOfOptions > 0) \
        { \
            for (int j=0;j < event->numberOfOptions; j++) \
            { \
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_EDGE: \
                        flags |= (1ULL<<18); \
                        break; \
                    case EVENT_OPTION_THRESHOLD: \
                        flags |= ((event->options[j].value & 0x1F) << 24); \
                        break; \
                    case EVENT_OPTION_OCCUPANCY: \
                        flags |= ((event->options[j].value & 0x3) << 14); \
                        break; \
                    case EVENT_OPTION_OCCUPANCY_INVERT: \
                        flags |= (1ULL<<30); \
                        break; \
                    case EVENT_OPTION_OCCUPANCY_EDGE: \
                        flags |= (1ULL<<31); \
                        break; \
                    case EVENT_OPTION_MATCH0: \
                        VERBOSEPRINTREG(cpu_id, MSR_UNC_PCU_PMON_BOX_FILTER, event->options[j].value, SETUP_WBOX_FILTER); \
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_PCU_PMON_BOX_FILTER, event->options[j].value)); \
                        break; \
                    default: \
                        break; \
                } \
            } \
        } \
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_WBOX); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, flags)); \
    }


#define IVB_SETUP_BASIC_BOX(id) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(id)) && \
            pci_checkDevice(counter_map[index].device, cpu_id)) \
    { \
        flags = (1ULL<<22)|(1ULL<<20); \
        flags |= (event->umask<<8) + event->eventId; \
        if (event->numberOfOptions > 0) \
        { \
            for (int j=0;j < event->numberOfOptions; j++) \
            { \
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_EDGE: \
                        flags |= (1ULL<<18); \
                        break; \
                    case EVENT_OPTION_THRESHOLD: \
                        flags |= ((event->options[j].value & 0xFF) << 24); \
                        break; \
                    default: \
                        break; \
                } \
            } \
        } \
        VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device, reg, flags, SETUP_##id); \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, counter_map[index].device, reg, flags)); \
    }

int perfmon_setupCounterThread_ivybridge(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t fixed_flags = 0x0ULL;
    uint32_t uflags = 0x0;
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

    IVB_FREEZE_UNCORE;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterIndex index = eventSet->events[i].index;
        uint64_t reg = counter_map[index].configRegister;
        PerfmonEvent *event = &(eventSet->events[i].event);
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        switch (counter_map[index].type)
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

            case MBOX0:
                IVB_SETUP_BASIC_BOX(MBOX0);
                //IVB_SETUP_MBOX(0);
                break;
            case MBOX1:
                IVB_SETUP_BASIC_BOX(MBOX1);
                //IVB_SETUP_MBOX(1);
                break;
            case MBOX2:
                IVB_SETUP_BASIC_BOX(MBOX2);
                //IVB_SETUP_MBOX(2);
                break;
            case MBOX3:
                IVB_SETUP_BASIC_BOX(MBOX3);
                //IVB_SETUP_MBOX(3);
                break;

            case MBOX0FIX:
                IVB_SETUP_MBOXFIX(0);
                break;
            case MBOX1FIX:
                IVB_SETUP_MBOXFIX(1);
                break;
            case MBOX2FIX:
                IVB_SETUP_MBOXFIX(2);
                break;
            case MBOX3FIX:
                IVB_SETUP_MBOXFIX(3);
                break;

            case SBOX0:
                IVB_SETUP_SBOX(0);
                break;

            case SBOX1:
                IVB_SETUP_SBOX(1);
                break;

            case CBOX0:
                IVB_SETUP_CBOX(0);
                break;
            case CBOX1:
                IVB_SETUP_CBOX(1);
                break;
            case CBOX2:
                IVB_SETUP_CBOX(2);
                break;
            case CBOX3:
                IVB_SETUP_CBOX(3);
                break;
            case CBOX4:
                IVB_SETUP_CBOX(4);
                break;
            case CBOX5:
                IVB_SETUP_CBOX(5);
                break;
            case CBOX6:
                IVB_SETUP_CBOX(6);
                break;
            case CBOX7:
                IVB_SETUP_CBOX(7);
                break;
            case CBOX8:
                IVB_SETUP_CBOX(8);
                break;
            case CBOX9:
                IVB_SETUP_CBOX(9);
                break;
            case CBOX10:
                IVB_SETUP_CBOX(10);
                break;
            case CBOX11:
                IVB_SETUP_CBOX(11);
                break;
            case CBOX12:
                IVB_SETUP_CBOX(12);
                break;
            case CBOX13:
                IVB_SETUP_CBOX(13);
                break;
            case CBOX14:
                IVB_SETUP_CBOX(14);
                break;

            case UBOX:
                IVB_SETUP_UBOX;
                break;
            case UBOXFIX:
                flags = (1ULL<<22)|(1ULL<<20);
                VERBOSEPRINTREG(cpu_id, reg, LLU_CAST flags, SETUP_UBOXFIX)
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags));
                break;

            case BBOX0:
                IVB_SETUP_BASIC_BOX(BBOX0);
                break;
            case BBOX1:
                IVB_SETUP_BASIC_BOX(BBOX1);
                break;

            case WBOX:
                IVB_SETUP_WBOX;
                break;

            case PBOX:
                IVB_SETUP_BASIC_BOX(PBOX);
                break;

            case RBOX0:
                IVB_SETUP_BASIC_BOX(RBOX0);
                break;
            case RBOX1:
                IVB_SETUP_BASIC_BOX(RBOX1);
                break;

            default:
                /* should never be reached */
                break;
        }
    }
    if (fixed_flags > 0x0)
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_FIXED_CTR_CTRL, LLU_CAST fixed_flags, SETUP_FIXED)
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, fixed_flags));
    }
    return 0;
}

#define CBOX_START(NUM) \
if(haveLock) { \
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C##NUM##_PMON_BOX_CTL, uflags));  \
}

#define MBOX_START(NUM) \
if(haveLock) { \
    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_##NUM,  PCI_UNC_MC_PMON_BOX_CTL, uflags)); \
}



int perfmon_startCountersThread_ivybridge(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t power = 0x0ULL;
    uint64_t fixed_flags = 0x0ULL;
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
            switch (counter_map[index].type)
            {
                case PMC:
                    if (eventSet->regTypeMask & REG_TYPE_MASK(PMC))
                    {
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, counter1, 0x0ULL));
                        fixed_flags |= (1ULL<<(index-OFFSET_PMC));  /* enable counter */
                    }
                    break;

                case FIXED:
                    if (eventSet->regTypeMask & REG_TYPE_MASK(FIXED))
                    {
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, counter1, 0x0ULL));
                        fixed_flags |= (1ULL<<(index+32));  /* enable fixed counter */
                    }
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
                    break;
            }
        }
    }

    IVB_UNFREEZE_UNCORE_AND_RESET_CTR;
    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST fixed_flags, UNFREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, fixed_flags));
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x30000000FULL));
    }
    return 0;
}

#define CBOX_STOP(NUM) \
if(haveLock) { \
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C##NUM##_PMON_BOX_CTL, uflags));  \
    CHECK_MSR_READ_ERROR(msr_read(cpu_id, ivybridge_counter_map[index].counterRegister, &eventSet->events[i].threadCounter[thread_id].counterData));    \
}

#define MBOX_STOP(NUM) \
if(haveLock) { \
    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_0_CH_##NUM ,  PCI_UNC_MC_PMON_BOX_CTL, uflags)); \
    CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_0_CH_##NUM , ivybridge_counter_map[index].counterRegister, (uint32_t*)&counter_result)); \
    eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);  \
    CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_0_CH_##NUM , ivybridge_counter_map[index].counterRegister2, (uint32_t*)&counter_result)); \
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


#define IVB_CHECK_OVERFLOW(offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t ovf_values = 0x0ULL; \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_PERF_GLOBAL_STATUS, &ovf_values)); \
        if (ovf_values & (1<<offset)) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
        } \
    }

#define IVB_CHECK_UNCORE_OVERFLOW(id, global_offset, box_offset) \
    if (haveLock && (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)) \
    { \
        uint64_t ovf_values = 0x0ULL; \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_UNC_U_PMON_GLOBAL_STATUS, &ovf_values)); \
        if (ovf_values & (1<<global_offset)) \
        { \
            uint64_t ovf_box = 0x0ULL; \
            if (ivybridge_box_map[id].isPci) \
            { \
                CHECK_PCI_READ_ERROR(pci_read(cpu_id, ivybridge_box_map[id].device, \
                            ivybridge_box_map[id].statusRegister, (uint32_t*)&ovf_box)); \
            } \
            else \
            { \
                CHECK_MSR_READ_ERROR(msr_read(cpu_id, ivybridge_box_map[id].statusRegister, &ovf_box)); \
            } \
            if (ovf_box & (1<<box_offset)) \
            { \
                eventSet->events[i].threadCounter[thread_id].overflows++; \
            } \
        } \
    }

// Read MSR counter register
#define IVB_READ_BOX(id, reg1) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) \
    { \
        VERBOSEPRINTREG(cpu_id, reg1, LLU_CAST counter_result, READ_BOX_##id) \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, reg1, &counter_result)); \
    }

#define IVB_READ_PCI_BOX(id, reg1, reg2) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id))) && \
                    pci_checkDevice(ivybridge_box_map[id].device, cpu_id)) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_PCI_READ_ERROR(pci_read(cpu_id, ivybridge_box_map[id].device, reg1, (uint32_t*)&tmp)); \
        counter_result = (tmp<<32); \
        CHECK_PCI_READ_ERROR(pci_read(cpu_id, ivybridge_box_map[id].device, reg2, (uint32_t*)&tmp)); \
        counter_result += tmp; \
        VERBOSEPRINTPCIREG(cpu_id, ivybridge_box_map[id].device, reg1, LLU_CAST counter_result, READ_PCI_BOX_##id) \
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

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    IVB_FREEZE_UNCORE_AND_RESET_CTL;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        counter_result= 0x0ULL;
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;
            switch (counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter1, &counter_result));
                    IVB_CHECK_OVERFLOW(index-cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;

                case FIXED:
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter1, &counter_result));
                    IVB_CHECK_OVERFLOW(index+32);
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
                        CHECK_TEMP_READ_ERROR(thermal_read(cpu_id, (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case MBOX0:
                    IVB_READ_PCI_BOX(MBOX0, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(MBOX0, 20, getCounterTypeOffset(index)+1)
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case MBOX1:
                    IVB_READ_PCI_BOX(MBOX1, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(MBOX0, 20, getCounterTypeOffset(index)+1)
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case MBOX2:
                    IVB_READ_PCI_BOX(MBOX2, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(MBOX2, 20, getCounterTypeOffset(index)+1)
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case MBOX3:
                    IVB_READ_PCI_BOX(MBOX3, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(MBOX3, 20, getCounterTypeOffset(index)+1)
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case MBOX0FIX:
                    IVB_READ_PCI_BOX(MBOX0FIX, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(MBOX0, 20, 0);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case MBOX1FIX:
                    IVB_READ_PCI_BOX(MBOX1FIX, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(MBOX1, 20, 0);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case MBOX2FIX:
                    IVB_READ_PCI_BOX(MBOX2FIX, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(MBOX2, 20, 0);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case MBOX3FIX:
                    IVB_READ_PCI_BOX(MBOX3FIX, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(MBOX3, 20, 0);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case SBOX0:
                    IVB_READ_PCI_BOX(SBOX0, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(SBOX0, 22, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case SBOX1:
                    IVB_READ_PCI_BOX(SBOX1, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(SBOX1, 23, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case SBOX0FIX:
                case SBOX1FIX:
                case SBOX2FIX:
                    CHECK_PCI_READ_ERROR(pci_read(cpu_id, counter_map[index].device,
                                    counter1, (uint32_t*)&counter_result));
                    VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device, counter1, LLU_CAST counter_result, READ_SBOX_FIXED)
                    switch (extractBitField(counter_result,3,0))
                    {
                        case 0x2:
                            counter_result = 5600000000ULL;
                            break;
                        case 0x3:
                            counter_result = 6400000000ULL;
                            break;
                        case 0x4:
                            counter_result = 7200000000ULL;
                            break;
                        case 0x5:
                            counter_result = 8000000000ULL;
                            break;
                        case 0x6:
                            counter_result = 8800000000ULL;
                            break;
                        case 0x7:
                            counter_result = 9600000000ULL;
                            break;
                        default:
                            counter_result = 0x0ULL;
                            break;
                    }
                    VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device, counter1, LLU_CAST counter_result, READ_SBOX_FIXED_REAL)
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX0:
                    IVB_READ_BOX(CBOX0, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX0, 3, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX1:
                    IVB_READ_BOX(CBOX1, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX1, 4, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX2:
                    IVB_READ_BOX(CBOX2, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX2, 5, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX3:
                    IVB_READ_BOX(CBOX3, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX3, 6, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX4:
                    IVB_READ_BOX(CBOX4, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX4, 7, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX5:
                    IVB_READ_BOX(CBOX5, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX5, 8, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX6:
                    IVB_READ_BOX(CBOX6, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX6, 9, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX7:
                    IVB_READ_BOX(CBOX7, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX7, 10, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX8:
                    IVB_READ_BOX(CBOX8, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX8, 11, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX9:
                    IVB_READ_BOX(CBOX9, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX9, 12, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX10:
                    IVB_READ_BOX(CBOX10, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX10, 13, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX11:
                    IVB_READ_BOX(CBOX11, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX11, 14, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX12:
                    IVB_READ_BOX(CBOX12, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX12, 15, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX13:
                    IVB_READ_BOX(CBOX13, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX13, 16, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX14:
                    IVB_READ_BOX(CBOX14, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX14, 17, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case UBOX:
                    IVB_READ_BOX(UBOX, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(UBOX, 1, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case UBOXFIX:
                    IVB_READ_BOX(UBOX, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(UBOX, 0, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case BBOX0:
                    IVB_READ_PCI_BOX(BBOX0, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(BBOX0, 18, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case BBOX1:
                    IVB_READ_PCI_BOX(BBOX1, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(BBOX1, 19, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case WBOX:
                    IVB_READ_PCI_BOX(WBOX, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(WBOX, 2, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;

                case PBOX:
                    IVB_READ_PCI_BOX(PBOX, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(PBOX, 26, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;

                case RBOX0:
                    IVB_READ_PCI_BOX(RBOX0, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(RBOX0, 24, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                case RBOX1:
                    IVB_READ_PCI_BOX(RBOX1, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(RBOX1, 25, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;

                default:
                    /* should never be reached */
                    break;
            }
        }
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
    }

    //CHECK_MSR_READ_ERROR(msr_read(cpu_id,MSR_PERF_GLOBAL_STATUS, &flags));
    //    printf ("Status: 0x%llX \n", LLU_CAST flags);
    //if ( (flags & 0x3) || (flags & (0x3ULL<<32)) ) 
    //{
        //printf ("Overflow occured \n");
    //}
    return 0;
}

int perfmon_readCountersThread_ivybridge(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t counter_result = 0x0ULL;
    uint64_t pmc_flags = 0x0ULL;
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_PERF_GLOBAL_CTRL, &pmc_flags));
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    IVB_FREEZE_UNCORE;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        counter_result = 0x0ULL;
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;
            switch (counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter1, &counter_result));
                    IVB_CHECK_OVERFLOW(index-OFFSET_PMC);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC)
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;

                case FIXED:
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter1, &counter_result));
                    IVB_CHECK_OVERFLOW(index+32);
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
                        CHECK_TEMP_READ_ERROR(thermal_read(cpu_id, (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case MBOX0:
                    IVB_READ_PCI_BOX(MBOX0, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(MBOX0, 20, getCounterTypeOffset(index)+1)
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case MBOX1:
                    IVB_READ_PCI_BOX(MBOX1, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(MBOX0, 20, getCounterTypeOffset(index)+1)
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case MBOX2:
                    IVB_READ_PCI_BOX(MBOX2, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(MBOX2, 20, getCounterTypeOffset(index)+1)
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case MBOX3:
                    IVB_READ_PCI_BOX(MBOX3, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(MBOX3, 20, getCounterTypeOffset(index)+1)
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case MBOX0FIX:
                    IVB_READ_PCI_BOX(MBOX0FIX, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(MBOX0, 20, 0);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case MBOX1FIX:
                    IVB_READ_PCI_BOX(MBOX1FIX, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(MBOX1, 20, 0);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case MBOX2FIX:
                    IVB_READ_PCI_BOX(MBOX2FIX, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(MBOX2, 20, 0);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case MBOX3FIX:
                    IVB_READ_PCI_BOX(MBOX3FIX, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(MBOX3, 20, 0);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case SBOX0:
                    IVB_READ_PCI_BOX(SBOX0, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(SBOX0, 22, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case SBOX1:
                    IVB_READ_PCI_BOX(SBOX1, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(SBOX1, 23, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case SBOX0FIX:
                case SBOX1FIX:
                case SBOX2FIX:
                    CHECK_PCI_READ_ERROR(pci_read(cpu_id, counter_map[index].device,
                                    counter1, (uint32_t*)&counter_result));
                    VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device, counter1,
                                                LLU_CAST counter_result, READ_SBOX_FIXED)
                    switch (extractBitField(counter_result,3,0))
                    {
                        case 0x2:
                            counter_result = 5600000000ULL;
                            break;
                        case 0x3:
                            counter_result = 6400000000ULL;
                            break;
                        case 0x4:
                            counter_result = 7200000000ULL;
                            break;
                        case 0x5:
                            counter_result = 8000000000ULL;
                            break;
                        case 0x6:
                            counter_result = 8800000000ULL;
                            break;
                        case 0x7:
                            counter_result = 9600000000ULL;
                            break;
                        default:
                            counter_result = 0x0ULL;
                            break;
                    }
                    VERBOSEPRINTPCIREG(cpu_id, counter_map[index].device, counter1,
                                                LLU_CAST counter_result, READ_SBOX_FIXED_REAL)
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX0:
                    IVB_READ_BOX(CBOX0, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX0, 3, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX1:
                    IVB_READ_BOX(CBOX1, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX1, 4, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX2:
                    IVB_READ_BOX(CBOX2, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX2, 5, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX3:
                    IVB_READ_BOX(CBOX3, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX3, 6, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX4:
                    IVB_READ_BOX(CBOX4, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX4, 7, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX5:
                    IVB_READ_BOX(CBOX5, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX5, 8, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX6:
                    IVB_READ_BOX(CBOX6, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX6, 9, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX7:
                    IVB_READ_BOX(CBOX7, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX7, 10, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX8:
                    IVB_READ_BOX(CBOX8, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX8, 11, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX9:
                    IVB_READ_BOX(CBOX9, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX9, 12, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX10:
                    IVB_READ_BOX(CBOX10, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX10, 13, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX11:
                    IVB_READ_BOX(CBOX11, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX11, 14, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX12:
                    IVB_READ_BOX(CBOX12, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX12, 15, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX13:
                    IVB_READ_BOX(CBOX13, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX13, 16, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case CBOX14:
                    IVB_READ_BOX(CBOX14, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(CBOX14, 17, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case UBOX:
                    IVB_READ_BOX(UBOX, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(UBOX, 1, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case UBOXFIX:
                    IVB_READ_BOX(UBOX, counter1);
                    IVB_CHECK_UNCORE_OVERFLOW(UBOX, 0, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case BBOX0:
                    IVB_READ_PCI_BOX(BBOX0, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(BBOX0, 18, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case BBOX1:
                    IVB_READ_PCI_BOX(BBOX1, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(BBOX1, 19, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case WBOX:
                    IVB_READ_PCI_BOX(WBOX, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(WBOX, 2, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case PBOX:
                    IVB_READ_PCI_BOX(PBOX, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(PBOX, 26, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case RBOX0:
                    IVB_READ_PCI_BOX(RBOX0, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(RBOX0, 24, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;
                case RBOX1:
                    IVB_READ_PCI_BOX(RBOX1, counter1, counter2);
                    IVB_CHECK_UNCORE_OVERFLOW(RBOX1, 25, getCounterTypeOffset(index));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                default:
                    break;
            }
        }
    }

    IVB_UNFREEZE_UNCORE;
    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, pmc_flags));
    }
    return 0;
}

