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

#include <perfmon_sandybridge_events.h>
#include <perfmon_sandybridge_counters.h>
#include <error.h>
#include <affinity.h>

static int perfmon_numCountersSandybridge = NUM_COUNTERS_SANDYBRIDGE;
static int perfmon_numArchEventsSandybridge = NUM_ARCH_EVENTS_SANDYBRIDGE;

#define OFFSET_PMC 3

int perfmon_init_sandybridge(int cpu_id)
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
        if ( cpuid_info.model == SANDYBRIDGE_EP )
        {
            /* Only root can access pci address space in direct mode */
            if ((accessClient_mode == DAEMON_AM_ACCESS_D) || (getuid() == 0))
            {
                uint32_t uflags = 0x10100U; /* enable freeze (bit 16), freeze (bit 8) */
                uint32_t confflags = (1<<22); /* enable flag */
                uint32_t fixedflags = (1<<19); /* reset fixed counter */
                if (pci_checkDevice(PCI_IMC_DEVICE_CH_0, cpu_id))
                {
                 
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTL_0, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTL_1, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTL_2, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTL_3, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_FIXED_CTL, fixedflags));
                    /* iMC counters need to be manually reset to zero */
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_0_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_0_B, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_1_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_1_B, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_2_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_2_B, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_3_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_CTR_3_B, 0x0U));
                }
                if (pci_checkDevice(PCI_IMC_DEVICE_CH_1, cpu_id))
                {
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTL_0, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTL_1, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTL_2, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTL_3, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_FIXED_CTL, fixedflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_0_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_0_B, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_1_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_1_B, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_2_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_2_B, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_3_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_CTR_3_B, 0x0U));
                }
                if (pci_checkDevice(PCI_IMC_DEVICE_CH_2, cpu_id))
                {
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTL_0, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTL_1, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTL_2, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTL_3, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_FIXED_CTL, fixedflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_0_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_0_B, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_1_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_1_B, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_2_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_2_B, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_3_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_CTR_3_B, 0x0U));
                }
                if (pci_checkDevice(PCI_IMC_DEVICE_CH_3, cpu_id))
                {
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTL_0, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTL_1, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTL_2, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTL_3, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_FIXED_CTL, fixedflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_0_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_0_B, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_1_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_1_B, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_2_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_2_B, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_3_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_CTR_3_B, 0x0U));
                }

                /* BBOX / Home Agent */
                if (pci_checkDevice(PCI_HA_DEVICE, cpu_id))
                {
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_BOX_CTL, uflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_CTL_0, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_CTL_1, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_CTL_2, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_CTL_3, confflags));
                    /* Manual reset of BBOX counters */
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_CTR_0_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_CTR_0_B, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_CTR_1_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_CTR_1_B, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_CTR_2_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_CTR_2_B, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_CTR_3_A, 0x0U));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,  PCI_UNC_HA_PMON_CTR_3_B, 0x0U));
                }

                /* UBOX */
                /* Enable (bit 22) and reset (bit 17) counter */
                uflags = 0x0U;
                uflags = (1<<17)|(1<<22);
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_U_PMON_CTL0, uflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_U_PMON_CTL1, uflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_U_UCLK_FIXED_CTL, 0x0U));

                /* CBOX */
                /*enable freeze (bit 16), freeze (bit 8), reset counter (bit 1), reset control (bit 0) */
                uflags = 0x10103U; 
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C0_PMON_BOX_CTL, uflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C1_PMON_BOX_CTL, uflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C2_PMON_BOX_CTL, uflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C3_PMON_BOX_CTL, uflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C4_PMON_BOX_CTL, uflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C5_PMON_BOX_CTL, uflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C6_PMON_BOX_CTL, uflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C7_PMON_BOX_CTL, uflags));

                /* FIXME: Not yet tested/ working due to BIOS issues on test
                 * machines */
                /* QPI registers can be zeroed with single write */
                if (pci_checkDevice(PCI_QPI_DEVICE_PORT_0, cpu_id))
                {
                    /*enable freeze (bit 16), freeze (bit 8), reset counter (bit 1), reset control (bit 0) */
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_BOX_CTL, uflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_CTL_0, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_CTL_1, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_CTL_2, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_CTL_3, confflags));
                }
                if (pci_checkDevice(PCI_QPI_DEVICE_PORT_1, cpu_id))
                {
                    /*enable freeze (bit 16), freeze (bit 8), reset counter (bit 1), reset control (bit 0) */ 
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_BOX_CTL, uflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_CTL_0, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_CTL_1, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_CTL_2, confflags));
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_CTL_3, confflags));
                }
                
                /* WBOX or Power Control */
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_PCU_PMON_BOX_CTL1, uflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_PCU_PMON_CTL0, confflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_PCU_PMON_CTL1, confflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_PCU_PMON_CTL2, confflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_PCU_PMON_CTL3, confflags));
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_PCU_PMON_BOX_FILTER, 0x0U));
            }
        }
    }
//    lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    return 0;
}

#define BOX_GATE_IVYB(channel,label) \
    if(haveLock) { \
        CHECK_PCI_READ_ERROR(pci_read(cpu_id, channel, reg, &uflags));  \
        uflags &= ~(0xFFFFU);  \
        uflags |= (event->umask<<8) + event->eventId;  \
        if (event->numberOfOptions > 0) \
        { \
            for(int j=0;j<event->numberOfOptions;j++) \
            { \
                switch(event->options[j].type) \
                { \
                    case EVENT_OPTION_THRESHOLD: \
                        uflags |= ((event->options[j].value << 24) & 0xFF000000); \
                        break; \
                    case EVENT_OPTION_INVERT: \
                        uflags |= (1<<23); \
                        break; \
                    case EVENT_OPTION_EDGE: \
                        uflags |= (1<<18); \
                        break; \
                } \
            } \
        } \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, channel,  reg, uflags));  \
    }

uint32_t add_fixed_config(RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t ret = 0x0U;
    for(j=0;j<event->numberOfOptions;j++)
    {
        switch (event->options[j].type)
        {
            case EVENT_OPTION_COUNT_KERNEL:
                if (index == 0)
                {
                    ret |= (1<<0);
                }
                else if (index == 1)
                {
                    ret |= (1<<4);
                }
                else if (index == 2)
                {
                    ret |= (1<<8);
                }
                break;
            default:
                break;
        }
    }
    return ret;
}

uint32_t add_pmc_config(RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t ret = 0x0U;
    for(j=0;j<event->numberOfOptions;j++)
    {
        switch (event->options[j].type)
        {
            case EVENT_OPTION_EDGE:
                ret |= (1<<18);
                break;
            case EVENT_OPTION_COUNT_KERNEL:
                ret |= (1<<17);
                break;
            default:
                break;
        }
    }
    return ret;
}


uint32_t add_cbox_filter(uint64_t eventMask, EventOptionType type, uint32_t value)
{
    uint32_t ret = 0x0;
    switch (type)
    {
        case EVENT_OPTION_OPCODE:
            if ((value == 0x180) ||
                (value == 0x181) ||
                (value == 0x182) ||
                (value == 0x187) ||
                (value == 0x18C) ||
                (value == 0x18D) ||
                (value == 0x190) ||
                (value == 0x191) ||
                (value == 0x192) ||
                (value == 0x194) ||
                (value == 0x195) ||
                (value == 0x19C) ||
                (value == 0x19E) ||
                (value == 0x1C4) ||
                (value == 0x1C5) ||
                (value == 0x1C8) ||
                (value == 0x1E4) ||
                (value == 0x1E5) ||
                (value == 0x1E6))
            {
                ret |= ((value << 23) & 0xFF800000);
            }
            else
            {
                ERROR_PRINT(Invalid value 0x%x for opcode option, value);
            }
            break;
        case EVENT_OPTION_STATE:
            if (value & 0x1F)
            {
                ret |= ((value << 18) & 0x7C0000);
            }
            else
            {
                ERROR_PRINT(Invalid value 0x%x for state option, value);
            }
            break;
        case EVENT_OPTION_NID:
            if (value >= 0x1 && value <= (affinityDomains.numberOfNumaDomains+1<<1))
            {
                ret |= ((value << 10) & 0x3FC00);
            }
            else
            {
                ERROR_PRINT(Invalid value 0x%x for node id option, value);
            }
            break;
        case EVENT_OPTION_TID:
            if (value <= 0xF)
            {
                ret |= ((value << 0) & 0x1F);
            }
            else
            {
                ERROR_PRINT(Invalid value 0x%x for thread id option, value);
            }
            break;
        default:
            printf("DEFAULT SWITCH\n");
            break;
    }
    return ret;
}

uint32_t add_cbox_config(RegisterIndex index, EventOptionType type, uint32_t value)
{
    uint32_t ret = 0x0U;
    switch (type)
    {
        case EVENT_OPTION_TID:
            ret |= (1<<19);
            break;
        case EVENT_OPTION_EDGE:
            ret |= (1<<18);
            break;
        case EVENT_OPTION_THRESHOLD:
            ret |= ((value << 24) & 0xFF000000);
            break;
        default:
            break;
    }
    return ret;
}

uint32_t add_ubox_config(RegisterIndex index, EventOptionType type, uint32_t value)
{
    uint32_t ret = 0x0U;
    switch (type)
    {
        case EVENT_OPTION_EDGE:
            ret |= (1<<18);
            break;
        case EVENT_OPTION_THRESHOLD:
            ret |= ((value << 24) & 0x1F000000);
            break;
        default:
            break;
    }
    return ret;
}

uint32_t add_bbox_config(RegisterIndex index, EventOptionType type, uint32_t value)
{
    uint32_t ret = 0x0U;
    switch (type)
    {
        case EVENT_OPTION_EDGE:
            ret |= (1<<18);
            break;
        case EVENT_OPTION_INVERT:
            ret |= (1<<23);
            break;
        case EVENT_OPTION_THRESHOLD:
            ret |= ((value << 24) & 0xFF000000);
            break;
        default:
            break;
    }
    return ret;
}

int add_bbox_match(RegisterIndex index, EventOptionType type, uint32_t value, uint32_t* opcodematch, uint32_t* addr0match, uint32_t* addr1match)
{
    int ret = 0;
    *opcodematch = 0x0U;
    *addr0match = 0x0U;
    *addr1match = 0x0U;

    switch (type)
    {
        case EVENT_OPTION_OPCODE:
            *opcodematch |= (value & 0x3F);
            ret = 1;
            break;
        case EVENT_OPTION_ADDR:
            *addr0match |= (extractBitField(value,0,26))<<5;
            *addr1match |= extractBitField(value,32,14) & 0x3FFF;
            ret = 2;
            break;
        default:
            break;
    }
    return ret;
}

uint32_t add_wbox_config(RegisterIndex index, EventOptionType type, uint32_t value)
{
    uint32_t ret = 0x0U;
    switch (type)
    {
        case EVENT_OPTION_EDGE:
            ret |= (1<<18);
            break;
        case EVENT_OPTION_INVERT:
            ret |= (1<<23);
            break;
        case EVENT_OPTION_THRESHOLD:
            ret |= ((value << 24) & 0x1F000000);
            break;
        default:
            break;
    }
    return ret;
}

uint32_t add_wbox_filter(RegisterIndex index, EventOptionType type, uint32_t value)
{
    if (type == EVENT_OPTION_OCCUPANCY)
    {
        return value;
    }
    return 0x0u;
}

int perfmon_setupCounterThread_sandybridge(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int i, j;
    int haveLock = 0;
    uint64_t flags;
    uint32_t uflags;

    int cpu_id = groupSet->threads[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    for (i=0;i < eventSet->numberOfEvents;i++)
    {
        PerfmonEvent *event = &(eventSet->events[i].event);
        RegisterIndex index = eventSet->events[i].index;
        uint64_t reg = sandybridge_counter_map[index].configRegister;
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        switch (sandybridge_counter_map[index].type)
        {
            case PMC:
                CHECK_MSR_READ_ERROR(msr_read(cpu_id,reg, &flags));
                flags &= ~(0xFFFFU);   /* clear lower 16bits */

                /* Intel with standard 8 bit event mask: [7:0] */
                flags |= (event->umask<<8) + event->eventId;

                if (event->cfgBits != 0) /* set custom cfg and cmask */
                {
                    flags &= ~(0xFFFFU<<16);  /* clear upper 16bits */
                    flags |= ((event->cmask<<8) + event->cfgBits)<<16;
                }

                if (event->numberOfOptions > 0)
                {
                    flags |= add_pmc_config(index, event);
                }

                DEBUG_PRINT(DEBUGLEV_DETAIL, Setting up reg 0x%x with value 0x%x, reg, flags);
                
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags));
                break;

            case FIXED:
                CHECK_MSR_READ_ERROR(msr_read(cpu_id,reg, &flags));
                if (event->numberOfOptions > 0)
                {
                    flags |= add_fixed_config(index,event);
                }
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags));
                break;

            case POWER:
                break;

            case MBOX0:
                BOX_GATE_IVYB(PCI_IMC_DEVICE_CH_0,MBOX0);
                break;

            case MBOX1:
                BOX_GATE_IVYB(PCI_IMC_DEVICE_CH_1,MBOX1);
                break;

            case MBOX2:
                BOX_GATE_IVYB(PCI_IMC_DEVICE_CH_2,MBOX2);
                break;

            case MBOX3:
                BOX_GATE_IVYB(PCI_IMC_DEVICE_CH_3,MBOX3);
                break;

            case CBOX0:
            case CBOX1:
            case CBOX2:
            case CBOX3:
            case CBOX4:
            case CBOX5:
            case CBOX6:
            case CBOX7:
                CHECK_MSR_READ_ERROR(msr_read(cpu_id, reg, (uint64_t*)&uflags));
                uflags &= ~(0xFFFFU);
                uflags |= (event->umask<<8) + event->eventId;
                uflags |= (1<<22);

                if (event->numberOfOptions > 0)
                {
                    uint32_t optflags = 0x0U;
                    for (j=0;j< event->numberOfOptions; j++)
                    {
                        optflags |= add_cbox_filter(event->optionMask, event->options[j].type, event->options[j].value);
                        uflags |= add_cbox_config(index, event->options[j].type, event->options[j].value);
                    }
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_C0_PMON_BOX_FILTER, optflags));
                }

                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, (uint64_t)uflags));
                break;

            case UBOX:
                CHECK_MSR_READ_ERROR(msr_read(cpu_id, reg, (uint64_t*)&uflags));
                uflags &= ~(0xFFFFU);
                uflags |= (event->umask<<8) + event->eventId;

                if (event->numberOfOptions > 0)
                {
                    for (j=0;j< event->numberOfOptions; j++)
                    {
                        uflags |= add_ubox_config(index, event->options[j].type, event->options[j].value);
                    }
                }

                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, uflags));
                break;
                
            case UBOXFIX:
                break;

            case SBOX0:

                /* CTO_COUNT event requires programming of MATCH/MASK registers */
                if (event->eventId != 0x38)
                {
                    BOX_GATE_IVYB(PCI_QPI_DEVICE_PORT_0,SBOX0);
                }
                else
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

                break;

            case SBOX1:

                /* CTO_COUNT event requires programming of MATCH/MASK registers */
                if (event->eventId != 0x38)
                {
                    BOX_GATE_IVYB(PCI_QPI_DEVICE_PORT_0,SBOX0);
                }
                else
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
                break;

            case BBOX0:
                if(haveLock) {
                    uint32_t opcode;
                    uint32_t addr0;
                    uint32_t addr1;
                    CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_HA_DEVICE, reg, &uflags));
                    uflags &= ~(0xFFFFU);
                    uflags |= (event->umask<<8) + event->eventId;
                    if (event->numberOfOptions > 0)
                    {
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_HA_DEVICE, PCI_UNC_HA_PMON_OPCODEMATCH, &opcode));
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_HA_DEVICE, PCI_UNC_HA_PMON_ADDRMATCH0, &addr0));
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_HA_DEVICE, PCI_UNC_HA_PMON_ADDRMATCH1, &addr1));
                        for (j=0;j<event->numberOfOptions;j++)
                        {
                            uflags |= add_bbox_config(index, event->options[j].type, event->options[j].value);
                            add_bbox_match(index, event->options[j].type, event->options[j].value, &opcode, &addr0, &addr1);
                        }
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE, 
                                            PCI_UNC_HA_PMON_OPCODEMATCH, opcode));
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE, 
                                            PCI_UNC_HA_PMON_ADDRMATCH0, addr0));
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE, 
                                            PCI_UNC_HA_PMON_ADDRMATCH1, addr1));
                    }
                    CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_HA_DEVICE,  reg, uflags));
                }
                break;

            case WBOX:
                if (haveLock)
                {
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, reg, (uint64_t*)&uflags));
                    uflags &= ~(0xFFFFU);
                    uflags |= event->eventId & 0xFF;
                    if (event->numberOfOptions > 0)
                    {
                        uint32_t filter = 0x0U;
                        CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_UNC_PCU_PMON_BOX_FILTER, (uint64_t*)&filter));
                        for(j=0;j<event->numberOfOptions;j++)
                        {
                            uflags |= add_wbox_config(index, event->options[j].type, event->options[j].value);
                            filter |= add_wbox_filter(index, event->options[j].type, event->options[j].value);
                        }
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_UNC_PCU_PMON_BOX_FILTER, (uint64_t)filter));
                    }
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, uflags));
                }

            default:
                /* should never be reached */
                break;
        }
    }
    return 0;
}


#define UNFREEZE_BOX(reg) \
    if(haveLock) { \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, reg, (uint64_t*)&uflags)); \
        uflags &= 0xFFFFFEFF; \
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Start MSR box with flags 0x%llX to register 0x%llX, \
            LLU_CAST uflags, LLU_CAST reg); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, (uint64_t)uflags)); \
    }

#define UNFREEZE_PCI_BOX(dev, reg) \
    if (haveLock) { \
        CHECK_PCI_READ_ERROR(pci_read(cpu_id, dev, reg, &uflags)); \
        uflags &= 0xFFFFFEFF; \
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Start PCI box with flags 0x%llX to register 0x%llX at device %d, \
            LLU_CAST uflags, LLU_CAST reg, dev); \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, dev, reg, uflags)); \
    }

int perfmon_startCountersThread_sandybridge(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint32_t tmp;
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
            switch (sandybridge_counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, sandybridge_counter_map[index].counterRegister, 0x0ULL));
                    flags |= (1<<(index-OFFSET_PMC));  /* enable counter */
                    break;

                case FIXED:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, sandybridge_counter_map[index].counterRegister, 0x0ULL));
                    flags |= (1ULL<<(index+32));  /* enable fixed counter */
                    break;

                case POWER:
                    if(haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, sandybridge_counter_map[index].counterRegister, &tmp));
                        eventSet->events[i].threadCounter[thread_id].counterData = tmp;
                    }
                    break;

                case MBOX0:
                    if(haveLock)
                    {
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                    }
                    break;

                case MBOX1:
                    if(haveLock)
                    {
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                    }
                    break;

                case MBOX2:
                    if(haveLock)
                    {
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                    }
                    break;

                case MBOX3:
                    if(haveLock)
                    {
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                    }
                    break;

                case MBOXFIX:
                    if(haveLock)
                    {
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_FIXED_CTL, 0x48000UL));
                    }
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
                    UNFREEZE_BOX(MSR_UNC_C0_PMON_BOX_CTL);
                    break;
                case CBOX1:
                    UNFREEZE_BOX(MSR_UNC_C1_PMON_BOX_CTL);
                    break;
                case CBOX2:
                    UNFREEZE_BOX(MSR_UNC_C2_PMON_BOX_CTL);
                    break;
                case CBOX3:
                    UNFREEZE_BOX(MSR_UNC_C3_PMON_BOX_CTL);
                    break;
                case CBOX4:
                    UNFREEZE_BOX(MSR_UNC_C4_PMON_BOX_CTL);
                    break;
                case CBOX5:
                    UNFREEZE_BOX(MSR_UNC_C5_PMON_BOX_CTL);
                    break;
                case CBOX6:
                    UNFREEZE_BOX(MSR_UNC_C6_PMON_BOX_CTL);
                    break;
                case CBOX7:
                    UNFREEZE_BOX(MSR_UNC_C7_PMON_BOX_CTL);
                    break;

                case UBOX:
                case UBOXFIX:
                    if(haveLock)
                    {
                        uflags = 0x0U;
                        uflags |= (1<<22);
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, sandybridge_counter_map[index].configRegister, uflags));
                    }
                    break;

                case BBOX0:
                    UNFREEZE_PCI_BOX(PCI_HA_DEVICE, PCI_UNC_HA_PMON_BOX_CTL);
                    break;

                default:
                    /* should never be reached */
                    break;
            }
        }
    }

    DEBUG_PLAIN_PRINT(DEBUGLEV_DETAIL, Start thread-local MSR counters);
    DEBUG_PRINT(DEBUGLEV_DETAIL, Write flags 0x%llX to register 0x%X ,
                    MSR_PERF_GLOBAL_CTRL, LLU_CAST flags);
    DEBUG_PRINT(DEBUGLEV_DETAIL, Write flags 0x%llX to register 0x%X ,
                    MSR_UNCORE_PERF_GLOBAL_CTRL, LLU_CAST uflags);

    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, flags));
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x30000000FULL));
    return 0;
}

#define READ_BOX() \
    if(haveLock) \
    { \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, sandybridge_counter_map[index].counterRegister, &counter_result)); \
        eventSet->events[i].threadCounter[thread_id].counterData = counter_result; \
    }

#define STOP_AND_READ_BOX(stopreg) \
    if(haveLock) \
    { \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, stopreg, uflags)) \
    } \
    READ_BOX()
    
#define READ_PCI_BOX(dev, reg1, reg2) \
    if(haveLock) \
    { \
        CHECK_PCI_READ_ERROR(pci_read(cpu_id, dev, reg1, (uint32_t*)&counter_result)); \
        eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32); \
        CHECK_PCI_READ_ERROR(pci_read(cpu_id, dev, reg2, (uint32_t*)&counter_result)); \
        eventSet->events[i].threadCounter[thread_id].counterData += counter_result; \
    }

#define STOP_AND_READ_PCI_BOX(dev, configreg, reg1, reg2) \
    if (haveLock) \
    {  \
        CHECK_PCI_READ_ERROR(pci_read(cpu_id, dev, configreg, &uflags)); \
        uflags &= ~(1<<22); \
        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, dev, configreg, uflags)); \
    } \
    READ_PCI_BOX(dev, reg1, reg2)

int perfmon_stopCountersThread_sandybridge(int thread_id, PerfmonEventSet* eventSet)
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
            RegisterIndex index = eventSet->events[i].index;
            switch (sandybridge_counter_map[index].type)
            {
                case PMC:

                case FIXED:
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, sandybridge_counter_map[index].counterRegister, &counter_result));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    break;

                case POWER:
                    if(haveLock)
                    {
                        CHECK_POWER_READ_ERROR(power_read(cpu_id, sandybridge_counter_map[index].counterRegister, 
                                (uint32_t*)&counter_result));
                        if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                        {
                            fprintf(stderr,"Overflow in power status register 0x%x, assuming single overflow\n",
                                                sandybridge_counter_map[index].counterRegister);
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

                case MBOX0:
                    if(haveLock)
                    {
                        uflags = 0x10100UL;
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_0, 
                                sandybridge_counter_map[index].counterRegister,
                                (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_0, 
                                sandybridge_counter_map[index].counterRegister2,
                                (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData += counter_result;
                    }
                    break;

                case MBOX1:
                    if(haveLock)
                    {
                        uflags = 0x10100UL;
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_1, 
                                sandybridge_counter_map[index].counterRegister,
                                (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_1, 
                                sandybridge_counter_map[index].counterRegister2,
                                (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData += counter_result;
                    }
                    break;

                case MBOX2:
                    if(haveLock)
                    {
                        uflags = 0x10100UL;
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_2, 
                                sandybridge_counter_map[index].counterRegister,
                                (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_2, 
                                sandybridge_counter_map[index].counterRegister2,
                                (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData += counter_result;
                    }
                    break;

                case MBOX3:
                    if(haveLock)
                    {
                        uflags = 0x10100UL;
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_BOX_CTL, uflags));
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_3, 
                                sandybridge_counter_map[index].counterRegister,
                                (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_3, 
                                sandybridge_counter_map[index].counterRegister2,
                                (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData += counter_result;
                    }
                    break;

                case MBOXFIX:
                    if(haveLock)
                    {
                        uflags = 0x10100UL;
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_FIXED_CTL, uflags));
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_0, 
                                sandybridge_counter_map[index].counterRegister,
                                (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_0, 
                                sandybridge_counter_map[index].counterRegister2,
                                (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData += counter_result;
                    }
                    break;

                case SBOX0:
                    if(haveLock)
                    {
                        uflags = 0x10100UL;
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_0,  PCI_UNC_QPI_PMON_BOX_CTL, uflags));
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_QPI_DEVICE_PORT_0, 
                                sandybridge_counter_map[index].counterRegister,
                                (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_QPI_DEVICE_PORT_0, 
                                sandybridge_counter_map[index].counterRegister2,
                                (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData += counter_result;
                    }
                    break;

                case SBOX1:
                    if(haveLock)
                    {
                        uflags = 0x10100UL;
                        CHECK_PCI_WRITE_ERROR(pci_write(cpu_id, PCI_QPI_DEVICE_PORT_1,  PCI_UNC_QPI_PMON_BOX_CTL, uflags));
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_QPI_DEVICE_PORT_1, 
                                sandybridge_counter_map[index].counterRegister,
                                (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);
                        CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_QPI_DEVICE_PORT_1, 
                                sandybridge_counter_map[index].counterRegister2,
                                (uint32_t*)&counter_result));
                        eventSet->events[i].threadCounter[thread_id].counterData += counter_result;
                    }
                    break;

                case CBOX0:
                    uflags = 0x10100UL;
                    STOP_AND_READ_BOX(MSR_UNC_C0_PMON_BOX_CTL);
                    break;
                case CBOX1:
                    uflags = 0x10100UL;
                    STOP_AND_READ_BOX(MSR_UNC_C1_PMON_BOX_CTL);
                    break;
                case CBOX2:
                    uflags = 0x10100UL;
                    STOP_AND_READ_BOX(MSR_UNC_C2_PMON_BOX_CTL);
                    break;
                case CBOX3:
                    uflags = 0x10100UL;
                    STOP_AND_READ_BOX(MSR_UNC_C3_PMON_BOX_CTL);
                    break;
                case CBOX4:
                    uflags = 0x10100UL;
                    STOP_AND_READ_BOX(MSR_UNC_C4_PMON_BOX_CTL);
                    break;
                case CBOX5:
                    uflags = 0x10100UL;
                    STOP_AND_READ_BOX(MSR_UNC_C5_PMON_BOX_CTL);
                    break;
                case CBOX6:
                    uflags = 0x10100UL;
                    STOP_AND_READ_BOX(MSR_UNC_C6_PMON_BOX_CTL);
                    break;
                case CBOX7:
                    uflags = 0x10100UL;
                    STOP_AND_READ_BOX(MSR_UNC_C7_PMON_BOX_CTL);
                    break;

                case UBOX:
                case UBOXFIX:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, sandybridge_counter_map[index].configRegister, 0x0U));
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, sandybridge_counter_map[index].counterRegister, &counter_result));
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;

                case BBOX0:
                    STOP_AND_READ_PCI_BOX(PCI_HA_DEVICE, sandybridge_counter_map[index].configRegister, 
                                    sandybridge_counter_map[index].counterRegister, sandybridge_counter_map[index].counterRegister2);

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

int perfmon_readCountersThread_sandybridge(int thread_id, PerfmonEventSet* eventSet)
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
            RegisterIndex index = eventSet->events[i].index;
            if ((sandybridge_counter_map[index].type == PMC) ||
                    (sandybridge_counter_map[index].type == FIXED))
            {
                CHECK_MSR_READ_ERROR(msr_read(cpu_id, sandybridge_counter_map[index].counterRegister, &counter_result));
                eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
            }
            else
            {
                if(haveLock)
                {
                    switch (sandybridge_counter_map[index].type)
                    {
                        case POWER:
                            CHECK_POWER_READ_ERROR(power_read(cpu_id, sandybridge_counter_map[index].counterRegister,
                                    (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result * power_info.energyUnit;
                            break;

                        case MBOX0:
                            CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_0,
                                    sandybridge_counter_map[index].counterRegister,
                                    (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);
                            CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_0,
                                    sandybridge_counter_map[index].counterRegister2,
                                    (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData += counter_result;
                            break;

                        case MBOX1:
                            CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_1,
                                    sandybridge_counter_map[index].counterRegister,
                                    (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);
                            CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_1,
                                    sandybridge_counter_map[index].counterRegister2,
                                    (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData += counter_result;
                            break;

                        case MBOX2:
                            CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_2,
                                    sandybridge_counter_map[index].counterRegister,
                                    (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);
                            CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_2,
                                    sandybridge_counter_map[index].counterRegister2,
                                    (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData += counter_result;
                            break;

                        case MBOX3:
                            CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_3,
                                    sandybridge_counter_map[index].counterRegister,
                                    (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData = (counter_result<<32);
                            CHECK_PCI_READ_ERROR(pci_read(cpu_id, PCI_IMC_DEVICE_CH_3,
                                    sandybridge_counter_map[index].counterRegister2,
                                    (uint32_t*)&counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData += counter_result;
                            break;

                        case UBOX:
                        case UBOXFIX:
                            CHECK_MSR_READ_ERROR(msr_read(cpu_id, sandybridge_counter_map[index].counterRegister, &counter_result));
                            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;

                        case CBOX0:
                            READ_BOX();
                            break;
                        case CBOX1:
                            READ_BOX();
                            break;
                        case CBOX2:
                            READ_BOX();
                            break;
                        case CBOX3:
                            READ_BOX();
                            break;
                        case CBOX4:
                            READ_BOX();
                            break;
                        case CBOX5:
                            READ_BOX();
                            break;
                        case CBOX6:
                            READ_BOX();
                            break;
                        case CBOX7:
                            READ_BOX();
                            break;
                    
                        case BBOX0:
                            READ_PCI_BOX(PCI_HA_DEVICE, sandybridge_counter_map[index].counterRegister,
                                                        sandybridge_counter_map[index].counterRegister2);

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

