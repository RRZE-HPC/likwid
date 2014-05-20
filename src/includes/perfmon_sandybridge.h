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

#include <perfmon_sandybridge_events.h>
#include <perfmon_sandybridge_groups.h>

#ifdef SNB_UNCORE
#define NUM_COUNTERS_SANDYBRIDGE 36
#else
#define NUM_COUNTERS_SANDYBRIDGE 11
#endif

static int perfmon_numCountersSandybridge = NUM_COUNTERS_SANDYBRIDGE;
static int perfmon_numGroupsSandybridge = NUM_GROUPS_SANDYBRIDGE;
static int perfmon_numArchEventsSandybridge = NUM_ARCH_EVENTS_SANDYBRIDGE;

static PerfmonCounterMap sandybridge_counter_map[NUM_COUNTERS_SANDYBRIDGE] = {
    {"FIXC0",PMC0},
    {"FIXC1",PMC1},
    {"FIXC2",PMC2},
    {"PMC0",PMC3},
    {"PMC1",PMC4},
    {"PMC2",PMC5},
    {"PMC3",PMC6},
    {"PWR0",PMC7},
    {"PWR1",PMC8},
    {"PWR2",PMC9},
    {"PWR3",PMC10},
#ifdef SNB_UNCORE
    {"MBOX0C0",PMC11},
    {"MBOX1C0",PMC12},
    {"MBOX2C0",PMC13},
    {"MBOX3C0",PMC14},
    {"MBOX0C1",PMC15},
    {"MBOX1C1",PMC16},
    {"MBOX2C1",PMC17},
    {"MBOX3C1",PMC18},
    {"MBOX0C2",PMC19},
    {"MBOX1C2",PMC20},
    {"MBOX2C2",PMC21},
    {"MBOX3C2",PMC22},
    {"MBOX0C3",PMC23},
    {"MBOX1C3",PMC24},
    {"MBOX2C3",PMC25},
    {"MBOX3C3",PMC26},
    {"MBOXFIX",PMC27},
    {"SBOX0P0",PMC28},
    {"SBOX1P0",PMC29},
    {"SBOX2P0",PMC30},
    {"SBOX3P0",PMC31},
    {"SBOX0P1",PMC32},
    {"SBOX1P1",PMC33},
    {"SBOX2P1",PMC34},
    {"SBOX3P1",PMC35}
#endif
};

#define OFFSET_PMC 3

void
perfmon_init_sandybridge(PerfmonThread *thread)
{
    uint64_t flags = 0x0ULL;
    int cpu_id = thread->processorId;

    for (int i=0; i<NUM_COUNTERS_SANDYBRIDGE; i++)
    {
        thread->counters[i].init = FALSE;
    }

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

    /* RAPL counters */
    thread->counters[PMC7].configRegister = MSR_PKG_ENERGY_STATUS;
    thread->counters[PMC7].counterRegister = 0x0;
    thread->counters[PMC7].type = POWER;
    thread->counters[PMC8].configRegister = MSR_PP0_ENERGY_STATUS;
    thread->counters[PMC8].counterRegister = 0x0;
    thread->counters[PMC8].type = POWER;
    thread->counters[PMC9].configRegister = MSR_PP1_ENERGY_STATUS;
    thread->counters[PMC9].counterRegister = 0x0;
    thread->counters[PMC9].type = POWER;
    thread->counters[PMC10].configRegister = MSR_DRAM_ENERGY_STATUS;
    thread->counters[PMC10].counterRegister = 0x0;
    thread->counters[PMC10].type = POWER;

#ifdef SNB_UNCORE
    /* IMC Counters: 4 48bit wide per memory channel */
    thread->counters[PMC11].configRegister = PCI_UNC_MC_PMON_CTL_0;
    thread->counters[PMC11].counterRegister = PCI_UNC_MC_PMON_CTR_0_A;
    thread->counters[PMC11].counterRegister2 = PCI_UNC_MC_PMON_CTR_0_B;
    thread->counters[PMC11].type = MBOX0;
    thread->counters[PMC12].configRegister = PCI_UNC_MC_PMON_CTL_1;
    thread->counters[PMC12].counterRegister = PCI_UNC_MC_PMON_CTR_1_A;
    thread->counters[PMC12].counterRegister2 = PCI_UNC_MC_PMON_CTR_1_B;
    thread->counters[PMC12].type = MBOX0;
    thread->counters[PMC13].configRegister = PCI_UNC_MC_PMON_CTL_2;
    thread->counters[PMC13].counterRegister = PCI_UNC_MC_PMON_CTR_2_A;
    thread->counters[PMC13].counterRegister2 = PCI_UNC_MC_PMON_CTR_2_B;
    thread->counters[PMC13].type = MBOX0;
    thread->counters[PMC14].configRegister = PCI_UNC_MC_PMON_CTL_3;
    thread->counters[PMC14].counterRegister = PCI_UNC_MC_PMON_CTR_3_A;
    thread->counters[PMC14].counterRegister2 = PCI_UNC_MC_PMON_CTR_3_B;
    thread->counters[PMC14].type = MBOX0;
    thread->counters[PMC15].configRegister = PCI_UNC_MC_PMON_CTL_0;
    thread->counters[PMC15].counterRegister = PCI_UNC_MC_PMON_CTR_0_A;
    thread->counters[PMC15].counterRegister2 = PCI_UNC_MC_PMON_CTR_0_B;
    thread->counters[PMC15].type = MBOX1;
    thread->counters[PMC16].configRegister = PCI_UNC_MC_PMON_CTL_1;
    thread->counters[PMC16].counterRegister = PCI_UNC_MC_PMON_CTR_1_A;
    thread->counters[PMC16].counterRegister2 = PCI_UNC_MC_PMON_CTR_1_B;
    thread->counters[PMC16].type = MBOX1;
    thread->counters[PMC17].configRegister = PCI_UNC_MC_PMON_CTL_2;
    thread->counters[PMC17].counterRegister = PCI_UNC_MC_PMON_CTR_2_A;
    thread->counters[PMC17].counterRegister2 = PCI_UNC_MC_PMON_CTR_2_B;
    thread->counters[PMC17].type = MBOX1;
    thread->counters[PMC18].configRegister = PCI_UNC_MC_PMON_CTL_3;
    thread->counters[PMC18].counterRegister = PCI_UNC_MC_PMON_CTR_3_A;
    thread->counters[PMC18].counterRegister2 = PCI_UNC_MC_PMON_CTR_3_B;
    thread->counters[PMC18].type = MBOX1;
    thread->counters[PMC19].configRegister = PCI_UNC_MC_PMON_CTL_0;
    thread->counters[PMC19].counterRegister = PCI_UNC_MC_PMON_CTR_0_A;
    thread->counters[PMC19].counterRegister2 = PCI_UNC_MC_PMON_CTR_0_B;
    thread->counters[PMC19].type = MBOX2;
    thread->counters[PMC20].configRegister = PCI_UNC_MC_PMON_CTL_1;
    thread->counters[PMC20].counterRegister = PCI_UNC_MC_PMON_CTR_1_A;
    thread->counters[PMC20].counterRegister2 = PCI_UNC_MC_PMON_CTR_1_B;
    thread->counters[PMC20].type = MBOX2;
    thread->counters[PMC21].configRegister = PCI_UNC_MC_PMON_CTL_2;
    thread->counters[PMC21].counterRegister = PCI_UNC_MC_PMON_CTR_2_A;
    thread->counters[PMC21].counterRegister2 = PCI_UNC_MC_PMON_CTR_2_B;
    thread->counters[PMC21].type = MBOX2;
    thread->counters[PMC22].configRegister = PCI_UNC_MC_PMON_CTL_3;
    thread->counters[PMC22].counterRegister = PCI_UNC_MC_PMON_CTR_3_A;
    thread->counters[PMC22].counterRegister2 = PCI_UNC_MC_PMON_CTR_3_B;
    thread->counters[PMC22].type = MBOX2;
    thread->counters[PMC23].configRegister = PCI_UNC_MC_PMON_CTL_0;
    thread->counters[PMC23].counterRegister = PCI_UNC_MC_PMON_CTR_0_A;
    thread->counters[PMC23].counterRegister2 = PCI_UNC_MC_PMON_CTR_0_B;
    thread->counters[PMC23].type = MBOX3;
    thread->counters[PMC24].configRegister = PCI_UNC_MC_PMON_CTL_1;
    thread->counters[PMC24].counterRegister = PCI_UNC_MC_PMON_CTR_1_A;
    thread->counters[PMC24].counterRegister2 = PCI_UNC_MC_PMON_CTR_1_B;
    thread->counters[PMC24].type = MBOX3;
    thread->counters[PMC25].configRegister = PCI_UNC_MC_PMON_CTL_2;
    thread->counters[PMC25].counterRegister = PCI_UNC_MC_PMON_CTR_2_A;
    thread->counters[PMC25].counterRegister2 = PCI_UNC_MC_PMON_CTR_2_B;
    thread->counters[PMC25].type = MBOX3;
    thread->counters[PMC26].configRegister = PCI_UNC_MC_PMON_CTL_3;
    thread->counters[PMC26].counterRegister = PCI_UNC_MC_PMON_CTR_3_A;
    thread->counters[PMC26].counterRegister2 = PCI_UNC_MC_PMON_CTR_3_B;
    thread->counters[PMC26].type = MBOX3;
    thread->counters[PMC27].configRegister = PCI_UNC_MC_PMON_FIXED_CTL;
    thread->counters[PMC27].counterRegister = PCI_UNC_MC_PMON_FIXED_CTR_A;
    thread->counters[PMC27].counterRegister2 = PCI_UNC_MC_PMON_FIXED_CTR_B;
    thread->counters[PMC27].type = MBOXFIX;

    /* QPI counters four 48bit  wide per port*/
    thread->counters[PMC28].configRegister = PCI_UNC_QPI_PMON_CTL_0;
    thread->counters[PMC28].counterRegister = PCI_UNC_QPI_PMON_CTR_0_A;
    thread->counters[PMC28].counterRegister2 = PCI_UNC_QPI_PMON_CTR_0_B;
    thread->counters[PMC28].type = SBOX0;
    thread->counters[PMC29].configRegister = PCI_UNC_QPI_PMON_CTL_1;
    thread->counters[PMC29].counterRegister = PCI_UNC_QPI_PMON_CTR_1_A;
    thread->counters[PMC29].counterRegister2 = PCI_UNC_QPI_PMON_CTR_1_B;
    thread->counters[PMC29].type = SBOX0;
    thread->counters[PMC30].configRegister = PCI_UNC_QPI_PMON_CTL_2;
    thread->counters[PMC30].counterRegister = PCI_UNC_QPI_PMON_CTR_2_A;
    thread->counters[PMC30].counterRegister2 = PCI_UNC_QPI_PMON_CTR_2_B;
    thread->counters[PMC30].type = SBOX0;
    thread->counters[PMC31].configRegister = PCI_UNC_QPI_PMON_CTL_3;
    thread->counters[PMC31].counterRegister = PCI_UNC_QPI_PMON_CTR_3_A;
    thread->counters[PMC31].counterRegister2 = PCI_UNC_QPI_PMON_CTR_3_B;
    thread->counters[PMC31].type = SBOX0;
    thread->counters[PMC32].configRegister = PCI_UNC_QPI_PMON_CTL_0;
    thread->counters[PMC32].counterRegister = PCI_UNC_QPI_PMON_CTR_0_A;
    thread->counters[PMC32].counterRegister2 = PCI_UNC_QPI_PMON_CTR_0_B;
    thread->counters[PMC32].type = SBOX1;
    thread->counters[PMC33].configRegister = PCI_UNC_QPI_PMON_CTL_1;
    thread->counters[PMC33].counterRegister = PCI_UNC_QPI_PMON_CTR_1_A;
    thread->counters[PMC33].counterRegister2 = PCI_UNC_QPI_PMON_CTR_1_B;
    thread->counters[PMC33].type = SBOX1;
    thread->counters[PMC34].configRegister = PCI_UNC_QPI_PMON_CTL_2;
    thread->counters[PMC34].counterRegister = PCI_UNC_QPI_PMON_CTR_2_A;
    thread->counters[PMC34].counterRegister2 = PCI_UNC_QPI_PMON_CTR_2_B;
    thread->counters[PMC34].type = SBOX1;
    thread->counters[PMC35].configRegister = PCI_UNC_QPI_PMON_CTL_3;
    thread->counters[PMC35].counterRegister = PCI_UNC_QPI_PMON_CTR_3_A;
    thread->counters[PMC35].counterRegister2 = PCI_UNC_QPI_PMON_CTR_3_B;
    thread->counters[PMC35].type = SBOX1;
#endif

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

#ifdef SNB_UNCORE
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id) ||
            lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id))
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
#endif
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


void
perfmon_setupCounterThread_sandybridge(int thread_id,
        PerfmonEvent* event,
        PerfmonCounterIndex index)
{
    int haveLock = 0;
    uint64_t flags;
    uint32_t uflags;
    uint64_t reg = perfmon_threadData[thread_id].counters[index].configRegister;
    int cpu_id = perfmon_threadData[thread_id].processorId;
    perfmon_threadData[thread_id].counters[index].init = TRUE;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    switch (perfmon_threadData[thread_id].counters[index].type)
    {
        case PMC:

            if (perfmon_verbose)
            {
                printf("[%d] perfmon_setup_counter PMC: Write Register 0x%llX , Flags: 0x%llX \n",
                        cpu_id,
                        LLU_CAST reg,
                        LLU_CAST flags);
            }

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
            break;

        case FIXED:
            break;

#ifdef SNB_UNCORE
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
#endif

        default:
            /* should never be reached */
            break;
    }
}

void
perfmon_startCountersThread_sandybridge(int thread_id)
{
    int i;
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint32_t uflags = 0x10000UL; /* Clear freeze bit */
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);

    for (i=0;i<NUM_PMC;i++) {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) {

            switch (perfmon_threadData[thread_id].counters[i].type)
            {
                case PMC:
                    msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister, 0x0ULL);
                    flags |= (1<<(i-OFFSET_PMC));  /* enable counter */
                    break;

                case FIXED:
                    msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister, 0x0ULL);
                    flags |= (1ULL<<(i+32));  /* enable fixed counter */
                    break;

                case POWER:
                    perfmon_threadData[thread_id].counters[i].counterRegister =
                        power_read(cpu_id, perfmon_threadData[thread_id].counters[i].configRegister);
                    break;
#ifdef SNB_UNCORE
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
#endif
                default:
                    /* should never be reached */
                    break;
            }
        }
    }

    if (perfmon_verbose)
    {
        printf("perfmon_start_counters: Write Register 0x%X , Flags: 0x%llX \n",MSR_PERF_GLOBAL_CTRL, LLU_CAST flags);
        printf("perfmon_start_counters: Write Register 0x%X , Flags: 0x%llX \n",MSR_UNCORE_PERF_GLOBAL_CTRL, LLU_CAST uflags);
    }

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, flags);
    msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x30000000FULL);

}

void 
perfmon_stopCountersThread_sandybridge(int thread_id)
{
    uint64_t flags;
    uint32_t uflags = 0x10100UL; /* Set freeze bit */
    uint64_t counter_result = 0x0ULL;
    int haveLock = 0;
    int i;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);

    for (i=0; i < NUM_COUNTERS_SANDYBRIDGE; i++) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            switch (perfmon_threadData[thread_id].counters[i].type)
            {
                case PMC:

                case FIXED:
                    perfmon_threadData[thread_id].counters[i].counterData = 
                        msr_read(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister);
                    break;

                case POWER:
                    if(haveLock)
                    {
                        perfmon_threadData[thread_id].counters[i].counterData =
                            power_info.energyUnit * ( power_read(cpu_id, perfmon_threadData[thread_id].counters[i].configRegister) -
                                    perfmon_threadData[thread_id].counters[i].counterRegister);
                    }
                    break;

#ifdef SNB_UNCORE
                case MBOX0:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_BOX_CTL, uflags);
                        counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_0, perfmon_threadData[thread_id].counters[i].counterRegister);
                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_IMC_DEVICE_CH_0, perfmon_threadData[thread_id].counters[i].counterRegister2);
                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                    }

                    break;

                case MBOX1:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_1,  PCI_UNC_MC_PMON_BOX_CTL, uflags);
                        counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_1, perfmon_threadData[thread_id].counters[i].counterRegister);
                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_IMC_DEVICE_CH_1, perfmon_threadData[thread_id].counters[i].counterRegister2);
                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                    }
                    break;

                case MBOX2:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_2,  PCI_UNC_MC_PMON_BOX_CTL, uflags);
                        counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_2, perfmon_threadData[thread_id].counters[i].counterRegister);
                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_IMC_DEVICE_CH_2, perfmon_threadData[thread_id].counters[i].counterRegister2);
                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                    }
                    break;

                case MBOX3:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_MC_PMON_BOX_CTL, uflags);
                        counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_3, perfmon_threadData[thread_id].counters[i].counterRegister);
                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_IMC_DEVICE_CH_3, perfmon_threadData[thread_id].counters[i].counterRegister2);
                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                    }
                    break;

                case MBOXFIX:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_0,  PCI_UNC_MC_PMON_FIXED_CTL, uflags);
                        counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_0, perfmon_threadData[thread_id].counters[i].counterRegister);
                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_IMC_DEVICE_CH_0, perfmon_threadData[thread_id].counters[i].counterRegister2);
                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                    }
                    break;

                case SBOX0:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_QPI_PMON_BOX_CTL, uflags);
                        counter_result = pci_read(cpu_id, PCI_QPI_DEVICE_PORT_0, perfmon_threadData[thread_id].counters[i].counterRegister);
                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_QPI_DEVICE_PORT_0, perfmon_threadData[thread_id].counters[i].counterRegister2);
                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                    }
                    break;

                case SBOX1:
                    if(haveLock)
                    {
                        pci_write(cpu_id, PCI_IMC_DEVICE_CH_3,  PCI_UNC_QPI_PMON_BOX_CTL, uflags);
                        counter_result = pci_read(cpu_id, PCI_QPI_DEVICE_PORT_1, perfmon_threadData[thread_id].counters[i].counterRegister);
                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_QPI_DEVICE_PORT_1, perfmon_threadData[thread_id].counters[i].counterRegister2);
                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                    }
                    break;
#endif

                default:
                    /* should never be reached */
                    break;
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
perfmon_readCountersThread_sandybridge(int thread_id)
{
    uint64_t counter_result = 0x0ULL;
    int haveLock = 0;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    for (int i=0; i<NUM_COUNTERS_SANDYBRIDGE; i++) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            if ((perfmon_threadData[thread_id].counters[i].type == PMC) ||
                    (perfmon_threadData[thread_id].counters[i].type == FIXED))
            {
                perfmon_threadData[thread_id].counters[i].counterData =
                    msr_read(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister);
            }
            else
            {
                if(haveLock)
                {
                    switch (perfmon_threadData[thread_id].counters[i].type)
                    {
                        case POWER:
                            perfmon_threadData[thread_id].counters[i].counterData =
                                power_info.energyUnit *  power_read(cpu_id, perfmon_threadData[thread_id].counters[i].configRegister);
                            break;

#ifdef SNB_UNCORE
                        case MBOX0:
                        counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_0, perfmon_threadData[thread_id].counters[i].counterRegister);
                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_IMC_DEVICE_CH_0, perfmon_threadData[thread_id].counters[i].counterRegister2);
                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                            break;

                        case MBOX1:
                        counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_1, perfmon_threadData[thread_id].counters[i].counterRegister);
                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_IMC_DEVICE_CH_1, perfmon_threadData[thread_id].counters[i].counterRegister2);
                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                            break;

                        case MBOX2:
                        counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_2, perfmon_threadData[thread_id].counters[i].counterRegister);
                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_IMC_DEVICE_CH_2, perfmon_threadData[thread_id].counters[i].counterRegister2);
                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                            break;

                        case MBOX3:
                        counter_result = pci_read(cpu_id, PCI_IMC_DEVICE_CH_3, perfmon_threadData[thread_id].counters[i].counterRegister);
                        counter_result = (counter_result<<32) +
                            pci_read(cpu_id, PCI_IMC_DEVICE_CH_3, perfmon_threadData[thread_id].counters[i].counterRegister2);
                        perfmon_threadData[thread_id].counters[i].counterData = counter_result;
                            break;
#endif

                        default:
                            /* should never be reached */
                            break;
                    }
                }
            }
        }
    }
}

