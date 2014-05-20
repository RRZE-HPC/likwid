/*
 * =======================================================================================
 *
 *      Filename:  perfmon_westmereEX.h
 *
 *      Description:  Header File of perfmon module for Westmere EX.
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

#include <perfmon_westmereEX_events.h>
#include <perfmon_westmereEX_groups.h>
#include <perfmon_westmereEX_counters.h>


static int perfmon_numCountersWestmereEX = NUM_COUNTERS_WESTMEREEX;
static int perfmon_numGroupsWestmereEX = NUM_GROUPS_WESTMEREEX;
static int perfmon_numArchEventsWestmereEX = NUM_ARCH_EVENTS_WESTMEREEX;

static PerfmonUnit westmereEX_PMunits[NUM_UNITS];

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void perfmon_init_westmereEX(PerfmonThread *thread)
{
    uint64_t flags = 0x0ULL;
    int cpu_id = thread->processorId;

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

    /* Initialize uncore */
    /* MBOX */
    thread->counters[PMC7].id  = 0;
    thread->counters[PMC8].id  = 1;
    thread->counters[PMC9].id  = 2;
    thread->counters[PMC10].id = 3;
    thread->counters[PMC11].id = 4;
    thread->counters[PMC12].id = 5;
    westmereEX_PMunits[MBOX0].ctrlRegister = MSR_M0_PMON_BOX_CTRL;
    westmereEX_PMunits[MBOX0].statusRegister = MSR_M0_PMON_BOX_STATUS;
    westmereEX_PMunits[MBOX0].ovflRegister = MSR_M0_PMON_BOX_OVF_CTRL;

    thread->counters[PMC13].id = 0;
    thread->counters[PMC14].id = 1;
    thread->counters[PMC15].id = 2;
    thread->counters[PMC16].id = 3;
    thread->counters[PMC17].id = 4;
    thread->counters[PMC18].id = 5;
    westmereEX_PMunits[MBOX1].ctrlRegister = MSR_M1_PMON_BOX_CTRL;
    westmereEX_PMunits[MBOX1].statusRegister = MSR_M1_PMON_BOX_STATUS;
    westmereEX_PMunits[MBOX1].ovflRegister = MSR_M1_PMON_BOX_OVF_CTRL;

    /* BBOX */
    thread->counters[PMC19].id = 0;
    thread->counters[PMC20].id = 1;
    thread->counters[PMC21].id = 2;
    thread->counters[PMC22].id = 3;
    westmereEX_PMunits[BBOX0].ctrlRegister = MSR_B0_PMON_BOX_CTRL;
    westmereEX_PMunits[BBOX0].statusRegister =  MSR_B0_PMON_BOX_STATUS;
    westmereEX_PMunits[BBOX0].ovflRegister = MSR_B0_PMON_BOX_OVF_CTRL;

    thread->counters[PMC23].id = 0;
    thread->counters[PMC24].id = 1;
    thread->counters[PMC25].id = 2;
    thread->counters[PMC26].id = 3;
    westmereEX_PMunits[BBOX1].ctrlRegister = MSR_B1_PMON_BOX_CTRL;
    westmereEX_PMunits[BBOX1].statusRegister =  MSR_B1_PMON_BOX_STATUS;
    westmereEX_PMunits[BBOX1].ovflRegister = MSR_B1_PMON_BOX_OVF_CTRL;

    /* RBOX */
    thread->counters[PMC27].id = 0;
    thread->counters[PMC28].id = 1;
    thread->counters[PMC29].id = 2;
    thread->counters[PMC30].id = 3;
    thread->counters[PMC31].id = 4;
    thread->counters[PMC32].id = 5;
    thread->counters[PMC33].id = 6;
    thread->counters[PMC34].id = 7;
    westmereEX_PMunits[RBOX0].ctrlRegister = MSR_R0_PMON_BOX_CTRL;
    westmereEX_PMunits[RBOX0].statusRegister =  MSR_R0_PMON_BOX_STATUS;
    westmereEX_PMunits[RBOX0].ovflRegister = MSR_R0_PMON_BOX_OVF_CTRL;

    thread->counters[PMC35].id = 0;
    thread->counters[PMC36].id = 1;
    thread->counters[PMC37].id = 2;
    thread->counters[PMC38].id = 3;
    thread->counters[PMC39].id = 4;
    thread->counters[PMC40].id = 5;
    thread->counters[PMC41].id = 6;
    thread->counters[PMC42].id = 7;
    westmereEX_PMunits[RBOX1].ctrlRegister = MSR_R1_PMON_BOX_CTRL;
    westmereEX_PMunits[RBOX1].statusRegister =  MSR_R1_PMON_BOX_STATUS;
    westmereEX_PMunits[RBOX1].ovflRegister = MSR_R1_PMON_BOX_OVF_CTRL;

    /* WBOX */
    thread->counters[PMC43].id = 0;
    thread->counters[PMC44].id = 1;
    thread->counters[PMC45].id = 2;
    thread->counters[PMC46].id = 3;
    thread->counters[PMC47].id = 31;
    westmereEX_PMunits[WBOX].ctrlRegister   = MSR_W_PMON_BOX_CTRL;
    westmereEX_PMunits[WBOX].statusRegister = MSR_W_PMON_BOX_STATUS;
    westmereEX_PMunits[WBOX].ovflRegister   = MSR_W_PMON_BOX_OVF_CTRL;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id) ||
            lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id))
    {
        msr_write(cpu_id, MSR_W_PMON_BOX_CTRL,  0x0ULL);
        msr_write(cpu_id, MSR_W_PMON_EVNT_SEL0, 0x0ULL);
        msr_write(cpu_id, MSR_W_PMON_EVNT_SEL1, 0x0ULL);
        msr_write(cpu_id, MSR_W_PMON_EVNT_SEL2, 0x0ULL);
        msr_write(cpu_id, MSR_W_PMON_EVNT_SEL3, 0x0ULL);
        msr_write(cpu_id, MSR_W_PMON_FIXED_CTR, 0x0ULL);

        msr_write(cpu_id, MSR_M0_PMON_BOX_CTRL,  0x0ULL);
        msr_write(cpu_id, MSR_M0_PMON_EVNT_SEL0, 0x0ULL);
        msr_write(cpu_id, MSR_M0_PMON_EVNT_SEL1, 0x0ULL);
        msr_write(cpu_id, MSR_M0_PMON_EVNT_SEL2, 0x0ULL);
        msr_write(cpu_id, MSR_M0_PMON_EVNT_SEL3, 0x0ULL);
        msr_write(cpu_id, MSR_M0_PMON_EVNT_SEL4, 0x0ULL);
        msr_write(cpu_id, MSR_M0_PMON_EVNT_SEL5, 0x0ULL);

        msr_write(cpu_id, MSR_M1_PMON_BOX_CTRL,  0x0ULL);
        msr_write(cpu_id, MSR_M1_PMON_EVNT_SEL0, 0x0ULL);
        msr_write(cpu_id, MSR_M1_PMON_EVNT_SEL1, 0x0ULL);
        msr_write(cpu_id, MSR_M1_PMON_EVNT_SEL2, 0x0ULL);
        msr_write(cpu_id, MSR_M1_PMON_EVNT_SEL3, 0x0ULL);
        msr_write(cpu_id, MSR_M1_PMON_EVNT_SEL4, 0x0ULL);
        msr_write(cpu_id, MSR_M1_PMON_EVNT_SEL5, 0x0ULL);

        msr_write(cpu_id, MSR_B0_PMON_BOX_CTRL,  0x0ULL);
        msr_write(cpu_id, MSR_B0_PMON_EVNT_SEL0, 0x0ULL);
        msr_write(cpu_id, MSR_B0_PMON_EVNT_SEL1, 0x0ULL);
        msr_write(cpu_id, MSR_B0_PMON_EVNT_SEL2, 0x0ULL);
        msr_write(cpu_id, MSR_B0_PMON_EVNT_SEL3, 0x0ULL);

        msr_write(cpu_id, MSR_B1_PMON_BOX_CTRL,  0x0ULL);
        msr_write(cpu_id, MSR_B1_PMON_EVNT_SEL0, 0x0ULL);
        msr_write(cpu_id, MSR_B1_PMON_EVNT_SEL1, 0x0ULL);
        msr_write(cpu_id, MSR_B1_PMON_EVNT_SEL2, 0x0ULL);
        msr_write(cpu_id, MSR_B1_PMON_EVNT_SEL3, 0x0ULL);

        msr_write(cpu_id, MSR_R0_PMON_BOX_CTRL,  0x0ULL);
        msr_write(cpu_id, MSR_R0_PMON_EVNT_SEL0, 0x0ULL);
        msr_write(cpu_id, MSR_R0_PMON_EVNT_SEL1, 0x0ULL);
        msr_write(cpu_id, MSR_R0_PMON_EVNT_SEL2, 0x0ULL);
        msr_write(cpu_id, MSR_R0_PMON_EVNT_SEL3, 0x0ULL);
        msr_write(cpu_id, MSR_R0_PMON_EVNT_SEL4, 0x0ULL);
        msr_write(cpu_id, MSR_R0_PMON_EVNT_SEL5, 0x0ULL);
        msr_write(cpu_id, MSR_R0_PMON_EVNT_SEL6, 0x0ULL);
        msr_write(cpu_id, MSR_R0_PMON_EVNT_SEL7, 0x0ULL);

        msr_write(cpu_id, MSR_R1_PMON_BOX_CTRL,   0x0ULL);
        msr_write(cpu_id, MSR_R1_PMON_EVNT_SEL8,  0x0ULL);
        msr_write(cpu_id, MSR_R1_PMON_EVNT_SEL9,  0x0ULL);
        msr_write(cpu_id, MSR_R1_PMON_EVNT_SEL10, 0x0ULL);
        msr_write(cpu_id, MSR_R1_PMON_EVNT_SEL11, 0x0ULL);
        msr_write(cpu_id, MSR_R1_PMON_EVNT_SEL12, 0x0ULL);
        msr_write(cpu_id, MSR_R1_PMON_EVNT_SEL13, 0x0ULL);
        msr_write(cpu_id, MSR_R1_PMON_EVNT_SEL14, 0x0ULL);
        msr_write(cpu_id, MSR_R1_PMON_EVNT_SEL15, 0x0ULL);

        {
            uint32_t ubflags = 0x0UL;
            ubflags |= (1<<29); /* reset all */
            msr_write(cpu_id, MSR_U_PMON_GLOBAL_CTRL, ubflags );
        }
    }
}

/* MBOX macros */

#define MBOX_GATE(NUM)  \
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
        fvc_flags |= (event->umask<<12);   \
        if (event->umask == 0x5)   \
        {   \
            fvc_flags |= (event->cmask<<6);   \
        }   \
        else   \
        {   \
            fvc_flags |= (event->cmask<<9);   \
        }   \
        msr_write(cpu_id, MSR_M##NUM##_PMON_ZDP, fvc_flags);   \
        VERBOSEPRINTREG(cpu_id, MSR_M##NUM##_PMON_ZDP, fvc_flags, FVC_EV0) \
    }   \
    break;   \
    case 0x07: /* FVC_EV1: FVC */   \
                                    flags |= (event->eventId<<9);   \
    {   \
        uint32_t fvc_flags = 0x0UL;   \
        fvc_flags |= (event->umask<<15);   \
        if (event->umask == 0x5)   \
        {   \
            fvc_flags |= (event->cmask<<6);   \
        }   \
        else   \
        {   \
            fvc_flags |= (event->cmask<<9);   \
        }   \
        msr_write(cpu_id, MSR_M##NUM##_PMON_ZDP, fvc_flags);   \
        VERBOSEPRINTREG(cpu_id, MSR_M##NUM##_PMON_ZDP, fvc_flags, FVC_EV1) \
    }   \
    break;   \
    case 0x08: /* FVC_EV2: FVC */   \
                                    flags |= (event->eventId<<9);   \
    {   \
        uint32_t fvc_flags = 0x0UL;   \
        fvc_flags |= (event->umask<<18);   \
        if (event->umask == 0x5)   \
        {   \
            fvc_flags |= (event->cmask<<6);   \
        }   \
        else   \
        {   \
            fvc_flags |= (event->cmask<<9);   \
        }   \
        msr_write(cpu_id, MSR_M##NUM##_PMON_ZDP, fvc_flags);   \
        VERBOSEPRINTREG(cpu_id, MSR_M##NUM##_PMON_ZDP, fvc_flags, FVC_EV2) \
    }   \
    break;   \
    case 0x09: /* FVC_EV3: FVC(ZDP) */   \
                                         flags |= (event->eventId<<9);   \
    {   \
        uint32_t fvc_flags = 0x0UL;   \
        fvc_flags |= (event->umask<<21);   \
        if (event->umask == 0x5)   \
        {   \
            fvc_flags |= (event->cmask<<6);   \
        }   \
        else   \
        {   \
            fvc_flags |= (event->cmask<<9);   \
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

/* RBOX macros */
#define RBOX_GATE(NUM)  \
    flags = 0x01ULL; /* set local enable flag */ \
switch (event->eventId) {  \
    case 0x00:  \
                flags |= (event->umask<<1); /* configure sub register */   \
    {  \
        uint32_t iperf_flags = 0x0UL;   \
        iperf_flags |= (event->cfgBits<<event->cmask); /* configure event */  \
        switch (event->umask) { /* pick correct iperf register */  \
            case 0x00: \
                       msr_write(cpu_id, MSR_R##NUM##_PMON_IPERF0_P0, iperf_flags);   \
            break; \
            case 0x01: \
                       msr_write(cpu_id, MSR_R##NUM##_PMON_IPERF1_P0, iperf_flags);   \
            break; \
            case 0x06: \
                       msr_write(cpu_id, MSR_R##NUM##_PMON_IPERF0_P1, iperf_flags);   \
            break; \
            case 0x07: \
                       msr_write(cpu_id, MSR_R##NUM##_PMON_IPERF1_P1, iperf_flags);   \
            break; \
            case 0x0C: \
                       msr_write(cpu_id, MSR_R##NUM##_PMON_IPERF0_P2, iperf_flags);   \
            break; \
            case 0x0D: \
                       msr_write(cpu_id, MSR_R##NUM##_PMON_IPERF1_P2, iperf_flags);   \
            break; \
            case 0x12: \
                       msr_write(cpu_id, MSR_R##NUM##_PMON_IPERF0_P3, iperf_flags);   \
            break; \
            case 0x13: \
                       msr_write(cpu_id, MSR_R##NUM##_PMON_IPERF1_P3, iperf_flags);   \
            break; \
        } } \
    break; \
    case 0x01: \
               flags |= (event->umask<<1); /* configure sub register */   \
    { \
        uint32_t qlx_flags = 0x0UL;   \
        qlx_flags |= (event->cfgBits); /* configure event */  \
        if (event->cmask) qlx_flags |= (event->cmask<<4);  \
        switch (event->umask) { /* pick correct qlx register */  \
            case 0x02: \
                       msr_write(cpu_id, MSR_R##NUM##_PMON_QLX_P0, qlx_flags);   \
            break; \
            case 0x03: \
                       msr_write(cpu_id, MSR_R##NUM##_PMON_QLX_P0, (qlx_flags<<8));   \
            break; \
            case 0x08: \
                       msr_write(cpu_id, MSR_R##NUM##_PMON_QLX_P0, qlx_flags);   \
            break; \
            case 0x09: \
                       msr_write(cpu_id, MSR_R##NUM##_PMON_QLX_P1, (qlx_flags<<8));   \
            break; \
            case 0x0E: \
                       msr_write(cpu_id, MSR_R##NUM##_PMON_QLX_P0, qlx_flags);   \
            break; \
            case 0x0F: \
                       msr_write(cpu_id, MSR_R##NUM##_PMON_QLX_P2, (qlx_flags<<8));   \
            break; \
            case 0x14: \
                       msr_write(cpu_id, MSR_R##NUM##_PMON_QLX_P0, qlx_flags);   \
            break; \
            case 0x15: \
                       msr_write(cpu_id, MSR_R##NUM##_PMON_QLX_P3, (qlx_flags<<8));   \
            break; \
        } } \
    break; \
}



void perfmon_setupCounterThread_westmereEX(
        int thread_id,
        PerfmonEvent* event,
        PerfmonCounterIndex index)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;;
    uint64_t reg = westmereEX_counter_map[index].configRegister;
    int cpu_id = perfmon_threadData[thread_id].processorId;
    perfmon_threadData[thread_id].counters[index].init = TRUE;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }

    switch (westmereEX_counter_map[index].type)
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
            if (haveLock)
            {
                MBOX_GATE(0);
                msr_write(cpu_id, reg , flags);
                VERBOSEPRINTREG(cpu_id, reg, flags, MBOX0_CTRL)
            }
            break;

        case MBOX1:
            if (haveLock)
            {
                MBOX_GATE(1);
                msr_write(cpu_id, reg , flags);
                VERBOSEPRINTREG(cpu_id, reg, flags, MBOX1_CTRL)
            }
            break;

        case BBOX0:

        case BBOX1:
            if (haveLock)
            {
                flags = 0x1ULL; /* set enable bit */
                flags |=  (event->eventId<<1);
                msr_write(cpu_id, reg , flags);
                VERBOSEPRINTREG(cpu_id, reg, flags, BBOX_CTRL)
            }
            break;

        case RBOX0:
            if (haveLock)
            {
                RBOX_GATE(0);
                msr_write(cpu_id, reg , flags);
                VERBOSEPRINTREG(cpu_id, reg, flags, RBOX0_CTRL)
            }
            break;

        case RBOX1:
            if (haveLock)
            {
                RBOX_GATE(1);
                msr_write(cpu_id, reg , flags);
                VERBOSEPRINTREG(cpu_id, reg, flags, RBOX1_CTRL)
            }
            break;

        case WBOX:
            if (haveLock)
            {
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
            }
            break;

        default:
            /* should never be reached */
            break;
    }
}

/* Actions for Performance Monitoring Session:
 *
 * Core Counters (counter is always enabled in PERVSEL register):
 * 1) Disable counters in global ctrl Register MSR_PERF_GLOBAL_CTRL
 * 2) Zero according counter registers
 * 3) Set enable bit in global register flag
 * 4) Write global register flag
 *
 * Uncore Counters (only one core per socket):
 * 1) Set reset flag in global U Box control register
 * 2) Zero according counter registers
 * 3) Set enable bit in Box control register
 * 4) Write according uncore Box ctrl register
 * 3) Set enable bit in global U Box control register
 * */

void perfmon_startCountersThread_westmereEX(int thread_id)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint32_t uflags[NUM_UNITS];
    int cpu_id = perfmon_threadData[thread_id].processorId;

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        uint32_t ubflags = 0x0UL;
        ubflags |= (1<<29); /* reset all */
        haveLock = 1;
        //        msr_write(cpu_id, MSR_U_PMON_GLOBAL_CTRL, ubflags );
        //       VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, ubflags, UBOX_GLOBAL_CTRL)
    }

    for ( int i=0; i<NUM_UNITS; i++ )
    {
        uflags[i] = 0x0UL;
    }

    for ( int i=0; i<NUM_PMC; i++ ) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) {
            if (westmereEX_counter_map[i].type == PMC)
            {
                msr_write(cpu_id, westmereEX_counter_map[i].counterRegister , 0x0ULL);
                flags |= (1<<(i-OFFSET_PMC));  /* enable counter */
            }
            else if (westmereEX_counter_map[i].type == FIXED)
            {
                msr_write(cpu_id, westmereEX_counter_map[i].counterRegister , 0x0ULL);
                flags |= (1ULL<<(i+32));  /* enable fixed counter */
            }
            else if (westmereEX_counter_map[i].type > UNCORE)
            {
                if(haveLock)
                {
                    msr_write(cpu_id, westmereEX_counter_map[i].counterRegister , 0x0ULL);
                    uflags[westmereEX_counter_map[i].type] |=
                        (1<<(perfmon_threadData[thread_id].counters[i].id));  /* enable uncore counter */
                }
            }
        }
    }

    VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, GLOBAL_CTRL);

    if (haveLock)
    {
        for ( int i=0; i<NUM_UNITS; i++ )
        {
            /* if counters are enabled write the according box ctrl register */
            if (uflags[i]) 
            {
                msr_write(cpu_id, westmereEX_PMunits[i].ctrlRegister, uflags[i]);
                VERBOSEPRINTREG(cpu_id, westmereEX_PMunits[i].ctrlRegister, LLU_CAST uflags[i], BOXCTRL);
            }
        }

        /* set global enable flag in U BOX ctrl register */
        uint32_t ubflags = 0x0UL;
        ubflags |= (1<<28); /* enable all */
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, LLU_CAST ubflags, UBOX_GLOBAL_CTRL);
        msr_write(cpu_id, MSR_U_PMON_GLOBAL_CTRL, ubflags );
    }
    /* Finally enable counters */
    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, flags);
    msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x30000000FULL);
}

void perfmon_stopCountersThread_westmereEX(int thread_id)
{
    int haveLock = 0;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        uint32_t ubflags = 0x0UL;
        haveLock = 1;
        //        ubflags |= (1<<29); /* reset all */
        msr_write(cpu_id, MSR_U_PMON_GLOBAL_CTRL, ubflags );
    }

    for ( int i=0; i<NUM_COUNTERS_WESTMEREEX; i++ ) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            if (westmereEX_counter_map[i].type > UNCORE)
            {
                if(haveLock)
                {
                    perfmon_threadData[thread_id].counters[i].counterData =
                        msr_read(cpu_id, westmereEX_counter_map[i].counterRegister);

                    VERBOSEPRINTREG(cpu_id, westmereEX_counter_map[i].counterRegister,
                            LLU_CAST perfmon_threadData[thread_id].counters[i].counterData, READ_UNCORE);
                }
            }
            else
            {
                perfmon_threadData[thread_id].counters[i].counterData =
                    msr_read(cpu_id, westmereEX_counter_map[i].counterRegister);

                VERBOSEPRINTREG(cpu_id, westmereEX_counter_map[i].counterRegister,
                        LLU_CAST perfmon_threadData[thread_id].counters[i].counterData, READ_CORE);
            }
        }
    }

#if 0
    flags = msr_read(cpu_id,MSR_PERF_GLOBAL_STATUS);
    printf ("Status: 0x%llX \n", LLU_CAST flags);
    if((flags & 0x3) || (flags & (0x3ULL<<32)) ) 
    {
        printf ("Overflow occured \n");
    }
#endif
}

void perfmon_readCountersThread_westmereEX(int thread_id)
{
    int haveLock = 0;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    for ( int i=0; i<NUM_COUNTERS_WESTMEREEX; i++ ) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            if (westmereEX_counter_map[i].type > UNCORE)
            {
                if(haveLock)
                {
                    perfmon_threadData[thread_id].counters[i].counterData =
                        msr_read(cpu_id, westmereEX_counter_map[i].counterRegister);
                }
            }
            else
            {
                perfmon_threadData[thread_id].counters[i].counterData =
                    msr_read(cpu_id, westmereEX_counter_map[i].counterRegister);
            }
        }
    }
}

