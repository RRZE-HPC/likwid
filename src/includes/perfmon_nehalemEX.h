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
 *               Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2013 Jan Treibig and Thomas Roehl
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
#include <perfmon_nehalemEX_counters.h>
#include <error.h>
#include <affinity.h>


static int perfmon_numArchEventsNehalemEX = NUM_ARCH_EVENTS_NEHALEMEX;
static int perfmon_numCountersNehalemEX = NUM_COUNTERS_NEHALEMEX;

/* This SUCKS: There are only subtle difference between NehalemEX
 * and Westmere EX Uncore. Still one of them is that one field is 
 * 1 bit shifted. Thank you Intel for this mess!!! Do you want 
 * to change the register definitions for every architecture?*/

int perfmon_init_nehalemEX(int cpu_id)
{
    lock_acquire((int*) &tile_lock[affinity_thread2tile_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL));
    return 0;
}

uint32_t nex_fixed_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint32_t flags = (1ULL<<(1+(index*4)));
    for(j = 0; j < event->numberOfOptions; j++)
    {
        switch (event->options[j].type)
        {
            case EVENT_OPTION_COUNT_KERNEL:
                flags |= (1ULL<<(index*4));
                break;
            case EVENT_OPTION_ANYTHREAD:
                flags |= (1ULL<<(2+(index*4)));
            default:
                break;
        }
    }
    return flags;
}

int nex_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    uint64_t offcore_flags = 0x0ULL;
    int haveTileLock = 0;
    uint64_t reg = counter_map[index].configRegister;
    if (tile_lock[affinity_thread2tile_lookup[cpu_id]] == cpu_id)
    {
        haveTileLock = 1;
    }

    flags |= (1ULL<<22)|(1ULL<<16);
    /* Intel with standard 8 bit event mask: [7:0] */
    flags |= (event->umask<<8) + event->eventId;

    if (event->cfgBits != 0 &&
       ((event->eventId != 0xB7) || (event->eventId != 0xBB)))
    {
        /* set custom cfg and cmask */
        flags |= ((event->cmask<<8) + event->cfgBits)<<16;
    }

    if (event->numberOfOptions > 0)
    {
        for (j = 0; j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                case EVENT_OPTION_INVERT:
                    flags |= (1ULL<<23);
                    break;
                case EVENT_OPTION_COUNT_KERNEL:
                    flags |= (1ULL<<17);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= (event->options[j].value & 0xFFULL)<<24;
                    break;
                case EVENT_OPTION_MATCH0:
                    offcore_flags |= (event->options[j].value & 0xFFULL);
                    break;
                case EVENT_OPTION_MATCH1:
                    offcore_flags |= (event->options[j].value & 0xF7ULL)<<8;
                    break;
                default:
                    break;
            }
        }
    }
    if ((haveTileLock) && (event->eventId == 0xB7))
    {
        if ((event->cfgBits != 0xFF) && (event->cmask != 0xFF))
        {
            offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);
        }
        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, offcore_flags));
    }

    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
    VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_PMC)

    return 0;
}


/* MBOX macros */

#define NEX_SETUP_MBOX(number)  \
    if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(MBOX##number))) \
    { \
        flags = 0x41ULL; \
        if ((event->numberOfOptions > 0) && ((event->cfgBits == 0x02) || (event->cfgBits == 0x04))) \
        { \
            for (int j=0; j < event->numberOfOptions; j++) \
            {\
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_MATCH0: \
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ADDR_MATCH, (event->options[j].value & 0x3FFFFFFFFULL))); \
                        VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ADDR_MATCH, (event->options[j].value & 0x3FFFFFFFFULL), MBOX##number##_ADDR_MATCH) \
                        break; \
                    case EVENT_OPTION_MASK0: \
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ADDR_MASK, (event->options[j].value & 0x1FFFFFFC0ULL)>>6)); \
                        VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ADDR_MASK, (event->options[j].value & 0x1FFFFFFC0ULL)>>6, MBOX##number##_ADDR_MASK) \
                        break; \
                    default: \
                        break; \
                } \
            } \
        } \
        switch (event->cfgBits)  \
        {  \
            case 0x00:   /* primary Event */  \
                flags |= (event->eventId & 0x1FULL)<<9;  \
                break;  \
            case 0x01: /* secondary Events */  \
                /* TODO fvid index is missing defaults to 0 */   \
                flags |= (1ULL<<7); /* toggle flag mode */   \
                flags |= (event->eventId & 0x7ULL)<<19;   \
                switch (event->eventId)   \
                {   \
                    case 0x00: /* CYCLES_DSP_FILL: DSP */   \
                        {   \
                            uint64_t dsp_flags = 0x0ULL;   \
                            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_DSP, &dsp_flags));   \
                            VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_DSP, dsp_flags, MBOX##number##_DSP_READ); \
                            dsp_flags |= (event->umask & 0xFULL)<<7;  \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_DSP, dsp_flags));   \
                            VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_DSP, dsp_flags, MBOX##number##_DSP); \
                        }   \
                        break;   \
                    case 0x01: /* CYCLES_SCHED_MODE: ISS */   \
                        {   \
                            uint64_t iss_flags = 0x0ULL;   \
                            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_ISS, &iss_flags));   \
                            VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ISS, iss_flags, MBOX##number##_ISS_READ); \
                            iss_flags |= (event->umask & 0x3ULL)<<4;   \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ISS, iss_flags));   \
                            VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ISS, iss_flags, MBOX##number##_ISS); \
                        }    \
                        break;   \
                    case 0x05: /* CYCLES_PGT_STATE: PGT */   \
                        {   \
                            uint64_t pgt_flags = 0x0ULL;   \
                            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_PGT, &pgt_flags));   \
                            VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_PGT, pgt_flags, MBOX##number##_PGT_READ); \
                            pgt_flags |= (event->umask & 0x1ULL)<<6;   \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_PGT, pgt_flags));   \
                            VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_PGT, pgt_flags, MBOX##number##_PGT); \
                        }    \
                        break;   \
                    case 0x06: /* BCMD_SCHEDQ_OCCUPANCY: MAP */   \
                        {   \
                            uint64_t map_flags = 0x0ULL;   \
                            CHECK_MSR_WRITE_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_MAP, &map_flags));   \
                            VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_MAP, map_flags, MBOX##number##_MAP_READ); \
                            map_flags |= (event->umask & 0xFULL)<<5;   \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_MAP, map_flags));   \
                            VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_MAP, map_flags, MBOX##number##_MAP); \
                        }   \
                        break;   \
                    case 0x04: /* CYCLES_RETRYQ_STARVED */ \
                    case 0x03: /* CYCLES_RETRYQ_MFULL */ \
                        break; \
                }    \
                break;   \
            case 0x02: /* DRAM_CMD: PLD/ISS */   \
                flags |= (event->eventId & 0x1FULL)<<9;  \
                {   \
                    uint64_t pld_flags = 0x0ULL;   \
                    uint64_t iss_flags = 0x0ULL;   \
                    CHECK_MSR_WRITE_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_PLD, &pld_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_PLD, pld_flags, MBOX##number##_PLD_READ); \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_ISS, &iss_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ISS, iss_flags, MBOX##number##_ISS_READ); \
                    pld_flags |= (event->umask & 0x1FULL)<<8;   \
                    if (event->cmask & 0xFULL != 0)   \
                    {   \
                        iss_flags |= (event->cmask & 0x7ULL)<<7;   \
                    }   \
                    if ((event->cmask & 0xF0ULL) != 0) \
                    { \
                        pld_flags |= (1ULL<<0); /* toggle cmd flag */   \
                    } \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_PLD, pld_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_PLD, pld_flags, MBOX##number##_PLD); \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ISS, iss_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ISS, iss_flags, MBOX##number##_ISS); \
                }   \
                break;   \
            case 0x03: /* DSP_FILL: DSP */   \
                flags |= (event->eventId & 0x1FULL)<<9;   \
                {   \
                    uint64_t dsp_flags = 0x0ULL;   \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_DSP, &dsp_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_DSP, dsp_flags, MBOX##number##_DSP_READ); \
                    dsp_flags |= (event->umask & 0xFULL)<<7;   \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_DSP, dsp_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_DSP, dsp_flags, MBOX##number##_DSP); \
                }   \
                break;   \
            case 0x05: /* FRM_TYPE: ISS */   \
                flags |= (event->eventId & 0x1FULL)<<9;   \
                if (((event->umask >= 0x0) && (event->umask <= 0x3)) || (event->umask == 0x8) || (event->umask == 0xC)) \
                {   \
                    uint64_t iss_flags = 0x0ULL;   \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_ISS, &iss_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ISS, iss_flags, MBOX##number##_ISS_READ); \
                    iss_flags |= event->umask & 0xFULL;   \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ISS, iss_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ISS, iss_flags, MBOX##number##_ISS); \
                }   \
                break;   \
            case 0x06: /* FVC_EV0: FVC */   \
                flags |= (event->eventId & 0x1FULL)<<9;   \
                {   \
                    uint64_t fvc_flags = 0x0ULL;   \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_ZDP, &fvc_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags, MBOX##number##_FVC_READ); \
                    fvc_flags |= (event->umask & 0x7ULL)<<11;   \
                    if (event->umask == 0x5)   \
                    {   \
                        fvc_flags |= (event->cmask & 0x7ULL)<<5;   \
                    }   \
                    else   \
                    {   \
                        fvc_flags |= (event->cmask & 0x7ULL)<<8;   \
                    }   \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ZDP, fvc_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags, MBOX##number##_FVC_EV0); \
                }   \
                break;   \
            case 0x07: /* FVC_EV1: FVC */   \
                flags |= (event->eventId & 0x1FULL)<<9;   \
                {   \
                    uint64_t fvc_flags = 0x0ULL;   \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_ZDP, &fvc_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags, MBOX##number##_FVC_READ); \
                    fvc_flags |= (event->umask & 0x7ULL)<<14;   \
                    if (event->umask == 0x5)   \
                    {   \
                        fvc_flags |= (event->cmask & 0x7ULL)<<5;   \
                    }   \
                    else   \
                    {   \
                        fvc_flags |= (event->cmask & 0x7ULL)<<8;   \
                    }   \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ZDP, fvc_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags, MBOX##number##_FVC_EV1); \
                }   \
                break;   \
            case 0x08: /* FVC_EV2: FVC */   \
                flags |= (event->eventId & 0x1FULL)<<9;   \
                {   \
                    uint64_t fvc_flags = 0x0ULL;   \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_ZDP, &fvc_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags, MBOX##number##_FVC_READ); \
                    fvc_flags |= (event->umask & 0x7ULL)<<17;   \
                    if (event->umask == 0x5)   \
                    {   \
                        fvc_flags |= (event->cmask & 0x7ULL)<<5;   \
                    }   \
                    else   \
                    {   \
                        fvc_flags |= (event->cmask & 0x7ULL)<<8;   \
                    }   \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ZDP, fvc_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags, MBOX##number##_FVC_EV2); \
                }   \
                break;   \
            case 0x09: /* FVC_EV3: FVC(ZDP) */   \
                flags |= (event->eventId & 0x1FULL)<<9;   \
                {   \
                    uint64_t fvc_flags = 0x0ULL;   \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_ZDP, &fvc_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags, MBOX##number##_FVC_READ); \
                    fvc_flags |= (event->umask<<20);   \
                    if (event->umask == 0x5)   \
                    {   \
                        fvc_flags |= (event->cmask & 0x7ULL)<<5;   \
                    }   \
                    else   \
                    {   \
                        fvc_flags |= (event->cmask & 0x7ULL)<<8;   \
                    }   \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ZDP, fvc_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags, MBOX##number##_FVC_EV3); \
                }   \
                break;   \
            case 0x16: /* PGT_PAGE_EV: PGT */   \
                flags |= (event->eventId & 0x1FULL)<<9;   \
                {   \
                    uint64_t pgt_flags = 0x0ULL;   \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_PGT, &pgt_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_PGT, pgt_flags, MBOX##number##_PGT_READ); \
                    pgt_flags |= (event->umask & 0x1ULL);   \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_PGT, pgt_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_PGT, pgt_flags, MBOX##number##_PGT); \
                }   \
                break;   \
            case 0x0D: /* THERM_TRP_DN: THR */   \
                flags |= (event->eventId & 0x1FULL)<<9;   \
                {   \
                    uint64_t thr_flags = 0x0ULL;   \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_MSC_THR, &thr_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_MSC_THR, thr_flags, MBOX##number##_THR_READ); \
                    if (event->cmask == 0x0) \
                    { \
                        thr_flags |= (1ULL<<3);   \
                    } \
                    else \
                    { \
                        thr_flags &= ~(1ULL<<3);   \
                        thr_flags |= (event->cmask & 0x7ULL) << 4; \
                    } \
                    thr_flags |= (event->umask & 0x3ULL)<<9;   \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_MSC_THR, thr_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_MSC_THR, thr_flags, MBOX##number##_THR); \
                }   \
                break;   \
            case 0x0E: /* THERM_TRP_UP: THR */   \
                flags |= (event->eventId & 0x1FULL)<<9;   \
                {   \
                    uint64_t thr_flags = 0x0ULL;   \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_MSC_THR, &thr_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_MSC_THR, thr_flags, MBOX##number##_THR_READ); \
                    if (event->cmask == 0x0) \
                    { \
                        thr_flags |= (1ULL<<3);   \
                    } \
                    else \
                    { \
                        thr_flags &= ~(1ULL<<3);   \
                        thr_flags |= (event->cmask & 0x7ULL) << 4; \
                    } \
                    thr_flags |= (event->umask & 0x3ULL)<<7;   \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_MSC_THR, thr_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_MSC_THR, thr_flags, MBOX##number##_THR); \
                }   \
                break;   \
        } \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));  \
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_MBOX##number) \
    }

#define NEX_SETUP_RBOX(number)  \
    if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(RBOX##number))) \
    { \
        flags = 0x01ULL; /* set local enable flag */ \
        switch (event->eventId) {  \
            case 0x00:  \
                flags |= (event->umask & 0x1FULL)<<1; /* configure sub register */   \
                {  \
                    uint32_t iperf_flags = 0x0UL;   \
                    iperf_flags |= (event->cfgBits<<event->cmask); /* configure event */  \
                    switch (event->umask) { /* pick correct iperf register */  \
                        case 0x00: \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_IPERF0_P0, iperf_flags));   \
                            break; \
                        case 0x01: \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_IPERF1_P0, iperf_flags));   \
                            break; \
                        case 0x06: \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_IPERF0_P1, iperf_flags));   \
                            break; \
                        case 0x07: \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_IPERF1_P1, iperf_flags));   \
                            break; \
                        case 0x0C: \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_IPERF0_P2, iperf_flags));   \
                            break; \
                        case 0x0D: \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_IPERF1_P2, iperf_flags));   \
                            break; \
                        case 0x12: \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_IPERF0_P3, iperf_flags));   \
                            break; \
                        case 0x13: \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_IPERF1_P3, iperf_flags));   \
                            break; \
                    } \
                } \
                break; \
            case 0x01: \
                flags |= (event->umask & 0x1FULL)<<1; /* configure sub register */   \
                { \
                    uint32_t qlx_flags = 0x0UL;   \
                    qlx_flags |= (event->cfgBits); /* configure event */  \
                    if (event->cmask) qlx_flags |= (event->cmask & 0x7ULL)<<4;  \
                    switch (event->umask) { /* pick correct qlx register */  \
                        case 0x02: \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_QLX_P0, qlx_flags));   \
                            break; \
                        case 0x03: \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_QLX_P0, (qlx_flags<<8)));   \
                            break; \
                        case 0x08: \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_QLX_P1, qlx_flags));   \
                            break; \
                        case 0x09: \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_QLX_P1, (qlx_flags<<8)));   \
                            break; \
                        case 0x0E: \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_QLX_P2, qlx_flags));   \
                            break; \
                        case 0x0F: \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_QLX_P2, (qlx_flags<<8)));   \
                            break; \
                        case 0x14: \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_QLX_P3, qlx_flags));   \
                            break; \
                        case 0x15: \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_QLX_P3, (qlx_flags<<8)));   \
                            break; \
                    } \
                } \
                break; \
        } \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags)); \
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_RBOX##number) \
    }

int nex_bbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x1ULL; /* set enable bit */
    uint64_t reg = counter_map[index].configRegister;
    RegisterType type = counter_map[index].type;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags |= (event->eventId<<1);
    if (event->numberOfOptions > 0)
    {
        for (j = 0; j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_MATCH0:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister1, event->options[j].value & 0xFFFFFFFFFFFFFFFULL));
                    VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister1, LLU_CAST event->options[j].value & 0xFFFFFFFFFFFFFFFULL, SETUP_BBOX_MATCH);
                    break;
                case EVENT_OPTION_MASK0:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister2, event->options[j].value & 0xFFFFFFFFFFFFFFFULL));
                    VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister2, LLU_CAST event->options[j].value & 0xFFFFFFFFFFFFFFFULL, SETUP_BBOX_MASK);
                    break;
                default:
                    break;
            }
        }
    }
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags));
    VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_BBOX);
    return 0;
}

int nex_cbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    uint64_t reg = counter_map[index].configRegister;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags = (1ULL<<22);
    flags |=(event->umask<<8) + event->eventId;
    if (event->numberOfOptions > 0)
    {
        for (j = 0; j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                case EVENT_OPTION_INVERT:
                    flags |= (1ULL<<23);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= ((event->options[j].value & 0x1FULL) << 24);
                    break;
                default:
                    break;
            }
        }
    }
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags));
    VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_CBOX);
    return 0;
}

int nex_wbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x0ULL;
    uint64_t reg = counter_map[index].configRegister;
    int j;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    flags |= (1ULL<<22); /* set enable bit */
    flags |= (event->umask<<8) + event->eventId;
    if (event->numberOfOptions > 0)
    {
        for (j = 0; j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                case EVENT_OPTION_INVERT:
                    flags |= (1ULL<<23);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= ((event->options[j].value & 0xFFULL) << 24);
                    break;
                default:
                    break;
            }
        }
    }
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags));
    VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_WBOX);
    return 0;
}

int nex_sbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    int match_mask = 0;
    uint64_t flags = 0x0ULL;
    uint64_t reg = counter_map[index].configRegister;
    RegisterType type = counter_map[index].type;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags = (1ULL<<22);
    flags |=(event->umask<<8) + event->eventId;
    if (event->numberOfOptions > 0)
    {
        if (event->eventId == 0x0)
        {
            for (j = 0; j < event->numberOfOptions; j++)
            {
                if ((event->options[j].type == EVENT_OPTION_MATCH0) ||
                    (event->options[j].type == EVENT_OPTION_MASK0))
                {
                    match_mask = 1;
                    break;
                }
            }
            if (match_mask) {
                
                if (type == SBOX0)
                {
                    VERBOSEPRINTREG(cpu_id, MSR_S0_PMON_MM_CFG, 0x0ULL, CLEAR_MM_CFG);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_S0_PMON_MM_CFG, 0x0ULL));
                }
                else
                {
                    VERBOSEPRINTREG(cpu_id, MSR_S1_PMON_MM_CFG, 0x0ULL, CLEAR_MM_CFG);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_S1_PMON_MM_CFG, 0x0ULL));
                }
            }
        }
        for (j = 0; j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                case EVENT_OPTION_INVERT:
                    flags |= (1ULL<<23);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= ((event->options[j].value & 0xFFULL) << 24);
                    break;
                case EVENT_OPTION_MATCH0:
                    if (event->eventId == 0x0)
                    {
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister1, event->options[j].value));
                        VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister1, LLU_CAST event->options[j].value, SETUP_SBOX_MATCH);
                    }
                    break;
                case EVENT_OPTION_MASK0:
                    if (event->eventId == 0x0)
                    {
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister2, event->options[j].value));
                        VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister2, LLU_CAST event->options[j].value, SETUP_SBOX_MASK);
                    }
                    break;
                default:
                    break;
            }
        }
        if (match_mask)
        {
            if (type == SBOX0)
            {
                VERBOSEPRINTREG(cpu_id, MSR_S0_PMON_MM_CFG, (1ULL<<63), SET_MM_CFG);
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_S0_PMON_MM_CFG, (1ULL<<63)));
            }
            else
            {
                VERBOSEPRINTREG(cpu_id, MSR_S1_PMON_MM_CFG, (1ULL<<63), SET_MM_CFG);
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_S1_PMON_MM_CFG, (1ULL<<63)));
            }
        }
    }
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags));
    VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_SBOX);
    return 0;
}

#define NEX_FREEZE_UNCORE \
    if (haveLock && (eventSet->regTypeMask & ~(0xF))) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, &tmp)); \
        tmp &= ~(1ULL<<28); \
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, LLU_CAST tmp, FREEZE_UNCORE) \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, tmp)); \
    }


int perfmon_setupCounterThread_nehalemEX(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int haveTileLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t fixed_flags = 0x0ULL;
    uint64_t ubox_flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }
    if (tile_lock[affinity_thread2tile_lookup[cpu_id]] == cpu_id)
    {
        haveTileLock = 1;
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    if (haveLock && (eventSet->regTypeMask & ~(0xFULL)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, 0x0ULL, FREEZE_UNCORE)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, 0x0ULL));
    }
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX0))))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M0_PMON_TIMESTAMP, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M0_PMON_DSP, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M0_PMON_ISS, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M0_PMON_MAP, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M0_PMON_MSC_THR, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M0_PMON_PGT, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M0_PMON_PLD, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M0_PMON_ZDP, 0x0ULL));
    }
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(MBOX1))))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M1_PMON_TIMESTAMP, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M1_PMON_DSP, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M1_PMON_ISS, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M1_PMON_MAP, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M1_PMON_MSC_THR, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M1_PMON_PGT, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M1_PMON_PLD, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M1_PMON_ZDP, 0x0ULL));
    }
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(RBOX0))))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R0_PMON_IPERF0_P0, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R0_PMON_IPERF0_P1, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R0_PMON_IPERF0_P2, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R0_PMON_IPERF0_P3, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R0_PMON_IPERF1_P0, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R0_PMON_IPERF1_P1, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R0_PMON_IPERF1_P2, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R0_PMON_IPERF1_P3, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R0_PMON_QLX_P0, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R0_PMON_QLX_P1, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R0_PMON_QLX_P2, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R0_PMON_QLX_P3, 0x0ULL));
    }
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(RBOX1))))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R1_PMON_IPERF0_P0, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R1_PMON_IPERF0_P1, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R1_PMON_IPERF0_P2, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R1_PMON_IPERF0_P3, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R1_PMON_IPERF1_P0, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R1_PMON_IPERF1_P1, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R1_PMON_IPERF1_P2, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R1_PMON_IPERF1_P3, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R1_PMON_QLX_P0, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R1_PMON_QLX_P1, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R1_PMON_QLX_P2, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R1_PMON_QLX_P3, 0x0ULL));
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
        {
            continue;
        }
        flags = 0x0ULL;
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = counter_map[index].configRegister;
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        switch (type)
        {
            case PMC:
                nex_pmc_setup(cpu_id, index, event);
                break;

            case FIXED:
                fixed_flags |= nex_fixed_setup(cpu_id, index, event);
                break;

            case MBOX0:
                NEX_SETUP_MBOX(0);
                break;

            case MBOX1:
                NEX_SETUP_MBOX(1);
                break;

            case BBOX0:
            case BBOX1:
                nex_bbox_setup(cpu_id, index, event);
                break;

            case RBOX0:
                NEX_SETUP_RBOX(0);
                break;

            case RBOX1:
                NEX_SETUP_RBOX(1);
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
                nex_cbox_setup(cpu_id, index, event);
                break;

            case SBOX0:
            case SBOX1:
                nex_sbox_setup(cpu_id, index, event);
                break;

            case WBOX:
                nex_wbox_setup(cpu_id, index, event);
                break;

            case WBOX0FIX:
                if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(WBOX0FIX)))
                {
                    flags = 0x1ULL;
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags));
                    VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_WBOXFIX)
                    eventSet->regTypeMask |= REG_TYPE_MASK(WBOX);
                }
                break;

            case UBOX:
                if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(UBOX)))
                {
                    flags |= (1ULL<<22); /* set enable bit */
                    flags |= event->eventId;
                    for (int j=0;j<event->numberOfOptions;j++)
                    {
                        if (event->options[j].type == EVENT_OPTION_EDGE)
                        {
                            flags |= (1ULL<<18);
                            break;
                        }
                    }
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags));
                    VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_UBOX);
                    ubox_flags = 0x1ULL;
                }
                break;

            default:
                break;
        }
    }

    if (fixed_flags != 0x0ULL)
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_FIXED_CTR_CTRL, LLU_CAST fixed_flags, SETUP_FIXED);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_FIXED_CTR_CTRL, fixed_flags));
    }
    if (ubox_flags != 0x0ULL)
    {
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, LLU_CAST ubox_flags, ACTIVATE_UBOX);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, ubox_flags));
    }
    return 0;
}

#define NEX_RESET_ALL_UNCORE_COUNTERS \
    if (haveLock && (eventSet->regTypeMask & ~(0xF))) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, &tmp)); \
        tmp |= (1ULL<<29); \
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, LLU_CAST tmp, RESET_ALL_UNCORE_COUNTERS); \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, tmp)); \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, 0x0U)); \
    }

#define NEX_UNFREEZE_UNCORE \
    if (haveLock && (eventSet->regTypeMask & ~(0xF))) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, &tmp)); \
        tmp |= (1ULL<<28); \
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, LLU_CAST tmp, UNFREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, tmp)); \
    }

#define NEX_UNFREEZE_BOX(id, flags) \
    if (haveLock && (eventSet->regTypeMask & (REG_TYPE_MASK(id)))) \
    { \
        VERBOSEPRINTREG(cpu_id, box_map[id].ctrlRegister, LLU_CAST flags, UNFREEZE_BOX); \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[id].ctrlRegister, flags)); \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[id].ovflRegister, flags)); \
    }

int perfmon_startCountersThread_nehalemEX(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t core_ctrl_flags = 0x0ULL;
    uint32_t uflags[NUM_UNITS] = { [0 ... NUM_UNITS-1] = 0x0U };
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    NEX_RESET_ALL_UNCORE_COUNTERS;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE) 
        {
            RegisterType type = eventSet->events[i].type;
            if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
            {
                continue;
            }
            RegisterIndex index = eventSet->events[i].index;
            uint64_t counter1 = counter_map[index].counterRegister;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter1, 0x0ULL));
                    core_ctrl_flags |= (1ULL<<(index-cpuid_info.perf_num_fixed_ctr));
                    break;
                case FIXED:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter1, 0x0ULL));
                    core_ctrl_flags |= (1ULL<<(index+32));
                    break;
                case WBOX0FIX:
                    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(WBOX0FIX)))
                    {
                        uflags[WBOX] |= (1ULL<<31);
                    }
                    break;
                default:
                    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(counter_map[index].type)))
                    {
                        uflags[counter_map[index].type] |= (1<<getCounterTypeOffset(index));
                    }
                    break;
            }
        }
    }

    if (haveLock)
    {
        for ( int i=0; i<NUM_UNITS; i++ )
        {
            if (uflags[i] != 0x0U)
            {
                NEX_UNFREEZE_BOX(i, uflags[i]);
            }
        }
    }

    NEX_UNFREEZE_UNCORE;

    /* Finally enable counters */
    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST core_ctrl_flags, GLOBAL_CTRL);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, core_ctrl_flags));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<63)|(1ULL<<62)|core_ctrl_flags));
    }
    return 0;
}

#define NEX_CHECK_OVERFLOW(id, offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, box_map[id].statusRegister, &tmp)); \
        if (tmp & (1ULL<<offset)) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[id].statusRegister, (tmp & (1ULL<<offset)))); \
        } \
    }

#define NEX_CHECK_UNCORE_OVERFLOW(id, offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, box_map[id].statusRegister, &tmp)); \
        if (tmp & (1ULL<<offset)) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[id].ovflRegister, (tmp & (1ULL<<offset)))); \
        } \
    }

int perfmon_stopCountersThread_nehalemEX(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST 0x0ULL, FREEZE_PMC_AND_FIXED);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    NEX_FREEZE_UNCORE;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterType type = eventSet->events[i].type;
            if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
            {
                continue;
            }
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = counter_map[index].configRegister;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter_map[index].counterRegister, &counter_result));
                    NEX_CHECK_OVERFLOW(PMC, index-cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter_map[index].counterRegister, LLU_CAST counter_result, READ_PMC);
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter_map[index].counterRegister, &counter_result));
                    NEX_CHECK_OVERFLOW(PMC, index+32);
                    VERBOSEPRINTREG(cpu_id, counter_map[index].counterRegister, LLU_CAST counter_result, READ_FIXED);
                    break;
                default:
                    if(haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(counter_map[index].type)))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter_map[index].counterRegister, &counter_result));
                        NEX_CHECK_UNCORE_OVERFLOW(counter_map[index].type, getCounterTypeOffset(index));
                        VERBOSEPRINTREG(cpu_id, counter_map[index].counterRegister, LLU_CAST counter_result, READ_UNCORE);
                    }
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
        }
    }

    return 0;
}

int perfmon_readCountersThread_nehalemEX(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t counter_result = 0x0ULL;
    uint64_t core_ctrl_flags = 0x0ULL;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, &core_ctrl_flags));
    }
    NEX_FREEZE_UNCORE;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterType type = eventSet->events[i].type;
            if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
            {
                continue;
            }
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint64_t counter = counter_map[index].counterRegister;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                    NEX_CHECK_OVERFLOW(PMC, index-cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, READ_PMC);
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                    NEX_CHECK_OVERFLOW(PMC, index+32);
                    VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, READ_FIXED);
                    break;
                default:
                    if(haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(counter_map[index].type)))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                        NEX_CHECK_UNCORE_OVERFLOW(counter_map[index].type, getCounterTypeOffset(index));
                        VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, READ_UNCORE);
                    }
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
        }
    }

    NEX_UNFREEZE_UNCORE;
    if ((eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED))) && (core_ctrl_flags != 0x0ULL))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, core_ctrl_flags));
    }
    return 0;
}

int perfmon_finalizeCountersThread_nehalemEX(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int haveTileLock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t ovf_values_core = (1ULL<<63)|(1ULL<<62);

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }
    if (tile_lock[affinity_thread2tile_lookup[cpu_id]] == cpu_id)
    {
        haveTileLock = 1;
    }
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!(eventSet->regTypeMask & (REG_TYPE_MASK(type))))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = counter_map[index].configRegister;
        PciDeviceIndex dev = counter_map[index].device;
        switch (type)
        {
            case PMC:
                ovf_values_core |= (1ULL<<(index-cpuid_info.perf_num_fixed_ctr));
                if ((haveTileLock) && (event->eventId == 0xB7))
                {
                    VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, 0x0ULL, CLEAR_OFFCORE_RESP0);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, 0x0ULL));
                }
                else if ((haveTileLock) && (event->eventId == 0xBB))
                {
                    VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP1, 0x0ULL, CLEAR_OFFCORE_RESP1);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP1, 0x0ULL));
                }
                break;
            case FIXED:
                ovf_values_core |= (1ULL<<(index+32));
                break;
            case MBOX0:
            case MBOX1:
                if (haveLock && ((event->cfgBits == 0x02) || (event->cfgBits == 0x04)))
                {
                    VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister1, 0x0ULL, CLEAR_MATCH0);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister1, 0x0ULL));
                    VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister2, 0x0ULL, CLEAR_MASK0);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister2, 0x0ULL));
                }
                break;
            case SBOX0:
                if (haveLock && (event->eventId == 0x00))
                {
                    VERBOSEPRINTREG(cpu_id, MSR_S0_PMON_MM_CFG, 0x0ULL, CLEAR_MM_CFG);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_S0_PMON_MM_CFG, 0x0ULL));
                    VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister1, 0x0ULL, CLEAR_MATCH0);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister1, 0x0ULL));
                    VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister2, 0x0ULL, CLEAR_MASK0);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister2, 0x0ULL));
                }
                break;
            case SBOX1:
                if (haveLock && (event->eventId == 0x00))
                {
                    VERBOSEPRINTREG(cpu_id, MSR_S1_PMON_MM_CFG, 0x0ULL, CLEAR_MM_CFG);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_S1_PMON_MM_CFG, 0x0ULL));
                    VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister1, 0x0ULL, CLEAR_MATCH0);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister1, 0x0ULL));
                    VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister2, 0x0ULL, CLEAR_MASK0);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister2, 0x0ULL));
                }
                break;
            case BBOX0:
            case BBOX1:
                if (haveLock && ((event->eventId == 0x01) ||
                                 (event->eventId == 0x02) ||
                                 (event->eventId == 0x03) ||
                                 (event->eventId == 0x04) ||
                                 (event->eventId == 0x05) ||
                                 (event->eventId == 0x06)))
                {
                    VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister1, 0x0ULL, CLEAR_MATCH0);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister1, 0x0ULL));
                    VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister2, 0x0ULL, CLEAR_MASK0);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister2, 0x0ULL));
                }
                break;
        }
        if ((reg) && (((dev == MSR_DEV) && (type < UNCORE)) || (((haveLock) && (type > UNCORE)))))
        {
            VERBOSEPRINTPCIREG(cpu_id, dev, reg, 0x0ULL, CLEAR_CTL);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, 0x0ULL));
        }
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, ovf_values_core, CLEAR_OVF_CTRL);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, ovf_values_core));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, CLEAR_PMC_AND_FIXED_CTRL);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    if (haveLock && (eventSet->regTypeMask & ~(0xFULL)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_OVF_CTRL, 0x0ULL, CLEAR_UNCORE_OVF);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_OVF_CTRL, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, 0x0ULL, CLEAR_UNCORE_CTRL);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, 0x0ULL));
    }
    return 0;
}
