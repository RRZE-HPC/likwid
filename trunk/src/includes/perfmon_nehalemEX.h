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

#include <perfmon_nehalemEX_events.h>
#include <error.h>
#include <affinity.h>

#define NUM_COUNTERS_NEHALEMEX 7

static int perfmon_numCountersNehalemEX = NUM_COUNTERS_NEHALEMEX;
static int perfmon_numArchEventsNehalemEX = NUM_ARCH_EVENTS_NEHALEMEX;

/* This SUCKS: There are only subtle difference between NehalemEX
 * and Westmere EX Uncore. Still one of them is that one field is 
 * 1 bit shifted. Thank you Intel for this mess!!! Do you want 
 * to change the register definitions for every architecture?*/

int perfmon_init_nehalemEX(int cpu_id)
{
    lock_acquire((int*) &tile_lock[affinity_core2tile_lookup[cpu_id]], cpu_id);
    if ( cpuid_info.model == NEHALEM_EX )
    {
        lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    }
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL));
    return 0;
}

/* MBOX macros */

#define NEX_SETUP_MBOX(number)  \
    if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(MBOX##number))) \
    { \
        flags = 0x41ULL; \
        if (event->numberOfOptions > 0 && (event->cfgBits == 0x02 || event->cfgBits == 0x04)) \
        { \
            for (int j=0; j < event->numberOfOptions; j++) \
            {\
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_MATCH0: \
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ADDR_MATCH, event->options[j].value)); \
                        VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, event->options[j].value, MBOX##number##_ADDR_MATCH) \
                        break; \
                    case EVENT_OPTION_MASK0: \
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ADDR_MASK, event->options[j].value)); \
                        VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, event->options[j].value, MBOX##number##_ADDR_MASK) \
                        break; \
                    default: \
                        break; \
                } \
            } \
        } \
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
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_DSP, dsp_flags));   \
                        }   \
                        break;   \
                    case 0x01: /* CYCLES_SCHED_MODE: ISS */   \
                        {   \
                            uint32_t iss_flags = 0x0UL;   \
                            iss_flags |= (event->umask<<4);   \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ISS, iss_flags));   \
                        }    \
                        break;   \
                    case 0x05: /* CYCLES_PGT_STATE: PGT */   \
                        {   \
                            uint32_t pgt_flags = 0x0UL;   \
                            pgt_flags |= (event->umask<<6);   \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_PGT, pgt_flags));   \
                        }    \
                        break;   \
                    case 0x06: /* BCMD_SCHEDQ_OCCUPANCY: MAP */   \
                        {   \
                            uint32_t map_flags = 0x0UL;   \
                            map_flags |= (event->umask<<6);   \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_MAP, map_flags));   \
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
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_PLD, pld_flags));   \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ISS, iss_flags));   \
                }   \
                break;   \
            case 0x03: /* DSP_FILL: DSP */   \
                flags |= (event->eventId<<9);   \
                {   \
                    uint64_t dsp_flags = 0x0ULL;   \
                    dsp_flags |= (event->umask<<7);   \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_DSP, dsp_flags));   \
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
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_PLD, pld_flags));   \
                }   \
                break;   \
            case 0x05: /* FRM_TYPE: ISS */   \
                flags |= (event->eventId<<9);   \
                {   \
                    uint32_t iss_flags = 0x0UL;   \
                    iss_flags |= event->umask;   \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ISS, iss_flags));   \
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
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ZDP, fvc_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags, FVC_EV0) \
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
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ZDP, fvc_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags, FVC_EV1) \
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
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ZDP, fvc_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags, FVC_EV2) \
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
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ZDP, fvc_flags));   \
            }   \
            break;   \
            case 0x0A: /* ISS_SCHED: ISS */   \
            flags |= (event->eventId<<9);   \
            {   \
                uint32_t iss_flags = 0x0UL;   \
                iss_flags |= (event->umask<<10);   \
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ISS, iss_flags));   \
            }   \
            break;   \
            case 0x0B: /* PGT_PAGE_EV: PGT */   \
            flags |= (event->eventId<<9);   \
            {   \
                uint32_t pgt_flags = 0x0UL;   \
                pgt_flags |= event->umask;   \
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_PGT, pgt_flags));   \
            }   \
            break;   \
            case 0x0C: /* PGT_PAGE_EV2: PGT */   \
            flags |= (event->eventId<<9);   \
            {   \
                uint32_t pgt_flags = 0x0UL;   \
                pgt_flags |= (event->umask<<11);   \
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_PGT, pgt_flags));   \
            }   \
            break;   \
            case 0x0D: /* THERM_TRP_DN: THR */   \
            flags |= (event->eventId<<9);   \
            {   \
                uint32_t thr_flags = 0x0UL;   \
                thr_flags |= (1<<3);   \
                thr_flags |= (event->umask<<9);   \
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_PGT, thr_flags));   \
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
                flags |= (event->umask<<1); /* configure sub register */   \
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
                flags |= (event->umask<<1); /* configure sub register */   \
                { \
                    uint32_t qlx_flags = 0x0UL;   \
                    qlx_flags |= (event->cfgBits); /* configure event */  \
                    if (event->cmask) qlx_flags |= (event->cmask<<4);  \
                    switch (event->umask) { /* pick correct qlx register */  \
                        case 0x02: \
                                   CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_QLX_P0, qlx_flags));   \
                        break; \
                        case 0x03: \
                                   CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_QLX_P0, (qlx_flags<<8)));   \
                        break; \
                        case 0x08: \
                                   CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_QLX_P0, qlx_flags));   \
                        break; \
                        case 0x09: \
                                   CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_QLX_P1, (qlx_flags<<8)));   \
                        break; \
                        case 0x0E: \
                                   CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_QLX_P0, qlx_flags));   \
                        break; \
                        case 0x0F: \
                                   CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_QLX_P2, (qlx_flags<<8)));   \
                        break; \
                        case 0x14: \
                                   CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_R##number##_PMON_QLX_P0, qlx_flags));   \
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

#define NEX_SETUP_BBOX(number) \
    if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(BBOX##number))) \
    { \
        flags = 0x1ULL; /* set enable bit */ \
        flags |=  (event->eventId<<1); \
        if (event->numberOfOptions > 0) \
        { \
            for (int j=0;j<event->numberOfOptions;j++) \
            { \
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_MATCH0: \
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_B##number##_PMON_MATCH, event->options[j].value)); \
                        VERBOSEPRINTREG(cpu_id, MSR_B##number##_PMON_MATCH, LLU_CAST event->options[j].value, SETUP_BBOX##number##_MATCH) \
                        break; \
                    case EVENT_OPTION_MASK0: \
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_B##number##_PMON_MASK, event->options[j].value)); \
                        VERBOSEPRINTREG(cpu_id, MSR_B##number##_PMON_MASK, LLU_CAST event->options[j].value, SETUP_BBOX##number##_MASK) \
                        break; \
                    default: \
                        break; \
                } \
            } \
        } \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags)); \
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_BBOX##number) \
    }

#define NEX_SETUP_CBOX(number) \
    if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(CBOX##number))) \
    { \
        flags = (1<<22); \
        flags |=(event->umask<<8) + event->eventId; \
        if (event->numberOfOptions > 0) \
        { \
            for (int j=0;j<event->numberOfOptions;j++) \
            { \
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_EDGE: \
                        flags |= (1<<18); \
                        break; \
                    case EVENT_OPTION_INVERT: \
                        flags |= (1<<23); \
                        break; \
                    case EVENT_OPTION_THRESHOLD: \
                        flags |= ((event->options[j].value & 0x1F) << 24); \
                        break; \
                    default: \
                        break; \
                } \
            } \
        } \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags)); \
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_CBOX##number) \
    }

#define NEX_SETUP_SBOX(number) \
    if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(SBOX##number))) \
    { \
        flags = (1<<22); \
        flags |=(event->umask<<8) + event->eventId; \
        if (event->numberOfOptions > 0) \
        { \
            for (int j=0;j<event->numberOfOptions;j++) \
            { \
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_EDGE: \
                        flags |= (1<<18); \
                        break; \
                    case EVENT_OPTION_INVERT: \
                        flags |= (1<<23); \
                        break; \
                    case EVENT_OPTION_THRESHOLD: \
                        flags |= ((event->options[j].value & 0x1F) << 24); \
                        break; \
                    case EVENT_OPTION_MATCH0: \
                        if (event->eventId == 0x0) \
                        { \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_B##number##_PMON_MATCH, event->options[j].value)); \
                            VERBOSEPRINTREG(cpu_id, MSR_S##number##_PMON_MATCH, \
                                    LLU_CAST event->options[j].value, SETUP_SBOX##number##_MATCH) \
                        } \
                        break; \
                    case EVENT_OPTION_MASK0: \
                        if (event->eventId == 0x0) \
                        { \
                            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_B##number##_PMON_MASK, event->options[j].value)); \
                            VERBOSEPRINTREG(cpu_id, MSR_S##number##_PMON_MASK, \
                                    LLU_CAST event->options[j].value, SETUP_SBOX##number##_MASK) \
                        } \
                        break; \
                    default: \
                        break; \
                } \
            } \
        } \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags)); \
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_SBOX##number) \
    }

#define NEX_FREEZE_UNCORE \
    if (haveLock && (eventSet->regTypeMask & ~(0xF))) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, &tmp)); \
        tmp &= ~(1<<28); \
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, LLU_CAST tmp, FREEZE_UNCORE) \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, tmp)); \
    }


int perfmon_setupCounterThread_nehalemEX(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t fixed_flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        haveLock = 1;
    }
    if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    if (haveLock && (eventSet->regTypeMask & ~(0xF)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, 0x0ULL, FREEZE_UNCORE)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, 0x0ULL));
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        flags = 0x0ULL;
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = counter_map[index].configRegister;
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        uint64_t offcore_flags = 0x0ULL;
        switch (counter_map[index].type)
        {
            case PMC:
                if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)))
                {
                    flags |= (1<<22)|(1<<16);
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
                        for (int j=0;j<event->numberOfOptions;j++)
                        {
                            switch (event->options[j].type)
                            {
                                case EVENT_OPTION_EDGE:
                                    flags |= (1<<18);
                                    break;
                                case EVENT_OPTION_INVERT:
                                    flags |= (1<<23);
                                    break;
                                case EVENT_OPTION_COUNT_KERNEL:
                                    flags |= (1<<17);
                                    break;
                                case EVENT_OPTION_MATCH0:
                                    offcore_flags |= (event->options[j].value & 0xFF)<<7;
                                    break;
                                case EVENT_OPTION_MATCH1:
                                    offcore_flags |= (event->options[j].value & 0xFF);
                                    break;
                                default:
                                    break;
                            }
                        }
                    }
                    if (event->eventId == 0xB7)
                    {
                        if (offcore_flags == 0x0ULL)
                        {
                            offcore_flags = ((event->cfgBits & 0xFF)<<7) | (event->cmask & 0xFF);
                        }
                        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE);
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, offcore_flags));
                    }
                    if (event->eventId == 0xBB)
                    {
                        if (offcore_flags == 0x0ULL)
                        {
                            offcore_flags = ((event->cfgBits & 0xFF)<<7) | (event->cmask & 0xFF);
                        }
                        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP1, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE);
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP1, offcore_flags));
                    }

                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
                    VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_PMC)
                }
                break;

            case FIXED:
                if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)))
                {
                    fixed_flags |= (0x2 << (4*index));
                }
                break;

            case MBOX0:
                NEX_SETUP_MBOX(0);
                break;

            case MBOX1:
                NEX_SETUP_MBOX(1);
                break;

            case BBOX0:
                NEX_SETUP_BBOX(0);
                break;
            case BBOX1:
                NEX_SETUP_BBOX(1);
                break;

            case RBOX0:
                NEX_SETUP_RBOX(0);
                break;

            case RBOX1:
                NEX_SETUP_RBOX(1);
                break;

            case CBOX0:
                NEX_SETUP_CBOX(0);
                break;
            case CBOX1:
                NEX_SETUP_CBOX(1);
                break;
            case CBOX2:
                NEX_SETUP_CBOX(2);
                break;
            case CBOX3:
                NEX_SETUP_CBOX(3);
                break;
            case CBOX4:
                NEX_SETUP_CBOX(4);
                break;
            case CBOX5:
                NEX_SETUP_CBOX(5);
                break;
            case CBOX6:
                NEX_SETUP_CBOX(6);
                break;
            case CBOX7:
                NEX_SETUP_CBOX(7);
                break;

            case SBOX0:
                NEX_SETUP_SBOX(0);
                break;
            case SBOX1:
                NEX_SETUP_SBOX(1);
                break;

            case WBOX:
                if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(WBOX)))
                {
                    flags |= (1<<22); /* set enable bit */
                    flags |= (event->umask<<8) + event->eventId;
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags));
                    VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_WBOX)
                }
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
                    flags |= (1<<22); /* set enable bit */
                    flags |= event->eventId;
                    for (int j=0;j<event->numberOfOptions;j++)
                    {
                        if (event->options[j].type == EVENT_OPTION_EDGE)
                        {
                            flags |= (1<<18);
                            break;
                        }
                    }
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags));
                    VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_UBOX)
                }
                break;

            default:
                /* should never be reached */
                break;
        }
    }

    if (fixed_flags != 0x0ULL)
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_FIXED_CTR_CTRL, LLU_CAST fixed_flags, SETUP_FIXED);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_FIXED_CTR_CTRL, fixed_flags));
    }
    return 0;
}

#define NEX_RESET_ALL_UNCORE_COUNTERS \
    if (haveLock && (eventSet->regTypeMask & ~(0xF))) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, &tmp)); \
        tmp |= (1<<29); \
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, LLU_CAST tmp, RESET_ALL_UNCORE_COUNTERS); \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, tmp)); \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, 0x0U)); \
    }

#define NEX_UNFREEZE_UNCORE \
    if (haveLock && (eventSet->regTypeMask & ~(0xF))) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, &tmp)); \
        tmp |= (1<<28); \
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
    uint64_t flags = 0x0ULL;
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
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t counter2 = counter_map[index].counterRegister2;
            int reg_offset = 0;
            switch (counter_map[index].type)
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
                        uflags[WBOX] |= (1<<31);
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
        if (tmp & (1<<offset)) \
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
        if (tmp & (1<<offset)) \
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
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = counter_map[index].configRegister;
            switch (counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter_map[index].counterRegister, 
                                                    &counter_result));
                    NEX_CHECK_OVERFLOW(PMC, index-cpuid_info.perf_num_fixed_ctr);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    VERBOSEPRINTREG(cpu_id, counter_map[index].counterRegister, LLU_CAST counter_result, READ_PMC);
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter_map[index].counterRegister, 
                                                    &counter_result));
                    NEX_CHECK_OVERFLOW(PMC, index+32);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    VERBOSEPRINTREG(cpu_id, counter_map[index].counterRegister, LLU_CAST counter_result, READ_FIXED);
                    break;
                default:
                    if(haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(counter_map[index].type)))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter_map[index].counterRegister, &counter_result));
                        NEX_CHECK_UNCORE_OVERFLOW(counter_map[index].type, getCounterTypeOffset(index));
                        eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                        VERBOSEPRINTREG(cpu_id, counter_map[index].counterRegister, LLU_CAST counter_result, READ_UNCORE);
                    }
                    break;
            }
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, 0x0ULL));
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
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = counter_map[index].configRegister;
            uint64_t counter = counter_map[index].counterRegister;
            switch (counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                    NEX_CHECK_OVERFLOW(PMC, index-cpuid_info.perf_num_fixed_ctr);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, READ_PMC);
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                    NEX_CHECK_OVERFLOW(PMC, index+32);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, READ_FIXED);
                    break;
                default:
                    if(haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(counter_map[index].type)))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                        NEX_CHECK_UNCORE_OVERFLOW(counter_map[index].type, getCounterTypeOffset(index));
                        eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                        VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, READ_UNCORE);
                    }
                    break;
            }
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
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t ovf_values_core = (1ULL<<63)|(1ULL<<62);

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = counter_map[index].configRegister;
        VERBOSEPRINTREG(cpu_id, reg, LLU_CAST 0x0ULL, READ_UNCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, 0x0ULL));
        switch (counter_map[index].type)
        {
            case PMC:
                ovf_values_core |= (1ULL<<(index-cpuid_info.perf_num_fixed_ctr));
                if (event->eventId == 0xB7)
                {
                    VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, 0x0ULL, CLEAR_OFFCORE_RESP0);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, 0x0ULL));
                }
                else if (event->eventId == 0xBB)
                {
                    VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP1, 0x0ULL, CLEAR_OFFCORE_RESP1);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP1, 0x0ULL));
                }
                break;
            case FIXED:
                ovf_values_core |= (1ULL<<(index+32));
                break;
            default:
                break;
        }
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, ovf_values_core, CLEAR_OVF_CTRL);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, ovf_values_core));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, CLEAR_PMC_AND_FIXED_CTRL);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }

    if (haveLock && (eventSet->regTypeMask & ~(0xF)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_OVF_CTRL, 0x0ULL, CLEAR_UNCORE_OVF);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_OVF_CTRL, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, 0x0ULL, CLEAR_UNCORE_CTRL);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, 0x0ULL));
    }
    return 0;
}
