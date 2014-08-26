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
#include <perfmon_westmereEX_counters.h>
#include <error.h>
#include <affinity.h>


static int perfmon_numCountersWestmereEX = NUM_COUNTERS_WESTMEREEX;
static int perfmon_numArchEventsWestmereEX = NUM_ARCH_EVENTS_WESTMEREEX;


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int perfmon_init_westmereEX(int cpu_id)
{
    uint64_t flags = 0x0ULL;
    if ( cpuid_info.model == WESTMERE_EX )
    {
        lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    }
    return 0;
}

/* MBOX macros */

#define WEX_SETUP_MBOX(number)  \
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
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_ADDR_MATCH, event->options[j].value)); \
                        VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, event->options[j].value, MBOX##number##_ADDR_MATCH) \
                        break; \
                    case EVENT_OPTION_MASK0: \
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_ADDR_MASK, event->options[j].value)); \
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
                            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_DSP, dsp_flags));   \
                            VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, dsp_flags, MBOX##number##_DSP) \
                        }   \
                        break;   \
                    case 0x01: /* CYCLES_SCHED_MODE: ISS */   \
                        {   \
                          uint32_t iss_flags = 0x0UL;   \
                          iss_flags |= (event->umask<<4);   \
                          CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_ISS, iss_flags));   \
                          VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, iss_flags, MBOX##number##_ISS) \
                        }    \
                        break;   \
                    case 0x05: /* CYCLES_PGT_STATE: PGT */   \
                        {   \
                         uint32_t pgt_flags = 0x0UL;   \
                         pgt_flags |= (event->umask<<6);   \
                         CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_PGT, pgt_flags));   \
                         VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, pgt_flags, MBOX##number##_PGT) \
                        }    \
                        break;   \
                    case 0x06: /* BCMD_SCHEDQ_OCCUPANCY: MAP */   \
                        {   \
                          uint32_t map_flags = 0x0UL;   \
                          map_flags |= (event->umask<<6);   \
                          CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_MAP, map_flags));   \
                          VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, map_flags, MBOX##number##_MAP) \
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
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_PLD, pld_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, pld_flags, MBOX##number##_PLD) \
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_ISS, iss_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, iss_flags, MBOX##number##_ISS) \
                }   \
                break;   \
            case 0x03: /* DSP_FILL: DSP */   \
                flags |= (event->eventId<<9);   \
                {   \
                    uint64_t dsp_flags = 0x0ULL;   \
                    dsp_flags |= (event->umask<<7);   \
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_DSP, dsp_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, dsp_flags, MBOX##number##_DSP) \
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
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_PLD, pld_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, pld_flags, MBOX##number##_PLD) \
                }   \
                break;   \
            case 0x05: /* FRM_TYPE: ISS */   \
                flags |= (event->eventId<<9);   \
                {   \
                    uint32_t iss_flags = 0x0UL;   \
                    iss_flags |= event->umask;   \
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_ISS, iss_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, iss_flags, MBOX##number##_ISS) \
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
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags, MBOX##number##_FVC) \
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
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags, MBOX##number##_FVC) \
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
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags, MBOX##number##_FVC) \
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
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags, MBOX##number##_FVC) \
                }   \
                break;   \
            case 0x0A: /* ISS_SCHED: ISS */   \
                flags |= (event->eventId<<9);   \
                {   \
                    uint32_t iss_flags = 0x0UL;   \
                    iss_flags |= (event->umask<<10);   \
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_ISS, iss_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, iss_flags, MBOX##number##_ISS) \
                }   \
                break;   \
            case 0x0B: /* PGT_PAGE_EV: PGT */   \
                flags |= (event->eventId<<9);   \
                {   \
                    uint32_t pgt_flags = 0x0UL;   \
                    pgt_flags |= event->umask;   \
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_PGT, pgt_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, pgt_flags, MBOX##number##_PGT) \
                }   \
                break;   \
            case 0x0C: /* PGT_PAGE_EV2: PGT */   \
                flags |= (event->eventId<<9);   \
                {   \
                    uint32_t pgt_flags = 0x0UL;   \
                    pgt_flags |= (event->umask<<11);   \
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_PGT, pgt_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, pgt_flags, MBOX##number##_PGT) \
                }   \
                break;   \
            case 0x0D: /* THERM_TRP_DN: THR */   \
                flags |= (event->eventId<<9);   \
                {   \
                    uint32_t thr_flags = 0x0UL;   \
                    thr_flags |= (1<<3);   \
                    thr_flags |= (event->umask<<9);   \
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_M##number##_PMON_PGT, thr_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, thr_flags, MBOX##number##_THR) \
                }   \
                break;   \
        } \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags)); \
    }

/* RBOX macros */
#define WEX_SETUP_RBOX(number)  \
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
                                   CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_R##number##_PMON_IPERF0_P0, iperf_flags));   \
                        break; \
                        case 0x01: \
                                   CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_R##number##_PMON_IPERF1_P0, iperf_flags));   \
                        break; \
                        case 0x06: \
                                   CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_R##number##_PMON_IPERF0_P1, iperf_flags));   \
                        break; \
                        case 0x07: \
                                   CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_R##number##_PMON_IPERF1_P1, iperf_flags));   \
                        break; \
                        case 0x0C: \
                                   CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_R##number##_PMON_IPERF0_P2, iperf_flags));   \
                        break; \
                        case 0x0D: \
                                   CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_R##number##_PMON_IPERF1_P2, iperf_flags));   \
                        break; \
                        case 0x12: \
                                   CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_R##number##_PMON_IPERF0_P3, iperf_flags));   \
                        break; \
                        case 0x13: \
                                   CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_R##number##_PMON_IPERF1_P3, iperf_flags));   \
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
                                   CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_R##number##_PMON_QLX_P0, qlx_flags));   \
                        break; \
                        case 0x03: \
                                   CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_R##number##_PMON_QLX_P0, (qlx_flags<<8)));   \
                        break; \
                        case 0x08: \
                                   CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_R##number##_PMON_QLX_P0, qlx_flags));   \
                        break; \
                        case 0x09: \
                                   CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_R##number##_PMON_QLX_P1, (qlx_flags<<8)));   \
                        break; \
                        case 0x0E: \
                                   CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_R##number##_PMON_QLX_P0, qlx_flags));   \
                        break; \
                        case 0x0F: \
                                   CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_R##number##_PMON_QLX_P2, (qlx_flags<<8)));   \
                        break; \
                        case 0x14: \
                                   CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_R##number##_PMON_QLX_P0, qlx_flags));   \
                        break; \
                        case 0x15: \
                                   CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_R##number##_PMON_QLX_P3, (qlx_flags<<8)));   \
                        break; \
                    } \
                } \
                break; \
        } \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags)); \
    }

#define WEX_SETUP_BBOX(number) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(BBOX##number))) \
    { \
        flags = 0x1ULL; \
        flags |=  (event->eventId<<1); \
        if (event->numberOfOptions > 0) \
        { \
            for (int j=0; j<event->numberOfOptions; j++) \
            { \
                switch (event->options[j].type) \
                { \
                    case EVENT_OPTION_MATCH0: \
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_B##number##_PMON_MATCH, event->options[j].value)); \
                        VERBOSEPRINTREG(cpu_id, MSR_B##number##_PMON_MATCH, event->options[j].value, BBOX##number_MATCH) \
                        break; \
                    case EVENT_OPTION_MASK0: \
                        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_B##number##_PMON_MASK, event->options[j].value)); \
                        VERBOSEPRINTREG(cpu_id, MSR_B##number##_PMON_MASK, event->options[j].value, BBOX##number_MASK) \
                        break; \
                    default: \
                        break; \
                } \
            } \
        } \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags)); \
        VERBOSEPRINTREG(cpu_id, reg, flags, BBOX##number) \
    } \

#define WEX_SETUP_CBOX(number) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(CBOX##number))) \
    { \
        flags = (1<<22); \
        flags |= (event->umask<<8) + event->eventId; \
        if (event->numberOfOptions > 0) \
        { \
            for (int j=0;j < event->numberOfOptions; j++) \
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
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags)); \
        VERBOSEPRINTREG(cpu_id, reg, flags, CBOX##number##_CTRL) \
    }

#define WEX_SETUP_SBOX(number) \
    if (haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(SBOX##number))) \
    { \
        int write_mm_cfg = 0; \
        flags = (1<<22); \
        flags |= (event->umask<<8) + event->eventId; \
        if (event->numberOfOptions > 0) \
        { \
            for (int j=0;j < event->numberOfOptions; j++) \
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
                            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_S##number##_PMON_MATCH ,event->options[j].value)); \
                            VERBOSEPRINTREG(cpu_id, MSR_S##number##_PMON_MATCH, event->options[j].value, SBOX##number##_MATCH) \
                            write_mm_cfg = 1; \
                        } \
                        break; \
                    case EVENT_OPTION_MASK0: \
                        if (event->eventId == 0x0) \
                        { \
                            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_S##number##_PMON_MASK ,event->options[j].value)); \
                            VERBOSEPRINTREG(cpu_id, MSR_S##number##_PMON_MASK, event->options[j].value, SBOX##number##_MASK) \
                            write_mm_cfg = 1; \
                        } \
                        break; \
                    default: \
                        break; \
                } \
            } \
        } \
        if (write_mm_cfg && event->eventId == 0x0) \
        { \
            VERBOSEPRINTREG(cpu_id, MSR_S##number##_PMON_MM_CFG, (1ULL<<63), SBOX##number##_MATCH_CTRL) \
            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_S##number##_PMON_MM_CFG ,(1ULL<<63))); \
        } \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags)); \
        VERBOSEPRINTREG(cpu_id, reg, flags, SBOX##number##_CTRL) \
    }

#define WEX_FREEZE_UNCORE \
    if (haveLock && (eventSet->regTypeMask & ~(0xF))) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_U_PMON_GLOBAL_CTRL, &tmp)); \
        tmp &= ~(1<<28); \
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, LLU_CAST tmp, FREEZE_UNCORE) \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_U_PMON_GLOBAL_CTRL, tmp)); \
    }

#define WEX_FREEZE_BOX(id) \
    if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(id))) \
    { \
        VERBOSEPRINTREG(cpu_id, westmereEX_box_map[id].ctrlRegister, LLU_CAST 0x0U, FREEZE_BOX_##id) \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, westmereEX_box_map[id].ctrlRegister, 0x0U)); \
    }

#define WEX_RESET_OVF_BOX(id) \
    if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(id))) \
    { \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, westmereEX_box_map[id].ovflRegister, 0xFFFFFFFF)); \
    }


int perfmon_setupCounterThread_westmereEX(int thread_id, PerfmonEventSet* eventSet)
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
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PEBS_ENABLE, 0x0ULL));
    }

    if (haveLock && (eventSet->regTypeMask & ~(0xF)))
    {
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_U_PMON_GLOBAL_CTRL, 0x0ULL));
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = westmereEX_counter_map[index].configRegister;
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        flags = 0x0ULL;
        switch (westmereEX_counter_map[index].type)
        {
            case PMC:
                /* Intel with standard 8 bit event mask: [7:0] */
                flags |= (1<<22)|(1<<16);
                flags |= (event->umask<<8) + event->eventId;

                if (event->cfgBits != 0) /* set custom cfg and cmask */
                {
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
                            default:
                                break;
                        }
                    }
                }
                CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags));
                VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_PMC)
                break;

            case FIXED:
                fixed_flags |= (0x2 << (4*index));
                break;

            case MBOX0:
                WEX_SETUP_MBOX(0);
                VERBOSEPRINTREG(cpu_id, reg, flags, MBOX0_CTRL)
                break;

            case MBOX1:
                WEX_SETUP_MBOX(1);
                VERBOSEPRINTREG(cpu_id, reg, flags, MBOX1_CTRL)
                break;

            case BBOX0:
                WEX_SETUP_BBOX(0);
                VERBOSEPRINTREG(cpu_id, reg, flags, BBOX0_CTRL)
                break;

            case BBOX1:
                WEX_SETUP_BBOX(1);
                VERBOSEPRINTREG(cpu_id, reg, flags, BBOX1_CTRL)
                break;

            case RBOX0:
                WEX_SETUP_RBOX(0)
                VERBOSEPRINTREG(cpu_id, reg, flags, RBOX0_CTRL)
                break;

            case RBOX1:
                WEX_SETUP_RBOX(1)
                VERBOSEPRINTREG(cpu_id, reg, flags, RBOX1_CTRL)
                break;

            case WBOX:
                if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(WBOX)))
                {
                    flags |= (1<<22); /* set enable bit */
                    flags |= (event->umask<<8) + event->eventId;

                    if (event->numberOfOptions > 0)
                    {
                        for (int j=0;j < event->numberOfOptions; j++)
                        {
                            switch (event->options[j].type)
                            {
                                case EVENT_OPTION_EDGE:
                                    flags |= (1<<18);
                                    break;
                                case EVENT_OPTION_INVERT:
                                    flags |= (1<<23);
                                    break;
                                case EVENT_OPTION_THRESHOLD:
                                    flags |= ((event->options[j].value & 0x1F) << 24);
                                    break;
                                default:
                                    break;
                            }
                        }
                    }
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags));
                    VERBOSEPRINTREG(cpu_id, reg, flags, WBOX_CTRL)
                }
                break;  

            case WBOX0FIX:
                if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(WBOX0FIX)))
                {
                    flags = 0x1;
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags));
                    VERBOSEPRINTREG(cpu_id, reg, LLU_CAST flags, WBOX0FIX_CTRL);
                    eventSet->regTypeMask |= REG_TYPE_MASK(WBOX);
                }
                break;

            case UBOX:
                if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(UBOX)))
                {
                    flags = (1<<22);
                    if (event->numberOfOptions > 0)
                    {
                        for (int j=0;j < event->numberOfOptions; j++)
                        {
                            switch (event->options[j].type)
                            {
                                case EVENT_OPTION_EDGE:
                                    flags |= (1<<18);
                                    break;
                                default:
                                    break;
                            }
                        }
                    }
                    VERBOSEPRINTREG(cpu_id, reg, LLU_CAST flags, UBOX_CTRL);
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg , flags));
                }
            case CBOX0:
                WEX_SETUP_CBOX(0);
                break;
            case CBOX1:
                WEX_SETUP_CBOX(1);
                break;
            case CBOX2:
                WEX_SETUP_CBOX(2);
                break;
            case CBOX3:
                WEX_SETUP_CBOX(3);
                break;
            case CBOX4:
                WEX_SETUP_CBOX(4);
                break;
            case CBOX5:
                WEX_SETUP_CBOX(5);
                break;
            case CBOX6:
                WEX_SETUP_CBOX(6);
                break;
            case CBOX7:
                WEX_SETUP_CBOX(7);
                break;
            case CBOX8:
                WEX_SETUP_CBOX(8);
                break;
            case CBOX9:
                WEX_SETUP_CBOX(9);
                break;
            case SBOX0:
                WEX_SETUP_SBOX(0);
                break;
            case SBOX1:
                WEX_SETUP_SBOX(1);
                break;
            default:
                /* should never be reached */
                break;
        }
    }

    if (fixed_flags != 0x0ULL)
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_FIXED_CTR_CTRL, LLU_CAST fixed_flags, SETUP_FIXED);
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, fixed_flags));
    }
    return 0;
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

#define WEX_RESET_ALL_UNCORE_COUNTERS \
    if (haveLock && (eventSet->regTypeMask & ~(0xF))) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_U_PMON_GLOBAL_CTRL, &tmp)); \
        tmp |= (1<<29); \
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, LLU_CAST tmp, RESET_ALL_UNCORE_COUNTERS); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_U_PMON_GLOBAL_CTRL, tmp)); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_U_PMON_GLOBAL_CTRL, 0x0U)); \
    }

#define WEX_UNFREEZE_UNCORE \
    if (haveLock && (eventSet->regTypeMask & ~(0xF))) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_U_PMON_GLOBAL_CTRL, &tmp)); \
        tmp |= (1<<28); \
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, LLU_CAST tmp, UNFREEZE_UNCORE); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_U_PMON_GLOBAL_CTRL, tmp)); \
    }

#define WEX_UNFREEZE_BOX(id, flags) \
    if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(id))) \
    { \
        VERBOSEPRINTREG(cpu_id, westmereEX_box_map[id].ctrlRegister, LLU_CAST flags, UNFREEZE_BOX); \
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, westmereEX_box_map[id].ctrlRegister, flags)); \
    }

int perfmon_startCountersThread_westmereEX(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t core_ctrl_flags = 0x0ULL;
    uint32_t uflags[NUM_UNITS] = { [0 ... NUM_UNITS-1] = 0x0U };
    int cpu_id = groupSet->threads[thread_id].processorId;

    //CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    WEX_RESET_ALL_UNCORE_COUNTERS;

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
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, counter1, 0x0ULL));
                    core_ctrl_flags |= (1ULL<<(index-OFFSET_PMC));
                    break;
                case FIXED:
                    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, counter1, 0x0ULL));
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
                        for (int j = index-1; j>=0; j--)
                        {
                            if (counter_map[index].type == counter_map[j].type)
                            {
                                reg_offset++;
                            }
                            else
                            {
                                break;
                            }
                        }
                        uflags[counter_map[index].type] |= (1<<reg_offset);
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
                //WEX_RESET_OVF_BOX(i);
                WEX_UNFREEZE_BOX(i, uflags[i]);
            }
        }
    }

    WEX_UNFREEZE_UNCORE;

    /* Finally enable counters */
    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST core_ctrl_flags, GLOBAL_CTRL);
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, core_ctrl_flags));
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x30000000FULL));
    }
    return 0;
}

#define WEX_CHECK_OVERFLOW(id, offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, westmereEX_box_map[id].statusRegister, &tmp)); \
        if (tmp & (1<<offset)) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
        } \
    }

#define WEX_CLEAR_OVERFLOW(id, offset) \
    CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, westmereEX_box_map[id].ctrlRegister, (1<<offset)));


#define WEX_CHECK_UNCORE_OVERFLOW(id, index) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t tmp = 0x0ULL; \
        int offset = 0; \
        for (int j= index-1; j>= NUM_COUNTERS_CORE_WESTMEREEX; j--) \
        { \
            if (counter_map[index].type == counter_map[j].type) \
            { \
                offset++; \
            } \
            else \
            { \
                break; \
            } \
        } \
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, westmereEX_box_map[id].statusRegister, &tmp)); \
        if (tmp & (1<<offset)) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
        } \
    }

int perfmon_stopCountersThread_westmereEX(int thread_id, PerfmonEventSet* eventSet)
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
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST 0x0ULL, GLOBAL_CTRL);
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    WEX_FREEZE_UNCORE;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            uint64_t reg = counter_map[index].configRegister;
            switch (counter_map[index].type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter_map[index].counterRegister, 
                                                    &counter_result));
                    WEX_CHECK_OVERFLOW(PMC, index-OFFSET_PMC);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    VERBOSEPRINTREG(cpu_id, counter_map[index].counterRegister, LLU_CAST counter_result, READ_PMC);
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter_map[index].counterRegister, 
                                                    &counter_result));
                    WEX_CHECK_OVERFLOW(PMC, index+32);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    VERBOSEPRINTREG(cpu_id, counter_map[index].counterRegister, LLU_CAST counter_result, READ_FIXED);
                    break;
                default:
                    if(haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(counter_map[index].type)))
                    {
                        CHECK_MSR_READ_ERROR(msr_read(cpu_id, counter_map[index].counterRegister, &counter_result));
                        WEX_CHECK_UNCORE_OVERFLOW(counter_map[index].type, index);
                        eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                        VERBOSEPRINTREG(cpu_id, counter_map[index].counterRegister, LLU_CAST counter_result, READ_UNCORE);
                    }
                    break;
            }
            CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, reg, 0x0ULL));
        }
    }

#if 0
    CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_PERF_GLOBAL_STATUS, &flags));
    printf ("Status: 0x%llX \n", LLU_CAST flags);
    if((flags & 0x3) || (flags & (0x3ULL<<32)) ) 
    {
        printf ("Overflow occured \n");
    }
#endif
    return 0;
}

int perfmon_readCountersThread_westmereEX(int thread_id, PerfmonEventSet* eventSet)
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
        CHECK_MSR_READ_ERROR(msr_read(cpu_id, MSR_PERF_GLOBAL_CTRL, &core_ctrl_flags));
    }
    WEX_FREEZE_UNCORE;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            if (westmereEX_counter_map[index].type > UNCORE)
            {
                if(haveLock)
                {
                    CHECK_MSR_READ_ERROR(msr_read(cpu_id, westmereEX_counter_map[index].counterRegister, &counter_result));
                    WEX_CHECK_UNCORE_OVERFLOW(counter_map[index].type, index);
                    eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                    VERBOSEPRINTREG(cpu_id, counter_map[index].counterRegister, LLU_CAST counter_result, READ_UNCORE);
                }
            }
            else if (westmereEX_counter_map[index].type == FIXED)
            {
                CHECK_MSR_READ_ERROR(msr_read(cpu_id, westmereEX_counter_map[index].counterRegister, &counter_result));
                WEX_CHECK_OVERFLOW(PMC, index+32);
                eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                VERBOSEPRINTREG(cpu_id, counter_map[index].counterRegister, LLU_CAST counter_result, READ_FIXED);
            }
            else if (westmereEX_counter_map[index].type == PMC)
            {
                CHECK_MSR_READ_ERROR(msr_read(cpu_id, westmereEX_counter_map[index].counterRegister, &counter_result));
                WEX_CHECK_OVERFLOW(PMC, index-OFFSET_PMC);
                eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
                VERBOSEPRINTREG(cpu_id, counter_map[index].counterRegister, LLU_CAST counter_result, READ_PMC);
            }
        }
    }

    WEX_UNFREEZE_UNCORE;
    if ((eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED))) && (core_ctrl_flags != 0x0ULL))
    {
        CHECK_MSR_WRITE_ERROR(msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, core_ctrl_flags));
    }
    return 0;
}

