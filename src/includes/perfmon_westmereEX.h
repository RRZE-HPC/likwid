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

#include <perfmon_westmereEX_events.h>
#include <perfmon_westmereEX_counters.h>
#include <error.h>
#include <affinity.h>


static int perfmon_numCountersWestmereEX = NUM_COUNTERS_WESTMEREEX;
static int perfmon_numArchEventsWestmereEX = NUM_ARCH_EVENTS_WESTMEREEX;


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int perfmon_init_westmereEX(int cpu_id)
{
    lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &tile_lock[affinity_thread2tile_lookup[cpu_id]], cpu_id);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL));
    return 0;
}

uint32_t wex_fixed_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
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

int wex_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    uint64_t offcore_flags = 0x0ULL;
    int haveLock = 0;
    if (tile_lock[affinity_thread2tile_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    flags = (1ULL<<22)|(1ULL<<16);
    /* Intel with standard 8 bit event mask: [7:0] */
    flags |= (event->umask<<8) + event->eventId;

    /* set custom cfg and cmask */
    if ((event->cfgBits != 0) &&
        (event->eventId != 0xB7) &&
        (event->eventId != 0xBB))
    {
        flags |= ((event->cmask<<8) + event->cfgBits)<<16;
    }

    if (event->numberOfOptions > 0)
    {
        for(j = 0; j < event->numberOfOptions; j++)
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
    if ((haveLock) && (event->eventId == 0xB7))
    {
        if ((event->cfgBits != 0xFF) && (event->cmask != 0xFF))
        {
            offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);
        }
        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, offcore_flags));
    }
    if ((haveLock) && (event->eventId == 0xBB))
    {
        if ((event->cfgBits != 0xFF) && (event->cmask != 0xFF))
        {
            offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);
        }
        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP1, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP1, offcore_flags));
    }
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, SETUP_PMC);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister , flags));
    return 0;
}

int wex_bbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    RegisterType type = counter_map[index].type;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags = 0x1ULL;
    flags |=  (event->eventId<<1);
    if (event->numberOfOptions > 0)
    {
        for(j = 0; j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_MATCH0:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister1, event->options[j].value));
                    VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister1, event->options[j].value, SETUP_BBOX_MATCH);
                    break;
                case EVENT_OPTION_MASK0:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister2, event->options[j].value));
                    VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister2, event->options[j].value, SETUP_BBOX_MASK);
                    break;
                default:
                    break;
            }
        }
    }
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_BBOX);
    return 0;
}

int wex_uncore_box_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags = (1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;
    if (event->numberOfOptions > 0)
    {
        for(j = 0; j < event->numberOfOptions; j++)
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
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_BOX);
    return 0;
}


int wex_sbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;
    int write_mm_cfg = 0;
    RegisterType type = counter_map[index].type;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags = (1ULL<<22);
    flags |= (event->umask<<8) + event->eventId;
    if (event->numberOfOptions > 0)
    {
        for(j = 0; j < event->numberOfOptions; j++)
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
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister1,event->options[j].value));
                        VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister1, event->options[j].value, SETUP_SBOX_MATCH);
                        write_mm_cfg = 1;
                    }
                    break;
                case EVENT_OPTION_MASK0:
                    if (event->eventId == 0x0)
                    {
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[type].filterRegister2,event->options[j].value));
                        VERBOSEPRINTREG(cpu_id, box_map[type].filterRegister1, event->options[j].value, SETUP_SBOX_MASK);
                        write_mm_cfg = 1;
                    }
                    break;
                default:
                    break;
            }
        }
    }
    if (write_mm_cfg && event->eventId == 0x0)
    {
        if (type == SBOX0)
        {
            VERBOSEPRINTREG(cpu_id, MSR_S0_PMON_MM_CFG, (1ULL<<63), SETUP_SBOX_MATCH_CTRL);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_S0_PMON_MM_CFG ,(1ULL<<63)));
        }
        else if (type == SBOX1)
        {
            VERBOSEPRINTREG(cpu_id, MSR_S1_PMON_MM_CFG, (1ULL<<63), SETUP_SBOX_MATCH_CTRL);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_S1_PMON_MM_CFG ,(1ULL<<63)));
        }
    }
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_SBOX);
    return 0;
}

int wex_ubox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    int j;
    uint64_t flags = 0x0ULL;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags = (1ULL<<22);
    flags |= (event->eventId & 0xFF);
    if (event->numberOfOptions > 0)
    {
        for(j = 0; j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<18);
                    break;
                default:
                    break;
            }
        }
    }
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, UBOX_CTRL);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister , flags));
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
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ADDR_MATCH, (event->options[j].value & 0x3FFFFFFFFULL))); \
                        VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, (event->options[j].value & 0x3FFFFFFFFULL), MBOX##number##_ADDR_MATCH) \
                        break; \
                    case EVENT_OPTION_MASK0: \
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ADDR_MASK, (event->options[j].value & 0x1FFFFFFC0ULL)>>6)); \
                        VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, (event->options[j].value & 0x1FFFFFFC0ULL)>>6, MBOX##number##_ADDR_MASK) \
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
                            VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_DSP, dsp_flags, MBOX##number##_DSP); \
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
                          iss_flags |= (event->umask & 0x7ULL)<<4;   \
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
                          CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_MAP, &map_flags));   \
                          VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_MAP, map_flags, MBOX##number##_MAP_READ); \
                          map_flags |= (event->umask & 0x7ULL)<<6;   \
                          CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_MAP, map_flags));   \
                          VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_MAP, map_flags, MBOX##number##_MAP); \
                        }   \
                        break;   \
                }    \
                break;   \
            case 0x02: /* DRAM_CMD: PLD/ISS */   \
                flags |= (event->eventId & 0x1FULL)<<9;   \
                {   \
                    uint64_t pld_flags = 0x0ULL;   \
                    uint64_t iss_flags = 0x0ULL;   \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_PLD, &pld_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_PLD, pld_flags, MBOX##number##_PLD_READ); \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_ISS, &iss_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ISS, iss_flags, MBOX##number##_ISS_READ); \
                    pld_flags |= (event->umask & 0x1FULL)<<8;   \
                    if ((event->cmask & 0xFULL) != 0)   \
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
            case 0x04: /* DRAM_MISC: PLD */   \
                flags |= (event->eventId & 0x1FULL)<<9;   \
                {   \
                    uint64_t pld_flags = 0x0ULL;   \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_PLD, &pld_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_PLD, pld_flags, MBOX##number##_PLD_READ); \
                    switch (event->cmask)   \
                    {   \
                        case 0x0:   \
                            pld_flags |= (1ULL<<16);   \
                            pld_flags |= (event->umask & 0x1FULL)<<19;   \
                            break;   \
                        case 0x1:   \
                            pld_flags |= (event->umask & 0x1ULL)<<18;   \
                            break;   \
                        case 0x2:   \
                            pld_flags |= (event->umask & 0x1ULL)<<17;   \
                            break;   \
                        case 0x3:   \
                            pld_flags |= (event->umask & 0x1ULL)<<7;   \
                            break;   \
                    }   \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_PLD, pld_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_PLD, pld_flags, MBOX##number##_PLD); \
                }   \
                break;   \
            case 0x05: /* FRM_TYPE: ISS */   \
                flags |= (event->eventId & 0x1FULL)<<9;   \
                {   \
                    uint64_t iss_flags = 0x0ULL;   \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_ISS, &iss_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ISS, iss_flags, MBOX##number##_ISS_READ); \
                    iss_flags |= (event->umask & 0xFULL);   \
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
                    fvc_flags |= (event->umask & 0x7ULL)<<12;   \
                    if (event->umask == 0x5)   \
                    {   \
                        fvc_flags |= (event->cmask & 0x7ULL)<<6;   \
                    }   \
                    else   \
                    {   \
                        fvc_flags |= (event->cmask & 0x7ULL)<<9;   \
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
                    fvc_flags |= (event->umask & 0x7ULL)<<15;   \
                    if (event->umask == 0x5)   \
                    {   \
                        fvc_flags |= (event->cmask & 0x7ULL)<<6;   \
                    }   \
                    else   \
                    {   \
                        fvc_flags |= (event->cmask & 0x7ULL)<<9;   \
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
                    fvc_flags |= (event->umask & 0x7ULL)<<18;   \
                    if (event->umask == 0x5)   \
                    {   \
                        fvc_flags |= (event->cmask & 0x7ULL)<<6;   \
                    }   \
                    else   \
                    {   \
                        fvc_flags |= (event->cmask & 0x7ULL)<<9;   \
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
                    fvc_flags |= (event->umask & 0x7ULL)<<21;   \
                    if (event->umask == 0x5)   \
                    {   \
                        fvc_flags |= (event->cmask & 0x7ULL)<<6;   \
                    }   \
                    else   \
                    {   \
                        fvc_flags |= (event->cmask & 0x7ULL)<<9;   \
                    }   \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ZDP, fvc_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ZDP, fvc_flags, MBOX##number##_FVC_EV3); \
                }   \
                break;   \
            case 0x0A: /* ISS_SCHED: ISS */   \
                flags |= (event->eventId & 0x1FULL)<<9;   \
                {   \
                    uint64_t iss_flags = 0x0ULL;   \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_ISS, &iss_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ISS, iss_flags, MBOX##number##_ISS_READ); \
                    iss_flags |= (event->umask & 0x1ULL)<<10;   \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_ISS, iss_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_ISS, iss_flags, MBOX##number##_ISS); \
                }   \
                break;   \
            case 0x0B: /* PGT_PAGE_EV: PGT */   \
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
            case 0x0C: /* PGT_PAGE_EV2: PGT */   \
                flags |= (event->eventId & 0x1FULL)<<9;   \
                {   \
                    uint64_t pgt_flags = 0x0ULL;   \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_PGT, &pgt_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_PGT, pgt_flags, MBOX##number##_PGT_READ); \
                    pgt_flags |= (event->umask & 0x1ULL)<<11;   \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_PGT, pgt_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_PGT, pgt_flags, MBOX##number##_PGT); \
                }   \
                break;   \
            case 0x0D: /* THERM_TRP_DN: THR */   \
                flags |= (event->eventId & 0x1FULL)<<9;   \
                {   \
                    uint64_t thr_flags = 0x0ULL;   \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_MSC_THR, &thr_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_MSC_THR, thr_flags, MBOX##number##_PGT_READ); \
                    thr_flags |= (event->umask & 0x3ULL)<<9;   \
                    if (event->cmask == 0x0) \
                    { \
                        thr_flags |= (1ULL<<3);   \
                    } \
                    else \
                    { \
                        thr_flags &= ~(1ULL<<3);   \
                        thr_flags |= (event->cmask & 0x7ULL)<<4;   \
                    } \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_MSC_THR, thr_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_MSC_THR, thr_flags, MBOX##number##_THR); \
                }   \
                break;   \
            case 0x0E: /* THERM_TRP_UP: THR */   \
                flags |= (event->eventId & 0x1FULL)<<9;   \
                if (event->cmask == 0x0) \
                {   \
                    uint64_t thr_flags = 0x0ULL;   \
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_M##number##_PMON_MSC_THR, &thr_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_MSC_THR, thr_flags, MBOX##number##_PGT_READ); \
                    thr_flags |= (event->umask & 0x3ULL)<<7;   \
                    if (event->cmask == 0x0) \
                    { \
                        thr_flags |= (1ULL<<3);   \
                    } \
                    else \
                    { \
                        thr_flags &= ~(1ULL<<3);   \
                        thr_flags |= (event->cmask & 0x7ULL)<<4;   \
                    } \
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_M##number##_PMON_MSC_THR, thr_flags));   \
                    VERBOSEPRINTREG(cpu_id, MSR_M##number##_PMON_MSC_THR, thr_flags, MBOX##number##_THR); \
                }   \
                break;   \
        } \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags)); \
    }

/* RBOX macros */
#define WEX_SETUP_RBOX(number)  \
    if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(RBOX##number))) \
    { \
        flags = 0x01ULL; /* set local enable flag */ \
        switch (event->eventId) {  \
            case 0x00:  \
                flags |= (event->umask & 0x1FULL)<<1; /* configure sub register */   \
                {  \
                    uint64_t iperf_flags = 0x0ULL;   \
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
                    uint64_t qlx_flags = 0x0ULL;   \
                    qlx_flags |= (event->cfgBits & 0xFULL); /* configure event */  \
                    if (event->cmask) qlx_flags |= (event->cmask & 0xFULL)<<4;  \
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
    }


int wex_uncore_freeze(int cpu_id, PerfmonEventSet* eventSet, int flags)
{
    uint64_t freeze_flags = 0x0ULL;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (eventSet->regTypeMask & ~(0xF))
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, &freeze_flags));
        freeze_flags &= ~(1ULL<<28);
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, LLU_CAST freeze_flags, FREEZE_UNCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, freeze_flags));
    }
    if (flags != FREEZE_FLAG_ONLYFREEZE)
    {
        if (flags & FREEZE_FLAG_CLEAR_CTR)
        {
            uint64_t clear_flags = 0x0ULL;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, &clear_flags));
            clear_flags |= 29;
            VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, LLU_CAST freeze_flags, CLEAR_UNCORE_CTR);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, freeze_flags));
        }
        else if (flags & FREEZE_FLAG_CLEAR_CTL)
        {
            for (int i=0;i < eventSet->numberOfEvents;i++)
            {
                uint32_t reg = counter_map[eventSet->events[i].index].configRegister;
                if (reg != 0x0ULL)
                {
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, 0x0ULL));
                    VERBOSEPRINTREG(cpu_id, reg, 0x0ULL, CLEAR_UNCORE_CTL);
                }
            }
        }

    }
    return 0;
}

int wex_uncore_unfreeze(int cpu_id, PerfmonEventSet* eventSet, int flags)
{
    uint64_t unfreeze_flags = 0x0ULL;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (flags != FREEZE_FLAG_ONLYFREEZE)
    {
        if (flags & FREEZE_FLAG_CLEAR_CTR)
        {
            uint64_t clear_flags = 0x0ULL;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, &clear_flags));
            clear_flags |= 29;
            VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, LLU_CAST clear_flags, CLEAR_UNCORE_CTR);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, clear_flags));
        }
        else if (flags & FREEZE_FLAG_CLEAR_CTL)
        {
            for (int i=0;i < eventSet->numberOfEvents;i++)
            {
                uint32_t reg = counter_map[eventSet->events[i].index].configRegister;
                if (reg != 0x0ULL)
                {
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, 0x0ULL));
                    VERBOSEPRINTREG(cpu_id, reg, 0x0ULL, CLEAR_UNCORE_CTL);
                }
            }
        }
    }
    if (eventSet->regTypeMask & ~(0xF))
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, &unfreeze_flags));
        unfreeze_flags |= (1ULL<<28);
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, LLU_CAST unfreeze_flags, UNFREEZE_UNCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, unfreeze_flags));
    }
    return 0;
}

#define WEX_RESET_OVF_BOX(id) \
    if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(id))) \
    { \
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[id].ovflRegister, 0xFFFFFFFF)); \
    }


int perfmon_setupCounterThread_westmereEX(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t fixed_flags = 0x0ULL;
    uint64_t ubox_flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint32_t uflags[NUM_UNITS] = { [0 ... NUM_UNITS-1] = 0x0U };

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (eventSet->regTypeMask & (REG_TYPE_MASK(FIXED)|REG_TYPE_MASK(PMC)))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, 0x0ULL));
    }

    if (haveLock && (eventSet->regTypeMask & ~(0xF)))
    {
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
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = counter_map[index].configRegister;
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        flags = 0x0ULL;
        switch (type)
        {
            case PMC:
                wex_pmc_setup(cpu_id, index, event);
                break;

            case FIXED:
                fixed_flags |= wex_fixed_setup(cpu_id, index, event);
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
            case BBOX1:
                wex_bbox_setup(cpu_id, index, event);
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
                wex_uncore_box_setup(cpu_id, index, event);
                break;

            case WBOX0FIX:
                if (haveLock && eventSet->regTypeMask & (REG_TYPE_MASK(WBOX0FIX)))
                {
                    flags = 0x1;
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags));
                    VERBOSEPRINTREG(cpu_id, reg, LLU_CAST flags, WBOX0FIX_CTRL);
                    eventSet->regTypeMask |= REG_TYPE_MASK(WBOX);
                }
                break;

            case UBOX:
                wex_ubox_setup(cpu_id, index, event);
                ubox_flags = 0x1ULL;

            case SBOX0:
            case SBOX1:
                wex_sbox_setup(cpu_id, index, event);
                break;
            default:
                break;
        }
        if (type != WBOX0FIX)
        {
            uflags[type] |= (1U<<getCounterTypeOffset(index));
        }
        else
        {
            uflags[WBOX] |= (1<<31);
        }
    }

    if (haveLock && (eventSet->regTypeMask & ~(0xF)))
    {
        for ( int i=0; i<NUM_UNITS; i++ )
        {
            if ((uflags[i] != 0x0ULL) && (i != WBOX0FIX))
            {
                VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, box_map[i].ctrlRegister, uflags[i], CLEAR_CTL);
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[i].ctrlRegister, uflags[i]));
                VERBOSEPRINTPCIREG(cpu_id, MSR_DEV, box_map[i].ovflRegister, uflags[i], CLEAR_OVF_CTL);
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[i].ovflRegister, uflags[i]));
            }
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


int perfmon_startCountersThread_westmereEX(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t core_ctrl_flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    //wex_uncore_freeze(cpu_id, eventSet, FREEZE_FLAG_CLEAR_CTR);

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
                default:
                    break;
            }
        }
    }


    wex_uncore_unfreeze(cpu_id, eventSet, FREEZE_FLAG_CLEAR_CTR);

    /* Finally enable counters */
    if (eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST core_ctrl_flags, GLOBAL_CTRL);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, core_ctrl_flags));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<63)|(1ULL<<62)|core_ctrl_flags));
    }
    return 0;
}

#define WEX_CHECK_OVERFLOW(id, offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t tmp = 0x0ULL; \
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, box_map[id].statusRegister, &tmp)); \
        if (tmp & (1ULL<<offset)) \
        { \
            eventSet->events[i].threadCounter[thread_id].overflows++; \
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[id].ovflRegister, (1ULL<<offset))); \
        } \
    }

#define WEX_CLEAR_OVERFLOW(id, offset) \
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[id].ctrlRegister, (1<<offset)));


#define WEX_CHECK_UNCORE_OVERFLOW(id, offset) \
    if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData) \
    { \
        uint64_t tmp = 0x0ULL; \
        int check_local = 0; \
        if ((id == SBOX0) || (id == SBOX1) || (id == WBOX) || (id == UBOX)) \
        { \
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_STATUS, &tmp)); \
            int gl_offset = -1; \
            switch (id) \
            { \
                case UBOX: \
                    gl_offset = 0; \
                    break; \
                case WBOX: \
                    gl_offset = 1; \
                    break; \
                case SBOX1: \
                    gl_offset = 2; \
                    break; \
                case SBOX0: \
                    gl_offset = 3; \
                    break; \
                default: \
                    break; \
            } \
            if ((gl_offset != -1) && (tmp & (1ULL<<gl_offset))) \
            { \
                check_local = 1; \
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_OVF_CTRL, (1ULL<<gl_offset))); \
            } \
        } \
        else \
        { \
            check_local = 1; \
        } \
        if (check_local) \
        { \
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, box_map[id].statusRegister, &tmp)); \
            if (tmp & (1ULL<<offset)) \
            { \
                eventSet->events[i].threadCounter[thread_id].overflows++; \
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, box_map[id].ovflRegister, (1ULL<<offset))); \
            } \
        } \
    }

int perfmon_stopCountersThread_westmereEX(int thread_id, PerfmonEventSet* eventSet)
{
    int i;
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
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    wex_uncore_freeze(cpu_id, eventSet, FREEZE_FLAG_CLEAR_CTL);

    for (i = 0; i < eventSet->numberOfEvents; i++)
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
            uint64_t counter1 = counter_map[index].counterRegister;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    WEX_CHECK_OVERFLOW(PMC, index-cpuid_info.perf_num_fixed_ctr);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC);
                    break;
                case FIXED:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    WEX_CHECK_OVERFLOW(PMC, index+32);
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED);
                    break;
                default:
                    if(haveLock && (eventSet->regTypeMask & REG_TYPE_MASK(type)))
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                        WEX_CHECK_UNCORE_OVERFLOW(type, index);
                        VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_UNCORE);
                    }
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
        }
    }

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
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, &core_ctrl_flags));
    }
    wex_uncore_freeze(cpu_id, eventSet, FREEZE_FLAG_ONLYFREEZE);

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
            uint64_t counter1 = counter_map[index].counterRegister;
            if (type > UNCORE)
            {
                if(haveLock)
                {
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    WEX_CHECK_UNCORE_OVERFLOW(counter_map[index].type, getCounterTypeOffset(index));
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_UNCORE);
                }
            }
            else if (type == FIXED)
            {
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                WEX_CHECK_OVERFLOW(PMC, index+32);
                VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_FIXED);
            }
            else if (type == PMC)
            {
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                WEX_CHECK_OVERFLOW(PMC, index-cpuid_info.perf_num_fixed_ctr);
                VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST counter_result, READ_PMC);
            }
            eventSet->events[i].threadCounter[thread_id].counterData = field64(counter_result, 0, box_map[type].regWidth);
        }
    }

    wex_uncore_unfreeze(cpu_id, eventSet, FREEZE_FLAG_ONLYFREEZE);
    if ((eventSet->regTypeMask & (REG_TYPE_MASK(PMC)|REG_TYPE_MASK(FIXED))) && (core_ctrl_flags != 0x0ULL))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, core_ctrl_flags));
    }
    return 0;
}


int perfmon_finalizeCountersThread_westmereEX(int thread_id, PerfmonEventSet* eventSet)
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
        uint64_t reg = counter_map[index].configRegister;
        PciDeviceIndex dev = counter_map[index].device;
        switch (type)
        {
            case PMC:
                ovf_values_core |= (1ULL<<(index-cpuid_info.perf_num_fixed_ctr));
                if ((haveTileLock) && (eventSet->events[i].event.eventId == 0xB7))
                {
                    VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, 0x0ULL, CLEAR_OFFCORE_RESP0);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, 0x0ULL));
                }
                else if ((haveTileLock) && (eventSet->events[i].event.eventId == 0xBB))
                {
                    VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP1, 0x0ULL, CLEAR_OFFCORE_RESP1);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP1, 0x0ULL));
                }
                break;
            case FIXED:
                ovf_values_core |= (1ULL<<(index+32));
                break;
            default:
                if (((haveLock) && (type > UNCORE)))
                {
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, 0x0ULL));
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
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, CLEAR_PMC_AND_FIXED_CTL);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, ovf_values_core, CLEAR_PMC_AND_FIXED_OVERFLOW);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, ovf_values_core));
    }
    if (haveLock && (eventSet->regTypeMask & ~(0xF)))
    {
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_CTRL, 0x0ULL, CLEAR_UNCORE_CTL);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_CTRL, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, MSR_U_PMON_GLOBAL_OVF_CTRL, 0x0ULL, CLEAR_UNCORE_OVERFLOW);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_U_PMON_GLOBAL_OVF_CTRL, 0x0ULL));
    }
    return 0;
}
