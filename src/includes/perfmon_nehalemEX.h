/*
 * =======================================================================================
 *
 *      Filename:  perfmon_nehalemEX.h
 *
 *      Description:  Header File of perfmon module for Intel Nehalem EX.
 *
 *      Version:   4.1
 *      Released:  13.6.2016
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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
#include <perfmon_nehalemEX_westmereEX_common.h>
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
    uint64_t reg = counter_map[index].configRegister;

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
    if (event->eventId == 0xB7)
    {
        if ((event->cfgBits != 0xFF) && (event->cmask != 0xFF))
        {
            offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);
        }
        VERBOSEPRINTREG(cpu_id, MSR_OFFCORE_RESP0, LLU_CAST offcore_flags, SETUP_PMC_OFFCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_OFFCORE_RESP0, offcore_flags));
    }
    if (flags != currentConfig[cpu_id][index])
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_PMC)
        currentConfig[cpu_id][index] = flags;
    }

    return 0;
}





int nex_mbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x41ULL;
    uint64_t subflags1 = 0x0ULL;
    uint64_t subflags2 = 0x0ULL;
    int number;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if (((counter_map[index].configRegister& 0xFF0) == 0xCA0) ||
       ((counter_map[index].configRegister& 0xFF0) == 0xCB0))
        number = 0;
    else
        number = 1;

    if (event->numberOfOptions > 0 && (event->cfgBits == 0x02 || event->cfgBits == 0x04))
    {
        for (int j=0; j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_MATCH0:
                    subflags2 = (event->options[j].value & 0x3FFFFFFFFULL);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ADDR_MATCH], subflags2));
                    VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][ADDR_MATCH], subflags2, SETUP_MBOX_ADDR_MATCH);
                    break;
                case EVENT_OPTION_MASK0:
                    subflags2 = ((event->options[j].value & 0x1FFFFFFC0ULL)>>6);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ADDR_MASK], subflags2));
                    VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][ADDR_MASK], subflags2, SETUP_MBOX_ADDR_MASK);
                    break;
                default:
                    break;
            }
        }
        subflags2 = 0x0ULL;
    }
    switch (event->cfgBits)
    {
        case 0x00:
            flags |= (event->eventId & 0x1FULL)<<9; 
            break;
        case 0x01:
            flags |= (1ULL<<7);
            flags |= (event->eventId & 0x7ULL)<<19;
            switch (event->eventId)
            {
                case 0x00:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][DSP], &subflags1));
                    subflags1 |= (event->umask & 0xFULL)<<7;
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][DSP], subflags1));
                    VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][DSP], subflags1, SETUP_MBOX_DSP);
                    break;
                case 0x01:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ISS], &subflags1));
                    subflags1 |= (event->umask & 0x7ULL)<<4;
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ISS], subflags1));
                    VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][ISS], subflags1, SETUP_MBOX_ISS);
                    break;
                case 0x05:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][PGT], &subflags1));
                    subflags1 |= (event->umask & 0x1ULL)<<6;
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][PGT], subflags1));
                    VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][PGT], subflags1, SETUP_MBOX_PGT);
                    break;
                case 0x06:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][MAP], &subflags1));
                    subflags1 |= (event->umask & 0x7ULL)<<6;
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][MAP], subflags1));
                    VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][MAP], subflags1, SETUP_MBOX_MAP);
                    break;
            }
            break;
        case 0x02:
            flags |= (event->eventId & 0x1FULL)<<9;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][PLD], &subflags1));
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ISS], &subflags2));
            subflags1 |= (event->umask & 0x1FULL)<<8;
            if ((event->cmask & 0xF0ULL) != 0)
            {
                subflags1 |= (1ULL<<0);
            }
            if ((event->cmask & 0xFULL) != 0)
            {
                subflags2 |= (event->cmask & 0x7ULL)<<7;
            }
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][PLD], subflags1));
            VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][PLD], subflags1, SETUP_MBOX_PLD);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ISS], subflags2));
            VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][ISS], subflags2, SETUP_MBOX_ISS);
            break;
        case 0x03:
            flags |= (event->eventId & 0x1FULL)<<9;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][DSP], &subflags1));
            subflags1 |= (event->umask & 0xFULL)<<7;
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][DSP], subflags1));
            VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][DSP], subflags1, SETUP_MBOX_DSP);
            break;
        case 0x04:
            flags |= (event->eventId & 0x1FULL)<<9;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][PLD], &subflags1));
            switch (event->cmask)
            {
                case 0x0:
                    subflags1 |= (1ULL<<16);
                    subflags1 |= (event->umask & 0x1FULL)<<19;
                    break;
                case 0x1:
                    subflags1 |= (event->umask & 0x1ULL)<<18;
                    break;
                case 0x2:
                    subflags1 |= (event->umask & 0x1ULL)<<17;
                    break;
                case 0x3:
                    subflags1 |= (event->umask & 0x1ULL)<<7;
                    break;
            }
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][PLD], subflags1));
            VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][PLD], subflags1, SETUP_MBOX_PLD);
            break;
        case 0x05:
            flags |= (event->eventId & 0x1FULL)<<9;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ISS], &subflags1));
            subflags1 |= (event->umask & 0xFULL);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ISS], subflags1));
            VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][ISS], subflags1, SETUP_MBOX_ISS);
            break;
        case 0x06:
            flags |= (event->eventId & 0x1FULL)<<9;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ZDP], &subflags1));
            subflags1 |= (event->umask & 0x7ULL)<<12;
            if (event->umask == 0x5)
            {
                subflags1 |= (event->cmask & 0x7ULL)<<6;
            }
            else
            {
                subflags1 |= (event->cmask & 0x7ULL)<<9;
            }
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ZDP], subflags1));
            VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][ZDP], subflags1, SETUP_MBOX_ZDP);
            break;
        case 0x07:
            flags |= (event->eventId & 0x1FULL)<<9;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ZDP], &subflags1));
            subflags1 |= (event->umask & 0x7ULL)<<15;
            if (event->umask == 0x5)
            {
                subflags1 |= (event->cmask & 0x7ULL)<<6;
            }
            else
            {
                subflags1 |= (event->cmask & 0x7ULL)<<9;
            }
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ZDP], subflags1));
            VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][ZDP], subflags1, SETUP_MBOX_ZDP);
            break;
        case 0x08:
            flags |= (event->eventId & 0x1FULL)<<9;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ZDP], &subflags1));
            subflags1 |= (event->umask & 0x7ULL)<<18;
            if (event->umask == 0x5)
            {
                subflags1 |= (event->cmask & 0x7ULL)<<6;
            }
            else
            {
                subflags1 |= (event->cmask & 0x7ULL)<<9;
            }
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ZDP], subflags1));
            VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][ZDP], subflags1, SETUP_MBOX_ZDP);
            break;
        case 0x09:
            flags |= (event->eventId & 0x1FULL)<<9;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ZDP], &subflags1));
            subflags1 |= (event->umask & 0x7ULL)<<21;
            if (event->umask == 0x5)
            {
                subflags1 |= (event->cmask & 0x7ULL)<<6;
            }
            else
            {
                subflags1 |= (event->cmask & 0x7ULL)<<9;
            }
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ZDP], subflags1));
            VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][ZDP], subflags1, SETUP_MBOX_ZDP);
            break;
        case 0x0A:
            flags |= (event->eventId & 0x1FULL)<<9;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ISS], &subflags1));
            subflags1 |= (event->umask & 0x1ULL)<<10;
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][ISS], subflags1));
            VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][ISS], subflags1, SETUP_MBOX_ISS);
            break;
        case 0x0B:
            flags |= (event->eventId & 0x1FULL)<<9;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][PGT], &subflags1));
            subflags1 |= (event->umask & 0x1ULL);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][PGT], subflags1));
            VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][PGT], subflags1, SETUP_MBOX_PGT);
            break;
        case 0x0C:
            flags |= (event->eventId & 0x1FULL)<<9;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][PGT], &subflags1));
            subflags1 |= (event->umask & 0x1ULL)<<11;
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][PGT], subflags1));
            VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][PGT], subflags1, SETUP_MBOX_PGT);
            break;
        case 0x0D:
            flags |= (event->eventId & 0x1FULL)<<9;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][THR], &subflags1));
            subflags1 |= (event->umask & 0x3ULL)<<9;
            if (event->cmask == 0x0)
            {
                subflags1 |= (1ULL<<3);
            }
            else
            {
                subflags1 &= ~(1ULL<<3);
                subflags1 |= (event->cmask & 0x7ULL)<<4;
            }
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][THR], subflags1));
            VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][THR], subflags1, SETUP_MBOX_THR);
            break;
        case 0x0E:
            flags |= (event->eventId & 0x1FULL)<<9;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][THR], &subflags1));
            subflags1 |= (event->umask & 0x3ULL)<<7;
            if (event->cmask == 0x0)
            {
                subflags1 |= (1ULL<<3);
            }
            else
            {
                subflags1 &= ~(1ULL<<3);
                subflags1 |= (event->cmask & 0x7ULL)<<4;
            }
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_mbox_regs[number][THR], subflags1));
            VERBOSEPRINTREG(cpu_id, nex_wex_mbox_regs[number][THR], subflags1, SETUP_MBOX_THR);
            break;
    }
    if (flags != currentConfig[cpu_id][index])
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_MBOX)
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}



int nex_rbox_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x01ULL;
    uint64_t subflags = 0x0ULL;
    int number;

    if (socket_lock[affinity_core2node_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }
    if ((counter_map[index].configRegister & 0xFF0) == 0xE10)
        number = 0;
    else if ((counter_map[index].configRegister & 0xFF0) == 0xE30)
        number = 1;

    switch (event->eventId) {
        case 0x00:
            flags |= (event->umask & 0x1FULL)<<1;
            subflags |= (event->cfgBits<<event->cmask);
            switch (event->umask)
            {
                case 0x00:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_rbox_regs[number][IPERF0][0], subflags));
                    break;
                case 0x01:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_rbox_regs[number][IPERF1][0], subflags));
                    break;
                case 0x06:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_rbox_regs[number][IPERF0][1], subflags));
                    break;
                case 0x07:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_rbox_regs[number][IPERF1][1], subflags));
                    break;
                case 0x0C:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_rbox_regs[number][IPERF0][2], subflags));
                    break;
                case 0x0D:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_rbox_regs[number][IPERF1][2], subflags));
                    break;
                case 0x12:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_rbox_regs[number][IPERF0][3], subflags));
                    break;
                case 0x13:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_rbox_regs[number][IPERF1][3], subflags));
                    break;
            }
            break;
        case 0x01:
            flags |= (event->umask & 0x1FULL)<<1;
            subflags |= (event->cfgBits & 0xFULL);
            if (event->cmask != 0x0)
            {
                subflags |= (event->cmask & 0xFULL)<<4;
            }
            switch (event->umask)
            {
                case 0x02:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_rbox_regs[number][QLX][0], subflags));
                    break;
                case 0x03:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_rbox_regs[number][QLX][0], (subflags<<8)));
                    break;
                case 0x08:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_rbox_regs[number][QLX][1], subflags));
                    break;
                case 0x09:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_rbox_regs[number][QLX][1], (subflags<<8)));
                    break;
                case 0x0E:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_rbox_regs[number][QLX][2], subflags));
                    break;
                case 0x0F:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_rbox_regs[number][QLX][2], (subflags<<8)));
                    break;
                case 0x14:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_rbox_regs[number][QLX][3], subflags));
                    break;
                case 0x15:
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, nex_wex_rbox_regs[number][QLX][3], (subflags<<8)));
                    break;
            }
            break;
    }
    if (flags != currentConfig[cpu_id][index])
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_RBOX)
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
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
    if (flags != currentConfig[cpu_id][index])
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags));
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_BBOX);
        currentConfig[cpu_id][index] = flags;
    }
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
    if (flags != currentConfig[cpu_id][index])
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags));
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_CBOX);
        currentConfig[cpu_id][index] = flags;
    }
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
    if (flags != currentConfig[cpu_id][index])
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags));
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_WBOX);
        currentConfig[cpu_id][index] = flags;
    }
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
    if (flags != currentConfig[cpu_id][index])
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg , flags));
        VERBOSEPRINTREG(cpu_id, reg, flags, SETUP_SBOX);
        currentConfig[cpu_id][index] = flags;
    }
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
            case MBOX1:
                nex_mbox_setup(cpu_id, index, event);
                break;

            case BBOX0:
            case BBOX1:
                nex_bbox_setup(cpu_id, index, event);
                break;

            case RBOX0:
            case RBOX1:
                nex_rbox_setup(cpu_id, index, event);
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
            eventSet->events[i].threadCounter[thread_id].startData = 0;
            eventSet->events[i].threadCounter[thread_id].counterData = 0;
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
            default:
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
