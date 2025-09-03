/*
 * =======================================================================================
 *
 *      Filename:  perfmon_zen5.h
 *
 *      Description:  Header file of perfmon module for AMD Family 1A (ZEN5)
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tg), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2017 RRZE, University Erlangen-Nuremberg
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
#ifndef PERFMON_ZEN5_H
#define PERFMON_ZEN5_H

#include <perfmon_zen5_events.h>
#include <perfmon_zen5_counters.h>
#include <error.h>
#include <affinity.h>
#include <cpuid.h>

static int perfmon_numCountersZen5 = NUM_COUNTERS_ZEN5;
static int perfmon_numArchEventsZen5 = NUM_ARCH_EVENTS_ZEN5;

int zen5_init_counter_map(int num_in_counters, RegisterMap * in_counters, int* num_out_counters, RegisterMap ** out_counters) {
    unsigned eax = 0x80000022, ebx = 0x0, ecx = 0x0, edx = 0x0;
    CPUID(eax, ebx, ecx, edx);
    int umc_count = (ebx >> 16) & 0xFF;
    int umc_units = bitMask_popcount(ecx);
    int umc_pmcs = umc_count/umc_units;
    int outcount = 0;
    DEBUG_PRINT(DEBUGLEV_DEVELOP, "Creating runtime counter map for AMD Zen5");
    RegisterMap * out = malloc((num_in_counters + umc_count) * sizeof(RegisterMap));
    if (!out) {
        return -ENOMEM;
    }
    memcpy(out, in_counters, num_in_counters * sizeof(RegisterMap));

    int umcoff = 0;
    for (int i = 0; i < umc_units; i++) {
        RegisterType unit_type = BBOX0+i;
        for (int j = 0; j < umc_pmcs; j++) {
            RegisterMap* out_umc = &out[num_in_counters+umcoff];
            out_umc->index = num_in_counters+umcoff;
            out_umc->type = unit_type;
            snprintf(out_umc->key, 127, "UMC%dC%d", i, j);
            out_umc->configRegister = MSR_AMD1A_UMC_PERFEVTSEL0 + umcoff;
            out_umc->counterRegister = MSR_AMD1A_UMC_PMC0 + umcoff;
            out_umc->counterRegister2 = 0x0;
            out_umc->device = MSR_DEV;
            out_umc->optionMask = ZEN5_VALID_OPTIONS_UMC;
            umcoff++;
        }
    }

    *num_out_counters = num_in_counters + umcoff;
    *out_counters = out;

    return 0;
}


int perfmon_init_zen5(int cpu_id)
{
    lock_acquire((int*) &socket_lock[affinity_thread2socket_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &core_lock[affinity_thread2core_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &sharedl3_lock[affinity_thread2sharedl3_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &numa_lock[affinity_thread2numa_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &die_lock[affinity_thread2die_lookup[cpu_id]], cpu_id);
    return 0;
}

int zen5_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent* event)
{
    uint64_t flags = 0x0ULL;

    // per default LIKWID counts in user-space
    flags |= (1ULL<<AMD_K17_PMC_USER_BIT);
    flags |= ((event->umask & AMD_K17_PMC_UNIT_MASK) << AMD_K17_PMC_UNIT_SHIFT);
    flags |= ((event->eventId & AMD_K17_PMC_EVSEL_MASK) << AMD_K17_PMC_EVSEL_SHIFT);
    flags |= (((event->eventId >> 8) & AMD_K17_PMC_EVSEL_MASK2) << AMD_K17_PMC_EVSEL_SHIFT2);

    // Currently no option for host/guest counting
    if (event->numberOfOptions > 0)
    {
        for(uint64_t j=0;j<event->numberOfOptions;j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_EDGE:
                    flags |= (1ULL<<AMD_K17_PMC_EDGE_BIT);
                    break;
                case EVENT_OPTION_COUNT_KERNEL:
                    flags |= (1ULL<<AMD_K17_PMC_KERNEL_BIT);
                    break;
                case EVENT_OPTION_INVERT:
                    flags |= (1ULL<<AMD_K17_PMC_INVERT_BIT);
                    break;
                case EVENT_OPTION_THRESHOLD:
                    flags |= (event->options[j].value & AMD_K17_PMC_THRES_MASK) << AMD_K17_PMC_THRES_SHIFT;
                    break;
                default:
                    break;
            }
        }
    }
    if (event->eventId == 0xFFF)
    {
        flags = 0x0ULL;
        flags |= ((event->eventId & AMD_K17_PMC_EVSEL_MASK) << AMD_K17_PMC_EVSEL_SHIFT);
        flags |= (((event->eventId >> 8) & AMD_K17_PMC_EVSEL_MASK2) << AMD_K17_PMC_EVSEL_SHIFT2);
        flags |= (1ULL<<22);
    }
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, "SETUP_PMC");
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int zen5_umc_setup(int cpu_id, RegisterIndex index, PerfmonEvent* event)
{
    uint64_t flags = 0x0ULL;
    int has_tid = 0;
    int has_cid = 0;
    int has_slice = 0;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags |= ((event->eventId & AMD_K1A_UMC_EVSEL_MASK) << AMD_K1A_UMC_EVSEL_SHIFT);

    if (event->numberOfOptions > 0)
    {
        for(int j = 0; j < (int)event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_MASK0:
                    flags |= ((uint64_t)(event->options[j].value & AMD_K1A_UMC_RWMASK_MASK)) << AMD_K1A_UMC_RWMASK_SHIFT;
                    break;
                default:
                    break;
            }
        }
    }

    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, "SETUP_UMC");
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}


int perfmon_setupCounterThread_zen5(int thread_id, PerfmonEventSet* eventSet)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t fixed_flags = 0x0ULL;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!TESTTYPE(eventSet, type))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        switch (type)
        {
            case PMC:
                zen5_pmc_setup(cpu_id, index, event);
                break;
            case CBOX0:
                zen4_cache_setup(cpu_id, index, event);
                break;
            case POWER:
                break;
            case FIXED:
                fixed_flags |= zen4_fixed_setup(cpu_id, index, event);
                break;
            case MBOX0:
                zen4_datafabric_setup(cpu_id, index, event);
                break;
            case BBOX0:
            case BBOX1:
            case BBOX2:
            case BBOX3:
            case BBOX4:
            case BBOX5:
            case BBOX6:
            case BBOX7:
            case BBOX8:
            case BBOX9:
            case BBOX10:
            case BBOX11:
            case BBOX12:
            case BBOX13:
            case BBOX14:
            case BBOX15:
            case BBOX16:
            case BBOX17:
            case BBOX18:
            case BBOX19:
            case BBOX20:
            case BBOX21:
            case BBOX22:
            case BBOX23:
            case BBOX24:
            case BBOX25:
            case BBOX26:
            case BBOX27:
            case BBOX28:
            case BBOX29:
            case BBOX30:
            case BBOX31:
            case BBOX32:
            case BBOX33:
            case BBOX34:
            case BBOX35:
            case BBOX36:
            case BBOX37:
            case BBOX38:
            case BBOX39:
            case BBOX40:
            case BBOX41:
            case BBOX42:
            case BBOX43:
            case BBOX44:
            case BBOX45:
            case BBOX46:
            case BBOX47:
            case BBOX48:
            case BBOX49:
            case BBOX50:
            case BBOX51:
            case BBOX52:
            case BBOX53:
            case BBOX54:
            case BBOX55:
            case BBOX56:
            case BBOX57:
            case BBOX58:
            case BBOX59:
            case BBOX60:
            case BBOX61:
            case BBOX62:
            case BBOX63:
                zen5_umc_setup(cpu_id, index, event);
                break;
            
            default:
                break;
        }
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
    }
    if ((fixed_flags > 0x0ULL))
    {
        uint64_t tmp = 0x0ULL;
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_AMD17_HW_CONFIG, &tmp));
        VERBOSEPRINTREG(cpu_id, MSR_AMD17_HW_CONFIG, LLU_CAST tmp, "READ_HW_CONFIG");
        tmp |= fixed_flags;
        VERBOSEPRINTREG(cpu_id, MSR_AMD17_HW_CONFIG, LLU_CAST tmp, "WRITE_HW_CONFIG");
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_AMD17_HW_CONFIG, tmp));
    }
    return 0;
}

int perfmon_startCountersThread_zen5(int thread_id, PerfmonEventSet* eventSet)
{
    int haveSLock = 0;
    int haveL3Lock = 0;
    int haveCLock = 0;
    uint64_t flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        haveSLock = 1;
    }
    if (sharedl3_lock[affinity_thread2sharedl3_lookup[cpu_id]] == cpu_id)
    {
        haveL3Lock = 1;
    }
    if (core_lock[affinity_thread2core_lookup[cpu_id]] == cpu_id)
    {
        haveCLock = 1;
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterType type = eventSet->events[i].type;
            if (!TESTTYPE(eventSet, type))
            {
                continue;
            }
            flags = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint32_t reg = counter_map[index].configRegister;
            uint32_t counter = counter_map[index].counterRegister;
            eventSet->events[i].threadCounter[thread_id].startData = 0;
            eventSet->events[i].threadCounter[thread_id].counterData = 0;
            if ((type == PMC) ||
                ((type == MBOX0) && (haveSLock)) ||
                ((type == CBOX0) && (haveL3Lock)))
            {
                VERBOSEPRINTREG(cpu_id, counter, LLU_CAST 0x0ULL, "RESET_CTR");
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter, 0x0ULL));
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, reg, &flags));
                VERBOSEPRINTREG(cpu_id, reg, LLU_CAST flags, "READ_CTRL");
                flags |= (1ULL << AMD_K17_ENABLE_BIT);  /* enable flag */
                VERBOSEPRINTREG(cpu_id, reg, LLU_CAST flags, "START_CTRL");
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
            }
            else if (type == POWER)
            {
                if (counter == MSR_AMD17_RAPL_PKG_STATUS && (!haveSLock))
                    continue;
                if (counter == MSR_AMD17_RAPL_CORE_STATUS && (!haveCLock))
                    continue;
                if (counter == MSR_AMD19_RAPL_L3_STATUS && (!haveL3Lock))
                    continue;
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &flags));
                eventSet->events[i].threadCounter[thread_id].startData = flags;
                VERBOSEPRINTREG(cpu_id, counter, LLU_CAST flags, "START_POWER");
            }
            else if (type == FIXED)
            {
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &flags));
                eventSet->events[i].threadCounter[thread_id].startData = field64(flags, 0, box_map[type].regWidth);
                VERBOSEPRINTREG(cpu_id, counter, LLU_CAST field64(flags, 0, box_map[type].regWidth), "START_FIXED");
            }
            else if (haveSLock && type >= BBOX0 && type <= BBOX63)
            {
                VERBOSEPRINTREG(cpu_id, counter, LLU_CAST 0x0ULL, "RESET_UMC_CTR");
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter, 0x0ULL));
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, reg, &flags));
                VERBOSEPRINTREG(cpu_id, reg, LLU_CAST flags, "READ_UMC_CTRL");
                flags |= (1ULL << AMD_K1A_UMC_ENABLE_BIT);  /* enable flag */
                VERBOSEPRINTREG(cpu_id, reg, LLU_CAST flags, "START_UMC_CTRL");
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
            }
            eventSet->events[i].threadCounter[thread_id].counterData = eventSet->events[i].threadCounter[thread_id].startData;
        }
    }
    return 0;
}

int perfmon_stopCountersThread_zen5(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags = 0x0ULL;
    int haveSLock = 0;
    int haveL3Lock = 0;
    int haveCLock = 0;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        haveSLock = 1;
    }
    if (sharedl3_lock[affinity_thread2sharedl3_lookup[cpu_id]] == cpu_id)
    {
        haveL3Lock = 1;
    }
    if (core_lock[affinity_thread2core_lookup[cpu_id]] == cpu_id)
    {
        haveCLock = 1;
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterType type = eventSet->events[i].type;
            if (!TESTTYPE(eventSet, type))
            {
                continue;
            }
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint32_t reg = counter_map[index].configRegister;
            uint32_t counter = counter_map[index].counterRegister;
            if ((type == PMC) ||
                ((type == MBOX0) && (haveSLock)) ||
                ((type == CBOX0) && (haveL3Lock)))
            {
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, reg, &flags));
                flags &= ~(1ULL<<AMD_K17_ENABLE_BIT);  /* clear enable flag */
                VERBOSEPRINTREG(cpu_id, reg, LLU_CAST flags, "STOP_CTRL");
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                VERBOSEPRINTREG(cpu_id, reg, LLU_CAST counter_result, "READ_CTR");
                if (field64(counter_result, 0, box_map[type].regWidth) < eventSet->events[i].threadCounter[thread_id].counterData)
                {
                    eventSet->events[i].threadCounter[thread_id].overflows++;
                    VERBOSEPRINTREG(cpu_id, reg, LLU_CAST counter_result, "OVERFLOW");
                }
            }
            else if (type == POWER)
            {
                if (counter == MSR_AMD17_RAPL_PKG_STATUS && (!haveSLock))
                    continue;
                if (counter == MSR_AMD17_RAPL_CORE_STATUS && (!haveCLock))
                    continue;
                if (counter == MSR_AMD19_RAPL_L3_STATUS && (!haveL3Lock))
                    continue;
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                {
                    eventSet->events[i].threadCounter[thread_id].overflows++;
                    VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, "OVERFLOW_POWER");
                }

                VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, "STOP_POWER");
            }
            else if (type == FIXED)
            {
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                counter_result = field64(counter_result, 0, box_map[type].regWidth);
                if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                {
                    eventSet->events[i].threadCounter[thread_id].overflows++;
                    VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, "OVERFLOW_FIXED");
                }
                VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, "STOP_FIXED");
            }
            else if (haveSLock && type >= BBOX0 && type <= BBOX63)
            {
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                VERBOSEPRINTREG(cpu_id, reg, LLU_CAST counter_result, "READ_UMC_CTR");
                if ((counter_result >> (box_map[type].regWidth-1)) > 0)
                {
                    eventSet->events[i].threadCounter[thread_id].overflows++;
                    VERBOSEPRINTREG(cpu_id, reg, LLU_CAST counter_result, "UMC_OVERFLOW");
                }
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, reg, &flags));
                flags &= ~(1ULL<<AMD_K1A_UMC_ENABLE_BIT);  /* clear enable flag */
                VERBOSEPRINTREG(cpu_id, reg, LLU_CAST flags, "STOP_UMC_CTRL");
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
            }
            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
        }
    }
    return 0;
}


int perfmon_readCountersThread_zen5(int thread_id, PerfmonEventSet* eventSet)
{
    int haveSLock = 0;
    int haveL3Lock = 0;
    int haveCLock = 0;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        haveSLock = 1;
    }
    if (sharedl3_lock[affinity_thread2sharedl3_lookup[cpu_id]] == cpu_id)
    {
        haveL3Lock = 1;
    }
    if (core_lock[affinity_thread2core_lookup[cpu_id]] == cpu_id)
    {
        haveCLock = 1;
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterType type = eventSet->events[i].type;
            if (!TESTTYPE(eventSet, type))
            {
                continue;
            }
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint32_t counter = counter_map[index].counterRegister;
            uint64_t* current = &(eventSet->events[i].threadCounter[thread_id].counterData);
            if ((type == PMC) ||
                ((type == MBOX0) && (haveSLock)) ||
                ((type == CBOX0) && (haveL3Lock)))
            {
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                VERBOSEPRINTREG(cpu_id, counter, counter_result, "READ_CTR");
                if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                {
                    eventSet->events[i].threadCounter[thread_id].overflows++;
                }
                *current = field64(counter_result, 0, box_map[type].regWidth);
            }
            else if (type == POWER)
            {
                if (counter == MSR_AMD17_RAPL_PKG_STATUS && (!haveSLock))
                    continue;
                if (counter == MSR_AMD17_RAPL_CORE_STATUS && (!haveCLock))
                    continue;
                if (counter == MSR_AMD19_RAPL_L3_STATUS && (!haveL3Lock))
                    continue;
                CHECK_POWER_READ_ERROR(power_read(cpu_id, counter, (uint64_t*)&counter_result));
                VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, "READ_POWER");
                if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                {
                    VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, "OVERFLOW_POWER");
                    eventSet->events[i].threadCounter[thread_id].overflows++;
                }
                *current = field64(counter_result, 0, box_map[type].regWidth);
            }
            else if (type == FIXED)
            {
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, "READ_FIXED");
                if (counter_result < eventSet->events[i].threadCounter[thread_id].counterData)
                {
                    VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, "OVERFLOW_FIXED");
                    eventSet->events[i].threadCounter[thread_id].overflows++;
                }
                *current = field64(counter_result, 0, box_map[type].regWidth);
            }
            else if (haveSLock && type >= BBOX0 && type <= BBOX63)
            {
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, "READ_UMC_CTR");
                if ((counter_result >> (box_map[type].regWidth-1)) > 0)
                {
                    eventSet->events[i].threadCounter[thread_id].overflows++;
                    VERBOSEPRINTREG(cpu_id, counter, LLU_CAST counter_result, "UMC_OVERFLOW");
                }
                *current = field64(counter_result, 0, box_map[type].regWidth);
            }
        }
    }
    return 0;
}


int perfmon_finalizeCountersThread_zen5(int thread_id, PerfmonEventSet* eventSet)
{
    int haveSLock = 0;
    int haveL3Lock = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        haveSLock = 1;
    }
    if (sharedl3_lock[affinity_thread2sharedl3_lookup[cpu_id]] == cpu_id)
    {
        haveL3Lock = 1;
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!TESTTYPE(eventSet, type))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        if ((type == PMC) ||
            ((type == MBOX0) && (haveSLock)) ||
            ((type == CBOX0) && (haveL3Lock)) ||
            ((type >= BBOX0 && type <= BBOX63) && (haveSLock)))
        {
            if (counter_map[index].configRegister != 0x0)
            {
                VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, 0x0ULL, "CLEAR_CTRL");
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, 0x0ULL));
            }
            if (counter_map[index].counterRegister != 0x0)
            {
                VERBOSEPRINTREG(cpu_id, counter_map[index].counterRegister, 0x0ULL, "CLEAR_CTR");
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].counterRegister, 0x0ULL));
            }
            eventSet->events[i].threadCounter[thread_id].init = FALSE;
        }
        else if (type == FIXED)
        {
            uint64_t tmp = 0x0ULL;
            CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_AMD17_HW_CONFIG, &tmp));
            if (tmp & (1ULL << AMD_K17_INST_RETIRE_ENABLE_BIT))
            {
                tmp &= ~(1ULL << AMD_K17_INST_RETIRE_ENABLE_BIT);
            }
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_AMD17_HW_CONFIG, tmp));
        }
    }
    return 0;
}



#endif
