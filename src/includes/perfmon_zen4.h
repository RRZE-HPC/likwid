/*
 * =======================================================================================
 *
 *      Filename:  perfmon_zen4.h
 *
 *      Description:  Header file of perfmon module for AMD Family 19 (ZEN4)
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
#ifndef PERFMON_ZEN4_H
#define PERFMON_ZEN4_H

#include <perfmon_zen4_events.h>
#include <perfmon_zen4_counters.h>
#include <error.h>
#include <affinity.h>
#include <cpuid.h>

static int perfmon_numCountersZen4 = NUM_COUNTERS_ZEN4;
static int perfmon_numArchEventsZen4 = NUM_ARCH_EVENTS_ZEN4;


int zen4_init_counter_map(int num_in_counters, RegisterMap * in_counters, int* num_out_counters, RegisterMap ** out_counters) {
    unsigned eax = 0x80000022, ebx = 0x0, ecx = 0x0, edx = 0x0;
    CPUID(eax, ebx, ecx, edx);
    int umc_count = (ebx >> 16) & 0xFF;
    int umc_units = bitMask_popcount(ecx);
    DEBUG_PRINT(DEBUGLEV_DEVELOP, "Creating runtime counter map for AMD Zen4 with %d UMC units", umc_units);
    if ((umc_count == 0) || (umc_units == 0)) {
        RegisterMap * out = malloc(num_in_counters * sizeof(RegisterMap));
        if (!out) {
            return -ENOMEM;
        }
        memcpy(out, in_counters, num_in_counters * sizeof(RegisterMap));
        *num_out_counters = num_in_counters;
        *out_counters = out;
    } else {
        int umc_pmcs = umc_count/umc_units;
        int outcount = 0;
        
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
                out_umc->configRegister = MSR_AMD19_UMC_PERFEVTSEL0 + umcoff;
                out_umc->counterRegister = MSR_AMD19_UMC_PMC0 + umcoff;
                out_umc->counterRegister2 = 0x0;
                out_umc->device = MSR_DEV;
                out_umc->optionMask = ZEN4_VALID_OPTIONS_UMC;
                umcoff++;
            }
        }
        *num_out_counters = num_in_counters + umcoff;
        *out_counters = out;
    }

    return 0;
}

int perfmon_init_zen4(int cpu_id)
{
    lock_acquire((int*) &socket_lock[affinity_thread2socket_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &core_lock[affinity_thread2core_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &sharedl3_lock[affinity_thread2sharedl3_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &numa_lock[affinity_thread2numa_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &die_lock[affinity_thread2die_lookup[cpu_id]], cpu_id);
    return 0;
}

int zen4_fixed_setup(int cpu_id, RegisterIndex index, PerfmonEvent* event)
{
    (void)cpu_id;
    (void)index;
    uint64_t flags = 0x0ULL;
    switch (event->eventId)
    {
        case 0x1:
            flags |= (1ULL << AMD_K17_INST_RETIRE_ENABLE_BIT);
            VERBOSEPRINTREG(cpu_id, 0x00, LLU_CAST flags, "SETUP_FIXC0");
            break;
        case 0x2:
        case 0x3:
            break;
        default:
            fprintf(stderr, "Unknown fixed event 0x%lX\n", event->eventId);
            break;
    }
    return flags;
}

int zen4_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent* event)
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
    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, "SETUP_PMC");
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int zen4_cache_setup(int cpu_id, RegisterIndex index, PerfmonEvent* event)
{
    uint64_t flags = 0x0ULL;
    int has_tid = 0;
    int has_cid = 0;
    int has_slice = 0;

    if (sharedl3_lock[affinity_thread2sharedl3_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags |= ((event->umask & AMD_K17_L3_UNIT_MASK) << AMD_K17_L3_UNIT_SHIFT);
    flags |= ((event->eventId & AMD_K17_L3_EVSEL_MASK) << AMD_K17_L3_EVSEL_SHIFT);
    if (event->numberOfOptions > 0)
    {
        for(uint64_t j=0;j<event->numberOfOptions;j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_TID:
                    flags |= ((uint64_t)(event->options[j].value & AMD_K17_L3_TID_MASK)) << AMD_K17_L3_TID_SHIFT;
                    has_tid = 1;
                    break;
                case EVENT_OPTION_CID:
                    flags |= ((uint64_t)(event->options[j].value & AMD_K17_L3_CID_MASK)) << AMD_K17_L3_CID_SHIFT;
                    has_cid = 1;
                    break;
                case EVENT_OPTION_SLICE:
                    flags |= ((uint64_t)(event->options[j].value & AMD_K17_L3_SLICE_MASK)) << AMD_K17_L3_SLICE_SHIFT;
                    has_slice = 1;
                    break;
                default:
                    break;
            }
        }
    }
    if (!has_tid)
        flags |= 0x3ULL << AMD_K17_L3_TID_SHIFT;
    if (!has_slice)
        flags |= 0x1ULL << AMD_K17_L3_ALL_SLICES_BIT;
    if (!has_cid)
        flags |= 0x1ULL << AMD_K17_L3_ALL_CORES_BIT;

    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, "SETUP_CBOX");
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}

int zen4_datafabric_setup(int cpu_id, RegisterIndex index, PerfmonEvent* event)
{
    uint64_t flags = 0x0ULL;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags |= ((event->eventId & AMD_K19_DF_EVSEL_MASK) << AMD_K19_DF_EVSEL_SHIFT);

    flags |= ((event->umask & AMD_K19_DF_UNIT_MASK) << AMD_K19_DF_UNIT_SHIFT);

    flags |= (((event->umask >> 8) & AMD_K19_DF_UNIT_MASK1) << AMD_K19_DF_UNIT_SHIFT1);

    flags |= (((event->eventId >> 8) & AMD_K19_DF_EVSEL_MASK1) << AMD_K19_DF_EVSEL_SHIFT1);

    flags |= (1ULL << AMD_K19_DF_ENABLE_OFFSET);


    if (flags != currentConfig[cpu_id][index])
    {
        VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, LLU_CAST flags, "SETUP_DF");
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
        currentConfig[cpu_id][index] = flags;
    }
    return 0;
}


int zen4_umc_setup(int cpu_id, RegisterIndex index, PerfmonEvent* event)
{
    uint64_t flags = 0x0ULL;
    int has_tid = 0;
    int has_cid = 0;
    int has_slice = 0;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] != cpu_id)
    {
        return 0;
    }

    flags |= ((event->eventId & AMD_K19_UMC_EVSEL_MASK) << AMD_K19_UMC_EVSEL_SHIFT);

    if (event->numberOfOptions > 0)
    {
        for(int j = 0; j < (int)event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_MASK0:
                    flags |= ((uint64_t)(event->options[j].value & AMD_K19_UMC_RWMASK_MASK)) << AMD_K19_UMC_RWMASK_SHIFT;
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

int perfmon_setupCounterThread_zen4(int thread_id, PerfmonEventSet* eventSet)
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
                zen4_pmc_setup(cpu_id, index, event);
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
                zen4_umc_setup(cpu_id, index, event);
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

int perfmon_startCountersThread_zen4(int thread_id, PerfmonEventSet* eventSet)
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
                PerfmonEvent *event = &(eventSet->events[i].event);
                if (counter == MSR_AMD17_RAPL_PKG_STATUS && (!haveSLock) && event->eventId == 0x02)
                    continue;
                else if (counter == MSR_AMD17_RAPL_CORE_STATUS && (!haveCLock) && event->eventId == 0x01)
                    continue;
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &flags));
                flags = field64(flags, 0, box_map[type].regWidth);
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
                flags |= (1ULL << AMD_K19_UMC_ENABLE_BIT);  /* enable flag */
                VERBOSEPRINTREG(cpu_id, reg, LLU_CAST flags, "START_UMC_CTRL");
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
            }
            eventSet->events[i].threadCounter[thread_id].counterData = eventSet->events[i].threadCounter[thread_id].startData;
        }
    }
    return 0;
}

int perfmon_stopCountersThread_zen4(int thread_id, PerfmonEventSet* eventSet)
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
                PerfmonEvent *event = &(eventSet->events[i].event);
                if (counter == MSR_AMD17_RAPL_PKG_STATUS && (!haveSLock) && event->eventId == 0x02)
                    continue;
                else if (counter == MSR_AMD17_RAPL_CORE_STATUS && (!haveCLock) && event->eventId == 0x01)
                    continue;
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                counter_result = field64(counter_result, 0, box_map[type].regWidth);
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
                flags &= ~(1ULL<<AMD_K19_UMC_ENABLE_BIT);  /* clear enable flag */
                VERBOSEPRINTREG(cpu_id, reg, LLU_CAST flags, "STOP_UMC_CTRL");
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, reg, flags));
            }
            eventSet->events[i].threadCounter[thread_id].counterData = counter_result;
        }
    }
    return 0;
}


int perfmon_readCountersThread_zen4(int thread_id, PerfmonEventSet* eventSet)
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
                PerfmonEvent *event = &(eventSet->events[i].event);
                if (counter == MSR_AMD17_RAPL_PKG_STATUS && (!haveSLock) && event->eventId == 0x02)
                    continue;
                else if (counter == MSR_AMD17_RAPL_CORE_STATUS && (!haveCLock) && event->eventId == 0x01)
                    continue;
                CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter, &counter_result));
                counter_result = field64(counter_result, 0, box_map[type].regWidth);
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


int perfmon_finalizeCountersThread_zen4(int thread_id, PerfmonEventSet* eventSet)
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

#endif //PERFMON_ZEN4_H
