/*
 * =======================================================================================
 *
 *      Filename:  perfmon.c
 *
 *      Description:  Main implementation of the performance monitoring module
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include <sys/types.h>

#include <types.h>
#include <likwid.h>
#include <bitUtil.h>
#include <timer.h>
#include <lock.h>
#include <perfmon.h>
#include <registers.h>
#include <topology.h>
#include <access.h>
#include <perfgroup.h>
#include <cpuid.h>

#include <perfmon_pm.h>
#include <perfmon_atom.h>
#include <perfmon_core2.h>
#include <perfmon_nehalem.h>
#include <perfmon_westmere.h>
#include <perfmon_westmereEX.h>
#include <perfmon_nehalemEX.h>
#include <perfmon_sandybridge.h>
#include <perfmon_ivybridge.h>
#include <perfmon_haswell.h>
#include <perfmon_phi.h>
#include <perfmon_knl.h>
#include <perfmon_k8.h>
#include <perfmon_k10.h>
#include <perfmon_interlagos.h>
#include <perfmon_kabini.h>
#include <perfmon_silvermont.h>
#include <perfmon_goldmont.h>
#include <perfmon_broadwell.h>
#include <perfmon_skylake.h>
#include <perfmon_zen.h>

#ifdef LIKWID_USE_PERFEVENT
#include <perfmon_perfevent.h>
#endif

/* #####   EXPORTED VARIABLES   ########################################### */

PerfmonEvent* eventHash = NULL;
RegisterMap* counter_map = NULL;
BoxMap* box_map = NULL;
PciDevice* pci_devices = NULL;
char** translate_types = NULL;

int perfmon_numCounters = 0;
int perfmon_numCoreCounters = 0;
int perfmon_numArchEvents = 0;
int perfmon_initialized = 0;
int perfmon_verbosity = DEBUGLEV_ONLY_ERROR;
int maps_checked = 0;
uint64_t currentConfig[MAX_NUM_THREADS][NUM_PMC] = { 0 };

PerfmonGroupSet* groupSet = NULL;
LikwidResults* markerResults = NULL;
int markerRegions = 0;

int (*perfmon_startCountersThread) (int thread_id, PerfmonEventSet* eventSet);
int (*perfmon_stopCountersThread) (int thread_id, PerfmonEventSet* eventSet);
int (*perfmon_readCountersThread) (int thread_id, PerfmonEventSet* eventSet);
int (*perfmon_setupCountersThread) (int thread_id, PerfmonEventSet* eventSet);
int (*perfmon_finalizeCountersThread) (int thread_id, PerfmonEventSet* eventSet);

int (*initThreadArch) (int cpu_id);
void perfmon_delEventSet(int groupID);

char* eventOptionTypeName[NUM_EVENT_OPTIONS] = {
    "NONE",
    "OPCODE",
    "MATCH0",
    "MATCH1",
    "MATCH2",
    "MATCH3",
    "MASK0",
    "MASK1",
    "MASK2",
    "MASK3",
    "NID",
    "TID",
    "STATE",
    "EDGEDETECT",
    "THRESHOLD",
    "INVERT",
    "KERNEL",
    "ANYTHREAD",
    "OCCUPANCY",
    "OCCUPANCY_FILTER",
    "OCCUPANCY_EDGEDETECT",
    "OCCUPANCY_INVERT",
    "IN_TRANSACTION",
    "IN_TRANSACTION_ABORTED"
#ifdef LIKWID_USE_PERFEVENT
    ,"PERF_PID"
    ,"PERF_FLAGS"
#endif
};

char* default_translate_types[NUM_UNITS] = {
    [FIXED] = "/sys/bus/event_source/devices/cpu",
    [PMC] = "/sys/bus/event_source/devices/cpu",
    [MBOX0] = "/sys/bus/event_source/devices/uncore_imc",
    [CBOX0] = "/sys/bus/event_source/devices/uncore_cbox_0",
    [CBOX1] = "/sys/bus/event_source/devices/uncore_cbox_1",
    [CBOX2] = "/sys/bus/event_source/devices/uncore_cbox_2",
    [CBOX3] = "/sys/bus/event_source/devices/uncore_cbox_3",
    [UBOX] = "/sys/bus/event_source/devices/uncore_arb",
    [POWER] = "/sys/bus/event_source/devices/power",
};

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int
getIndexAndType (bstring reg, RegisterIndex* index, RegisterType* type)
{
    int ret = FALSE;

    for (int i=0; i< perfmon_numCounters; i++)
    {
        if (biseqcstr(reg, counter_map[i].key))
        {
            *index = counter_map[i].index;
            *type = counter_map[i].type;
            ret = TRUE;
            break;
        }
    }
    return ret;
}

static RegisterType
checkAccess(bstring reg, RegisterIndex index, RegisterType oldtype, int force)
{
    int err = 0;
    uint64_t tmp = 0x0ULL;
    RegisterType type = oldtype;
    int (*ownstrcmp)(const char*, const char*);
    ownstrcmp = &strcmp;
    int testcpu = groupSet->threads[0].processorId;
    int firstpmcindex = -1;

    for (int i=0; i< perfmon_numCounters; i++)
    {
        if (counter_map[i].type == PMC && firstpmcindex < 0)
        {
            firstpmcindex = i;
            break;
        }
    }

    if (cpuid_info.isIntel && type == PMC && (index - firstpmcindex) >= cpuid_info.perf_num_ctr)
    {
        fprintf(stderr,
                "WARN: Counter %s is only available with deactivated HyperThreading. Counter results defaults to 0.\n",
                bdata(reg));
        return NOTYPE;
    }
    if (type == NOTYPE)
    {
        DEBUG_PRINT(DEBUGLEV_INFO, WARNING: Counter %s not available on the current system. Counter results defaults to 0.,bdata(reg));
        return NOTYPE;
    }
    if (ownstrcmp(bdata(reg), counter_map[index].key) != 0)
    {
        DEBUG_PRINT(DEBUGLEV_INFO, WARNING: Counter %s does not exist ,bdata(reg));
        return NOTYPE;
    }
    err = HPMcheck(counter_map[index].device, 0);
    if (!err)
    {
        DEBUG_PRINT(DEBUGLEV_INFO, WARNING: The device for counter %s does not exist ,bdata(reg));
        return NOTYPE;
    }
    if ((type != THERMAL) && (type != POWER) && (type != WBOX0FIX))
    {
        int check_settings = 1;
        uint32_t reg = counter_map[index].configRegister;
        if (reg == 0x0)
        {
            reg = counter_map[index].counterRegister;
            check_settings = 0;
        }
        err = HPMread(testcpu, counter_map[index].device, reg, &tmp);
        if (err != 0)
        {
            if (err == -ENODEV)
            {
                DEBUG_PRINT(DEBUGLEV_DETAIL, Device %s not accessible on this machine,
                                         pci_devices[box_map[type].device].name);
            }
            else
            {
                DEBUG_PRINT(DEBUGLEV_DETAIL, Counter %s not readable on this machine,
                                             counter_map[index].key);
            }
            type = NOTYPE;
        }
        else if (tmp == 0x0ULL)
        {
            err = HPMwrite(testcpu, counter_map[index].device, reg, 0x0ULL);
            if (err != 0)
            {
                if (err == -ENODEV)
                {
                    DEBUG_PRINT(DEBUGLEV_DETAIL, Device %s not accessible on this machine,
                                             pci_devices[box_map[type].device].name);
                }
                else
                {
                    DEBUG_PRINT(DEBUGLEV_DETAIL, Counter %s not writeable on this machine,
                                             counter_map[index].key);
                }
                type = NOTYPE;
            }
            check_settings = 0;
        }
        if ((check_settings) && (tmp != 0x0ULL))
        {
            if (force == 1)
            {
                DEBUG_PRINT(DEBUGLEV_DETAIL, Counter %s has bits set (0x%llx) but we are forced to overwrite them,
                                             counter_map[index].key, LLU_CAST tmp);
                err = HPMwrite(testcpu, counter_map[index].device, reg, 0x0ULL);
            }
            else if ((force == 0) && ((type != FIXED)&&(type != THERMAL)&&(type != POWER)&&(type != WBOX0FIX)))
            {
                fprintf(stderr, "ERROR: The selected register %s is in use.\n", counter_map[index].key);
                fprintf(stderr, "Please run likwid with force option (-f, --force) to overwrite settings\n");
                exit(EXIT_SUCCESS);
            }
        }
    }
    else if ((type == POWER) || (type == WBOX0FIX) || (type == THERMAL))
    {
        err = HPMread(testcpu, MSR_DEV, counter_map[index].counterRegister, &tmp);
        if (err != 0)
        {
            DEBUG_PRINT(DEBUGLEV_DETAIL, Counter %s not readable on this machine,
                                         counter_map[index].key);
            type = NOTYPE;
        }
    }
    else
    {
        type = NOTYPE;
    }
    return type;
}

static int
checkCounter(bstring counterName, const char* limit)
{
    int i;
    struct bstrList* tokens;
    int ret = FALSE;
    bstring limitString = bfromcstr(limit);

    tokens = bsplit(limitString,'|');
    for(i=0; i<tokens->qty; i++)
    {
        if(bstrncmp(counterName, tokens->entry[i], blength(tokens->entry[i])))
        {
            ret = FALSE;
        }
        else
        {
            ret = TRUE;
            break;
        }
    }
    bdestroy(limitString);
    bstrListDestroy(tokens);
    return ret;
}

static int
getEvent(bstring event_str, bstring counter_str, PerfmonEvent* event)
{
    int ret = FALSE;
    int (*ownstrncmp)(const char *, const char *, size_t);
    ownstrncmp = &strncmp;
    for (int i=0; i< perfmon_numArchEvents; i++)
    {
        if (biseqcstr(event_str, eventHash[i].name))
        {
            *event = eventHash[i];
            ret = TRUE;
            break;
        }
    }

    return ret;
}

static int
assignOption(PerfmonEvent* event, bstring entry, int index, EventOptionType type, int zero_value)
{
    int found_double = -1;
    int return_index = index;
    long long unsigned int value;
    for (int k = 0; k < index; k++)
    {
        if (event->options[k].type == type)
        {
            found_double = k;
            break;
        }
    }
    if (found_double >= 0)
    {
        index = found_double;
    }
    else
    {
        return_index++;
    }
    event->options[index].type = type;
    if (zero_value)
    {
        event->options[index].value = 0;
    }
    else
    {
        value = 0;
        sscanf(bdata(entry), "%llx", &value);
        event->options[index].value = value;
    }
    return return_index;
}

static int
parseOptions(struct bstrList* tokens, PerfmonEvent* event, RegisterIndex index)
{
    int i,j;
    struct bstrList* subtokens;

    for (i = event->numberOfOptions; i < MAX_EVENT_OPTIONS; i++)
    {
        event->options[i].type = EVENT_OPTION_NONE;
    }
    if (tokens->qty-2 > MAX_EVENT_OPTIONS)
    {
        return -ERANGE;
    }


    for (i=2;i<tokens->qty;i++)
    {
        subtokens = bsplit(tokens->entry[i],'=');
        btolower(subtokens->entry[0]);
        if (subtokens->qty == 1)
        {
            if (biseqcstr(subtokens->entry[0], "edgedetect") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_EDGE, 1);
            }
            else if (biseqcstr(subtokens->entry[0], "invert") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_INVERT, 1);
            }
            else if (biseqcstr(subtokens->entry[0], "kernel") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_COUNT_KERNEL, 1);
            }
            else if (biseqcstr(subtokens->entry[0], "anythread") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_ANYTHREAD, 1);
            }
            else if (biseqcstr(subtokens->entry[0], "occ_edgedetect") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_OCCUPANCY_EDGE, 1);
            }
            else if (biseqcstr(subtokens->entry[0], "occ_invert") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_OCCUPANCY_INVERT, 1);
            }
            else if (biseqcstr(subtokens->entry[0], "in_trans") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_IN_TRANS, 1);
            }
            else if (biseqcstr(subtokens->entry[0], "in_trans_aborted") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_IN_TRANS_ABORT, 1);
            }
            else
            {
                fprintf(stderr, "WARN: Option '%s' unknown, skipping option\n", bdata(subtokens->entry[0]));
                continue;
            }
            event->options[event->numberOfOptions].value = 0;
        }
        else if (subtokens->qty == 2)
        {
            if (biseqcstr(subtokens->entry[0], "opcode") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_OPCODE, 0);
            }
            else if (biseqcstr(subtokens->entry[0], "match0") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_MATCH0, 0);
            }
            else if (biseqcstr(subtokens->entry[0], "match1") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_MATCH1, 0);
            }
            else if (biseqcstr(subtokens->entry[0], "match2") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_MATCH2, 0);
            }
            else if (biseqcstr(subtokens->entry[0], "match3") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_MATCH3, 0);
            }
            else if (biseqcstr(subtokens->entry[0], "mask0") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_MASK0, 0);
            }
            else if (biseqcstr(subtokens->entry[0], "mask1") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_MASK1, 0);
            }
            else if (biseqcstr(subtokens->entry[0], "mask2") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_MASK2, 0);
            }
            else if (biseqcstr(subtokens->entry[0], "mask3") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_MASK3, 0);
            }
            else if (biseqcstr(subtokens->entry[0], "nid") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_NID, 0);
            }
            else if (biseqcstr(subtokens->entry[0], "tid") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_TID, 0);
            }
            else if (biseqcstr(subtokens->entry[0], "state") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_STATE, 0);
            }
            else if (biseqcstr(subtokens->entry[0], "threshold") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_THRESHOLD, 0);
            }
            else if (biseqcstr(subtokens->entry[0], "occupancy") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_OCCUPANCY, 0);
            }
            else if (biseqcstr(subtokens->entry[0], "occ_filter") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_OCCUPANCY_FILTER, 0);
            }
#ifdef LIKWID_USE_PERFEVENT
            else if (biseqcstr(subtokens->entry[0], "perf_pid") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_PERF_PID, 0);
            }
            else if (biseqcstr(subtokens->entry[0], "perf_flags") == 1)
            {
                event->numberOfOptions = assignOption(event, subtokens->entry[1],
                                    event->numberOfOptions, EVENT_OPTION_PERF_FLAGS, 0);
            }
#endif
            else
            {
                fprintf(stderr, "WARN: Option '%s' unknown, skipping option\n", bdata(subtokens->entry[0]));
                continue;
            }
        }
        bstrListDestroy(subtokens);
    }
    for(i=event->numberOfOptions-1;i>=0;i--)
    {
#ifdef LIKWID_USE_PERFEVENT
        if (event->options[i].type != EVENT_OPTION_PERF_PID && event->options[i].type != EVENT_OPTION_PERF_FLAGS && !(OPTIONS_TYPE_MASK(event->options[i].type) & (counter_map[index].optionMask|event->optionMask)))
#else
        if (!(OPTIONS_TYPE_MASK(event->options[i].type) & (counter_map[index].optionMask|event->optionMask)))
#endif
        {
            DEBUG_PRINT(DEBUGLEV_INFO,Removing Option %s not valid for register %s,
                        eventOptionTypeName[event->options[i].type],
                        counter_map[index].key);
            event->options[i].type = EVENT_OPTION_NONE;
            event->numberOfOptions--;
        }
    }

    for(i=0;i<event->numberOfOptions;i++)
    {
        if (event->options[i].type == EVENT_OPTION_EDGE)
        {
            int threshold_set = FALSE;
            for (j=0;j<event->numberOfOptions;j++)
            {
                if (event->options[i].type == EVENT_OPTION_THRESHOLD)
                {
                    threshold_set = TRUE;
                    break;
                }
            }
            if ((threshold_set == FALSE) && (event->numberOfOptions < MAX_EVENT_OPTIONS))
            {
                event->options[event->numberOfOptions].type = EVENT_OPTION_THRESHOLD;
                event->options[event->numberOfOptions].value = 0x1;
                event->numberOfOptions++;
            }
            else
            {
                ERROR_PLAIN_PRINT(Cannot set threshold option to default. no more space in options list);
            }
        }
        else if (event->options[i].type == EVENT_OPTION_OCCUPANCY)
        {
            int threshold_set = FALSE;
            int edge_set = FALSE;
            int invert_set = FALSE;
            for (j=0;j<event->numberOfOptions;j++)
            {
                if (event->options[i].type == EVENT_OPTION_THRESHOLD)
                {
                    threshold_set = TRUE;
                    break;
                }
                if (event->options[i].type == EVENT_OPTION_EDGE)
                {
                    edge_set = TRUE;
                    break;
                }
                if (event->options[i].type == EVENT_OPTION_INVERT)
                {
                    invert_set = TRUE;
                    break;
                }
            }
            if ((threshold_set == FALSE) && (event->numberOfOptions < MAX_EVENT_OPTIONS) &&
                (edge_set == TRUE || invert_set == TRUE ))
            {
                event->options[event->numberOfOptions].type = EVENT_OPTION_THRESHOLD;
                event->options[event->numberOfOptions].value = 0x1;
                event->numberOfOptions++;
            }
            else
            {
                ERROR_PLAIN_PRINT(Cannot set threshold option to default. no more space in options list);
            }
        }
    }

    return event->numberOfOptions;
}

static double
calculateResult(int groupId, int eventId, int threadId)
{
    PerfmonEventSetEntry* event;
    PerfmonCounter* counter;
    int cpu_id;
    double result = 0.0;
    uint64_t maxValue = 0ULL;
    if (groupSet->groups[groupId].events[eventId].type == NOTYPE)
        return result;

    event = &(groupSet->groups[groupId].events[eventId]);
    counter = &(event->threadCounter[threadId]);
    if (counter->overflows == 0)
    {
        result = (double) (counter->counterData - counter->startData);
    }
    else if (counter->overflows > 0)
    {
        maxValue = perfmon_getMaxCounterValue(counter_map[event->index].type);
        result += (double) ((maxValue - counter->startData) + counter->counterData);
        if (counter->overflows > 1)
        {
            result += (double) ((counter->overflows-1) * maxValue);
        }
        counter->overflows = 0;
    }
    if (counter_map[event->index].type == POWER)
    {
        result *= power_getEnergyUnit(getCounterTypeOffset(event->index));
    }
    else if (counter_map[event->index].type == THERMAL)
    {
        result = (double)counter->counterData;
    }
    return result;
}

int
getCounterTypeOffset(int index)
{
    int off = 0;
    for (int j=index-1;j>=0;j--)
    {
        if (counter_map[index].type == counter_map[j].type)
        {
            off++;
        }
        else
        {
            break;
        }
    }
    return off;
}

void
perfmon_setVerbosity(int level)
{
    if ((level >= DEBUGLEV_ONLY_ERROR) && (level <= DEBUGLEV_DEVELOP))
        perfmon_verbosity = level;
}

void
perfmon_check_counter_map(int cpu_id)
{
    int own_hpm = 0;
    if (perfmon_numCounters == 0 || perfmon_numArchEvents == 0)
    {
        ERROR_PLAIN_PRINT(Counter and event maps not initialized.);
        return;
    }
    if (maps_checked)
        return;
#ifndef LIKWID_USE_PERFEVENT
    if (!HPMinitialized())
    {
        HPMinit();
        if (HPMaddThread(cpu_id) != 0)
        {
            ERROR_PLAIN_PRINT(Cannot check counters without access to performance counters)
            return;
        }
        own_hpm = 1;
    }
#endif
    int startpmcindex = -1;
    for (int i=0;i<perfmon_numCounters;i++)
    {
        if (counter_map[i].type == NOTYPE)
        {
            continue;
        }
        if (counter_map[i].type == PMC && startpmcindex < 0)
        {
            startpmcindex = i;
        }
        if (cpuid_info.isIntel &&
            counter_map[i].type == PMC &&
            (counter_map[i].index - counter_map[startpmcindex].index) >= cpuid_info.perf_num_ctr)
        {
            counter_map[i].type = NOTYPE;
            counter_map[i].optionMask = 0x0ULL;
        }
#ifndef LIKWID_USE_PERFEVENT
        if (HPMcheck(counter_map[i].device, cpu_id))
        {
            uint32_t reg = counter_map[i].configRegister;
            uint64_t tmp = 0x0ULL;
            if (reg == 0x0U)
                reg = counter_map[i].counterRegister;
            int err = HPMread(cpu_id, counter_map[i].device, reg, &tmp);
            if (err)
            {
                counter_map[i].type = NOTYPE;
                counter_map[i].optionMask = 0x0ULL;
            }
        }
        else
        {
            counter_map[i].type = NOTYPE;
            counter_map[i].optionMask = 0x0ULL;
        }
#endif
    }
    if (own_hpm)
        HPMfinalize();
    for (int i=0; i<perfmon_numArchEvents; i++)
    {
        int found = 0;
        bstring estr = bfromcstr(eventHash[i].name);
        if (i > 0 && strlen(eventHash[i-1].limit) != 0 && strcmp(eventHash[i-1].limit, eventHash[i].limit) == 0)
        {
            bdestroy(estr);
            continue;
        }
        for (int j=0;j<perfmon_numCounters; j++)
        {
            if (counter_map[j].type == NOTYPE)
                continue;
            PerfmonEvent event;
            bstring cstr = bfromcstr(counter_map[j].key);
            if (getEvent(estr, cstr, &event))
            {
                found = 1;
                bdestroy(cstr);
                break;
            }
            bdestroy(cstr);
        }
        bdestroy(estr);
        if (!found)
        {
            eventHash[i].limit = "";
        }
    }
    maps_checked = 1;
}

void
perfmon_init_maps(void)
{
    uint32_t eax, ebx, ecx, edx;
    if (eventHash != NULL && counter_map != NULL && box_map != NULL && perfmon_numCounters > 0 && perfmon_numArchEvents > 0)
        return;
    switch ( cpuid_info.family )
    {
        case P6_FAMILY:

            switch ( cpuid_info.model )
            {
                case PENTIUM_M_BANIAS:
                case PENTIUM_M_DOTHAN:
                    eventHash = pm_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEvents_pm;
                    counter_map = pm_counter_map;
                    box_map = pm_box_map;
                    perfmon_numCounters = perfmon_numCounters_pm;
                    translate_types = default_translate_types;
                    break;

                case ATOM_45:
                case ATOM_32:
                case ATOM_22:
                case ATOM:
                    eventHash = atom_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsAtom;
                    counter_map = core2_counter_map;
                    perfmon_numCounters = perfmon_numCountersCore2;
                    box_map = core2_box_map;
                    translate_types = default_translate_types;
                    break;

                case ATOM_SILVERMONT_E:
                case ATOM_SILVERMONT_C:
                case ATOM_SILVERMONT_Z1:
                case ATOM_SILVERMONT_Z2:
                case ATOM_SILVERMONT_F:
                case ATOM_SILVERMONT_AIR:
                    eventHash = silvermont_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsSilvermont;
                    counter_map = silvermont_counter_map;
                    box_map = silvermont_box_map;
                    perfmon_numCounters = perfmon_numCountersSilvermont;
                    perfmon_numCoreCounters = perfmon_numCoreCountersSilvermont;
                    translate_types = default_translate_types;
                    break;

                case ATOM_SILVERMONT_GOLD:
                case ATOM_DENVERTON:
                case ATOM_GOLDMONT_PLUS:
                    eventHash = goldmont_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsGoldmont;
                    counter_map = goldmont_counter_map;
                    box_map = goldmont_box_map;
                    perfmon_numCounters = perfmon_numCountersGoldmont;
                    perfmon_numCoreCounters = perfmon_numCoreCountersGoldmont;
                    translate_types = default_translate_types;
                    break;

                case CORE_DUO:
                    ERROR_PLAIN_PRINT(Unsupported Processor);
                    break;

                case XEON_MP:
                case CORE2_65:
                case CORE2_45:
                    eventHash = core2_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsCore2;
                    counter_map = core2_counter_map;
                    perfmon_numCounters = perfmon_numCountersCore2;
                    box_map = core2_box_map;
                    translate_types = default_translate_types;
                    break;

                case NEHALEM_EX:
                    eventHash = nehalemEX_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsNehalemEX;
                    counter_map = nehalemEX_counter_map;
                    perfmon_numCounters = perfmon_numCountersNehalemEX;
                    box_map = nehalemEX_box_map;
                    translate_types = default_translate_types;
                    break;

                case WESTMERE_EX:
                    eventHash = westmereEX_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsWestmereEX;
                    counter_map = westmereEX_counter_map;
                    perfmon_numCounters = perfmon_numCountersWestmereEX;
                    box_map = westmereEX_box_map;
                    translate_types = default_translate_types;
                    break;

                case NEHALEM_BLOOMFIELD:
                case NEHALEM_LYNNFIELD:
                case NEHALEM_LYNNFIELD_M:
                    eventHash = nehalem_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsNehalem;
                    counter_map = nehalem_counter_map;
                    perfmon_numCounters = perfmon_numCountersNehalem;
                    box_map = nehalem_box_map;
                    translate_types = default_translate_types;
                    break;

                case NEHALEM_WESTMERE_M:
                case NEHALEM_WESTMERE:
                    eventHash = westmere_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsWestmere;
                    counter_map = nehalem_counter_map;
                    perfmon_numCounters = perfmon_numCountersNehalem;
                    box_map = nehalem_box_map;
                    translate_types = default_translate_types;
                    break;

                case IVYBRIDGE_EP:
                    pci_devices = ivybridgeEP_pci_devices;
                    translate_types = ivybridgeEP_translate_types;
                    box_map = ivybridgeEP_box_map;
                    eventHash = ivybridgeEP_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsIvybridgeEP;
                    counter_map = ivybridgeEP_counter_map;
                    perfmon_numCounters = perfmon_numCountersIvybridgeEP;
                    perfmon_numCoreCounters = perfmon_numCoreCountersIvybridgeEP;
                    translate_types = ivybridgeEP_translate_types;
                    break;
                case IVYBRIDGE:
                    translate_types = default_translate_types;
                    eventHash = ivybridge_arch_events;
                    box_map = ivybridge_box_map;
                    perfmon_numArchEvents = perfmon_numArchEventsIvybridge;
                    counter_map = ivybridge_counter_map;
                    perfmon_numCounters = perfmon_numCountersIvybridge;
                    perfmon_numCoreCounters = perfmon_numCoreCountersIvybridge;
                    translate_types = default_translate_types;
                    break;

                case HASWELL_EP:
                    eventHash = haswellEP_arch_events;
                    translate_types = haswellEP_translate_types;
                    perfmon_numArchEvents = perfmon_numArchEventsHaswellEP;
                    counter_map = haswellEP_counter_map;
                    perfmon_numCounters = perfmon_numCountersHaswellEP;
                    perfmon_numCoreCounters = perfmon_numCoreCountersHaswellEP;
                    box_map = haswellEP_box_map;
                    pci_devices = haswellEP_pci_devices;
                    translate_types = haswellEP_translate_types;
                    break;
                case HASWELL:
                case HASWELL_M1:
                case HASWELL_M2:
                    eventHash = haswell_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsHaswell;
                    counter_map = haswell_counter_map;
                    perfmon_numCounters = perfmon_numCountersHaswell;
                    perfmon_numCoreCounters = perfmon_numCoreCountersHaswell;
                    box_map = haswell_box_map;
                    translate_types = default_translate_types;
                    break;

                case SANDYBRIDGE_EP:
                    pci_devices = sandybridgeEP_pci_devices;
                    translate_types = sandybridgeEP_translate_types;
                    box_map = sandybridgeEP_box_map;
                    eventHash = sandybridgeEP_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsSandybridgeEP;
                    counter_map = sandybridgeEP_counter_map;
                    perfmon_numCounters = perfmon_numCountersSandybridgeEP;
                    perfmon_numCoreCounters = perfmon_numCoreCountersSandybridgeEP;
                    translate_types = sandybridgeEP_translate_types;
                    break;
                case SANDYBRIDGE:
                    box_map = sandybridge_box_map;
                    eventHash = sandybridge_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsSandybridge;
                    counter_map = sandybridge_counter_map;
                    perfmon_numCounters = perfmon_numCountersSandybridge;
                    perfmon_numCoreCounters = perfmon_numCoreCountersSandybridge;
                    translate_types = default_translate_types;
                    break;

                case BROADWELL:
                case BROADWELL_E3:
                    box_map = broadwell_box_map;
                    eventHash = broadwell_arch_events;
                    counter_map = broadwell_counter_map;
                    perfmon_numArchEvents = perfmon_numArchEventsBroadwell;
                    perfmon_numCounters = perfmon_numCountersBroadwell;
                    perfmon_numCoreCounters = perfmon_numCoreCountersBroadwell;
                    translate_types = default_translate_types;
                    break;
                case BROADWELL_D:
                    pci_devices = broadwelld_pci_devices;
                    translate_types = broadwellEP_translate_types;
                    box_map = broadwelld_box_map;
                    eventHash = broadwelld_arch_events;
                    counter_map = broadwelld_counter_map;
                    perfmon_numArchEvents = perfmon_numArchEventsBroadwellD;
                    perfmon_numCounters = perfmon_numCountersBroadwellD;
                    perfmon_numCoreCounters = perfmon_numCoreCountersBroadwellD;
                    translate_types = broadwellEP_translate_types;
                    break;
                case BROADWELL_E:
                    pci_devices = broadwellEP_pci_devices;
                    box_map = broadwellEP_box_map;
                    eventHash = broadwellEP_arch_events;
                    translate_types = broadwellEP_translate_types;
                    counter_map = broadwellEP_counter_map;
                    perfmon_numArchEvents = perfmon_numArchEventsBroadwellEP;
                    perfmon_numCounters = perfmon_numCountersBroadwellEP;
                    perfmon_numCoreCounters = perfmon_numCoreCountersBroadwellEP;
                    translate_types = broadwellEP_translate_types;
                    break;

                case SKYLAKE1:
                case SKYLAKE2:
                case KABYLAKE1:
                case KABYLAKE2:
                    box_map = skylake_box_map;
                    eventHash = skylake_arch_events;
                    counter_map = skylake_counter_map;
                    perfmon_numArchEvents = perfmon_numArchEventsSkylake;
                    perfmon_numCounters = perfmon_numCountersSkylake;
                    perfmon_numCoreCounters = perfmon_numCoreCountersSkylake;
                    translate_types = default_translate_types;
                    break;
                case SKYLAKEX:
                    box_map = skylakeX_box_map;
                    eventHash = skylakeX_arch_events;
                    counter_map = skylakeX_counter_map;
                    perfmon_numArchEvents = perfmon_numArchEventsSkylakeX;
                    perfmon_numCounters = perfmon_numCountersSkylakeX;
                    perfmon_numCoreCounters = perfmon_numCoreCountersSkylakeX;
                    translate_types = skylakeX_translate_types;
                    break;

                case XEON_PHI_KNL:
                    pci_devices = knl_pci_devices;
                    eventHash = knl_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsKNL;
                    counter_map = knl_counter_map;
                    box_map = knl_box_map;
                    perfmon_numCounters = perfmon_numCountersKNL;
                    translate_types = knl_translate_types;
                    break;

                default:
                    ERROR_PLAIN_PRINT(Unsupported Processor);
                    break;
            }
            break;

        case MIC_FAMILY:

            switch ( cpuid_info.model )
            {
                case XEON_PHI:
                    eventHash = phi_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsPhi;
                    counter_map = phi_counter_map;
                    box_map = phi_box_map;
                    perfmon_numCounters = perfmon_numCountersPhi;
                    translate_types = default_translate_types;
                    break;

                default:
                    ERROR_PLAIN_PRINT(Unsupported Processor);
                    break;
            }
            break;

        case K8_FAMILY:
            eventHash = k8_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsK8;
            counter_map = k10_counter_map;
            box_map = k10_box_map;
            perfmon_numCounters = perfmon_numCountersK10;
            translate_types = default_translate_types;
            break;

        case K10_FAMILY:
            eventHash = k10_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsK10;
            counter_map = k10_counter_map;
            box_map = k10_box_map;
            perfmon_numCounters = perfmon_numCountersK10;
            translate_types = default_translate_types;
            break;

        case K15_FAMILY:
            eventHash = interlagos_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsInterlagos;
            counter_map = interlagos_counter_map;
            box_map = interlagos_box_map;
            perfmon_numCounters = perfmon_numCountersInterlagos;
            translate_types = default_translate_types;
            break;

        case K16_FAMILY:
            eventHash = kabini_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsKabini;
            counter_map = kabini_counter_map;
            box_map = kabini_box_map;
            perfmon_numCounters = perfmon_numCountersKabini;
            translate_types = default_translate_types;
            break;

        case ZEN_FAMILY:
            eventHash = zen_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsZen;
            counter_map = zen_counter_map;
            box_map = zen_box_map;
            perfmon_numCounters = perfmon_numCountersZen;
            translate_types = zen_translate_types;
            break;

        default:
            ERROR_PLAIN_PRINT(Unsupported Processor);
            break;
    }
    return;
}

void
perfmon_init_funcs(int* init_power, int* init_temp)
{
    int initialize_power = FALSE;
    int initialize_thermal = FALSE;
#ifndef LIKWID_USE_PERFEVENT
    switch ( cpuid_info.family )
    {
        case P6_FAMILY:

            switch ( cpuid_info.model )
            {
                case PENTIUM_M_BANIAS:
                case PENTIUM_M_DOTHAN:
                    initThreadArch = perfmon_init_pm;
                    perfmon_startCountersThread = perfmon_startCountersThread_pm;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_pm;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_pm;
                    perfmon_readCountersThread = perfmon_readCountersThread_pm;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_pm;
                    break;

                case ATOM_45:
                case ATOM_32:
                case ATOM_22:
                case ATOM:
                    initThreadArch = perfmon_init_core2;
                    perfmon_startCountersThread = perfmon_startCountersThread_core2;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_core2;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_core2;
                    perfmon_readCountersThread = perfmon_readCountersThread_core2;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_core2;
                    break;

                case ATOM_SILVERMONT_E:
                case ATOM_SILVERMONT_C:
                case ATOM_SILVERMONT_Z1:
                case ATOM_SILVERMONT_Z2:
                case ATOM_SILVERMONT_F:
                case ATOM_SILVERMONT_AIR:
                    initialize_power = TRUE;
                    initialize_thermal = TRUE;
                    initThreadArch = perfmon_init_silvermont;
                    perfmon_startCountersThread = perfmon_startCountersThread_silvermont;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_silvermont;
                    perfmon_setupCountersThread = perfmon_setupCountersThread_silvermont;
                    perfmon_readCountersThread = perfmon_readCountersThread_silvermont;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_silvermont;
                    break;

                case ATOM_SILVERMONT_GOLD:
                case ATOM_DENVERTON:
                case ATOM_GOLDMONT_PLUS:
                    initialize_power = TRUE;
                    initialize_thermal = TRUE;
                    initThreadArch = perfmon_init_goldmont;
                    perfmon_startCountersThread = perfmon_startCountersThread_goldmont;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_goldmont;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_goldmont;
                    perfmon_readCountersThread = perfmon_readCountersThread_goldmont;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_goldmont;
                    break;

                case CORE_DUO:
                    ERROR_PLAIN_PRINT(Unsupported Processor);
                    break;

                case XEON_MP:
                case CORE2_65:
                case CORE2_45:
                    initThreadArch = perfmon_init_core2;
                    perfmon_startCountersThread = perfmon_startCountersThread_core2;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_core2;
                    perfmon_readCountersThread = perfmon_readCountersThread_core2;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_core2;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_core2;
                    break;

                case NEHALEM_EX:
                    initThreadArch = perfmon_init_nehalemEX;
                    perfmon_startCountersThread = perfmon_startCountersThread_nehalemEX;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_nehalemEX;
                    perfmon_readCountersThread = perfmon_readCountersThread_nehalemEX;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_nehalemEX;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_nehalemEX;
                    break;

                case WESTMERE_EX:
                    initThreadArch = perfmon_init_westmereEX;
                    perfmon_startCountersThread = perfmon_startCountersThread_westmereEX;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_westmereEX;
                    perfmon_readCountersThread = perfmon_readCountersThread_westmereEX;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_westmereEX;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_westmereEX;
                    break;

                case NEHALEM_BLOOMFIELD:
                case NEHALEM_LYNNFIELD:
                    initialize_thermal = TRUE;
                    initThreadArch = perfmon_init_nehalem;
                    perfmon_startCountersThread = perfmon_startCountersThread_nehalem;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_nehalem;
                    perfmon_readCountersThread = perfmon_readCountersThread_nehalem;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_nehalem;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_nehalem;
                    break;

                case NEHALEM_WESTMERE_M:
                case NEHALEM_WESTMERE:
                    initialize_thermal = TRUE;
                    initThreadArch = perfmon_init_nehalem;
                    perfmon_startCountersThread = perfmon_startCountersThread_nehalem;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_nehalem;
                    perfmon_readCountersThread = perfmon_readCountersThread_nehalem;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_nehalem;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_nehalem;
                    break;

                case IVYBRIDGE_EP:
                case IVYBRIDGE:
                    initialize_power = TRUE;
                    initialize_thermal = TRUE;
                    initThreadArch = perfmon_init_ivybridge;
                    perfmon_startCountersThread = perfmon_startCountersThread_ivybridge;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_ivybridge;
                    perfmon_readCountersThread = perfmon_readCountersThread_ivybridge;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_ivybridge;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_ivybridge;
                    break;

                case HASWELL_EP:
                case HASWELL:
                case HASWELL_M1:
                case HASWELL_M2:
                    initialize_power = TRUE;
                    initialize_thermal = TRUE;
                    initThreadArch = perfmon_init_haswell;
                    perfmon_startCountersThread = perfmon_startCountersThread_haswell;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_haswell;
                    perfmon_readCountersThread = perfmon_readCountersThread_haswell;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_haswell;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_haswell;
                    break;

                case SANDYBRIDGE_EP:
                case SANDYBRIDGE:
                    initialize_power = TRUE;
                    initialize_thermal = TRUE;
                    initThreadArch = perfmon_init_sandybridge;
                    perfmon_startCountersThread = perfmon_startCountersThread_sandybridge;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_sandybridge;
                    perfmon_readCountersThread = perfmon_readCountersThread_sandybridge;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_sandybridge;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_sandybridge;
                    break;

                case BROADWELL:
                case BROADWELL_E:
                case BROADWELL_D:
                case BROADWELL_E3:
                    initialize_power = TRUE;
                    initialize_thermal = TRUE;
                    initThreadArch = perfmon_init_broadwell;
                    perfmon_startCountersThread = perfmon_startCountersThread_broadwell;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_broadwell;
                    perfmon_readCountersThread = perfmon_readCountersThread_broadwell;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_broadwell;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_broadwell;
                    break;

                case SKYLAKE1:
                case SKYLAKE2:
                case SKYLAKEX:
                case KABYLAKE1:
                case KABYLAKE2:
                    initialize_power = TRUE;
                    initialize_thermal = TRUE;
                    initThreadArch = perfmon_init_skylake;
                    perfmon_startCountersThread = perfmon_startCountersThread_skylake;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_skylake;
                    perfmon_readCountersThread = perfmon_readCountersThread_skylake;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_skylake;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_skylake;
                    break;

                case XEON_PHI_KNL:
                    initialize_power = TRUE;
                    initialize_thermal = TRUE;
                    initThreadArch = perfmon_init_knl;
                    perfmon_startCountersThread = perfmon_startCountersThread_knl;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_knl;
                    perfmon_readCountersThread = perfmon_readCountersThread_knl;
                    perfmon_setupCountersThread = perfmon_setupCountersThread_knl;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_knl;
                    break;

                default:
                    ERROR_PLAIN_PRINT(Unsupported Processor);
                    break;
            }
            break;

        case MIC_FAMILY:

            switch ( cpuid_info.model )
            {
                case XEON_PHI:
                    initThreadArch = perfmon_init_phi;
                    perfmon_startCountersThread = perfmon_startCountersThread_phi;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_phi;
                    perfmon_readCountersThread = perfmon_readCountersThread_phi;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_phi;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_phi;
                    break;

                default:
                    ERROR_PLAIN_PRINT(Unsupported Processor);
                    break;
            }
            break;

        case K8_FAMILY:
            initThreadArch = perfmon_init_k10;
            perfmon_startCountersThread = perfmon_startCountersThread_k10;
            perfmon_stopCountersThread = perfmon_stopCountersThread_k10;
            perfmon_readCountersThread = perfmon_readCountersThread_k10;
            perfmon_setupCountersThread = perfmon_setupCounterThread_k10;
            perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_k10;
            break;

        case K10_FAMILY:
            initThreadArch = perfmon_init_k10;
            perfmon_startCountersThread = perfmon_startCountersThread_k10;
            perfmon_stopCountersThread = perfmon_stopCountersThread_k10;
            perfmon_readCountersThread = perfmon_readCountersThread_k10;
            perfmon_setupCountersThread = perfmon_setupCounterThread_k10;
            perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_k10;
            break;

        case K15_FAMILY:
            initThreadArch = perfmon_init_interlagos;
            perfmon_startCountersThread = perfmon_startCountersThread_interlagos;
            perfmon_stopCountersThread = perfmon_stopCountersThread_interlagos;
            perfmon_readCountersThread = perfmon_readCountersThread_interlagos;
            perfmon_setupCountersThread = perfmon_setupCounterThread_interlagos;
            perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_interlagos;
            break;

        case K16_FAMILY:
            initThreadArch = perfmon_init_kabini;
            perfmon_startCountersThread = perfmon_startCountersThread_kabini;
            perfmon_stopCountersThread = perfmon_stopCountersThread_kabini;
            perfmon_readCountersThread = perfmon_readCountersThread_kabini;
            perfmon_setupCountersThread = perfmon_setupCounterThread_kabini;
            perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_kabini;
            break;

        case ZEN_FAMILY:
            initThreadArch = perfmon_init_zen;
            initialize_power = TRUE;
            perfmon_startCountersThread = perfmon_startCountersThread_zen;
            perfmon_stopCountersThread = perfmon_stopCountersThread_zen;
            perfmon_readCountersThread = perfmon_readCountersThread_zen;
            perfmon_setupCountersThread = perfmon_setupCounterThread_zen;
            perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_zen;
            break;

        default:
            ERROR_PLAIN_PRINT(Unsupported Processor);
            break;
    }
#else
    initThreadArch = perfmon_init_perfevent;
    perfmon_startCountersThread = perfmon_startCountersThread_perfevent;
    perfmon_stopCountersThread = perfmon_stopCountersThread_perfevent;
    perfmon_readCountersThread = perfmon_readCountersThread_perfevent;
    perfmon_setupCountersThread = perfmon_setupCountersThread_perfevent;
    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_perfevent;
#endif
    *init_power = initialize_power;
    *init_temp = initialize_thermal;
}

int
perfmon_init(int nrThreads, const int* threadsToCpu)
{
    int i;
    int ret;
    int initialize_power = FALSE;
    int initialize_thermal = FALSE;

    if (perfmon_initialized == 1)
    {
        return 0;
    }

    if (nrThreads <= 0)
    {
        ERROR_PRINT(Number of threads must be greater than 0 but only %d given,nrThreads);
        return -EINVAL;
    }

    if (!lock_check())
    {
        ERROR_PLAIN_PRINT(Access to performance monitoring registers locked);
        return -EINVAL;
    }

    if ((cpuid_info.family == 0) && (cpuid_info.model == 0))
    {
        ERROR_PLAIN_PRINT(Topology module not inialized. Needed to determine current CPU type);
        return -ENODEV;
    }

    /* Check threadsToCpu array if only valid cpu_ids are listed */
    if (groupSet != NULL)
    {
        /* TODO: Decision whether setting new thread count and adjust processorIds
         *          or just exit like implemented now
         */
        return -EEXIST;
    }

    groupSet = (PerfmonGroupSet*) malloc(sizeof(PerfmonGroupSet));
    if (groupSet == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate group descriptor);
        return -ENOMEM;
    }
    groupSet->threads = (PerfmonThread*) malloc(nrThreads * sizeof(PerfmonThread));
    if (groupSet->threads == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate set of threads);
        free(groupSet);
        return -ENOMEM;
    }
    groupSet->numberOfThreads = nrThreads;
    groupSet->numberOfGroups = 0;
    groupSet->numberOfActiveGroups = 0;
    groupSet->groups = NULL;
    groupSet->activeGroup = -1;

    for(i=0; i<MAX_NUM_NODES; i++) socket_lock[i] = LOCK_INIT;
    for(i=0; i<MAX_NUM_THREADS; i++)
    {
        tile_lock[i] = LOCK_INIT;
        core_lock[i] = LOCK_INIT;
        sharedl3_lock[i] = LOCK_INIT;
    }


    /* Initialize access interface */
#ifndef LIKWID_USE_PERFEVENT
    ret = HPMinit();
    if (ret)
    {
        ERROR_PLAIN_PRINT(Cannot set access functions);
        free(groupSet->threads);
        free(groupSet);
        return ret;
    }
#endif
    timer_init();

    /* Initialize maps pointer to current architecture maps */
    perfmon_init_maps();

    /* Initialize function pointer to current architecture functions */
    perfmon_init_funcs(&initialize_power, &initialize_thermal);

    /* Store thread information and reset counters for processor*/
    /* If the arch supports it, initialize power and thermal measurements */
    for(i=0;i<nrThreads;i++)
    {
#ifndef LIKWID_USE_PERFEVENT
        ret = HPMaddThread(threadsToCpu[i]);
        if (ret != 0)
        {
            ERROR_PLAIN_PRINT(Cannot get access to performance counters);
            free(groupSet->threads);
            free(groupSet);
            return ret;
        }

        ret = HPMcheck(MSR_DEV, threadsToCpu[i]);
        if (ret != 1)
        {
            fprintf(stderr, "Cannot get access to MSRs. Please check permissions to the MSRs\n");
            free(groupSet->threads);
            free(groupSet);
            return -EACCES;
        }
#endif
        groupSet->threads[i].thread_id = i;
        groupSet->threads[i].processorId = threadsToCpu[i];

        if (initialize_power == TRUE)
        {
            power_init(threadsToCpu[i]);
        }
        if (initialize_thermal == TRUE)
        {
            thermal_init(threadsToCpu[i]);
        }
        initThreadArch(threadsToCpu[i]);
    }
    perfmon_initialized = 1;
    return 0;
}

void
perfmon_finalize(void)
{
    int group, event;
    int thread;
    if (perfmon_initialized == 0)
    {
        return;
    }
    if (groupSet == NULL)
    {
        return;
    }
    for(group=0;group < groupSet->numberOfActiveGroups; group++)
    {
        for (thread=0;thread< groupSet->numberOfThreads; thread++)
        {
            perfmon_finalizeCountersThread(thread, &(groupSet->groups[group]));
        }
        for (event=0;event < groupSet->groups[group].numberOfEvents; event++)
        {
            if (groupSet->groups[group].events[event].threadCounter)
                free(groupSet->groups[group].events[event].threadCounter);
        }
        if (groupSet->groups[group].events != NULL)
            free(groupSet->groups[group].events);
        perfmon_delEventSet(group);
        groupSet->groups[group].state = STATE_NONE;
    }
    if (groupSet->groups != NULL)
        free(groupSet->groups);
    if (groupSet->threads != NULL)
        free(groupSet->threads);
    groupSet->activeGroup = -1;
    if (groupSet)
        free(groupSet);
    for (group=0; group < MAX_NUM_THREADS; group++)
    {
        memset(currentConfig[group], 0, NUM_PMC * sizeof(uint64_t));
    }
    if (markerResults != NULL)
    {
        perfmon_destroyMarkerResults();
    }
    power_finalize();
#ifndef LIKWID_USE_PERFEVENT
    HPMfinalize();
#endif
    perfmon_initialized = 0;
    groupSet = NULL;
    return;
}

int
perfmon_addEventSet(const char* eventCString)
{
    int i, j, err, isPerfGroup = 0;
    bstring eventBString;
    struct bstrList* eventtokens;
    PerfmonEventSet* eventSet;
    PerfmonEventSetEntry* event;
    char* cstringcopy;
    Configuration_t config;
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    config = get_configuration();

    if (eventCString == NULL)
    {
        DEBUG_PLAIN_PRINT(DEBUGLEV_INFO, Event string is empty. Trying environment variable LIKWID_EVENTS);
        eventCString = getenv("LIKWID_EVENTS");
        if (eventCString == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot read event string. Also event string from environment variable is empty);
            return -EINVAL;
        }
    }

    if (strchr(eventCString, '-') != NULL)
    {
        ERROR_PLAIN_PRINT(Event string contains invalid character -);
        return -EINVAL;
    }
    if (strchr(eventCString, '.') != NULL)
    {
        ERROR_PLAIN_PRINT(Event string contains invalid character .);
        return -EINVAL;
    }
    if (groupSet->numberOfActiveGroups == 0)
    {
        groupSet->groups = (PerfmonEventSet*) malloc(sizeof(PerfmonEventSet));
        if (groupSet->groups == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate initialize of event group list);
            return -ENOMEM;
        }
        groupSet->numberOfGroups = 1;
        groupSet->numberOfActiveGroups = 0;
        groupSet->activeGroup = -1;

        /* Only one group exists by now */
        groupSet->groups[0].rdtscTime = 0;
        groupSet->groups[0].runTime = 0;
        groupSet->groups[0].numberOfEvents = 0;
    }

    if ((groupSet->numberOfActiveGroups > 0) && (groupSet->numberOfActiveGroups == groupSet->numberOfGroups))
    {
        groupSet->numberOfGroups++;
        groupSet->groups = (PerfmonEventSet*)realloc(groupSet->groups, groupSet->numberOfGroups*sizeof(PerfmonEventSet));
        if (groupSet->groups == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate additional group);
            return -ENOMEM;
        }
        groupSet->groups[groupSet->numberOfActiveGroups].rdtscTime = 0;
        groupSet->groups[groupSet->numberOfActiveGroups].runTime = 0;
        groupSet->groups[groupSet->numberOfActiveGroups].numberOfEvents = 0;
        DEBUG_PLAIN_PRINT(DEBUGLEV_INFO, Allocating new group structure for group.);
    }
    DEBUG_PRINT(DEBUGLEV_INFO, Currently %d groups of %d active,
                    groupSet->numberOfActiveGroups+1,
                    groupSet->numberOfGroups+1);
    cstringcopy = malloc((strlen(eventCString)+1)*sizeof(char));
    if (!cstringcopy)
        return -ENOMEM;
    strcpy(cstringcopy, eventCString);
    char* perf_pid = strstr(eventCString, "PERF_PID");
    if (perf_pid != NULL)
    {
#ifdef LIKWID_USE_PERFEVENT
        snprintf(cstringcopy, strlen(eventCString)-strlen(perf_pid), "%s", eventCString);
#endif
    }

    if (strchr(cstringcopy, ':') == NULL)
    {
        err = read_group(config->groupPath, cpuid_info.short_name,
                         cstringcopy,
                         &groupSet->groups[groupSet->numberOfActiveGroups].group);
        if (err == -EACCES)
        {
            ERROR_PRINT(Access to performance group %s not allowed, cstringcopy);
            return err;
        }
        else if (err == -ENODEV)
        {
            ERROR_PRINT(Performance group %s only available with deactivated HyperThreading, eventCString);
            return err;
        }
        else if (err < 0)
        {
            ERROR_PRINT(Cannot read performance group %s, cstringcopy);
            return err;
        }
        isPerfGroup = 1;
    }
    else
    {
        err = custom_group(cstringcopy, &groupSet->groups[groupSet->numberOfActiveGroups].group);
        if (err)
        {
            ERROR_PRINT(Cannot transform %s to performance group, cstringcopy);
            return err;
        }
    }
    char * evstr = get_eventStr(&groupSet->groups[groupSet->numberOfActiveGroups].group);
    if (perf_pid != NULL)
    {
        char* tmp = realloc(evstr, strlen(evstr)+strlen(perf_pid)+1);
        if (!tmp)
        {
            return -ENOMEM;
        }
        else
        {
            evstr = tmp;
            strcat(evstr, ":");
            strcat(evstr, perf_pid);
        }
    }
    free(cstringcopy);
    eventBString = bfromcstr(evstr);
    eventtokens = bsplit(eventBString,',');
    free(evstr);
    bdestroy(eventBString);

    eventSet = &(groupSet->groups[groupSet->numberOfActiveGroups]);
    eventSet->events = (PerfmonEventSetEntry*) malloc(eventtokens->qty * sizeof(PerfmonEventSetEntry));
    if (eventSet->events == NULL)
    {
        ERROR_PRINT(Cannot allocate event list for group %d\n, groupSet->numberOfActiveGroups);
        return -ENOMEM;
    }
    eventSet->numberOfEvents = 0;

    eventSet->regTypeMask1 = 0x0ULL;
    eventSet->regTypeMask2 = 0x0ULL;
    eventSet->regTypeMask3 = 0x0ULL;
    eventSet->regTypeMask4 = 0x0ULL;

    int forceOverwrite = 0;
    int valid_events = 0;
    if (getenv("LIKWID_FORCE") != NULL)
    {
        forceOverwrite = atoi(getenv("LIKWID_FORCE"));
    }
    for(i=0;i<eventtokens->qty;i++)
    {
        event = &(eventSet->events[i]);
        struct bstrList* subtokens = bsplit(eventtokens->entry[i],':');
        if (subtokens->qty < 2)
        {
            ERROR_PRINT(Cannot parse event descriptor %s, bdata(eventtokens->entry[i]));
            bstrListDestroy(subtokens);
            continue;
        }
        else
        {
            if (!getIndexAndType(subtokens->entry[1], &event->index, &event->type))
            {
                fprintf(stderr, "WARN: Counter %s not defined for current architecture\n", bdata(subtokens->entry[1]));
                event->type = NOTYPE;
                goto past_checks;
            }
#ifndef LIKWID_USE_PERFEVENT
            event->type = checkAccess(subtokens->entry[1], event->index, event->type, forceOverwrite);
            if (event->type == NOTYPE)
            {
                DEBUG_PRINT(DEBUGLEV_INFO, Cannot access counter register %s, bdata(subtokens->entry[1]));
                event->type = NOTYPE;
                goto past_checks;
            }
#else
            char* path = translate_types[counter_map[event->index].type];
            struct stat st;
            if (path == NULL || stat(path, &st) != 0)
            {
                DEBUG_PRINT(DEBUGLEV_INFO, Cannot access counter register %s, bdata(subtokens->entry[1]));
                event->type = NOTYPE;
                goto past_checks;
            }
#endif

            if (!getEvent(subtokens->entry[0], subtokens->entry[1], &event->event))
            {
                fprintf(stderr, "WARN: Event %s not found for current architecture\n", bdata(subtokens->entry[0]));
                event->type = NOTYPE;
                goto past_checks;
            }
#ifndef LIKWID_USE_PERFEVENT
            if (!checkCounter(subtokens->entry[1], event->event.limit))
            {
                fprintf(stderr, "WARN: Register %s not allowed for event %s (limit %s)\n", bdata(subtokens->entry[1]),bdata(subtokens->entry[0]),event->event.limit);
                event->type = NOTYPE;
                goto past_checks;
            }
#endif
            if (parseOptions(subtokens, &event->event, event->index) < 0)
            {
                event->type = NOTYPE;
                goto past_checks;
            }

            SETTYPE(eventSet, event->type);

            for (int e = 0; e < eventSet->numberOfEvents; e++)
            {
                if (event->index == eventSet->events[e].index)
                {
                    fprintf(stderr, "WARN: Counter %s already used in event set, skipping\n", counter_map[event->index].key);
                    event->type = NOTYPE;
                    break;
                }
            }

past_checks:
            event->threadCounter = (PerfmonCounter*) malloc(
                groupSet->numberOfThreads * sizeof(PerfmonCounter));

            if (event->threadCounter == NULL)
            {
                ERROR_PRINT(Cannot allocate counter for all threads in group %d,groupSet->numberOfActiveGroups);
                //bstrListDestroy(subtokens);
                continue;
            }
            for(j=0;j<groupSet->numberOfThreads;j++)
            {
                event->threadCounter[j].counterData = 0;
                event->threadCounter[j].startData = 0;
                event->threadCounter[j].fullResult = 0.0;
                event->threadCounter[j].lastResult = 0.0;
                event->threadCounter[j].overflows = 0;
                event->threadCounter[j].init = FALSE;
            }

            eventSet->numberOfEvents++;

            if (event->type != NOTYPE)
            {
                valid_events++;
                DEBUG_PRINT(DEBUGLEV_INFO,
                        Added event %s for counter %s to group %d,
                        event->event.name,
                        counter_map[event->index].key,
                        groupSet->numberOfActiveGroups);
            }
        }
        bstrListDestroy(subtokens);
    }
    bstrListDestroy(eventtokens);
    int fixed_counters = 0;
    char fix[] = "FIXC";
    char* ptr;
    ptr = strstr(eventCString, fix);
    if (cpuid_info.isIntel && !ptr)
    {
        fixed_counters = cpuid_info.perf_num_fixed_ctr;
    }

    if (((valid_events > fixed_counters) || isPerfGroup) &&
        ((eventSet->regTypeMask1 != 0x0ULL) ||
        (eventSet->regTypeMask2 != 0x0ULL) ||
        (eventSet->regTypeMask3 != 0x0ULL) ||
        (eventSet->regTypeMask4 != 0x0ULL)))
    {
        eventSet->state = STATE_NONE;
        groupSet->numberOfActiveGroups++;
        return groupSet->numberOfActiveGroups-1;
    }
    else
    {
        fprintf(stderr,"ERROR: No event in given event string can be configured.\n");
        fprintf(stderr,"       Either the events or counters do not exist for the\n");
        fprintf(stderr,"       current architecture. If event options are set, they might\n");
        fprintf(stderr,"       be invalid.\n");
        return -EINVAL;
    }
}

void
perfmon_delEventSet(int groupID)
{
    if (groupID >= groupSet->numberOfGroups || groupID < 0)
        return;
    return_group(&groupSet->groups[groupID].group);
    return;
}

int
__perfmon_setupCountersThread(int thread_id, int groupId)
{
    int i = 0;
    int ret = 0;
    if (groupId >= groupSet->numberOfActiveGroups)
    {
        ERROR_PRINT(Group %d does not exist in groupSet, groupId);
        return -ENOENT;
    }

    ret = perfmon_setupCountersThread(thread_id, &groupSet->groups[groupId]);
    if (ret < 0)
    {
        fprintf(stderr, "Setup of counters failed for thread %d\n", (ret+1)*-1);
        return ret;
    }

    groupSet->activeGroup = groupId;
    return 0;
}

int
perfmon_setupCounters(int groupId)
{
    int i;
    int ret = 0;
    if (!lock_check())
    {
        ERROR_PLAIN_PRINT(Access to performance monitoring registers locked);
        return -ENOLCK;
    }
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (unlikely(groupSet == NULL))
    {
        return -EINVAL;
    }

    if (groupId >= groupSet->numberOfActiveGroups)
    {
        ERROR_PRINT(Group %d does not exist in groupSet, groupId);
        return -ENOENT;
    }

    for(i=0;i<groupSet->numberOfThreads;i++)
    {
        ret = __perfmon_setupCountersThread(groupSet->threads[i].thread_id, groupId);
        if (ret != 0)
        {
            return ret;
        }
    }
    groupSet->groups[groupId].state = STATE_SETUP;
    return 0;
}

int
__perfmon_startCounters(int groupId)
{
    int i = 0, j = 0;
    int ret = 0;
    if (groupSet->groups[groupId].state != STATE_SETUP)
    {
        return -EINVAL;
    }
    if (!lock_check())
    {
        ERROR_PLAIN_PRINT(Access to performance monitoring registers locked);
        return -ENOLCK;
    }
    for(;i<groupSet->numberOfThreads;i++)
    {
        for (j=0; j<perfmon_getNumberOfEvents(groupId); j++)
            groupSet->groups[groupId].events[j].threadCounter[i].overflows = 0;
        ret = perfmon_startCountersThread(groupSet->threads[i].thread_id, &groupSet->groups[groupId]);
        if (ret)
        {
            return -groupSet->threads[i].thread_id-1;
        }
    }
    groupSet->groups[groupId].state = STATE_START;
    timer_start(&groupSet->groups[groupId].timer);
    return 0;
}

int perfmon_startCounters(void)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (unlikely(groupSet == NULL))
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (groupSet->activeGroup < 0)
    {
        ERROR_PLAIN_PRINT(Cannot find group to start);
        return -EINVAL;
    }
    return __perfmon_startCounters(groupSet->activeGroup);
}

int perfmon_startGroupCounters(int groupId)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (unlikely(groupSet == NULL))
    {
        return -EINVAL;
    }
    if (((groupId < 0) || (groupId >= groupSet->numberOfActiveGroups)) && (groupSet->activeGroup >= 0))
    {
        groupId = groupSet->activeGroup;
    }
    else
    {
        ERROR_PLAIN_PRINT(Cannot find group to start);
        return -EINVAL;
    }
    return __perfmon_startCounters(groupId);
}

int
__perfmon_stopCounters(int groupId)
{
    int i = 0;
    int j = 0;
    int ret = 0;
    double result = 0.0;

    if (!lock_check())
    {
        ERROR_PLAIN_PRINT(Access to performance monitoring registers locked);
        return -ENOLCK;
    }

    timer_stop(&groupSet->groups[groupId].timer);

    for (i = 0; i<groupSet->numberOfThreads; i++)
    {
        ret = perfmon_stopCountersThread(groupSet->threads[i].thread_id, &groupSet->groups[groupId]);
        if (ret)
        {
            return -groupSet->threads[i].thread_id-1;
        }
    }

    for (i=0; i<perfmon_getNumberOfEvents(groupId); i++)
    {
        for (j=0; j<perfmon_getNumberOfThreads(); j++)
        {
            result = (double)calculateResult(groupId, i, j);
            groupSet->groups[groupId].events[i].threadCounter[j].lastResult = result;
            groupSet->groups[groupId].events[i].threadCounter[j].fullResult += result;
        }
    }
    groupSet->groups[groupId].state = STATE_SETUP;
    groupSet->groups[groupId].rdtscTime =
                timer_print(&groupSet->groups[groupId].timer);
    groupSet->groups[groupId].runTime += groupSet->groups[groupId].rdtscTime;
    return 0;
}

int
perfmon_stopCounters(void)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (unlikely(groupSet == NULL))
    {
        return -EINVAL;
    }
    if (groupSet->activeGroup < 0)
    {
        ERROR_PLAIN_PRINT(Cannot find group to start);
        return -EINVAL;
    }
    if (groupSet->groups[groupSet->activeGroup].state != STATE_START)
    {
        return -EINVAL;
    }
    return __perfmon_stopCounters(groupSet->activeGroup);
}

int
perfmon_stopGroupCounters(int groupId)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (unlikely(groupSet == NULL))
    {
        return -EINVAL;
    }
    if (((groupId < 0) || (groupId >= groupSet->numberOfActiveGroups)) && (groupSet->activeGroup >= 0))
    {
        groupId = groupSet->activeGroup;
    }
    else
    {
        ERROR_PLAIN_PRINT(Cannot find group to start);
        return -EINVAL;
    }
    if (groupSet->groups[groupId].state != STATE_START)
    {
        return -EINVAL;
    }
    return __perfmon_stopCounters(groupId);
}

int
__perfmon_readCounters(int groupId, int threadId)
{
    int ret = 0;
    int i = 0, j = 0;
    double result = 0.0;
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (((groupId < 0) || (groupId >= groupSet->numberOfActiveGroups)) && (groupSet->activeGroup >= 0))
    {
        groupId = groupSet->activeGroup;
    }
    if (groupSet->groups[groupId].state != STATE_START)
    {
        return -EINVAL;
    }
    timer_stop(&groupSet->groups[groupId].timer);
    groupSet->groups[groupId].rdtscTime = timer_print(&groupSet->groups[groupId].timer);
    groupSet->groups[groupId].runTime += groupSet->groups[groupId].rdtscTime;
    if (threadId == -1)
    {
        for (threadId = 0; threadId<groupSet->numberOfThreads; threadId++)
        {
            ret = perfmon_readCountersThread(threadId, &groupSet->groups[groupId]);
            if (ret)
            {
                return -threadId-1;
            }
            for (j=0; j < groupSet->groups[groupId].numberOfEvents; j++)
            {
                if (groupSet->groups[groupId].events[j].type != NOTYPE)
                {
                    result = (double)calculateResult(groupId, j, threadId);
                    groupSet->groups[groupId].events[j].threadCounter[threadId].lastResult = result;
                    groupSet->groups[groupId].events[j].threadCounter[threadId].fullResult += result;
                    groupSet->groups[groupId].events[j].threadCounter[threadId].startData =
                        groupSet->groups[groupId].events[j].threadCounter[threadId].counterData;
                }
            }
        }
    }
    else if ((threadId >= 0) && (threadId < groupSet->numberOfThreads))
    {
        ret = perfmon_readCountersThread(threadId, &groupSet->groups[groupId]);
        if (ret)
        {
            return -threadId-1;
        }
        for (j=0; j < groupSet->groups[groupId].numberOfEvents; j++)
        {
            result = (double)calculateResult(groupId, j, threadId);
            groupSet->groups[groupId].events[j].threadCounter[threadId].lastResult = result;
            groupSet->groups[groupId].events[j].threadCounter[threadId].fullResult += result;
            groupSet->groups[groupId].events[j].threadCounter[threadId].startData =
                groupSet->groups[groupId].events[j].threadCounter[threadId].counterData;
        }
}
    timer_start(&groupSet->groups[groupId].timer);
    return 0;
}

int
perfmon_readCounters(void)
{
    return __perfmon_readCounters(-1,-1);
}

int
perfmon_readCountersCpu(int cpu_id)
{
    int i;
    int thread_id = -1;
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    for(i=0;i<groupSet->numberOfThreads;i++)
    {
        if (groupSet->threads[i].processorId == cpu_id)
        {
            thread_id = groupSet->threads[i].thread_id;
            break;
        }
    }
    if (thread_id < 0)
    {
        ERROR_PRINT(Failed to read counters for CPU %d, cpu_id);
        return -thread_id;
    }
    i = __perfmon_readCounters(groupSet->activeGroup, thread_id);
    return i;
}

int
perfmon_readGroupCounters(int groupId)
{
    return __perfmon_readCounters(groupId, -1);
}

int
perfmon_readGroupThreadCounters(int groupId, int threadId)
{
    return __perfmon_readCounters(groupId, threadId);
}

int
perfmon_isUncoreCounter(char* counter)
{
    char fix[] = "FIXC";
    char pmc[] = "PMC";
    char upmc[] = "UPMC";
    char tmp[] = "TMP";
    char *ptr = NULL;
    ptr = strstr(counter, fix);
    if (ptr)
    {
        return 0;
    }
    ptr = NULL;
    ptr = strstr(counter, tmp);
    if (ptr)
    {
        return 0;
    }
    ptr = NULL;
    ptr = strstr(counter, pmc);
    if (ptr)
    {
        ptr = strstr(counter, upmc);
        if (!ptr)
        {
            return 0;
        }
    }
    return 1;
}

double
perfmon_getResult(int groupId, int eventId, int threadId)
{
    if (unlikely(groupSet == NULL))
    {
        return NAN;
    }
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return NAN;
    }
    if (groupSet->numberOfActiveGroups == 0)
    {
        return NAN;
    }
    if ((groupId < 0) && (groupSet->activeGroup >= 0))
    {
        groupId = groupSet->activeGroup;
    }
    if (eventId >= groupSet->groups[groupId].numberOfEvents)
    {
        printf("ERROR: EventID greater than defined events\n");
        return NAN;
    }
    if (threadId >= groupSet->numberOfThreads)
    {
        printf("ERROR: ThreadID greater than defined threads\n");
        return NAN;
    }
    if (groupSet->groups[groupId].events[eventId].type == NOTYPE)
        return NAN;

    if ((groupSet->groups[groupId].events[eventId].threadCounter[threadId].fullResult == 0) ||
        (groupSet->groups[groupId].events[eventId].type == THERMAL) ||
        (groupSet->groups[groupId].events[eventId].type == QBOX0FIX) ||
        (groupSet->groups[groupId].events[eventId].type == QBOX1FIX) ||
        (groupSet->groups[groupId].events[eventId].type == QBOX2FIX) ||
        (groupSet->groups[groupId].events[eventId].type == SBOX0FIX) ||
        (groupSet->groups[groupId].events[eventId].type == SBOX1FIX) ||
        (groupSet->groups[groupId].events[eventId].type == SBOX2FIX))
    {
        return groupSet->groups[groupId].events[eventId].threadCounter[threadId].lastResult;
    }
    return groupSet->groups[groupId].events[eventId].threadCounter[threadId].fullResult;
}

double
perfmon_getLastResult(int groupId, int eventId, int threadId)
{
    if (unlikely(groupSet == NULL))
    {
        return 0;
    }
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return 0;
    }
    if (groupSet->numberOfActiveGroups == 0)
    {
        return 0;
    }
    if ((groupId < 0) && (groupSet->activeGroup >= 0))
    {
        groupId = groupSet->activeGroup;
    }
    if (eventId >= groupSet->groups[groupId].numberOfEvents)
    {
        printf("ERROR: EventID greater than defined events\n");
        return 0;
    }
    if (threadId >= groupSet->numberOfThreads)
    {
        printf("ERROR: ThreadID greater than defined threads\n");
        return 0;
    }
    if (groupSet->groups[groupId].events[eventId].type == NOTYPE)
        return 0;

    return groupSet->groups[groupId].events[eventId].threadCounter[threadId].lastResult;
}

double
perfmon_getMetric(int groupId, int metricId, int threadId)
{
    int e = 0;
    double result = 0;
    bstring vars = bformat("");
    bstring varlist = bformat("");
    char split[2] = ":";
    if (unlikely(groupSet == NULL))
    {
        return NAN;
    }
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return NAN;
    }
    if (groupSet->numberOfActiveGroups == 0)
    {
        return NAN;
    }
    if ((groupId < 0) && (groupSet->activeGroup >= 0))
    {
        groupId = groupSet->activeGroup;
    }
    if (groupSet->groups[groupId].group.nmetrics == 0)
    {
        return NAN;
    }
    if ((metricId < 0) || (metricId >= groupSet->groups[groupId].group.nmetrics))
    {
        return NAN;
    }
    char* f = groupSet->groups[groupId].group.metricformulas[metricId];
    for (e=0;e<groupSet->groups[groupId].numberOfEvents;e++)
    {
        if (groupSet->groups[groupId].events[e].type != NOTYPE)
        {
            char *ctr = strtok(groupSet->groups[groupId].group.counters[e], split);
            calc_add_dbl_var(ctr, perfmon_getResult(groupId, e, threadId), vars, varlist);
        }
        else
        {
            char *ctr = strtok(groupSet->groups[groupId].group.counters[e], split);
            if (strstr(f, ctr) != NULL)
            {
                bdestroy(vars);
                bdestroy(varlist);
                return NAN;
            }
        }
    }
    if (groupSet->groups[groupId].group.lua_funcs)
    {
        if (calc_set_user_funcs(groupSet->groups[groupId].group.lua_funcs))
        {
            free(groupSet->groups[groupId].group.lua_funcs);
            groupSet->groups[groupId].group.lua_funcs = NULL;
        }
    }
    calc_add_dbl_var("time", perfmon_getLastTimeOfGroup(groupId), vars, varlist);
    calc_add_dbl_var("inverseClock", 1.0/timer_getCycleClock(), vars, varlist);
    int cpu = 0, sock_cpu = 0, err = 0;
    for (e=0; e<groupSet->numberOfThreads; e++)
    {
        if (groupSet->threads[e].thread_id == threadId)
        {
            cpu = groupSet->threads[e].processorId;
        }
    }
    sock_cpu = socket_lock[affinity_thread2socket_lookup[cpu]];
    if (cpu != sock_cpu)
    {
        for (e=0; e<groupSet->numberOfThreads; e++)
        {
            if (groupSet->threads[e].processorId == sock_cpu)
            {
                sock_cpu = groupSet->threads[e].thread_id;
            }
        }
        calc_add_int_var("SOCKET_CPU", sock_cpu, vars, varlist);
        for (e=0;e<groupSet->groups[groupId].numberOfEvents;e++)
        {
            if (perfmon_isUncoreCounter(groupSet->groups[groupId].group.counters[e]) &&
                !perfmon_isUncoreCounter(groupSet->groups[groupId].group.metricformulas[metricId]))
            {
                if (groupSet->groups[groupId].events[e].type != NOTYPE)
                {
                    char *ctr = strtok(groupSet->groups[groupId].group.counters[e], split);
                    double res = perfmon_getResult(groupId, e, sock_cpu);
                    err = calc_add_dbl_var(ctr, res, vars, varlist);
                    if (err < 0)
                    {
                        DEBUG_PRINT(DEBUGLEV_DEVELOP, Cannot add socket result of counter %s for thread %d, groupSet->groups[groupId].group.counters[e], threadId);
                    }
                }
            }
        }
    }
    else
    {
        calc_add_int_var("SOCKET_CPU", cpu, vars, varlist);
    }
    
    e = calc_metric(cpu, f, vars, varlist, &result);
    bdestroy(vars);
    bdestroy(varlist);
    if (e < 0)
    {
        result = NAN;
    }
    return result;
}

double
perfmon_getLastMetric(int groupId, int metricId, int threadId)
{
    int e = 0;
    double result = 0;
    bstring vars = bformat("");
    bstring varlist = bformat("");
    char split[2] = ":";
    if (unlikely(groupSet == NULL))
    {
        return NAN;
    }
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return NAN;
    }
    if (groupSet->numberOfActiveGroups == 0)
    {
        return NAN;
    }
    if ((groupId < 0) && (groupSet->activeGroup >= 0))
    {
        groupId = groupSet->activeGroup;
    }
    if (groupSet->groups[groupId].group.nmetrics == 0)
    {
        return NAN;
    }
    if ((metricId < 0) || (metricId >= groupSet->groups[groupId].group.nmetrics))
    {
        return NAN;
    }
    char* f = groupSet->groups[groupId].group.metricformulas[metricId];
    for (e=0;e<groupSet->groups[groupId].numberOfEvents;e++)
    {
        if (groupSet->groups[groupId].events[e].type != NOTYPE)
        {
            char *ctr = strtok(groupSet->groups[groupId].group.counters[e], split);
            calc_add_dbl_var(ctr, perfmon_getLastResult(groupId, e, threadId), vars, varlist);
        }
        else
        {
            char *ctr = strtok(groupSet->groups[groupId].group.counters[e], split);
            if (strstr(f, ctr) != NULL)
            {
                bdestroy(vars);
                bdestroy(varlist);
                return NAN;
            }
        }
    }
    if (groupSet->groups[groupId].group.lua_funcs)
    {
        if (calc_set_user_funcs(groupSet->groups[groupId].group.lua_funcs))
        {
            free(groupSet->groups[groupId].group.lua_funcs);
            groupSet->groups[groupId].group.lua_funcs = NULL;
        }
    }
    calc_add_dbl_var("time", perfmon_getLastTimeOfGroup(groupId), vars, varlist);
    calc_add_dbl_var("inverseClock", 1.0/timer_getCycleClock(), vars, varlist);
    int cpu = 0, sock_cpu = 0, err = 0;
    for (e=0; e<groupSet->numberOfThreads; e++)
    {
        if (groupSet->threads[e].thread_id == threadId)
        {
            cpu = groupSet->threads[e].processorId;
        }
    }
    calc_add_int_var("CPU", cpu, vars, varlist);
    sock_cpu = socket_lock[affinity_thread2socket_lookup[cpu]];
    if (cpu != sock_cpu)
    {
        for (e=0; e<groupSet->numberOfThreads; e++)
        {
            if (groupSet->threads[e].processorId == sock_cpu)
            {
                sock_cpu = groupSet->threads[e].thread_id;
            }
        }
        calc_add_int_var("SOCKET_CPU", sock_cpu, vars, varlist);
        for (e=0;e<groupSet->groups[groupId].numberOfEvents;e++)
        {
            if (perfmon_isUncoreCounter(groupSet->groups[groupId].group.counters[e]) &&
                !perfmon_isUncoreCounter(groupSet->groups[groupId].group.metricformulas[metricId]))
            {
                if (groupSet->groups[groupId].events[e].type != NOTYPE)
                {
                    char *ctr = strtok(groupSet->groups[groupId].group.counters[e], split);
                    double res = perfmon_getLastResult(groupId, e, sock_cpu);
                    err = calc_add_dbl_var(ctr, res, vars, varlist);
                    if (err < 0)
                    {
                        DEBUG_PRINT(DEBUGLEV_DEVELOP, Cannot add socket result of counter %s for thread %d, groupSet->groups[groupId].group.counters[e], threadId);
                    }
                }
            }
        }
    }
    else
    {
        calc_add_int_var("SOCKET_CPU", cpu, vars, varlist);
    }
    
    e = calc_metric(cpu, f, vars, varlist, &result);
    bdestroy(vars);
    bdestroy(varlist);
    if (e < 0)
    {
        result = NAN;
    }
    return result;
}

int
__perfmon_switchActiveGroupThread(int thread_id, int new_group)
{
    int ret = 0;
    int i = 0;
    GroupState state;
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }

    timer_stop(&groupSet->groups[groupSet->activeGroup].timer);
    groupSet->groups[groupSet->activeGroup].rdtscTime =
                timer_print(&groupSet->groups[groupSet->activeGroup].timer);
    groupSet->groups[groupSet->activeGroup].runTime += groupSet->groups[groupSet->activeGroup].rdtscTime;
    state = groupSet->groups[groupSet->activeGroup].state;

    if (state == STATE_START)
    {
        ret = perfmon_stopCounters();
    }

    if (state == STATE_SETUP)
    {
        for(i=0; i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
        {
            groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].init = FALSE;
        }
    }
    ret = perfmon_setupCounters(new_group);
    if (ret != 0)
    {
        return ret;
    }
    if (groupSet->groups[groupSet->activeGroup].state == STATE_SETUP)
    {
        ret = perfmon_startCounters();
        if (ret != 0)
        {
            return ret;
        }
    }
    return 0;
}

int
perfmon_switchActiveGroup(int new_group)
{
    int i = 0;
    int ret = 0;
    for(i=0;i<groupSet->numberOfThreads;i++)
    {
        ret = __perfmon_switchActiveGroupThread(groupSet->threads[i].thread_id, new_group);
        if (ret != 0)
        {
            return ret;
        }
    }
    return 0;
}

int
perfmon_getNumberOfGroups(void)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    return groupSet->numberOfActiveGroups;
}

int
perfmon_getIdOfActiveGroup(void)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    return groupSet->activeGroup;
}

int
perfmon_getNumberOfThreads(void)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    return groupSet->numberOfThreads;
}

int
perfmon_getNumberOfEvents(int groupId)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (groupId < 0)
    {
        groupId = groupSet->activeGroup;
    }
    return groupSet->groups[groupId].numberOfEvents;
}

double
perfmon_getTimeOfGroup(int groupId)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (groupId < 0)
    {
        groupId = groupSet->activeGroup;
    }
    return groupSet->groups[groupId].runTime;
}

double
perfmon_getLastTimeOfGroup(int groupId)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (groupId < 0)
    {
        groupId = groupSet->activeGroup;
    }
    return groupSet->groups[groupId].rdtscTime;
}

uint64_t
perfmon_getMaxCounterValue(RegisterType type)
{
    int width = 48;
    uint64_t tmp = 0x0ULL;
    if (box_map && (box_map[type].regWidth > 0))
    {
        width = box_map[type].regWidth;
    }
    for(int i=0;i<width;i++)
    {
        tmp |= (1ULL<<i);
    }
    return tmp;
}

char*
perfmon_getEventName(int groupId, int eventId)
{
    if (unlikely(groupSet == NULL))
    {
        return NULL;
    }
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return NULL;
    }
    if (groupSet->numberOfActiveGroups == 0)
    {
        return NULL;
    }
    if ((groupId < 0) && (groupSet->activeGroup >= 0))
    {
        groupId = groupSet->activeGroup;
    }
    if ((groupSet->groups[groupId].group.nevents == 0) ||
        (eventId > groupSet->groups[groupId].group.nevents))
    {
        return NULL;
    }
    return groupSet->groups[groupId].group.events[eventId];
}

char*
perfmon_getCounterName(int groupId, int eventId)
{
    if (unlikely(groupSet == NULL))
    {
        return NULL;
    }
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return NULL;
    }
    if (groupSet->numberOfActiveGroups == 0)
    {
        return NULL;
    }
    if ((groupId < 0) && (groupSet->activeGroup >= 0))
    {
        groupId = groupSet->activeGroup;
    }
    if ((groupSet->groups[groupId].group.nevents == 0) ||
        (eventId > groupSet->groups[groupId].group.nevents))
    {
        return NULL;
    }
    return groupSet->groups[groupId].group.counters[eventId];
}

char*
perfmon_getMetricName(int groupId, int metricId)
{
    if (unlikely(groupSet == NULL))
    {
        return NULL;
    }
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return NULL;
    }
    if (groupSet->numberOfActiveGroups == 0)
    {
        return NULL;
    }
    if ((groupId < 0) && (groupSet->activeGroup >= 0))
    {
        groupId = groupSet->activeGroup;
    }
    if (groupSet->groups[groupId].group.nmetrics == 0)
    {
        return NULL;
    }
    return groupSet->groups[groupId].group.metricnames[metricId];
}

char*
perfmon_getGroupName(int groupId)
{
    if (unlikely(groupSet == NULL))
    {
        return NULL;
    }
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return NULL;
    }
    if (groupSet->numberOfActiveGroups == 0)
    {
        return NULL;
    }
    if ((groupId < 0) && (groupSet->activeGroup >= 0))
    {
        groupId = groupSet->activeGroup;
    }
    return groupSet->groups[groupId].group.groupname;
}

char*
perfmon_getGroupInfoShort(int groupId)
{
    if (unlikely(groupSet == NULL))
    {
        return NULL;
    }
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return NULL;
    }
    if (groupSet->numberOfActiveGroups == 0)
    {
        return NULL;
    }
    if ((groupId < 0) && (groupSet->activeGroup >= 0))
    {
        groupId = groupSet->activeGroup;
    }
    return groupSet->groups[groupId].group.shortinfo;
}

char*
perfmon_getGroupInfoLong(int groupId)
{
    if (unlikely(groupSet == NULL))
    {
        return NULL;
    }
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return NULL;
    }
    if (groupSet->numberOfActiveGroups == 0)
    {
        return NULL;
    }
    if ((groupId < 0) && (groupSet->activeGroup >= 0))
    {
        groupId = groupSet->activeGroup;
    }
    return groupSet->groups[groupId].group.longinfo;
}

int
perfmon_getGroups(char*** groups, char*** shortinfos, char*** longinfos)
{
    int ret = 0;
    init_configuration();
    Configuration_t config = get_configuration();
    ret = get_groups(config->groupPath, cpuid_info.short_name, groups, shortinfos, longinfos);
    return ret;
}

void
perfmon_returnGroups(int nrgroups, char** groups, char** shortinfos, char** longinfos)
{
    return_groups(nrgroups, groups, shortinfos, longinfos);
}

int
perfmon_getNumberOfMetrics(int groupId)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (groupId < 0)
    {
        groupId = groupSet->activeGroup;
    }
    return groupSet->groups[groupId].group.nmetrics;
}

void
perfmon_printMarkerResults()
{
    int i = 0, j = 0, k = 0;
    for (i=0; i<markerRegions; i++)
    {
        printf("Region %d : %s\n", i, bdata(markerResults[i].tag));
        printf("Group %d\n", markerResults[i].groupID);
        for (j=0;j<markerResults[i].threadCount; j++)
        {
            printf("Thread %d on CPU %d\n", j, markerResults[i].cpulist[j]);
            printf("\t Measurement time %f sec\n", markerResults[i].time[j]);
            printf("\t Call count %d\n", markerResults[i].count[j]);
            for(k=0;k<markerResults[i].eventCount;k++)
            {
                printf("\t Event %d : %f\n", k, markerResults[i].counters[j][k]);
            }
        }
    }
}

int
perfmon_getNumberOfRegions()
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (markerResults == NULL)
    {
        return 0;
    }
    return markerRegions;
}

int
perfmon_getGroupOfRegion(int region)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= markerRegions)
    {
        return -EINVAL;
    }
    if (markerResults == NULL)
    {
        return 0;
    }
    return markerResults[region].groupID;
}

char*
perfmon_getTagOfRegion(int region)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return NULL;
    }
    if (region < 0 || region >= markerRegions)
    {
        return NULL;
    }
    if (markerResults == NULL)
    {
        return NULL;
    }
    return bdata(markerResults[region].tag);
}

int
perfmon_getEventsOfRegion(int region)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= markerRegions)
    {
        return -EINVAL;
    }
    if (markerResults == NULL)
    {
        return 0;
    }
    return markerResults[region].eventCount;
}

int
perfmon_getMetricsOfRegion(int region)
{
    if (region < 0 || region >= markerRegions)
    {
        return -EINVAL;
    }
    if (markerResults == NULL)
    {
        return 0;
    }
    return perfmon_getNumberOfMetrics(markerResults[region].groupID);
}

int
perfmon_getThreadsOfRegion(int region)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= markerRegions)
    {
        return -EINVAL;
    }
    if (markerResults == NULL)
    {
        return 0;
    }
    return markerResults[region].threadCount;
}

int
perfmon_getCpulistOfRegion(int region, int count, int* cpulist)
{
    int i;
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= markerRegions)
    {
        return -EINVAL;
    }
    if (markerResults == NULL)
    {
        return 0;
    }
    if (cpulist == NULL)
    {
        return -EINVAL;
    }
    for (i=0; i< MIN(count, markerResults[region].threadCount); i++)
    {
        cpulist[i] = markerResults[region].cpulist[i];
    }
    return MIN(count, markerResults[region].threadCount);
}

double
perfmon_getTimeOfRegion(int region, int thread)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= markerRegions)
    {
        return -EINVAL;
    }
    if (thread < 0 || thread >= groupSet->numberOfThreads)
    {
        return -EINVAL;
    }
    if (markerResults == NULL || markerResults[region].time == NULL)
    {
        return 0.0;
    }
    return markerResults[region].time[thread];
}

int
perfmon_getCountOfRegion(int region, int thread)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= markerRegions)
    {
        return -EINVAL;
    }
    if (thread < 0 || thread >= groupSet->numberOfThreads)
    {
        return -EINVAL;
    }
    if (markerResults == NULL || markerResults[region].count == NULL)
    {
        return 0.0;
    }
    return markerResults[region].count[thread];
}

double
perfmon_getResultOfRegionThread(int region, int event, int thread)
{
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= markerRegions)
    {
        return -EINVAL;
    }
    if (markerResults == NULL)
    {
        return 0;
    }
    if (thread < 0 || thread >= markerResults[region].threadCount)
    {
        return -EINVAL;
    }
    if (event < 0 || event >= markerResults[region].eventCount)
    {
        return -EINVAL;
    }
    if (markerResults[region].counters[thread] == NULL)
    {
        return 0.0;
    }
    return markerResults[region].counters[thread][event];
}

double
perfmon_getMetricOfRegionThread(int region, int metricId, int threadId)
{
    int e = 0, err = 0;
    double result = 0.0;
    char split[2] = ":";
    bstring vars = bformat("");
    bstring varlist = bformat("");
    if (perfmon_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return NAN;
    }
    if (region < 0 || region >= markerRegions)
    {
        return NAN;
    }
    if (markerResults == NULL)
    {
        return NAN;
    }
    if (threadId < 0 || threadId >= markerResults[region].threadCount)
    {
        return NAN;
    }
    if (metricId < 0 || metricId >= groupSet->groups[markerResults[region].groupID].group.nmetrics)
    {
        return NAN;
    }

    char* f = groupSet->groups[markerResults[region].groupID].group.metricformulas[metricId];
    for (e=0;e<markerResults[region].eventCount;e++)
    {
        if (groupSet->groups[markerResults[region].groupID].events[e].type != NOTYPE)
        {
            char* ctr = strtok(groupSet->groups[markerResults[region].groupID].group.counters[e], split);
            double res = perfmon_getResultOfRegionThread(region, e, threadId);
            calc_add_dbl_var(ctr, res, vars, varlist);
        }
        else
        {
            char *ctr = strtok(groupSet->groups[markerResults[region].groupID].group.counters[e], split);
            if (strstr(f, ctr) != NULL)
            {
                bdestroy(vars);
                bdestroy(varlist);
                return NAN;
            }
        }
    }
    if (groupSet->groups[markerResults[region].groupID].group.lua_funcs)
    {
        if (calc_set_user_funcs(groupSet->groups[markerResults[region].groupID].group.lua_funcs))
        {
            free(groupSet->groups[markerResults[region].groupID].group.lua_funcs);
            groupSet->groups[markerResults[region].groupID].group.lua_funcs = NULL;
        }
    }
    calc_add_dbl_var("time", perfmon_getTimeOfRegion(region, threadId), vars, varlist);
    calc_add_dbl_var("inverseClock", 1.0/timer_getCycleClock(), vars, varlist);
    int cpu = 0, sock_cpu = 0;
    for (e=0; e<groupSet->numberOfThreads; e++)
    {
        if (groupSet->threads[e].thread_id == threadId)
        {
            cpu = groupSet->threads[e].processorId;
        }
    }
    
    sock_cpu = socket_lock[affinity_thread2socket_lookup[cpu]];
    if (cpu != sock_cpu)
    {
        for (e=0; e<groupSet->numberOfThreads; e++)
        {
            if (groupSet->threads[e].processorId == sock_cpu)
            {
                sock_cpu = groupSet->threads[e].thread_id;
            }
        }
        calc_add_int_var("SOCKET_CPU", sock_cpu, vars, varlist);
        for (e=0;e<markerResults[region].eventCount;e++)
        {
            if (perfmon_isUncoreCounter(groupSet->groups[markerResults[region].groupID].group.counters[e]) &&
                !perfmon_isUncoreCounter(groupSet->groups[markerResults[region].groupID].group.metricformulas[metricId]))
            {
                if (groupSet->groups[markerResults[region].groupID].events[e].type != NOTYPE)
                {
                    char *ctr = strtok(groupSet->groups[markerResults[region].groupID].group.counters[e], split);
                    double res = perfmon_getResultOfRegionThread(region, e, sock_cpu);
                    err = calc_add_dbl_var(ctr, res, vars, varlist);
                    if (err < 0)
                    {
                        DEBUG_PRINT(DEBUGLEV_DEVELOP, Cannot add socket result of counter %s for thread %d, groupSet->groups[markerResults[region].groupID].group.counters[e], threadId);
                    }
                }
            }
        }
    }
    else
    {
        calc_add_int_var("SOCKET_CPU", cpu, vars, varlist);
    }
    
    err = calc_metric(cpu, f, vars, varlist, &result);
    bdestroy(vars);
    bdestroy(varlist);
    if (err < 0)
    {
        ERROR_PRINT(Cannot calculate formula %s, f);
        return NAN;
    }
    return result;
}

int
perfmon_readMarkerFile(const char* filename)
{
    FILE* fp = NULL;
    int i = 0;
    int ret = 0;
    char buf[2048];
    buf[0] = '\0';
    char *ptr = NULL;
    int nr_regions = 0;
    int cpus = 0, groups = 0, regions = 0;

    if (filename == NULL)
    {
        return -EINVAL;
    }
    if (access(filename, R_OK))
    {
        return -EINVAL;
    }
    fp = fopen(filename, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Error opening file %s\n", filename);
    }
    ptr = fgets(buf, sizeof(buf), fp);
    ret = sscanf(buf, "%d %d %d", &cpus, &regions, &groups);
    if (ret != 3)
    {
        fprintf(stderr, "Marker file missformatted.\n");
        return -EINVAL;
    }
    //markerResults = malloc(regions * sizeof(LikwidResults));
    markerResults = realloc(markerResults, regions * sizeof(LikwidResults));
    if (markerResults == NULL)
    {
        fprintf(stderr, "Failed to allocate %lu bytes for the marker results storage\n", regions * sizeof(LikwidResults));
        return -ENOMEM;
    }
    int* regionCPUs = (int*)malloc(regions * sizeof(int));
    if (regionCPUs == NULL)
    {
        fprintf(stderr, "Failed to allocate %lu bytes for temporal cpu count storage\n", regions * sizeof(int));
        return -ENOMEM;
    }
    markerRegions = regions;
    groupSet->numberOfThreads = cpus;
    for ( uint32_t i=0; i < regions; i++ )
    {
        regionCPUs[i] = 0;
        markerResults[i].threadCount = cpus;
        markerResults[i].time = (double*) malloc(cpus * sizeof(double));
        if (!markerResults[i].time)
        {
            fprintf(stderr, "Failed to allocate %lu bytes for the time storage\n", cpus * sizeof(double));
            break;
        }
        markerResults[i].count = (uint32_t*) malloc(cpus * sizeof(uint32_t));
        if (!markerResults[i].count)
        {
            fprintf(stderr, "Failed to allocate %lu bytes for the count storage\n", cpus * sizeof(uint32_t));
            break;
        }
        markerResults[i].cpulist = (int*) malloc(cpus * sizeof(int));
        if (!markerResults[i].count)
        {
            fprintf(stderr, "Failed to allocate %lu bytes for the cpulist storage\n", cpus * sizeof(int));
            break;
        }
        markerResults[i].counters = (double**) malloc(cpus * sizeof(double*));
        if (!markerResults[i].counters)
        {
            fprintf(stderr, "Failed to allocate %lu bytes for the counter result storage\n", cpus * sizeof(double*));
            break;
        }
    }
    while (fgets(buf, sizeof(buf), fp))
    {
        if (strchr(buf,':'))
        {
            int regionid = 0, groupid = -1;
            char regiontag[100];
            char* ptr = NULL;
            char* colonptr = NULL;
            regiontag[0] = '\0';
            ret = sscanf(buf, "%d:%s", &regionid, regiontag);

            ptr = strrchr(regiontag,'-');
            colonptr = strchr(buf,':');
            if (ret != 2 || ptr == NULL || colonptr == NULL)
            {
                fprintf(stderr, "Line %s not a valid region description\n", buf);
                continue;
            }
            groupid = atoi(ptr+1);
            snprintf(regiontag, strlen(regiontag)-strlen(ptr)+1, "%s", &(buf[colonptr-buf+1]));
            markerResults[regionid].groupID = groupid;
            markerResults[regionid].tag = bfromcstr(regiontag);
            nr_regions++;
        }
        else
        {
            int regionid = 0, groupid = 0, cpu = 0, count = 0, nevents = 0;
            int cpuidx = 0, eventidx = 0;
            double time = 0;
            char remain[1024];
            remain[0] = '\0';
            ret = sscanf(buf, "%d %d %d %d %lf %d %[^\t\n]", &regionid, &groupid, &cpu, &count, &time, &nevents, remain);
            if (ret != 7)
            {
                fprintf(stderr, "Line %s not a valid region values line\n", buf);
                continue;
            }
            if (cpu >= 0)
            {
                cpuidx = regionCPUs[regionid];
                markerResults[regionid].cpulist[cpuidx] = cpu;
                markerResults[regionid].eventCount = nevents;
                markerResults[regionid].time[cpuidx] = time;
                markerResults[regionid].count[cpuidx] = count;
                markerResults[regionid].counters[cpuidx] = malloc(nevents * sizeof(double));

                eventidx = 0;
                ptr = strtok(remain, " ");
                while (ptr != NULL && eventidx < nevents)
                {
                    sscanf(ptr, "%lf", &(markerResults[regionid].counters[cpuidx][eventidx]));
                    ptr = strtok(NULL, " ");
                    eventidx++;
                }
                regionCPUs[regionid]++;
            }
        }
    }
    for ( uint32_t i=0; i < regions; i++ )
    {
        markerResults[i].threadCount = regionCPUs[i];
    }
    free(regionCPUs);
    fclose(fp);
    return nr_regions;
}

void
perfmon_destroyMarkerResults()
{
    int i = 0, j = 0;
    if (markerResults != NULL)
    {
        for (i = 0; i < markerRegions; i++)
        {
            free(markerResults[i].time);
            free(markerResults[i].count);
            free(markerResults[i].cpulist);
            for (j = 0; j < markerResults[i].threadCount; j++)
            {
                free(markerResults[i].counters[j]);
            }
            free(markerResults[i].counters);
            bdestroy(markerResults[i].tag);
        }
        free(markerResults);
    }
}

