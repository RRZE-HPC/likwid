/*
 * =======================================================================================
 *
 *      Filename:  perfmon.c
 *
 *      Description:  Implementation of perfmon Module.
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include <sys/types.h>

#include <types.h>
#include <bitUtil.h>
#include <bstrlib.h>
#include <error.h>
#include <timer.h>
#include <accessClient.h>
#include <msr.h>
#include <pci.h>
#include <lock.h>
#include <cpuid.h>
#include <power.h>
#include <thermal.h>
#include <perfmon.h>
#include <registers.h>


/* #####   EXPORTED VARIABLES   ########################################### */

int perfmon_verbose = 0;

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static PerfmonEvent* eventHash;
static PerfmonCounterMap* counter_map;
static TimerData timeData;
static double rdtscTime;
static int perfmon_numCounters;
static int perfmon_numArchEvents;
static int perfmon_numThreads;
static PerfmonThread* perfmon_threadData;
static int socket_fd = -1;
static int socket_lock[MAX_NUM_NODES];

/* #####   PROTOTYPES  -  LOCAL TO THIS SOURCE FILE   ##################### */

static void initThread(int , int );

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

/* #####   Architecture Specific Code  #################################### */
//#include <perfmon_pm.h>
//#include <perfmon_atom.h>
//#include <perfmon_core2.h>
//#include <perfmon_nehalem.h>
//#include <perfmon_westmere.h>
//#include <perfmon_westmereEX.h>
//#include <perfmon_nehalemEX.h>
//#include <perfmon_sandybridge.h>
//#include <perfmon_ivybridge.h>
//#include <perfmon_haswell.h>
//#include <perfmon_phi.h>
//#include <perfmon_k8.h>
//#include <perfmon_k10.h>
//#include <perfmon_interlagos.h>
//#include <perfmon_kabini.h>

/* #####  EXPORTED  FUNCTION POINTERS   ################################### */
void (*perfmon_startCountersThread) (int thread_id);
void (*perfmon_stopCountersThread) (int thread_id);
void (*perfmon_readCountersThread) (int thread_id);
void (*perfmon_setupCounterThread) (int thread_id,
        PerfmonEvent* event, PerfmonCounterIndex index);

/* #####   FUNCTION POINTERS  -  LOCAL TO THIS SOURCE FILE ################ */

static void (*initThreadArch) (PerfmonThread *thread);

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static void
initThread(int thread_id, int cpu_id)
{
    for (int i=0; i<NUM_PMC; i++)
    {
        perfmon_threadData[thread_id].counters[i].init = FALSE;
    }

    perfmon_threadData[thread_id].processorId = cpu_id;
    initThreadArch(&perfmon_threadData[thread_id]);
}


static int
getIndex (bstring reg, PerfmonCounterIndex* index)
{
    for (int i=0; i< perfmon_numCounters; i++)
    {
        if (biseqcstr(reg, counter_map[i].key))
        {
            *index = counter_map[i].index;
            return TRUE;
        }
    }

    return FALSE;
}

static int
getEvent(bstring event_str, PerfmonEvent* event)
{
    for (int i=0; i< perfmon_numArchEvents; i++)
    {
        if (biseqcstr(event_str, eventHash[i].name))
        {
            *event = eventHash[i];
            return TRUE;
        }
    }

    return FALSE;
}

static int
checkCounter(bstring counterName, const char* limit)
{
    int i;
    struct bstrList* tokens;
    int value = FALSE;
    bstring limitString = bfromcstr(limit);

    tokens = bstrListCreate();
    tokens = bsplit(limitString,'|');

    for(i=0; i<tokens->qty; i++)
    {
        if(bstrncmp(counterName, tokens->entry[i], blength(tokens->entry[i])))
        {
            value = FALSE;
        }
        else
        {
            value = TRUE;
            break;
        }
    }

    bdestroy(limitString);
    bstrListDestroy(tokens);
    return value;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
perfmon_setupEventSet(char* eventCString, PerfmonEventSet* eventSet)
{
    struct bstrList* tokens;
    struct bstrList* subtokens;
    bstring q = bfromcstr(eventCString);

    tokens = bsplit(q,',');
    eventSet->numberOfEvents = tokens->qty;
    eventSet->events = (PerfmonEventSetEntry*)
        malloc(eventSet->numberOfEvents * sizeof(PerfmonEventSetEntry));

    for (int i=0;i<tokens->qty;i++)
    {
        subtokens = bsplit(tokens->entry[i],':');

        if ( subtokens->qty != 2 )
        {
            ERROR_PLAIN_PRINT(Error in parsing event string);
        }
        else
        {
            /* get register index */
#if 0
            if (!getIndex(bstrcpy(subtokens->entry[1], &eventSet->events[i].index)))
            {
                ERROR_PRINT(Counter register %s not supported,bdata(
                            eventSetConfig->events[i].counterName));
            }

            /* setup event */
            if (!getEvent(subtokens->entry[0], &eventSet->events[i].event))
            {
                ERROR_PRINT(Event %s not found for current architecture,
                        bdata(eventSetConfig->events[i].eventName));
            }

            /* is counter allowed for event */
            if (!checkCounter(subtokens->entry[1], eventSet->events[i].event.limit))
            {
                ERROR_PRINT(Register not allowed  for event  %s,
                        bdata(eventSetConfig->events[i].eventName));
            }
#endif

        }

        bstrListDestroy(subtokens);
    }

    bstrListDestroy(tokens);
}

int
perfmon_setupCounters(PerfmonEventSet* eventSet)
{
#if 0
    for (int j=0; j<eventSet.numberOfEvents; j++)
    {
        for (int i=0; i<perfmon_numThreads; i++)
        {
            perfmon_setupCounterThread(i,
                    &eventSet.events[j].event,
                    eventSet.events[j].index);
        }
    }
#endif

}

void
perfmon_startCounters(void)
{
    for (int i=0; i<perfmon_numThreads; i++)
    {
        perfmon_startCountersThread(i);
    }

    timer_start(&timeData);
}

void
perfmon_stopCounters(void)
{
    timer_stop(&timeData);

    for (int i=0; i<perfmon_numThreads; i++)
    {
        perfmon_stopCountersThread(i);
    }

    rdtscTime = timer_print(&timeData);
}

void
perfmon_readCounters(void)
{
    for (int i=0; i<perfmon_numThreads; i++)
    {
        perfmon_readCountersThread(i);
    }
}

void
perfmon_init(int numThreads_local, int threads[])
{
    perfmon_numThreads = numThreads_local;
    perfmon_threadData = (PerfmonThread*)
        malloc(perfmon_numThreads * sizeof(PerfmonThread));

    for(int i=0; i<MAX_NUM_NODES; i++) socket_lock[i] = LOCK_INIT;

    if (accessClient_mode != DAEMON_AM_DIRECT)
    {
        accessClient_init(&socket_fd);
    }

    msr_init(socket_fd);

#if 0
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
                    perfmon_numCounters = perfmon_numCounters_pm;
                    initThreadArch = perfmon_init_pm;
                    printDerivedMetrics = perfmon_printDerivedMetrics_pm;
                    perfmon_startCountersThread = perfmon_startCountersThread_pm;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_pm;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_pm;
                    break;

                case ATOM_45:

                case ATOM_32:

                case ATOM_22:

                case ATOM:

                    eventHash = atom_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsAtom;
                    counter_map = core2_counter_map;
                    perfmon_numCounters = perfmon_numCountersCore2;
                    initThreadArch = perfmon_init_core2;
                    printDerivedMetrics = perfmon_printDerivedMetricsAtom;
                    perfmon_startCountersThread = perfmon_startCountersThread_core2;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_core2;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_core2;
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
                    initThreadArch = perfmon_init_core2;
                    printDerivedMetrics = perfmon_printDerivedMetricsCore2;
                    logDerivedMetrics = perfmon_logDerivedMetricsCore2;
                    perfmon_startCountersThread = perfmon_startCountersThread_core2;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_core2;
                    perfmon_readCountersThread = perfmon_readCountersThread_core2;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_core2;
                    break;

                case NEHALEM_EX:

                    eventHash = nehalemEX_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsNehalemEX;
                    counter_map = westmereEX_counter_map;
                    perfmon_numCounters = perfmon_numCountersWestmereEX;
                    initThreadArch = perfmon_init_westmereEX;
                    printDerivedMetrics = perfmon_printDerivedMetricsNehalemEX;
                    logDerivedMetrics = perfmon_logDerivedMetricsNehalemEX;
                    perfmon_startCountersThread = perfmon_startCountersThread_westmereEX;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_westmereEX;
                    perfmon_readCountersThread = perfmon_readCountersThread_westmereEX;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_nehalemEX;
                    break;

                case WESTMERE_EX:

                    eventHash = westmereEX_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsWestmereEX;
                    counter_map = westmereEX_counter_map;
                    perfmon_numCounters = perfmon_numCountersWestmereEX;
                    initThreadArch = perfmon_init_westmereEX;
                    printDerivedMetrics = perfmon_printDerivedMetricsWestmereEX;
                    logDerivedMetrics = perfmon_logDerivedMetricsWestmereEX;
                    perfmon_startCountersThread = perfmon_startCountersThread_westmereEX;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_westmereEX;
                    perfmon_readCountersThread = perfmon_readCountersThread_westmereEX;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_westmereEX;
                    break;

                case NEHALEM_BLOOMFIELD:

                case NEHALEM_LYNNFIELD:

                    thermal_init(0);

                    eventHash = nehalem_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsNehalem;
                    counter_map = nehalem_counter_map;
                    perfmon_numCounters = perfmon_numCountersNehalem;
                    initThreadArch = perfmon_init_nehalem;
                    printDerivedMetrics = perfmon_printDerivedMetricsNehalem;
                    logDerivedMetrics = perfmon_logDerivedMetricsNehalem;
                    perfmon_startCountersThread = perfmon_startCountersThread_nehalem;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_nehalem;
                    perfmon_readCountersThread = perfmon_readCountersThread_nehalem;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_nehalem;
                    break;

                case NEHALEM_WESTMERE_M:

                case NEHALEM_WESTMERE:

                    thermal_init(0);

                    eventHash = westmere_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsWestmere;
                    counter_map = nehalem_counter_map;
                    perfmon_numCounters = perfmon_numCountersNehalem;
                    initThreadArch = perfmon_init_nehalem;
                    printDerivedMetrics = perfmon_printDerivedMetricsWestmere;
                    logDerivedMetrics = perfmon_logDerivedMetricsWestmere;
                    perfmon_startCountersThread = perfmon_startCountersThread_nehalem;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_nehalem;
                    perfmon_readCountersThread = perfmon_readCountersThread_nehalem;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_nehalem;
                    break;

                case IVYBRIDGE:

                case IVYBRIDGE_EP:

                    power_init(0); /* FIXME Static coreId is dangerous */
                    thermal_init(0);
                    pci_init(socket_fd);

                    eventHash = ivybridge_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsIvybridge;
                    counter_map = ivybridge_counter_map;
                    perfmon_numCounters = perfmon_numCountersIvybridge;
                    initThreadArch = perfmon_init_ivybridge;
                    printDerivedMetrics = perfmon_printDerivedMetricsIvybridge;
                    logDerivedMetrics = perfmon_logDerivedMetricsIvybridge;
                    perfmon_startCountersThread = perfmon_startCountersThread_ivybridge;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_ivybridge;
                    perfmon_readCountersThread = perfmon_readCountersThread_ivybridge;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_ivybridge;
                    break;

                case HASWELL:

                case HASWELL_EX:

                case HASWELL_M1:

                case HASWELL_M2:

                    power_init(0); /* FIXME Static coreId is dangerous */
                    thermal_init(0);

                    eventHash = haswell_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsHaswell;
                    counter_map = haswell_counter_map;
                    perfmon_numCounters = perfmon_numCountersHaswell;
                    initThreadArch = perfmon_init_haswell;
                    printDerivedMetrics = perfmon_printDerivedMetricsHaswell;
                    logDerivedMetrics = perfmon_logDerivedMetricsHaswell;
                    perfmon_startCountersThread = perfmon_startCountersThread_haswell;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_haswell;
                    perfmon_readCountersThread = perfmon_readCountersThread_haswell;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_haswell;
                    break;

                case SANDYBRIDGE:

                case SANDYBRIDGE_EP:

                    power_init(0); /* FIXME Static coreId is dangerous */
                    thermal_init(0);
                    pci_init(socket_fd);

                    eventHash = sandybridge_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsSandybridge;
                    counter_map = sandybridge_counter_map;
                    perfmon_numCounters = perfmon_numCountersSandybridge;
                    initThreadArch = perfmon_init_sandybridge;
                    printDerivedMetrics = perfmon_printDerivedMetricsSandybridge;
                    logDerivedMetrics = perfmon_logDerivedMetricsSandybridge;
                    perfmon_startCountersThread = perfmon_startCountersThread_sandybridge;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_sandybridge;
                    perfmon_readCountersThread = perfmon_readCountersThread_sandybridge;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_sandybridge;
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
                    perfmon_numCounters = perfmon_numCountersPhi;
                    initThreadArch = perfmon_init_phi;
                    printDerivedMetrics = perfmon_printDerivedMetricsPhi;
                    logDerivedMetrics = perfmon_logDerivedMetricsPhi;
                    perfmon_startCountersThread = perfmon_startCountersThread_phi;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_phi;
                    perfmon_readCountersThread = perfmon_readCountersThread_phi;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_phi;
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
            perfmon_numCounters = perfmon_numCountersK10;
            initThreadArch = perfmon_init_k10;
            printDerivedMetrics = perfmon_printDerivedMetricsK8;
            logDerivedMetrics = perfmon_logDerivedMetricsK8;
            perfmon_startCountersThread = perfmon_startCountersThread_k10;
            perfmon_stopCountersThread = perfmon_stopCountersThread_k10;
            perfmon_readCountersThread = perfmon_readCountersThread_k10;
            perfmon_setupCounterThread = perfmon_setupCounterThread_k10;
            break;

        case K10_FAMILY:
            eventHash = k10_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsK10;
            counter_map = k10_counter_map;
            perfmon_numCounters = perfmon_numCountersK10;
            initThreadArch = perfmon_init_k10;
            printDerivedMetrics = perfmon_printDerivedMetricsK10;
            logDerivedMetrics = perfmon_logDerivedMetricsK10;
            perfmon_startCountersThread = perfmon_startCountersThread_k10;
            perfmon_stopCountersThread = perfmon_stopCountersThread_k10;
            perfmon_readCountersThread = perfmon_readCountersThread_k10;
            perfmon_setupCounterThread = perfmon_setupCounterThread_k10;
            break;

        case K15_FAMILY:
            eventHash = interlagos_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsInterlagos;
            counter_map = interlagos_counter_map;
            perfmon_numCounters = perfmon_numCountersInterlagos;
            initThreadArch = perfmon_init_interlagos;
            printDerivedMetrics = perfmon_printDerivedMetricsInterlagos;
            logDerivedMetrics = perfmon_logDerivedMetricsInterlagos;
            perfmon_startCountersThread = perfmon_startCountersThread_interlagos;
            perfmon_stopCountersThread = perfmon_stopCountersThread_interlagos;
            perfmon_readCountersThread = perfmon_readCountersThread_interlagos;
            perfmon_setupCounterThread = perfmon_setupCounterThread_interlagos;
            break;

        case K16_FAMILY:
            eventHash = kabini_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsKabini;
            counter_map = kabini_counter_map;
            perfmon_numCounters = perfmon_numCountersKabini;
            initThreadArch = perfmon_init_kabini;
            printDerivedMetrics = perfmon_printDerivedMetricsKabini;
            logDerivedMetrics = perfmon_logDerivedMetricsKabini;
            perfmon_startCountersThread = perfmon_startCountersThread_kabini;
            perfmon_stopCountersThread = perfmon_stopCountersThread_kabini;
            perfmon_readCountersThread = perfmon_readCountersThread_kabini;
            perfmon_setupCounterThread = perfmon_setupCounterThread_kabini;
           break;

        default:
            ERROR_PLAIN_PRINT(Unsupported Processor);
            break;
    }

#endif

    for (int i=0; i<perfmon_numThreads; i++)
    {
        initThread(i,threads[i]);
    }
}

void
perfmon_finalize(void)
{
    free(perfmon_threadData);
    msr_finalize();
    pci_finalize();
    accessClient_finalize(socket_fd);
}

