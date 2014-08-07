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
#include <topology.h>
#include <power.h>
#include <thermal.h>
#include <perfmon.h>
#include <registers.h>


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
#include <perfmon_k8.h>
#include <perfmon_k10.h>
#include <perfmon_interlagos.h>
#include <perfmon_kabini.h>
#include <perfmon_silvermont.h>


PerfmonEvent* eventHash;
RegisterMap* counter_map;
int perfmon_numCounters;
int perfmon_numArchEvents;

int socket_fd = -1;

PerfmonGroupSet* groupSet = NULL;

int (*perfmon_startCountersThread) (int thread_id, PerfmonEventSet* eventSet);
int (*perfmon_stopCountersThread) (int thread_id, PerfmonEventSet* eventSet);
int (*perfmon_readCountersThread) (int thread_id, PerfmonEventSet* eventSet);
int (*perfmon_setupCountersThread) (int thread_id, PerfmonEventSet* eventSet);

int (*initThreadArch) (int cpu_id);

int
perfmon_initThread(/*PerfmonGroupSet* groupSet, int groupId,*/ int thread_id, int cpu_id)
{
    int i, j;
    
    for (i=0;i<groupSet->numberOfActiveGroups;i++)
    {
        for (i=0;i<groupSet->groups[i].numberOfEvents;i++)
        {
            groupSet->groups[i].events[j].threadCounter[thread_id].init = FALSE;
        }
    }
    return initThreadArch(cpu_id);
}

static int
getIndexAndType (bstring reg, RegisterIndex* index, RegisterType* type)
{
    for (int i=0; i< perfmon_numCounters; i++)
    {
        if (biseqcstr(reg, counter_map[i].key))
        {
            *index = counter_map[i].index;
            *type = counter_map[i].type;
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

void
perfmon_printCounters(FILE* OUTSTREAM)
{
    fprintf(OUTSTREAM,"This architecture has %d counters.\n", perfmon_numCounters);
    fprintf(OUTSTREAM,"Counters names:  ");

    for (int i=0; i<perfmon_numCounters; i++)
    {
        fprintf(OUTSTREAM,"%s",counter_map[i].key);
        if (i != perfmon_numCounters-1)
        {
            fprintf(OUTSTREAM,"\t");
        }
    }
    fprintf(OUTSTREAM,".\n");
}

void
perfmon_printEvents(FILE* OUTSTREAM)
{
    int i;

    fprintf(OUTSTREAM,"This architecture has %d events.\n", perfmon_numArchEvents);
    fprintf(OUTSTREAM,"Event tags (tag, id, umask, counters):\n");

    for (i=0; i<perfmon_numArchEvents; i++)
    {
        fprintf(OUTSTREAM,"%s, 0x%X, 0x%X, %s \n",
                eventHash[i].name,
                eventHash[i].eventId,
                eventHash[i].umask,
                eventHash[i].limit);
    }
}

void
perfmon_init_maps(void)
{
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
                    break;

                case ATOM_45:

                case ATOM_32:

                case ATOM_22:

                case ATOM:

                    eventHash = atom_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsAtom;
                    counter_map = core2_counter_map;
                    perfmon_numCounters = perfmon_numCountersCore2;
                    break;

                case ATOM_SILVERMONT:
                    eventHash = silvermont_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsSilvermont;
                    counter_map = silvermont_counter_map;
                    perfmon_numCounters = perfmon_numCountersSilvermont;
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
                    break;

                case NEHALEM_EX:

                    eventHash = nehalemEX_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsNehalemEX;
                    counter_map = westmereEX_counter_map;
                    perfmon_numCounters = perfmon_numCountersWestmereEX;

                    break;

                case WESTMERE_EX:

                    eventHash = westmereEX_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsWestmereEX;
                    counter_map = westmereEX_counter_map;
                    perfmon_numCounters = perfmon_numCountersWestmereEX;
                    break;

                case NEHALEM_BLOOMFIELD:

                case NEHALEM_LYNNFIELD:

                    eventHash = nehalem_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsNehalem;
                    counter_map = nehalem_counter_map;
                    perfmon_numCounters = perfmon_numCountersNehalem;
                    break;

                case NEHALEM_WESTMERE_M:

                case NEHALEM_WESTMERE:
                    eventHash = westmere_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsWestmere;
                    counter_map = nehalem_counter_map;
                    perfmon_numCounters = perfmon_numCountersNehalem;
                    break;

                case IVYBRIDGE:

                case IVYBRIDGE_EP:
                    eventHash = ivybridge_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsIvybridge;
                    counter_map = ivybridge_counter_map;
                    perfmon_numCounters = perfmon_numCountersIvybridge;
                    break;

                case HASWELL:

                case HASWELL_EX:

                case HASWELL_M1:

                case HASWELL_M2:
                    eventHash = haswell_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsHaswell;
                    counter_map = haswell_counter_map;
                    perfmon_numCounters = perfmon_numCountersHaswell;
                    break;

                case SANDYBRIDGE:

                case SANDYBRIDGE_EP:
                    eventHash = sandybridge_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsSandybridge;
                    counter_map = sandybridge_counter_map;
                    perfmon_numCounters = perfmon_numCountersSandybridge;
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
            break;

        case K10_FAMILY:
            eventHash = k10_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsK10;
            counter_map = k10_counter_map;
            perfmon_numCounters = perfmon_numCountersK10;
            break;

        case K15_FAMILY:
            eventHash = interlagos_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsInterlagos;
            counter_map = interlagos_counter_map;
            perfmon_numCounters = perfmon_numCountersInterlagos;
            break;

        case K16_FAMILY:
            eventHash = kabini_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsKabini;
            counter_map = kabini_counter_map;
            perfmon_numCounters = perfmon_numCountersKabini;
           break;

        default:
            ERROR_PLAIN_PRINT(Unsupported Processor);
            break;
    }
    return;
}


int
perfmon_init(int nrThreads, int threadsToCpu[])
{
    int i;
    int ret;
    int initialize_power = FALSE;
    int initialize_thermal = FALSE;

    if (nrThreads <= 0)
    {
        ERROR_PRINT("Number of threads must be greater than 0, given %d",nrThreads);
        return -EINVAL;
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
        ERROR_PLAIN_PRINT("Cannot allocate group descriptor");
        return -ENOMEM;
    }
    
    groupSet->threads = (PerfmonThread*) malloc(nrThreads * sizeof(PerfmonThread));
    if (groupSet->threads == NULL)
    {
        ERROR_PLAIN_PRINT("Cannot allocate set of threads");
        free(groupSet);
        return -ENOMEM;
    }
    
    groupSet->groups = (PerfmonEventSet*) malloc(sizeof(PerfmonEventSet));
    if (groupSet->groups == NULL)
    {
        ERROR_PLAIN_PRINT("Cannot allocate set of groups");
        free(groupSet->threads);
        free(groupSet);
        return -ENOMEM;
    }
    
    groupSet->numberOfGroups = 1;
    groupSet->numberOfActiveGroups = 0;
    groupSet->numberOfThreads = nrThreads;
    
    
    /* Only one group exists by now */
    groupSet->groups[0].rdtscTime = 0;
    groupSet->groups[0].numberOfEvents = 0;
    
    for(i=0; i<MAX_NUM_NODES; i++) socket_lock[i] = LOCK_INIT;
    
    if (accessClient_mode != DAEMON_AM_DIRECT)
    {
        accessClient_init(&socket_fd);
    }
    
    ret = msr_init(socket_fd);

    if (ret)
    {
        ERROR_PLAIN_PRINT("Initialization of MSR device accesses failed");
        free(groupSet->groups);
        free(groupSet->threads);
        free(groupSet);
        return ret;
    }
    

    timer_init();
    
    perfmon_init_maps();

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
                    //perfmon_readCountersThread = perfmon_readCountersThread_pm;
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
                    break;

                case ATOM_SILVERMONT:
                    initialize_power = TRUE;
                    initialize_thermal = TRUE;
                    initThreadArch = perfmon_init_silvermont;
                    perfmon_startCountersThread = perfmon_startCountersThread_silvermont;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_silvermont;
                    perfmon_setupCountersThread = perfmon_setupCountersThread_silvermont;
                    perfmon_readCountersThread = perfmon_readCountersThread_silvermont;
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
                    break;

                case NEHALEM_EX:
                    initThreadArch = perfmon_init_westmereEX;
                    perfmon_startCountersThread = perfmon_startCountersThread_westmereEX;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_westmereEX;
                    perfmon_readCountersThread = perfmon_readCountersThread_westmereEX;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_nehalemEX;
                    break;

                case WESTMERE_EX:
                    initThreadArch = perfmon_init_westmereEX;
                    perfmon_startCountersThread = perfmon_startCountersThread_westmereEX;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_westmereEX;
                    perfmon_readCountersThread = perfmon_readCountersThread_westmereEX;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_westmereEX;
                    break;

                case NEHALEM_BLOOMFIELD:

                case NEHALEM_LYNNFIELD:
                    initialize_thermal = TRUE;
                    initThreadArch = perfmon_init_nehalem;
                    perfmon_startCountersThread = perfmon_startCountersThread_nehalem;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_nehalem;
                    perfmon_readCountersThread = perfmon_readCountersThread_nehalem;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_nehalem;
                    break;

                case NEHALEM_WESTMERE_M:

                case NEHALEM_WESTMERE:
                    initialize_thermal = TRUE;
                    initThreadArch = perfmon_init_nehalem;
                    perfmon_startCountersThread = perfmon_startCountersThread_nehalem;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_nehalem;
                    perfmon_readCountersThread = perfmon_readCountersThread_nehalem;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_nehalem;
                    break;

                case IVYBRIDGE:

                case IVYBRIDGE_EP:

                    initialize_power = TRUE;
                    initialize_thermal = TRUE;
                    pci_init(socket_fd);
                    initThreadArch = perfmon_init_ivybridge;
                    perfmon_startCountersThread = perfmon_startCountersThread_ivybridge;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_ivybridge;
                    perfmon_readCountersThread = perfmon_readCountersThread_ivybridge;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_ivybridge;
                    break;

                case HASWELL:

                case HASWELL_EX:

                case HASWELL_M1:

                case HASWELL_M2:
                    initialize_power = TRUE;
                    initialize_thermal = TRUE;
                    initThreadArch = perfmon_init_haswell;
                    perfmon_startCountersThread = perfmon_startCountersThread_haswell;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_haswell;
                    perfmon_readCountersThread = perfmon_readCountersThread_haswell;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_haswell;
                    break;

                case SANDYBRIDGE:

                case SANDYBRIDGE_EP:
                    initialize_power = TRUE;
                    initialize_thermal = TRUE;
                    pci_init(socket_fd);
                    initThreadArch = perfmon_init_sandybridge;
                    perfmon_startCountersThread = perfmon_startCountersThread_sandybridge;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_sandybridge;
                    perfmon_readCountersThread = perfmon_readCountersThread_sandybridge;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_sandybridge;
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
            break;

        case K10_FAMILY:
            initThreadArch = perfmon_init_k10;
            perfmon_startCountersThread = perfmon_startCountersThread_k10;
            perfmon_stopCountersThread = perfmon_stopCountersThread_k10;
            perfmon_readCountersThread = perfmon_readCountersThread_k10;
            perfmon_setupCountersThread = perfmon_setupCounterThread_k10;
            break;

        case K15_FAMILY:
            initThreadArch = perfmon_init_interlagos;
            perfmon_startCountersThread = perfmon_startCountersThread_interlagos;
            perfmon_stopCountersThread = perfmon_stopCountersThread_interlagos;
            perfmon_readCountersThread = perfmon_readCountersThread_interlagos;
            perfmon_setupCountersThread = perfmon_setupCounterThread_interlagos;
            break;

        case K16_FAMILY:
            initThreadArch = perfmon_init_kabini;
            perfmon_startCountersThread = perfmon_startCountersThread_kabini;
            perfmon_stopCountersThread = perfmon_stopCountersThread_kabini;
            perfmon_readCountersThread = perfmon_readCountersThread_kabini;
            perfmon_setupCountersThread = perfmon_setupCounterThread_kabini;
           break;

        default:
            ERROR_PLAIN_PRINT(Unsupported Processor);
            break;
    }

    /* Store thread information and reset counters for processor*/
    /* If the arch supports it, initialize power and thermal measurements */
    for(i=0;i<nrThreads;i++)
    {
        if (initialize_power == TRUE)
        {
            power_init(threadsToCpu[i]);
        }
        if (initialize_thermal == TRUE)
        {
            thermal_init(threadsToCpu[i]);
        }
        groupSet->threads[i].thread_id = i;
        groupSet->threads[i].processorId = threadsToCpu[i];
        perfmon_initThread(i, threadsToCpu[i]);
    }
    
    return 0;
}

void 
perfmon_finalize(void)
{
    int group, event;
    
    for(group=0;group < groupSet->numberOfGroups; group++)
    {
        for (event=0;event < groupSet->groups[group].numberOfEvents; event++)
        {
            free(groupSet->groups[group].events[event].threadCounter);
        }
        free(groupSet->groups[group].events);
    }
    
    free(groupSet->threads);
    free(groupSet);
    msr_finalize();
    pci_finalize();
    accessClient_finalize(socket_fd);
    return;
}

int 
perfmon_addEventSet(char* eventCString)
{
    int i, j;
    int groupIndex;
    bstring eventBString;
    struct bstrList* tokens;
    struct bstrList* subtokens;
    PerfmonEventSet* eventSet;
    PerfmonEventSetEntry* event;
    RegisterType reg_type;
    
    if (eventCString == NULL)
    {
        ERROR_PLAIN_PRINT(Event string is empty. Trying environment variable LIKWID_EVENTS);
        eventCString = getenv("LIKWID_EVENTS");
        if (eventCString == NULL)
        {
            ERROR_PLAIN_PRINT(Event string from environment variable is empty);
            return -EINVAL;
        }
    }
    
    if (strchr(eventCString, '-') != NULL)
    {
        return -EINVAL;
    }
    if (strchr(eventCString, '.') != NULL)
    {
        return -EINVAL;
    }
    if (strchr(eventCString, ';') != NULL)
    {
        return -EINVAL;
    }
    
    if (groupSet->numberOfActiveGroups == groupSet->numberOfGroups)
    {
        groupSet->numberOfGroups++;
        groupSet->groups = (PerfmonEventSet*)realloc(groupSet->groups, groupSet->numberOfGroups*sizeof(PerfmonEventSet));
        groupSet->groups[groupSet->numberOfActiveGroups].rdtscTime = 0;
    }
    
    
    eventBString = bfromcstr(eventCString);
    tokens = bsplit(eventBString,',');
    bdestroy(eventBString);
    
    eventSet = &(groupSet->groups[groupSet->numberOfActiveGroups]);
    
    eventSet->events = (PerfmonEventSetEntry*) malloc(tokens->qty * sizeof(PerfmonEventSetEntry));
    
    if (eventSet->events == NULL)
    {
        fprintf(stderr,"Cannot allocate event list for group %d\n", groupSet->numberOfActiveGroups);
        bstrListDestroy(tokens);
        return -ENOMEM;
    }
    eventSet->numberOfEvents = 0;
    eventSet->measureFixed = 0;
    eventSet->measurePMC = 0;
    eventSet->measurePMCUncore = 0;
    eventSet->measurePCIUncore = 0;

    for(i=0;i<tokens->qty;i++)
    {
        event = &(groupSet->groups[groupSet->numberOfActiveGroups].events[i]);

        subtokens = bsplit(tokens->entry[i],':');
        if (subtokens->qty != 2)
        {
            fprintf(stderr,"Cannot parse event descriptor %s\n",tokens->entry[i]);
            bstrListDestroy(subtokens);
            continue;
        }
        else
        {
            if (!getIndexAndType(subtokens->entry[1], &event->index, &reg_type))
            {
                ERROR_PRINT(Counter register %s not supported,bdata(
                        subtokens->entry[1]));
                bstrListDestroy(subtokens);
                continue;
            }

            if (!getEvent(subtokens->entry[0], &event->event))
            {
                ERROR_PRINT(Event %s not found for current architecture,
                     bdata(subtokens->entry[0]));
                bstrListDestroy(subtokens);
                continue;
            }
           
            if (!checkCounter(subtokens->entry[1], event->event.limit))
            {
                ERROR_PRINT(Register %s not allowed for event %s,
                     bdata(subtokens->entry[1]),bdata(subtokens->entry[0]));
                bstrListDestroy(subtokens);
                continue;
            }

            eventSet->numberOfEvents++;
            switch (reg_type)
            {
                case FIXED:
                    eventSet->measureFixed = 1;
                    break;
                case PMC:
                    eventSet->measurePMC = 1;
                    break;
                case UNCORE:
                    eventSet->measurePMCUncore = 1;
                case MBOX0:
                case MBOX1:
                case MBOX2:
                case MBOX3:
                case MBOXFIX:
                case BBOX0:
                case BBOX1:
                case RBOX0:
                case RBOX1:
                case WBOX:
                case SBOX0:
                case SBOX1:
                case SBOX2:
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
                case CBOX10:
                case CBOX11:
                case PBOX:
                    eventSet->measurePCIUncore = 1;       
            }

            event->threadCounter = (PerfmonCounter*) malloc(
                groupSet->numberOfThreads * sizeof(PerfmonCounter));

            if (event->threadCounter == NULL)
            {
                ERROR_PRINT(Cannot allocate counter for all threads in group %d,groupSet->numberOfActiveGroups);
                bstrListDestroy(subtokens);
                continue;
            }
            for(j=0;j<groupSet->numberOfThreads;j++)
            {
                event->threadCounter[j].counterData = 0;
                event->threadCounter[j].startData = 0;
                event->threadCounter[j].overflows = 0;
            }
        }
        bstrListDestroy(subtokens);
    }
    bstrListDestroy(tokens);
    groupSet->numberOfActiveGroups++;
    return groupSet->numberOfActiveGroups-1;
}

int
perfmon_setupCounters(int groupId)
{
    int i;
    if (groupId >= groupSet->numberOfActiveGroups)
    {
        ERROR_PRINT(Group %d does not exist in groupSet, groupId);
        return -ENOENT;
    }
    
    for(i=0;i<groupSet->numberOfThreads;i++)
    {
        CHECK_AND_RETURN_ERROR(perfmon_setupCountersThread(i, &groupSet->groups[groupId]),
            Setup of counters failed);
    }
    groupSet->activeGroup = groupId;
    return 0;
}

int
__perfmon_startCounters(int groupId)
{
    int i = 0;
    int ret = 0;
    if ((groupId < 0) || (groupId >= groupSet->numberOfActiveGroups))
    {
        groupId = groupSet->activeGroup;
    }
    for(;i<groupSet->numberOfThreads;i++)
    {
        ret = perfmon_startCountersThread(i, &groupSet->groups[groupId]);
        if (ret)
        {
            return -i;
        }
    }
    timer_start(&groupSet->groups[groupId].timer);
    return 0;
}

int perfmon_startCounters(void)
{
    return __perfmon_startCounters(-1);
}

int perfmon_startGroupCounters(int groupId)
{
    return __perfmon_startGroupCounters(groupId);
}

int
__perfmon_stopCounters(int groupId)
{
    int i = 0;
    int ret = 0;

    if ((groupId < 0) || (groupId >= groupSet->numberOfActiveGroups))
    {
        groupId = groupSet->activeGroup;
    }

    timer_stop(&groupSet->groups[groupId].timer);

    for (; i<groupSet->numberOfThreads; i++)
    {
        ret = perfmon_stopCountersThread(i, &groupSet->groups[groupId]);
        if (ret)
        {
            return -i;
        }
    }

    groupSet->groups[groupSet->activeGroup].rdtscTime += 
                timer_print(&groupSet->groups[groupId].timer);
    return 0;
}

int perfmon_stopCounters(void)
{
    return __perfmon_stopCounters(-1);
}

int perfmon_stopGroupCounters(int groupId)
{
    return __perfmon_stopCounters(groupId);
}

int
__perfmon_readCounters(int groupId)
{
    int i = 0;
    int ret = 0;

    if ((groupId < 0) || (groupId >= groupSet->numberOfActiveGroups))
    {
        groupId = groupSet->activeGroup;
    }

    for (; i<groupSet->numberOfThreads; i++)
    {
        ret = perfmon_readCountersThread(i, &groupSet->groups[groupId]);
        if (ret)
        {
            return -i;
        }
    }
    return 0;
}

int perfmon_readCounters(void)
{
    return __perfmon_readCounters(-1);
}

int perfmon_readGroupCounters(int groupId)
{
    return __perfmon_readCounters(groupId);
}

uint64_t
perfmon_getResult(int groupId, int eventId, int threadId)
{
    uint64_t result = 0x0ULL;
    PerfmonEventSetEntry* event;
    PerfmonCounter* counter;
    if (unlikely(groupSet == NULL))
    {
        return 0;
    }
    if (groupId < 0)
    {
        groupId = groupSet->activeGroup;
    }
    if (eventId >= groupSet->groups[groupId].numberOfEvents)
    {
        return 0;
    }
    if (threadId >= groupSet->numberOfThreads)
    {
        return 0;
    }
    event = &(groupSet->groups[groupId].events[eventId]);
    counter = &(event->threadCounter[threadId]);
    
    result = counter->counterData;
    
    if ((counter_map[event->index].type == FIXED) || 
                    (counter_map[event->index].type == PMC))
    {
        result += counter->overflows * get_maxPerfCounterValue();
    }
    else if (counter_map[event->index].type == POWER)
    {
        result += counter->overflows * get_maxPowerCounterValue();
        result -= counter->startData;
        result *= power_info.energyUnit;
    }
    return result;
}

int
perfmon_switchActiveGroup(int new_group)
{
    int ret;
    ret = perfmon_stopCounters();
    if (ret != 0)
    {
        return ret;
    }
    ret = perfmon_setupCounters(new_group);
    if (ret != 0)
    {
        return ret;
    }
    ret = perfmon_startCounters();
    if (ret != 0)
    {
        return ret;
    }
    return 0;
}

int
perfmon_getNumberOfGroups(void)
{
    return groupSet->numberOfActiveGroups;
}

int
perfmon_getIdOfActiveGroup(void)
{
    return groupSet->activeGroup;
}

int
perfmon_getNumberOfThreads(void)
{
    return groupSet->numberOfThreads;
}

int
perfmon_getNumberOfEvents(int groupId)
{
    return groupSet->groups[groupId].numberOfEvents;
}

double
perfmon_getTimeOfGroup(int groupId)
{
    return groupSet->groups[groupId].rdtscTime * 1.E06;
}

int 
perfmon_accessClientInit(void)
{
    if (accessClient_mode != DAEMON_AM_DIRECT)
    {
        accessClient_init(&socket_fd);
        msr_init(socket_fd);
    }
}
