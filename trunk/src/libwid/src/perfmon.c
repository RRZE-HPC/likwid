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


static PerfmonEvent* eventHash;
static PerfmonCounterMap* counter_map;
static PerfmonGroupMap* group_map;
static PerfmonGroupHelp* group_help;
static int perfmon_numCounters;
static int perfmon_numArchEvents;
static int perfmon_numGroups;

static int socket_fd = -1;

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



int
perfmon_init(int nrThreads, int threadsToCpu[])
{
    int i;
    int ret;
    char func_name[100];

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
    
    /*ret =*/power_init(0); /* FIXME Static coreId is dangerous */
    /*ret =*/thermal_init(0);
    timer_init();
    
    
     switch ( cpuid_info.family )
    {
        case P6_FAMILY:

            switch ( cpuid_info.model )
            {
                case PENTIUM_M_BANIAS:

                case PENTIUM_M_DOTHAN:

                    eventHash = pm_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEvents_pm;

                    perfmon_numGroups = perfmon_numGroups_pm;

                    counter_map = pm_counter_map;
                    perfmon_numCounters = perfmon_numCounters_pm;

                    initThreadArch = perfmon_init_pm;
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

                    perfmon_numGroups = perfmon_numGroupsAtom;

                    counter_map = core2_counter_map;
                    perfmon_numCounters = perfmon_numCountersCore2;

                    initThreadArch = perfmon_init_core2;
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

                    perfmon_numGroups = perfmon_numGroupsCore2;

                    counter_map = core2_counter_map;
                    perfmon_numCounters = perfmon_numCountersCore2;

                    initThreadArch = perfmon_init_core2;
                    perfmon_startCountersThread = perfmon_startCountersThread_core2;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_core2;
                    perfmon_readCountersThread = perfmon_readCountersThread_core2;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_core2;
                    break;

                case NEHALEM_EX:

                    eventHash = nehalemEX_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsNehalemEX;

                    group_map = nehalemEX_group_map;
                    group_help = nehalemEX_group_help;
                    perfmon_numGroups = perfmon_numGroupsNehalemEX;

                    counter_map = westmereEX_counter_map;
                    perfmon_numCounters = perfmon_numCountersWestmereEX;

                    initThreadArch = perfmon_init_westmereEX;
                    perfmon_startCountersThread = perfmon_startCountersThread_westmereEX;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_westmereEX;
                    perfmon_readCountersThread = perfmon_readCountersThread_westmereEX;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_nehalemEX;
                    break;

                case WESTMERE_EX:

                    eventHash = westmereEX_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsWestmereEX;

                    group_map = westmereEX_group_map;
                    group_help = westmereEX_group_help;
                    perfmon_numGroups = perfmon_numGroupsWestmereEX;

                    counter_map = westmereEX_counter_map;
                    perfmon_numCounters = perfmon_numCountersWestmereEX;

                    initThreadArch = perfmon_init_westmereEX;
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

                    group_map = nehalem_group_map;
                    group_help = nehalem_group_help;
                    perfmon_numGroups = perfmon_numGroupsNehalem;

                    counter_map = nehalem_counter_map;
                    perfmon_numCounters = perfmon_numCountersNehalem;

                    initThreadArch = perfmon_init_nehalem;
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

                    group_map = westmere_group_map;
                    group_help = westmere_group_help;
                    perfmon_numGroups = perfmon_numGroupsWestmere;

                    counter_map = nehalem_counter_map;
                    perfmon_numCounters = perfmon_numCountersNehalem;

                    initThreadArch = perfmon_init_nehalem;
                    perfmon_startCountersThread = perfmon_startCountersThread_nehalem;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_nehalem;
                    perfmon_readCountersThread = perfmon_readCountersThread_nehalem;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_nehalem;
                    break;

                case IVYBRIDGE:

                case IVYBRIDGE_EP:

                    power_init(0);
                    thermal_init(0);
                    pci_init(socket_fd); 

                    eventHash = ivybridge_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsIvybridge;

                    group_map = ivybridge_group_map;
                    group_help = ivybridge_group_help;
                    perfmon_numGroups = perfmon_numGroupsIvybridge;

                    counter_map = ivybridge_counter_map;
                    perfmon_numCounters = perfmon_numCountersIvybridge;

                    initThreadArch = perfmon_init_ivybridge;
                    perfmon_startCountersThread = perfmon_startCountersThread_ivybridge;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_ivybridge;
                    perfmon_readCountersThread = perfmon_readCountersThread_ivybridge;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_ivybridge;
                    break;

                case HASWELL:

                case HASWELL_EX:

                case HASWELL_M1:

                case HASWELL_M2:

                    power_init(0);
                    thermal_init(0);

                    eventHash = haswell_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsHaswell;
                    perfmon_numGroups = perfmon_numGroupsHaswell;

                    counter_map = haswell_counter_map;
                    perfmon_numCounters = perfmon_numCountersHaswell;

                    initThreadArch = perfmon_init_haswell;
                    perfmon_startCountersThread = perfmon_startCountersThread_haswell;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_haswell;
                    perfmon_readCountersThread = perfmon_readCountersThread_haswell;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_haswell;
                    break;

                case SANDYBRIDGE:

                case SANDYBRIDGE_EP:

                    power_init(0);
                    thermal_init(0);
                    pci_init(socket_fd);

                    eventHash = sandybridge_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsSandybridge;

                    perfmon_numGroups = perfmon_numGroupsSandybridge;

                    counter_map = sandybridge_counter_map;
                    perfmon_numCounters = perfmon_numCountersSandybridge;

                    initThreadArch = perfmon_init_sandybridge;
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

                    group_map = phi_group_map;
                    group_help = phi_group_help;
                    perfmon_numGroups = perfmon_numGroupsPhi;

                    counter_map = phi_counter_map;
                    perfmon_numCounters = perfmon_numCountersPhi;

                    initThreadArch = perfmon_init_phi;
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

            group_map = k8_group_map;
            group_help = k8_group_help;
            perfmon_numGroups = perfmon_numGroupsK8;

            counter_map = k10_counter_map;
            perfmon_numCounters = perfmon_numCountersK10;

            initThreadArch = perfmon_init_k10;
            perfmon_startCountersThread = perfmon_startCountersThread_k10;
            perfmon_stopCountersThread = perfmon_stopCountersThread_k10;
            perfmon_readCountersThread = perfmon_readCountersThread_k10;
            perfmon_setupCounterThread = perfmon_setupCounterThread_k10;
            break;

        case K10_FAMILY:
            eventHash = k10_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsK10;

            group_map = k10_group_map;
            group_help = k10_group_help;
            perfmon_numGroups = perfmon_numGroupsK10;

            counter_map = k10_counter_map;
            perfmon_numCounters = perfmon_numCountersK10;

            initThreadArch = perfmon_init_k10;
            perfmon_startCountersThread = perfmon_startCountersThread_k10;
            perfmon_stopCountersThread = perfmon_stopCountersThread_k10;
            perfmon_readCountersThread = perfmon_readCountersThread_k10;
            perfmon_setupCounterThread = perfmon_setupCounterThread_k10;
            break;

        case K15_FAMILY:
            eventHash = interlagos_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsInterlagos;

            group_map = interlagos_group_map;
            group_help = interlagos_group_help;
            perfmon_numGroups = perfmon_numGroupsInterlagos;

            counter_map = interlagos_counter_map;
            perfmon_numCounters = perfmon_numCountersInterlagos;

            initThreadArch = perfmon_init_interlagos;
            perfmon_startCountersThread = perfmon_startCountersThread_interlagos;
            perfmon_stopCountersThread = perfmon_stopCountersThread_interlagos;
            perfmon_readCountersThread = perfmon_readCountersThread_interlagos;
            perfmon_setupCounterThread = perfmon_setupCounterThread_interlagos;
            break;

        case K16_FAMILY:
            eventHash = kabini_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsKabini;

            group_map = kabini_group_map;
            group_help = kabini_group_help;
            perfmon_numGroups = perfmon_numGroupsKabini;

            counter_map = kabini_counter_map;
            perfmon_numCounters = perfmon_numCountersKabini;

            initThreadArch = perfmon_init_kabini;
            perfmon_startCountersThread = perfmon_startCountersThread_kabini;
            perfmon_stopCountersThread = perfmon_stopCountersThread_kabini;
            perfmon_readCountersThread = perfmon_readCountersThread_kabini;
            perfmon_setupCounterThread = perfmon_setupCounterThread_kabini;
           break;

        default:
            ERROR_PLAIN_PRINT(Unsupported Processor);
            break;
    }
    
    /* Store thread information and reset counters for processor*/
    for(i=0;i<nrThreads;i++)
    {
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
    
    if (eventCString == NULL)
    {
        fprintf(stderr,"Event string is empty\nTrying environment variable LIKWID_EVENTS\n");
        eventCString = getenv("LIKWID_EVENTS");
        if (eventCString == NULL)
        {
            fprintf(stderr,"Event string from environment variable is empty");
            return -EINVAL;
        }
        
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
        ERROR_PRINT(Cannot allocate event list for group %d, groupSet->numberOfActiveGroups);
        bstrListDestroy(tokens);
        return -ENOMEM;
    }
    eventSet->numberOfEvents = 0;
    
       
       for(i=0;i<tokens->qty;i++)
       {
           event = &(groupSet->groups[groupSet->numberOfActiveGroups].events[i]);
           
           subtokens = bsplit(tokens->entry[i],':');

           if (subtokens->qty != 2)
           {
               ERROR_PRINT(Cannot parse event descriptor %s,tokens->entry[i]);
               bstrListDestroy(subtokens);
               continue;
           }
           else
           {
               if (!getIndex(subtokens->entry[1], &event->index))
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
perfmon_startCounters(void)
{
    int i;
    for(i=0;i<groupSet->numberOfThreads;i++)
    {
        perfmon_startCountersThread(i, &groupSet->groups[groupSet->activeGroup]);
    }
    
    
    timer_start(&groupSet->groups[groupSet->activeGroup].timer);
    return 0;
}

int
perfmon_stopCounters(void)
{
    timer_stop(&groupSet->groups[groupSet->activeGroup].timer);

    for (int i=0; i<groupSet->numberOfThreads; i++)
    {
        perfmon_stopCountersThread(i, &groupSet->groups[groupSet->activeGroup]);
    }

    groupSet->groups[groupSet->activeGroup].rdtscTime += 
                timer_print(&groupSet->groups[groupSet->activeGroup].timer);
    return 0;
}

int
perfmon_readCounters(void)
{
    for (int i=0; i<groupSet->numberOfThreads; i++)
    {
        perfmon_readCountersThread(i, &groupSet->groups[groupSet->activeGroup]);
    }
    return 0;
}

uint64_t
perfmon_getResult(int groupId, int eventId, int threadId)
{
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
    return groupSet->groups[groupId].events[eventId].threadCounter[threadId].counterData;
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
perfmon_getNumberOfActiveGroup(void)
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
