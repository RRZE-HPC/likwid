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
#include <msr.h>
#include <pci.h>
#include <lock.h>
#include <perfmon.h>
#include <registers.h>
#include <topology.h>
#include <access.h>

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
#include <perfmon_broadwell.h>


PerfmonEvent* eventHash;
RegisterMap* counter_map = NULL;
BoxMap* box_map = NULL;
PciDevice* pci_devices = NULL;
int perfmon_numCounters = 0;
int perfmon_numCoreCounters = 0;
int perfmon_numArchEvents = 0;
int perfmon_verbosity = DEBUGLEV_ONLY_ERROR;

int socket_fd = -1;
int thread_sockets[MAX_NUM_THREADS] = { [0 ... MAX_NUM_THREADS-1] = -1};


PerfmonGroupSet* groupSet = NULL;

int (*perfmon_startCountersThread) (int thread_id, PerfmonEventSet* eventSet);
int (*perfmon_stopCountersThread) (int thread_id, PerfmonEventSet* eventSet);
int (*perfmon_readCountersThread) (int thread_id, PerfmonEventSet* eventSet);
int (*perfmon_setupCountersThread) (int thread_id, PerfmonEventSet* eventSet);
int (*perfmon_finalizeCountersThread) (int thread_id, PerfmonEventSet* eventSet);

int (*initThreadArch) (int cpu_id);


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
    "COUNT_KERNEL",
    "ANYTHREAD",
    "OCCUPANCY",
    "OCCUPANCY_FILTER",
    "OCCUPANCY_EDGEDETECT",
    "OCCUPANCY_INVERT",
    "IN_TRANSACTION",
    "IN_TRANSACTION_ABORTED"
};

static int
getIndexAndType (bstring reg, RegisterIndex* index, RegisterType* type)
{
    int err = 0;
    int ret = FALSE;
    uint64_t tmp;
    int (*ownstrcmp)(const char*, const char*);
    ownstrcmp = &strcmp;
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
    if (ret && (ownstrcmp(bdata(reg), counter_map[*index].key) != 0))
    {
        *type = NOTYPE;
        return FALSE;
    }
    if (!pci_checkDevice(counter_map[*index].device, 0))
    {
        *type = NOTYPE;
        return FALSE;
    }
    if ((ret) && (*type != THERMAL) && (*type != POWER) && (*type != WBOX0FIX))
    {
        err = HPMread(0, counter_map[*index].device, counter_map[*index].counterRegister, &tmp);
        if (err != 0)
        {
            if (err == -ENODEV)
            {
                DEBUG_PRINT(DEBUGLEV_DETAIL, Device %s not accessible on this machine,
                                         pci_devices[box_map[*type].device].name);
            }
            else
            {
                DEBUG_PRINT(DEBUGLEV_DETAIL, Counter %s not readable on this machine,
                                             counter_map[*index].key);
            }
            *type = NOTYPE;
            ret = FALSE;
        }
        else if (tmp == 0x0)
        {
            err = HPMwrite(0, counter_map[*index].device, counter_map[*index].counterRegister, 0x0U);
            if (err != 0)
            {
                if (err == -ENODEV)
                {
                    DEBUG_PRINT(DEBUGLEV_DETAIL, Device %s not accessible on this machine,
                                             pci_devices[box_map[*type].device].name);
                }
                else
                {
                    DEBUG_PRINT(DEBUGLEV_DETAIL, Counter %s not writeable on this machine,
                                             counter_map[*index].key);
                }
                *type = NOTYPE;
                ret = FALSE;
            }
        }
        /*else
        {
            printf("Err %d Tmp 0x%llx\n", err, tmp);
            DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, Counter %s already in use. Skipping setup of this event,
                                             counter_map[*index].key);
            *type = NOTYPE;
        }*/
    }
    else if ((ret) && ((*type == POWER) || (*type == WBOX0FIX) || (*type == THERMAL)))
    {
        err = HPMread(0, MSR_DEV, counter_map[*index].counterRegister, &tmp);
        if (err != 0)
        {
            DEBUG_PRINT(DEBUGLEV_DETAIL, Counter %s not readable on this machine,
                                         counter_map[*index].key);
            *type = NOTYPE;
            ret = FALSE;
        }
    }
    else
    {
        *type = NOTYPE;
        ret = FALSE;
    }
    return ret;
}

static int
checkCounter(bstring counterName, const char* limit)
{
    int i;
    struct bstrList* tokens;
    int ret = FALSE;
    bstring limitString = bfromcstr(limit);

    tokens = bstrListCreate();
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
            if (!checkCounter(counter_str, eventHash[i].limit))
            {
                continue;
            }
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
        bstrListDestroy(tokens);
        return -ERANGE;
    }

    subtokens = bstrListCreate();

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
            else
            {
                continue;
            }
            //sscanf(bdata(subtokens->entry[1]), "%x", &(event->options[event->numberOfOptions].value));
        }
        //event->numberOfOptions++;
    }
    for(i=event->numberOfOptions-1;i>=0;i--)
    {
        if (!(OPTIONS_TYPE_MASK(event->options[i].type) & (counter_map[index].optionMask|event->optionMask)))
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

    bstrListDestroy(subtokens);
    return event->numberOfOptions;
}

int
getCounterTypeOffset(int index)
{
    int off = 0;
    for (int j=index-1;j>=NUM_COUNTERS_CORE_IVYBRIDGE;j--)
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
perfmon_init_maps(void)
{
    box_map = NULL;
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
                    break;

                case NEHALEM_EX:
                    eventHash = nehalemEX_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsNehalemEX;
                    counter_map = nehalemEX_counter_map;
                    perfmon_numCounters = perfmon_numCountersNehalemEX;
                    box_map = nehalemEX_box_map;
                    break;

                case WESTMERE_EX:
                    eventHash = westmereEX_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsWestmereEX;
                    counter_map = westmereEX_counter_map;
                    perfmon_numCounters = perfmon_numCountersWestmereEX;
                    box_map = westmereEX_box_map;
                    break;

                case NEHALEM_BLOOMFIELD:
                case NEHALEM_LYNNFIELD:
                case NEHALEM_LYNNFIELD_M:
                    eventHash = nehalem_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsNehalem;
                    counter_map = nehalem_counter_map;
                    perfmon_numCounters = perfmon_numCountersNehalem;
                    box_map = nehalem_box_map;
                    break;

                case NEHALEM_WESTMERE_M:
                case NEHALEM_WESTMERE:
                    eventHash = westmere_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsWestmere;
                    counter_map = nehalem_counter_map;
                    perfmon_numCounters = perfmon_numCountersNehalem;
                    box_map = nehalem_box_map;
                    break;

                case IVYBRIDGE_EP:
                    pci_devices = ivybridgeEP_pci_devices;
                    box_map = ivybridgeEP_box_map;
                    eventHash = ivybridgeEP_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsIvybridgeEP;
                    counter_map = ivybridgeEP_counter_map;
                    perfmon_numCounters = perfmon_numCountersIvybridgeEP;
                    perfmon_numCoreCounters = perfmon_numCoreCountersIvybridgeEP;
                    break;
                case IVYBRIDGE:
                    eventHash = ivybridge_arch_events;
                    box_map = ivybridge_box_map;
                    perfmon_numArchEvents = perfmon_numArchEventsIvybridge;
                    counter_map = ivybridge_counter_map;
                    perfmon_numCounters = perfmon_numCountersIvybridge;
                    perfmon_numCoreCounters = perfmon_numCoreCountersIvybridge;
                    break;

                case HASWELL_EP:
                    eventHash = haswellEP_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsHaswellEP;
                    counter_map = haswellEP_counter_map;
                    perfmon_numCounters = perfmon_numCountersHaswellEP;
                    perfmon_numCoreCounters = perfmon_numCoreCountersHaswellEP;
                    box_map = haswellEP_box_map;
                    pci_devices = haswellEP_pci_devices;
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
                    break;

                case SANDYBRIDGE_EP:
                    pci_devices = sandybridgeEP_pci_devices;
                    box_map = sandybridgeEP_box_map;
                    eventHash = sandybridgeEP_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsSandybridgeEP;
                    counter_map = sandybridgeEP_counter_map;
                    perfmon_numCounters = perfmon_numCountersSandybridgeEP;
                    perfmon_numCoreCounters = perfmon_numCoreCountersSandybridgeEP;
                    break;
                case SANDYBRIDGE:
                    box_map = sandybridge_box_map;
                    eventHash = sandybridge_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsSandybridge;
                    counter_map = sandybridge_counter_map;
                    perfmon_numCounters = perfmon_numCountersSandybridge;
                    perfmon_numCoreCounters = perfmon_numCoreCountersSandybridge;
                    break;

                case BROADWELL:
                case BROADWELL_E:
                case BROADWELL_D:
                    box_map = broadwell_box_map;
                    eventHash = broadwell_arch_events;
                    counter_map = broadwell_counter_map;
                    perfmon_numArchEvents = perfmon_numArchEventsBroadwell;
                    perfmon_numCounters = perfmon_numCountersBroadwell;
                    perfmon_numCoreCounters = perfmon_numCoreCountersBroadwell;
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
            break;

        case K10_FAMILY:
            eventHash = k10_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsK10;
            counter_map = k10_counter_map;
            box_map = k10_box_map;
            perfmon_numCounters = perfmon_numCountersK10;
            break;

        case K15_FAMILY:
            eventHash = interlagos_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsInterlagos;
            counter_map = interlagos_counter_map;
            box_map = interlagos_box_map;
            perfmon_numCounters = perfmon_numCountersInterlagos;
            break;

        case K16_FAMILY:
            eventHash = kabini_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsKabini;
            counter_map = kabini_counter_map;
            box_map = kabini_box_map;
            perfmon_numCounters = perfmon_numCountersKabini;
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
                    initialize_power = TRUE;
                    initialize_thermal = TRUE;
                    initThreadArch = perfmon_init_broadwell;
                    perfmon_startCountersThread = perfmon_startCountersThread_broadwell;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_broadwell;
                    perfmon_readCountersThread = perfmon_readCountersThread_broadwell;
                    perfmon_setupCountersThread = perfmon_setupCounterThread_broadwell;
                    perfmon_finalizeCountersThread = perfmon_finalizeCountersThread_broadwell;
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

        default:
            ERROR_PLAIN_PRINT(Unsupported Processor);
            break;
    }
    *init_power = initialize_power;
    *init_temp = initialize_thermal;
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
        ERROR_PRINT(Number of threads must be greater than 0 but only %d given,nrThreads);
        return -EINVAL;
    }
    
    if (!lock_check())
    {
        ERROR_PLAIN_PRINT(Access to performance monitoring registers locked);
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

    for(i=0; i<MAX_NUM_NODES; i++) socket_lock[i] = LOCK_INIT;
    for(i=0; i<MAX_NUM_THREADS; i++) tile_lock[i] = LOCK_INIT;

    /* Initialize maps pointer to current architecture maps */
    perfmon_init_maps();

    /* Initialize access interface */
    ret = HPMaddThread(threadsToCpu[0]);
    if (ret)
    {
        ERROR_PLAIN_PRINT(Cannot get access to performance counters);
        free(groupSet->threads);
        free(groupSet);
        return ret;
    }
    timer_init();

    
    /* Initialize function pointer to current architecture functions */
    perfmon_init_funcs(&initialize_power, &initialize_thermal);

    /* Store thread information and reset counters for processor*/
    /* If the arch supports it, initialize power and thermal measurements */
    for(i=0;i<nrThreads;i++)
    {
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
    return 0;
}

void 
perfmon_finalize(void)
{
    int group, event;
    int thread;
    for(group=0;group < groupSet->numberOfGroups; group++)
    {
        for (thread=0;thread< groupSet->numberOfThreads; thread++)
        {
            perfmon_finalizeCountersThread(thread, &(groupSet->groups[group]));
        }
        for (event=0;event < groupSet->groups[group].numberOfEvents; event++)
        {
            free(groupSet->groups[group].events[event].threadCounter);
        }
        free(groupSet->groups[group].events);
    }
    
    free(groupSet->threads);
    free(groupSet);
    HPMfinalize();
    power_finalize();
    return;
}

int 
perfmon_addEventSet(char* eventCString)
{
    int i, j;
    bstring eventBString;
    struct bstrList* eventtokens;
    struct bstrList* subtokens;
    PerfmonEventSet* eventSet;
    PerfmonEventSetEntry* event;

    if (eventCString == NULL)
    {
        DEBUG_PLAIN_PRINT(1, Event string is empty. Trying environment variable LIKWID_EVENTS);
        eventCString = getenv("LIKWID_EVENTS");
        if (eventCString == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot read event string. Also event string from environment variable is empty);
            return -EINVAL;
        }
    }

    if (strchr(eventCString, '-') != NULL)
    {
        ERROR_PLAIN_PRINT(Event string contains valid character -);
        return -EINVAL;
    }
    if (strchr(eventCString, '.') != NULL)
    {
        ERROR_PLAIN_PRINT(Event string contains valid character .);
        return -EINVAL;
    }
    if (groupSet->numberOfGroups == 0)
    {
        groupSet->groups = (PerfmonEventSet*) malloc(sizeof(PerfmonEventSet));
        if (groupSet->groups == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate initialize of event group list);
            return -ENOMEM;
        }

        groupSet->numberOfGroups = 1;
        groupSet->numberOfActiveGroups = 0;

        /* Only one group exists by now */
        groupSet->groups[0].rdtscTime = 0;
        groupSet->groups[0].runTime = 0;
        groupSet->groups[0].numberOfEvents = 0;
    }
    
    if (groupSet->numberOfActiveGroups == groupSet->numberOfGroups)
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

    eventSet = &(groupSet->groups[groupSet->numberOfActiveGroups]);

    eventBString = bfromcstr(eventCString);
    eventtokens = bstrListCreate();
    eventtokens = bsplit(eventBString,',');
    bdestroy(eventBString);
    eventSet->events = (PerfmonEventSetEntry*) malloc(eventtokens->qty * sizeof(PerfmonEventSetEntry));
    
    if (eventSet->events == NULL)
    {
        ERROR_PRINT(Cannot allocate event list for group %d\n, groupSet->numberOfActiveGroups);
        return -ENOMEM;
    }
    eventSet->numberOfEvents = 0;
    eventSet->regTypeMask = 0x0ULL;

    subtokens = bstrListCreate();
    
    for(i=0;i<eventtokens->qty;i++)
    {
        event = &(eventSet->events[i]);
        subtokens = bsplit(eventtokens->entry[i],':');
        if (subtokens->qty < 2)
        {
            fprintf(stderr,"Cannot parse event descriptor %s\n", bdata(eventtokens->entry[i]));
            continue;
        }
        else
        {
            if (!getIndexAndType(subtokens->entry[1], &event->index, &event->type))
            {
                fprintf(stderr,"Counter register %s not supported or PCI device not available\n",bdata(
                        subtokens->entry[1]));
                event->type = NOTYPE;
                goto past_checks;
            }

            if (!getEvent(subtokens->entry[0], subtokens->entry[1], &event->event))
            {
                fprintf(stderr,"Event %s not found for current architecture\n",
                     bdata(subtokens->entry[0]));
                event->type = NOTYPE;
                goto past_checks;
            }
           
            if (!checkCounter(subtokens->entry[1], event->event.limit))
            {
                fprintf(stderr,"Register %s not allowed for event %s\n",
                     bdata(subtokens->entry[1]),bdata(subtokens->entry[0]));
                event->type = NOTYPE;
                goto past_checks;
            }
            if (parseOptions(subtokens, &event->event, event->index) < 0)
            {
                fprintf(stderr,"Cannot parse options in %s\n", bdata(eventtokens->entry[i]));
                event->type = NOTYPE;
                goto past_checks;
            }
            
            eventSet->regTypeMask |= REG_TYPE_MASK(event->type);
past_checks:
            event->threadCounter = (PerfmonCounter*) malloc(
                groupSet->numberOfThreads * sizeof(PerfmonCounter));

            if (event->threadCounter == NULL)
            {
                ERROR_PRINT(Cannot allocate counter for all threads in group %d,groupSet->numberOfActiveGroups);
                continue;
            }
            for(j=0;j<groupSet->numberOfThreads;j++)
            {
                event->threadCounter[j].counterData = 0;
                event->threadCounter[j].startData = 0;
                event->threadCounter[j].overflows = 0;
                event->threadCounter[j].init = FALSE;
            }

            eventSet->numberOfEvents++;

            if (event->type != NOTYPE)
            {
                DEBUG_PRINT(DEBUGLEV_INFO,
                        Added event %s for counter %s to group %d,
                        event->event.name,
                        counter_map[event->index].key,
                        groupSet->numberOfActiveGroups);
            }
        }
    }
    bstrListDestroy(subtokens);
    bstrListDestroy(eventtokens);
    groupSet->numberOfActiveGroups++;
    if ((eventSet->numberOfEvents > 0) && (eventSet->regTypeMask != 0x0ULL))
    {
        return groupSet->numberOfActiveGroups-1;
    }
    else
    {
        fprintf(stderr,"No event in given event string can be configured\n");
        return -EINVAL;
    }
}

int
__perfmon_setupCountersThread(int thread_id, int groupId)
{
    int i;
    if (groupId >= groupSet->numberOfActiveGroups)
    {
        ERROR_PRINT(Group %d does not exist in groupSet, groupId);
        return -ENOENT;
    }

    CHECK_AND_RETURN_ERROR(perfmon_setupCountersThread(thread_id, &groupSet->groups[groupId]),
            Setup of counters failed);

    groupSet->activeGroup = groupId;
    return 0;
}

int
perfmon_setupCounters(int groupId)
{
    int i;
    int ret = 0;
    for(i=0;i<groupSet->numberOfThreads;i++)
    {
        ret = __perfmon_setupCountersThread(groupSet->threads[i].thread_id, groupId);
        if (ret != 0)
        {
            return ret;
        }
    }
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
        ret = perfmon_startCountersThread(groupSet->threads[i].thread_id, &groupSet->groups[groupId]);
        if (ret)
        {
            return -groupSet->threads[i].thread_id-1;
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
    return __perfmon_startCounters(groupId);
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
        ret = perfmon_stopCountersThread(groupSet->threads[i].thread_id, &groupSet->groups[groupId]);
        if (ret)
        {
            return -groupSet->threads[i].thread_id-1;
        }
    }

    groupSet->groups[groupId].rdtscTime =
                timer_print(&groupSet->groups[groupId].timer);
    groupSet->groups[groupId].runTime += groupSet->groups[groupId].rdtscTime;
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
__perfmon_readCounters(int groupId, int threadId)
{
    int ret = 0;

    if ((groupId < 0) || (groupId >= groupSet->numberOfActiveGroups))
    {
        groupId = groupSet->activeGroup;
    }

    if (threadId == -1)
    {
        for (threadId = 0; threadId<groupSet->numberOfThreads; threadId++)
        {
            ret = perfmon_readCountersThread(threadId, &groupSet->groups[groupId]);
            if (ret)
            {
                return -threadId-1;
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
    }
    return 0;
}

int perfmon_readCounters(void)
{
    return __perfmon_readCounters(-1,-1);
}

int perfmon_readCountersCpu(int cpu_id)
{
    int i;
    int thread_id = 0;
    for(i=0;i<groupSet->numberOfThreads;i++)
    {
        if (groupSet->threads[i].processorId == cpu_id)
        {
            thread_id = groupSet->threads[i].thread_id;
            break;
        }
    }
    return perfmon_readCountersThread(thread_id, &groupSet->groups[groupSet->activeGroup]);
}

int perfmon_readGroupCounters(int groupId)
{
    return __perfmon_readCounters(groupId,-1);
}
int perfmon_readGroupThreadCounters(int groupId, int threadId)
{
    return __perfmon_readCounters(groupId,threadId);
}


double
perfmon_getResult(int groupId, int eventId, int threadId)
{
    double result = 0.0;
    PerfmonEventSetEntry* event;
    PerfmonCounter* counter;
    int cpu_id;
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
        printf("ERROR: EventID greater than defined events\n");
        return 0;
    }
    if (threadId >= groupSet->numberOfThreads)
    {
        printf("ERROR: ThreadID greater than defined threads\n");
        return 0;
    }
    event = &(groupSet->groups[groupId].events[eventId]);
    counter = &(event->threadCounter[threadId]);
    cpu_id = groupSet->threads[threadId].processorId;

    if (counter->overflows == 0)
    {
        result = (double) (counter->counterData - counter->startData);
    }
    else if (counter->overflows > 0)
    {
        result += (double) ((perfmon_getMaxCounterValue(counter_map[event->index].type) - counter->startData) + counter->counterData);
        counter->overflows--;
    }
    result += (double) (counter->overflows * perfmon_getMaxCounterValue(counter_map[event->index].type));

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

int __perfmon_switchActiveGroupThread(int thread_id, int new_group)
{
    int ret;
    int i;
    ret = perfmon_stopCounters();
    if (ret != 0)
    {
        return ret;
    }
    for(i=0; i<groupSet->groups[groupSet->activeGroup].numberOfEvents;i++)
    {
        groupSet->groups[groupSet->activeGroup].events[i].threadCounter[thread_id].init = FALSE;
    }
    /*for(i=0; i<groupSet->groups[new_group].numberOfEvents;i++)
    {
        groupSet->groups[new_group].events[i].threadCounter[cpu_id].init = TRUE;
    }*/
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
perfmon_switchActiveGroup(int new_group)
{
    int i=0;
    int ret=0;
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
    if (groupId < 0)
    {
        groupId = groupSet->activeGroup;
    }
    return groupSet->groups[groupId].numberOfEvents;
}

double
perfmon_getTimeOfGroup(int groupId)
{
    if (groupId < 0)
    {
        groupId = groupSet->activeGroup;
    }
    return groupSet->groups[groupId].runTime;
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



