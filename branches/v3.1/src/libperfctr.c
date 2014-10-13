/*
 * =======================================================================================
 *
 *      Filename:  libperfctr.c
 *
 *      Description:  Marker API interface of module perfmon
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
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
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>

#include <error.h>
#include <types.h>
#include <bitUtil.h>
#include <bstrlib.h>
#include <cpuid.h>
#include <numa.h>
#include <affinity.h>
#include <lock.h>
#include <tree.h>
#include <accessClient.h>
#include <msr.h>
#include <pci.h>
#include <power.h>
#include <thermal.h>
#include <timer.h>
#include <hashTable.h>
#include <registers.h>
#include <likwid.h>

#include <perfmon_core2_counters.h>
#include <perfmon_haswell_counters.h>
#include <perfmon_interlagos_counters.h>
#include <perfmon_kabini_counters.h>
#include <perfmon_k10_counters.h>
#include <perfmon_nehalem_counters.h>
#include <perfmon_phi_counters.h>
#include <perfmon_pm_counters.h>
#include <perfmon_sandybridge_counters.h>
#include <perfmon_ivybridge_counters.h>
#include <perfmon_westmereEX_counters.h>
#include <perfmon_silvermont_counters.h>


/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int perfmon_numCounters=0;     /* total number of counters */
static int perfmon_numCountersCore=0; /* max index of core counters */
static int perfmon_numCountersUncore=0; /* max index of conventional uncore counters */
static PerfmonCounterMap* perfmon_counter_map = NULL;
static int socket_lock[MAX_NUM_NODES];
static int thread_socketFD[MAX_NUM_THREADS];
static int hasPCICounters = 0;
static int likwid_init = 0;
static BitMask counterMask;

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define gettid() syscall(SYS_gettid)

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

void str2BitMask(const char* str, BitMask* mask)
{
    char* endptr;
    errno = 0;
    struct bstrList* tokens;
    bstring q = bfromcstralloc (60, str);
    tokens = bsplit(q,' ');

    for (int i=0; i<tokens->qty; i++)
    {
        uint64_t val =  strtoull((char*) tokens->entry[i]->data, &endptr, 16);

        if ((errno == ERANGE && val == LONG_MAX ) || (errno != 0 && val == 0))
        {
            ERROR;
        }

        if (endptr == str)
        {
            ERROR_PLAIN_PRINT(No digits were found);
        }

        mask->mask[i] = val;
    }

    bstrListDestroy(tokens);
    bdestroy(q);
}

static int getProcessorID(cpu_set_t* cpu_set)
{
    int processorId;

    for (processorId=0;processorId<MAX_NUM_THREADS;processorId++)
    {
        if (CPU_ISSET(processorId,cpu_set))
        {
            break;
        }
    }
    return processorId;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void likwid_markerInit(void)
{
    int cpuId = likwid_getProcessorId();
    char* modeStr = getenv("LIKWID_MODE");
    char* maskStr = getenv("LIKWID_MASK");

    if ((modeStr != NULL) && (maskStr != NULL))
    {
        likwid_init = 1;
    }
    else
    {
        return;
    }

    if (!lock_check())
    {
        fprintf(stderr,"Access to performance counters is locked.\n");
        exit(EXIT_FAILURE);
    }

    cpuid_init();
    numa_init();
    affinity_init();
    timer_init();
    hashTable_init();

    for(int i=0; i<MAX_NUM_THREADS; i++) thread_socketFD[i] = -1;
    for(int i=0; i<MAX_NUM_NODES; i++) socket_lock[i] = LOCK_INIT;

    accessClient_mode = atoi(modeStr);
    str2BitMask(maskStr, &counterMask);

    if (accessClient_mode != DAEMON_AM_DIRECT)
    {
        accessClient_init(&thread_socketFD[cpuId]);
    }

    msr_init(thread_socketFD[cpuId]);
    thermal_init(cpuId);

    switch ( cpuid_info.family )
    {
        case P6_FAMILY:

            switch ( cpuid_info.model )
            {
                case PENTIUM_M_BANIAS:

                case PENTIUM_M_DOTHAN:

                    perfmon_counter_map = pm_counter_map;
                    perfmon_numCounters = NUM_COUNTERS_PM;
                    perfmon_numCountersCore = NUM_COUNTERS_CORE_PM;
                    break;

                case ATOM_45:

                case ATOM_32:

                case ATOM_22:

                case ATOM:

                    perfmon_counter_map = core2_counter_map;
                    perfmon_numCounters = NUM_COUNTERS_CORE2;
                    perfmon_numCountersCore = NUM_COUNTERS_CORE_CORE2;
                    break;

                case ATOM_SILVERMONT:

                    power_init(0);
                    perfmon_counter_map = silvermont_counter_map;
                    perfmon_numCounters = NUM_COUNTERS_SILVERMONT;
                    perfmon_numCountersCore = NUM_COUNTERS_CORE_SILVERMONT;
                    break;

                case CORE_DUO:
                    ERROR_PLAIN_PRINT(Unsupported Processor);
                    break;

                case XEON_MP:

                case CORE2_65:

                case CORE2_45:

                    perfmon_counter_map = core2_counter_map;
                    perfmon_numCounters = NUM_COUNTERS_CORE2;
                    perfmon_numCountersCore = NUM_COUNTERS_CORE_CORE2;
                    break;

                case NEHALEM_EX:

                case WESTMERE_EX:

                    perfmon_counter_map = westmereEX_counter_map;
                    perfmon_numCounters = NUM_COUNTERS_WESTMEREEX;
                    perfmon_numCountersCore = NUM_COUNTERS_CORE_WESTMEREEX;
                    perfmon_numCountersUncore = NUM_COUNTERS_UNCORE_WESTMEREEX;
                    break;

                case NEHALEM_BLOOMFIELD:

                case NEHALEM_LYNNFIELD:

                case NEHALEM_WESTMERE_M:

                case NEHALEM_WESTMERE:

                    perfmon_counter_map = nehalem_counter_map;
                    perfmon_numCounters = NUM_COUNTERS_NEHALEM;
                    perfmon_numCountersCore = NUM_COUNTERS_CORE_NEHALEM;
                    perfmon_numCountersUncore = NUM_COUNTERS_UNCORE_NEHALEM;
                    break;

                case IVYBRIDGE:

                case IVYBRIDGE_EP:

                    {
                        int socket_fd = thread_socketFD[cpuId];
                        hasPCICounters = 1;
                        power_init(0); /* FIXME Static coreId is dangerous */
                        pci_init(socket_fd);
                        perfmon_counter_map = ivybridge_counter_map;
                        perfmon_numCounters = NUM_COUNTERS_IVYBRIDGE;
                        perfmon_numCountersCore = NUM_COUNTERS_CORE_IVYBRIDGE;
                        perfmon_numCountersUncore = NUM_COUNTERS_UNCORE_IVYBRIDGE;
                    }
                    break;

                case HASWELL:

                case HASWELL_EX:

                case HASWELL_M1:

                case HASWELL_M2:

                    power_init(0); /* FIXME Static coreId is dangerous */

                    perfmon_counter_map = haswell_counter_map;
                    perfmon_numCounters = NUM_COUNTERS_HASWELL;
                    perfmon_numCountersCore = NUM_COUNTERS_CORE_HASWELL;
                    break;

                case SANDYBRIDGE:

                case SANDYBRIDGE_EP:

                    {
                        int socket_fd = thread_socketFD[cpuId];
                        hasPCICounters = 1;
                        power_init(0); /* FIXME Static coreId is dangerous */
                        pci_init(socket_fd);
                        perfmon_counter_map = sandybridge_counter_map;
                        perfmon_numCounters = NUM_COUNTERS_SANDYBRIDGE;
                        perfmon_numCountersCore = NUM_COUNTERS_CORE_SANDYBRIDGE;
                        perfmon_numCountersUncore = NUM_COUNTERS_UNCORE_SANDYBRIDGE;
                    }
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

                    perfmon_counter_map = phi_counter_map;
                    perfmon_numCounters = NUM_COUNTERS_PHI;
                    perfmon_numCountersCore = NUM_COUNTERS_CORE_PHI;
                    break;

                default:
                    ERROR_PLAIN_PRINT(Unsupported Processor);
                    break;
            }
            break;

        case K8_FAMILY:

            perfmon_counter_map = k10_counter_map;
            perfmon_numCounters = NUM_COUNTERS_K10;
            perfmon_numCountersCore = NUM_COUNTERS_CORE_K10;
            break;

        case K10_FAMILY:

            perfmon_counter_map = k10_counter_map;
            perfmon_numCounters = NUM_COUNTERS_K10;
            perfmon_numCountersCore = NUM_COUNTERS_CORE_K10;
            break;

        case K15_FAMILY:

            perfmon_counter_map = interlagos_counter_map;
            perfmon_numCounters = NUM_COUNTERS_INTERLAGOS;
            perfmon_numCountersCore = NUM_COUNTERS_CORE_INTERLAGOS;
            break;

        case K16_FAMILY:

            perfmon_counter_map = kabini_counter_map;
            perfmon_numCounters = NUM_COUNTERS_KABINI;
            perfmon_numCountersCore = NUM_COUNTERS_CORE_KABINI;
            break;

        default:
            ERROR_PLAIN_PRINT(Unsupported Processor);
            break;
    }
}

void likwid_markerThreadInit(void)
{
    if ( ! likwid_init )
    {
        return;
    }

    int cpuId = likwid_getProcessorId();

    if (accessClient_mode != DAEMON_AM_DIRECT)
    {
        if (thread_socketFD[cpuId] == -1)
        {
            accessClient_init(&thread_socketFD[cpuId]);
        }
    }
}

/* File format
 * 1 numberOfThreads numberOfRegions
 * 2 regionID:regionTag0
 * 3 regionID:regionTag1
 * 4 regionID threadID countersvalues(space separated)
 * 5 regionID threadID countersvalues
 */
void likwid_markerClose(void)
{
    FILE *file = NULL;
    LikwidResults* results = NULL;
    int numberOfThreads;
    int numberOfRegions;

    if ( ! likwid_init )
    {
        return;
    }

    hashTable_finalize(&numberOfThreads, &numberOfRegions, &results);

    file = fopen(getenv("LIKWID_FILEPATH"),"w");

    if (file != NULL)
    {
        fprintf(file,"%d %d\n",numberOfThreads,numberOfRegions);

        for (int i=0; i<numberOfRegions; i++)
        {
            fprintf(file,"%d:%s\n",i,bdata(results[i].tag));
        }

        for (int i=0; i<numberOfRegions; i++)
        {
            for (int j=0; j<numberOfThreads; j++)
            {
                fprintf(file,"%d ",i);
                fprintf(file,"%d ",j);
                fprintf(file,"%u ",results[i].count[j]);
                fprintf(file,"%e ",results[i].time[j]);

                for (int k=0; k<NUM_PMC; k++)
                {
                    fprintf(file,"%e ",results[i].counters[j][k]);
                }
                fprintf(file,"\n");
            }
        }
        fclose(file);
    }

    for (int i=0;i<numberOfRegions; i++)
    {
        for (int j=0;j<numberOfThreads; j++)
        {
            free(results[i].counters[j]);
        }
        free(results[i].time);
        bdestroy(results[i].tag);
        free(results[i].count);
        free(results[i].counters);
    }

    if (results != NULL)
    {
        free(results);
    }

    msr_finalize();
    pci_finalize();

    for (int i=0; i<MAX_NUM_THREADS; i++)
    {
        accessClient_finalize(thread_socketFD[i]);
        thread_socketFD[i] = -1;
    }
}


void likwid_markerStartRegion(const char* regionTag)
{
    if ( ! likwid_init )
    {
        return;
    }

    bstring tag = bfromcstralloc(100, regionTag);
    LikwidThreadResults* results;
    uint64_t res;
    int cpu_id = hashTable_get(tag, &results);
    bdestroy(tag);
    int socket_fd = thread_socketFD[cpu_id];

    if (accessClient_mode != DAEMON_AM_DIRECT)
    {
        if (socket_fd == -1)
        {
            printf("ERROR: Invalid socket file handle on processor %d. \
                    Did you call likwid_markerThreadInit() ?\n", cpu_id);
        }
    }

    results->count++;

    /* Core specific counters */
    for ( int i=0; i<perfmon_numCountersCore; i++ )
    {
        bitMask_test(res,counterMask,i);
        if ( res )
        {
            if (perfmon_counter_map[i].type != THERMAL)
            {
                results->StartPMcounters[i] =
                    (double) msr_tread(
                            socket_fd,
                            cpu_id,
                            perfmon_counter_map[i].counterRegister);
            }
        }
    }

    /* Uncore specific counters */
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id) ||
            lock_acquire((int*)
                &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id))
    {
        /* Conventional Uncore counters */
        for ( int i=perfmon_numCountersCore; i<perfmon_numCountersUncore; i++ )
        {
            bitMask_test(res,counterMask,i);
            if ( res )
            {
                if (perfmon_counter_map[i].type != POWER)
                {
                    results->StartPMcounters[i] =
                        (double) msr_tread(
                                socket_fd,
                                cpu_id,
                                perfmon_counter_map[i].counterRegister);
                }
                else
                {
                    results->StartPMcounters[i] =
                        (double) power_tread(
                                socket_fd,
                                cpu_id,
                                perfmon_counter_map[i].counterRegister);
                }
            }
        }

        /* PCI Uncore counters */
        if ( hasPCICounters && (accessClient_mode != DAEMON_AM_DIRECT) )
        {
            for ( int i=perfmon_numCountersUncore; i<perfmon_numCounters; i++ )
            {
                bitMask_test(res,counterMask,i);
                if ( res )
                {
                    uint64_t counter_result =
                        pci_tread(
                                socket_fd,
                                cpu_id,
                                perfmon_counter_map[i].device,
                                perfmon_counter_map[i].counterRegister);

                    counter_result = (counter_result<<32) +
                        pci_tread(
                                socket_fd,
                                cpu_id,
                                perfmon_counter_map[i].device,
                                perfmon_counter_map[i].counterRegister2);

                    results->StartPMcounters[perfmon_counter_map[i].index] =
                        (double) counter_result;
                }
            }
        }
    }

    timer_start(&(results->startTime));
}

#define READ_END_MEM_CHANNEL(channel, reg, cid)                      \
    counter_result = pci_tread(socket_fd, cpu_id, channel, reg##_A); \
    counter_result = (counter_result<<32) +                          \
    pci_tread(socket_fd, cpu_id, channel, reg##_B);                  \
    results->PMcounters[cid] += (double) counter_result - results->StartPMcounters[cid]


/* TODO: Readout hash at the end. Compute result at the end of the function to
 * keep overhead in region low */

void likwid_markerStopRegion(const char* regionTag)
{
    if (! likwid_init)
    {
        return;
    }

    TimerData timestamp;
    timer_stop(&timestamp);
    int cpu_id = likwid_getProcessorId();
    uint64_t res;
    int socket_fd = thread_socketFD[cpu_id];
    double PMcounters[NUM_PMC];

    /* Core specific counters */
    for ( int i=0; i<perfmon_numCountersCore; i++ )
    {
        bitMask_test(res,counterMask,i);
        if ( res )
        {
            if (perfmon_counter_map[i].type != THERMAL)
            {
                PMcounters[i] = (double) msr_tread(
                        socket_fd,
                        cpu_id,
                        perfmon_counter_map[i].counterRegister);
            }
            else
            {
                PMcounters[i] = (double) thermal_read(cpu_id);
            }
        }
    }

    /* Uncore specific counters */
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        /* Conventional Uncore counters */
        for ( int i=perfmon_numCountersCore; i<perfmon_numCountersUncore; i++ )
        {
            bitMask_test(res,counterMask,i);
            if ( res )
            {
                if (perfmon_counter_map[i].type != POWER)
                {
                    PMcounters[i] = (double) msr_tread(
                            socket_fd,
                            cpu_id,
                            perfmon_counter_map[i].counterRegister);
                }
                else
                {
                    PMcounters[i] = (double) power_tread(
                            socket_fd,
                            cpu_id,
                            perfmon_counter_map[i].counterRegister);
                }
            }
        }

        /* PCI Uncore counters */
        if ( hasPCICounters && (accessClient_mode != DAEMON_AM_DIRECT) )
        {
            for ( int i=perfmon_numCountersUncore; i<perfmon_numCounters; i++ )
            {
                bitMask_test(res,counterMask,i);
                if ( res )
                {
                    uint64_t counter_result =
                        pci_tread(
                                socket_fd,
                                cpu_id,
                                perfmon_counter_map[i].device,
                                perfmon_counter_map[i].counterRegister);

                    counter_result = (counter_result<<32) +
                        pci_tread(
                                socket_fd,
                                cpu_id,
                                perfmon_counter_map[i].device,
                                perfmon_counter_map[i].counterRegister2);

                    PMcounters[i] = (double) counter_result;
                }
            }
        }
    }

    bstring tag = bfromcstralloc(100, regionTag);
    LikwidThreadResults* results;
    hashTable_get(tag, &results);
    results->startTime.stop = timestamp.stop;
    results->time += timer_print(&(results->startTime));
    bdestroy(tag);

    /* Accumulate the results */
    /* Core counters */
    for ( int i=0; i<perfmon_numCountersCore; i++ )
    {
        bitMask_test(res,counterMask,i);
        if ( res )
        {
            if (perfmon_counter_map[i].type != THERMAL)
            {
                results->PMcounters[i] += (PMcounters[i] - results->StartPMcounters[i]);
            }
            else
            {
                results->PMcounters[i] = PMcounters[i];
            }
        }
    }

    /* Uncore counters */
    if ((socket_lock[affinity_core2node_lookup[cpu_id]] == cpu_id))
    {
        for ( int i=perfmon_numCountersCore; i<perfmon_numCounters; i++ )
        {
            bitMask_test(res,counterMask,i);
            if ( res )
            {
                if ( perfmon_counter_map[i].type == POWER )
                {
                    results->PMcounters[i] += power_info.energyUnit *
                        (PMcounters[i] - results->StartPMcounters[i]);
                }
                else
                {
                    results->PMcounters[i] += (PMcounters[i] - results->StartPMcounters[i]);
                }
            }
        }
    }
}

int  likwid_getProcessorId()
{
    cpu_set_t  cpu_set;
    CPU_ZERO(&cpu_set);
    sched_getaffinity(gettid(),sizeof(cpu_set_t), &cpu_set);

    return getProcessorID(&cpu_set);
}

#ifdef HAS_SCHEDAFFINITY
int  likwid_pinThread(int processorId)
{
    int ret;
    cpu_set_t cpuset;
    pthread_t thread;

    thread = pthread_self();
    CPU_ZERO(&cpuset);
    CPU_SET(processorId, &cpuset);
    ret = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);

    if (ret != 0)
    {
        ERROR;
        return FALSE;
    }

    return TRUE;
}
#endif


int  likwid_pinProcess(int processorId)
{
    int ret;
    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    CPU_SET(processorId, &cpuset);
    ret = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

    if (ret < 0)
    {
        ERROR;
        return FALSE;
    }

    return TRUE;
}


