 /* =======================================================================================
 *
 *      Filename:  rocmon.c
 *
 *      Description:  Main implementation of the performance monitoring module
 *                    for AMD GPUs
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tg), thomas.roehl@googlemail.com
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
#ifdef LIKWID_WITH_ROCMON

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include <types.h>
#include <sys/types.h>
#include <inttypes.h>

#include <likwid.h>
#include <bstrlib.h>
#include <error.h>
#include <dlfcn.h>

#include <likwid.h>
#ifdef HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE
#undef HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE
#endif
#ifndef LIKWID_ROCPROF_SDK
#include <rocmon_v1_types.h>
#include <rocmon_v1.h>
#else
#include <rocmon_sdk_types.h>
#include <rocmon_sdk.h>
#endif

#include <amd_smi/amdsmi.h>



void
rocmon_finalize(void)
{
#ifndef LIKWID_ROCPROF_SDK
    rocmon_v1_finalize();
#else
    rocmon_sdk_finalize();
#endif
    return;
}

int
rocmon_init(int numGpus, const int* gpuIds)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_init(numGpus, gpuIds);
#else
    return rocmon_sdk_init(numGpus, gpuIds);
#endif
}

int
rocmon_addEventSet(const char* eventString, int* gid)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_addEventSet(eventString, gid);
#else
    return rocmon_sdk_addEventSet(eventString, gid);
#endif
}



int
rocmon_setupCounters(int gid)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_setupCounters(gid);
#else
    return rocmon_sdk_setupCounters(gid);
#endif
}



int
rocmon_startCounters(void)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_startCounters();
#else
    return rocmon_sdk_startCounters();
#endif
}



int
rocmon_stopCounters(void)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_stopCounters();
#else
    return rocmon_sdk_stopCounters();
#endif
}


int
rocmon_readCounters(void)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_readCounters();
#else
    return rocmon_sdk_readCounters();
#endif
}


double
rocmon_getResult(int gpuIdx, int groupId, int eventId)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getResult(gpuIdx, groupId, eventId);
#else
    return rocmon_sdk_getResult(gpuIdx, groupId, eventId);
#endif
}


// TODO: multiple groups
double
rocmon_getLastResult(int gpuIdx, int groupId, int eventId)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getLastResult(gpuIdx, groupId, eventId);
#else
    return rocmon_sdk_getLastResult(gpuIdx, groupId, eventId);
#endif
}


int
rocmon_getEventsOfGpu(int gpuIdx, EventList_rocm_t* list)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getEventsOfGpu(gpuIdx, list);
#else
    return rocmon_sdk_getEventsOfGpu(gpuIdx, list);
#endif
}

void
rocmon_freeEventsOfGpu(EventList_rocm_t list)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_freeEventsOfGpu(list);
#else
    return rocmon_sdk_freeEventsOfGpu(list);
#endif
}


int
rocmon_switchActiveGroup(int newGroupId)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_switchActiveGroup(newGroupId);
#else
    return rocmon_sdk_switchActiveGroup(newGroupId);
#endif
}


int
rocmon_getNumberOfGroups(void)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getNumberOfGroups();
#else
    return rocmon_sdk_getNumberOfGroups();
#endif
}


int
rocmon_getIdOfActiveGroup(void)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getIdOfActiveGroup();
#else
    return rocmon_sdk_getIdOfActiveGroup();
#endif
}


int
rocmon_getNumberOfGPUs(void)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getNumberOfGPUs();
#else
    return rocmon_sdk_getNumberOfGPUs();
#endif
}


int
rocmon_getNumberOfEvents(int groupId)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getNumberOfEvents(groupId);
#else
    return rocmon_sdk_getNumberOfEvents(groupId);
#endif
}


int
rocmon_getNumberOfMetrics(int groupId)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getNumberOfMetrics(groupId);
#else
    return rocmon_sdk_getNumberOfMetrics(groupId);
#endif
}


double
rocmon_getTimeOfGroup(int groupId)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getTimeOfGroup(groupId);
#else
    return rocmon_sdk_getTimeOfGroup(groupId);
#endif
}


double
rocmon_getLastTimeOfGroup(int groupId)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getLastTimeOfGroup(groupId);
#else
    return rocmon_sdk_getLastTimeOfGroup(groupId);
#endif
}


double
rocmon_getTimeToLastReadOfGroup(int groupId)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getTimeToLastReadOfGroup(groupId);
#else
    return rocmon_sdk_getTimeToLastReadOfGroup(groupId);
#endif
}


char*
rocmon_getEventName(int groupId, int eventId)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getEventName(groupId, eventId);
#else
    return rocmon_sdk_getEventName(groupId, eventId);
#endif
}


char*
rocmon_getCounterName(int groupId, int eventId)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getCounterName(groupId, eventId);
#else
    return rocmon_sdk_getCounterName(groupId, eventId);
#endif
}


char*
rocmon_getMetricName(int groupId, int metricId)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getMetricName(groupId, metricId);
#else
    return rocmon_sdk_getMetricName(groupId, metricId);
#endif
}


char* 
rocmon_getGroupName(int groupId)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getGroupName(groupId);
#else
    return rocmon_sdk_getGroupName(groupId);
#endif
}


char*
rocmon_getGroupInfoShort(int groupId)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getGroupInfoShort(groupId);
#else
    return rocmon_sdk_getGroupInfoShort(groupId);
#endif
}


char*
rocmon_getGroupInfoLong(int groupId)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getGroupInfoLong(groupId);
#else
    return rocmon_sdk_getGroupInfoLong(groupId);
#endif
}


int
rocmon_getGroups(char*** groups, char*** shortinfos, char*** longinfos)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_getGroups(groups, shortinfos, longinfos);
#else
    return rocmon_sdk_getGroups(groups, shortinfos, longinfos);
#endif
}


int
rocmon_returnGroups(int nrgroups, char** groups, char** shortinfos, char** longinfos)
{
#ifndef LIKWID_ROCPROF_SDK
    return rocmon_v1_returnGroups(nrgroups, groups, shortinfos, longinfos);
#else
    return rocmon_sdk_returnGroups(nrgroups, groups, shortinfos, longinfos);
#endif
}

void rocmon_setVerbosity(int level)
{
    if (level >= DEBUGLEV_ONLY_ERROR && level <= DEBUGLEV_DEVELOP)
    {
        likwid_rocmon_verbosity = level;
    }
}

#endif /* LIKWID_WITH_ROCMON */
