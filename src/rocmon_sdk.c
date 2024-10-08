 /* =======================================================================================
 *
 *      Filename:  rocmon_sdk.c
 *
 *      Description:  Main implementation of the performance monitoring module
 *                    for AMD GPUs with ROCm >= 6.2
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

#include <rocmon_sdk_types.h>

static bool rocmon_initialized = FALSE;
int likwid_rocmon_verbosity = DEBUGLEV_ONLY_ERROR;

int
rocmon_sdk_init(int numGpus, const int* gpuIds)
{
    return 0;
}


void
rocmon_sdk_finalize(void)
{
    return;
}


int
rocmon_sdk_addEventSet(const char* eventString, int* gid)
{
    return 0;
}

int
rocmon_sdk_setupCounters(int gid)
{
    return 0;
}


int
rocmon_sdk_startCounters(void)
{
    return 0;
}

int
rocmon_sdk_stopCounters(void)
{
    return 0;
}


int
rocmon_sdk_readCounters(void)
{
    return 0;
}


double
rocmon_sdk_getResult(int gpuIdx, int groupId, int eventId)
{
    return 0.0;
}


// TODO: multiple groups
double
rocmon_sdk_getLastResult(int gpuIdx, int groupId, int eventId)
{
    return 0.0;
}


int
rocmon_sdk_getEventsOfGpu(int gpuIdx, EventList_rocm_t* list)
{
    return -EINVAL;
}

void
rocmon_sdk_freeEventsOfGpu(EventList_rocm_t list)
{
    return;
}


int
rocmon_sdk_switchActiveGroup(int newGroupId)
{
    return 0;
}


int
rocmon_sdk_getNumberOfGroups(void)
{
    return 0;
}


int
rocmon_sdk_getIdOfActiveGroup(void)
{
    return 0;
}


int
rocmon_sdk_getNumberOfGPUs(void)
{
    return 0;
}


int
rocmon_sdk_getNumberOfEvents(int groupId)
{
    return 0;
}


int
rocmon_sdk_getNumberOfMetrics(int groupId)
{
    return 0;
}


double
rocmon_sdk_getTimeOfGroup(int groupId)
{
    return 0;
}


double
rocmon_sdk_getLastTimeOfGroup(int groupId)
{
    return 0;
}


double
rocmon_sdk_getTimeToLastReadOfGroup(int groupId)
{
    return 0;
}


char*
rocmon_sdk_getEventName(int groupId, int eventId)
{
    return NULL;
}


char*
rocmon_sdk_getCounterName(int groupId, int eventId)
{
    return NULL;
}


char*
rocmon_sdk_getMetricName(int groupId, int metricId)
{
    return NULL;
}


char* 
rocmon_sdk_getGroupName(int groupId)
{
    return NULL;
}


char*
rocmon_sdk_getGroupInfoShort(int groupId)
{
    return NULL;
}


char*
rocmon_sdk_getGroupInfoLong(int groupId)
{
    return NULL;
}


int
rocmon_sdk_getGroups(char*** groups, char*** shortinfos, char*** longinfos)
{
    init_configuration();
    Configuration_t config = get_configuration();

    return perfgroup_getGroups(config->groupPath, "amd_gpu_sdk", groups, shortinfos, longinfos);
}


int
rocmon_sdk_returnGroups(int nrgroups, char** groups, char** shortinfos, char** longinfos)
{
    perfgroup_returnGroups(nrgroups, groups, shortinfos, longinfos);
}


#endif /* LIKWID_WITH_ROCMON */
