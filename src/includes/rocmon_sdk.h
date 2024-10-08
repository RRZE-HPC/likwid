/*
 * =======================================================================================
 *
 *      Filename:  rocmon_sdk.h
 *
 *      Description:  Header File of rocmon module for ROCm >= 6.2.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tg), thomas.gruber@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2019 RRZE, University Erlangen-Nuremberg
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
#ifndef LIKWID_ROCMON_SDK_H
#define LIKWID_ROCMON_SDK_H

int rocmon_sdk_init(int numGpus, const int* gpuIds);
void rocmon_sdk_finalize(void);
int rocmon_sdk_addEventSet(const char* eventString, int* gid);
int rocmon_sdk_setupCounters(int gid);
int rocmon_sdk_startCounters(void);
int rocmon_sdk_stopCounters(void);
int rocmon_sdk_readCounters(void);
double rocmon_sdk_getResult(int gpuIdx, int groupId, int eventId);
double rocmon_sdk_getLastResult(int gpuIdx, int groupId, int eventId);
int rocmon_sdk_getEventsOfGpu(int gpuIdx, EventList_rocm_t* list);
void rocmon_sdk_freeEventsOfGpu(EventList_rocm_t list);
int rocmon_sdk_switchActiveGroup(int newGroupId);
int rocmon_sdk_getNumberOfGroups(void);
int rocmon_sdk_getIdOfActiveGroup(void);
int rocmon_sdk_getNumberOfGPUs(void);
int rocmon_sdk_getNumberOfEvents(int groupId);
int rocmon_sdk_getNumberOfMetrics(int groupId);
double rocmon_sdk_getTimeOfGroup(int groupId);
double rocmon_sdk_getLastTimeOfGroup(int groupId);
double rocmon_sdk_getTimeToLastReadOfGroup(int groupId);
char* rocmon_sdk_getEventName(int groupId, int eventId);
char* rocmon_sdk_getCounterName(int groupId, int eventId);
char* rocmon_sdk_getMetricName(int groupId, int metricId);
char* rocmon_sdk_getGroupName(int groupId);
char* rocmon_sdk_getGroupInfoShort(int groupId);
char* rocmon_sdk_getGroupInfoLong(int groupId);
int rocmon_sdk_getGroups(char*** groups, char*** shortinfos, char*** longinfos);
int rocmon_sdk_returnGroups(int nrgroups, char** groups, char** shortinfos, char** longinfos);


#endif /* LIKWID_ROCMON_SDK_H */

