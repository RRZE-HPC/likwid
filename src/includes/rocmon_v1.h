/*
 * =======================================================================================
 *
 *      Filename:  rocmon_v1.h
 *
 *      Description:  Header File of rocmon module for ROCm < 6.2.
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
#ifndef LIKWID_ROCMON_V1_H
#define LIKWID_ROCMON_V1_H

int rocmon_v1_init(int numGpus, const int* gpuIds);
void rocmon_v1_finalize(void);
int rocmon_v1_addEventSet(const char* eventString, int* gid);
int rocmon_v1_setupCounters(int gid);
int rocmon_v1_startCounters(void);
int rocmon_v1_stopCounters(void);
int rocmon_v1_readCounters(void);
double rocmon_v1_getResult(int gpuIdx, int groupId, int eventId);
double rocmon_v1_getLastResult(int gpuIdx, int groupId, int eventId);
int rocmon_v1_getEventsOfGpu(int gpuIdx, EventList_rocm_t* list);
void rocmon_v1_freeEventsOfGpu(EventList_rocm_t list);
int rocmon_v1_switchActiveGroup(int newGroupId);
int rocmon_v1_getNumberOfGroups(void);
int rocmon_v1_getIdOfActiveGroup(void);
int rocmon_v1_getNumberOfGPUs(void);
int rocmon_v1_getNumberOfEvents(int groupId);
int rocmon_v1_getNumberOfMetrics(int groupId);
double rocmon_v1_getTimeOfGroup(int groupId);
double rocmon_v1_getLastTimeOfGroup(int groupId);
double rocmon_v1_getTimeToLastReadOfGroup(int groupId);
char* rocmon_v1_getEventName(int groupId, int eventId);
char* rocmon_v1_getCounterName(int groupId, int eventId);
char* rocmon_v1_getMetricName(int groupId, int metricId);
char* rocmon_v1_getGroupName(int groupId);
char* rocmon_v1_getGroupInfoShort(int groupId);
char* rocmon_v1_getGroupInfoLong(int groupId);
int rocmon_v1_getGroups(char*** groups, char*** shortinfos, char*** longinfos);
int rocmon_v1_returnGroups(int nrgroups, char** groups, char** shortinfos, char** longinfos);


#endif /* LIKWID_ROCMON_V1_H */

