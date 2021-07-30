/*
 * =======================================================================================
 *
 *      Filename:  nvmon_nvml.h
 *
 *      Description:  Header File of nvmon module (NVML backend).
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
#ifndef LIKWID_NVMON_NVML_H
#define LIKWID_NVMON_NVML_H

#include <nvmon_types.h>

int nvml_init();
void nvml_finalize();
int nvml_getEventsOfGpu(int gpuId, NvmonEventList_t* output);
void nvml_returnEventsOfGpu(NvmonEventList_t list);
int nvml_addEventSet(char** events, int numEvents);
int nvml_setupCounters(int gid);
int nvml_startCounters();
int nvml_stopCounters();
int nvml_readCounters();
int nvml_getNumberOfEvents(int groupId);
int nvml_getResult(int gpuIdx, int groupId, int eventId);
int nvml_getLastResult(int gpuIdx, int groupId, int eventId);
double nvml_getTimeOfGroup(int groupId);
double nvml_getLastTimeOfGroup(int groupId);

#endif /* LIKWID_NVMON_NVML_H */
