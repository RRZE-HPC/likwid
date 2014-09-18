/*
 * =======================================================================================
 *
 *      Filename:  likwid.h
 *
 *      Description:  Header File of likwid marker API
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

#ifndef LIKWID_H
#define LIKWID_H

#include <types.h>
#include <topology.h>
#include <error.h>
#include <bstrlib.h>


#ifdef LIKWID_PERFMON
#define LIKWID_MARKER_INIT likwid_markerInit()
#define LIKWID_MARKER_THREADINIT likwid_markerThreadInit()
#define LIKWID_MARKER_START(reg) likwid_markerStartRegion(reg)
#define LIKWID_MARKER_STOP(reg) likwid_markerStopRegion(reg)
#define LIKWID_MARKER_CLOSE likwid_markerClose()
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_START(reg)
#define LIKWID_MARKER_STOP(reg)
#define LIKWID_MARKER_CLOSE
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* marker API routines */
extern void likwid_markerInit(void);
extern void likwid_markerThreadInit(void);
extern void likwid_markerClose(void);
extern int likwid_markerStartRegion(const char* regionTag);
extern int likwid_markerStopRegion(const char* regionTag);

/* utility routines */
extern int  likwid_getProcessorId();
extern int  likwid_pinProcess(int processorId);
extern int  likwid_pinThread(int processorId);

/* configuration routines */
extern int init_configuration(void);
extern int destroy_configuration(void);
extern Configuration_t get_configuration(void);

/* topology routines */
extern int topology_init(void);
extern CpuTopology_t get_cpuTopology(void);
extern CpuInfo_t get_cpuInfo(void);
extern void topology_finalize(void);

/* numa routines */
extern int numa_init(void);
extern NumaTopology_t get_numaTopology(void);
extern void numa_setInterleaved(int* processorList, int numberOfProcessors);
extern void numa_membind(void* ptr, size_t size, int domainId);
extern void numa_finalize(void);
extern int likwid_getNumberOfNodes(void);

/* affinity routines */
extern void affinity_init();
extern AffinityDomains_t get_affinityDomains(void);
extern void affinity_pinProcess(int processorId);
extern void affinity_pinThread(int processorId);
extern void affinity_finalize();

/* accessClient routines */
extern void accessClient_setaccessmode(int mode);
extern void accessClient_init(int* socket_fd);
extern void accessClient_finalize(int socket_fd);

/* perfmon routines */
extern int perfmon_addEventSet(char* eventCString);
extern int perfmon_setupCounters(int groupId);
extern int perfmon_startCounters(void);
extern int perfmon_stopCounters(void);
extern int perfmon_readCounters(void);
extern int perfmon_readCountersCpu(int cpu_id);
extern int perfmon_initThread(int thread_id, int cpu_id);
extern int perfmon_init(int nrThreads, int threadsToCpu[]);
extern void perfmon_init_maps(void);
extern void perfmon_finalize(void);
extern int perfmon_switchActiveGroup(int new_group);
extern int perfmon_accessClientInit(void);
extern double perfmon_getResult(int groupId, int eventId, int threadId);
extern int perfmon_getNumberOfGroups(void);
extern int perfmon_getNumberOfEvents(int groupId);
extern double perfmon_getTimeOfGroup(int groupId);
extern int perfmon_getIdOfActiveGroup(void);
extern int perfmon_getNumberOfThreads(void);

/* power routines */
extern int power_init(int cpuId);
extern PowerInfo_t get_powerInfo(void);
extern int power_read(int cpuId, uint64_t reg, uint32_t *data);
extern int power_tread(int socket_fd, int cpuId, uint64_t reg, uint32_t *data);
extern int power_start(PowerData_t data, int cpuId, PowerType type);
extern int power_stop(PowerData_t data, int cpuId, PowerType type);

/* thermal routines */
extern void thermal_init(int cpuId);
extern int thermal_read(int cpuId, uint32_t *data);
extern int thermal_tread(int socket_fd, int cpuId, uint32_t *data);

/* daemon routines */
extern int daemon_start(uint64_t duration);
extern int daemon_stop(int sig);

/* memsweep routines */
extern void memsweep_domain(int domainId);
extern void memsweep_threadGroup(int* processorList, int numberOfProcessors);

#ifdef __cplusplus
}
#endif

#endif /*LIKWID_H*/
