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

/** \addtogroup MarkerAPI Marker API module
*  @{
*/
/*!
\def LIKWID_MARKER_INIT
Shortcut for likwid_markerInit() if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_MARKER_THREADINIT
Shortcut for likwid_markerThreadInit() if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_MARKER_START(reg)
Shortcut for likwid_markerStartRegion() with \a reg as regionTag if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_MARKER_STOP(reg)
Shortcut for likwid_markerStopRegion() with \a reg as regionTag if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_MARKER_CLOSE
Shortcut for likwid_markerClose() if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/** @}*/

#ifdef LIKWID_PERFMON
#define LIKWID_MARKER_INIT likwid_markerInit()
#define LIKWID_MARKER_THREADINIT likwid_markerThreadInit()
#define LIKWID_MARKER_SWITCH likwid_markerNextGroup()
#define LIKWID_MARKER_START(reg) likwid_markerStartRegion(reg)
#define LIKWID_MARKER_STOP(reg) likwid_markerStopRegion(reg)
#define LIKWID_MARKER_CLOSE likwid_markerClose()
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_START(reg)
#define LIKWID_MARKER_STOP(reg)
#define LIKWID_MARKER_CLOSE
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
################################################################################
# Marker API related functions
################################################################################
*/
/** \addtogroup MarkerAPI Marker API module
*  @{
*/
/*! \brief Initialize LIKWID's marker API

Must be called in serial region of the application to set up basic data structures
of LIKWID. 
Reads environment variables: 
- LIKWID_MODE (access mode)
- LIKWID_MASK (event bitmask)
- LIKWID_EVENTS (event string)
- LIKWID_THREADS (cpu list separated by ,)
- LIKWID_GROUPS (amount of groups)
*/
extern void likwid_markerInit(void);
/*! \brief Initialize LIKWID's marker API for the current thread

Must be called in parallel region of the application to set up basic data structures
of LIKWID. Before you can call likwid_markerThreadInit() you have to call likwid_markerInit().

*/
extern void likwid_markerThreadInit(void);
/*! \brief Select next group to measure

Must be called in parallel region of the application to switch group on every CPU.
*/
extern void likwid_markerNextGroup(void);
/*! \brief Close LIKWID's marker API

Must be called in serial region of the application. It gathers all data of regions and
writes them out to a file (filepath in env variable LIKWID_FILEPATH).
*/
extern void likwid_markerClose(void);
/*! \brief Start a measurement region

Reads the values of all configured counters and saves the results under the name given
in regionTag.
@param regionTag [in] Store data using this string
@return Error code of start operation
*/
extern int likwid_markerStartRegion(const char* regionTag);
/*! \brief Stop a measurement region

Reads the values of all configured counters and saves the results under the name given
in regionTag. The measurement data of the stopped region gets summed up in global region counters.
@param regionTag [in] Store data using this string
@return Error code of stop operation
*/
extern int likwid_markerStopRegion(const char* regionTag);

/* utility routines */
/*! \brief Get CPU ID of the current process/thread

Returns the ID of the CPU the current process or thread is running on.
@return current CPU ID
*/
extern int  likwid_getProcessorId();
/*! \brief Pin the current process to given CPU

Pin the current process to the given CPU ID. The process cannot be scheduled to 
another CPU after pinning but the pinning can be changed anytime with this function.
@param [in] processorId CPU ID to pin the current process to
@return error code (1 for success, 0 for error)
*/
extern int  likwid_pinProcess(int processorId);
/*! \brief Pin the current thread to given CPU

Pin the current thread to the given CPU ID. The thread cannot be scheduled to 
another CPU after pinning but the pinning can be changed anytime with this function
@param [in] processorId CPU ID to pin the current thread to
@return error code (1 for success, 0 for error)
*/
extern int  likwid_pinThread(int processorId);
/** @}*/
/*
################################################################################
# Config file related functions
################################################################################
*/
/** \addtogroup Config Config file module
*  @{
*/
/*! \brief Read the config file of LIKWID, if it exists

Search for LIKWID config file and read the values in
Currently the paths /usr/local/etc/likwid.cfg, /etc/likwid.cfg and ./likwid.cfg 
are checked
@return error code (0 for success, -EFAULT if no file can be found)
*/
extern int init_configuration(void);
/*! \brief Destroy the config structure

Destroys the current config structure and frees all allocated memory for path names
@return error code (0 for success, -EFAULT if config structure not initialized)
*/
extern int destroy_configuration(void);


/*! \brief Retrieve the config structure

Get the initialized configuration
\sa Configuration_t
@return Configuration_t (pointer to internal Configuration structure)
*/
extern Configuration_t get_configuration(void);
/** @}*/
/* 
################################################################################
# CPU topology related functions
################################################################################
*/
/** \addtogroup CPUTopology CPU information module
*  @{
*/
/*! \brief Initialize topology information

CpuInfo_t and CpuTopology_t are initialized by either HWLOC, CPUID/ProcFS or topology file if present
\sa CpuInfo_t and CpuTopology_t
@return always 0
*/
extern int topology_init(void);
/*! \brief Retrieve CPU topology of the current machine

\sa CpuTopology_t
@return CpuTopology_t (pointer to internal cpuid_topology structure)
*/
extern CpuTopology_t get_cpuTopology(void);
/*! \brief Retrieve CPU information of the current machine

Get the previously initialized CPU info structure containing number of CPUs/Threads
\sa CpuInfo_t
@return CpuInfo_t (pointer to internal cpuid_info structure)
*/
extern CpuInfo_t get_cpuInfo(void);
/*! \brief Destroy topology structures CpuInfo_t and CpuTopology_t.

Retrieved pointers to the structures are not valid anymore after this function call
\sa CpuInfo_t and CpuTopology_t
*/
extern void topology_finalize(void);
/** @}*/
/* 
################################################################################
# NUMA related functions
################################################################################
*/
/** \addtogroup NumaTopology NUMA memory topology
 *  @{
 */
/*! \brief Initialize NUMA information

Initialize NUMA information NumaTopology_t using either HWLOC or CPUID/ProcFS. If
a topology config file is present it is read at topology_init() and fills \a NumaTopology_t
\sa NumaTopology_t
@return error code (0 for success, -1 if initialization failed)
*/
extern int numa_init(void);
/*! \brief Retrieve NUMA information of the current machine

Get the previously initialized NUMA info structure
\sa NumaTopology_t
@return NumaTopology_t (pointer to internal numa_info structure)
*/
extern NumaTopology_t get_numaTopology(void);
/*! \brief Set memory allocation policy to interleaved

Set the memory allocation policy to interleaved for given list of CPUs
@param [in] processorList List of processors
@param [in] numberOfProcessors Length of processor list
*/
extern void numa_setInterleaved(int* processorList, int numberOfProcessors);
/*! \brief Allocate memory from a specific specific NUMA node
@param [in,out] ptr Start pointer of memory
@param [in] size Size for the allocation
@param [in] domainId ID of NUMA node for the allocation
*/
extern void numa_membind(void* ptr, size_t size, int domainId);
/*! \brief Destroy NUMA information structure

Destroys the NUMA information structure NumaTopology_t. Retrieved pointers
to the structures are not valid anymore after this function call
\sa NumaTopology_t
*/
extern void numa_finalize(void);
/*! \brief Retrieve the number of NUMA nodes

Returns the number of NUMA nodes of the current machine. Can also be read out of
NumaTopology_t
\sa NumaTopology_t
@return Number of NUMA nodes
*/
extern int likwid_getNumberOfNodes(void);
/** @}*/
/* 
################################################################################
# Affinity domains related functions
################################################################################
*/
/** \addtogroup AffinityDomains
 *  @{
 */
/*! \brief Initialize affinity information

Initialize affinity information AffinityDomains_t using the data of the structures
\a CpuInfo_t, CpuTopology_t and NumaTopology_t
\sa AffinityDomains_t
*/
extern void affinity_init();
/*! \brief Retrieve affinity structure

Get the previously initialized affinity info structure
\sa AffinityDomains_t
@return AffinityDomains_t (pointer to internal affinityDomains structure)
*/
extern AffinityDomains_t get_affinityDomains(void);
/*! \brief Pin process to a CPU

Pin process to a CPU. Duplicate of likwid_pinProcess()
@param [in] processorId CPU ID for pinning
*/
extern void affinity_pinProcess(int processorId);
/*! \brief Pin thread to a CPU

Pin thread to a CPU. Duplicate of likwid_pinThread()
@param [in] processorId CPU ID for pinning
*/
extern void affinity_pinThread(int processorId);
/*! \brief Return the CPU ID where the current process runs.

@return CPU ID
*/
int affinity_processGetProcessorId();
/*! \brief Destroy affinity information structure

Destroys the affinity information structure AffinityDomains_t. Retrieved pointers
to the structures are not valid anymore after this function call
\sa AffinityDomains_t
*/
extern void affinity_finalize();
/** @}*/
/* 
################################################################################
# Access client related functions
################################################################################
*/
/** \addtogroup AccessClient Access client module
 *  @{
 */
/*! \brief Set accessClient mode

Sets the mode how the MSR and PCI registers should be accessed. 0 for direct access (propably root priviledges required) and 1 for accesses through the access daemon. It must be called before accessClient_init()
@param [in] mode (0=direct, 1=daemon)
*/
extern void accessClient_setaccessmode(int mode);
/*! \brief Initialize socket for communication with the access daemon

Initializes the file descriptor by starting and connecting to a new access daemon.
@param [out] socket_fd Pointer to socket file descriptor
*/
extern void accessClient_init(int* socket_fd);
/*! \brief Destroy socket for communication with the access daemon

Destroys the file descriptor by disconnecting and shutting down the access daemon
@param [in] socket_fd socket file descriptor
*/
extern void accessClient_finalize(int socket_fd);
/** @}*/
/* 
################################################################################
# Performance monitoring related functions
################################################################################
*/
/** \addtogroup PerfMon Performance monitoring module
 *  @{
 */
/*! \brief Initialize performance monitoring facility

Initialize the performance monitoring feature by creating basic data structures.
The access mode must already be set when calling perfmon_init()
@param [in] nrThreads Amount of threads
@param [in] threadsToCpu List of CPUs
@return error code (0 on success, -ERRORCODE on failure)
*/
extern int perfmon_init(int nrThreads, int threadsToCpu[]);
extern void perfmon_accessClientInit(void);
/*! \brief Initialize performance monitoring maps

Initialize the performance monitoring maps for counters, events and Uncore boxes#
for the current architecture. topology_init() and numa_init() must be called before calling
perfmon_init_maps()
\sa RegisterMap list, PerfmonEvent list and BoxMap list
*/
extern void perfmon_init_maps(void);
/*! \brief Add an event string to LIKWID

A event string looks like <eventname>:<countername>(:<option1>:<options2>),...
The eventname, countername and options are checked if they are available.
@param [in] eventCString Event string
@return Returns the ID of the new eventSet
*/
extern int perfmon_addEventSet(char* eventCString);
/*! \brief Setup all performance monitoring counters of an eventSet

A event string looks like <eventname>:<countername>(:<option1>:<options2>),...
The eventname, countername and options are checked if they are available.
@param [in] groupId (returned from perfmon_addEventSet()
@return error code (-ENOENT if groupId is invalid and -1 if the counters of one CPU cannot be set up)
*/
extern int perfmon_setupCounters(int groupId);
/*! \brief Start performance monitoring counters

Start the counters that have been previously set up by perfmon_setupCounters().
The counter registered are zeroed before enabling the counters
@return 0 on success and -(thread_id+1) for error
*/
extern int perfmon_startCounters(void);
/*! \brief Stop performance monitoring counters 

Stop the counters that have been previously started by perfmon_startCounters().
All config registers get zeroed before reading the counter register.
@return 0 on success and -(thread_id+1) for error
*/
extern int perfmon_stopCounters(void);
/*! \brief Read the performance monitoring counters on all CPUs

Read the counters that have been previously started by perfmon_startCounters().
The counters are stopped directly to avoid interference of LIKWID with the measured
code. Before returning, the counters are started again.
@return 0 on success and -(thread_id+1) for error
*/
extern int perfmon_readCounters(void);
/*! \brief Read the performance monitoring counters on one CPU

Read the counters that have been previously started by perfmon_startCounters().
The counters are stopped directly to avoid interference of LIKWID with the measured
code. Before returning, the counters are started again. Only one CPU is read.
@param [in] cpu_id CPU ID of the CPU that should be read
@return 0 on success and -(thread_id+1) for error
*/
extern int perfmon_readCountersCpu(int cpu_id);
/*! \brief Switch the active eventSet to a new one

Stops the currently running counters, switches the eventSet by setting up the
counters and start the counters.
@param [in] new_group ID of group that should be switched to.
@return 0 on success and -(thread_id+1) for error
*/
extern int perfmon_switchActiveGroup(int new_group);
/*! \brief Close the perfomance monitoring facility of LIKWID

Deallocates all internal data that is used during performance monitoring. Also
the counter values are not accessible after this function.
*/
extern void perfmon_finalize(void);
/*! \brief Get the results of the specified group, counter and thread

Get the result of the last measurement cycle. The function takes care of happened
overflows and if the counter values need to be calculated with multipliers.
@param [in] groupId ID of the group that should be read
@param [in] eventId ID of the event that should be read
@param [in] threadId ID of the thread/cpu that should be read
@return The counter result
*/
extern double perfmon_getResult(int groupId, int eventId, int threadId);
/*! \brief Get the number of configured event groups

@return Number of groups
*/
extern int perfmon_getNumberOfGroups(void);
/*! \brief Get the number of configured eventSets in group

@param [in] groupId ID of group
@return Number of eventSets
*/
extern int perfmon_getNumberOfEvents(int groupId);
/*! \brief Get the measurement time a group

@param [in] groupId ID of group
@return Time in seconds the event group was measured
*/
extern double perfmon_getTimeOfGroup(int groupId);
/*! \brief Get the ID of the currently set up event group

@return Number of active group
*/
extern int perfmon_getIdOfActiveGroup(void);
/*! \brief Get the number of threads specified at perfmon_init()

@return Number of threads
*/
extern int perfmon_getNumberOfThreads(void);
/** @}*/

/* 
################################################################################
# Power measurements related functions
################################################################################
*/
/** \addtogroup PowerMon Power and Energy monitoring module
 *  @{
 */
/*! \brief Initialize power measurements on specific CPU

Additionally, it reads basic information about the power measurements like 
minimal measurement time.
@param [in] cpuId Initialize power facility for this CPU
@return error code
*/
extern int power_init(int cpuId);
/*! \brief Get a pointer to the power facility information

@return PowerInfo_t pointer
\sa PowerInfo_t
*/
extern PowerInfo_t get_powerInfo(void);
/*! \brief Read the current power value

@param [in] cpuId Read power facility for this CPU
@param [in] reg Power register
@param [out] data Power data
*/
extern int power_read(int cpuId, uint64_t reg, uint32_t *data);
/*! \brief Read the current power value using a specific communication socket

@param [in] socket_fd Communication socket for the read operation
@param [in] cpuId Read power facility for this CPU
@param [in] reg Power register
@param [out] data Power data
*/
extern int power_tread(int socket_fd, int cpuId, uint64_t reg, uint32_t *data);
/*! \brief Start power measurements

@param [in,out] data Data structure holding start and stop values for power measurements
@param [in] cpuId Start power facility for this CPU
@param [in] type Which type should be measured
@return error code
*/
extern int power_start(PowerData_t data, int cpuId, PowerType type);
/*! \brief Stop power measurements

@param [in,out] data Data structure holding start and stop values for power measurements
@param [in] cpuId Start power facility for this CPU
@param [in] type Which type should be measured
@return error code
*/
extern int power_stop(PowerData_t data, int cpuId, PowerType type);
/*! \brief Print power measurements gathered by power_start() and power_stop()

@param [in] data Data structure holding start and stop values for power measurements
@return Consumed energy in Joules
*/
extern double power_printEnergy(PowerData* data);
/** @}*/

/* 
################################################################################
# Thermal measurements related functions
################################################################################
*/
/** \addtogroup ThermalMon Thermal monitoring module
 *  @{
 */
/*! \brief Initialize thermal measurements on specific CPU

@param [in] cpuId Initialize thermal facility for this CPU
*/
extern void thermal_init(int cpuId);
/*! \brief Read the current thermal value

@param [in] cpuId Read thermal facility for this CPU
@param [out] data Thermal data
*/
extern int thermal_read(int cpuId, uint32_t *data);
/*! \brief Read the current thermal value using a specific communication socket

@param [in] socket_fd Communication socket for the read operation
@param [in] cpuId Read thermal facility for this CPU
@param [out] data Thermal data
*/
extern int thermal_tread(int socket_fd, int cpuId, uint32_t *data);
/** @}*/

/* 
################################################################################
# Timeline daemon related functions
################################################################################
*/
/** \addtogroup Daemon Timeline daemon module
 *  @{
 */
/*! \brief Start timeline daemon

Starts the timeline daemon which reads and prints the counter values after each \a duration time
@param [in] duration Time interval in ns
@return 0 on success and -EFAULT if counters cannot be started
*/
extern int daemon_start(uint64_t duration);
/*! \brief Stop timeline daemon

Stop the timeline daemon using the signal \a sig
@param [in] sig Signal code to kill the daemon (see signal.h for signal codes)
@return 0 on success and the negative error code at failure
*/
extern int daemon_stop(int sig);
/** @}*/

/* 
################################################################################
# Memory sweeping related functions
################################################################################
*/
/** \addtogroup MemSweep Memory sweeping module
 *  @{
 */
/*! \brief Sweeping the memory of a NUMA node

Sweeps (zeros) the memory of NUMA node with ID \a domainId
@param [in] domainId NUMA node ID
*/
extern void memsweep_domain(int domainId);
/*! \brief Sweeping the memory of all NUMA nodes covered by CPU list

Sweeps (zeros) the memory of all NUMA nodes containing the CPUs in \a processorList
@param [in] processorList List of CPU IDs
@param [in] numberOfProcessors Number of CPUs in list
*/
extern void memsweep_threadGroup(int* processorList, int numberOfProcessors);
/** @}*/
#ifdef __cplusplus
}
#endif

#endif /*LIKWID_H*/
