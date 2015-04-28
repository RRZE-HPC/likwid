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
 *      Authors:  Thomas Roehl (tr), thomas.roehl@googlemail.com
 *                Jan Treibig (jt), jan.treibig@gmail.com,
 *
 *      Project:  likwid
 *
 *      Copyright (C) 2015 Thomas Roehl
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

#include <stdint.h>
#include <errno.h>
#include <string.h>

#include <bstrlib.h>

#define DEBUGLEV_ONLY_ERROR 0
#define DEBUGLEV_INFO 1
#define DEBUGLEV_DETAIL 2
#define DEBUGLEV_DEVELOP 3

extern int perfmon_verbosity;

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
\def LIKWID_MARKER_START(regionTag)
Shortcut for likwid_markerStartRegion() with \a regionTag if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_MARKER_STOP(regionTag)
Shortcut for likwid_markerStopRegion() with \a regionTag if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
Shortcut for likwid_markerGetResults() for \a regionTag if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_MARKER_SWITCH
Shortcut for likwid_markerNextGroup() if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
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
#define LIKWID_MARKER_START(regionTag) likwid_markerStartRegion(regionTag)
#define LIKWID_MARKER_STOP(regionTag) likwid_markerStopRegion(regionTag)
#define LIKWID_MARKER_CLOSE likwid_markerClose()
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count) likwid_markerGetResults(regionTag, nevents, events, time, count)
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
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

/*! \brief Get accumulated data of a code region

Get the accumulated data of the current thread for the given regionTag.
@param regionTag [in] Print data using this string
@param nr_events [in] Length of events array
@param events [out] Events array for the intermediate results
@param time [out] Accumulated measurement time
@param count [out] Call count of the code region
*/
extern void likwid_markerGetRegion(const char* regionTag, int nr_events, double* events, double *time, int *count);
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
# Access client related functions
################################################################################
*/
/** \addtogroup AccessClient Access client module
 *  @{
 */

/*! \brief Enum for the access modes

LIKWID supports multiple access modes to the MSR and PCI performance monitoring 
registers. For direct access the user must have enough priviledges to access the
MSR and PCI devices. The daemon mode forwards the operations to a daemon with 
higher priviledges.
*/
typedef enum {
    ACCESSMODE_DIRECT = 0, /*!< \brief Access performance monitoring registers directly */
    ACCESSMODE_DAEMON = 1 /*!< \brief Use the access daemon to access the registers */
} AccessMode;

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
# Config file related functions
################################################################################
*/
/** \addtogroup Config Config file module
*  @{
*/
/*! \brief Structure holding values of the configuration file

LIKWID supports the definition of runtime values in a configuration file. The 
most important configurations in most cases are the path the access daemon and 
the corresponding access mode. In order to avoid reading in the system topology
at each start, a path to a topology file can be set. The other values are mostly
used internally.
*/
typedef struct {
    char* topologyCfgFileName; /*!< \brief Path to the topology file */
    char* daemonPath; /*!< \brief Path of the access daemon */
    AccessMode daemonMode; /*!< \brief Access mode to the MSR and PCI registers */
    int maxNumThreads; /*!< \brief Maximum number of HW threads */
    int maxNumNodes; /*!< \brief Maximum number of NUMA nodes */
} Configuration;

/** \brief Pointer for exporting the Configuration data structure */
typedef Configuration* Configuration_t;
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
/*! \brief Structure with general CPU information

General information covers CPU family, model, name and current clock and vendor 
specific information like the version of Intel's performance monitoring facility.
*/
typedef struct {
    uint32_t family; /*!< \brief CPU family ID*/
    uint32_t model; /*!< \brief CPU model ID */
    uint32_t stepping; /*!< \brief Stepping (version) of the CPU */
    uint64_t clock; /*!< \brief Current clock frequency of the executing CPU*/
    int      turbo; /*!< \brief Flag if CPU has a turbo mode */
    char*  osname; /*!< \brief Name of the CPU reported by OS */
    char*  name; /*!< \brief Name of the CPU as identified by LIKWID */
    char*  short_name; /*!< \brief Short name of the CPU*/
    char*  features; /*!< \brief String with all features supported by the CPU*/
    int         isIntel; /*!< \brief Flag if it is an Intel CPU*/
    int     supportUncore; /*!< \brief Flag if system has Uncore performance monitors */
    uint32_t featureFlags; /*!< \brief Mask of all features supported by the CPU*/
    uint32_t perf_version; /*!< \brief Version of Intel's performance monitoring facility */
    uint32_t perf_num_ctr; /*!< \brief Number of general purpose core-local performance monitoring counters */
    uint32_t perf_width_ctr; /*!< \brief Bit width of fixed and general purpose counters */
    uint32_t perf_num_fixed_ctr; /*!< \brief Number of fixed purpose core-local performance monitoring counters */
} CpuInfo;

/*! \brief Structure with IDs of a HW thread

For each HW thread this structure stores the ID of the thread inside a CPU, the
CPU core ID of the HW thread and the CPU socket ID.
\extends CpuTopology
*/
typedef struct {
    uint32_t threadId; /*!< \brief ID of HW thread inside the CPU core */
    uint32_t coreId; /*!< \brief ID of CPU core that executes the HW thread */
    uint32_t packageId; /*!< \brief ID of CPU socket containing the HW thread */
    uint32_t apicId; /*!< \brief ID of HW thread retrieved through the Advanced Programmable Interrupt Controller */
    uint32_t inCpuSet; /*!< \brief ID of HW thread inside the CPU core */
} HWThread;

/*! \brief Enum of possible caches

CPU caches can have different tasks and hold different kind of data. This enum lists all shapes used in all supported CPUs
\extends CacheLevel
*/
typedef enum {
    NOCACHE=0, /*!< \brief No cache used as undef value */
    DATACACHE, /*!< \brief Cache holding data cache lines */
    INSTRUCTIONCACHE, /*!< \brief Cache holding instruction cache lines */
    UNIFIEDCACHE, /*!< \brief Cache holding both instruction and data cache lines */
    ITLB, /*!< \brief Translation Lookaside Buffer cache for instruction pages */
    DTLB /*!< \brief Translation Lookaside Buffer cache for data pages */
} CacheType;

/*! \brief Structure describing a cache level

CPUs are connected to a cache hierarchy with different amount of caches at each level. The CacheLevel structure holds general information about the cache.
\extends CpuTopology
*/
typedef struct {
    uint32_t level; /*!< \brief Level of the cache in the hierarchy */
    CacheType type; /*!< \brief Type of the cache */
    uint32_t associativity; /*!< \brief Amount of cache lines hold by each set */
    uint32_t sets; /*!< \brief Amount of sets */
    uint32_t lineSize; /*!< \brief Size in bytes of one cache line */
    uint32_t size; /*!< \brief Size in bytes of the cache */
    uint32_t threads; /*!< \brief Number of HW thread connected to the cache */
    uint32_t inclusive; /*!< \brief Flag if cache is inclusive (holds also cache lines available in caches nearer to the CPU) or exclusive */
} CacheLevel;

/*! \brief Structure describing the topology of the HW threads in the system

This structure describes the topology at HW thread level like the amount of HW threads, how they are distributed over the CPU sockets/packages and how the caching hierarchy is assembled.
*/
typedef struct {
    uint32_t numHWThreads; /*!< \brief Amount of HW threads in the system and length of \a threadPool */
    uint32_t activeHWThreads; /*!< \brief Amount of HW threads in the system and length of \a threadPool */
    uint32_t numSockets; /*!< \brief Amount of CPU sockets/packages in the system */
    uint32_t numCoresPerSocket; /*!< \brief Amount of physical cores in one CPU socket/package */
    uint32_t numThreadsPerCore; /*!< \brief Amount of HW threads in one physical CPU core */
    uint32_t numCacheLevels; /*!< \brief Amount of caches for each HW thread and length of \a cacheLevels */
    HWThread* threadPool; /*!< \brief List of all HW thread descriptions */
    CacheLevel*  cacheLevels; /*!< \brief List of all caches in the hierarchy */
    struct treeNode* topologyTree; /*!< \brief Anchor for a tree structure describing the system topology */
} CpuTopology;

/*! \brief Variable holding the global cpu information structure */
extern CpuInfo cpuid_info;
/*! \brief Variable holding the global cpu topology structure */
extern CpuTopology cpuid_topology;

/** \brief Pointer for exporting the CpuInfo data structure */
typedef CpuInfo* CpuInfo_t;
/** \brief Pointer for exporting the CpuTopology data structure */
typedef CpuTopology* CpuTopology_t;
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
/** \addtogroup NumaTopology NUMA memory topology module
 *  @{
 */
/*! \brief CPUs in NUMA node and general information about a NUMA domain

The NumaNode structure describes the topology and holds general information of a
NUMA node. The structure is filled by calling numa_init() by either the HWLOC 
library or by evaluating the /proc filesystem.
\extends NumaTopology
*/
typedef struct {
    uint32_t id; /*!< \brief ID of the NUMA node */
    uint64_t totalMemory; /*!< \brief Amount of memory in the NUMA node */
    uint64_t freeMemory; /*!< \brief Amount of free memory in the NUMA node */
    uint32_t numberOfProcessors; /*!< \brief umber of processors covered by the NUMA node and length of \a processors */
    uint32_t*  processors; /*!< \brief List of HW threads in the NUMA node */
    uint32_t*  processorsCompact; /*!< \brief Currently unused */
    uint32_t numberOfDistances; /*!< \brief Amount of distances to the other NUMA nodes in the system and self  */
    uint32_t*  distances; /*!< \brief List of distances to the other NUMA nodes and self */
} NumaNode;


/*! \brief  The NumaTopology structure describes all NUMA nodes in the current system.
*/
typedef struct {
    uint32_t numberOfNodes; /*!< \brief Number of NUMA nodes in the system and length of \a nodes  */
    NumaNode* nodes; /*!< \brief List of NUMA nodes */
} NumaTopology;

/*! \brief Variable holding the global NUMA information structure */
extern NumaTopology numa_info;

/** \brief Pointer for exporting the NumaTopology data structure */
typedef NumaTopology* NumaTopology_t;

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
/** \addtogroup AffinityDomains Thread affinity module
 *  @{
 */

/*! \brief The AffinityDomain data structure describes a single domain in the current system

The AffinityDomain data structure describes a single domain in the current system. Example domains are NUMA nodes, CPU sockets/packages or LLC (Last Level Cache) cache domains.
\extends AffinityDomains
*/
typedef struct {
    bstring tag; /*!< \brief Bstring with the ID for the affinity domain. Currently possible values: N (node), SX (socket/package X), CX (LLC cache domain X) and MX (memory domain X) */
    uint32_t numberOfProcessors; /*!< \brief Number of HW threads in the domain and length of \a processorList */
    uint32_t numberOfCores; /*!< \brief Number of CPU cores in the domain */
    int*  processorList; /*!< \brief List of HW thread IDs in the domain */
} AffinityDomain;

/*! \brief The AffinityDomains data structure holds different count variables describing the
various system layers

Affinity domains are for example the amount of NUMA domains, CPU sockets/packages or LLC 
(Last Level Cache) cache domains of the current machine. Moreover a list of
\a domains holds the processor lists for each domain that are used for
scheduling processes to domain specific HW threads. Some amounts are duplicates
or derivation of values in \a CpuInfo, \a CpuTopology and \a NumaTopology.
*/
typedef struct {
    uint32_t numberOfSocketDomains; /*!< \brief Number of CPU sockets/packages in the system */
    uint32_t numberOfNumaDomains; /*!< \brief Number of NUMA nodes in the system */
    uint32_t numberOfProcessorsPerSocket; /*!< \brief Number of HW threads per socket/package in the system */
    uint32_t numberOfCacheDomains; /*!< \brief Number of LLC caches in the system */
    uint32_t numberOfCoresPerCache; /*!< \brief Number of HW threads per LLC cache in the system */
    uint32_t numberOfProcessorsPerCache; /*!< \brief Number of CPU cores per LLC cache in the system */
    uint32_t numberOfAffinityDomains; /*!< \brief Number of affinity domains in the current system  and length of \a domains array */
    AffinityDomain* domains; /*!< \brief List of all domains in the system */
} AffinityDomains;

/** \brief Pointer for exporting the AffinityDomains data structure */
typedef AffinityDomains* AffinityDomains_t;

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
extern int affinity_processGetProcessorId();
/*! \brief Return the CPU ID where the current thread runs.

@return CPU ID
*/
extern int affinity_threadGetProcessorId();
/*! \brief Destroy affinity information structure

Destroys the affinity information structure AffinityDomains_t. Retrieved pointers
to the structures are not valid anymore after this function call
\sa AffinityDomains_t
*/
extern void affinity_finalize();
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

/*! \brief Initialize performance monitoring maps

Initialize the performance monitoring maps for counters, events and Uncore boxes#
for the current architecture. topology_init() and numa_init() must be called before calling
perfmon_init_maps()
\sa RegisterMap list, PerfmonEvent list and BoxMap list
*/
extern void perfmon_init_maps(void);
/*! \brief Add an event string to LIKWID

A event string looks like Eventname:Countername(:Option1:Option2:...),...
The eventname, countername and options are checked if they are available.
@param [in] eventCString Event string
@return Returns the ID of the new eventSet
*/
extern int perfmon_addEventSet(char* eventCString);
/*! \brief Setup all performance monitoring counters of an eventSet

A event string looks like Eventname:Countername(:Option1:Option2:...),...
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
# Time measurements related functions
################################################################################
*/

/** \addtogroup TimerMon Time measurement module
 *  @{
 */

/*! \brief Struct defining the start and stop time of a time interval
\extends TimerData
*/
typedef union
{
    uint64_t int64; /*!< \brief Cycle count in 64 bit */
    struct {uint32_t lo, hi;} int32; /*!< \brief Cycle count stored in two 32 bit fields */
} TscCounter;

/*! \brief Struct defining the start and stop time of a time interval
*/
typedef struct {
    TscCounter start; /*!< \brief Cycles at start */
    TscCounter stop; /*!< \brief Cycles at stop */
} TimerData;

/*! \brief Initialize timer by retrieving baseline frequency and cpu clock
*/
extern void timer_init( void );
/*! \brief Return the measured interval in seconds

@param [in] time Structure holding the cycle count at start and stop
@return Time in seconds
*/
extern double timer_print( TimerData* time);
/*! \brief Return the measured interval in cycles

@param [in] time Structure holding the cycle count at start and stop
@return Time in cycles
*/
extern uint64_t timer_printCycles( TimerData* time);
/*! \brief Return the CPU clock determined at timer_init

@return CPU clock
*/
extern uint64_t timer_getCpuClock( void );
/*! \brief Return the baseline CPU clock determined at timer_init

@return Baseline CPU clock
*/
extern uint64_t timer_getBaseline( void );
/*! \brief Start time measurement

@param [in,out] time Structure holding the cycle count at start
*/
extern void timer_start( TimerData* time );
/*! \brief Stop time measurement

@param [in,out] time Structure holding the cycle count at stop
*/
extern void timer_stop ( TimerData* time);

/** @}*/

/* 
################################################################################
# Power measurements related functions
################################################################################
*/
/** \addtogroup PowerMon Power and Energy monitoring module
 *  @{
 */

/*!
\def NUM_POWER_DOMAINS
Amount of currently supported RAPL domains
*/
#define NUM_POWER_DOMAINS 4
/*! \brief List of all RAPL domain names
*/
extern const char* power_names[NUM_POWER_DOMAINS];

/*!
\def POWER_DOMAIN_SUPPORT_STATUS
Flag to check in PowerDomain's supportFlag if the status msr registers are available
*/
#define POWER_DOMAIN_SUPPORT_STATUS (1ULL<<0)
/*!
\def POWER_DOMAIN_SUPPORT_LIMIT
Flag to check in PowerDomain's supportFlag if the limit msr registers are available
*/
#define POWER_DOMAIN_SUPPORT_LIMIT (1ULL<<1)
/*!
\def POWER_DOMAIN_SUPPORT_POLICY
Flag to check in PowerDomain's supportFlag if the policy msr registers are available
*/
#define POWER_DOMAIN_SUPPORT_POLICY (1ULL<<2)
/*!
\def POWER_DOMAIN_SUPPORT_PERF
Flag to check in PowerDomain's supportFlag if the perf msr registers are available
*/
#define POWER_DOMAIN_SUPPORT_PERF (1ULL<<3)
/*!
\def POWER_DOMAIN_SUPPORT_INFO
Flag to check in PowerDomain's supportFlag if the info msr registers are available
*/
#define POWER_DOMAIN_SUPPORT_INFO (1ULL<<4)


/*! \brief Information structure of CPU's turbo mode
\extends PowerInfo
*/
typedef struct {
    int numSteps; /*!< \brief Amount of turbo mode steps/frequencies */
    double* steps; /*!< \brief List of turbo mode steps */
} TurboBoost;

/*! \brief Enum for all supported RAPL domains
\extends PowerDomain
*/
typedef enum {
    PKG = 0, /*!< \brief PKG domain, mostly one CPU socket/package */
    PP0 = 1, /*!< \brief PP0 domain, not clearly defined by Intel */
    PP1 = 2, /*!< \brief PP1 domain, not clearly defined by Intel */
    DRAM = 3 /*!< \brief DRAM domain, the memory modules */
} PowerType;

/*! \brief Structure describing an RAPL power domain
\extends PowerInfo
*/
typedef struct {
    PowerType type; /*!< \brief Identifier which RAPL domain is managed by this struct */
    uint32_t supportFlags; /*!< \brief Bitmask which features are supported by the power domain */
    double energyUnit; /*!< \brief Multiplier for energy measurements */
    double tdp; /*!< \brief Thermal Design Power (maximum amount of heat generated by the CPU) */
    double minPower; /*!< \brief Minimal power consumption of the CPU */
    double maxPower; /*!< \brief Maximal power consumption of the CPU */
    double maxTimeWindow; /*!< \brief Minimal power measurement interval */
} PowerDomain;

/*! \brief Information structure of CPU's power measurement facility
*/
typedef struct {
    double baseFrequency; /*!< \brief Base frequency of the CPU */
    double minFrequency; /*!< \brief Minimal frequency of the CPU */
    TurboBoost turbo; /*!< \brief Turbo boost information */
    int hasRAPL; /*!< \brief RAPL support flag */
    double powerUnit; /*!< \brief Multiplier for power measurements */
    double timeUnit; /*!< \brief Multiplier for time information */
    PowerDomain domains[NUM_POWER_DOMAINS]; /*!< \brief List of power domains */
} PowerInfo;

/*! \brief Power measurement data for start/stop measurements
*/
typedef struct {
    int domain; /*!< \brief RAPL domain identifier */
    uint32_t before; /*!< \brief Counter state at start */
    uint32_t after; /*!< \brief Counter state at stop */
} PowerData;

/*! \brief Variable holding the global power information structure */
extern PowerInfo power_info;

/** \brief Pointer for exporting the PowerInfo data structure */
typedef PowerInfo* PowerInfo_t;
/** \brief Pointer for exporting the PowerData data structure */
typedef PowerData* PowerData_t;

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
/*! \brief Get energy Unit

@param [in] domain RAPL domain ID
@return Power unit of the given RAPL domain
*/
extern double power_getEnergyUnit(int domain);

/*! \brief Get the values of the limit register of a domain

@param [in] cpuId CPU ID
@param [in] domain RAPL domain ID
@param [out] power Power limit
@param [out] time Time limit
@return error code
*/
int power_limitGet(int cpuId, PowerType domain, double* power, double* time);

/*! \brief Set the values of the limit register of a domain

@param [in] cpuId CPU ID
@param [in] domain RAPL domain ID
@param [in] power Power limit
@param [in] time Time limit
@param [in] doClamping Activate clamping (going below OS-requested power level)
@return error code
*/
int power_limitSet(int cpuId, PowerType domain, double power, double time, int doClamping);

/*! \brief Get the state of a power limit, activated or deactivated

@param [in] cpuId CPU ID
@param [in] domain RAPL domain ID
@return state, 1 for active, 0 for inactive
*/
int power_limitState(int cpuId, PowerType domain);

/*! \brief Free space of power_unit
*/
extern void power_finalize(void);
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
