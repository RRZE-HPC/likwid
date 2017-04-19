/*
 * =======================================================================================
 *
 *      Filename:  likwid.h
 *
 *      Description:  Header File of likwid API
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Authors:  Thomas Roehl (tr), thomas.roehl@googlemail.com
 *
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

#define LIKWID_VERSION "VERSION.RELEASE.MINORVERSION"
#define LIKWID_COMMIT GITCOMMIT

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
\def LIKWID_MARKER_REGISTER(regionTag)
Shortcut for likwid_markerRegisterRegion() with \a regionTag if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
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
#define LIKWID_MARKER_REGISTER(regionTag) likwid_markerRegisterRegion(regionTag)
#define LIKWID_MARKER_START(regionTag) likwid_markerStartRegion(regionTag)
#define LIKWID_MARKER_STOP(regionTag) likwid_markerStopRegion(regionTag)
#define LIKWID_MARKER_CLOSE likwid_markerClose()
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count) likwid_markerGetRegion(regionTag, nevents, events, time, count)
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
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
extern void likwid_markerInit(void) __attribute__ ((visibility ("default") ));
/*! \brief Initialize LIKWID's marker API for the current thread

Must be called in parallel region of the application to set up basic data structures
of LIKWID. Before you can call likwid_markerThreadInit() you have to call likwid_markerInit().

*/
extern void likwid_markerThreadInit(void) __attribute__ ((visibility ("default") ));
/*! \brief Select next group to measure

Must be called in parallel region of the application to switch group on every CPU.
*/
extern void likwid_markerNextGroup(void) __attribute__ ((visibility ("default") ));
/*! \brief Close LIKWID's marker API

Must be called in serial region of the application. It gathers all data of regions and
writes them out to a file (filepath in env variable LIKWID_FILEPATH).
*/
extern void likwid_markerClose(void) __attribute__ ((visibility ("default") ));
/*! \brief Register a measurement region

Initializes the hashTable entry in order to reduce execution time of likwid_markerStartRegion()
@param regionTag [in] Initialize data using this string
@return Error code
*/
extern int likwid_markerRegisterRegion(const char* regionTag) __attribute__ ((visibility ("default") ));
/*! \brief Start a measurement region

Reads the values of all configured counters and saves the results under the name given
in regionTag.
@param regionTag [in] Store data using this string
@return Error code of start operation
*/
extern int likwid_markerStartRegion(const char* regionTag) __attribute__ ((visibility ("default") ));
/*! \brief Stop a measurement region

Reads the values of all configured counters and saves the results under the name given
in regionTag. The measurement data of the stopped region gets summed up in global region counters.
@param regionTag [in] Store data using this string
@return Error code of stop operation
*/
extern int likwid_markerStopRegion(const char* regionTag) __attribute__ ((visibility ("default") ));

/*! \brief Get accumulated data of a code region

Get the accumulated data of the current thread for the given regionTag.
@param regionTag [in] Print data using this string
@param nr_events [in,out] Length of events array
@param events [out] Events array for the intermediate results
@param time [out] Accumulated measurement time
@param count [out] Call count of the code region
*/
extern void likwid_markerGetRegion(const char* regionTag, int* nr_events, double* events, double *time, int *count) __attribute__ ((visibility ("default") ));
/* utility routines */
/*! \brief Get CPU ID of the current process/thread

Returns the ID of the CPU the current process or thread is running on.
@return current CPU ID
*/
extern int  likwid_getProcessorId() __attribute__ ((visibility ("default") ));
/*! \brief Pin the current process to given CPU

Pin the current process to the given CPU ID. The process cannot be scheduled to
another CPU after pinning but the pinning can be changed anytime with this function.
@param [in] processorId CPU ID to pin the current process to
@return error code (1 for success, 0 for error)
*/
extern int  likwid_pinProcess(int processorId) __attribute__ ((visibility ("default") ));
/*! \brief Pin the current thread to given CPU

Pin the current thread to the given CPU ID. The thread cannot be scheduled to
another CPU after pinning but the pinning can be changed anytime with this function
@param [in] processorId CPU ID to pin the current thread to
@return error code (1 for success, 0 for error)
*/
extern int  likwid_pinThread(int processorId) __attribute__ ((visibility ("default") ));
/** @}*/

/*
################################################################################
# Access client related functions
################################################################################
*/
/** \addtogroup Access Access module
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

/*! \brief Set access mode

Sets the mode how the MSR and PCI registers should be accessed. 0 for direct access (propably root priviledges required) and 1 for accesses through the access daemon. It must be called before HPMinit()
@param [in] mode (0=direct, 1=daemon)
*/
extern void HPMmode(int mode) __attribute__ ((visibility ("default") ));
/*! \brief Initialize access module

Initialize the module internals to either the MSR/PCI files or the access daemon
@return error code (0 for sccess)
*/
extern int HPMinit() __attribute__ ((visibility ("default") ));
/*! \brief Add CPU to access module

Add the given CPU to the access module. This opens the commnunication to either the MSR/PCI files or the access daemon.
@param [in] cpu_id CPU that should be enabled for measurements
@return error code (0 for success, -ENODEV if access cannot be initialized
*/
extern int HPMaddThread(int cpu_id) __attribute__ ((visibility ("default") ));
/*! \brief Close connections

Close the connections to the MSR/PCI files or the access daemon
*/
extern void HPMfinalize() __attribute__ ((visibility ("default") ));
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
    char* configFileName; /*!< \brief Path to the configuration file */
    char* topologyCfgFileName; /*!< \brief Path to the topology file */
    char* daemonPath; /*!< \brief Path of the access daemon */
    char* groupPath; /*!< \brief Path of default performance group directory */
    AccessMode daemonMode; /*!< \brief Access mode to the MSR and PCI registers */
    int maxNumThreads; /*!< \brief Maximum number of HW threads */
    int maxNumNodes; /*!< \brief Maximum number of NUMA nodes */
} Configuration;

/** \brief Pointer for exporting the Configuration data structure */
typedef Configuration* Configuration_t;
/*! \brief Read the config file of LIKWID, if it exists

Search for LIKWID config file and read the values in
Currently the paths /usr/local/etc/likwid.cfg, /etc/likwid.cfg and the path
defined in config.mk are checked.
@return error code (0 for success, -EFAULT if no file can be found)
*/
extern int init_configuration(void) __attribute__ ((visibility ("default") ));
/*! \brief Destroy the config structure

Destroys the current config structure and frees all allocated memory for path names
@return error code (0 for success, -EFAULT if config structure not initialized)
*/
extern int destroy_configuration(void) __attribute__ ((visibility ("default") ));


/*! \brief Retrieve the config structure

Get the initialized configuration
\sa Configuration_t
@return Configuration_t (pointer to internal Configuration structure)
*/
extern Configuration_t get_configuration(void) __attribute__ ((visibility ("default") ));

/*! \brief Set group path in the config struction

Set group path in the config struction. The path must be a directory.
@param [in] path
@return error code (0 for success, -ENOMEM if reallocation failed, -ENOTDIR if no directoy)
*/
extern int config_setGroupPath(const char* path) __attribute__ ((visibility ("default") ));

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

CpuInfo_t and CpuTopology_t are initialized by either HWLOC, CPUID/ProcFS or topology file if present. The topology file name can be configured in the configuration file. Furthermore, the paths /etc/likwid_topo.cfg and &lt;PREFIX&gt;/etc/likwid_topo.cfg are checked.
\sa CpuInfo_t and CpuTopology_t
@return always 0
*/
extern int topology_init(void) __attribute__ ((visibility ("default") ));
/*! \brief Retrieve CPU topology of the current machine

\sa CpuTopology_t
@return CpuTopology_t (pointer to internal cpuid_topology structure)
*/
extern CpuTopology_t get_cpuTopology(void) __attribute__ ((visibility ("default") ));
/*! \brief Retrieve CPU information of the current machine

Get the previously initialized CPU info structure containing number of CPUs/Threads
\sa CpuInfo_t
@return CpuInfo_t (pointer to internal cpuid_info structure)
*/
extern CpuInfo_t get_cpuInfo(void) __attribute__ ((visibility ("default") ));
/*! \brief Destroy topology structures CpuInfo_t and CpuTopology_t.

Retrieved pointers to the structures are not valid anymore after this function call
\sa CpuInfo_t and CpuTopology_t
*/
extern void topology_finalize(void) __attribute__ ((visibility ("default") ));
/*! \brief Print all supported architectures
*/
extern void print_supportedCPUs(void) __attribute__ ((visibility ("default") ));
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
extern int numa_init(void) __attribute__ ((visibility ("default") ));
/*! \brief Retrieve NUMA information of the current machine

Get the previously initialized NUMA info structure
\sa NumaTopology_t
@return NumaTopology_t (pointer to internal numa_info structure)
*/
extern NumaTopology_t get_numaTopology(void) __attribute__ ((visibility ("default") ));
/*! \brief Set memory allocation policy to interleaved

Set the memory allocation policy to interleaved for given list of CPUs
@param [in] processorList List of processors
@param [in] numberOfProcessors Length of processor list
*/
extern void numa_setInterleaved(const int* processorList, int numberOfProcessors) __attribute__ ((visibility ("default") ));
/*! \brief Allocate memory from a specific specific NUMA node
@param [in,out] ptr Start pointer of memory
@param [in] size Size for the allocation
@param [in] domainId ID of NUMA node for the allocation
*/
extern void numa_membind(void* ptr, size_t size, int domainId) __attribute__ ((visibility ("default") ));
/*! \brief Destroy NUMA information structure

Destroys the NUMA information structure NumaTopology_t. Retrieved pointers
to the structures are not valid anymore after this function call
\sa NumaTopology_t
*/
extern void numa_finalize(void) __attribute__ ((visibility ("default") ));
/*! \brief Retrieve the number of NUMA nodes

Returns the number of NUMA nodes of the current machine. Can also be read out of
NumaTopology_t
\sa NumaTopology_t
@return Number of NUMA nodes
*/
extern int likwid_getNumberOfNodes(void) __attribute__ ((visibility ("default") ));
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
extern void affinity_init() __attribute__ ((visibility ("default") ));
/*! \brief Retrieve affinity structure

Get the previously initialized affinity info structure
\sa AffinityDomains_t
@return AffinityDomains_t (pointer to internal affinityDomains structure)
*/
extern AffinityDomains_t get_affinityDomains(void) __attribute__ ((visibility ("default") ));
/*! \brief Pin process to a CPU

Pin process to a CPU. Duplicate of likwid_pinProcess()
@param [in] processorId CPU ID for pinning
*/
extern void affinity_pinProcess(int processorId) __attribute__ ((visibility ("default") ));
/*! \brief Pin processes to a CPU

Pin processes to a CPU. Creates a cpuset with the given processor IDs
@param [in] cpu_count Number of processors in processorIds
@param [in] processorIds Array of processor IDs
*/
extern void affinity_pinProcesses(int cpu_count, const int* processorIds) __attribute__ ((visibility ("default") ));
/*! \brief Pin thread to a CPU

Pin thread to a CPU. Duplicate of likwid_pinThread()
@param [in] processorId CPU ID for pinning
*/
extern void affinity_pinThread(int processorId) __attribute__ ((visibility ("default") ));
/*! \brief Return the CPU ID where the current process runs.

@return CPU ID
*/
extern int affinity_processGetProcessorId() __attribute__ ((visibility ("default") ));
/*! \brief Return the CPU ID where the current thread runs.

@return CPU ID
*/
extern int affinity_threadGetProcessorId() __attribute__ ((visibility ("default") ));
/*! \brief Destroy affinity information structure

Destroys the affinity information structure AffinityDomains_t. Retrieved pointers
to the structures are not valid anymore after this function call
\sa AffinityDomains_t
*/
extern void affinity_finalize() __attribute__ ((visibility ("default") ));
/** @}*/

/*
################################################################################
# CPU string parsing related functions
################################################################################
*/
/** \addtogroup CPUParse CPU string parser module
 *  @{
 */

/*! \brief Read CPU selection string and resolve to available CPU numbers

Reads the CPU selection string and fills the given list with the CPU numbers
defined in the selection string. This function is a interface function for the
different selection modes: scatter, expression, logical and physical.
@param [in] cpustring Selection string
@param [in,out] cpulist List of CPUs
@param [in] length Length of cpulist
@return error code (>0 on success for the returned list length, -ERRORCODE on failure)
*/
extern int cpustr_to_cpulist(const char* cpustring, int* cpulist, int length)  __attribute__ ((visibility ("default") ));
/*! \brief Read NUMA node selection string and resolve to available NUMA node numbers

Reads the NUMA node selection string and fills the given list with the NUMA node numbers
defined in the selection string.
@param [in] nodestr Selection string
@param [out] nodes List of available NUMA nodes
@param [in] length Length of NUMA node list
@return error code (>0 on success for the returned list length, -ERRORCODE on failure)
*/
extern int nodestr_to_nodelist(const char* nodestr, int* nodes, int length)  __attribute__ ((visibility ("default") ));
/*! \brief Read CPU socket selection string and resolve to available CPU socket numbers

Reads the CPU socket selection string and fills the given list with the CPU socket numbers
defined in the selection string.
@param [in] sockstr Selection string
@param [out] sockets List of available CPU sockets
@param [in] length Length of CPU socket list
@return error code (>0 on success for the returned list length, -ERRORCODE on failure)
*/
extern int sockstr_to_socklist(const char* sockstr, int* sockets, int length)  __attribute__ ((visibility ("default") ));

/** @}*/

/*
################################################################################
# Performance monitoring related functions
################################################################################
*/
/** \addtogroup PerfMon Performance monitoring module
 *  @{
 */

/*! \brief Get all groups

Checks the configured performance group path for the current architecture and
returns all found group names
@return Amount of found performance groups
*/
extern int perfmon_getGroups(char*** groups, char*** shortinfos, char*** longinfos) __attribute__ ((visibility ("default") ));

/*! \brief Free all group information

@param [in] nrgroups Number of groups
@param [in] groups List of group names
@param [in] shortinfos List of short information string about group
@param [in] longinfos List of long information string about group
*/
extern void perfmon_returnGroups(int nrgroups, char** groups, char** shortinfos, char** longinfos) __attribute__ ((visibility ("default") ));

/*! \brief Initialize performance monitoring facility

Initialize the performance monitoring feature by creating basic data structures.
The access mode must already be set when calling perfmon_init()
@param [in] nrThreads Amount of threads
@param [in] threadsToCpu List of CPUs
@return error code (0 on success, -ERRORCODE on failure)
*/
extern int perfmon_init(int nrThreads, const int* threadsToCpu) __attribute__ ((visibility ("default") ));

/*! \brief Initialize performance monitoring maps

Initialize the performance monitoring maps for counters, events and Uncore boxes#
for the current architecture. topology_init() and numa_init() must be called before calling
perfmon_init_maps()
\sa RegisterMap list, PerfmonEvent list and BoxMap list
*/
extern void perfmon_init_maps(void) __attribute__ ((visibility ("default") ));
/*! \brief Check the performance monitoring maps whether counters and events are available

Checks each counter and event in the performance monitoring maps for their availibility on
the current system. topology_init(), numa_init() and perfmon_init_maps() must be called before calling
perfmon_check_counter_map().
\sa RegisterMap list, PerfmonEvent list and BoxMap list
*/
extern void perfmon_check_counter_map(int cpu_id) __attribute__ ((visibility ("default") ));
/*! \brief Add an event string to LIKWID

A event string looks like Eventname:Countername(:Option1:Option2:...),...
The eventname, countername and options are checked if they are available.
@param [in] eventCString Event string
@return Returns the ID of the new eventSet
*/
extern int perfmon_addEventSet(const char* eventCString) __attribute__ ((visibility ("default") ));
/*! \brief Setup all performance monitoring counters of an eventSet

A event string looks like Eventname:Countername(:Option1:Option2:...),...
The eventname, countername and options are checked if they are available.
@param [in] groupId (returned from perfmon_addEventSet()
@return error code (-ENOENT if groupId is invalid and -1 if the counters of one CPU cannot be set up)
*/
extern int perfmon_setupCounters(int groupId) __attribute__ ((visibility ("default") ));
/*! \brief Start performance monitoring counters

Start the counters that have been previously set up by perfmon_setupCounters().
The counter registered are zeroed before enabling the counters
@return 0 on success and -(thread_id+1) for error
*/
extern int perfmon_startCounters(void) __attribute__ ((visibility ("default") ));
/*! \brief Stop performance monitoring counters

Stop the counters that have been previously started by perfmon_startCounters().
All config registers get zeroed before reading the counter register.
@return 0 on success and -(thread_id+1) for error
*/
extern int perfmon_stopCounters(void) __attribute__ ((visibility ("default") ));
/*! \brief Read the performance monitoring counters on all CPUs

Read the counters that have been previously started by perfmon_startCounters().
The counters are stopped directly to avoid interference of LIKWID with the measured
code. Before returning, the counters are started again.
@return 0 on success and -(thread_id+1) for error
*/
extern int perfmon_readCounters(void) __attribute__ ((visibility ("default") ));
/*! \brief Read the performance monitoring counters on one CPU

Read the counters that have been previously started by perfmon_startCounters().
The counters are stopped directly to avoid interference of LIKWID with the measured
code. Before returning, the counters are started again. Only one CPU is read.
@param [in] cpu_id CPU ID of the CPU that should be read
@return 0 on success and -(thread_id+1) for error
*/
extern int perfmon_readCountersCpu(int cpu_id) __attribute__ ((visibility ("default") ));
/*! \brief Read the performance monitoring counters of all threads in a group

Read the counters that have been previously started by perfmon_startCounters().
The counters are stopped directly to avoid interference of LIKWID with the measured
code. Before returning, the counters are started again.
@param [in] groupId Read the counters for all threads taking part in group
@return 0 on success and -(thread_id+1) for error
*/
extern int perfmon_readGroupCounters(int groupId) __attribute__ ((visibility ("default") ));
/*! \brief Read the performance monitoring counters of on thread in a group

Read the counters that have been previously started by perfmon_startCounters().
The counters are stopped directly to avoid interference of LIKWID with the measured
code. Before returning, the counters are started again. Only one thread's CPU is read.
@param [in] groupId Read the counters defined in group identified with groupId
@param [in] threadId Read the counters for the thread
@return 0 on success and -(thread_id+1) for error
*/
extern int perfmon_readGroupThreadCounters(int groupId, int threadId) __attribute__ ((visibility ("default") ));
/*! \brief Switch the active eventSet to a new one

Stops the currently running counters, switches the eventSet by setting up the
counters and start the counters.
@param [in] new_group ID of group that should be switched to.
@return 0 on success and -(thread_id+1) for error
*/
extern int perfmon_switchActiveGroup(int new_group) __attribute__ ((visibility ("default") ));
/*! \brief Close the perfomance monitoring facility of LIKWID

Deallocates all internal data that is used during performance monitoring. Also
the counter values are not accessible after this function.
*/
extern void perfmon_finalize(void) __attribute__ ((visibility ("default") ));
/*! \brief Get the results of the specified group, counter and thread

Get the result of all measurement cycles. The function takes care of happened
overflows and if the counter values need to be calculated with multipliers.
@param [in] groupId ID of the group that should be read
@param [in] eventId ID of the event that should be read
@param [in] threadId ID of the thread/cpu that should be read
@return The counter result
*/
extern double perfmon_getResult(int groupId, int eventId, int threadId) __attribute__ ((visibility ("default") ));
/*! \brief Get the last results of the specified group, counter and thread

Get the result of the last measurement cycle. The function takes care of happened
overflows and if the counter values need to be calculated with multipliers.
@param [in] groupId ID of the group that should be read
@param [in] eventId ID of the event that should be read
@param [in] threadId ID of the thread/cpu that should be read
@return The counter result
*/
extern double perfmon_getLastResult(int groupId, int eventId, int threadId) __attribute__ ((visibility ("default") ));
/*! \brief Get the metric result of the specified group, counter and thread

Get the metric result of all measurement cycles. It reads all raw results for the given groupId and threadId.
@param [in] groupId ID of the group that should be read
@param [in] metricId ID of the metric that should be calculated
@param [in] threadId ID of the thread/cpu that should be read
@return The metric result
*/
extern double perfmon_getMetric(int groupId, int metricId, int threadId) __attribute__ ((visibility ("default") ));
/*! \brief Get the last metric result of the specified group, counter and thread

Get the metric result of the last measurement cycle. It reads all raw results for the given groupId and threadId.
@param [in] groupId ID of the group that should be read
@param [in] metricId ID of the metric that should be calculated
@param [in] threadId ID of the thread/cpu that should be read
@return The metric result
*/
extern double perfmon_getLastMetric(int groupId, int metricId, int threadId) __attribute__ ((visibility ("default") ));

/*! \brief Get the number of configured event groups

@return Number of groups
*/
extern int perfmon_getNumberOfGroups(void) __attribute__ ((visibility ("default") ));
/*! \brief Get the number of configured eventSets in group

@param [in] groupId ID of group
@return Number of eventSets
*/
extern int perfmon_getNumberOfEvents(int groupId) __attribute__ ((visibility ("default") ));
/*! \brief Get the accumulated measurement time a group

@param [in] groupId ID of group
@return Time in seconds the event group was measured
*/
extern double perfmon_getTimeOfGroup(int groupId) __attribute__ ((visibility ("default") ));
/*! \brief Get the ID of the currently set up event group

@return Number of active group
*/
extern int perfmon_getIdOfActiveGroup(void) __attribute__ ((visibility ("default") ));
/*! \brief Get the number of threads specified at perfmon_init()

@return Number of threads
*/
extern int perfmon_getNumberOfThreads(void) __attribute__ ((visibility ("default") ));


/*! \brief Set verbosity of LIKWID library

*/
extern void perfmon_setVerbosity(int verbose) __attribute__ ((visibility ("default") ));

/*! \brief Get the event name of the specified group and event

Get the metric name as defined in the performance group file
@param [in] groupId ID of the group that should be read
@param [in] eventId ID of the event that should be returned
@return The event name or NULL in case of failure
*/
extern char* perfmon_getEventName(int groupId, int eventId) __attribute__ ((visibility ("default") ));
/*! \brief Get the counter name of the specified group and event

Get the counter name as defined in the performance group file
@param [in] groupId ID of the group that should be read
@param [in] eventId ID of the event of which the counter should be returned
@return The counter name or NULL in case of failure
*/
extern char* perfmon_getCounterName(int groupId, int eventId) __attribute__ ((visibility ("default") ));
/*! \brief Get the name group

Get the name of group. Either it is the name of the performance group or "Custom"
@param [in] groupId ID of the group that should be read
@return The group name or NULL in case of failure
*/
extern char* perfmon_getGroupName(int groupId) __attribute__ ((visibility ("default") ));
/*! \brief Get the metric name of the specified group and metric

Get the metric name as defined in the performance group file
@param [in] groupId ID of the group that should be read
@param [in] metricId ID of the metric that should be calculated
@return The metric name or NULL in case of failure
*/
extern char* perfmon_getMetricName(int groupId, int metricId) __attribute__ ((visibility ("default") ));
/*! \brief Get the short informational string of the specified group

Returns the short information string as defined by performance groups or "Custom"
in case of custom event sets
@param [in] groupId ID of the group that should be read
@return The short information or NULL in case of failure
*/
extern char* perfmon_getGroupInfoShort(int groupId) __attribute__ ((visibility ("default") ));
/*! \brief Get the long descriptive string of the specified group

Returns the long descriptive string as defined by performance groups or NULL
in case of custom event sets
@param [in] groupId ID of the group that should be read
@return The long description or NULL in case of failure
*/
extern char* perfmon_getGroupInfoLong(int groupId) __attribute__ ((visibility ("default") ));

/*! \brief Get the number of configured metrics for group

@param [in] groupId ID of group
@return Number of metrics
*/
extern int perfmon_getNumberOfMetrics(int groupId) __attribute__ ((visibility ("default") ));

/*! \brief Get the last measurement time a group

@param [in] groupId ID of group
@return Time in seconds the event group was measured the last time
*/
extern double perfmon_getLastTimeOfGroup(int groupId) __attribute__ ((visibility ("default") ));

/*! \brief Read the output file of the Marker API
@param [in] filename Filename with Marker API results
@return 0 or negative error number
*/
extern int perfmon_readMarkerFile(const char* filename) __attribute__ ((visibility ("default") ));
/*! \brief Free space for read in Marker API file
*/
extern void perfmon_destroyMarkerResults() __attribute__ ((visibility ("default") ));
/*! \brief Get the number of regions listed in Marker API result file

@return Number of regions
*/
extern int perfmon_getNumberOfRegions() __attribute__ ((visibility ("default") ));
/*! \brief Get the groupID of a region

@param [in] region ID of region
@return Group ID of region
*/
extern int perfmon_getGroupOfRegion(int region) __attribute__ ((visibility ("default") ));
/*! \brief Get the tag of a region
@param [in] region ID of region
@return tag of region
*/
extern char* perfmon_getTagOfRegion(int region) __attribute__ ((visibility ("default") ));
/*! \brief Get the number of events of a region
@param [in] region ID of region
@return Number of events of region
*/
extern int perfmon_getEventsOfRegion(int region) __attribute__ ((visibility ("default") ));
/*! \brief Get the number of metrics of a region
@param [in] region ID of region
@return Number of metrics of region
*/
extern int perfmon_getMetricsOfRegion(int region) __attribute__ ((visibility ("default") ));
/*! \brief Get the number of threads of a region
@param [in] region ID of region
@return Number of threads of region
*/
extern int perfmon_getThreadsOfRegion(int region) __attribute__ ((visibility ("default") ));
/*! \brief Get the cpulist of a region
@param [in] region ID of region
@param [in] count Length of cpulist array
@param [in,out] cpulist cpulist array
@return Number of threads of region or count, whatever is lower
*/
extern int perfmon_getCpulistOfRegion(int region, int count, int* cpulist)  __attribute__ ((visibility ("default") ));
/*! \brief Get the accumulated measurement time of a region for a thread
@param [in] region ID of region
@param [in] thread ID of thread
@return Measurement time of a region for a thread
*/
extern double perfmon_getTimeOfRegion(int region, int thread) __attribute__ ((visibility ("default") ));
/*! \brief Get the call count of a region for a thread
@param [in] region ID of region
@param [in] thread ID of thread
@return Call count of a region for a thread
*/
extern int perfmon_getCountOfRegion(int region, int thread) __attribute__ ((visibility ("default") ));
/*! \brief Get the event result of a region for an event and thread
@param [in] region ID of region
@param [in] event ID of event
@param [in] thread ID of thread
@return Result of a region for an event and thread
*/
extern double perfmon_getResultOfRegionThread(int region, int event, int thread) __attribute__ ((visibility ("default") ));
/*! \brief Get the metric result of a region for a metric and thread
@param [in] region ID of region
@param [in] metricId ID of metric
@param [in] threadId ID of thread
@return Metric result of a region for a thread
*/
extern double perfmon_getMetricOfRegionThread(int region, int metricId, int threadId) __attribute__ ((visibility ("default") ));

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
extern void timer_init( void ) __attribute__ ((visibility ("default") ));
/*! \brief Return the measured interval in seconds

@param [in] time Structure holding the cycle count at start and stop
@return Time in seconds
*/
extern double timer_print( const TimerData* time) __attribute__ ((visibility ("default") ));
/*! \brief Return the measured interval in cycles

@param [in] time Structure holding the cycle count at start and stop
@return Time in cycles
*/
extern uint64_t timer_printCycles( const TimerData* time) __attribute__ ((visibility ("default") ));
/*! \brief Reset values in TimerData

@param [in] time Structure holding the cycle count at start and stop
*/
extern void timer_reset( TimerData* time ) __attribute__ ((visibility ("default") ));
/*! \brief Return the CPU clock determined at timer_init

@return CPU clock
*/
extern uint64_t timer_getCpuClock( void ) __attribute__ ((visibility ("default") ));
/*! \brief Return the current CPU clock read from sysfs

@return CPU clock
*/
extern uint64_t timer_getCpuClockCurrent( int cpu_id ) __attribute__ ((visibility ("default") ));
/*! \brief Return the cycles clock determined at timer_init

@return cycle clock
*/
extern uint64_t timer_getCycleClock( void ) __attribute__ ((visibility ("default") ));
/*! \brief Return the baseline CPU clock determined at timer_init

@return Baseline CPU clock
*/
extern uint64_t timer_getBaseline( void ) __attribute__ ((visibility ("default") ));
/*! \brief Start time measurement

@param [in,out] time Structure holding the cycle count at start
*/
extern void timer_start( TimerData* time ) __attribute__ ((visibility ("default") ));
/*! \brief Stop time measurement

@param [in,out] time Structure holding the cycle count at stop
*/
extern void timer_stop ( TimerData* time) __attribute__ ((visibility ("default") ));
/*! \brief Sleep for specified usecs

@param [in] usec Amount of usecs to sleep
*/
extern int timer_sleep(unsigned long usec) __attribute__ ((visibility ("default") ));

/*! \brief Finalize timer module

*/
extern void timer_finalize(void) __attribute__ ((visibility ("default") ));

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
extern const char* power_names[NUM_POWER_DOMAINS] __attribute__ ((visibility ("default") ));

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
    double uncoreMinFreq; /*!< \brief Minimal uncore frequency */
    double uncoreMaxFreq; /*!< \brief Maximal uncore frequency */
    uint8_t perfBias; /*!< \brief Performance energy bias */
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

/*! \brief Initialize energy measurements on specific CPU

Additionally, it reads basic information about the energy measurements like
minimal measurement time.
@param [in] cpuId Initialize energy facility for this CPU
@return error code
*/
extern int power_init(int cpuId) __attribute__ ((visibility ("default") ));
/*! \brief Get a pointer to the energy facility information

@return PowerInfo_t pointer
\sa PowerInfo_t
*/
extern PowerInfo_t get_powerInfo(void) __attribute__ ((visibility ("default") ));
/*! \brief Read the current power value

@param [in] cpuId Read energy facility for this CPU
@param [in] reg Energy register
@param [out] data Energy data
*/
extern int power_read(int cpuId, uint64_t reg, uint32_t *data) __attribute__ ((visibility ("default") ));
/*! \brief Read the current energy value using a specific communication socket

@param [in] socket_fd Communication socket for the read operation
@param [in] cpuId Read energy facility for this CPU
@param [in] reg Energy register
@param [out] data Energy data
*/
extern int power_tread(int socket_fd, int cpuId, uint64_t reg, uint32_t *data) __attribute__ ((visibility ("default") ));
/*! \brief Start energy measurements

@param [in,out] data Data structure holding start and stop values for energy measurements
@param [in] cpuId Start energy facility for this CPU
@param [in] type Which type should be measured
@return error code
*/
extern int power_start(PowerData_t data, int cpuId, PowerType type) __attribute__ ((visibility ("default") ));
/*! \brief Stop energy measurements

@param [in,out] data Data structure holding start and stop values for energy measurements
@param [in] cpuId Start energy facility for this CPU
@param [in] type Which type should be measured
@return error code
*/
extern int power_stop(PowerData_t data, int cpuId, PowerType type) __attribute__ ((visibility ("default") ));
/*! \brief Print energy measurements gathered by power_start() and power_stop()

@param [in] data Data structure holding start and stop values for energy measurements
@return Consumed energy in Joules
*/
extern double power_printEnergy(const PowerData* data) __attribute__ ((visibility ("default") ));
/*! \brief Get energy Unit

@param [in] domain RAPL domain ID
@return Energy unit of the given RAPL domain
*/
extern double power_getEnergyUnit(int domain) __attribute__ ((visibility ("default") ));

/*! \brief Get the values of the limit register of a domain
NOT IMPLEMENTED

@param [in] cpuId CPU ID
@param [in] domain RAPL domain ID
@param [out] power Energy limit
@param [out] time Time limit
@return error code
*/
int power_limitGet(int cpuId, PowerType domain, double* power, double* time) __attribute__ ((visibility ("default") ));

/*! \brief Set the values of the limit register of a domain
NOT IMPLEMENTED

@param [in] cpuId CPU ID
@param [in] domain RAPL domain ID
@param [in] power Energy limit
@param [in] time Time limit
@param [in] doClamping Activate clamping (going below OS-requested power level)
@return error code
*/
int power_limitSet(int cpuId, PowerType domain, double power, double time, int doClamping) __attribute__ ((visibility ("default") ));

/*! \brief Get the state of a energy limit, activated or deactivated
NOT IMPLEMENTED

@param [in] cpuId CPU ID
@param [in] domain RAPL domain ID
@return state, 1 for active, 0 for inactive
*/
int power_limitState(int cpuId, PowerType domain) __attribute__ ((visibility ("default") ));

/*! \brief Free space of power_unit
*/
extern void power_finalize(void) __attribute__ ((visibility ("default") ));
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
extern void thermal_init(int cpuId) __attribute__ ((visibility ("default") ));
/*! \brief Read the current thermal value

@param [in] cpuId Read thermal facility for this CPU
@param [out] data Thermal data
*/
extern int thermal_read(int cpuId, uint32_t *data) __attribute__ ((visibility ("default") ));
/*! \brief Read the current thermal value using a specific communication socket

@param [in] socket_fd Communication socket for the read operation
@param [in] cpuId Read thermal facility for this CPU
@param [out] data Thermal data
*/
extern int thermal_tread(int socket_fd, int cpuId, uint32_t *data) __attribute__ ((visibility ("default") ));
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
extern void memsweep_domain(int domainId) __attribute__ ((visibility ("default") ));
/*! \brief Sweeping the memory of all NUMA nodes covered by CPU list

Sweeps (zeros) the memory of all NUMA nodes containing the CPUs in \a processorList
@param [in] processorList List of CPU IDs
@param [in] numberOfProcessors Number of CPUs in list
*/
extern void memsweep_threadGroup(const int* processorList, int numberOfProcessors) __attribute__ ((visibility ("default") ));
/** @}*/

/*
################################################################################
# CPU feature related functions
################################################################################
*/
/** \addtogroup CpuFeatures Retrieval and manipulation of processor features
 *  @{
 */
/*! \brief Enumeration of all CPU related features.
*/
typedef enum {
    FEAT_HW_PREFETCHER=0, /*!< \brief Hardware prefetcher */
    FEAT_CL_PREFETCHER, /*!< \brief Adjacent cache line prefetcher */
    FEAT_DCU_PREFETCHER, /*!< \brief DCU L1 data cache prefetcher */
    FEAT_IP_PREFETCHER, /*!< \brief IP L1 data cache prefetcher */
    FEAT_FAST_STRINGS, /*!< \brief Fast-strings feature */
    FEAT_THERMAL_CONTROL, /*!< \brief Automatic Thermal Control Circuit */
    FEAT_PERF_MON, /*!< \brief Hardware performance monitoring */
    FEAT_FERR_MULTIPLEX, /*!< \brief FERR# Multiplexing, must be 1 for XAPIC interrupt model */
    FEAT_BRANCH_TRACE_STORAGE, /*!< \brief Branch Trace Storage */
    FEAT_XTPR_MESSAGE, /*!< \brief xTPR Message to set processor priority */
    FEAT_PEBS, /*!< \brief Precise Event Based Sampling (PEBS) */
    FEAT_SPEEDSTEP, /*!< \brief Enhanced Intel SpeedStep Technology to reduce energy consumption*/
    FEAT_MONITOR, /*!< \brief MONITOR/MWAIT feature to monitor write-back stores*/
    FEAT_SPEEDSTEP_LOCK, /*!< \brief Enhanced Intel SpeedStep Technology Select Lock */
    FEAT_CPUID_MAX_VAL, /*!< \brief Limit CPUID Maxval */
    FEAT_XD_BIT, /*!< \brief Execute Disable Bit */
    FEAT_DYN_ACCEL, /*!< \brief Intel Dynamic Acceleration */
    FEAT_TURBO_MODE, /*!< \brief Intel Turbo Mode */
    FEAT_TM2, /*!< \brief Thermal Monitoring 2 */
    CPUFEATURES_MAX
} CpuFeature;

/*! \brief Initialize the internal feature variables for all CPUs

Initialize the internal feature variables for all CPUs
*/
extern void cpuFeatures_init() __attribute__ ((visibility ("default") ));
/*! \brief Print state of all CPU features for a given CPU

Print state of all CPU features for a given CPU
@param [in] cpu CPU ID
*/
extern void cpuFeatures_print(int cpu) __attribute__ ((visibility ("default") ));
/*! \brief Get state of a CPU feature for a given CPU

Get state of a CPU feature for a given CPU
@param [in] cpu CPU ID
@param [in] type CPU feature
@return State of CPU feature (1=enabled, 0=disabled)
*/
extern int cpuFeatures_get(int cpu, CpuFeature type)  __attribute__ ((visibility ("default") ));
/*! \brief Get the name of a CPU feature

Get the name of a CPU feature
@param [in] type CPU feature
@return Name of the CPU feature or NULL if feature is not available
*/
extern char* cpuFeatures_name(CpuFeature type)  __attribute__ ((visibility ("default") ));
/*! \brief Enable a CPU feature for a specific CPU

Enable a CPU feature for a specific CPU. Only the state of the prefetchers can be changed, all other features return -EINVAL
@param [in] cpu CPU ID
@param [in] type CPU feature
@param [in] print Print outcome of operation
@return Status of operation (0=success, all others are erros, either by MSR access or invalid feature)
*/
extern int cpuFeatures_enable(int cpu, CpuFeature type, int print) __attribute__ ((visibility ("default") ));
/*! \brief Disable a CPU feature for a specific CPU

Disable a CPU feature for a specific CPU. Only the state of the prefetchers can be changed, all other features return -EINVAL
@param [in] cpu CPU ID
@param [in] type CPU feature
@param [in] print Print outcome of operation
@return Status of operation (0=success, all others are erros, either by MSR access or invalid feature)
*/
extern int cpuFeatures_disable(int cpu, CpuFeature type, int print) __attribute__ ((visibility ("default") ));
/** @}*/


/*
################################################################################
# CPU frequency related functions
################################################################################
*/
/** \addtogroup CpuFreq Retrieval and manipulation of processor clock frequencies
 *  @{
 */
/*! \brief Get the current clock frequency of a core

Get the current clock frequency of a core
@param [in] cpu_id CPU ID
@return Frequency or 0 in case of errors
*/
extern uint64_t freq_getCpuClockCurrent(const int cpu_id ) __attribute__ ((visibility ("default") ));
/*! \brief Set the current clock frequency of a core

Set the current clock frequency of a core
@param [in] cpu_id CPU ID
@param [in] freq Frequency in kHz
@return Frequency or 0 in case of errors
*/
extern uint64_t freq_setCpuClockCurrent(const int cpu_id, const uint64_t freq) __attribute__ ((visibility ("default") ));
/*! \brief Get the maximal clock frequency of a core

Get the maximal clock frequency of a core
@param [in] cpu_id CPU ID
@return Frequency or 0 in case of errors
*/
extern uint64_t freq_getCpuClockMax(const int cpu_id ) __attribute__ ((visibility ("default") ));
/*! \brief Set the maximal clock frequency of a core

Set the maximal clock frequency of a core
@param [in] cpu_id CPU ID
@param [in] freq Frequency in kHz
@return Frequency or 0 in case of errors
*/
extern uint64_t freq_setCpuClockMax(const int cpu_id, const uint64_t freq) __attribute__ ((visibility ("default") ));
/*! \brief Get the minimal clock frequency of a core

Get the minimal clock frequency of a core
@param [in] cpu_id CPU ID
@return Frequency or 0 in case of errors
*/
extern uint64_t freq_getCpuClockMin(const int cpu_id ) __attribute__ ((visibility ("default") ));
/*! \brief Set the minimal clock frequency of a core

Set the minimal clock frequency of a core
@param [in] cpu_id CPU ID
@param [in] freq Frequency in kHz
@return Frequency or 0 in case of errors
*/
extern uint64_t freq_setCpuClockMin(const int cpu_id, const uint64_t freq) __attribute__ ((visibility ("default") ));
/*! \brief Get the frequency governor of a core

Get the frequency governor of a core. The returned string must be freed by the caller.
@param [in] cpu_id CPU ID
@return Governor or NULL in case of errors
*/
extern char * freq_getGovernor(const int cpu_id ) __attribute__ ((visibility ("default") ));
/*! \brief Set the frequency governor of a core

Set the frequency governor of a core.
@param [in] cpu_id CPU ID
@param [in] gov Governor
@return 1 or 0 in case of errors
*/
extern int freq_setGovernor(const int cpu_id, const char* gov) __attribute__ ((visibility ("default") ));
/*! \brief Get the available frequencies of a core

Get the available frequencies of a core. The returned string must be freed by the caller.
@param [in] cpu_id CPU ID
@return String with available frequencies or NULL in case of errors
*/
extern char * freq_getAvailFreq(const int cpu_id ) __attribute__ ((visibility ("default") ));
/*! \brief Get the available frequency governors of a core

Get the available frequency governors of a core. The returned string must be freed by the caller.
@param [in] cpu_id CPU ID
@return String with available frequency governors or NULL in case of errors
*/
extern char * freq_getAvailGovs(const int cpu_id ) __attribute__ ((visibility ("default") ));
/*! \brief Get the name of the currently active cpufreq driver

Get the name of the currently active cpufreq driver. The returned string must be freed by the caller.
@param [in] cpu_id CPU ID
@return String with active cpufreq driver or NULL in case of errors
*/
extern char * freq_getDriver(const int cpu_id ) __attribute__ ((visibility ("default") ));

/*! \brief Set the minimal Uncore frequency

Set the minimal Uncore frequency. Since the ranges are not documented, valid frequencies are from minimal CPU clock to maximal Turbo clock. If selecting a frequency at the borders, please check the result with the UNCORE_CLOCK event to be effective.
@param [in] socket_id ID of socket
@param [in] freq Frequency in MHz
@return 0 for success, -ERROR at failure
*/
extern int freq_setUncoreFreqMin(const int socket_id, const uint64_t freq) __attribute__ ((visibility ("default") ));

/*! \brief Get the minimal Uncore frequency

Get the minimal Uncore frequency.
@param [in] socket_id ID of socket
@return frequency in MHz or 0 at failure
*/
extern uint64_t freq_getUncoreFreqMin(const int socket_id) __attribute__ ((visibility ("default") ));

/*! \brief Set the maximal Uncore frequency

Set the maximal Uncore frequency. Since the ranges are not documented, valid frequencies are from minimal CPU clock to maximal Turbo clock. If selecting a frequency at the borders, please check the result with the UNCORE_CLOCK event to be effective.
@param [in] socket_id ID of socket
@param [in] freq Frequency in MHz
@return 0 for success, -ERROR at failure
*/
extern int freq_setUncoreFreqMax(const int socket_id, const uint64_t freq) __attribute__ ((visibility ("default") ));

/*! \brief Get the maximal Uncore frequency

Get the maximal Uncore frequency.
@param [in] socket_id ID of socket
@return frequency in MHz or 0 at failure
*/
extern uint64_t freq_getUncoreFreqMax(const int socket_id) __attribute__ ((visibility ("default") ));
/** @}*/

#ifdef __cplusplus
}
#endif

#endif /*LIKWID_H*/
