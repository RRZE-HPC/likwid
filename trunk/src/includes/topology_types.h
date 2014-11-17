/*
 * =======================================================================================
 *
 *      Filename:  topology_types.h
 *
 *      Description:  Types file for cpuid module.
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

#ifndef CPUID_TYPES_H
#define CPUID_TYPES_H

/** \addtogroup CPUTopology CPU information module
*  @{
*/

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

/*! \brief Enum of possible CPU features

CPUs implement different features that likely improve application performance if
optimized using the feature. The list contains all features that are currently 
supported by LIKWID. LIKWID does not perform any action based on these features,
it gathers the data only for output purposes. It is not a complete list.
\extends CpuInfo
*/
typedef enum {
    SSE3=0, /*!< \brief Streaming SIMD Extensions 3 */
    MMX, /*!< \brief Multi Media Extension */
    SSE, /*!< \brief Streaming SIMD Extensions */
    SSE2, /*!< \brief Streaming SIMD Extensions 2 */
    MONITOR, /*!< \brief MONITOR and MWAIT instructions (part of SSE3) */
    ACPI, /*!< \brief Advanced Configuration and Power Interface */
    RDTSCP, /*!< \brief Serializing Read of the Time Stamp Counter */
    VMX, /*!< \brief Virtual Machine eXtensions (VT-x) */
    EIST, /*!< \brief Enhanced Intel SpeedStep */
    TM, /*!< \brief Thermal Monitor */
    TM2, /*!< \brief Thermal Monitor 2 */
    AES, /*!< \brief AES instruction set */
    RDRAND, /*!< \brief Random numbers from an on-chip hardware random number generator */
    SSSE3, /*!< \brief Supplemental Streaming SIMD Extensions 3 */
    SSE41, /*!< \brief Streaming SIMD Extensions 4.1 */
    SSE42, /*!< \brief Streaming SIMD Extensions 4.2 */
    AVX, /*!< \brief Advanced Vector Extensions */
    FMA /*!< \brief Fused multiply-add (FMA3) */
} FeatureBit;


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
} HWThread;

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
    uint32_t numSockets; /*!< \brief Amount of CPU sockets/packages in the system */
    uint32_t numCoresPerSocket; /*!< \brief Amount of physical cores in one CPU socket/package */
    uint32_t numThreadsPerCore; /*!< \brief Amount of HW threads in one physical CPU core */
    uint32_t numCacheLevels; /*!< \brief Amount of caches for each HW thread and length of \a cacheLevels */
    HWThread* threadPool; /*!< \brief List of all HW thread descriptions */
    CacheLevel*  cacheLevels; /*!< \brief List of all caches in the hierarchy */
    struct treeNode* topologyTree; /*!< \brief Anchor for a tree structure describing the system topology */
} CpuTopology;

/** \brief Pointer for exporting the CpuInfo data structure */
typedef CpuInfo* CpuInfo_t;
/** \brief Pointer for exporting the CpuTopology data structure */
typedef CpuTopology* CpuTopology_t;


/** @}*/
#endif /*CPUID_TYPES_H*/
