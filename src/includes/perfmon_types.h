/*
 * =======================================================================================
 *
 *      Filename:  perfmon_types.h
 *
 *      Description:  Header File of perfmon module.
 *                    Configures and reads out performance counters
 *                    on x86 based architectures. Supports multi threading.
 *
 *      Version:   5.4.0
 *      Released:  15.11.2024
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2024 RRZE, University Erlangen-Nuremberg
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

#ifndef PERFMON_TYPES_H
#define PERFMON_TYPES_H

#include <bstrlib.h>
#include <timer.h>
#include <inttypes.h>
#include <perfgroup.h>

#define MAX_EVENT_OPTIONS NUM_EVENT_OPTIONS

/* #####   EXPORTED TYPE DEFINITIONS   #################################### */

/** \addtogroup PerfMon
 *  @{
 */
/////////////////////////////////////////////

/*! \brief Enum of possible event and counter options

List of internally used IDs for all event and counter options that are supported
by LIKWID.
\extends PerfmonEventOption
*/
typedef enum {
    EVENT_OPTION_NONE = 0, /*!< \brief No option, used as False value */
    EVENT_OPTION_OPCODE, /*!< \brief Match opcode */
    EVENT_OPTION_MATCH0, /*!< \brief Match0 register */
    EVENT_OPTION_MATCH1, /*!< \brief Match1 register */
    EVENT_OPTION_MATCH2, /*!< \brief Match2 register */
    EVENT_OPTION_MATCH3, /*!< \brief Match3 register */
    EVENT_OPTION_MASK0, /*!< \brief Mask0 register */
    EVENT_OPTION_MASK1, /*!< \brief Mask1 register */
    EVENT_OPTION_MASK2, /*!< \brief Mask2 register */
    EVENT_OPTION_MASK3, /*!< \brief Mask3 register */
    EVENT_OPTION_NID, /*!< \brief Set NUMA node IDs */
    EVENT_OPTION_TID, /*!< \brief Set Thread IDs */
    EVENT_OPTION_CID, /*!< \brief Set Core IDs */
    EVENT_OPTION_SLICE, /*!< \brief Set Slice IDs, often when the L3 is assembled with different sections */
    EVENT_OPTION_STATE, /*!< \brief Match for state */
    EVENT_OPTION_EDGE, /*!< \brief Increment counter at each edge */
    EVENT_OPTION_THRESHOLD, /*!< \brief Increment only if exceeding threshold */
    EVENT_OPTION_INVERT, /*!< \brief Invert behavior of EVENT_OPTION_THRESHOLD, hence increment only below threshold */
    EVENT_OPTION_COUNT_KERNEL, /*!< \brief Also count events when in kernel space */
    EVENT_OPTION_ANYTHREAD, /*!< \brief Increment counter at events of all HW threads in the core */
    EVENT_OPTION_OCCUPANCY, /*!< \brief Count occupancy not occurrences */
    EVENT_OPTION_OCCUPANCY_FILTER, /*!< \brief Filter for occupancy counting */
    EVENT_OPTION_OCCUPANCY_EDGE, /*!< \brief Increment occupancy counter at detection of an edge */
    EVENT_OPTION_OCCUPANCY_INVERT, /*!< \brief Invert filter for occupancy counting */
    EVENT_OPTION_IN_TRANS, /*!< \brief Count events during transactions */
    EVENT_OPTION_IN_TRANS_ABORT, /*!< \brief Count events that aborted during transactions */
    EVENT_OPTION_GENERIC_CONFIG, /*!< \brief Configuration bitmask for generic event */
    EVENT_OPTION_GENERIC_UMASK, /*!< \brief Umask bitmask for generic event */
#ifdef LIKWID_USE_PERFEVENT
    EVENT_OPTION_PERF_PID, /*!< \brief PID parameter to use in the perf_event_open call */
    EVENT_OPTION_PERF_FLAGS, /*!< \brief FLAGS parameters to use in the perf_event_open call */
#endif
#ifdef _ARCH_PPC
    EVENT_OPTION_PMC, /*!< \brief Specify which PMC counter should be used */
    EVENT_OPTION_PMCXSEL,
    EVENT_OPTION_UNCORE_CONFIG, /*!< \brief Configuration bitmask for event ID for NEST (Uncore) measurements */

#endif
    NUM_EVENT_OPTIONS /*!< \brief Amount of defined options */
} EventOptionType;

/*! \brief Enum of possible states of an event group

List of states for event groups
*/
typedef enum {
    STATE_NONE = 0, /*!< \brief Not configured, not started and not stopped */
    STATE_SETUP, /*!< \brief The event set hold by group is configured */
    STATE_START, /*!< \brief The event set hold by group is current running */
} GroupState;

/*! \brief List of option names

List of strings for all event and counter options used for matching and output
*/
extern char* eventOptionTypeName[NUM_EVENT_OPTIONS];

/** \brief Bitmask with no event/counter option set */
#define EVENT_OPTION_NONE_MASK 0x0ULL
/** \brief Define for easily creating an bitmask of all configured event/counter options */
#define OPTIONS_TYPE_MASK(type) \
        (((type == EVENT_OPTION_NONE)||(type >= NUM_EVENT_OPTIONS)) ? \
        EVENT_OPTION_NONE_MASK : \
        (1ULL<<type))


/** @cond */
#define EVENT_OPTION_OPCODE_MASK (1ULL<<EVENT_OPTION_OPCODE)
#define EVENT_OPTION_MATCH0_MASK (1ULL<<EVENT_OPTION_MATCH0)
#define EVENT_OPTION_MATCH1_MASK (1ULL<<EVENT_OPTION_MATCH1)
#define EVENT_OPTION_MATCH2_MASK (1ULL<<EVENT_OPTION_MATCH2)
#define EVENT_OPTION_MATCH3_MASK (1ULL<<EVENT_OPTION_MATCH3)
#define EVENT_OPTION_MASK0_MASK (1ULL<<EVENT_OPTION_MASK0)
#define EVENT_OPTION_MASK1_MASK (1ULL<<EVENT_OPTION_MASK1)
#define EVENT_OPTION_MASK2_MASK (1ULL<<EVENT_OPTION_MASK2)
#define EVENT_OPTION_MASK3_MASK (1ULL<<EVENT_OPTION_MASK3)
#define EVENT_OPTION_NID_MASK (1ULL<<EVENT_OPTION_NID)
#define EVENT_OPTION_TID_MASK (1ULL<<EVENT_OPTION_TID)
#define EVENT_OPTION_CID_MASK (1ULL<<EVENT_OPTION_CID)
#define EVENT_OPTION_SLICE_MASK (1ULL<<EVENT_OPTION_SLICE)
#define EVENT_OPTION_STATE_MASK (1ULL<<EVENT_OPTION_STATE)
#define EVENT_OPTION_EDGE_MASK (1ULL<<EVENT_OPTION_EDGE)
#define EVENT_OPTION_THRESHOLD_MASK (1ULL<<EVENT_OPTION_THRESHOLD)
#define EVENT_OPTION_INVERT_MASK (1ULL<<EVENT_OPTION_INVERT)
#define EVENT_OPTION_COUNT_KERNEL_MASK (1ULL<<EVENT_OPTION_COUNT_KERNEL)
#define EVENT_OPTION_ANYTHREAD_MASK (1ULL<<EVENT_OPTION_ANYTHREAD)
#define EVENT_OPTION_OCCUPANCY_MASK (1ULL<<EVENT_OPTION_OCCUPANCY)
#define EVENT_OPTION_OCCUPANCY_FILTER_MASK (1ULL<<EVENT_OPTION_OCCUPANCY_FILTER)
#define EVENT_OPTION_OCCUPANCY_EDGE_MASK (1ULL<<EVENT_OPTION_OCCUPANCY_EDGE)
#define EVENT_OPTION_OCCUPANCY_INVERT_MASK (1ULL<<EVENT_OPTION_OCCUPANCY_INVERT)
#define EVENT_OPTION_IN_TRANS_MASK (1ULL<<EVENT_OPTION_IN_TRANS)
#define EVENT_OPTION_IN_TRANS_ABORT_MASK (1ULL<<EVENT_OPTION_IN_TRANS_ABORT)
#define EVENT_OPTION_GENERIC_CONFIG_MASK (1ULL<<EVENT_OPTION_GENERIC_CONFIG)
#define EVENT_OPTION_GENERIC_UMASK_MASK (1ULL<<EVENT_OPTION_GENERIC_UMASK)
/** @endcond */

/*! \brief Structure specifying thread to CPU relation

Threads are always numbered incrementally. This structure is used in order to
resolve the real HW thread ID.
\extends PerfmonGroupSet
*/
typedef struct {
    int             thread_id; /*!< \brief Thread ID how it is used internally */
    int             processorId; /*!< \brief Real HW thread ID */
} PerfmonThread;

/*! \brief Structure specifying event/counter options and their value

Most options set a bitfield in registers and their values are stored in this structure.
If an option is a binary option, the value is set to 1.
\extends PerfmonEvent
*/
typedef struct {
    EventOptionType      type; /*!< \brief Type of the option */
    uint64_t             value; /*!< \brief Value of the option */
} PerfmonEventOption;

/*! \brief Structure specifying an performance monitoring event

This structure holds the configuration data for an event. It groups the name,
the allowed counters and internally used values like event ID and masks. Moreover
the event options are hold here.
\extends PerfmonEventSetEntry
*/
typedef struct {
    const char*     name; /*!< \brief Name of the event */
    char*           limit; /*!< \brief Valid counters for the event */
    uint64_t        eventId; /*!< \brief ID of the event */
    uint64_t        umask; /*!< \brief Most events need to specify a mask to limit counting */
    uint64_t        cfgBits; /*!< \brief Misc configuration bits */
    uint64_t        cmask; /*!< \brief Misc mask bits */
    uint64_t         numberOfOptions; /*!< \brief Number of options for the event */
    uint64_t        optionMask; /*!< \brief Bitmask for fast check of set options */
    PerfmonEventOption options[NUM_EVENT_OPTIONS]; /*!< \brief List of options */
} PerfmonEvent;

/*! \brief Structure describing performance monitoring counter data

Each event holds one of these structures for each thread to store the counter
data, if it is configured and the amount of happened overflows.
\extends PerfmonEventSetEntry
*/
typedef struct {
    int         init; /*!< \brief Flag if corresponding control register is set up properly */
    int         id; /*!< \brief Offset in higher level control register, e.g. position of enable bit */
    int         overflows; /*!< \brief Amount of overflows */
    uint64_t    startData; /*!< \brief Start data from the counter */
    uint64_t    counterData; /*!< \brief Intermediate data from the counters */
    double      lastResult; /*!< \brief Last measurement result*/
    double      fullResult; /*!< \brief Aggregated measurement result */
#if defined(__x86_64__) || defined(__i386__) || defined(__ARM_ARCH_8A) || defined(__ARM_ARCH_7A__)
    uint64_t    _padding[2]; /*!< \brief Padding to one  64B cache line */
#endif
#if defined(_ARCH_PPC)
    uint64_t    _padding[10]; /*!< \brief Padding to one 128B cache line */
#endif
} PerfmonCounter;


/*! \brief Structure specifying an performance monitoring event

An eventSet consists of an event and a counter and the read counter values.
\extends PerfmonEventSet
*/
typedef struct {
    PerfmonEvent        event; /*!< \brief Event configuration */
    RegisterIndex       index; /*!< \brief Index of the counter register in the counter map */
    RegisterType        type; /*!< \brief Type of the counter register and event */
    PerfmonCounter*     threadCounter; /*!< \brief List of counter data for each thread, list length is \a numberOfThreads in PerfmonGroupSet */
} PerfmonEventSetEntry;

/*! \brief Structure specifying an performance monitoring event group

A PerfmonEventSet holds a set of event and counter combinations and some global information about all eventSet entries
\extends PerfmonGroupSet
*/
typedef struct {
    int                   numberOfEvents; /*!< \brief Number of eventSets in \a events */
    PerfmonEventSetEntry* events; /*!< \brief List of eventSets */
    TimerData             timer; /*!< \brief Time information how long the counters were running */
    double                rdtscTime; /*!< \brief Evaluation of the Time information in seconds */
    double                runTime; /*!< \brief Sum of all time information in seconds that the group was running */
    uint64_t              regTypeMask1; /*!< \brief Bitmask1 for easy checks which types are included in the eventSet */
    uint64_t              regTypeMask2; /*!< \brief Bitmask2 for easy checks which types are included in the eventSet */
    uint64_t              regTypeMask3; /*!< \brief Bitmask3 for easy checks which types are included in the eventSet */
    uint64_t              regTypeMask4; /*!< \brief Bitmask4 for easy checks which types are included in the eventSet */
    uint64_t              regTypeMask5; /*!< \brief Bitmask5 for easy checks which types are included in the eventSet */
    uint64_t              regTypeMask6; /*!< \brief Bitmask6 for easy checks which types are included in the eventSet */
    GroupState            state; /*!< \brief Current state of the event group (configured, started, none) */
    GroupInfo             group; /*!< \brief Structure holding the performance group information */
} PerfmonEventSet;

/*! \brief Structure specifying all performance monitoring event groups

The global PerfmonGroupSet structure holds all eventSets and threads that are
configured to measure. Only one eventSet can be measured at a time but the groups
can be switched to perform some kind of multiplexing.
*/
typedef struct {
    int              numberOfGroups; /*!< \brief List length of \a groups*/
    int              numberOfActiveGroups; /*!< \brief Amount of added eventSets. Only those eventSets can be accessed in \a groups. */
    int              activeGroup; /*!< \brief Currently active eventSet */
    PerfmonEventSet* groups; /*!< \brief List of eventSets */
    int              numberOfThreads; /*!< \brief Amount of threads in \a threads */
    PerfmonThread*   threads; /*!< \brief List of threads */
} PerfmonGroupSet;

/** \brief List of counter with name, config register, counter registers and
if needed PCI device */
extern RegisterMap* counter_map;
/** \brief List of boxes with name, config register, counter registers and if
needed PCI device. Mainly used in Uncore handling but also core-local counters
are defined as a box. */
extern BoxMap* box_map;
/** \brief List of events available for the current architecture */
extern PerfmonEvent* eventHash;
/** \brief List of PCI devices available for the current architecture */
extern PciDevice* pci_devices;
/** @}*/

/* perfmon datatypes */
extern PerfmonGroupSet *groupSet;
extern int perfmon_numCounters;
extern int perfmon_numCoreCounters;
extern int perfmon_numUncoreCounters;
extern int perfmon_numArchEvents;

/*! \brief Structure holding information for Intel's uncore discovery mechanism introduced with SapphireRapids */
typedef struct {
    char* name; /*!< \brief Name of unit */
    int discovery_type; /*!< \brief Intel's device discovery type */
    int max_devices; /*!< \brief Maximal amont of devices for this discovery type */
    PciDeviceIndex base_device; /*!< \brief LIKWID unit device base */
} PerfmonUncoreDiscovery;


#endif /*PERFMON_TYPES_H*/
