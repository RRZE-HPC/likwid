/*
 * =======================================================================================
 *
 *      Filename:  nvmon_types.h
 *
 *      Description:  Header File of nvmon module.
 *                    Configures and reads out performance counters
 *                    on NVIDIA GPUs. Supports multi GPUs.
 *
 *      Version:   5.1.1
 *      Released:  31.03.2021
 *
 *      Author:   Thomas Gruber (tg), thomas.gruber@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 RRZE, University Erlangen-Nuremberg
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
#ifndef LIKWID_NVMON_TYPES_H
#define LIKWID_NVMON_TYPES_H

#include <perfmon_types.h>
#include <bstrlib.h>
#include <inttypes.h>
#include <perfgroup.h>

#include <cupti.h>

#define NVMON_DEFAULT_STR_LEN 1024

typedef enum {
    NVMON_CUPTI_EVENT,
    NVMON_NVML_EVENT,
    NVMON_PERFWORKS_EVENT,
    NVMON_NO_TYPE
} NvmonEventType;

typedef enum {
    ENTITY_EVENT_TYPE_MONOTONIC = 0,
    ENTITY_TYPE_INSTANT,
} NvmonEventResultType;

typedef struct {
    double startValue;
    double stopValue;
    double currentValue;
    double lastValue;
    double fullValue;
    double maxValue;
    int overflows;
    NvmonEventResultType type;
} NvmonEventResult;

typedef struct {
    int domainId;
    CUpti_EventDomainID cuDomainId;
    int eventId;
    CUpti_EventID cuEventId;
    char name[NVMON_DEFAULT_STR_LEN];
    char real[NVMON_DEFAULT_STR_LEN];
    char description[NVMON_DEFAULT_STR_LEN];
    int active;
    NvmonEventType type;
} NvmonEvent;
typedef NvmonEvent* NvmonEvent_t;

typedef struct {
    int eventId;
    int groupId;
    int idxInSet;
    int deviceId;
    uint32_t numInstances;
    uint32_t numTotalInstances;
    CUpti_EventID cuEventId;
    CUpti_EventDomainID cuDomainId;
    CUpti_EventGroup cuGroup;
    CUpti_EventGroupSet *cuGroupSet;

} NvmonActiveEvent;
typedef NvmonActiveEvent* NvmonActiveEvent_t;

/*! \brief Structure specifying an performance monitoring event group

A NvmonEventSet holds a set of event and counter combinations and some global information about all eventSet entries
\extends NvmonGroupSet
*/
typedef struct {
    int                   id; /*!< \brief ID of event set */
    int                   numberOfEvents; /*!< \brief Number of events in \a events */
    NvmonEvent_t*         nvEvents; /*!< \brief List of events with length numberOfEvents */
    NvmonEventResult*     results; /* \brief List of event results with length numberOfEvents */
    CUpti_EventID*        cuEventIDs; /*!< \brief List of CUPTI event IDs with length numberOfEvents */
    TimerData             timer; /*!< \brief Time information how long the counters were running */
    double                rdtscTime; /*!< \brief Evaluation of the Time information in seconds */
    double                runTime; /*!< \brief Sum of all time information in seconds that the group was running */
    struct bstrList*      events;
    uint32_t numStages;
    uint8_t* configImage;
    size_t configImageSize;
    uint8_t* counterDataImage;
    size_t counterDataImageSize;
    uint8_t* counterDataScratchBuffer;
    size_t counterDataScratchBufferSize;
    uint8_t* counterDataImagePrefix;
    size_t counterDataImagePrefixSize;
    uint8_t* counterAvailabilityImage;
    size_t counterAvailabilityImageSize;
} NvmonEventSet;


typedef enum {
    LIKWID_NVMON_CUPTI_BACKEND = 0,
    LIKWID_NVMON_PERFWORKS_BACKEND,
} NvmonBackends;

typedef struct {
    int deviceId;
    CUdevice cuDevice;
    CUcontext context;
    int numEventSets;
    int activeEventSet;
    CUpti_EventGroupSets *cuEventSets;
    GHashTable* eventHash;
    GHashTable* evIdHash;
    int numAllEvents;
    NvmonEvent_t *allevents;
    int numNvEventSets;
    NvmonEventSet* nvEventSets;
    char *name;
    char *chip;
    int numActiveEvents;
    NvmonActiveEvent *activeEvents;
    int numActiveCuGroups;
    CUpti_EventGroupSet **activeCuGroups;
    uint64_t timeStart;
    uint64_t timeRead;
    uint64_t timeStop;
    NvmonBackends backend;
} NvmonDevice;
typedef NvmonDevice* NvmonDevice_t;

/*typedef struct {*/
/*    int numDevices;*/
/*    NvmonDevice_t *devices;*/
/*} NvmonControl;*/
/*typedef NvmonControl* NvmonControl_t;*/


typedef struct {
    int  (*getEventList)(int gpuId, NvmonEventList_t* list);
    int (*createDevice)(int id, NvmonDevice *dev);
    void (*freeDevice)(NvmonDevice* dev);
    int (*addEvents)(NvmonDevice* device, const char* eventString);
    int (*setupCounters)(NvmonDevice* device, NvmonEventSet* eventSet);
    int (*startCounters)(NvmonDevice* device);
    int (*stopCounters)(NvmonDevice* device);
    int (*readCounters)(NvmonDevice* device);
    int (*finalizeCounters)(NvmonDevice* device);
} NvmonFunctions;


/*! \brief Structure specifying all performance monitoring event groups

The global NvmonGroupSet structure holds all eventSets and threads that are
configured to measure. Only one eventSet can be measured at a time but the groups
can be switched to perform some kind of multiplexing.
*/
typedef struct {
    int              numberOfGroups; /*!< \brief List length of \a groups*/
    int              numberOfActiveGroups; /*!< \brief Amount of added eventSets. Only those eventSets can be accessed in \a groups. */
    GroupState       state; /*!< \brief Current state of the event group (configured, started, none) */
    GroupInfo        *groups; /*!< \brief Structure holding the performance group information */
    int              activeGroup; /*!< \brief Currently active eventSet */
    //NvmonEventSet*   groups; /*!< \brief List of eventSets */
    int              numberOfGPUs; /*!< \brief Amount of GPUs in \a gpus */
    NvmonDevice*     gpus; /*!< \brief List of GPUs */
    int              numberOfBackends;
    NvmonFunctions*  backends[3];
} NvmonGroupSet;

extern NvmonGroupSet* nvGroupSet;


#endif
