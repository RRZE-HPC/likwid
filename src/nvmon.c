/*
 * =======================================================================================
 *
 *      Filename:  nvmon.c
 *
 *      Description:  Main implementation of the performance monitoring module
 *                    for NVIDIA GPUs
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tg), thomas.roehl@googlemail.com
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

/*
 * Currently this code assumes a single context per device. In most cases there is only
 * a single context but since other interfaces allow multiple contexts it might be
 * necessary. On the other hand, the other interfaces assume that the same events
 * are measured on all devices while this interface allows different set of events
 * per device
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include <sys/types.h>

#include <dlfcn.h>
#include <cupti.h>
#include <cuda_runtime_api.h>


#include <types.h>
#include <likwid.h>
#include <bitUtil.h>
#include <timer.h>
#include <lock.h>
#include <ghash.h>
#include <error.h>
#include <bstrlib.h>

#include <nvmon_types.h>
#include <libnvctr_types.h>

#include <likwid.h>

static int nvmon_initialized = 0;
NvmonGroupSet* nvGroupSet = NULL;
static int likwid_nvmon_verbosity = 0;

LikwidNvResults* gMarkerResults = NULL;
int gMarkerRegions = 0;


//static NvmonControl_t likwid_nvmon_control = NULL;
/* Copy from PAPI's cuda component (BSD License)
 * @author  Asim YarKhan yarkhan@icl.utk.edu (updated in 2017 to support CUDA metrics)
 * @author  Asim YarKhan yarkhan@icl.utk.edu (updated in 2015 for multiple CUDA contexts/devices)
 * @author  Heike Jagode (First version, in collaboration with Robert Dietrich, TU Dresden) jagode@icl.utk.edu
 */
void (*_dl_non_dynamic_init) (void) __attribute__ ((weak));

#define CU_CALL( call, handleerror )                                    \
    do {                                                                \
        CUresult _status = (call);                                      \
        if (_status != CUDA_SUCCESS) {                                  \
            fprintf(stderr, "Error: function %s failed with error %d.\n", #call, _status); \
            handleerror;                                                \
        }                                                               \
    } while (0)

#define CUPTI_CALL(call, handleerror)                                 \
    do {                                                                \
        CUptiResult _status = (call);                                   \
        if (_status != CUPTI_SUCCESS) {                                 \
            const char *errstr;                                         \
            (*cuptiGetResultString)(_status, &errstr);               \
            fprintf(stderr, "Error: function %s failed with error %s.\n", #call, errstr); \
            handleerror;                                                \
        }                                                               \
    } while (0)

#define CUDA_CALL( call, handleerror )                                \
    do {                                                                \
        cudaError_t _status = (call);                                   \
        if (_status != cudaSuccess) {                                   \
            fprintf(stderr, "Error: function %s failed with error %d.\n", #call, _status); \
            handleerror;                                                \
        }                                                               \
    } while (0)

#define CUAPIWEAK __attribute__( ( weak ) )
#define DECLARECUFUNC(funcname, funcsig) CUresult CUAPIWEAK funcname funcsig;  CUresult( *funcname##Ptr ) funcsig;
DECLARECUFUNC(cuCtxGetCurrent, (CUcontext *));
DECLARECUFUNC(cuCtxSetCurrent, (CUcontext));
DECLARECUFUNC(cuDeviceGet, (CUdevice *, int));
DECLARECUFUNC(cuDeviceGetCount, (int *));
DECLARECUFUNC(cuDeviceGetName, (char *, int, CUdevice));
DECLARECUFUNC(cuInit, (unsigned int));
DECLARECUFUNC(cuCtxPopCurrent, (CUcontext * pctx));
DECLARECUFUNC(cuCtxPushCurrent, (CUcontext pctx));
DECLARECUFUNC(cuCtxSynchronize, ());

#define CUDAAPIWEAK __attribute__( ( weak ) )
#define DECLARECUDAFUNC(funcname, funcsig) cudaError_t CUDAAPIWEAK funcname funcsig;  cudaError_t( *funcname##Ptr ) funcsig;
DECLARECUDAFUNC(cudaGetDevice, (int *));
DECLARECUDAFUNC(cudaSetDevice, (int));
DECLARECUDAFUNC(cudaFree, (void *));

#define CUPTIAPIWEAK __attribute__( ( weak ) )
#define DECLARECUPTIFUNC(funcname, funcsig) CUptiResult CUPTIAPIWEAK funcname funcsig;  CUptiResult( *funcname##Ptr ) funcsig;
DECLARECUPTIFUNC(cuptiEventGroupGetAttribute, (CUpti_EventGroup eventGroup, CUpti_EventGroupAttribute attrib, size_t * valueSize, void *value));
DECLARECUPTIFUNC(cuptiDeviceGetEventDomainAttribute, (CUdevice device, CUpti_EventDomainID eventDomain, CUpti_EventDomainAttribute attrib, size_t * valueSize, void *value));
DECLARECUPTIFUNC(cuptiEventGroupReadEvent, (CUpti_EventGroup eventGroup, CUpti_ReadEventFlags flags, CUpti_EventID event, size_t * eventValueBufferSizeBytes, uint64_t *        eventValueBuffer));
DECLARECUPTIFUNC(cuptiEventGroupSetAttribute, (CUpti_EventGroup eventGroup, CUpti_EventGroupAttribute attrib, size_t valueSize, void *value));
DECLARECUPTIFUNC(cuptiEventGroupSetDisable, (CUpti_EventGroupSet * eventGroupSet));
DECLARECUPTIFUNC(cuptiEventGroupSetEnable, (CUpti_EventGroupSet * eventGroupSet));
DECLARECUPTIFUNC(cuptiEventGroupSetsCreate, (CUcontext context, size_t eventIdArraySizeBytes, CUpti_EventID * eventIdArray, CUpti_EventGroupSets ** eventGroupPasses));
DECLARECUPTIFUNC(cuptiEventGroupSetsDestroy, (CUpti_EventGroupSets * eventGroupSets));
DECLARECUPTIFUNC(cuptiGetTimestamp, (uint64_t * timestamp));
DECLARECUPTIFUNC(cuptiSetEventCollectionMode, (CUcontext context, CUpti_EventCollectionMode mode));
DECLARECUPTIFUNC(cuptiDeviceEnumEventDomains, (CUdevice, size_t *, CUpti_EventDomainID *));
DECLARECUPTIFUNC(cuptiDeviceGetNumEventDomains, (CUdevice, uint32_t *));
DECLARECUPTIFUNC(cuptiEventDomainEnumEvents, (CUpti_EventDomainID, size_t *, CUpti_EventID *));
DECLARECUPTIFUNC(cuptiEventDomainGetNumEvents, (CUpti_EventDomainID, uint32_t *));
DECLARECUPTIFUNC(cuptiEventGetAttribute, (CUpti_EventID, CUpti_EventAttribute, size_t *, void *));
DECLARECUPTIFUNC(cuptiGetResultString, (CUptiResult result, const char **str));


static void *dl_libcuda = NULL;
static void *dl_libcudart = NULL;
static void *dl_libcupti = NULL;

static int
link_libraries(void)
{
#define DLSYM_AND_CHECK( dllib, name ) dlsym( dllib, name ); if ( dlerror() != NULL ) { return -1; }

    /* Attempt to guess if we were statically linked to libc, if so bail */
    if(_dl_non_dynamic_init != NULL) {
        return -1;
    }
    /* Need to link in the cuda libraries, if not found disable the component */
    dl_libcuda = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_libcuda)
    {
        fprintf(stderr, "CUDA library libcuda.so not found.");
        return -1;
    }
    cuCtxGetCurrentPtr = DLSYM_AND_CHECK(dl_libcuda, "cuCtxGetCurrent");
    cuCtxSetCurrentPtr = DLSYM_AND_CHECK(dl_libcuda, "cuCtxSetCurrent");
    cuDeviceGetPtr = DLSYM_AND_CHECK(dl_libcuda, "cuDeviceGet");
    cuDeviceGetCountPtr = DLSYM_AND_CHECK(dl_libcuda, "cuDeviceGetCount");
    cuDeviceGetNamePtr = DLSYM_AND_CHECK(dl_libcuda, "cuDeviceGetName");
    cuInitPtr = DLSYM_AND_CHECK(dl_libcuda, "cuInit");
    cuCtxPopCurrentPtr = DLSYM_AND_CHECK(dl_libcuda, "cuCtxPopCurrent");
    cuCtxPushCurrentPtr = DLSYM_AND_CHECK(dl_libcuda, "cuCtxPushCurrent");
    cuCtxSynchronizePtr = DLSYM_AND_CHECK(dl_libcuda, "cuCtxSynchronize");

    dl_libcudart = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if (!dl_libcudart)
    {
        fprintf(stderr, "CUDA runtime library libcudart.so not found.");
        return -1;
    }
    cudaGetDevicePtr = DLSYM_AND_CHECK(dl_libcudart, "cudaGetDevice");
    cudaSetDevicePtr = DLSYM_AND_CHECK(dl_libcudart, "cudaSetDevice");
    cudaFreePtr = DLSYM_AND_CHECK(dl_libcudart, "cudaFree");

    dl_libcupti = dlopen("libcupti.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_libcupti)
    {
        fprintf(stderr, "CUDA runtime library libcupti.so not found.");
        return -1;
    }
    cuptiDeviceGetEventDomainAttributePtr = DLSYM_AND_CHECK(dl_libcupti, "cuptiDeviceGetEventDomainAttribute");
    cuptiEventGroupGetAttributePtr = DLSYM_AND_CHECK(dl_libcupti, "cuptiEventGroupGetAttribute");
    cuptiEventGroupReadEventPtr = DLSYM_AND_CHECK(dl_libcupti, "cuptiEventGroupReadEvent");
    cuptiEventGroupSetAttributePtr = DLSYM_AND_CHECK(dl_libcupti, "cuptiEventGroupSetAttribute");
    cuptiEventGroupSetDisablePtr = DLSYM_AND_CHECK(dl_libcupti, "cuptiEventGroupSetDisable");
    cuptiEventGroupSetEnablePtr = DLSYM_AND_CHECK(dl_libcupti, "cuptiEventGroupSetEnable");
    cuptiEventGroupSetsCreatePtr = DLSYM_AND_CHECK(dl_libcupti, "cuptiEventGroupSetsCreate");
    cuptiEventGroupSetsDestroyPtr = DLSYM_AND_CHECK(dl_libcupti, "cuptiEventGroupSetsDestroy");
    cuptiGetTimestampPtr = DLSYM_AND_CHECK(dl_libcupti, "cuptiGetTimestamp");
    cuptiSetEventCollectionModePtr = DLSYM_AND_CHECK(dl_libcupti, "cuptiSetEventCollectionMode");
    cuptiDeviceEnumEventDomainsPtr = DLSYM_AND_CHECK(dl_libcupti, "cuptiDeviceEnumEventDomains");
    cuptiDeviceGetNumEventDomainsPtr = DLSYM_AND_CHECK(dl_libcupti, "cuptiDeviceGetNumEventDomains");
    cuptiEventDomainEnumEventsPtr = DLSYM_AND_CHECK(dl_libcupti, "cuptiEventDomainEnumEvents");
    cuptiEventDomainGetNumEventsPtr = DLSYM_AND_CHECK(dl_libcupti, "cuptiEventDomainGetNumEvents");
    cuptiEventGetAttributePtr = DLSYM_AND_CHECK(dl_libcupti, "cuptiEventGetAttribute");
    cuptiGetResultStringPtr = DLSYM_AND_CHECK(dl_libcupti, "cuptiGetResultString");
    return 0;
}

static int
init_cuda(void)
{
    CUresult cuErr = (*cuInitPtr)(0);
    if (cuErr != CUDA_SUCCESS)
    {
        fprintf(stderr, "CUDA cannot be found and initialized (cuInit failed).\n");
        return -ENODEV;
    }
    return 0;
}

static int
get_numDevices(void)
{
    CUresult cuErr;
    int count = 0;
    cuErr = (*cuDeviceGetCountPtr)(&count);
    if(cuErr == CUDA_ERROR_NOT_INITIALIZED)
    {
        int ret = init_cuda();
        if (ret == 0)
        {
            cuErr = (*cuDeviceGetCountPtr)(&count);
        }
        else
        {
            return ret;
        }
    }
    return count;
}

/*static int*/
/*nvmon_create_control(void)*/
/*{*/
/*    if (!likwid_nvmon_control)*/
/*    {*/
/*        likwid_nvmon_control = malloc(sizeof(NvmonControl));*/
/*        likwid_nvmon_control->numDevices = 0;*/
/*        likwid_nvmon_control->devices = NULL;*/
/*    }*/
/*}*/

static NvmonDevice_t
free_device(NvmonDevice_t dev)
{
    int j = 0;
    GHashTableIter iter;
    char* name = NULL;
    uint32_t *id = NULL;
    NvmonEvent_t event = NULL;
    if (dev)
    {
        if (dev->context)
        {
            cuCtxDestroy(dev->context);
            dev->context = NULL;
        }
        if (dev->cuEventSets)
        {
            CUPTI_CALL((*cuptiEventGroupSetsDestroyPtr)(dev->cuEventSets), return NULL);
            dev->cuEventSets = NULL;
        }
        // We just need to free the keys in the hash tables.
/*        g_hash_table_iter_init(&iter, dev->eventHash);*/
/*        while (g_hash_table_iter_next(&iter, (gpointer)name, (gpointer)&event))*/
/*        {*/
/*            free(name);*/
/*        }*/
/*        g_hash_table_iter_init(&iter, dev->evIdHash);*/
/*        while (g_hash_table_iter_next(&iter, (gpointer)id, (gpointer)&event))*/
/*        {*/
/*            free(id);*/
/*        }*/
        // The event objects are freed here
        if (dev->allevents)
        {
            for (j = 0; j < dev->numAllEvents; j++)
                free(dev->allevents[j]);
        }
        if (dev->activeEvents)
        {
            free(dev->activeEvents);
            dev->activeEvents = NULL;
            dev->numActiveEvents = 0;
        }
/*        if (dev->activeCuGroups)*/
/*        {*/
/*            printf("%p\n", dev->activeCuGroups);*/
/*            free(dev->activeCuGroups);*/
/*            dev->activeCuGroups = NULL;*/
/*            dev->numActiveCuGroups = 0;*/
/*        }*/
        if (dev->nvEventSets)
        {
            for (j = 0; j < dev->numNvEventSets; j++)
            {
                free(dev->nvEventSets[j].results);
            }
            free(dev->nvEventSets);
            dev->nvEventSets = NULL;
            dev->numNvEventSets = 0;
        }
        memset(dev, 0, sizeof(NvmonDevice));
        //free(dev);
    }
    return NULL;
}

static void print_event(NvmonEvent_t event)
{
    printf("Event %s: %s\n", event->name, event->description);
}

static int
create_device(int id, NvmonDevice *dev)
{
    int j = 0, k = 0, c = 0;
    int numDomains = 0;
    CUpti_EventDomainID* eventDomainIds = NULL;
    int eventIdx = 0;
    uint32_t totalEvents = 0;

    // Allocate new device space
/*    NvmonDevice_t dev = malloc(sizeof(NvmonDevice));*/
/*    if (!dev) return free_device(dev);*/
/*    memset(dev, 0, sizeof(NvmonDevice));*/
    dev->deviceId = id;
    dev->cuEventSets = NULL;
    dev->context = NULL;
    dev->activeEvents = NULL;
    dev->numActiveEvents = 0;
    dev->numNvEventSets = 0;
    dev->nvEventSets = NULL;

    // Assign device ID and get cuDevice from CUDA
    dev->deviceId = id;
    CU_CALL((*cuDeviceGetPtr)(&dev->cuDevice, id), return -1);
    // Get the name of the device
/*    CU_CALL((*cuDeviceGetNamePtr)(dev->name, NVMON_DEFAULT_STR_LEN-1, dev->cuDevice), return free_device(dev));*/
/*    dev->name[NVMON_DEFAULT_STR_LEN-1] = '\0';*/
    // Get the number of event domains of the device
    CUPTI_CALL((*cuptiDeviceGetNumEventDomainsPtr)(dev->cuDevice, &numDomains), return -1);
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Nvmon: Dev %d Domains %d, id, numDomains);

    // Get the domain IDs for the device
    size_t domainarraysize = numDomains * sizeof(CUpti_EventDomainID);
    eventDomainIds = malloc(domainarraysize);
    if (!eventDomainIds) return -ENOMEM;
    CUPTI_CALL((*cuptiDeviceEnumEventDomainsPtr)(dev->cuDevice, &domainarraysize, eventDomainIds), return -1);


    // Count the events in all domains to allocate the event list
    dev->numAllEvents = 0;
    for (j = 0; j < numDomains; j++)
    {
        uint32_t domainNumEvents = 0;
        CUpti_EventDomainID domainID = eventDomainIds[j];
        CUPTI_CALL((*cuptiEventDomainGetNumEventsPtr)(domainID, &domainNumEvents), return -1);
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Nvmon: Dev %d Domain %d Events %d, id, j, domainNumEvents);
        totalEvents += domainNumEvents;
    }
    // Now we now how many events are provided by the device, so allocate a big enough event list
    dev->allevents = malloc(totalEvents * sizeof(NvmonEvent_t));
    if (!dev->allevents)
    {
        free(eventDomainIds);
        return -ENOMEM;
    }

    // We use hash tables for faster access
    dev->eventHash = g_hash_table_new(g_str_hash, g_str_equal);
    dev->evIdHash = g_hash_table_new(g_int64_hash, g_int64_equal);

    for (j = 0; j < numDomains; j++)
    {
        uint32_t domainNumEvents = 0;
        CUpti_EventDomainID domainID = eventDomainIds[j];
        // How many events are provided by the domain
        CUPTI_CALL((*cuptiEventDomainGetNumEventsPtr)(domainID, &domainNumEvents), return -1);
        size_t tmpSize = domainNumEvents * sizeof(CUpti_EventID);
        // Allocate space for all CUPTI event IDs in the domain
        CUpti_EventID* cuEventIds = malloc(tmpSize);
        // Get the CUPTI events
        CUPTI_CALL((*cuptiEventDomainEnumEventsPtr)(domainID, &tmpSize, cuEventIds), return -1);
        for (k = 0; k < domainNumEvents; k++)
        {
            CUpti_EventID eventId = cuEventIds[k];
            // Name and description are limited in length
            size_t tmpSizeBytesName = (NVMON_DEFAULT_STR_LEN-1) * sizeof(char);
            size_t tmpSizeBytesDesc = (NVMON_DEFAULT_STR_LEN-1) * sizeof(char);

            NvmonEvent_t event = malloc(sizeof(NvmonEvent));
            if (!event)
            {
                free(cuEventIds);
                free(dev->allevents);
                free(eventDomainIds);
                return -ENOMEM;
            }
            // Get event name and description
            CUPTI_CALL((*cuptiEventGetAttributePtr)(eventId, CUPTI_EVENT_ATTR_NAME, &tmpSizeBytesName, event->name), return -1);
            CUPTI_CALL((*cuptiEventGetAttributePtr)(eventId, CUPTI_EVENT_ATTR_LONG_DESCRIPTION, &tmpSizeBytesDesc, event->description), return -1);
            event->name[tmpSizeBytesName/sizeof(char)] = '\0';
            event->description[tmpSizeBytesDesc/sizeof(char)] = '\0';
            // LIKWID events are all uppercase, so transform the event names
            c = 0;
            while (event->name[c] != '\0')
            {
                event->name[c] = toupper(event->name[c]);
                c++;
            }
            // Save all gathered information in a NvmonEvent object

            event->cuEventId = eventId;
            event->eventId = dev->numAllEvents;
            event->cuDomainId = domainID;
            event->domainId = j;
            event->active = 0;
            //DEBUG_PRINT(DEBUGLEV_DETAIL, New Event %d CuEvent %d Domain %d CuDomain %d Name %s, event->eventId, (int)event->cuEventId, event->domainId, (int)event->cuDomainId, event->name);
            // Add the object to the event list
            dev->allevents[dev->numAllEvents] = event;
            dev->numAllEvents++;
            // Add the object to the hash tables
            char* nameKey = g_strdup(event->name);
            //printf("Add event %s\n", nameKey);
            CUpti_EventID* idKey = malloc(sizeof(CUpti_EventID));
            if (idKey)
            {
                *idKey = event->cuEventId;
                // Key is event name
                g_hash_table_insert(dev->eventHash, (gpointer)nameKey, (gpointer)event);
                // Key is CUPTI event ID
                g_hash_table_insert(dev->evIdHash, (gpointer)idKey, (gpointer)event);
                //print_event(event);
            }
        }
    }
    return 0;
}



static void print_active_event(NvmonActiveEvent_t event)
{
    printf("Event %d\n", event->idxInSet);
}

int
nvmon_init(int nrGpus, const int* gpuIds)
{
    int idx = 0;
    GpuTopology_t gtopo = NULL;
    if (nvmon_initialized == 1)
    {
        return 0;
    }

    if (nrGpus <= 0)
    {
        ERROR_PRINT(Number of gpus must be greater than 0 but only %d given, nrGpus);
        return -EINVAL;
    }

    if (!lock_check())
    {
        ERROR_PLAIN_PRINT(Access to performance monitoring locked);
        return -EINVAL;
    }

    if (nvGroupSet != NULL)
    {
        return -EEXIST;
    }
    int ret = link_libraries();
    if (ret < 0)
    {
        return -ENODEV;
    }

    ret = topology_gpu_init();
    if (ret != EXIT_SUCCESS)
    {
        return -ENODEV;
    }
    gtopo = get_gpuTopology();

    nvGroupSet = (NvmonGroupSet*) malloc(sizeof(NvmonGroupSet));
    if (nvGroupSet == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate group descriptor);
        return -ENOMEM;
    }
    nvGroupSet->gpus = (NvmonDevice*) malloc(nrGpus * sizeof(NvmonDevice));
    if (nvGroupSet->gpus == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate set of GPUs);
        free(nvGroupSet);
        nvGroupSet = NULL;
        return -ENOMEM;
    }
    nvGroupSet->numberOfGPUs = nrGpus;
    nvGroupSet->numberOfGroups = 0;
    nvGroupSet->numberOfActiveGroups = 0;
    nvGroupSet->groups = NULL;
    nvGroupSet->activeGroup = -1;

    for (int i = 0; i < nrGpus; i++)
    {
        if (gpuIds[i] < 0) continue;
        int available = 0;
        for (int j = 0; j < gtopo->numDevices; j++)
        {
            if (gtopo->devices[j].devid == gpuIds[i])
            {
                available = 1;
                break;
            }
        }
        if (!available)
        {
            ERROR_PRINT(No device with ID %d, gpuIds[i]);
            free(nvGroupSet->gpus);
            free(nvGroupSet);
            nvGroupSet = NULL;
            return -ENOMEM;
        }
        ret = create_device(gpuIds[i], &nvGroupSet->gpus[idx]);
        idx++;
    }



    nvmon_initialized = 1;
    return 0;
}



void
nvmon_finalize(void)
{
    if (nvmon_initialized && nvGroupSet)
    {
        for (int i = 0; i < nvGroupSet->numberOfGPUs; i++)
        {
            free_device(&nvGroupSet->gpus[i]);
        }
        free(nvGroupSet->gpus);
        free(nvGroupSet);
        nvGroupSet = NULL;
        nvmon_initialized = 0;
    }
}



int
nvmon_addEventSet(const char* eventCString)
{
    int i = 0, j = 0, k = 0, l = 0, m = 0, err = 0;
    int isPerfGroup = 0;
    int curDeviceId = -1;
    int devId = 0;
    int configuredEvents = 0;
    CUcontext curContext;
    NvmonDevice_t device = NULL;
    Configuration_t config = NULL;
    GroupInfo ginfo;

    struct bstrList* eventtokens;
    if (!eventCString)
    {
        return -EINVAL;
    }
    if (!nvmon_initialized)
    {
        return -EFAULT;
    }

    config = get_configuration();

    if (nvGroupSet->numberOfActiveGroups == nvGroupSet->numberOfGroups)
    {
        GroupInfo* tmpInfo = (GroupInfo*)realloc(nvGroupSet->groups, (nvGroupSet->numberOfGroups+1)*sizeof(GroupInfo));
        if (tmpInfo == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate additional group);
            return -ENOMEM;
        }
        nvGroupSet->groups = tmpInfo;
        nvGroupSet->numberOfGroups++;
        DEBUG_PLAIN_PRINT(DEBUGLEV_INFO, Allocating new group structure for group.);
    }
    DEBUG_PRINT(DEBUGLEV_INFO, NVMON: Currently %d groups of %d active,
                                        nvGroupSet->numberOfActiveGroups+1,
                                        nvGroupSet->numberOfGroups+1);

    bstring eventBString = bfromcstr(eventCString);
    if (bstrchrp(eventBString, ':', 0) != BSTR_ERR)
    {
        DEBUG_PLAIN_PRINT(DEBUGLEV_DEVELOP, Custom eventset);
        err = custom_group(eventCString, &nvGroupSet->groups[nvGroupSet->numberOfGroups-1]);
        if (err)
        {
            ERROR_PRINT(Cannot transform %s to performance group, eventCString);
            return err;
        }
    }
    else
    {
        DEBUG_PLAIN_PRINT(DEBUGLEV_DEVELOP, Performance group);
        err = read_group(config->groupPath, "nvidiagpu",
                         eventCString,
                         &nvGroupSet->groups[nvGroupSet->numberOfGroups-1]);
        if (err == -EACCES)
        {
            ERROR_PRINT(Access to performance group %s not allowed, eventCString);
            return err;
        }
        else if (err == -ENODEV)
        {
            ERROR_PRINT(Performance group %s only available with deactivated HyperThreading, eventCString);
            return err;
        }
        else if (err < 0)
        {
            ERROR_PRINT(Cannot read performance group %s, eventCString);
            return err;
        }
        isPerfGroup = 1;
    }
    bdestroy(eventBString);
    char * evstr = get_eventStr(&nvGroupSet->groups[nvGroupSet->numberOfGroups-1]);
    eventBString = bfromcstr(evstr);
    DEBUG_PRINT(DEBUGLEV_DEVELOP, EventStr %s, evstr);
    eventtokens = bsplit(eventBString, ',');
    bdestroy(eventBString);



    CUDA_CALL((*cudaGetDevicePtr)(&curDeviceId), return -EFAULT);
    CUDA_CALL((*cudaFreePtr)(NULL), return -EFAULT);
    CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);


    for (devId = 0; devId < nvGroupSet->numberOfGPUs; devId++)
    {
        device = &nvGroupSet->gpus[devId];
        size_t sizeBytes = (eventtokens->qty) * sizeof(CUpti_EventID);

        int popContext = 0;
        CUpti_EventGroupSets * cuEventSets = NULL;

        NvmonEventSet* tmpEventSet = realloc(device->nvEventSets, (device->numNvEventSets+1)*sizeof(NvmonEventSet));
        if (!tmpEventSet)
        {
            ERROR_PRINT(Cannot enlarge GPU %d eventSet list, device->deviceId);
            err = -ENOMEM;
            continue;
        }
        device->nvEventSets = tmpEventSet;
        NvmonEventSet* devEventSet = &device->nvEventSets[device->numNvEventSets];

        devEventSet->nvEvents = (NvmonEvent_t*) malloc(eventtokens->qty * sizeof(NvmonEvent_t));
        if (devEventSet->nvEvents == NULL)
        {
            ERROR_PRINT(Cannot allocate event list for group %d\n, groupSet->numberOfActiveGroups);
            return -ENOMEM;
        }
        devEventSet->cuEventIDs = (CUpti_EventID*) malloc(eventtokens->qty * sizeof(CUpti_EventID));
        if (devEventSet->cuEventIDs == NULL)
        {
            ERROR_PRINT(Cannot allocate event ID list for group %d\n, groupSet->numberOfActiveGroups);
            free(devEventSet->nvEvents);
            return -ENOMEM;
        }
        devEventSet->results = malloc(eventtokens->qty * sizeof(NvmonEventResult));
        if (devEventSet->cuEventIDs == NULL)
        {
            ERROR_PRINT(Cannot allocate result list for group %d\n, groupSet->numberOfActiveGroups);
            free(devEventSet->cuEventIDs);
            free(devEventSet->nvEvents);
            return -ENOMEM;
        }
        memset(devEventSet->results, 0, eventtokens->qty * sizeof(NvmonEventResult));
        devEventSet->numberOfEvents = 0;

        // If the device has no context, check whether the current context is
        // used already by another device
        if (!device->context)
        {
            int contextDevice = -1;
            for (j = 0; j < nvGroupSet->numberOfGPUs; j++)
            {
                NvmonDevice_t dev = &nvGroupSet->gpus[j];
                if (dev->context == curContext)
                {
                    contextDevice = j;
                    break;
                }
            }
            if (contextDevice < 0)
            {
                // Current context is _not_ used by another device, use it for the
                // current device
                device->context = curContext;
            }
            else
            {
                // Create a new context and assign it to the current event device
                CUDA_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
                CUDA_CALL((*cudaFreePtr)(NULL), return -EFAULT);
                CU_CALL((*cuCtxGetCurrentPtr)(&device->context), return -EFAULT);
            }
        }
        else if (device->context != curContext)
        {
            // The device context is not the current one, so add it to the
            // context stack
            CU_CALL((*cuCtxPushCurrentPtr)(device->context), return -EFAULT);
            popContext = 1;
        }


        CUPTI_CALL((*cuptiSetEventCollectionModePtr)(device->context, CUPTI_EVENT_COLLECTION_MODE_KERNEL), return -EFAULT);

        for (i = 0; i < eventtokens->qty; i++)
        {
            struct bstrList* evset = bsplit(eventtokens->entry[i], ':');
            if (evset->qty != 2)
            {
                ERROR_PRINT(NVMON: Event %s invalid: Format <event>:<gpucounter>, bdata(eventtokens->entry[i]));
            }
            if (blength(evset->entry[0]) == 0 || blength(evset->entry[1]) == 0)
            {
                ERROR_PRINT(NVMON: Event %s invalid: Format <event>:<gpucounter>, bdata(eventtokens->entry[i]));
            }
            NvmonEvent_t event = g_hash_table_lookup(device->eventHash, (gpointer)bdata(evset->entry[0]));
            if (!event)
            {
                DEBUG_PRINT(DEBUGLEV_INFO, NVMON: Event %s unknown. Skipping..., bdata(evset->entry[0]));
                continue; //unknown event
            }
            else
            {
                devEventSet->cuEventIDs[devEventSet->numberOfEvents] = event->cuEventId;
                devEventSet->nvEvents[devEventSet->numberOfEvents] = event;
                devEventSet->numberOfEvents++;
                size_t s = devEventSet->numberOfEvents*sizeof(CUpti_EventID);

                CUPTI_CALL((*cuptiEventGroupSetsCreatePtr)(device->context, s, devEventSet->cuEventIDs, &cuEventSets), devEventSet->numberOfEvents--;);
                if (cuEventSets->numSets > 1)
                {
                    ERROR_PRINT(Error adding event %s. Multiple measurement runs are required. skipping event ..., bdata(evset->entry[i]));
                    continue;
                }
            }
        }
        // Create an eventset with the currently configured and current event
        if(popContext)
        {
            CU_CALL((*cuCtxPopCurrentPtr)(&device->context), return -EFAULT);
        }
        device->numNvEventSets++;
    }

    // Check whether group has any event in any device
    nvGroupSet->numberOfActiveGroups++;
    return (nvGroupSet->numberOfActiveGroups-1);
}

int nvmon_setupCounters(int gid)
{
    int j = 0, k = 0, l = 0, m = 0;
    int devId = 0, eid = 0;
    int oldDevId;
    CUcontext curContext;
    NvmonDevice_t device = NULL;
    if ((!nvGroupSet) || (!nvmon_initialized) || (gid < 0))
    {
        return -EFAULT;
    }

    // Currently we are on which device?
    CUDA_CALL((*cudaGetDevicePtr)(&oldDevId), return -EFAULT);
    // This is a workaround to (eventually create and) get the current context
    CUDA_CALL((*cudaFreePtr)(NULL), return -EFAULT);
    CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);

    if (gid >= 0 && gid < nvGroupSet->numberOfGroups)
    {
        for (devId = 0; devId < nvGroupSet->numberOfGPUs; devId++)
        {
            int popContext = 0;
            device = &nvGroupSet->gpus[devId];
            NvmonEventSet* devEventSet = &device->nvEventSets[gid];
            CUpti_EventGroupSets * cuEventSets = NULL;
            if (devEventSet->numberOfEvents == 0)
            {
                DEBUG_PRINT(DEBUGLEV_DETAIL, Skipping GPU%d it has no events in group %d, device->deviceId, gid);
                continue;
            }

            // If the device has no context, check whether the current context is
            // used already by another device
            if (!device->context)
            {
                int contextDevice = -1;
                for (j = 0; j < nvGroupSet->numberOfGPUs; j++)
                {
                    NvmonDevice_t dev = &nvGroupSet->gpus[j];
                    if (dev->context == curContext)
                    {
                        contextDevice = j;
                        break;
                    }
                }
                if (contextDevice < 0)
                {
                    // Current context is _not_ used by another device, use it for the
                    // current device
                    device->context = curContext;
                    printf("Reuse context %ld for device %d\n", device->context, device->deviceId);
                }
                else
                {
                    // Create a new context and assign it to the current event device
                    CUDA_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
                    CUDA_CALL((*cudaFreePtr)(NULL), return -EFAULT);
                    CU_CALL((*cuCtxGetCurrentPtr)(&device->context), return -EFAULT);
                    printf("New context %ld for device %d\n", device->context, device->deviceId);
                }
            }
            else if (device->context != curContext)
            {
                // The device context is not the current one, so add it to the
                // context stack
                CU_CALL((*cuCtxPushCurrentPtr)(device->context), return -EFAULT);
                popContext = 1;
            }


            size_t grpEventIdsSize = devEventSet->numberOfEvents * sizeof(CUpti_EventID);
            CUPTI_CALL((*cuptiEventGroupSetsCreatePtr)(device->context, grpEventIdsSize, devEventSet->cuEventIDs, &cuEventSets), m++;);

            // Allocate temporary array to hold the group event IDs
            CUpti_EventID *grpEventIds = malloc(grpEventIdsSize);
            if (!grpEventIds)
            {
                return -ENOMEM;
            }

            // Delete current activeEvent list
            if (device->activeEvents)
            {
                free(device->activeEvents);
                device->activeEvents = NULL;
                device->numActiveEvents = 0;
            }
            // Delete current activeCuGroups list
            if (device->activeCuGroups)
            {
                free(device->activeCuGroups);
                device->activeCuGroups = NULL;
                device->numActiveCuGroups = 0;
            }

            // Create a new activeEvent list
            device->activeEvents = malloc(devEventSet->numberOfEvents * sizeof(NvmonActiveEvent));
            if (!device->activeEvents)
            {
                free(grpEventIds);
                grpEventIds = NULL;
                return -ENOMEM;
            }
            // Create a new activeCuGroups list
            device->activeCuGroups = malloc(cuEventSets->numSets * sizeof(CUpti_EventGroupSet**));
            if (!device->activeCuGroups)
            {
                free(device->activeEvents);
                device->activeEvents = NULL;
                free(grpEventIds);
                grpEventIds = NULL;
                return -ENOMEM;
            }




            // Run over eventset and store all information we need for start/stop/reads in NvmonActiveEvent_t
            CUpti_EventGroup curGroup;
            uint32_t curNumInstances = 0, curNumTotalInstances = 0;
            CUpti_EventGroupSet *curEventGroupSet;
            for (j = 0; j < cuEventSets->numSets; j++)
            {
                size_t sizeofuint32t = sizeof(uint32_t);
                uint32_t numGEvents = 0, numGInstances = 0, numTotalGInstances = 0;
                CUpti_EventGroupSet* groupset = &cuEventSets->sets[j];

                for (k = 0; k < groupset->numEventGroups; k++)
                {
                    uint32_t one = 1;
                    CUpti_EventGroup group = groupset->eventGroups[k];
                    // Get the number of events in the group
                    CUPTI_CALL((*cuptiEventGroupGetAttributePtr)(group, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &sizeofuint32t, &numGEvents), return -EFAULT);
                    // Get the CUPTI event IDs in the group
                    CUPTI_CALL((*cuptiEventGroupGetAttributePtr)(group, CUPTI_EVENT_GROUP_ATTR_EVENTS, &grpEventIdsSize, grpEventIds), return -EFAULT);
                    // If we don't set this, each event has only a single instance but we want to measure all instances
                    CUPTI_CALL((*cuptiEventGroupSetAttributePtr)(group, CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(uint32_t), &one), return -EFAULT);
                    // Get instance count for a group
                    CUPTI_CALL((*cuptiEventGroupGetAttributePtr)(group, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &sizeofuint32t, &numGInstances), return -EFAULT);

                    for (l = 0; l < numGEvents; l++)
                    {
                        for (m = 0; m <  devEventSet->numberOfEvents; m++)
                        {
                            if (devEventSet->cuEventIDs[m] == grpEventIds[l])
                            {
                                CUpti_EventDomainID did = devEventSet->nvEvents[m]->cuDomainId;
                                // Get total instance count for a group
                                CUPTI_CALL((*cuptiDeviceGetEventDomainAttributePtr)(device->cuDevice, did, CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &sizeofuint32t, &numTotalGInstances), return -EFAULT);
                                device->activeEvents[m].eventId = devEventSet->nvEvents[m]->eventId;
                                device->activeEvents[m].idxInSet = m;
                                device->activeEvents[m].groupId = gid;
                                device->activeEvents[m].cuEventId = devEventSet->nvEvents[m]->cuEventId;
                                device->activeEvents[m].cuDomainId = did;
                                device->activeEvents[m].numTotalInstances = numTotalGInstances;
                                device->activeEvents[m].cuGroup = group;
                                device->activeEvents[m].cuGroupSet = groupset;
                                device->activeEvents[m].numInstances = numGInstances;
                                device->activeEvents[m].deviceId = device->deviceId;

                                int found = 0;
                                for (int i = 0; i < device->numActiveCuGroups; i++)
                                {
                                    if (device->activeCuGroups[i] == groupset)
                                    {
                                        found = 1;
                                    }
                                }
                                if (!found)
                                {
                                    device->activeCuGroups[device->numActiveCuGroups] = groupset;
                                    device->numActiveCuGroups++;
                                }
                                // Mark event as active. This is used to avoid measuring the same event on the same device twice
                                devEventSet->nvEvents[m]->active = 1;
                                DEBUG_PRINT(DEBUGLEV_INFO, Setup event %s (%d) for GPU %d, devEventSet->nvEvents[m]->name, device->activeEvents[m].cuEventId, device->deviceId);
                                device->numActiveEvents++;
                            }
                        }
                    }
                }
            }
            free(grpEventIds);
            if(popContext)
            {
                CU_CALL((*cuCtxPopCurrentPtr)(&device->context), return -EFAULT);
            }
        }
    }
    nvGroupSet->activeGroup = gid;

    return 0;
}


int
nvmon_startCounters(void)
{
    int i = 0, j = 0, k = 0;
    int oldDevId = -1;
    uint64_t timestamp = 0;
    CUcontext curContext;
    if ((!nvGroupSet) || (!nvmon_initialized) || (nvGroupSet->activeGroup < 0))
    {
        return -EFAULT;
    }

    // Currently we are on which device?
    CUDA_CALL((*cudaGetDevicePtr)(&oldDevId), return -EFAULT);
    // Take the timestamp, we assign it later for all devices
    CUPTI_CALL((*cuptiGetTimestampPtr)(&timestamp), return -EFAULT);
    // This is a workaround to (eventually create and) get the current context
    CUDA_CALL((*cudaFreePtr)(NULL), return -EFAULT);
    CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);

    for (i = 0; i < nvGroupSet->numberOfGPUs; i++)
    {
        int popContext = 0;
        uint32_t one = 1;
        NvmonDevice_t device = &nvGroupSet->gpus[i];
        if (device->numActiveCuGroups == 0)
        {
            DEBUG_PRINT(DEBUGLEV_DETAIL, Skipping GPU%d it has no events in group %d, device->deviceId, nvGroupSet->activeGroup);
            continue;
        }
        if (device->deviceId != oldDevId)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Change GPU device %d -> %d, oldDevId, device->deviceId);
            CUDA_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
            CUDA_CALL((*cudaGetDevicePtr)(&oldDevId), return -EFAULT);
        }
        CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);
        device->timeStart = timestamp;
        device->timeRead = timestamp;

        // Are we in the proper context?
        if (device->context != curContext)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Push Context %ld -> %ld for device %d, curContext, device->context, device->deviceId);
            CU_CALL((*cuCtxPushCurrentPtr)(device->context), return -EFAULT);
            popContext = 1;
        }
        else
        {
            // Although we are already in the right context, we set it again
            // to be sure
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Set Context %ld for device %d, device->context, device->deviceId);
            CU_CALL((*cuCtxSetCurrentPtr)(device->context), return -EFAULT);
        }

        NvmonEventSet* nvEventSet = &device->nvEventSets[nvGroupSet->activeGroup];
        for (j = 0; j < nvEventSet->numberOfEvents; j++)
        {
            NvmonEventResult* res = &nvEventSet->results[j];
            res->startValue = 0.0;
            res->stopValue = 0.0;
            res->currentValue = 0.0;
            res->fullValue = 0.0;
            res->overflows = 0;
        }

        for (j = 0; j < device->numActiveCuGroups; j++)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Enable group %ld, device->activeCuGroups[j]);
            CUPTI_CALL((*cuptiEventGroupSetEnablePtr)(device->activeCuGroups[j]), return -EFAULT);
        }

        // If we added the device context to the stack, pop it again
        if (popContext)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Pop Context %ld for device %d, device->context, device->deviceId);
            CU_CALL((*cuCtxPopCurrentPtr)(&device->context), return -EFAULT);
        }
    }

    return 0;
}

int
nvmon_stopCounters(void)
{
    int i = 0, j = 0, k = 0;
    int oldDevId = -1;
    uint64_t timestamp = 0;
    CUcontext curContext;
    if ((!nvGroupSet) || (!nvmon_initialized) || (nvGroupSet->activeGroup < 0))
    {
        return -EFAULT;
    }

    // Currently we are on which device?
    CUDA_CALL((*cudaGetDevicePtr)(&oldDevId), return -EFAULT);
    // Take the timestamp, we assign it later for all devices
    CUPTI_CALL((*cuptiGetTimestampPtr)(&timestamp), return -EFAULT);


    for (i = 0; i < nvGroupSet->numberOfGPUs; i++)
    {
        int popContext = 0;
        uint32_t one = 1;
        int maxTotalInstances = 0;
        size_t valuesSize = 0;
        NvmonDevice_t device = &nvGroupSet->gpus[i];
        if (device->deviceId != oldDevId)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Change GPU device %d -> %d, oldDevId, device->deviceId);
            CUDA_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
            CUDA_CALL((*cudaGetDevicePtr)(&oldDevId), return -EFAULT);
        }
        CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);
        NvmonEventSet* nvEventSet = &device->nvEventSets[nvGroupSet->activeGroup];
        if (device->numActiveCuGroups == 0)
        {
            DEBUG_PRINT(DEBUGLEV_DETAIL, Skipping GPU%d it has no events in group %d, device->deviceId, nvGroupSet->activeGroup);
            continue;
        }
        device->timeStop = timestamp;

        // Are we in the proper context?
        if (device->context != curContext)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Push Context %ld -> %ld for device %d, curContext, device->context, device->deviceId);
            CU_CALL((*cuCtxPushCurrentPtr)(device->context), return -EFAULT);
            popContext = 1;
        }
        else
        {
            // Although we are already in the right context, we set it again
            // to be sure
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Set Context %ld for device %d, device->context, device->deviceId);
            CU_CALL((*cuCtxSetCurrentPtr)(device->context), return -EFAULT);
        }

        for (j = 0; j < device->numActiveEvents; j++)
        {
            maxTotalInstances = MAX(maxTotalInstances, device->activeEvents[j].numTotalInstances);
        }
        uint64_t *tmpValues = (uint64_t *) malloc(maxTotalInstances * sizeof(uint64_t));

        for (j = 0; j < device->numActiveEvents; j++)
        {
            NvmonActiveEvent_t event = &device->activeEvents[j];
            valuesSize = sizeof(uint64_t) * event->numTotalInstances;
            memset(tmpValues, 0, valuesSize);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read Grp %ld Ev %ld for device %d, event->cuGroup, event->cuEventId, device->deviceId);
            CUPTI_CALL((*cuptiEventGroupReadEventPtr)(event->cuGroup, CUPTI_EVENT_READ_FLAG_NONE, event->cuEventId, &valuesSize, tmpValues), return -EFAULT);
            uint64_t valuesum = 0;
            for (k = 0; k < event->numInstances; k++)
            {
                valuesum += tmpValues[k];
            }
            NvmonEventResult* res = &nvEventSet->results[event->idxInSet];
            res->stopValue = (double)valuesum;
            res->lastValue = res->currentValue;
            res->fullValue += res->stopValue - res->startValue;
            res->lastValue += res->stopValue - res->currentValue;
            res->currentValue = (double)valuesum;
        }
        for (j = 0; j < device->numActiveCuGroups; j++)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Disable group %ld, device->activeCuGroups[j]);
            CUPTI_CALL((*cuptiEventGroupSetDisablePtr)(device->activeCuGroups[j]), return -EFAULT);
        }
        // If we added the device context to the stack, pop it again
        if (popContext)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Pop Context %ld for device %d, device->context, device->deviceId);
            CU_CALL((*cuCtxPopCurrentPtr)(&device->context), return -EFAULT);
        }
    }

    return 0;
}

int nvmon_readCounters(void)
{

    int i = 0, j = 0, k = 0, l = 0, m = 0;
    int oldDevId = -1;
    uint64_t timestamp = 0;
    CUcontext curContext;
    size_t sizeofuint32num = sizeof(uint32_t);
    int maxTotalInstances = 0;
    if ((!nvGroupSet) || (!nvmon_initialized) || (nvGroupSet->activeGroup < 0))
    {
        return -EFAULT;
    }

    for (i = 0; i < nvGroupSet->numberOfGPUs; i++)
    {
        NvmonDevice_t device = &nvGroupSet->gpus[i];
        for (j = 0; j < device->numActiveEvents; j++)
        {
            maxTotalInstances = MAX(maxTotalInstances, device->activeEvents[j].numTotalInstances);
        }
    }
    // In this array we collect the instance values of an events (summed up later)
    size_t valuesSize = sizeof(uint64_t) * maxTotalInstances;
    uint64_t *tmpValues = (uint64_t *) malloc(valuesSize);
    if (!tmpValues)
    {
        ERROR_PRINT(Not enough memory to allocate space for instance values);
        return -ENOMEM;
    }

    // Currently we are on which device?
    CUDA_CALL((*cudaGetDevicePtr)(&oldDevId), return -EFAULT);
    // Take the timestamp, we assign it later for all devices
    CUPTI_CALL((*cuptiGetTimestampPtr)(&timestamp), return -EFAULT);
/*    // This is a workaround to (eventually create and) get the current context*/
/*    CUDA_CALL((*cudaFreePtr)(NULL), return -EFAULT);*/
/*    CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);*/

    for (i = 0; i < nvGroupSet->numberOfGPUs; i++)
    {
        int popContext = 0;
        NvmonDevice_t device = &nvGroupSet->gpus[i];
        if (device->numActiveEvents == 0)
        {
            DEBUG_PRINT(DEBUGLEV_DETAIL, Skipping GPU%d it has no events in group %d, device->deviceId, nvGroupSet->activeGroup);
            continue;
        }
        if (device->deviceId != oldDevId)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Change GPU device %d -> %d, oldDevId, device->deviceId);
            CUDA_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
            CUDA_CALL((*cudaGetDevicePtr)(&oldDevId), return -EFAULT);
        }
        CUDA_CALL((*cudaFreePtr)(NULL), return -EFAULT);
        CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);
        device->timeRead = timestamp;

        // Are we in the proper context?
        if (curContext != device->context)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Push Context %ld -> %ld for device %d, curContext, device->context, device->deviceId);
            CU_CALL((*cuCtxPushCurrentPtr)(device->context), return -EFAULT);
            popContext = 1;
        }
        else
        {
            // Although we are already in the right context, we set it again
            // to be sure
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Set Context %ld for device %d, device->context, device->deviceId);
            CU_CALL((*cuCtxSetCurrentPtr)(device->context), return -EFAULT);
        }
        // Synchronize devices. I'm not sure whether this is required as each
        // device measures it's own events
        CU_CALL((*cuCtxSynchronizePtr)(), return -EFAULT);
        NvmonEventSet* nvEventSet = &device->nvEventSets[nvGroupSet->activeGroup];

        for (j = 0; j < device->numActiveEvents; j++)
        {
            NvmonActiveEvent_t event = &device->activeEvents[j];
            // Empty space for instance values
            valuesSize = sizeof(uint64_t) * event->numTotalInstances;
            memset(tmpValues, 0, valuesSize);
            // Read all instance values
            //printf("%d %d %d\n", event->cuGroup, event->cuEventId, valuesSize);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read Grp %ld Ev %ld for device %d, event->cuGroup, event->cuEventId, device->deviceId);
            CUPTI_CALL((*cuptiEventGroupReadEventPtr)(event->cuGroup, CUPTI_EVENT_READ_FLAG_NONE, event->cuEventId, &valuesSize, tmpValues), return -EFAULT);
            // Sum all instance values
            uint64_t valuesum = 0;
            for (k = 0; k < event->numInstances; k++)
            {
                valuesum += tmpValues[k];
            }

            NvmonEventResult* res = &nvEventSet->results[event->idxInSet];
            res->lastValue = res->currentValue;
            res->currentValue = (double)valuesum;
            res->fullValue += res->currentValue - res->startValue;
            res->lastValue += res->currentValue - res->lastValue;
        }
        if (popContext)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Pop Context %ld for device %d, device->context, device->deviceId);
            CU_CALL((*cuCtxPopCurrentPtr)(&device->context), return -EFAULT);
        }

    }
    free(tmpValues);
    return 0;
}

double nvmon_getResult(int groupId, int eventId, int gpuId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (nvGroupSet->activeGroup < 0))
    {
        return -EFAULT;
    }
    if (gpuId < 0 || gpuId >= nvGroupSet->numberOfGPUs)
    {
        return -EFAULT;
    }
    NvmonDevice *device = &nvGroupSet->gpus[gpuId];
    if (groupId < 0 || groupId >= device->numNvEventSets)
    {
        return -EFAULT;
    }
    NvmonEventSet* evset = &device->nvEventSets[nvGroupSet->activeGroup];
    if (eventId < 0 || eventId >= evset->numberOfEvents)
    {
        return -EFAULT;
    }
    return evset->results[eventId].fullValue;
}


int
nvmon_switchActiveGroup(int new_group)
{
    nvmon_stopCounters();
    nvmon_setupCounters(new_group);
    nvmon_startCounters();
    return 0;
}

double nvmon_getLastResult(int groupId, int eventId, int gpuId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (nvGroupSet->activeGroup < 0))
    {
        return -EFAULT;
    }
    if (gpuId < 0 || gpuId >= nvGroupSet->numberOfGPUs)
    {
        return -EFAULT;
    }
    NvmonDevice *device = &nvGroupSet->gpus[gpuId];
    if (groupId < 0 || groupId >= device->numNvEventSets)
    {
        return -EFAULT;
    }
    NvmonEventSet* evset = &device->nvEventSets[nvGroupSet->activeGroup];
    if (eventId < 0 || eventId >= evset->numberOfEvents)
    {
        return -EFAULT;
    }
    return evset->results[eventId].lastValue;
}





int nvmon_getNumberOfGroups(void)
{
    if ((!nvGroupSet) || (!nvmon_initialized))
    {
        return -EFAULT;
    }
    return nvGroupSet->numberOfActiveGroups;
}
int nvmon_getIdOfActiveGroup(void)
{
    if ((!nvGroupSet) || (!nvmon_initialized))
    {
        return -EFAULT;
    }
    return nvGroupSet->activeGroup;
}

int nvmon_getNumberOfGPUs(void)
{
    if ((!nvGroupSet) || (!nvmon_initialized))
    {
        return -EFAULT;
    }
    return nvGroupSet->numberOfGPUs;
}

int nvmon_getNumberOfEvents(int groupId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return -EFAULT;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    return ginfo->nevents;
}

double nvmon_getTimeOfGroup(int groupId)
{
    int i = 0;
    double t = 0;
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return -EFAULT;
    }
    for (i = 0; i < nvGroupSet->numberOfGPUs; i++)
    {
        t = MAX(t, (double)(nvGroupSet->gpus[i].timeStop - nvGroupSet->gpus[i].timeStart));
    }
    return t*1E-9;
}


double nvmon_getLastTimeOfGroup(int groupId)
{
    int i = 0;
    double t = 0;
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return -EFAULT;
    }
    for (i = 0; i < nvGroupSet->numberOfGPUs; i++)
    {
        t = MAX(t, (double)(nvGroupSet->gpus[i].timeStop - nvGroupSet->gpus[i].timeRead));
    }
    return t*1E-9;
}

char* nvmon_getEventName(int groupId, int eventId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    if ((eventId < 0) || (eventId >= ginfo->nevents))
    {
        return NULL;
    }
    return ginfo->events[eventId];
}

char* nvmon_getCounterName(int groupId, int eventId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    if ((eventId < 0) || (eventId >= ginfo->nevents))
    {
        return NULL;
    }
    return ginfo->counters[eventId];
}

char* nvmon_getMetricName(int groupId, int metricId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    if ((metricId < 0) || (metricId >= ginfo->nmetrics))
    {
        return NULL;
    }
    return ginfo->metricnames[metricId];
}

char* nvmon_getGroupName(int groupId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    return ginfo->groupname;
}

char* nvmon_getGroupInfoShort(int groupId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    return ginfo->shortinfo;
}

char* nvmon_getGroupInfoLong(int groupId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return NULL;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    return ginfo->longinfo;
}

int nvmon_getGroups(char*** groups, char*** shortinfos, char*** longinfos)
{
    int ret = 0;
    init_configuration();
    Configuration_t config = get_configuration();
    ret = get_groups(config->groupPath, "nvidiagpu", groups, shortinfos, longinfos);
    return ret;
}

int nvmon_returnGroups(int nrgroups, char** groups, char** shortinfos, char** longinfos)
{
    return_groups(nrgroups, groups, shortinfos, longinfos);
}

int nvmon_getNumberOfMetrics(int groupId)
{
    if ((!nvGroupSet) || (!nvmon_initialized) || (groupId < 0) || groupId >= nvGroupSet->numberOfActiveGroups)
    {
        return -EFAULT;
    }
    GroupInfo* ginfo = &nvGroupSet->groups[groupId];
    return ginfo->nmetrics;
}

void nvmon_setVerbosity(int level)
{
    if (level >= DEBUGLEV_ONLY_ERROR && level <= DEBUGLEV_DEVELOP)
    {
        likwid_nvmon_verbosity = level;
    }
}


double nvmon_getMetric(int groupId, int metricId, int gpuId)
{
    return 0.0;
}

double nvmon_getLastMetric(int groupId, int metricId, int gpuId)
{
    return 0.0;
}

double nvmon_getMetricOfRegionGpu(int region, int metricId, int threadId)
{
    return 0.0;
}


void
nvmon_printMarkerResults()
{
    int i = 0, j = 0, k = 0;
    for (i=0; i<gMarkerRegions; i++)
    {
        printf("Region %d : %s\n", i, bdata(gMarkerResults[i].tag));
        printf("Group %d\n", gMarkerResults[i].groupID);
        for (j=0;j<gMarkerResults[i].gpuCount; j++)
        {
            printf("GPU %d:\n", j, gMarkerResults[i].gpulist[j]);
            printf("\t Measurement time %f sec\n", gMarkerResults[i].time[j]);
            printf("\t Call count %d\n", gMarkerResults[i].count[j]);
            for(k=0;k<gMarkerResults[i].eventCount;k++)
            {
                printf("\t Event %d : %f\n", k, gMarkerResults[i].counters[j][k]);
            }
        }
    }
}

int
nvmon_readMarkerFile(const char* filename)
{
    int ret = 0, i = 0;
    FILE* fp = NULL;
    char buf[2048];
    buf[0] = '\0';
    char *ptr = NULL;
    int gpus = 0, groups = 0, regions = 0;
    int nr_regions = 0;

    if (filename == NULL)
    {
        return -EINVAL;
    }
    if (access(filename, R_OK))
    {
        return -EINVAL;
    }
    fp = fopen(filename, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Error opening file %s\n", filename);
    }
    ptr = fgets(buf, sizeof(buf), fp);
    ret = sscanf(buf, "%d %d %d", &gpus, &regions, &groups);
    if (ret != 3)
    {
        fprintf(stderr, "GPUMarker file missformatted.\n");
        return -EINVAL;
    }
    gMarkerResults = realloc(gMarkerResults, regions * sizeof(LikwidNvResults));
    if (gMarkerResults == NULL)
    {
        fprintf(stderr, "Failed to allocate %lu bytes for the marker results storage\n", regions * sizeof(LikwidNvResults));
        return -ENOMEM;
    }
    int* regionGPUs = (int*)malloc(regions * sizeof(int));
    if (regionGPUs == NULL)
    {
        fprintf(stderr, "Failed to allocate %lu bytes for temporal gpu count storage\n", regions * sizeof(int));
        return -ENOMEM;
    }
    gMarkerRegions = regions;
    for ( uint32_t i=0; i < regions; i++ )
    {
        regionGPUs[i] = 0;
        gMarkerResults[i].gpuCount = gpus;
        gMarkerResults[i].time = (double*) malloc(gpus * sizeof(double));
        if (!gMarkerResults[i].time)
        {
            fprintf(stderr, "Failed to allocate %lu bytes for the time storage\n", gpus * sizeof(double));
            break;
        }
        gMarkerResults[i].count = (uint32_t*) malloc(gpus * sizeof(uint32_t));
        if (!gMarkerResults[i].count)
        {
            fprintf(stderr, "Failed to allocate %lu bytes for the count storage\n", gpus * sizeof(uint32_t));
            break;
        }
        gMarkerResults[i].gpulist = (int*) malloc(gpus * sizeof(int));
        if (!gMarkerResults[i].count)
        {
            fprintf(stderr, "Failed to allocate %lu bytes for the gpulist storage\n", gpus * sizeof(int));
            break;
        }
        gMarkerResults[i].counters = (double**) malloc(gpus * sizeof(double*));
        if (!gMarkerResults[i].counters)
        {
            fprintf(stderr, "Failed to allocate %lu bytes for the counter result storage\n", gpus * sizeof(double*));
            break;
        }
    }
    while (fgets(buf, sizeof(buf), fp))
    {
        if (strchr(buf,':'))
        {
            int regionid = 0, groupid = -1;
            char regiontag[100];
            char* ptr = NULL;
            char* colonptr = NULL;
            regiontag[0] = '\0';
            ret = sscanf(buf, "%d:%s", &regionid, regiontag);

            ptr = strrchr(regiontag,'-');
            colonptr = strchr(buf,':');
            if (ret != 2 || ptr == NULL || colonptr == NULL)
            {
                fprintf(stderr, "Line %s not a valid region description\n", buf);
                continue;
            }
            groupid = atoi(ptr+1);
            snprintf(regiontag, strlen(regiontag)-strlen(ptr)+1, "%s", &(buf[colonptr-buf+1]));
            gMarkerResults[regionid].groupID = groupid;
            gMarkerResults[regionid].tag = bfromcstr(regiontag);
            nr_regions++;
        }
        else
        {
            int regionid = 0, groupid = 0, gpu = 0, count = 0, nevents = 0;
            int gpuidx = 0, eventidx = 0;
            double time = 0;
            char remain[1024];
            remain[0] = '\0';
            ret = sscanf(buf, "%d %d %d %d %lf %d %[^\t\n]", &regionid, &groupid, &gpu, &count, &time, &nevents, remain);
            if (ret != 7)
            {
                fprintf(stderr, "Line %s not a valid region values line\n", buf);
                continue;
            }
            if (gpu >= 0)
            {
                gpuidx = regionGPUs[regionid];
                gMarkerResults[regionid].gpulist[gpuidx] = gpu;
                gMarkerResults[regionid].eventCount = nevents;
                gMarkerResults[regionid].time[gpuidx] = time;
                gMarkerResults[regionid].count[gpuidx] = count;
                gMarkerResults[regionid].counters[gpuidx] = malloc(nevents * sizeof(double));

                eventidx = 0;
                ptr = strtok(remain, " ");
                while (ptr != NULL && eventidx < nevents)
                {
                    sscanf(ptr, "%lf", &(gMarkerResults[regionid].counters[gpuidx][eventidx]));
                    ptr = strtok(NULL, " ");
                    eventidx++;
                }
                regionGPUs[regionid]++;
            }
        }
    }
    for ( uint32_t i=0; i < regions; i++ )
    {
        gMarkerResults[i].gpuCount = regionGPUs[i];
    }
    free(regionGPUs);
    fclose(fp);
    return nr_regions;
}

void
nvmon_destroyMarkerResults()
{
    int i = 0, j = 0;
    if (gMarkerResults != NULL)
    {
        for (i = 0; i < gMarkerRegions; i++)
        {
            free(gMarkerResults[i].time);
            free(gMarkerResults[i].count);
            free(gMarkerResults[i].gpulist);
            for (j = 0; j < gMarkerResults[i].gpuCount; j++)
            {
                free(gMarkerResults[i].counters[j]);
            }
            free(gMarkerResults[i].counters);
            bdestroy(gMarkerResults[i].tag);
        }
        free(gMarkerResults);
        gMarkerResults = NULL;
        gMarkerRegions = 0;
    }
}


int
nvmon_getCountOfRegion(int region, int gpu)
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return -EINVAL;
    }
    if (gpu < 0 || gpu >= gMarkerResults[region].gpuCount)
    {
        return -EINVAL;
    }
    if (gMarkerResults[region].count == NULL)
    {
        return 0;
    }
    return gMarkerResults[region].count[gpu];
}

double
nvmon_getTimeOfRegion(int region, int gpu)
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return -EINVAL;
    }
    if (gpu < 0 || gpu >= gMarkerResults[region].gpuCount)
    {
        return -EINVAL;
    }
    if (gMarkerResults[region].time == NULL)
    {
        return 0.0;
    }
    return gMarkerResults[region].time[gpu];
}

int
nvmon_getGpulistOfRegion(int region, int count, int* gpulist)
{
    int i;
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return -EINVAL;
    }
    if (gpulist == NULL)
    {
        return -EINVAL;
    }
    for (i=0; i< MIN(count, gMarkerResults[region].gpuCount); i++)
    {
        gpulist[i] = gMarkerResults[region].gpulist[i];
    }
    return MIN(count, gMarkerResults[region].gpuCount);
}

int
nvmon_getGpusOfRegion(int region)
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return -EINVAL;
    }
    return gMarkerResults[region].gpuCount;
}

int
nvmon_getMetricsOfRegion(int region)
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return -EINVAL;
    }
    return nvmon_getNumberOfMetrics(gMarkerResults[region].groupID);
}

int
nvmon_getNumberOfRegions()
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return -EINVAL;
    }
    return gMarkerRegions;
}

int
nvmon_getGroupOfRegion(int region)
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return -EINVAL;
    }
    return gMarkerResults[region].groupID;
}

char*
nvmon_getTagOfRegion(int region)
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return NULL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return NULL;
    }
    return bdata(gMarkerResults[region].tag);
}

int
nvmon_getEventsOfRegion(int region)
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Nvmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return -EINVAL;
    }
    return gMarkerResults[region].eventCount;
}

double nvmon_getResultOfRegionGpu(int region, int event, int gpu)
{
    if (gMarkerResults == NULL)
    {
        ERROR_PLAIN_PRINT(Perfmon module not properly initialized);
        return -EINVAL;
    }
    if (region < 0 || region >= gMarkerRegions)
    {
        return -EINVAL;
    }
    if (gpu < 0 || gpu >= gMarkerResults[region].gpuCount)
    {
        return -EINVAL;
    }
    if (event < 0 || event >= gMarkerResults[region].eventCount)
    {
        return -EINVAL;
    }
    if (gMarkerResults[region].counters[gpu] == NULL)
    {
        return 0.0;
    }
    return gMarkerResults[region].counters[gpu][event];
}
