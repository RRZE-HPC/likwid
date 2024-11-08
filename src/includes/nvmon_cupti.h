/*
 * =======================================================================================
 *
 *      Filename:  nvmon_cupti.h
 *
 *      Description:  Header File of nvmon module (CUPTI backend).
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

#ifndef LIKWID_NVMON_CUPTI_H
#define LIKWID_NVMON_CUPTI_H


#include <cuda.h>
#include <cupti.h>

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
            (*cuptiGetResultStringPtr)(_status, &errstr);               \
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

#ifndef CUAPIWEAK
#define CUAPIWEAK __attribute__( ( weak ) )
#endif
#define DECLARECUFUNC(funcname, funcsig) CUresult CUAPIWEAK funcname funcsig;  CUresult( *funcname##Ptr ) funcsig;
DECLARECUFUNC(cuCtxGetCurrent, (CUcontext *));
DECLARECUFUNC(cuCtxSetCurrent, (CUcontext));
DECLARECUFUNC(cuCtxDestroy, (CUcontext));
DECLARECUFUNC(cuDeviceGet, (CUdevice *, int));
DECLARECUFUNC(cuDeviceGetCount, (int *));
DECLARECUFUNC(cuDeviceGetName, (char *, int, CUdevice));
DECLARECUFUNC(cuInit, (unsigned int));
DECLARECUFUNC(cuCtxPopCurrent, (CUcontext * pctx));
DECLARECUFUNC(cuCtxPushCurrent, (CUcontext pctx));
DECLARECUFUNC(cuCtxSynchronize, ());

#ifndef CUDAAPIWEAK
#define CUDAAPIWEAK __attribute__( ( weak ) )
#endif
#define DECLARECUDAFUNC(funcname, funcsig) cudaError_t CUDAAPIWEAK funcname funcsig;  cudaError_t( *funcname##Ptr ) funcsig;
DECLARECUDAFUNC(cudaGetDevice, (int *));
DECLARECUDAFUNC(cudaSetDevice, (int));
DECLARECUDAFUNC(cudaFree, (void *));

#ifndef CUPTIAPIWEAK
#define CUPTIAPIWEAK __attribute__( ( weak ) )
#endif
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
link_cputi_libraries(void)
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
    cuCtxDestroyPtr = DLSYM_AND_CHECK(dl_libcuda, "cuCtxDestroy");

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


static int check_nv_context(NvmonDevice_t device, CUcontext currentContext)
{
    int j = 0;
    int need_pop = 0;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Current context %ld DevContext %ld, currentContext, device->context);
    if (!device->context)
    {
        int context_of_dev = -1;
        for (j = 0; j < nvGroupSet->numberOfGPUs; j++)
        {
            NvmonDevice_t dev = &nvGroupSet->gpus[j];
            if (dev->context == currentContext)
            {
                context_of_dev = j;
                break;
            }
        }
        if (context_of_dev < 0 && !device->context)
        {
            device->context = currentContext;
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Reuse context %ld for device %d, device->context, device->deviceId);
        }
        else
        {
            CUDA_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
            CUDA_CALL((*cudaFreePtr)(NULL), return -EFAULT);
            CU_CALL((*cuCtxGetCurrentPtr)(&device->context), return -EFAULT);
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, New context %ld for device %d, device->context, device->deviceId);
        }
    }
    else if (device->context != currentContext)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Use context %ld for device %d, device->context, device->deviceId);
        CU_CALL((*cuCtxPushCurrentPtr)(device->context), return -EFAULT);
        need_pop = 1;
    }
    else
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Context %ld fits for device %d, device->context, device->deviceId);
    }
    return need_pop;
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

void nvmon_cupti_freeDevice(NvmonDevice_t dev)
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
            CU_CALL((*cuCtxDestroyPtr)(dev->context), j++);
            dev->context = NULL;
        }
        if (dev->cuEventSets)
        {
            CUPTI_CALL((*cuptiEventGroupSetsDestroyPtr)(dev->cuEventSets), j++);
            dev->cuEventSets = NULL;
        }

        // The event objects are freed here

        if (dev->activeEvents)
        {
            free(dev->activeEvents);
            dev->activeEvents = NULL;
            dev->numActiveEvents = 0;
        }

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
        if (dev->allevents)
        {
            for (j = 0; j < dev->numAllEvents; j++)
            {
                if (dev->allevents[j])
                {
                    free(dev->allevents[j]);
                }
            }
        }
        memset(dev, 0, sizeof(NvmonDevice));
    }
    return;
}

int
nvmon_cupti_createDevice(int id, NvmonDevice *dev)
{
    int c = 0;
    unsigned numDomains = 0;
    CUpti_EventDomainID* eventDomainIds = NULL;
    int eventIdx = 0;
    uint32_t totalEvents = 0;


    dev->deviceId = id;
    dev->cuEventSets = NULL;
    dev->context = 0UL;
    dev->activeEvents = NULL;
    dev->numActiveEvents = 0;
    dev->numNvEventSets = 0;
    dev->nvEventSets = NULL;

    if ((!dl_libcuda) || (!dl_libcudart) || (!dl_libcupti))
    {
        int err = link_cputi_libraries();
        if (err < 0)
        {
            return -1;
        }
    }

    // Assign device ID and get cuDevice from CUDA
    CU_CALL((*cuDeviceGetPtr)(&dev->cuDevice, id), return -1);

    // Get the number of event domains of the device
    CUPTI_CALL((*cuptiDeviceGetNumEventDomainsPtr)(dev->cuDevice, &numDomains), return -1);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Nvmon: Dev %d Domains %d, id, numDomains);

    // Get the domain IDs for the device
    size_t domainarraysize = numDomains * sizeof(CUpti_EventDomainID);
    eventDomainIds = malloc(domainarraysize);
    if (!eventDomainIds) return -ENOMEM;
    CUPTI_CALL((*cuptiDeviceEnumEventDomainsPtr)(dev->cuDevice, &domainarraysize, eventDomainIds), return -1);


    // Count the events in all domains to allocate the event list
    dev->numAllEvents = 0;
    for (unsigned j = 0; j < numDomains; j++)
    {
        uint32_t domainNumEvents = 0;
        CUpti_EventDomainID domainID = eventDomainIds[j];
        CUPTI_CALL((*cuptiEventDomainGetNumEventsPtr)(domainID, &domainNumEvents), return -1);
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Nvmon: Dev %d Domain %d Events %d, id, j, domainNumEvents);
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

    for (unsigned j = 0; j < numDomains; j++)
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
        for (unsigned k = 0; k < domainNumEvents; k++)
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
            event->type = NVMON_CUPTI_EVENT;
            event->active = 0;
            //GPUDEBUG_PRINT(DEBUGLEV_DETAIL, New Event %d CuEvent %d Domain %d CuDomain %d Name %s, event->eventId, (int)event->cuEventId, event->domainId, (int)event->cuDomainId, event->name);
            // Add the object to the event list
            dev->allevents[dev->numAllEvents] = event;
            dev->numAllEvents++;
            // Add the object to the hash tables
            char* nameKey = g_strdup(event->name);

            CUpti_EventID* idKey = malloc(sizeof(CUpti_EventID));
            if (idKey)
            {
                *idKey = event->cuEventId;
                // Key is event name
                g_hash_table_insert(dev->eventHash, (gpointer)nameKey, (gpointer)event);
                // Key is CUPTI event ID
                g_hash_table_insert(dev->evIdHash, (gpointer)idKey, (gpointer)event);
            }
        }
    }

/*    CUDA_CALL((*cudaSetDevicePtr)(dev->deviceId), return -EFAULT);*/
/*    CUDA_CALL((*cudaFreePtr)(NULL), return -EFAULT);*/
/*    CU_CALL((*cuCtxGetCurrentPtr)(&dev->context), return -EFAULT);*/
    return 0;
}



int nvmon_cupti_getEventsOfGpu(int gpuId, NvmonEventList_t* list)
{
    int ret = 0;
    NvmonDevice device;
    int err = nvmon_cupti_createDevice(gpuId, &device);
    if (!err)
    {
        NvmonEventList_t l = malloc(sizeof(NvmonEventList));
        if (l)
        {
            l->events = malloc(sizeof(NvmonEventListEntry) * device.numAllEvents);
            if (l->events)
            {
                for (int i = 0; i < device.numAllEvents; i++)
                {
                    NvmonEventListEntry* out = &l->events[i];
                    NvmonEvent_t event = device.allevents[i];
                    out->name = malloc(strlen(event->name)+2);
                    if (out->name)
                    {
                        ret = snprintf(out->name, strlen(event->name)+1, "%s", event->name);
                        if (ret > 0)
                        {
                            out->name[ret] = '\0';
                        }
                    }
                    out->desc = malloc(strlen(event->description)+2);
                    if (out->desc)
                    {
                        ret = snprintf(out->desc, strlen(event->description)+1, "%s", event->description);
                        if (ret > 0)
                        {
                            out->desc[ret] = '\0';
                        }
                    }
                    out->limit = malloc(10*sizeof(char));
                    if (out->limit)
                    {
                        switch (event->type)
                        {
                            case NVMON_CUPTI_EVENT:
                                ret = snprintf(out->limit, 9, "GPU");
                                if (ret > 0) out->limit[ret] = '\0';
                                break;
                            default:
                                break;
                        }
                    }
                }
                l->numEvents = device.numAllEvents;
                *list = l;
            }
            else
            {
                free(l);
                nvmon_cupti_freeDevice(&device);
                return -ENOMEM;
            }
        }
    }
    else
    {
        ERROR_PRINT(No such device %d, gpuId);
    }
    return 0;
}



int
nvmon_cupti_addEventSets(NvmonDevice_t device, const char* eventString)
{
    int i = 0;
    int err = 0;
    int curDeviceId = -1;
    CUcontext curContext;
    struct bstrList* eventtokens = NULL;

    CUDA_CALL((*cudaGetDevicePtr)(&curDeviceId), return -EFAULT);
    CUDA_CALL((*cudaFreePtr)(NULL), return -EFAULT);
    CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);

    bstring eventBString = bfromcstr(eventString);
    eventtokens = bsplit(eventBString, ',');
    bdestroy(eventBString);

    if (curDeviceId != device->deviceId)
    {
        CUDA_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
    }

    size_t sizeBytes = (eventtokens->qty) * sizeof(CUpti_EventID);

    int popContext = 0;
    CUpti_EventGroupSets * cuEventSets = NULL;

    NvmonEventSet* tmpEventSet = realloc(device->nvEventSets, (device->numNvEventSets+1)*sizeof(NvmonEventSet));
    if (!tmpEventSet)
    {
        ERROR_PRINT(Cannot enlarge GPU %d eventSet list, device->deviceId);
        return -ENOMEM;
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
    popContext = check_nv_context(device, curContext);


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
            GPUDEBUG_PRINT(DEBUGLEV_INFO, NVMON: Event %s unknown. Skipping..., bdata(evset->entry[0]));
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
    devEventSet->id = nvGroupSet->numberOfActiveGroups;
    // Create an eventset with the currently configured and current event
    if(popContext)
    {
        CU_CALL((*cuCtxPopCurrentPtr)(&device->context), return -EFAULT);
    }
    device->numNvEventSets++;
    return 0;
}

int nvmon_cupti_setupCounters(NvmonDevice_t device, NvmonEventSet* eventSet)
{
    int popContext = 0;
    int oldDevId = -1;
    CUpti_EventGroupSets * cuEventSets = NULL;
    CUcontext curContext;

    if (eventSet->numberOfEvents == 0)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DETAIL, Skipping GPU%d it has no events in group %d, device->deviceId, eventSet->id);
        return -EINVAL;
    }
    // Currently we are on which device?
    CUDA_CALL((*cudaGetDevicePtr)(&oldDevId), return -EFAULT);
    // This is a workaround to (eventually create and) get the current context
    CUDA_CALL((*cudaFreePtr)(NULL), return -EFAULT);
    CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);

    popContext = check_nv_context(device, curContext);


    size_t grpEventIdsSize = eventSet->numberOfEvents * sizeof(CUpti_EventID);
    CUPTI_CALL((*cuptiEventGroupSetsCreatePtr)(device->context, grpEventIdsSize, eventSet->cuEventIDs, &cuEventSets), return -1;);
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
    device->activeEvents = malloc(eventSet->numberOfEvents * sizeof(NvmonActiveEvent));
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
    for (unsigned j = 0; j < cuEventSets->numSets; j++)
    {
        size_t sizeofuint32t = sizeof(uint32_t);
        uint32_t numGEvents = 0, numGInstances = 0, numTotalGInstances = 0;
        CUpti_EventGroupSet* groupset = &cuEventSets->sets[j];

        for (unsigned k = 0; k < groupset->numEventGroups; k++)
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

            for (unsigned l = 0; l < numGEvents; l++)
            {
                for (int m = 0; m <  eventSet->numberOfEvents; m++)
                {
                    if (eventSet->cuEventIDs[m] == grpEventIds[l])
                    {
                        CUpti_EventDomainID did = eventSet->nvEvents[m]->cuDomainId;
                        // Get total instance count for a group
                        CUPTI_CALL((*cuptiDeviceGetEventDomainAttributePtr)(device->cuDevice, did, CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &sizeofuint32t, &numTotalGInstances), return -EFAULT);
                        device->activeEvents[m].eventId = eventSet->nvEvents[m]->eventId;
                        device->activeEvents[m].idxInSet = m;
                        device->activeEvents[m].groupId = eventSet->id;
                        device->activeEvents[m].cuEventId = eventSet->nvEvents[m]->cuEventId;
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
                        eventSet->nvEvents[m]->active = 1;
                        GPUDEBUG_PRINT(DEBUGLEV_INFO, Setup event %s (%d) for GPU %d, eventSet->nvEvents[m]->name, device->activeEvents[m].cuEventId, device->deviceId);
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
    return 0;
}

int nvmon_cupti_startCounters(NvmonDevice_t device)
{
    int j = 0;
    CUcontext curContext;
    int popContext = 0;
    uint32_t one = 1;
    uint64_t timestamp = 0;
    int oldDevId = -1;

    // Currently we are on which device?
    CUDA_CALL((*cudaGetDevicePtr)(&oldDevId), return -EFAULT);
    // Take the timestamp, we assign it later for all devices
    CUPTI_CALL((*cuptiGetTimestampPtr)(&timestamp), return -EFAULT);
    // This is a workaround to (eventually create and) get the current context
    CUDA_CALL((*cudaFreePtr)(NULL), return -EFAULT);
    CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);


    //NvmonDevice_t device = &nvGroupSet->gpus[i];
    if (device->numActiveCuGroups == 0)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DETAIL, Skipping GPU%d it has no events in group %d, device->deviceId, nvGroupSet->activeGroup);
        return 0;
    }
    if (device->deviceId != oldDevId)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Change GPU device %d -> %d, oldDevId, device->deviceId);
        CUDA_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
        CUDA_CALL((*cudaGetDevicePtr)(&oldDevId), return -EFAULT);
    }
    CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);
    device->timeStart = timestamp;
    device->timeRead = timestamp;

    // Are we in the proper context?
    popContext = check_nv_context(device, curContext);

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
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Enable group %ld on Dev %d, device->activeCuGroups[j], device->deviceId);
        CUPTI_CALL((*cuptiEventGroupSetEnablePtr)(device->activeCuGroups[j]), return -EFAULT);
    }

    // If we added the device context to the stack, pop it again
    if (popContext)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Pop Context %ld for device %d, device->context, device->deviceId);
        CU_CALL((*cuCtxPopCurrentPtr)(&device->context), return -EFAULT);
    }
    return 0;
}


int nvmon_cupti_stopCounters(NvmonDevice_t device)
{
    (void)device; // unused

    int oldDevId = -1;
    uint64_t timestamp = 0;
    CUcontext curContext;

    CUDA_CALL((*cudaGetDevicePtr)(&oldDevId), return -EFAULT);
    // Take the timestamp, we assign it later for all devices
    CUPTI_CALL((*cuptiGetTimestampPtr)(&timestamp), return -EFAULT);


    for (int i = 0; i < nvGroupSet->numberOfGPUs; i++)
    {
        int popContext = 0;
        uint32_t one = 1;
        unsigned maxTotalInstances = 0;
        size_t valuesSize = 0;
        NvmonDevice_t device = &nvGroupSet->gpus[i];
        if (device->deviceId != oldDevId)
        {
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Change GPU device %d -> %d, oldDevId, device->deviceId);
            CUDA_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
            CUDA_CALL((*cudaGetDevicePtr)(&oldDevId), return -EFAULT);
        }
        CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);
        NvmonEventSet* nvEventSet = &device->nvEventSets[nvGroupSet->activeGroup];
        if (device->numActiveCuGroups == 0)
        {
            GPUDEBUG_PRINT(DEBUGLEV_DETAIL, Skipping GPU%d it has no events in group %d, device->deviceId, nvGroupSet->activeGroup);
            continue;
        }
        device->timeStop = timestamp;

        // Are we in the proper context?
        popContext = check_nv_context(device, curContext);

        for (int j = 0; j < device->numActiveEvents; j++)
        {
            maxTotalInstances = MAX(maxTotalInstances, device->activeEvents[j].numTotalInstances);
        }
        uint64_t *tmpValues = (uint64_t *) malloc(maxTotalInstances * sizeof(uint64_t));

        for (int j = 0; j < device->numActiveEvents; j++)
        {
            NvmonActiveEvent_t event = &device->activeEvents[j];
            valuesSize = sizeof(uint64_t) * event->numTotalInstances;
            memset(tmpValues, 0, valuesSize);
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Read Grp %ld Ev %ld for device %d, event->cuGroup, event->cuEventId, device->deviceId);
            CUPTI_CALL((*cuptiEventGroupReadEventPtr)(event->cuGroup, CUPTI_EVENT_READ_FLAG_NONE, event->cuEventId, &valuesSize, tmpValues), free(tmpValues); return -EFAULT);
            uint64_t valuesum = 0;
            for (unsigned k = 0; k < event->numInstances; k++)
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
        for (int j = 0; j < device->numActiveCuGroups; j++)
        {
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Disable group %ld, device->activeCuGroups[j]);
            CUPTI_CALL((*cuptiEventGroupSetDisablePtr)(device->activeCuGroups[j]), return -EFAULT);
        }
        // If we added the device context to the stack, pop it again
        if (popContext)
        {
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Pop Context %ld for device %d, device->context, device->deviceId);
            CU_CALL((*cuCtxPopCurrentPtr)(&device->context), return -EFAULT);
        }
    }
    return 0;
}


int nvmon_cupti_readCounters(NvmonDevice_t device)
{
    int oldDevId = -1;
    uint64_t timestamp = 0;
    CUcontext curContext;
    size_t sizeofuint32num = sizeof(uint32_t);
    unsigned maxTotalInstances = 0;


    for (int j = 0; j < device->numActiveEvents; j++)
    {
        maxTotalInstances = MAX(maxTotalInstances, device->activeEvents[j].numTotalInstances);
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
    CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);


    int popContext = 0;
    if (device->numActiveEvents == 0)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DETAIL, Skipping GPU%d it has no events in group %d, device->deviceId, nvGroupSet->activeGroup);
        return 0;
    }
    if (device->deviceId != oldDevId)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Change GPU device %d -> %d, oldDevId, device->deviceId);
        CUDA_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
        CUDA_CALL((*cudaGetDevicePtr)(&oldDevId), return -EFAULT);
    }
    CUDA_CALL((*cudaFreePtr)(NULL), return -EFAULT);
    CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);
    device->timeRead = timestamp;

    // Are we in the proper context?
    popContext = check_nv_context(device, curContext);
    // Synchronize devices. I'm not sure whether this is required as each
    // device measures it's own events
    CU_CALL((*cuCtxSynchronizePtr)(), return -EFAULT);
    NvmonEventSet* nvEventSet = &device->nvEventSets[nvGroupSet->activeGroup];

    for (int j = 0; j < device->numActiveEvents; j++)
    {
        NvmonActiveEvent_t event = &device->activeEvents[j];
        // Empty space for instance values
        valuesSize = sizeof(uint64_t) * event->numTotalInstances;
        memset(tmpValues, 0, valuesSize);
        // Read all instance values
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Read Grp %ld Ev %ld for device %d, event->cuGroup, event->cuEventId, device->deviceId);
        CUPTI_CALL((*cuptiEventGroupReadEventPtr)(event->cuGroup, CUPTI_EVENT_READ_FLAG_NONE, event->cuEventId, &valuesSize, tmpValues), return -EFAULT);
        // Sum all instance values
        uint64_t valuesum = 0;
        for (unsigned k = 0; k < event->numInstances; k++)
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
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Pop Context %ld for device %d, device->context, device->deviceId);
        CU_CALL((*cuCtxPopCurrentPtr)(&device->context), return -EFAULT);
    }
    free(tmpValues);
    return 0;
}


NvmonFunctions nvmon_cupti_functions = {
    .freeDevice = nvmon_cupti_freeDevice,
    .createDevice = nvmon_cupti_createDevice,
    .getEventList = nvmon_cupti_getEventsOfGpu,
    .addEvents = nvmon_cupti_addEventSets,
    .setupCounters = nvmon_cupti_setupCounters,
    .startCounters = nvmon_cupti_startCounters,
    .stopCounters = nvmon_cupti_stopCounters,
    .readCounters = nvmon_cupti_readCounters,
};

#endif
