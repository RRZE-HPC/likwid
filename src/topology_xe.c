/*
 * =======================================================================================
 *
 *      Filename:  topology_xe.c
 *
 *      Description:  Topology module for Intel GPUs
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tg), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2020 RRZE, University Erlangen-Nuremberg
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
 // /opt/intel/oneapi/dal/2021.2.0/include/services/internal/sycl/level_zero_types.h
 // /opt/intel/oneapi/compiler/2021.2.0/linux/lib/libpi_level_zero.so
 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <unistd.h>
#include <sys/types.h>
#include <error.h>

#include <likwid.h>
#include <types.h>

#include <dlfcn.h>
//#include <sycl/level_zero_types.h>
#include <ze_api.h>
#include <zes_api.h>


void (*_dl_non_dynamic_init) (void) __attribute__ ((weak));

static void *topo_dl_libxe = NULL;
static void *topo_dl_libxeloader = NULL;

#define XE_CALL( call, handleerror )                                    \
    do {                                                                \
        ze_result_t _status = (call);                                      \
        if (_status != ZE_RESULT_SUCCESS) {                                  \
            fprintf(stderr, "Error: function %s failed with error %d.\n", #call, _status); \
            handleerror;                                                \
        }                                                               \
    } while (0)
#define XEAPIWEAK __attribute__( ( weak ) )
#define DECLAREXEFUNC(funcname, funcsig)  ze_result_t XEAPIWEAK ( *funcname##Ptr ) funcsig = NULL;

#ifndef ERROR_PLAIN_PRINT
#define str(x) #x
#define ERROR_PLAIN_PRINT(msg) \
   fprintf(stderr,  "ERROR - [%s:%s:%d] " str(msg) "\n", __FILE__, __func__,__LINE__);
#endif


static int accel_topology_xe_initialized = 0;
XeGpuTopology xeGpuTopology = {0, NULL};

int likwid_xemon_verbosity = DEBUGLEV_ONLY_ERROR;

#ifdef LIKWID_WITH_XEMON
DECLAREXEFUNC(zeInit, (ze_init_flags_t));
DECLAREXEFUNC(zeDriverGet, (uint32_t *, ze_driver_handle_t*));
DECLAREXEFUNC(zeDeviceGet, (ze_driver_handle_t , uint32_t *, ze_device_handle_t *));
DECLAREXEFUNC(zeDeviceGetProperties, (ze_device_handle_t , ze_device_properties_t *));
DECLAREXEFUNC(zeDeviceGetComputeProperties, (ze_device_handle_t, ze_device_compute_properties_t *));
DECLAREXEFUNC(zeDeviceGetModuleProperties, (ze_device_handle_t, ze_device_module_properties_t *));
DECLAREXEFUNC(zeDeviceGetCommandQueueGroupProperties, (ze_device_handle_t, uint32_t *, ze_command_queue_group_properties_t *));
DECLAREXEFUNC(zeDriverGetApiVersion, (ze_driver_handle_t, ze_api_version_t *));
DECLAREXEFUNC(zeDeviceGetMemoryProperties, (ze_device_handle_t, uint32_t *, ze_device_memory_properties_t *));
//DECLAREXEFUNC(zeDeviceGetMemoryAccessProperties, (ze_device_handle_t, ze_device_memory_access_properties_t *));
DECLAREXEFUNC(zeDeviceGetCacheProperties, (ze_device_handle_t, uint32_t *, ze_device_cache_properties_t *));
DECLAREXEFUNC(zeDeviceGetImageProperties, (ze_device_handle_t, ze_device_image_properties_t *));
DECLAREXEFUNC(zeDeviceGetExternalMemoryProperties, (ze_device_handle_t, ze_device_external_memory_properties_t *));
DECLAREXEFUNC(zeDeviceGetP2PProperties, (ze_device_handle_t, ze_driver_handle_t, ze_device_p2p_properties_t *));

//DECLAREXEFUNC(zesDevicePciGetProperties, (zes_device_handle_t, zes_pci_properties_t *));
DECLAREXEFUNC(zesDevicePciGetProperties, (zes_device_handle_t, zes_pci_properties_t *));
// TODO Sysman stuff for PCIe

static int ze_error(ze_result_t err)
{
    switch(err)
    {
        case ZE_RESULT_SUCCESS:
            return 0;
            break;
        case ZE_RESULT_ERROR_DEVICE_LOST:
            fprintf(stderr, "Device lost\n");
            return -ENODEV;
            break;
        case ZE_RESULT_NOT_READY:
            fprintf(stderr, "Device not ready\n");
            return -ENODEV;
            break;
        case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
            fprintf(stderr, "Not enough host memory\n");
            return -ENOMEM;
            break;
        case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
            fprintf(stderr, "Not enough device memory\n");
            return -ENOMEM;
            break;
        case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
            fprintf(stderr, "Build error\n");
            return -EINVAL;
            break;
        case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
            fprintf(stderr, "Build error\n");
            return -EINVAL;
            break;
        case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
            fprintf(stderr, "No permission\n");
            return -EPERM;
            break;
        case ZE_RESULT_ERROR_NOT_AVAILABLE:
            fprintf(stderr, "Not available\n");
            return -ENODEV;
            break;
        case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
            fprintf(stderr, "Dependency not available\n");
            return -ENODEV;
            break;
        case ZE_RESULT_ERROR_UNINITIALIZED:
            fprintf(stderr, "Not initialized\n");
            return -ENODEV;
            break;
        case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
            fprintf(stderr, "Version not supported\n");
            return -ENODEV;
            break;
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
            fprintf(stderr, "Feature not supported\n");
            return -ENODEV;
            break;
        case ZE_RESULT_ERROR_INVALID_ARGUMENT:
            fprintf(stderr, "Invalid argument\n");
            return -EINVAL;
            break;
        case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
            fprintf(stderr, "Invalid NULL handle\n");
            return -EINVAL;
            break;
        case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
            fprintf(stderr, "Handle busy\n");
            return -EBUSY;
            break;
        case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
            fprintf(stderr, "Invalid NULL pointer\n");
            return -EBUSY;
            break;
        default:
            fprintf(stderr, "Unknown error\n");
            break;
    }
    return -1;
}

static int
xetopo_link_libraries(void)
{
#define DLSYM_AND_CHECK( dllib, name ) dlsym( dllib, name ); if ( dlerror() != NULL ) { return -1; }
    /* Attempt to guess if we were statically linked to libc, if so bail */
    if(_dl_non_dynamic_init != NULL) {
        return -1;
    }
    topo_dl_libxe = dlopen("libze_loader.so", RTLD_NOW | RTLD_GLOBAL);
    if (!topo_dl_libxe)
    {
        fprintf(stderr, "Intel Xe library libpi_level_zero.so not found: %s\n", dlerror());
        return -1;
    }
/*    topo_dl_libxeloader = dlopen("libze_loader.so", RTLD_NOW | RTLD_GLOBAL);*/
/*    if (!topo_dl_libxe)*/
/*    {*/
/*        fprintf(stderr, "Intel Ze loader library libze_loader.so not found: %s\n", dlerror());*/
/*        return -1;*/
/*    }*/
    zeInitPtr = DLSYM_AND_CHECK(topo_dl_libxe, "zeInit");
    zeDriverGetPtr = DLSYM_AND_CHECK(topo_dl_libxe, "zeDriverGet");
    zeDeviceGetPtr = DLSYM_AND_CHECK(topo_dl_libxe, "zeDeviceGet");
    zeDeviceGetPropertiesPtr = DLSYM_AND_CHECK(topo_dl_libxe, "zeDeviceGetProperties");
    zeDeviceGetComputePropertiesPtr = DLSYM_AND_CHECK(topo_dl_libxe, "zeDeviceGetComputeProperties");
    zeDeviceGetModulePropertiesPtr = DLSYM_AND_CHECK(topo_dl_libxe, "zeDeviceGetModuleProperties");
    zeDeviceGetCommandQueueGroupPropertiesPtr = DLSYM_AND_CHECK(topo_dl_libxe, "zeDeviceGetCommandQueueGroupProperties");
    zeDeviceGetMemoryPropertiesPtr = DLSYM_AND_CHECK(topo_dl_libxe, "zeDeviceGetMemoryProperties");
    //zeDeviceGetMemoryAccessPropertiesPtr = DLSYM_AND_CHECK(topo_dl_libxe, "zeDeviceGetMemoryAccessProperties");
    zeDeviceGetCachePropertiesPtr = DLSYM_AND_CHECK(topo_dl_libxe, "zeDeviceGetCacheProperties");
    zeDeviceGetImagePropertiesPtr = DLSYM_AND_CHECK(topo_dl_libxe, "zeDeviceGetImageProperties");
    zeDeviceGetExternalMemoryPropertiesPtr = DLSYM_AND_CHECK(topo_dl_libxe, "zeDeviceGetExternalMemoryProperties");
    zeDeviceGetP2PPropertiesPtr = DLSYM_AND_CHECK(topo_dl_libxe, "zeDeviceGetP2PProperties");
    zeDriverGetApiVersionPtr = DLSYM_AND_CHECK(topo_dl_libxe, "zeDriverGetApiVersion");
    zesDevicePciGetPropertiesPtr = DLSYM_AND_CHECK(topo_dl_libxe, "zesDevicePciGetProperties");
    return 0;
}

static int
xetopo_init(void)
{
    ze_result_t err = (*zeInitPtr)(ZE_INIT_FLAG_GPU_ONLY);
    if (err != ZE_RESULT_SUCCESS)
    {
        fprintf(stderr, "Intel Xe API cannot be found and initialized (zeInit failed): ");
        return ze_error(err);
    }
    return 0;
}


static int
xetopo_get_numaNode(zes_pci_address_t pcidata)
{
    char fname[1024];
    char buff[100];
    int ret = snprintf(fname, 1023, "/sys/bus/pci/devices/%04x:%02x:%02x.%01x/numa_node", pcidata.domain,
                                                                                          pcidata.bus,
                                                                                          pcidata.device,
                                                                                          pcidata.function);
    if (ret > 0)
    {
        fname[ret] = '\0';
        FILE* fp = fopen(fname, "r");
        if (fp)
        {
            ret = fread(buff, sizeof(char), 99, fp);
            int numa_node = atoi(buff);
            fclose(fp);
            return numa_node;
        }
    }
    return -1;
}

static int xetopo_gpu_cleanup(int idx, int err)
{
/*    if (accel_topology_xe_initialized)*/
/*    {*/
        for (int j = idx; j >= 0; j--)
        {
            AccelDevXe* dev = &xeGpuTopology.devices[j];
/*            if (dev->name)*/
/*            {*/
/*                free(dev->name);*/
/*                dev->name = NULL;*/
/*            }*/
            if (dev->short_name)
            {
                free(dev->short_name);
                dev->short_name = NULL;
            }
            if (dev->cmdQueues)
            {
                free(dev->cmdQueues);
                dev->cmdQueues = NULL;
                dev->numCmdQueues = 0;
            }
            if (dev->memories)
            {
                free(dev->memories);
                dev->memories = NULL;
                dev->numMemories = 0;
            }
            if (dev->caches)
            {
                free(dev->caches);
                dev->caches = NULL;
                dev->numCaches = 0;
            }
        }
/*    }*/
    return err;
}


int
topology_xe_init()
{
    uint32_t i = 0, d = 0;
    int ret = 0;
    int xe_version = 0;
    ze_api_version_t ze_version;
    uint32_t num_devices = 0;

    if (accel_topology_xe_initialized)
    {
        return EXIT_SUCCESS;
    }
    ret = xetopo_link_libraries();
    if (ret != 0)
    {
        ERROR_PLAIN_PRINT(Cannot open Xe library to fill GPU topology);
        return EXIT_FAILURE;
    }
    ret = xetopo_init();
    if (ret != 0)
    {
        ERROR_PLAIN_PRINT(Cannot initialize Xe library to fill GPU topology);
        return EXIT_FAILURE;
    }
    uint32_t driverCount = 0;
    XE_CALL((*zeDriverGetPtr)(&driverCount, NULL), return -1);
    XEDEBUG_PRINT(DEBUGLEV_DEVELOP, Allocating %ld bytes for all %d drivers (tmp), driverCount * sizeof(ze_driver_handle_t), driverCount);

    ze_driver_handle_t* allDrivers = malloc(driverCount * sizeof(ze_driver_handle_t));
    if (!allDrivers)
    {
        ERROR_PLAIN_PRINT(Cannot allocate data for Xe drivers);
        return -ENOMEM;
    }
    XE_CALL((*zeDriverGetPtr)(&driverCount, allDrivers), free(allDrivers); return -1;);
    for(i = 0; i < driverCount; ++i)
    {
        uint32_t deviceCount = 0;
        XE_CALL((*zeDeviceGetPtr)(allDrivers[i], &deviceCount, NULL), free(allDrivers); return -1;);
        num_devices += deviceCount;
        uint64_t msize = (deviceCount+1) * sizeof(ze_device_handle_t);
        XEDEBUG_PRINT(DEBUGLEV_DEVELOP, Allocating %ld bytes for all %d devices of dev %d (tmp), msize, deviceCount, i);
        ze_device_handle_t* allDevices = malloc(msize);
        if (!allDevices)
        {
            ERROR_PLAIN_PRINT(Cannot allocate data for Xe devices);
            free(allDrivers);
            return -ENOMEM;
        }
        memset(allDevices, 0, msize);
        XE_CALL((*zeDeviceGetPtr)(allDrivers[i], &deviceCount, allDevices), free(allDrivers); free(allDevices); return -1;);
        for(d = 0; d < deviceCount; ++d)
        {
            ze_device_properties_t device_properties = {};
            XE_CALL((*zeDeviceGetPropertiesPtr)(allDevices[d], &device_properties), free(allDrivers); free(allDevices); return -1;);
            if(ZE_DEVICE_TYPE_GPU == device_properties.type)
            {
                num_devices++;
            }
        }
        XEDEBUG_PRINT(DEBUGLEV_DEVELOP, Freeing %ld bytes for all %d devices of dev %d (tmp), deviceCount * sizeof(ze_device_handle_t), deviceCount, i);
        free(allDevices);
        allDevices = NULL;
    }
    XEDEBUG_PRINT(DEBUGLEV_INFO, System provides %d drivers with %d GPU devices, driverCount, num_devices);
    if (num_devices == 0)
    {
        xeGpuTopology.devices = NULL;
        xeGpuTopology.numDevices = 0;
        free(allDrivers);
        return EXIT_SUCCESS;
    }
    XEDEBUG_PRINT(DEBUGLEV_DEVELOP, Allocating %ld bytes for all %d drivers and devices, num_devices * sizeof(AccelDevXe), num_devices);
    xeGpuTopology.devices = malloc(num_devices * sizeof(AccelDevXe));
    if (!xeGpuTopology.devices)
    {
        free(allDrivers);
        xeGpuTopology.numDevices = 0;
        return -ENOMEM;
    }
    int idx = 0;
    for(i = 0; i < driverCount && idx < num_devices; ++i)
    {
        uint32_t deviceCount = 0;
        XE_CALL((*zeDeviceGetPtr)(allDrivers[i], &deviceCount, NULL), free(allDrivers); return -1;);
        XEDEBUG_PRINT(DEBUGLEV_DEVELOP, Allocating %ld bytes for all %d devices of drv %d (tmp), deviceCount * sizeof(ze_device_handle_t), deviceCount, i);
        ze_device_handle_t* allDevices = malloc(deviceCount * sizeof(ze_device_handle_t));
        if (!allDevices)
        {
            ERROR_PLAIN_PRINT(Cannot allocate data for Xe devices);
            free(allDrivers);
            return -ENOMEM;
        }
        XE_CALL((*zeDeviceGetPtr)(allDrivers[i], &deviceCount, allDevices), free(allDrivers); free(allDevices); return -1;);
        for(d = 0; d < deviceCount && idx < num_devices; ++d)
        {
            // WORKAROUND: If we use a single ze_device_properties_t variable, the call of
            // zeDeviceGetProperties smashes the stack. It seems to be only required if we
            // access some specific struct attributes since we don't need it when looking
            // up the number of devices (line 331)
            ze_device_properties_t device_properties[2];
            ze_device_compute_properties_t compute_properties;
            ze_device_module_properties_t module_properties;
            

            XE_CALL((*zeDeviceGetPropertiesPtr)(allDevices[d], device_properties), free(allDrivers); free(allDevices); return -1;);
            if(ZE_DEVICE_TYPE_GPU == device_properties[0].type)
            {
                AccelDevXe *dev = &xeGpuTopology.devices[idx];
                XEDEBUG_PRINT(DEBUGLEV_DEVELOP, Drv %d Dev %d Idx %d, i, d, idx);
                dev->cmdQueues = NULL;
                dev->memories = NULL;
                ret = snprintf(dev->name, ZE_MAX_DEVICE_NAME-1, "%s", device_properties[0].name);
                if (ret >= 0) dev->name[ret] = '\0';
                dev->short_name = NULL;
                dev->drvid = i;
                dev->devid = d;
                dev->vendorId = device_properties[0].vendorId;
                dev->deviceId = device_properties[0].deviceId;
                dev->coreClockRate = device_properties[0].coreClockRate;
                dev->maxMemAllocSize = device_properties[0].maxMemAllocSize;
                dev->maxHardwareContexts = device_properties[0].maxHardwareContexts;
                dev->maxCommandQueuePriority = device_properties[0].maxCommandQueuePriority;
                dev->numThreadsPerEU = device_properties[0].numThreadsPerEU;
                dev->physicalEUSimdWidth = device_properties[0].physicalEUSimdWidth;
                dev->numEUsPerSubslice = device_properties[0].numEUsPerSubslice;
                dev->numSubslicesPerSlice = device_properties[0].numSubslicesPerSlice;
                dev->numSlices = device_properties[0].numSlices;
                dev->timerResolution = device_properties[0].timerResolution;
                dev->timestampValidBits = device_properties[0].timestampValidBits;
                dev->kernelTimestampValidBits = device_properties[0].kernelTimestampValidBits;

                // compute properties
                XE_CALL((*zeDeviceGetComputePropertiesPtr)(allDevices[d], &compute_properties), free(allDrivers); free(allDevices); return -1;);
                dev->maxTotalGroupSize = compute_properties.maxTotalGroupSize;
                dev->maxGroupSizeX = compute_properties.maxGroupSizeX;
                dev->maxGroupSizeY = compute_properties.maxGroupSizeY;
                dev->maxGroupSizeZ = compute_properties.maxGroupSizeZ;
                dev->maxGroupCountX = compute_properties.maxGroupCountX;
                dev->maxGroupCountY = compute_properties.maxGroupCountY;
                dev->maxGroupCountZ = compute_properties.maxGroupCountZ;
                dev->maxSharedLocalMemory = compute_properties.maxSharedLocalMemory;
                dev->numSubGroupSizes = compute_properties.numSubGroupSizes;
                for (int j = 0; j < MIN(dev->numSubGroupSizes, ZE_SUBGROUPSIZE_COUNT); j++)
                {
                    dev->subGroupSizes[j] = compute_properties.subGroupSizes[j];
                }
                XEDEBUG_PRINT(DEBUGLEV_DEVELOP, Drv %d Dev %d Idx %d END COMP PROPS, i, d, idx);

                dev->cmdQueues = NULL;
                dev->numCmdQueues = 0;
                if (zeDeviceGetCommandQueueGroupPropertiesPtr) {
                    uint32_t num_queues = 0;
                    XE_CALL((*zeDeviceGetCommandQueueGroupPropertiesPtr)(allDevices[d], &num_queues, NULL), free(allDrivers); free(allDevices); return -1;);
                    XEDEBUG_PRINT(DEBUGLEV_DEVELOP, Allocating %ld bytes for %d cmdqueues (tmp), num_queues * sizeof(ze_command_queue_group_properties_t), num_queues);
                    ze_command_queue_group_properties_t* tmp = malloc(num_queues * sizeof(ze_command_queue_group_properties_t));
                    if (tmp)
                    {
                        XE_CALL((*zeDeviceGetCommandQueueGroupPropertiesPtr)(allDevices[d], &num_queues, tmp), free(allDrivers); free(allDevices); return -1;);
                        XEDEBUG_PRINT(DEBUGLEV_DEVELOP, Drv %d Dev %d Idx %d ALLOC %ld FOR %d CMDQUEUE, i, d, xeGpuTopology.numDevices, num_queues * sizeof(AccelDevXeCmdQueue), num_queues);
                        dev->cmdQueues = malloc(num_queues * sizeof(AccelDevXeCmdQueue));
                        if (dev->cmdQueues)
                        {
                            for (int j = 0; j < num_queues; j++)
                            {
                                dev->cmdQueues[j].numQueues = tmp[j].numQueues;
                                dev->cmdQueues[j].maxMemoryFillPatternSize = tmp[j].maxMemoryFillPatternSize;
                                for (int k = 0; k < MAX_CMD_QUEUE_FLAGS; k++)
                                {
                                    if (tmp[j].flags & (1ULL<<k))
                                    {
                                        dev->cmdQueues[j].flags |= AccelDevXeCmdQueueMask(k);
                                    }
                                }
                            }
                            dev->numCmdQueues = num_queues;
                        }
                        XEDEBUG_PRINT(DEBUGLEV_DEVELOP, Freeing %ld bytes for %d cmdqueues (tmp), num_queues * sizeof(ze_command_queue_group_properties_t), num_queues);
                        free(tmp);
                    }
                }

                dev->memories = NULL;
                dev->numMemories = 0;
                if (zeDeviceGetMemoryPropertiesPtr)
                {
                    uint32_t num_mems = 0;
                    XE_CALL((*zeDeviceGetMemoryPropertiesPtr)(allDevices[d], &num_mems, NULL), free(allDrivers); free(allDevices); return -1;);
                    ze_device_memory_properties_t* tmp = malloc(num_mems * sizeof(ze_device_memory_properties_t));
                    if (tmp)
                    {
                        XE_CALL((*zeDeviceGetMemoryPropertiesPtr)(allDevices[d], &num_mems, tmp), free(allDrivers); free(allDevices); return -1;);
                        dev->memories = malloc(num_mems * sizeof(AccelDevXeMemory));
                        if (dev->memories)
                        {
                            for (int j = 0; j < num_mems; j++)
                            {
                                dev->memories[j].maxClockRate = tmp[j].maxClockRate;
                                dev->memories[j].maxBusWidth = tmp[j].maxBusWidth;
                                dev->memories[j].totalSize = tmp[j].totalSize;
                                ret = snprintf(dev->memories[j].name, ZE_MAX_DEVICE_NAME, "%s", tmp[j].name);
                                if (ret >= 0) dev->memories[j].name[ret] = '\0';
                                for (int k = 0; k < MAX_MEMORY_FLAGS; k++)
                                {
                                    if (tmp[j].flags & (1ULL<<k))
                                    {
                                        dev->cmdQueues[j].flags |= AccelDevXeMemoryMask(k);
                                    }
                                }
                            }
                            dev->numMemories = num_mems;
                        }
                        free(tmp);
                    }
                }
                
                dev->caches = NULL;
                dev->numCaches = 0;
                if (zeDeviceGetCachePropertiesPtr)
                {
                    uint32_t num_caches = 0;
                    XE_CALL((*zeDeviceGetCachePropertiesPtr)(allDevices[d], &num_caches, NULL), free(allDrivers); free(allDevices); return -1;);
                    ze_device_cache_properties_t* tmp = malloc(num_caches * sizeof(ze_device_cache_properties_t));
                    if (tmp)
                    {
                        XE_CALL((*zeDeviceGetCachePropertiesPtr)(allDevices[d], &num_caches, tmp), free(allDrivers); free(allDevices); return -1;);
                        dev->caches = malloc(num_caches * sizeof(AccelDevXeCache));
                        if (dev->caches)
                        {
                            for (int j = 0; j < num_caches; j++)
                            {
                                dev->caches[j].id = j;
                                dev->caches[j].cacheSize = tmp[j].cacheSize;
                                for (int k = 0; k < MAX_CACHE_FLAGS; k++)
                                {
                                    if (tmp[j].flags & (1ULL<<k))
                                    {
                                        dev->caches[j].flags |= AccelDevXeCacheMask(k);
                                    }
                                }
                            }
                            dev->numCaches = num_caches;
                        }
                        free(tmp);
                    }
                }
                if (zesDevicePciGetPropertiesPtr)
                {
                    zes_pci_properties_t pciprop;
                    XE_CALL((*zesDevicePciGetPropertiesPtr)(allDevices[d], &pciprop), free(allDrivers); free(allDevices); return -1;);
                    dev->pciBus = pciprop.address.bus;
                    dev->pciDom = pciprop.address.domain;
                    dev->pciDev = pciprop.address.device;
                    dev->numaNode = xetopo_get_numaNode(pciprop.address);
                }


/*                xeGpuTopology.devices[xeGpuTopology.numDevices].uuid = device_properties[0].uuid;*/
/*                //ret = xetopo_fillDevice(allDrivers[i], allDevices[d], &xeGpuTopology.devices[xeGpuTopology.numDevices]);*/
/*                if (ret < 0)*/
/*                {*/
/*                    return xetopo_gpu_cleanup(xeGpuTopology.numDevices, ret);*/
/*                }*/
                XEDEBUG_PRINT(DEBUGLEV_DEVELOP, Drv %d Dev %d Idx %d INC IDX, i, d, idx);
                idx++;
                if (idx == num_devices)
                {
                    XEDEBUG_PRINT(DEBUGLEV_DEVELOP, Drv %d Dev %d Idx %d BREAK, i, d, idx);
                    break;
                }
            }
        }
        XEDEBUG_PRINT(DEBUGLEV_DEVELOP, Freeing %ld bytes for all %d devices of drv %d (tmp), deviceCount * sizeof(ze_device_handle_t), deviceCount, i);
        free(allDevices);
        if (idx == num_devices)
        {
            XEDEBUG_PRINT(DEBUGLEV_DEVELOP, Drv %d Dev %d Idx %d BREAK, i, deviceCount, idx);
            break;
        }
    }
    XEDEBUG_PRINT(DEBUGLEV_DEVELOP, Freeing %ld bytes for all %d drivers (tmp), driverCount * sizeof(ze_driver_handle_t), driverCount);
    free(allDrivers);
    xeGpuTopology.numDevices = idx;
    accel_topology_xe_initialized = 1;
    return EXIT_SUCCESS;
}

void
topology_xe_finalize(void)
{
    if (accel_topology_xe_initialized)
    {
        int ret = xetopo_gpu_cleanup(xeGpuTopology.numDevices-1, 0);
        if (topo_dl_libxe)
        {
            dlclose(topo_dl_libxe);
            topo_dl_libxe = NULL;
        }
    }
}

AccelXeTopology_t
get_xeGpuTopology(void)
{
    if (accel_topology_xe_initialized)
    {
        return &xeGpuTopology;
    }
}

void xemon_setVerbosity(int verbosity)
{
    if (verbosity >= 0 && verbosity <= DEBUGLEV_DEVELOP)
    {
        likwid_xemon_verbosity = verbosity;
    }
}


#endif /* LIKWID_WITH_XEMON */
