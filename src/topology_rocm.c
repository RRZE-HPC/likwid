/*
 * =======================================================================================
 *
 *      Filename:  topology_rocm.c
 *
 *      Description:  Topology module for AMD GPUs
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tg), thomas.roehl@googlemail.com
 *                Various people at HPE
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
 #ifdef LIKWID_WITH_ROCMON

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include <sys/types.h>
#include <errno.h>
#include <ctype.h>
#include <assert.h>

#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#include <dlfcn.h>

#include <error.h>
#include <likwid.h>


// Variables
static void *topo_dl_libhip = NULL;
static int topo_rocm_initialized = 0;
static RocmTopology _rocmTopology = {0, NULL};


// HIP function declarations
#define HIPWEAK __attribute__( ( weak ) )
#define DECLAREHIPFUNC(funcname, funcsig) hipError_t HIPWEAK funcname funcsig;  hipError_t ( *funcname##TopoPtr ) funcsig;

DECLAREHIPFUNC(hipGetProcAddress, (const char *symbol, void **pfn, int hipVersion, uint64_t flags, hipDriverProcAddressQueryResult *symbolStatus))

typedef hipError_t (*hipGetDeviceProperties_t)(hipDeviceProp_t *prop, int deviceId);
hipGetDeviceProperties_t hipGetDevicePropertiesFunc;

typedef hipError_t (*hipGetDeviceCount_t)(int *count);
hipGetDeviceCount_t hipGetDeviceCountFunc;

static int
topo_link_libraries(void)
{
#define DLSYM_AND_CHECK( dllib, name ) dlsym( dllib, name ); if ( dlerror() != NULL ) { return -1; }

    /* Need to link in the ROCm HIP libraries */
    topo_dl_libhip = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!topo_dl_libhip)
    {
        fprintf(stderr, "ROCm HIP library libamdrocm64.so not found.\n");
        return -1;
    }

    // Link HIP functions
    hipGetProcAddressTopoPtr = DLSYM_AND_CHECK(topo_dl_libhip, "hipGetProcAddress");

    hipError_t res;
    int hipVersion = HIP_VERSION; // Use the HIP version defined in hip_version.h
    uint64_t flags = 0; // No special flags
    hipDriverProcAddressQueryResult symbolStatus;


    res = (*hipGetProcAddressTopoPtr)("hipGetDeviceProperties", (void**)&hipGetDevicePropertiesFunc, hipVersion, flags, &symbolStatus);
    if (res != hipSuccess) {
        ERROR_PRINT("Failed to obtaian hipGetDeviceProperties address");
        return EXIT_FAILURE;
    }
    res = (*hipGetProcAddressTopoPtr)("hipGetDeviceCount", (void**)&hipGetDeviceCountFunc, hipVersion, flags, &symbolStatus);
    if (res != hipSuccess) {
        ERROR_PRINT("Failed to obtaian hipGetDeviceCount address");
        return EXIT_FAILURE;
    }

    return 0;
}

static int
topo_get_numDevices(void)
{
    int count = 0;

    hipError_t err = hipGetDeviceCountFunc(&count);
    if (err == hipErrorNoDevice)
    {
        return 0;
    }

    return count;
}

static int
topo_get_numNode(int pci_domain, int pci_bus, int pci_dev)
{
    char fname[1024];
    char buff[100];
    int ret = snprintf(fname, 1023, "/sys/bus/pci/devices/%04x:%02x:%02x.%1x/numa_node", pci_domain, pci_bus, pci_dev, 0);
    if (ret > 0)
    {
        fname[ret] = '\0';
        printf("Reading %s for NUMA node\n", fname);
        FILE* fp = fopen(fname, "r");
        if (fp)
        {
            ret = fread(buff, sizeof(char), 99, fp);
            int numa_node = atoi(buff);
            printf("Got node %d\n", numa_node);
            fclose(fp);
            return numa_node;
        }
    }
    return -1;
}

static void
topo_rocm_cleanup(int numDevices)
{
#define FREE_IF_NOT_NULL(var) if ( var ) { free( var ); }

    for (int i = 0; i < numDevices; i++)
    {
        RocmDevice *device = &_rocmTopology.devices[i];

        FREE_IF_NOT_NULL(device->name)
    }

    FREE_IF_NOT_NULL(_rocmTopology.devices)
}

static int
topo_gpu_init(RocmDevice *device, int deviceId)
{
    hipError_t err;
    hipDeviceProp_t props;

    device->devid = deviceId;
    device->name = NULL;
    device->short_name = "amd_gpu";

    // Get HIP device properties
    err = hipGetDevicePropertiesFunc(&props, deviceId);
    if (err == hipErrorInvalidDevice)
    {
        ERROR_PRINT("GPU %d is not a valid device", deviceId);
        return -ENODEV;
    }
    if (err != hipSuccess)
    {
        ERROR_PRINT("Failed to retreive properties for GPU %d", deviceId);
        return EXIT_FAILURE;
    }

    // Copy info from props
    device->mem = props.totalGlobalMem;
    device->ccapMajor = props.major;
    device->ccapMinor = props.minor;
    device->maxThreadsPerBlock = props.maxThreadsPerBlock;
    device->maxThreadsDim[0] = props.maxThreadsDim[0];
    device->maxThreadsDim[1] = props.maxThreadsDim[1];
    device->maxThreadsDim[2] = props.maxThreadsDim[2];
    device->maxGridSize[0] = props.maxGridSize[0];
    device->maxGridSize[1] = props.maxGridSize[1];
    device->maxGridSize[2] = props.maxGridSize[2];
    device->sharedMemPerBlock = props.sharedMemPerBlock;
    device->totalConstantMemory = props.totalConstMem;
    device->simdWidth = props.warpSize;
    device->memPitch = props.memPitch;
    device->regsPerBlock = props.regsPerBlock;
    device->clockRatekHz = props.clockRate;
    device->textureAlign = props.textureAlignment;
    device->l2Size = props.l2CacheSize;
    device->memClockRatekHz = props.memoryClockRate;
    device->pciBus = props.pciBusID;
    device->pciDev = props.pciDeviceID;
    device->pciDom = props.pciDomainID;
    device->numMultiProcs = props.multiProcessorCount;
    device->maxThreadPerMultiProc = props.maxThreadsPerMultiProcessor;
    device->memBusWidth = props.memoryBusWidth;
    device->ecc = props.ECCEnabled;
    device->mapHostMem = props.canMapHostMemory;
    device->integrated = props.integrated;
    device->numaNode = -1;
    if ((device->pciDom != 0) && (device->pciBus != 0) && (device->pciDev != 0))
    {
        device->numaNode = topo_get_numNode(device->pciDom, device->pciBus, device->pciDev);
    }

    // Copy Name
    device->name = malloc(256 * sizeof(char));
    if (!device->name)
    {
        ERROR_PRINT("Cannot allocate space for name of GPU %d", deviceId);
        return -ENOMEM;
    }
    strncpy(device->name, props.name, 256);
    device->name[255] = '\0';

    return 0;
}

int
topology_rocm_init()
{
    int ret = 0;

    // Do not initialize twice
    if (topo_rocm_initialized)
    {
        return EXIT_SUCCESS;
    }

    // Link required functions from dynamic libraries
    ret = topo_link_libraries();
    if (ret != 0)
    {
        ERROR_PRINT("Cannot open ROCm HIP library to fill GPU topology");
        return EXIT_FAILURE;
    }

    // Get number of devices to initialize
    int num_devs = topo_get_numDevices();
    if (num_devs < 0)
    {
        ERROR_PRINT("Cannot get number of devices from ROCm HIP library");
        return EXIT_FAILURE;
    }

    // Allocate memory for device information
    if (num_devs > 0)
    {
        _rocmTopology.devices = malloc(num_devs * sizeof(RocmDevice));
        if (!_rocmTopology.devices)
        {
            return -ENOMEM;
        }
    }

    // Initialize devices
    for (int i = 0; i < num_devs; i++)
    {
        ret = topo_gpu_init(&_rocmTopology.devices[i], i);
        if (ret != 0)
        {
            topo_rocm_cleanup(i+1);
            return ret;
        }
    }

    // Finished
    _rocmTopology.numDevices = num_devs;
    topo_rocm_initialized = 1;
    return EXIT_SUCCESS;
}

void
topology_rocm_finalize(void)
{
    if (topo_rocm_initialized)
    {
        topo_rocm_cleanup(_rocmTopology.numDevices);
    }
    topo_rocm_initialized = 0;
}

RocmTopology_t
get_rocmTopology(void)
{
    if (topo_rocm_initialized)
    {
        return &_rocmTopology;
    }
}

#endif /* LIKWID_WITH_ROCMON */
