/*
 * =======================================================================================
 *
 *      Filename:  topology_gpu.c
 *
 *      Description:  Topology module for GPUs
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
#include <dlfcn.h>

#include <error.h>
#include <likwid.h>


// Variables
static void *topo_dl_libhip = NULL;
static int topo_gpu_initialized = 0;
static GpuTopology_rocm topo_gpuTopology = {0, NULL};


// HIP function declarations
#define HIPWEAK __attribute__( ( weak ) )
#define DECLAREHIPFUNC(funcname, funcsig) hipError_t HIPWEAK funcname funcsig;  hipError_t ( *funcname##TopoPtr ) funcsig;

DECLAREHIPFUNC(hipGetDeviceCount, (int *count))
DECLAREHIPFUNC(hipGetDeviceProperties, (hipDeviceProp_t *prop, int deviceId))


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
    hipGetDeviceCountTopoPtr = DLSYM_AND_CHECK(topo_dl_libhip, "hipGetDeviceCount");
    hipGetDevicePropertiesTopoPtr = DLSYM_AND_CHECK(topo_dl_libhip, "hipGetDeviceProperties");

    return 0;
}

static int
topo_get_numDevices(void)
{
    int count = 0;

    hipError_t err = (*hipGetDeviceCountTopoPtr)(&count);
    if (err == hipErrorNoDevice)
    {
        return 0;
    }

    return count;
}

static int
topo_get_numNode(int pci_bus, int pci_dev, int pci_domain)
{
    char fname[1024];
    char buff[100];
    int ret = snprintf(fname, 1023, "/sys/bus/pci/devices/0000:%02x:%02x.%1x/numa_node", pci_bus, pci_dev, pci_domain);
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

static void
topo_gpu_cleanup(int numDevices)
{
#define FREE_IF_NOT_NULL(var) if ( var ) { free( var ); }

    for (int i = 0; i < numDevices; i++)
    {
        GpuDevice_rocm *device = &topo_gpuTopology.devices[i];

        FREE_IF_NOT_NULL(device->name)
    }

    FREE_IF_NOT_NULL(topo_gpuTopology.devices)
}

static int
topo_gpu_init(GpuDevice_rocm *device, int deviceId)
{
    hipError_t err;
    hipDeviceProp_t props;

    device->devid = deviceId;
    device->name = NULL;
    device->short_name = "amd_gpu";

    // Get HIP device properties
    err = (*hipGetDevicePropertiesTopoPtr)(&props, deviceId);
    if (err == hipErrorInvalidDevice)
    {
        ERROR_PRINT(GPU %d is not a valid device, deviceId);
        return -ENODEV;
    }
    if (err != hipSuccess)
    {
        ERROR_PRINT(Failed to retreive properties for GPU %d, deviceId);
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
    device->numaNode = topo_get_numNode(device->pciBus, device->pciDev, device->pciDom);

    // Copy Name
    device->name = malloc(256 * sizeof(char));
    if (!device->name)
    {
        ERROR_PRINT(Cannot allocate space for name of GPU %d, deviceId);
        return -ENOMEM;
    }
    strncpy(device->name, props.name, 256);
    device->name[255] = '\0';

    return 0;
}

int
topology_gpu_init_rocm()
{
    int ret = 0;

    // Do not initialize twice
    if (topo_gpu_initialized)
    {
        return EXIT_SUCCESS;
    }

    // Link required functions from dynamic libraries
    ret = topo_link_libraries();
    if (ret != 0)
    {
        ERROR_PLAIN_PRINT(Cannot open ROCm HIP library to fill GPU topology);
        return EXIT_FAILURE;
    }

    // Get number of devices to initialize
    int num_devs = topo_get_numDevices();
    if (num_devs < 0)
    {
        ERROR_PLAIN_PRINT(Cannot get number of devices from ROCm HIP library);
        return EXIT_FAILURE;
    }

    // Allocate memory for device information
    if (num_devs > 0)
    {
        topo_gpuTopology.devices = malloc(num_devs * sizeof(GpuDevice_rocm));
        if (!topo_gpuTopology.devices)
        {
            return -ENOMEM;
        }
    }

    // Initialize devices
    for (int i = 0; i < num_devs; i++)
    {
        ret = topo_gpu_init(&topo_gpuTopology.devices[i], i);
        if (ret != 0)
        {
            topo_gpu_cleanup(i+1);
            return ret;
        }
    }

    // Finished
    topo_gpuTopology.numDevices = num_devs;
    topo_gpu_initialized = 1;
    return EXIT_SUCCESS;
}

void
topology_gpu_finalize_rocm(void)
{
    if (topo_gpu_initialized)
    {
        topo_gpu_cleanup(topo_gpuTopology.numDevices);
    }
}

GpuTopology_rocm_t
get_gpuTopology_rocm(void)
{
    if (topo_gpu_initialized)
    {
        return &topo_gpuTopology;
    }
}

#endif /* LIKWID_WITH_ROCMON */
