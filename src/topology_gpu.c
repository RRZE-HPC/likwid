/*
 * =======================================================================================
 *
 *      Filename:  topology_gpu.c
 *
 *      Description:  Topology module for GPUs
 *
 *      Version:   5.0
 *      Released:  10.11.2019
 *
 *      Author:   Thomas Gruber (tg), thomas.roehl@googlemail.com
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

#include <cupti.h>
#include <dlfcn.h>
#include <cuda.h>

#include <error.h>
#include <likwid.h>

#define CU_CALL( call, handleerror )                                    \
    do {                                                                \
        CUresult _status = (call);                                      \
        if (_status != CUDA_SUCCESS) {                                  \
            fprintf(stderr, "Error: function %s failed with error %d.\n", #call, _status); \
            handleerror;                                                \
        }                                                               \
    } while (0)

#define CUAPIWEAK __attribute__( ( weak ) )
#define DECLARECUFUNC(funcname, funcsig) CUresult CUAPIWEAK funcname funcsig;  CUresult( *funcname##Ptr ) funcsig;

/* Copy from PAPI's cuda component (BSD License)
 * @author  Asim YarKhan yarkhan@icl.utk.edu (updated in 2017 to support CUDA metrics)
 * @author  Asim YarKhan yarkhan@icl.utk.edu (updated in 2015 for multiple CUDA contexts/devices)
 * @author  Heike Jagode (First version, in collaboration with Robert Dietrich, TU Dresden) jagode@icl.utk.edu
 */
void (*_dl_non_dynamic_init) (void) __attribute__ ((weak));

static void *topo_dl_libcuda = NULL;
static int topology_gpu_initialized = 0;
GpuTopology gpuTopology = {0, NULL};

#ifdef LIKWID_WITH_NVMON

DECLARECUFUNC(cuDeviceGet, (CUdevice *, int));
DECLARECUFUNC(cuDeviceGetCount, (int *));
DECLARECUFUNC(cuDeviceGetName, (char *, int, CUdevice));
DECLARECUFUNC(cuInit, (unsigned int));
DECLARECUFUNC(cuDeviceComputeCapability, (int*, int*, CUdevice));
DECLARECUFUNC(cuDeviceGetAttribute, (int*, CUdevice_attribute, CUdevice));
DECLARECUFUNC(cuDeviceGetProperties, (CUdevprop* prop, CUdevice));
DECLARECUFUNC(cuDeviceTotalMem, (size_t*, CUdevice));
DECLARECUFUNC(cuDeviceTotalMem_v2, (size_t*, CUdevice));


static int
topo_link_libraries(void)
{
#define DLSYM_AND_CHECK( dllib, name ) dlsym( dllib, name ); if ( dlerror() != NULL ) { return -1; }

    /* Attempt to guess if we were statically linked to libc, if so bail */
    if(_dl_non_dynamic_init != NULL) {
        return -1;
    }
    /* Need to link in the cuda libraries, if not found disable the component */
    topo_dl_libcuda = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!topo_dl_libcuda)
    {
        fprintf(stderr, "CUDA library libcuda.so not found.\n");
        return -1;
    }
    cuDeviceGetPtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceGet");
    cuDeviceGetCountPtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceGetCount");
    cuDeviceGetNamePtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceGetName");
    cuInitPtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuInit");
    cuDeviceComputeCapabilityPtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceComputeCapability");
    cuDeviceGetAttributePtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceGetAttribute");
    cuDeviceGetPropertiesPtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceGetProperties");
    cuDeviceTotalMemPtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceTotalMem");
    cuDeviceTotalMem_v2Ptr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceTotalMem_v2");
    return 0;
}

static int
topo_init_cuda(void)
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
topo_get_numDevices(void)
{
    CUresult cuErr;
    int count = 0;
    cuErr = (*cuDeviceGetCountPtr)(&count);
    if(cuErr == CUDA_ERROR_NOT_INITIALIZED)
    {
        int ret = topo_init_cuda();
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

static int topology_gpu_cleanup(int idx, int err)
{
    for (int j = idx; j >= 0; j--)
    {
        free(gpuTopology.devices[j].name);
    }
    return err;
}

int
topology_gpu_init()
{
    int ret = 0;
    if (topology_gpu_initialized)
    {
        return EXIT_SUCCESS;
    }
    ret = topo_link_libraries();
    if (ret != 0)
    {
        ERROR_PLAIN_PRINT(Cannot open CUDA library to fill GPU topology);
        return EXIT_FAILURE;
    }
    int num_devs = topo_get_numDevices();
    if (num_devs < 0)
    {
        ERROR_PLAIN_PRINT(Cannot get number of devices from CUDA library);
        return EXIT_FAILURE;
    }
    if (num_devs > 0)
    {
        gpuTopology.devices = malloc(num_devs * sizeof(GpuDevice));
        if (!gpuTopology.devices)
        {
            return -ENOMEM;
        }
        for (int i = 0; i < num_devs; i++)
        {
            CUdevice dev;
            CU_CALL((*cuDeviceGetPtr)(&dev, i), return topology_gpu_cleanup(i-1, -ENODEV));
            size_t s = 0;
#if __CUDA_API_VERSION >= 10000
            CU_CALL((*cuDeviceTotalMem_v2Ptr)(&s, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            if (s == 0)
            {
                CU_CALL((*cuDeviceTotalMemPtr)(&s, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            }
#else
            CU_CALL((*cuDeviceTotalMemPtr)(&s, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
#endif
            gpuTopology.devices[i].mem = (unsigned long long)s;
            gpuTopology.devices[i].name = malloc(1024 * sizeof(char));
            if (!gpuTopology.devices[i].name)
            {
                ERROR_PRINT(Cannot allocate space for name of GPU %d, i);
                return topology_gpu_cleanup(i-1, -ENOMEM);
            }
            CU_CALL((*cuDeviceGetNamePtr)(gpuTopology.devices[i].name, 1023, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            gpuTopology.devices[i].name[1024] = '\0';
            gpuTopology.devices[i].devid = i;

            CU_CALL((*cuDeviceComputeCapabilityPtr)(&gpuTopology.devices[i].ccapMajor, &gpuTopology.devices[i].ccapMinor, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            CUdevprop props;
            CU_CALL((*cuDeviceGetPropertiesPtr)(&props, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            gpuTopology.devices[i].maxThreadsPerBlock = props.maxThreadsPerBlock;
            gpuTopology.devices[i].maxThreadsDim[0] = props.maxThreadsDim[0];
            gpuTopology.devices[i].maxThreadsDim[1] = props.maxThreadsDim[1];
            gpuTopology.devices[i].maxThreadsDim[2] = props.maxThreadsDim[2];
            gpuTopology.devices[i].maxGridSize[0] = props.maxGridSize[0];
            gpuTopology.devices[i].maxGridSize[1] = props.maxGridSize[1];
            gpuTopology.devices[i].maxGridSize[2] = props.maxGridSize[2];
            gpuTopology.devices[i].sharedMemPerBlock = props.sharedMemPerBlock;
            gpuTopology.devices[i].totalConstantMemory = props.totalConstantMemory;
            gpuTopology.devices[i].simdWidth = props.SIMDWidth;
            gpuTopology.devices[i].memPitch = props.memPitch;
            gpuTopology.devices[i].clockRatekHz = props.clockRate;
            gpuTopology.devices[i].textureAlign = props.textureAlign;
            CU_CALL((*cuDeviceGetAttributePtr)(&gpuTopology.devices[i].l2Size, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            CU_CALL((*cuDeviceGetAttributePtr)(&gpuTopology.devices[i].memClockRatekHz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            CU_CALL((*cuDeviceGetAttributePtr)(&gpuTopology.devices[i].memClockRatekHz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            CU_CALL((*cuDeviceGetAttributePtr)(&gpuTopology.devices[i].pciBus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            CU_CALL((*cuDeviceGetAttributePtr)(&gpuTopology.devices[i].pciDev, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            CU_CALL((*cuDeviceGetAttributePtr)(&gpuTopology.devices[i].pciDom, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            CU_CALL((*cuDeviceGetAttributePtr)(&gpuTopology.devices[i].maxBlockRegs, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            CU_CALL((*cuDeviceGetAttributePtr)(&gpuTopology.devices[i].numMultiProcs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            CU_CALL((*cuDeviceGetAttributePtr)(&gpuTopology.devices[i].maxThreadPerMultiProc, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            CU_CALL((*cuDeviceGetAttributePtr)(&gpuTopology.devices[i].memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            CU_CALL((*cuDeviceGetAttributePtr)(&gpuTopology.devices[i].unifiedAddrSpace, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            CU_CALL((*cuDeviceGetAttributePtr)(&gpuTopology.devices[i].ecc, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            CU_CALL((*cuDeviceGetAttributePtr)(&gpuTopology.devices[i].asyncEngines, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            CU_CALL((*cuDeviceGetAttributePtr)(&gpuTopology.devices[i].mapHostMem, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            CU_CALL((*cuDeviceGetAttributePtr)(&gpuTopology.devices[i].integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev), return topology_gpu_cleanup(i-1, -ENOMEM));
            CU_CALL((*cuDeviceGetAttributePtr)(&gpuTopology.devices[i].surfaceAlign, CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, dev), return topology_gpu_cleanup(i-1, -ENOMEM));

            gpuTopology.devices[i].numaNode = topo_get_numNode(gpuTopology.devices[i].pciBus, gpuTopology.devices[i].pciDev, gpuTopology.devices[i].pciDom);
        }
        gpuTopology.numDevices = num_devs;
    }
    topology_gpu_initialized = 1;
    return EXIT_SUCCESS;
}


void
topology_gpu_finalize(void)
{
    if (topology_gpu_initialized)
    {
        int ret = topology_gpu_cleanup(gpuTopology.numDevices-1, 0);
    }
}

GpuTopology_t
get_gpuTopology(void)
{
    if (topology_gpu_initialized)
    {
        return &gpuTopology;
    }
}

#endif /* LIKWID_WITH_NVMON */
