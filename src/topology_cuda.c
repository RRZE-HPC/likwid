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
            ERROR_PRINT(Function %s failed with error %d, #call, _status); \
            handleerror;                                                \
        }                                                               \
    } while (0)

#define CUAPIWEAK __attribute__( ( weak ) )
#define DECLARECUFUNC(funcname, funcsig) CUresult CUAPIWEAK funcname funcsig;  CUresult( *funcname##TopoPtr ) funcsig;


#define CUDA_CALL( call, handleerror )                                \
    do {                                                                \
        cudaError_t _status = (call);                                   \
        if (_status != cudaSuccess) {                                   \
            ERROR_PRINT(Function %s failed with error %d, #call, _status); \
            handleerror;                                                \
        }                                                               \
    } while (0)

#define CUDAAPIWEAK __attribute__( ( weak ) )
#define DECLARECUDAFUNC(funcname, funcsig) cudaError_t CUDAAPIWEAK funcname funcsig;  cudaError_t( *funcname##TopoPtr ) funcsig;

/* Copy from PAPI's cuda component (BSD License)
 * @author  Asim YarKhan yarkhan@icl.utk.edu (updated in 2017 to support CUDA metrics)
 * @author  Asim YarKhan yarkhan@icl.utk.edu (updated in 2015 for multiple CUDA contexts/devices)
 * @author  Heike Jagode (First version, in collaboration with Robert Dietrich, TU Dresden) jagode@icl.utk.edu
 */
void (*_dl_non_dynamic_init) (void) __attribute__ ((weak));

static void *topo_dl_libcuda = NULL;
static void *topo_dl_libcudart = NULL;
static int topology_cuda_initialized = 0;
CudaTopology cudaTopology = {0, NULL};

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

DECLARECUDAFUNC(cudaDriverGetVersion, (int*));
DECLARECUDAFUNC(cudaRuntimeGetVersion, (int*))

static int
cuda_topo_link_libraries(void)
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
        DEBUG_PRINT(DEBUGLEV_INFO, CUDA library libcuda.so not found);
        return -1;
    }
    topo_dl_libcudart = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if (!topo_dl_libcudart)
    {
        DEBUG_PRINT(DEBUGLEV_INFO, CUDA library libcudart.so not found);
        return -1;
    }
    cuDeviceGetTopoPtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceGet");
    cuDeviceGetCountTopoPtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceGetCount");
    cuDeviceGetNameTopoPtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceGetName");
    cuInitTopoPtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuInit");
    cuDeviceComputeCapabilityTopoPtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceComputeCapability");
    cuDeviceGetAttributeTopoPtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceGetAttribute");
    cuDeviceGetPropertiesTopoPtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceGetProperties");
    cuDeviceTotalMemTopoPtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceTotalMem");
    
    cudaDriverGetVersionTopoPtr = DLSYM_AND_CHECK(topo_dl_libcudart, "cudaDriverGetVersion");
    cudaRuntimeGetVersionTopoPtr = DLSYM_AND_CHECK(topo_dl_libcudart, "cudaRuntimeGetVersion");
    
#if CUDA_VERSION >= 10000
    cuDeviceTotalMem_v2TopoPtr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceTotalMem_v2");
#endif
    
    return 0;
}

static int
cuda_topo_init(void)
{
    CUresult cuErr = (*cuInitTopoPtr)(0);
    if (cuErr != CUDA_SUCCESS)
    {
        DEBUG_PRINT(DEBUGLEV_INFO, CUDA cannot be found and initialized (cuInit failed): %d, cuErr);
        return -ENODEV;
    }
    return 0;
}

static int
cuda_topo_get_numDevices(void)
{
    int count = 0;
    CUresult cuErr = (*cuDeviceGetCountTopoPtr)(&count);
    if (cuErr == CUDA_SUCCESS)
        return count;

    int ret = cuda_topo_init();
    if (ret < 0)
        return ret;

    cuErr = (*cuDeviceGetCountTopoPtr)(&count);
    if (cuErr == CUDA_SUCCESS)
        return count;
    DEBUG_PRINT(DEBUGLEV_INFO, uDeviceGetCount failed even though cuda_topo_init succeeded: %d, cuErr);
    return -ELIBACC;
}

static int
cuda_topo_get_numNode(int pci_bus, int pci_dev, int pci_domain)
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

static int topology_cuda_cleanup(int idx, int err)
{
    for (int j = idx; j >= 0; j--)
    {
        free(cudaTopology.devices[j].name);
        free(cudaTopology.devices[j].short_name);
    }
    return err;
}

int
topology_cuda_init()
{
    int i = 0;
    int ret = 0;
    int cuda_version = 0;
    int cudart_version = 0;
    if (topology_cuda_initialized)
    {
        return EXIT_SUCCESS;
    }
    ret = cuda_topo_link_libraries();
    if (ret != 0)
    {
        return EXIT_FAILURE;
    }
    int num_devs = cuda_topo_get_numDevices();
    if (num_devs < 0)
    {
        return -ENODEV;
    }
    CUDA_CALL((*cudaDriverGetVersionTopoPtr)(&cuda_version), ret = -1; goto topology_gpu_init_error;);
    CUDA_CALL((*cudaRuntimeGetVersionTopoPtr)(&cudart_version), ret = -1; goto topology_gpu_init_error;);
    if (num_devs > 0)
    {
        cudaTopology.devices = malloc(num_devs * sizeof(CudaDevice));
        if (!cudaTopology.devices)
        {
            return -ENOMEM;
        }
        for (i = 0; i < num_devs; i++)
        {
            CUdevice dev;
            cudaTopology.devices[i].name = NULL;
            cudaTopology.devices[i].short_name = NULL;
            CU_CALL((*cuDeviceGetTopoPtr)(&dev, i), ret = -ENODEV; goto topology_gpu_init_error;);
            size_t s = 0;
#if CUDA_VERSION >= 10000
            if (cuda_version >= 10000 && cudart_version >= 10000)
            {
                CU_CALL((*cuDeviceTotalMem_v2TopoPtr)(&s, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
                if (s == 0)
                {
                    CU_CALL((*cuDeviceTotalMemTopoPtr)(&s, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
                }
            }
            else
            {
                CU_CALL((*cuDeviceTotalMemTopoPtr)(&s, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            }
#else
            CU_CALL((*cuDeviceTotalMemTopoPtr)(&s, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
#endif
            const size_t NAME_LONG_MAX = 1024;
            const size_t NAME_SHORT_MAX = 64;
            cudaTopology.devices[i].mem = (unsigned long long)s;
            cudaTopology.devices[i].name = calloc(NAME_LONG_MAX, sizeof(char));
            if (!cudaTopology.devices[i].name)
            {
                ERROR_PRINT(Cannot allocate space for name of GPU %d, i);
                ret = -ENOMEM;
                goto topology_gpu_init_error;
            }
            CU_CALL((*cuDeviceGetNameTopoPtr)(cudaTopology.devices[i].name, NAME_LONG_MAX-1, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            cudaTopology.devices[i].devid = i;
            cudaTopology.devices[i].short_name = calloc(NAME_SHORT_MAX, sizeof(char));
            if (!cudaTopology.devices[i].short_name)
            {
                ERROR_PRINT(Cannot allocate space for short name of GPU %d, i);
                ret = -ENOMEM;
                goto topology_gpu_init_error;
            }

            CU_CALL((*cuDeviceComputeCapabilityTopoPtr)(&cudaTopology.devices[i].ccapMajor, &cudaTopology.devices[i].ccapMinor, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            if (cudaTopology.devices[i].ccapMajor < 7)
            {
                ret = snprintf(cudaTopology.devices[i].short_name, NAME_SHORT_MAX, "nvidia_gpu_cc_lt_7");
            }
            else if (cudaTopology.devices[i].ccapMajor >= 7)
            {
                ret = snprintf(cudaTopology.devices[i].short_name, NAME_SHORT_MAX, "nvidia_gpu_cc_ge_7");
            }
            CUdevprop props;
            CU_CALL((*cuDeviceGetPropertiesTopoPtr)(&props, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            cudaTopology.devices[i].maxThreadsPerBlock = props.maxThreadsPerBlock;
            cudaTopology.devices[i].maxThreadsDim[0] = props.maxThreadsDim[0];
            cudaTopology.devices[i].maxThreadsDim[1] = props.maxThreadsDim[1];
            cudaTopology.devices[i].maxThreadsDim[2] = props.maxThreadsDim[2];
            cudaTopology.devices[i].maxGridSize[0] = props.maxGridSize[0];
            cudaTopology.devices[i].maxGridSize[1] = props.maxGridSize[1];
            cudaTopology.devices[i].maxGridSize[2] = props.maxGridSize[2];
            cudaTopology.devices[i].sharedMemPerBlock = props.sharedMemPerBlock;
            cudaTopology.devices[i].totalConstantMemory = props.totalConstantMemory;
            cudaTopology.devices[i].simdWidth = props.SIMDWidth;
            cudaTopology.devices[i].memPitch = props.memPitch;
            cudaTopology.devices[i].clockRatekHz = props.clockRate;
            cudaTopology.devices[i].textureAlign = props.textureAlign;
            CU_CALL((*cuDeviceGetAttributeTopoPtr)(&cudaTopology.devices[i].l2Size, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            CU_CALL((*cuDeviceGetAttributeTopoPtr)(&cudaTopology.devices[i].memClockRatekHz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            CU_CALL((*cuDeviceGetAttributeTopoPtr)(&cudaTopology.devices[i].memClockRatekHz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            CU_CALL((*cuDeviceGetAttributeTopoPtr)(&cudaTopology.devices[i].pciBus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            CU_CALL((*cuDeviceGetAttributeTopoPtr)(&cudaTopology.devices[i].pciDev, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            CU_CALL((*cuDeviceGetAttributeTopoPtr)(&cudaTopology.devices[i].pciDom, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            CU_CALL((*cuDeviceGetAttributeTopoPtr)(&cudaTopology.devices[i].maxBlockRegs, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            CU_CALL((*cuDeviceGetAttributeTopoPtr)(&cudaTopology.devices[i].numMultiProcs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            CU_CALL((*cuDeviceGetAttributeTopoPtr)(&cudaTopology.devices[i].maxThreadPerMultiProc, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            CU_CALL((*cuDeviceGetAttributeTopoPtr)(&cudaTopology.devices[i].memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            CU_CALL((*cuDeviceGetAttributeTopoPtr)(&cudaTopology.devices[i].unifiedAddrSpace, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            CU_CALL((*cuDeviceGetAttributeTopoPtr)(&cudaTopology.devices[i].ecc, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            CU_CALL((*cuDeviceGetAttributeTopoPtr)(&cudaTopology.devices[i].asyncEngines, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            CU_CALL((*cuDeviceGetAttributeTopoPtr)(&cudaTopology.devices[i].mapHostMem, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            CU_CALL((*cuDeviceGetAttributeTopoPtr)(&cudaTopology.devices[i].integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev), ret = -ENOMEM; goto topology_gpu_init_error;);
            CU_CALL((*cuDeviceGetAttributeTopoPtr)(&cudaTopology.devices[i].surfaceAlign, CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, dev), ret = -ENOMEM; goto topology_gpu_init_error;);

            cudaTopology.devices[i].numaNode = cuda_topo_get_numNode(cudaTopology.devices[i].pciBus, cudaTopology.devices[i].pciDev, cudaTopology.devices[i].pciDom);
        }
        cudaTopology.numDevices = num_devs;
    }
    topology_cuda_initialized = 1;
    return EXIT_SUCCESS;
topology_gpu_init_error:
    for (int j = 0; j < i; j++)
    {
        topology_cuda_cleanup(i-1, 0);
    }
    return ret;
}


void
topology_cuda_finalize(void)
{
    if (topology_cuda_initialized)
    {
        int ret = topology_cuda_cleanup(cudaTopology.numDevices-1, 0);
    }
}

CudaTopology_t
get_cudaTopology(void)
{
    if (topology_cuda_initialized)
    {
        return &cudaTopology;
    }
    ERROR_PRINT(Cannot get CUDA topology before initialization);
    return NULL;
}

#endif /* LIKWID_WITH_NVMON */
