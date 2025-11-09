/*
 * =======================================================================================
 *
 *      Filename:  topology_cuda.c
 *
 *      Description:  Topology module for Nvidia GPUs
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

#define CU_CALL(handleerror, func, ...)                            \
    do {                                                \
        CUresult s_ = (*func##_ptr)(__VA_ARGS__);    \
        if (s_ != CUDA_SUCCESS) {                        \
            const char *errstr = NULL;\
            cuGetErrorString_ptr(s_, &errstr);\
            ERROR_PRINT("Error: function %s failed with error: '%s' (CUresult=%d).", #func, errstr, s_);   \
            handleerror;\
        }                                               \
    } while (0)

#define DECLARECUFUNC(funcname, ...) CUresult __attribute__((weak)) funcname(__VA_ARGS__);  static CUresult (*funcname##_ptr)(__VA_ARGS__);


#define CUDA_CALL(handleerror, func, ...)                            \
    do {                                                \
        cudaError_t s_ = (*func##_ptr)(__VA_ARGS__);    \
        if (s_ != cudaSuccess) {                        \
            ERROR_PRINT("Error: function %s failed with error: '%s' (cudaError_t=%d).", #func, cudaGetErrorString_ptr(s_), s_);   \
            handleerror;\
        }                                               \
    } while (0)

#define DECLARECUDAFUNC(funcname, ...) cudaError_t __attribute__((weak)) funcname(__VA_ARGS__);  static cudaError_t (*funcname##_ptr)(__VA_ARGS__);

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

DECLARECUFUNC(cuDeviceGet, CUdevice *, int);
DECLARECUFUNC(cuDeviceGetCount, int *);
DECLARECUFUNC(cuDeviceGetName, char *, int, CUdevice);
DECLARECUFUNC(cuGetErrorString, CUresult, const char **);
DECLARECUFUNC(cuInit, unsigned int);
DECLARECUFUNC(cuDeviceComputeCapability, int*, int*, CUdevice);
DECLARECUFUNC(cuDeviceGetAttribute, int*, CUdevice_attribute, CUdevice);
DECLARECUFUNC(cuDeviceGetProperties, CUdevprop* prop, CUdevice);
DECLARECUFUNC(cuDeviceTotalMem, size_t*, CUdevice);
DECLARECUFUNC(cuDeviceTotalMem_v2, size_t*, CUdevice);

DECLARECUDAFUNC(cudaDriverGetVersion, int*);
DECLARECUDAFUNC(cudaRuntimeGetVersion, int*)
const char * __attribute__((weak)) cudaGetErrorString(cudaError_t);
static const char *(*cudaGetErrorString_ptr)(cudaError_t);

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
        DEBUG_PRINT(DEBUGLEV_INFO, "CUDA library libcuda.so not found");
        return -1;
    }
    topo_dl_libcudart = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if (!topo_dl_libcudart)
    {
        DEBUG_PRINT(DEBUGLEV_INFO, "CUDA library libcudart.so not found");
        return -1;
    }
    cuDeviceGet_ptr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceGet");
    cuDeviceGetCount_ptr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceGetCount");
    cuDeviceGetName_ptr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceGetName");
    cuInit_ptr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuInit");
    cuDeviceComputeCapability_ptr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceComputeCapability");
    cuDeviceGetAttribute_ptr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceGetAttribute");
    cuDeviceGetProperties_ptr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceGetProperties");
    cuDeviceTotalMem_ptr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceTotalMem");
    cuGetErrorString_ptr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuGetErrorString");
    
    cudaDriverGetVersion_ptr = DLSYM_AND_CHECK(topo_dl_libcudart, "cudaDriverGetVersion");
    cudaRuntimeGetVersion_ptr = DLSYM_AND_CHECK(topo_dl_libcudart, "cudaRuntimeGetVersion");
    cudaGetErrorString_ptr = DLSYM_AND_CHECK(topo_dl_libcudart, "cudaGetErrorString");
    
#if CUDA_VERSION >= 10000
    cuDeviceTotalMem_v2_ptr = DLSYM_AND_CHECK(topo_dl_libcuda, "cuDeviceTotalMem_v2");
#endif
    
    return 0;
}

static int
cuda_topo_init(void)
{
    CUresult cuErr = cuInit_ptr(0);
    if (cuErr != CUDA_SUCCESS)
    {
        const char *cuErrString;
        cuGetErrorString_ptr(cuErr, &cuErrString);
        DEBUG_PRINT(DEBUGLEV_INFO, "CUDA cannot be found and initialized (cuInit failed): %d (%s)", cuErr, cuErrString);
        return -ENODEV;
    }
    return 0;
}

static int
cuda_topo_get_numDevices(void)
{
    int count = 0;
    CUresult cuErr = cuDeviceGetCount_ptr(&count);
    if (cuErr == CUDA_SUCCESS)
        return count;

    int ret = cuda_topo_init();
    if (ret < 0)
        return ret;

    cuErr = cuDeviceGetCount_ptr(&count);
    if (cuErr == CUDA_SUCCESS)
        return count;
    DEBUG_PRINT(DEBUGLEV_INFO, "uDeviceGetCount failed even though cuda_topo_init succeeded: %d", cuErr);
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
        return ret;
    }
    int num_devs = cuda_topo_get_numDevices();
    if (num_devs < 0)
    {
        return -ENODEV;
    }
    CUDA_CALL(ret = -1; goto topology_gpu_init_error, cudaDriverGetVersion,&cuda_version);
    CUDA_CALL(ret = -1; goto topology_gpu_init_error, cudaRuntimeGetVersion,&cudart_version);
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
            CU_CALL(ret = -ENODEV; goto topology_gpu_init_error, cuDeviceGet, &dev, i);
            size_t s = 0;
#if CUDA_VERSION >= 10000
            if (cuda_version >= 10000 && cudart_version >= 10000)
            {
                CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceTotalMem_v2, &s, dev);
                if (s == 0)
                {
                    CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceTotalMem, &s, dev);
                }
            }
            else
            {
                CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceTotalMem, &s, dev);
            }
#else
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceTotalMem, &s, dev);
#endif
            const size_t NAME_LONG_MAX = 1024;
            const size_t NAME_SHORT_MAX = 64;
            cudaTopology.devices[i].mem = (unsigned long long)s;
            cudaTopology.devices[i].name = calloc(NAME_LONG_MAX, sizeof(char));
            if (!cudaTopology.devices[i].name)
            {
                ERROR_PRINT("Cannot allocate space for name of GPU %d", i);
                ret = -ENOMEM;
                goto topology_gpu_init_error;
            }
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetName, cudaTopology.devices[i].name, NAME_LONG_MAX-1, dev);
            cudaTopology.devices[i].devid = i;
            cudaTopology.devices[i].short_name = calloc(NAME_SHORT_MAX, sizeof(char));
            if (!cudaTopology.devices[i].short_name)
            {
                ERROR_PRINT("Cannot allocate space for short name of GPU %d", i);
                ret = -ENOMEM;
                goto topology_gpu_init_error;
            }

            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceComputeCapability, &cudaTopology.devices[i].ccapMajor, &cudaTopology.devices[i].ccapMinor, dev);
            if (cudaTopology.devices[i].ccapMajor < 7)
            {
                ret = snprintf(cudaTopology.devices[i].short_name, NAME_SHORT_MAX, "nvidia_gpu_cc_lt_7");
            }
            else if (cudaTopology.devices[i].ccapMajor >= 7)
            {
                ret = snprintf(cudaTopology.devices[i].short_name, NAME_SHORT_MAX, "nvidia_gpu_cc_ge_7");
            }
            CUdevprop props;
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetProperties, &props, dev);
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
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetAttribute, &cudaTopology.devices[i].l2Size, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetAttribute, &cudaTopology.devices[i].memClockRatekHz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetAttribute, &cudaTopology.devices[i].memClockRatekHz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetAttribute, &cudaTopology.devices[i].pciBus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev);
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetAttribute, &cudaTopology.devices[i].pciDev, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev);
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetAttribute, &cudaTopology.devices[i].pciDom, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev);
            // TODO: Get PCI function through nvmlDeviceGetPciInfo_v3, nvmlPciInfo_t->function
            cudaTopology.devices[i].pciFunc = 0;
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetAttribute, &cudaTopology.devices[i].maxBlockRegs, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev);
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetAttribute, &cudaTopology.devices[i].numMultiProcs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetAttribute, &cudaTopology.devices[i].maxThreadPerMultiProc, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetAttribute, &cudaTopology.devices[i].memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetAttribute, &cudaTopology.devices[i].unifiedAddrSpace, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev);
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetAttribute, &cudaTopology.devices[i].ecc, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, dev);
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetAttribute, &cudaTopology.devices[i].asyncEngines, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, dev);
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetAttribute, &cudaTopology.devices[i].mapHostMem, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev);
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetAttribute, &cudaTopology.devices[i].integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev);
            CU_CALL(ret = -ENOMEM; goto topology_gpu_init_error, cuDeviceGetAttribute, &cudaTopology.devices[i].surfaceAlign, CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, dev);

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
        topology_cuda_cleanup(cudaTopology.numDevices-1, 0);
}

CudaTopology_t
get_cudaTopology(void)
{
    if (topology_cuda_initialized)
    {
        return &cudaTopology;
    }
    ERROR_PRINT("Cannot get CUDA topology before initialization");
    return NULL;
}

#endif /* LIKWID_WITH_NVMON */
