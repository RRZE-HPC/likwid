/*
 * =======================================================================================
 *
 *      Filename:  nvmon_perfworks.h
 *
 *      Description:  Header File of nvmon module (PerfWorks backend).
 *
 *      Version:   5.2
 *      Released:  17.6.2021
 *
 *      Author:   Thomas Gruber (tg), thomas.gruber@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 NHR@FAU, University Erlangen-Nuremberg
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
#ifndef LIKWID_NVMON_PERFWORKS_H
#define LIKWID_NVMON_PERFWORKS_H




#include <cuda.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <cuda_runtime_api.h>

#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>


#if defined(CUDART_VERSION) && CUDART_VERSION >= 10000

static void *dl_perfworks_libcuda = NULL;
static void *dl_libhost = NULL;
static void *dl_cupti = NULL;
static void *dl_perfworks_libcudart = NULL;

#define LIKWID_CU_CALL( call, handleerror )                                    \
    do {                                                                \
        CUresult _status = (call);                                      \
        if (_status != CUDA_SUCCESS) {                                  \
            fprintf(stderr, "Error: function %s failed with error %d.\n", #call, _status); \
            handleerror;                                                \
        }                                                               \
    } while (0)

#define LIKWID_NVPW_API_CALL(call, handleerror) \
    do                                                                                   \
    {                          \
        NVPA_Status _status = (call);                                                          \
        if(_status != NVPA_STATUS_SUCCESS)                                                \
        {                                                                                \
            fprintf(stderr, "Error: function %s failed with error %d.\n", #call, _status); \
            handleerror;                                                               \
        }                                                                                \
    } while(0)

#define LIKWID_CUPTI_API_CALL(call, handleerror)                                            \
    do {                                                                           \
        CUptiResult _status = (call);                                         \
        if (_status != CUPTI_SUCCESS) {                                            \
            fprintf(stderr, "Error: function %s failed with error %d.\n", #call, _status);                    \
            handleerror;                                                             \
        }                                                                          \
    } while (0)

#define LIKWID_CUDA_API_CALL( call, handleerror )                                \
    do {                                                                \
        cudaError_t _status = (call);                                   \
        if (_status != cudaSuccess) {                                   \
            fprintf(stderr, "Error: function %s failed with error %d.\n", #call, _status); \
            handleerror;                                                \
        }                                                               \
    } while (0)


/* This definitions are used for CUDA 10.1 */
#if defined(CUDART_VERSION) && CUDART_VERSION < 11000
typedef struct CUpti_Profiler_GetCounterAvailability_Params
{
    size_t structSize;
    void* pPriv;
    CUcontext ctx;
    size_t counterAvailabilityImageSize;
    uint8_t* pCounterAvailabilityImage;
} CUpti_Profiler_GetCounterAvailability_Params;
#define CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE sizeof(CUpti_Profiler_GetCounterAvailability_Params)

CUptiResult cuptiProfilerGetCounterAvailability(CUpti_Profiler_GetCounterAvailability_Params* params)
{
    return CUPTI_SUCCESS;
}

typedef struct {
    size_t structSize;
    void* pPriv;
    NVPA_RawMetricsConfig* pRawMetricsConfig;
    uint8_t* pCounterAvailabilityImage;
} NVPW_RawMetricsConfig_SetCounterAvailability_Params;
#define NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE sizeof(NVPW_RawMetricsConfig_SetCounterAvailability_Params)

NVPA_Status NVPW_RawMetricsConfig_SetCounterAvailability(NVPW_RawMetricsConfig_SetCounterAvailability_Params* params)
{
    return NVPA_STATUS_SUCCESS;
}
#endif /* End of definitions for CUDA 10.1 */

#define CUpti_Device_GetChipName_Params_STRUCT_SIZE10 16
#define CUpti_Device_GetChipName_Params_STRUCT_SIZE11 32

#define CUpti_Profiler_SetConfig_Params_STRUCT_SIZE10 56
#define CUpti_Profiler_SetConfig_Params_STRUCT_SIZE11 58

#define CUpti_Profiler_EndPass_Params_STRUCT_SIZE10 24
#define CUpti_Profiler_EndPass_Params_STRUCT_SIZE11 41

#define CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE10 24
#define CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE11 40

#ifndef DECLARE_CUFUNC
#define CUAPIWEAK __attribute__( ( weak ) )
#define DECLARE_CUFUNC(funcname, funcsig) CUresult CUAPIWEAK funcname funcsig;  CUresult( *funcname##Ptr ) funcsig;
#endif

DECLARE_CUFUNC(cuCtxGetCurrent, (CUcontext *));
DECLARE_CUFUNC(cuCtxSetCurrent, (CUcontext));
DECLARE_CUFUNC(cuCtxDestroy, (CUcontext));
DECLARE_CUFUNC(cuDeviceGet, (CUdevice *, int));
DECLARE_CUFUNC(cuDeviceGetCount, (int *));
DECLARE_CUFUNC(cuDeviceGetName, (char *, int, CUdevice));
DECLARE_CUFUNC(cuInit, (unsigned int));
DECLARE_CUFUNC(cuCtxPopCurrent, (CUcontext * pctx));
DECLARE_CUFUNC(cuCtxPushCurrent, (CUcontext pctx));
DECLARE_CUFUNC(cuCtxSynchronize, ());
DECLARE_CUFUNC(cuDeviceGetAttribute, (int *, CUdevice_attribute, CUdevice));
DECLARE_CUFUNC(cuCtxCreate, (CUcontext *, unsigned int, CUdevice));
DECLARE_CUFUNC(cuDevicePrimaryCtxRetain, (CUcontext *, CUdevice));

#ifndef DECLARE_CUDAFUNC
#define CUDAAPIWEAK __attribute__( ( weak ) )
#define DECLARE_CUDAFUNC(funcname, funcsig) cudaError_t CUDAAPIWEAK funcname funcsig;  cudaError_t( *funcname##Ptr ) funcsig;
#endif
DECLARE_CUDAFUNC(cudaGetDevice, (int *));
DECLARE_CUDAFUNC(cudaSetDevice, (int));
DECLARE_CUDAFUNC(cudaFree, (void *));
DECLARE_CUDAFUNC(cudaDriverGetVersion, (int *));
DECLARE_CUDAFUNC(cudaRuntimeGetVersion, (int *));


#ifndef DECLARE_NVPWFUNC
#define NVPWAPIWEAK __attribute__( ( weak ) )
#define DECLARE_NVPWFUNC(fname, fsig) NVPA_Status NVPWAPIWEAK fname fsig; NVPA_Status( *fname##Ptr ) fsig;
#endif

DECLARE_NVPWFUNC(NVPW_GetSupportedChipNames, (NVPW_GetSupportedChipNames_Params* params));
DECLARE_NVPWFUNC(NVPW_CUDA_MetricsContext_Create, (NVPW_CUDA_MetricsContext_Create_Params* params));
DECLARE_NVPWFUNC(NVPW_MetricsContext_Destroy, (NVPW_MetricsContext_Destroy_Params * params));
DECLARE_NVPWFUNC(NVPW_MetricsContext_GetMetricNames_Begin, (NVPW_MetricsContext_GetMetricNames_Begin_Params* params));
DECLARE_NVPWFUNC(NVPW_MetricsContext_GetMetricNames_End, (NVPW_MetricsContext_GetMetricNames_End_Params* params));
DECLARE_NVPWFUNC(NVPW_InitializeHost, (NVPW_InitializeHost_Params* params));
DECLARE_NVPWFUNC(NVPW_MetricsContext_GetMetricProperties_Begin, (NVPW_MetricsContext_GetMetricProperties_Begin_Params* p));
DECLARE_NVPWFUNC(NVPW_MetricsContext_GetMetricProperties_End, (NVPW_MetricsContext_GetMetricProperties_End_Params* p));
DECLARE_NVPWFUNC(NVPA_RawMetricsConfig_Create, (const NVPA_RawMetricsConfigOptions*, NVPA_RawMetricsConfig**));
DECLARE_NVPWFUNC(NVPW_RawMetricsConfig_Destroy, (NVPW_RawMetricsConfig_Destroy_Params* params));
DECLARE_NVPWFUNC(NVPW_RawMetricsConfig_BeginPassGroup, (NVPW_RawMetricsConfig_BeginPassGroup_Params* params));
DECLARE_NVPWFUNC(NVPW_RawMetricsConfig_EndPassGroup, (NVPW_RawMetricsConfig_EndPassGroup_Params* params));
DECLARE_NVPWFUNC(NVPW_RawMetricsConfig_AddMetrics, (NVPW_RawMetricsConfig_AddMetrics_Params* params));
DECLARE_NVPWFUNC(NVPW_RawMetricsConfig_GenerateConfigImage, (NVPW_RawMetricsConfig_GenerateConfigImage_Params* params));
DECLARE_NVPWFUNC(NVPW_RawMetricsConfig_GetConfigImage, (NVPW_RawMetricsConfig_GetConfigImage_Params* params));
DECLARE_NVPWFUNC(NVPW_CounterDataBuilder_Create, (NVPW_CounterDataBuilder_Create_Params* params));
DECLARE_NVPWFUNC(NVPW_CounterDataBuilder_Destroy, (NVPW_CounterDataBuilder_Destroy_Params* params));
DECLARE_NVPWFUNC(NVPW_CounterDataBuilder_AddMetrics, (NVPW_CounterDataBuilder_AddMetrics_Params* params));
DECLARE_NVPWFUNC(NVPW_CounterDataBuilder_GetCounterDataPrefix, (NVPW_CounterDataBuilder_GetCounterDataPrefix_Params* params));
DECLARE_NVPWFUNC(NVPW_CounterData_GetNumRanges, (NVPW_CounterData_GetNumRanges_Params* params));
DECLARE_NVPWFUNC(NVPW_Profiler_CounterData_GetRangeDescriptions, (NVPW_Profiler_CounterData_GetRangeDescriptions_Params* params));
DECLARE_NVPWFUNC(NVPW_MetricsContext_SetCounterData, (NVPW_MetricsContext_SetCounterData_Params* params));
DECLARE_NVPWFUNC(NVPW_MetricsContext_EvaluateToGpuValues, (NVPW_MetricsContext_EvaluateToGpuValues_Params* params));
DECLARE_NVPWFUNC(NVPW_RawMetricsConfig_GetNumPasses, (NVPW_RawMetricsConfig_GetNumPasses_Params* params));
DECLARE_NVPWFUNC(NVPW_RawMetricsConfig_SetCounterAvailability, (NVPW_RawMetricsConfig_SetCounterAvailability_Params* params));

#ifndef DECLARE_CUPTIFUNC
#define CUPTIAPIWEAK __attribute__( ( weak ) )
#define DECLARE_CUPTIFUNC(funcname, funcsig) CUptiResult CUPTIAPIWEAK funcname funcsig;  CUptiResult( *funcname##Ptr ) funcsig;
#endif
DECLARE_CUPTIFUNC(cuptiDeviceGetChipName, (CUpti_Device_GetChipName_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerInitialize, (CUpti_Profiler_Initialize_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerDeInitialize, (CUpti_Profiler_DeInitialize_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerCounterDataImageCalculateSize, (CUpti_Profiler_CounterDataImage_CalculateSize_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerCounterDataImageInitialize, (CUpti_Profiler_CounterDataImage_Initialize_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerCounterDataImageCalculateScratchBufferSize, (CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerCounterDataImageInitializeScratchBuffer, (CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params* params));

DECLARE_CUPTIFUNC(cuptiProfilerBeginSession, (CUpti_Profiler_BeginSession_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerSetConfig, (CUpti_Profiler_SetConfig_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerBeginPass, (CUpti_Profiler_BeginPass_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerEnableProfiling, (CUpti_Profiler_EnableProfiling_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerPushRange, (CUpti_Profiler_PushRange_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerPopRange, (CUpti_Profiler_PopRange_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerDisableProfiling, (CUpti_Profiler_DisableProfiling_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerEndPass, (CUpti_Profiler_EndPass_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerFlushCounterData, (CUpti_Profiler_FlushCounterData_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerUnsetConfig, (CUpti_Profiler_UnsetConfig_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerEndSession, (CUpti_Profiler_EndSession_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerGetCounterAvailability, (CUpti_Profiler_GetCounterAvailability_Params* params));


#ifndef DLSYM_AND_CHECK
#define DLSYM_AND_CHECK( dllib, name ) dlsym( dllib, name ); if ( dlerror() != NULL ) { return -1; }
#endif


static int cuptiProfiler_initialized = 0;
static int cuda_runtime_version = 0;
static int cuda_version = 0;





static int
link_perfworks_libraries(void)
{
    /* Attempt to guess if we were statically linked to libc, if so bail */
    if(_dl_non_dynamic_init != NULL) {
        return -1;
    }
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Init PerfWorks Libaries);
    dl_perfworks_libcuda = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_perfworks_libcuda || dlerror() != NULL)
    {
        fprintf(stderr, "CUDA library libcuda.so not found.");
        return -1;
    }
    cuCtxGetCurrentPtr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuCtxGetCurrent");
    cuCtxSetCurrentPtr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuCtxSetCurrent");
    cuDeviceGetPtr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuDeviceGet");
    cuDeviceGetCountPtr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuDeviceGetCount");
    cuDeviceGetNamePtr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuDeviceGetName");
    cuInitPtr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuInit");
    cuCtxPopCurrentPtr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuCtxPopCurrent");
    cuCtxPushCurrentPtr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuCtxPushCurrent");
    cuCtxSynchronizePtr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuCtxSynchronize");
    cuCtxDestroyPtr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuCtxDestroy");
    cuDeviceGetAttributePtr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuDeviceGetAttribute");
    cuCtxCreatePtr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuCtxCreate");
    cuDevicePrimaryCtxRetainPtr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuDevicePrimaryCtxRetain");

    dl_perfworks_libcudart = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if (!dl_perfworks_libcudart)
    {
        fprintf(stderr, "CUDA runtime library libcudart.so not found.");
        return -1;
    }
    cudaGetDevicePtr = DLSYM_AND_CHECK(dl_perfworks_libcudart, "cudaGetDevice");
    cudaSetDevicePtr = DLSYM_AND_CHECK(dl_perfworks_libcudart, "cudaSetDevice");
    cudaFreePtr = DLSYM_AND_CHECK(dl_perfworks_libcudart, "cudaFree");
    cudaDriverGetVersionPtr = DLSYM_AND_CHECK(dl_perfworks_libcudart, "cudaDriverGetVersion");
    cudaRuntimeGetVersionPtr = DLSYM_AND_CHECK(dl_perfworks_libcudart, "cudaRuntimeGetVersion");

    dl_libhost = dlopen("libnvperf_host.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if (!dl_libhost || dlerror() != NULL)
    {
        fprintf(stderr, "CUpti library libnvperf_host.so not found.");
        return -1;
    }
    NVPW_GetSupportedChipNamesPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_GetSupportedChipNames");
    NVPW_CUDA_MetricsContext_CreatePtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_CUDA_MetricsContext_Create");
    NVPW_MetricsContext_DestroyPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_MetricsContext_Destroy");
    NVPW_MetricsContext_GetMetricNames_BeginPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_MetricsContext_GetMetricNames_Begin");
    NVPW_MetricsContext_GetMetricNames_EndPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_MetricsContext_GetMetricNames_End");
    NVPW_InitializeHostPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_InitializeHost");
    NVPW_MetricsContext_GetMetricProperties_BeginPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_MetricsContext_GetMetricProperties_Begin");
    NVPW_MetricsContext_GetMetricProperties_EndPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_MetricsContext_GetMetricProperties_End");

    NVPA_RawMetricsConfig_CreatePtr = DLSYM_AND_CHECK(dl_libhost, "NVPA_RawMetricsConfig_Create");
    NVPW_RawMetricsConfig_DestroyPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_RawMetricsConfig_Destroy");
    NVPW_RawMetricsConfig_BeginPassGroupPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_RawMetricsConfig_BeginPassGroup");
    NVPW_RawMetricsConfig_EndPassGroupPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_RawMetricsConfig_EndPassGroup")
    NVPW_RawMetricsConfig_AddMetricsPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_RawMetricsConfig_AddMetrics");
    NVPW_RawMetricsConfig_GenerateConfigImagePtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_RawMetricsConfig_GenerateConfigImage");
    NVPW_RawMetricsConfig_GetConfigImagePtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_RawMetricsConfig_GetConfigImage");

    NVPW_CounterDataBuilder_CreatePtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_CounterDataBuilder_Create");
    NVPW_CounterDataBuilder_DestroyPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_CounterDataBuilder_Destroy");
    NVPW_CounterDataBuilder_AddMetricsPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_CounterDataBuilder_AddMetrics");
    NVPW_CounterDataBuilder_GetCounterDataPrefixPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_CounterDataBuilder_GetCounterDataPrefix");

    NVPW_CounterData_GetNumRangesPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_CounterData_GetNumRanges");
    NVPW_Profiler_CounterData_GetRangeDescriptionsPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_Profiler_CounterData_GetRangeDescriptions");
    NVPW_MetricsContext_SetCounterDataPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_MetricsContext_SetCounterData");
    NVPW_MetricsContext_EvaluateToGpuValuesPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_MetricsContext_EvaluateToGpuValues");
    NVPW_RawMetricsConfig_GetNumPassesPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_RawMetricsConfig_GetNumPasses");


    dl_cupti = dlopen("libcupti.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if (!dl_cupti || dlerror() != NULL)
    {
        fprintf(stderr, "CUpti library libcupti.so not found.");
        return -1;
    }
    cuptiProfilerInitializePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerInitialize");
    cuptiProfilerDeInitializePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerDeInitialize");
    cuptiDeviceGetChipNamePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceGetChipName");
    cuptiProfilerCounterDataImageCalculateSizePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageCalculateSize");
    cuptiProfilerCounterDataImageInitializePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageInitialize");
    cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageCalculateScratchBufferSize");
    cuptiProfilerCounterDataImageInitializeScratchBufferPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageInitializeScratchBuffer");
    cuptiProfilerBeginSessionPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerBeginSession");
    cuptiProfilerSetConfigPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerSetConfig");
    cuptiProfilerBeginPassPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerBeginPass");
    cuptiProfilerEnableProfilingPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerEnableProfiling");
    cuptiProfilerPushRangePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerPushRange");
    cuptiProfilerPopRangePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerPopRange");
    cuptiProfilerDisableProfilingPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerDisableProfiling");
    cuptiProfilerEndPassPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerEndPass");
    cuptiProfilerFlushCounterDataPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerFlushCounterData");
    cuptiProfilerUnsetConfigPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerUnsetConfig");
    cuptiProfilerEndSessionPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerEndSession");
    
    dlerror();
    int curDeviceId = -1;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Run cuInit);
    LIKWID_CU_CALL((*cuInitPtr)(0), return -EFAULT);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Run cuDeviceGetCount);
    LIKWID_CU_CALL((*cuDeviceGetCountPtr)(&curDeviceId), return -EFAULT);
    // GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Run cudaGetDevice);
    // LIKWID_CUDA_API_CALL((*cudaGetDevicePtr)(&curDeviceId), return -EFAULT);
    CUdevice dev;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Run cuDeviceGet);
    LIKWID_CU_CALL((*cuDeviceGetPtr)(&dev, 0), return -EFAULT);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Run cuDeviceGetAttribute for major CC);
    LIKWID_CU_CALL((*cuDeviceGetAttributePtr)(&curDeviceId, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), return -EFAULT);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Run cuDeviceGetAttribute for minor CC);
    LIKWID_CU_CALL((*cuDeviceGetAttributePtr)(&curDeviceId, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), return -EFAULT);

    LIKWID_CUDA_API_CALL((*cudaDriverGetVersionPtr)(&cuda_version), return -EFAULT);
    LIKWID_CUDA_API_CALL((*cudaRuntimeGetVersionPtr)(&cuda_runtime_version), return -EFAULT);

    if (cuda_version >= 11000 && cuda_runtime_version >= 11000)
    {
        cuptiProfilerGetCounterAvailabilityPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerGetCounterAvailability");
        NVPW_RawMetricsConfig_SetCounterAvailabilityPtr = DLSYM_AND_CHECK(dl_libhost, "NVPW_RawMetricsConfig_SetCounterAvailability");
    }
    else
    {
        cuptiProfilerGetCounterAvailabilityPtr = &cuptiProfilerGetCounterAvailability;
        NVPW_RawMetricsConfig_SetCounterAvailabilityPtr = &NVPW_RawMetricsConfig_SetCounterAvailability;
    }

    return 0;
}

static void
release_perfworks_libraries(void)
{
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Finalize PerfWorks Libaries);
    if (dl_perfworks_libcuda)
    {
        dlclose(dl_perfworks_libcuda);
        dl_perfworks_libcuda = NULL;
    }
    if (dl_perfworks_libcudart)
    {
        dlclose(dl_perfworks_libcudart);
        dl_perfworks_libcudart = NULL;
    }
    if (dl_libhost)
    {
        dlclose(dl_libhost);
        dl_libhost = NULL;
    }
    if (dl_cupti)
    {
        dlclose(dl_cupti);
        dl_cupti = NULL;
    }
}

static int perfworks_check_nv_context(NvmonDevice_t device, CUcontext currentContext)
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
        if (context_of_dev < 0)// && !device->context)
        {
            device->context = currentContext;
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Reuse context %ld for device %d, device->context, device->deviceId);
        }
        else
        {
            LIKWID_CUDA_API_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
            LIKWID_CUDA_API_CALL((*cudaFreePtr)(NULL), return -EFAULT);
            //LIKWID_CUDA_API_CALL((*cuCtxGetCurrentPtr)(), return -EFAULT);
            LIKWID_CUDA_API_CALL((*cuDevicePrimaryCtxRetainPtr)(&device->context, device->cuDevice), return -EFAULT);
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, New context %ld for device %d, device->context, device->deviceId);
        }
    }
    else if (device->context != currentContext)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Use context %ld for device %d, device->context, device->deviceId);
        LIKWID_CUDA_API_CALL((*cuCtxPushCurrentPtr)(device->context), return -EFAULT);
        need_pop = 1;
    }
    else
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Context %ld fits for device %d, device->context, device->deviceId);
    }
    return need_pop;
}

static int cuptiProfiler_init()
{
    if (!cuptiProfiler_initialized)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Init CUpti Profiler);
        if (dl_perfworks_libcuda == NULL ||
            dl_perfworks_libcudart == NULL ||
            dl_libhost == NULL ||
            dl_cupti == NULL)
        {
            if (link_perfworks_libraries() < 0)
                return -1;
        }
        // LIKWID_CU_CALL((*cuInitPtr)(0), return -1);
        // CUdevice dev;
        // LIKWID_CU_CALL((*cuDeviceGetPtr)(&dev, 0), return -1);
        CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
        LIKWID_CUPTI_API_CALL((*cuptiProfilerInitializePtr)(&profilerInitializeParams), return -1);
        NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
        LIKWID_NVPW_API_CALL((*NVPW_InitializeHostPtr)(&initializeHostParams), return -1);
        cuptiProfiler_initialized = 1;
    }
}

static void cuptiProfiler_finalize()
{
    if (cuptiProfiler_initialized)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Finalize CUpti Profiler);
        CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
        LIKWID_CUPTI_API_CALL((*cuptiProfilerDeInitializePtr)(&profilerDeInitializeParams), cuptiProfiler_initialized = 0; return;);
        cuptiProfiler_initialized = 0;
        release_perfworks_libraries();
    }
}


static int nvmon_perfworks_parse_metric(char* inoutmetric, int* isolated, int* keepInstances)
{
    if (!inoutmetric)
        return 0;
    int len = strlen(inoutmetric);

    bstring outmetric = bfromcstr(inoutmetric);
    int newline = bstrchrp(outmetric, '\n', 0);
    if (newline != BSTR_ERR)
    {
        bdelete(outmetric, newline, 1);
    }
    btrimws(outmetric);
    if (blength(outmetric) > 0)
    {
        *keepInstances = 0;
        if (bchar(outmetric, blength(outmetric)-1) == '+')
        {
            *keepInstances = 1;
            bdelete(outmetric, blength(outmetric)-1, 1);
        }
        if (blength(outmetric) > 0)
        {
            *isolated = 1;
            if (bchar(outmetric, blength(outmetric)-1) == '$')
            {
                bdelete(outmetric, blength(outmetric)-1, 1);
            }
            else if (bchar(outmetric, blength(outmetric)-1) == '&')
            {
                *isolated = 0;
                bdelete(outmetric, blength(outmetric)-1, 1);
            }
            if (blength(outmetric) > 0)
            {
                int ret = snprintf(inoutmetric, len, "%s", bdata(outmetric));
                if (ret > 0)
                {
                    inoutmetric[ret] = '\0';
                }
                bdestroy(outmetric);
                return 1;
            } 
        }
    }
    return 0;
    
}

// static int expand_metric(NVPA_MetricsContext* context, char* inmetric, struct bstrList* events)
// {
//     int iso = 0;
//     int keep = 0;
//     nvmon_perfworks_parse_metric(inmetric, &iso, &keep);
//     keep = 1;
//     NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = { NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE };
//     getMetricPropertiesBeginParams.pMetricsContext = context;
//     getMetricPropertiesBeginParams.pMetricName = inmetric;
//     LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_GetMetricProperties_BeginPtr)(&getMetricPropertiesBeginParams), return -EFAULT);

//     for (char** dep = getMetricPropertiesBeginParams.ppRawMetricDependencies; *dep ; ++dep)
//     {
//         bstrListAddChar(events, *dep);
//     }
//     NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams = { NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE };
//     getMetricPropertiesEndParams.pMetricsContext = context;
//     LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_GetMetricProperties_EndPtr)(&getMetricPropertiesEndParams), return -EFAULT);
//     return 0;
// }

static int combineBstrLists(struct bstrList* inout, struct bstrList* add, int no_duplicates)
{
    int i, j;
    for (i = 0; i < add->qty; i++)
    {
        if (no_duplicates)
        {
            int found = 0;
            for (j = 0; j < inout->qty; j++)
            {
                if (bstrcmp(add->entry[i], inout->entry[j]) == BSTR_OK)
                {
                    found = 1;
                    break;
                }
            }
            if (!found)
            {
                bstrListAdd(inout, add->entry[i]);
            }
        }
        else
        {
            bstrListAdd(inout, add->entry[i]);
        }
    }
}

void nvmon_perfworks_freeDevice(NvmonDevice_t dev)
{
    if (dev)
    {
        if (dev->chip)
        {
            free(dev->chip);
            dev->chip = NULL;
        }
        if (dev->allevents)
        {
            int i = 0;
            for (i = 0; i < dev->numAllEvents; i++)
            {
                free(dev->allevents[i]);
            }
            free(dev->allevents);
            dev->allevents = NULL;
            dev->numAllEvents = 0;
        }
    }
}

static void prepare_metric_name(bstring metric)
{
    bstring double_us = bfromcstr("__");
    bstring us = bfromcstr("_");
    bstring dot = bfromcstr(".");
    btrimws(metric);
    int newline = bstrchrp(metric, '\n', 0);
    if (newline != BSTR_ERR)
    {
        bdelete(metric, newline, 1);
    }
    btoupper(metric);

    bfindreplace(metric, double_us, us, 0);
    bfindreplace(metric, dot, us, 0);

    bdestroy(double_us);
    bdestroy(us);
    bdestroy(dot);
}


int
nvmon_perfworks_createDevice(int id, NvmonDevice *dev)
{
    int ierr = 0;
    size_t i = 0;
    int count = 0;

    if (dl_perfworks_libcuda == NULL ||
        dl_perfworks_libcudart == NULL ||
        dl_libhost == NULL ||
        dl_cupti == NULL)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, link_perfworks_libraries in createDevice);
        ierr = link_perfworks_libraries();
        if (ierr < 0)
        {
            return -ENODEV;
        }
    }

    LIKWID_CU_CALL((*cuDeviceGetCountPtr)(&count), return -1);
    if (count == 0)
    {
        printf("No GPUs found\n");
        return -1;
    }
    if (id < 0 || id >= count)
    {
        printf("GPU %d not available\n", id);
        return -1;
    }
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Found %d GPUs, count);
    

    // Assign device ID and get cuDevice from CUDA
    CU_CALL((*cuDeviceGetPtr)(&dev->cuDevice, id), return -1);
    dev->deviceId = id;
    dev->context = 0UL;

    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Current GPU %d, id);
    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    LIKWID_CUPTI_API_CALL((*cuptiProfilerInitializePtr)(&profilerInitializeParams), return -1);

    NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
    LIKWID_NVPW_API_CALL((*NVPW_InitializeHostPtr)(&initializeHostParams), return -1;);
    /* Dirty hack to support CUDA 10.1 and CUDA 11.0 insted of
    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    */
    size_t getChipNameParams_size = 0;
    if (cuda_runtime_version < 11000)
    {
        getChipNameParams_size = CUpti_Device_GetChipName_Params_STRUCT_SIZE10;
    }
    else
    {
        getChipNameParams_size = CUpti_Device_GetChipName_Params_STRUCT_SIZE11;
    }
    CUpti_Device_GetChipName_Params getChipNameParams = { getChipNameParams_size };
    /* End of dirty hack */
    getChipNameParams.deviceIndex = id;
    LIKWID_CUPTI_API_CALL((*cuptiDeviceGetChipNamePtr)(&getChipNameParams), return -1;);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Current GPU chip %s, getChipNameParams.pChipName);
    dev->chip = malloc(strlen(getChipNameParams.pChipName)+2);
    if (dev->chip)
    {
        int ret = snprintf(dev->chip, strlen(getChipNameParams.pChipName)+1, "%s", getChipNameParams.pChipName);
        if (ret > 0)
        {
            dev->chip[ret] = '\0';
        }
    }

    

    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
    metricsContextCreateParams.pChipName = dev->chip;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Create metric context for chip '%s', dev->chip);
    LIKWID_NVPW_API_CALL((*NVPW_CUDA_MetricsContext_CreatePtr)(&metricsContextCreateParams), return -1);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Create metric context done);

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;

    NVPW_MetricsContext_GetMetricNames_Begin_Params getMetricNameBeginParams = { NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE };
    getMetricNameBeginParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    getMetricNameBeginParams.hidePeakSubMetrics = 1;
    getMetricNameBeginParams.hidePerCycleSubMetrics = 1;
    getMetricNameBeginParams.hidePctOfPeakSubMetrics = 1;
    //getMetricNameBeginParams.hidePctOfPeakSubMetricsOnThroughputs = 1;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Create metric context getMetricNames);
    LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_GetMetricNames_BeginPtr)(&getMetricNameBeginParams), return -1);

    NVPW_MetricsContext_GetMetricNames_End_Params getMetricNameEndParams = { NVPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE };
    getMetricNameEndParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;



    dev->allevents = malloc(getMetricNameBeginParams.numMetrics * sizeof(NvmonEvent_t));
    if (dev->allevents)
    {
        for (i = 0; i < getMetricNameBeginParams.numMetrics; i++)
        {
            NvmonEvent_t event = malloc(sizeof(NvmonEvent));
            if (event)
            {
                memset(event, 0, sizeof(NvmonEvent));
                bstring t = bfromcstr(getMetricNameBeginParams.ppMetricNames[i]);
                prepare_metric_name(t);

                int ret = snprintf(event->name, NVMON_DEFAULT_STR_LEN-1, "%s", bdata(t));
                if (ret > 0)
                {
                    event->name[ret] = '\0';
                }
                bdestroy(t);
                ret = snprintf(event->real, NVMON_DEFAULT_STR_LEN-1, "%s", getMetricNameBeginParams.ppMetricNames[i]);
                if (ret > 0)
                {
                    event->real[ret] = '\0';
                }
                event->eventId = i;
                event->type = NVMON_PERFWORKS_EVENT;
                dev->allevents[i] = event;
                
            }
            else
            {
                ierr = -ENOMEM;
                break;
            }
        }
        dev->numAllEvents = i;
    }
    else
    {
        ierr = -ENOMEM;

    }
    dev->nvEventSets = NULL;
    dev->numNvEventSets = 0;
    dev->activeEventSet = -1;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Destroy metric context getMetricNames);
    LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_GetMetricNames_EndPtr)(&getMetricNameEndParams), return -1);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Destroy metric context);
    LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_DestroyPtr)(&metricsContextDestroyParams), return -1);
    return ierr;

}

int nvmon_perfworks_getEventsOfGpu(int gpuId, NvmonEventList_t* list)
{
    int ret = 0;
    NvmonDevice device;
    int err = nvmon_perfworks_createDevice(gpuId, &device);
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
                    out->limit = malloc(10*sizeof(char));
                    if (out->limit)
                    {
                        ret = snprintf(out->limit, 9, "GPU");
                        if (ret > 0)
                        {
                            out->limit[ret] = '\0';
                        }
                    }
                    out->desc = NULL;
 
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

static int nvmon_perfworks_getMetricRequests3(NVPA_MetricsContext* context,
                                              struct bstrList* events,
                                              NVPA_RawMetricRequest** requests)
{
    int i = 0;
    int j = 0;
    int isolated = 1;
    int keepInstances = 1;

    int raw_metrics = 0;
    for (i = 0; i < events->qty; i++)
    {
        //nvmon_perfworks_parse_metric(events->entry[i], &isolated, &keepInstances);
        keepInstances = 1; /* Bug in Nvidia API */
        NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = { NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE };
        getMetricPropertiesBeginParams.pMetricsContext = context;
        getMetricPropertiesBeginParams.pMetricName = bdata(events->entry[i]);
        NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams = { NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE };
        getMetricPropertiesEndParams.pMetricsContext = context;
        LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_GetMetricProperties_BeginPtr)(&getMetricPropertiesBeginParams), return -1);
        for (const char** dep = getMetricPropertiesBeginParams.ppRawMetricDependencies; *dep ; ++dep)
            raw_metrics++;
        LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_GetMetricProperties_EndPtr)(&getMetricPropertiesEndParams), return -1);
    }

    NVPA_RawMetricRequest* reqs = (NVPA_RawMetricRequest*) malloc(raw_metrics * sizeof(NVPA_RawMetricRequest));
    if (!reqs)
        return -ENOMEM;

    raw_metrics = 0;

    for (i = 0; i < events->qty; i++)
    {
        NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = { NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE };
        getMetricPropertiesBeginParams.pMetricsContext = context;
        getMetricPropertiesBeginParams.pMetricName = bdata(events->entry[i]);
        NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams = { NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE };
        getMetricPropertiesEndParams.pMetricsContext = context;
        LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_GetMetricProperties_BeginPtr)(&getMetricPropertiesBeginParams), free(reqs); return -1);
        for (const char** dep = getMetricPropertiesBeginParams.ppRawMetricDependencies; *dep; ++dep)
        {
            NVPA_RawMetricRequest* req = &reqs[raw_metrics];
            char* s = (char*)malloc((strlen(*dep)+2) * sizeof(char));
            int ret = snprintf(s, strlen(*dep)+1, "%s", *dep);
            s[ret] = '\0';
            req->pMetricName = (const char*)s;
            req->isolated = isolated;
            req->keepInstances = keepInstances;
            raw_metrics++;
        }
        LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_GetMetricProperties_EndPtr)(&getMetricPropertiesEndParams), free(reqs); return -1);
    }
    *requests = reqs;
    return raw_metrics;
}


static int nvmon_perfworks_getMetricRequests(NVPA_MetricsContext* context, struct bstrList* events, NVPA_RawMetricRequest** requests)
{
    int i = 0;
    int isolated = 1;
    int keepInstances = 1;
    struct bstrList* temp = bstrListCreate();
    const char ** raw_events = NULL;
    int num_raw = 0;
    for (i = 0; i < events->qty; i++)
    {
        //nvmon_perfworks_parse_metric(events->entry[i], &isolated, &keepInstances);
        keepInstances = 1; /* Bug in Nvidia API */
        NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = { NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE };
        NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams = { NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE };
        getMetricPropertiesBeginParams.pMetricsContext = context;
        getMetricPropertiesBeginParams.pMetricName = bdata(events->entry[i]);
        getMetricPropertiesEndParams.pMetricsContext = context;
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Metric %s, bdata(events->entry[i]));
        LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_GetMetricProperties_BeginPtr)(&getMetricPropertiesBeginParams), bstrListDestroy(temp); return -EFAULT);

        int count = 0;
        for (const char** dep = getMetricPropertiesBeginParams.ppRawMetricDependencies; *dep ; ++dep)
        {
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Metric depend %s, *dep);
            bstrListAddChar(temp, (char*)*dep);
        }
        
        LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_GetMetricProperties_EndPtr)(&getMetricPropertiesEndParams), bstrListDestroy(temp); return -EFAULT);

    }
    int num_reqs = 0;
    NVPA_RawMetricRequest* reqs = malloc((temp->qty+1) * NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE);
    if (!reqs)
    {
        bstrListDestroy(temp);
        return -ENOMEM;
    }
    for (i = 0; i < temp->qty; i++)
    {
        NVPA_RawMetricRequest* req = &reqs[num_reqs];
        char* s = malloc((blength(temp->entry[i])+2) * sizeof(char));
        if (s)
        {
            int ret = snprintf(s, blength(temp->entry[i])+1, "%s", bdata(temp->entry[i]));
            if (ret > 0)
            {
                s[ret] = '\0';
            }
            req->structSize = NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE;
            req->pMetricName = s;
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Metric Request %s, s);
            req->isolated = isolated;
            req->keepInstances = keepInstances;
            num_reqs++;
        }
        
    }
    bstrListDestroy(temp);
    *requests = reqs;
    return num_reqs;
}

static int nvmon_perfworks_createConfigImage(char* chip, struct bstrList* events, uint8_t **configImage, uint8_t* availImage)
{
    int i = 0;
    int ierr = 0;
    uint8_t* cimage = NULL;
    NVPA_RawMetricRequest* reqs = NULL;
    NVPA_RawMetricsConfig* pRawMetricsConfig = NULL;

    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
    metricsContextCreateParams.pChipName = chip;
    LIKWID_NVPW_API_CALL((*NVPW_CUDA_MetricsContext_CreatePtr)(&metricsContextCreateParams), return -1;);
    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;

    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Create config image for chip %s, chip);
    int num_reqs = nvmon_perfworks_getMetricRequests3(metricsContextCreateParams.pMetricsContext, events, &reqs);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Create config image for chip %s with %d metric requests, chip, num_reqs);

    NVPA_RawMetricsConfigOptions metricsConfigOptions = { NVPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE };
    metricsConfigOptions.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
    metricsConfigOptions.pChipName = chip;
    
    LIKWID_NVPW_API_CALL((*NVPA_RawMetricsConfig_CreatePtr)(&metricsConfigOptions, &pRawMetricsConfig), ierr = -1; goto nvmon_perfworks_createConfigImage_out);
    NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = { NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE };
    rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;

    if(availImage)
    {
        NVPW_RawMetricsConfig_SetCounterAvailability_Params setCounterAvailabilityParams = {NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE};
        setCounterAvailabilityParams.pRawMetricsConfig = pRawMetricsConfig;
        setCounterAvailabilityParams.pCounterAvailabilityImage = availImage;
        LIKWID_NVPW_API_CALL((*NVPW_RawMetricsConfig_SetCounterAvailabilityPtr)(&setCounterAvailabilityParams), ierr = -1; goto nvmon_perfworks_createConfigImage_out);
    }

    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = { NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE };
    beginPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    LIKWID_NVPW_API_CALL((*NVPW_RawMetricsConfig_BeginPassGroupPtr)(&beginPassGroupParams), ierr = -1; goto nvmon_perfworks_createConfigImage_out;);


    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = { NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE };
    addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
    addMetricsParams.pRawMetricRequests = reqs;
    addMetricsParams.numMetricRequests = num_reqs;
    LIKWID_NVPW_API_CALL((*NVPW_RawMetricsConfig_AddMetricsPtr)(&addMetricsParams), ierr = -1; goto nvmon_perfworks_createConfigImage_out;);

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = { NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE };
    endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    LIKWID_NVPW_API_CALL((*NVPW_RawMetricsConfig_EndPassGroupPtr)(&endPassGroupParams), ierr = -1; goto nvmon_perfworks_createConfigImage_out);

    NVPW_RawMetricsConfig_GetNumPasses_Params getNumPassesParams = { NVPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE };
    getNumPassesParams.pRawMetricsConfig = pRawMetricsConfig;
    LIKWID_NVPW_API_CALL((*NVPW_RawMetricsConfig_GetNumPassesPtr)(&getNumPassesParams), ierr = -1; goto nvmon_perfworks_createConfigImage_out);
    if (getNumPassesParams.numPipelinedPasses + getNumPassesParams.numIsolatedPasses > 1)
    {
        errno = 1;
        ierr = -errno;
        ERROR_PRINT(Given GPU eventset requires multiple passes. Currently not supported.)
        goto nvmon_perfworks_createConfigImage_out;
    }

    NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams = { NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE };
    generateConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
    LIKWID_NVPW_API_CALL((*NVPW_RawMetricsConfig_GenerateConfigImagePtr)(&generateConfigImageParams), ierr = -1; goto nvmon_perfworks_createConfigImage_out);

    NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = { NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE };
    getConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
    getConfigImageParams.bytesAllocated = 0;
    getConfigImageParams.pBuffer = NULL;
    LIKWID_NVPW_API_CALL((*NVPW_RawMetricsConfig_GetConfigImagePtr)(&getConfigImageParams), ierr = -1; goto nvmon_perfworks_createConfigImage_out);

    cimage = malloc(getConfigImageParams.bytesCopied);
    if (!cimage)
    {
        ierr = -ENOMEM;
        goto nvmon_perfworks_createConfigImage_out;
    }
    int ci_size = getConfigImageParams.bytesCopied;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Allocated %d byte for configImage, ci_size);

    getConfigImageParams.bytesAllocated = getConfigImageParams.bytesCopied;
    getConfigImageParams.pBuffer = cimage;
    LIKWID_NVPW_API_CALL((*NVPW_RawMetricsConfig_GetConfigImagePtr)(&getConfigImageParams), free(cimage); ierr = -1; goto nvmon_perfworks_createConfigImage_out);

    

nvmon_perfworks_createConfigImage_out:
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, nvmon_perfworks_createConfigImage_out enter %d, ierr);
    LIKWID_NVPW_API_CALL((*NVPW_RawMetricsConfig_DestroyPtr)(&rawMetricsConfigDestroyParams), return -1;);
    LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_DestroyPtr)(&metricsContextDestroyParams), return -1;);
    for (i = 0; i < num_reqs; i++)
    {
        free((void*)reqs[i].pMetricName);
    }
    free(reqs);
    if (ierr == 0)
    {
        ierr = ci_size;
        *configImage = cimage;
    }
    else
    {
        free(cimage);
    }
    
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, nvmon_perfworks_createConfigImage returns %d, ierr);
    return ierr;
}

static int nvmon_perfworks_createCounterDataPrefixImage(char* chip, struct bstrList* events, uint8_t **cdpImage)
{
    int i = 0;
    int ierr = 0;
    NVPA_RawMetricRequest* reqs = NULL;
    uint8_t* cdp = NULL;

    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };

    metricsContextCreateParams.pChipName = chip;
    LIKWID_NVPW_API_CALL((*NVPW_CUDA_MetricsContext_CreatePtr)(&metricsContextCreateParams), ierr = -1; goto nvmon_perfworks_createCounterDataPrefixImage_out);

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;

    int num_reqs = nvmon_perfworks_getMetricRequests3(metricsContextCreateParams.pMetricsContext, events, &reqs);

    NVPW_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = { NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE };
    counterDataBuilderCreateParams.pChipName = chip;
    LIKWID_NVPW_API_CALL((*NVPW_CounterDataBuilder_CreatePtr)(&counterDataBuilderCreateParams), ierr = -1; goto nvmon_perfworks_createCounterDataPrefixImage_out);

    NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = { NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE };
    counterDataBuilderDestroyParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;

    NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = { NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE };
    addMetricsParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
    addMetricsParams.pRawMetricRequests = reqs;
    addMetricsParams.numMetricRequests = num_reqs;
    LIKWID_NVPW_API_CALL((*NVPW_CounterDataBuilder_AddMetricsPtr)(&addMetricsParams), ierr = -1; goto nvmon_perfworks_createCounterDataPrefixImage_out);

    size_t counterDataPrefixSize = 0;
    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = { NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE };
    getCounterDataPrefixParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
    getCounterDataPrefixParams.bytesAllocated = 0;
    getCounterDataPrefixParams.pBuffer = NULL;
    LIKWID_NVPW_API_CALL((*NVPW_CounterDataBuilder_GetCounterDataPrefixPtr)(&getCounterDataPrefixParams), ierr = -1; goto nvmon_perfworks_createCounterDataPrefixImage_out);

    cdp = malloc(getCounterDataPrefixParams.bytesCopied+10);
    if (!cdp)
    {
        ierr = -ENOMEM;
        goto nvmon_perfworks_createCounterDataPrefixImage_out;
    }
    int pi_size = getCounterDataPrefixParams.bytesCopied;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Allocated %d byte for configPrefixImage, pi_size);

    getCounterDataPrefixParams.bytesAllocated = getCounterDataPrefixParams.bytesCopied+10;
    getCounterDataPrefixParams.pBuffer = cdp;
    LIKWID_NVPW_API_CALL((*NVPW_CounterDataBuilder_GetCounterDataPrefixPtr)(&getCounterDataPrefixParams), free(cdp); ierr = -1; goto nvmon_perfworks_createCounterDataPrefixImage_out);

    

nvmon_perfworks_createCounterDataPrefixImage_out:
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, nvmon_perfworks_createCounterDataPrefixImage_out enter %d, ierr);
    // for (i = 0; i < num_reqs; i++)
    // {
    //     free((void*)reqs[i].pMetricName);
    // }
    // free(reqs);
    LIKWID_NVPW_API_CALL((*NVPW_CounterDataBuilder_DestroyPtr)(&counterDataBuilderDestroyParams), ierr = -1;);
    LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_DestroyPtr)(&metricsContextDestroyParams), ierr = -1);
    for (i = 0; i < num_reqs; i++)
    {
        free((void*)reqs[i].pMetricName);
    }
    free(reqs);
    if (ierr == 0)
    {
        ierr = pi_size;
        *cdpImage = cdp;
    }
    else
    {
        free(cdp);
    }
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, nvmon_perfworks_createCounterDataPrefixImage returns %d, ierr);
    return ierr;
}


int
nvmon_perfworks_addEventSet(NvmonDevice_t device, const char* eventString)
{
    int i = 0, j = 0;
    int curDeviceId = -1;
    CUcontext curContext;
    struct bstrList* tmp, *eventtokens = NULL;
    int gid = -1;
    uint8_t* configImage = NULL;
    uint8_t* prefixImage = NULL;
    uint8_t* availImage = NULL;
    size_t availImageSize = 0;

    //cuptiProfiler_init();

    LIKWID_CUDA_API_CALL((*cudaGetDevicePtr)(&curDeviceId), return -EFAULT);
    LIKWID_CUDA_API_CALL((*cudaFreePtr)(NULL), return -EFAULT);
    LIKWID_CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Add events to GPU device %d with context %u, device->deviceId, curContext);

    if (curDeviceId != device->deviceId)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Switching to GPU device %d, device->deviceId);
        LIKWID_CUDA_API_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
    }

    int popContext = perfworks_check_nv_context(device, curContext);
    if (popContext < 0)
    {
        errno = -popContext;
        ERROR_PRINT(Failed to get context);
    }

    bstring eventBString = bfromcstr(eventString);
    tmp = bsplit(eventBString, ',');
    bdestroy(eventBString);

    eventtokens = bstrListCreate();

    for (i = 0; i < tmp->qty; i++)
    {
        struct bstrList* parts = bsplit(tmp->entry[i], ':');
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, %s, bdata(parts->entry[0]));
        for (j = 0; j < device->numAllEvents; j++)
        {
            bstring bname = bfromcstr(device->allevents[j]->name);
            if (bstrcmp(parts->entry[0], bname) == BSTR_OK)
            {
                bstrListAddChar(eventtokens, device->allevents[j]->real);
                GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Adding real event %s, device->allevents[j]->real);
            }
        }
        bstrListDestroy(parts);
    }
    if (eventtokens->qty == 0)
    {
        ERROR_PRINT(No event in eventset);
        bstrListDestroy(eventtokens);
        if(popContext > 0)
        {
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Pop Context %ld for device %d, device->context, device->deviceId);
            LIKWID_CU_CALL((*cuCtxPopCurrentPtr)(&device->context), return -EFAULT);
        }
        if (curDeviceId != device->deviceId)
        {
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Switching to GPU device %d, device->deviceId);
            LIKWID_CUDA_API_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
        }
        return -EFAULT;
    }
    
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Increase size of eventSet space on device %d, device->deviceId);
    NvmonEventSet* tmpEventSet = realloc(device->nvEventSets, (device->numNvEventSets+1)*sizeof(NvmonEventSet));
    if (!tmpEventSet)
    {
        ERROR_PRINT(Cannot enlarge GPU %d eventSet list, device->deviceId);
        return -ENOMEM;
    }
    device->nvEventSets = tmpEventSet;
    NvmonEventSet* newEventSet = &device->nvEventSets[device->numNvEventSets];
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Filling eventset %d on device %d, device->numNvEventSets, device->deviceId);


    if (cuda_version >= 11000 && cuda_runtime_version >= 11000)
    {
        CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
        getCounterAvailabilityParams.ctx = device->context;
        LIKWID_CUPTI_API_CALL((*cuptiProfilerGetCounterAvailabilityPtr)(&getCounterAvailabilityParams), return -EFAULT);
        
        availImage = malloc(getCounterAvailabilityParams.counterAvailabilityImageSize);
        if (!availImage)
        {
            return -ENOMEM;
        }
        getCounterAvailabilityParams.ctx = device->context;
        getCounterAvailabilityParams.pCounterAvailabilityImage = availImage;
        LIKWID_CUPTI_API_CALL((*cuptiProfilerGetCounterAvailabilityPtr)(&getCounterAvailabilityParams), return -EFAULT);
        availImageSize = getCounterAvailabilityParams.counterAvailabilityImageSize;
    }

    int ci_size = nvmon_perfworks_createConfigImage(device->chip, eventtokens, &configImage, availImage);
    int pi_size = nvmon_perfworks_createCounterDataPrefixImage(device->chip, eventtokens, &prefixImage);
    
    
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Filling eventset %d on device %d, device->numNvEventSets, device->deviceId);
    if (configImage && prefixImage)
    {
        newEventSet->configImage = configImage;
        newEventSet->configImageSize = (size_t)ci_size;
        newEventSet->counterDataImagePrefix = prefixImage;
        newEventSet->counterDataImagePrefixSize = (size_t)pi_size;
        newEventSet->counterDataImage = NULL;
        newEventSet->counterDataImageSize = 0; 
        newEventSet->counterDataScratchBuffer = NULL;
        newEventSet->counterDataScratchBufferSize = 0;
        newEventSet->counterAvailabilityImage = availImage;
        newEventSet->counterAvailabilityImageSize = availImageSize;
        newEventSet->events = eventtokens;
        newEventSet->numberOfEvents = tmp->qty;
        newEventSet->id = device->numNvEventSets;
        gid = device->numNvEventSets;
        newEventSet->results = malloc(tmp->qty * sizeof(NvmonEventResult));
        if (newEventSet->results == NULL)
        {
            ERROR_PRINT(Cannot allocate result list for group %d\n, gid);
            return -ENOMEM;
        }
        memset(newEventSet->results, 0, tmp->qty * sizeof(NvmonEventResult));
        device->numNvEventSets++;
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Adding eventset %d, gid);
    }

    bstrListDestroy(tmp);
    //bstrListDestroy(eventtokens);
    

    if(popContext > 0)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Pop Context %ld for device %d, device->context, device->deviceId);
        LIKWID_CU_CALL((*cuCtxPopCurrentPtr)(&device->context), return -EFAULT);
    }
    if (curDeviceId != device->deviceId)
    {
        LIKWID_CUDA_API_CALL((*cudaSetDevicePtr)(curDeviceId), return -EFAULT);
    }
    return gid;
}


static int nvmon_perfworks_setupCounterImageData(NvmonEventSet* eventSet)//int size, uint8_t** counterDataImage, uint8_t** counterDataScratchBuffer, uint8_t* counterDataImagePrefix)
{
    int cimage_size = 0;

    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.counterDataPrefixSize = eventSet->counterDataImagePrefixSize;
    counterDataImageOptions.pCounterDataPrefix = eventSet->counterDataImagePrefix;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, counterDataPrefixSize %ld, eventSet->counterDataImagePrefixSize);
    counterDataImageOptions.maxNumRanges = 1;
    counterDataImageOptions.maxNumRangeTreeNodes = 2;
    counterDataImageOptions.maxRangeNameLength = NVMON_DEFAULT_STR_LEN-1;

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};

    calculateSizeParams.pOptions = &counterDataImageOptions;
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

    LIKWID_CUPTI_API_CALL((*cuptiProfilerCounterDataImageCalculateSizePtr)(&calculateSizeParams), return -EFAULT);

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.pOptions = &counterDataImageOptions;
    initializeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Resize counterDataImage to %ld, calculateSizeParams.counterDataImageSize);
    uint8_t* tmp = realloc(eventSet->counterDataImage, calculateSizeParams.counterDataImageSize);
    if (!tmp)
    {
        return -ENOMEM;
    }
    eventSet->counterDataImage = tmp;
    eventSet->counterDataImageSize = calculateSizeParams.counterDataImageSize;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Resized counterDataImage to %ld, eventSet->counterDataImageSize);
    //initializeParams.pCounterDataImage = &(eventSet->counterDataImage)[0];
    initializeParams.pCounterDataImage = eventSet->counterDataImage;
    LIKWID_CUPTI_API_CALL((*cuptiProfilerCounterDataImageInitializePtr)(&initializeParams), return -EFAULT);

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = {CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    scratchBufferSizeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    scratchBufferSizeParams.pCounterDataImage = eventSet->counterDataImage;
    LIKWID_CUPTI_API_CALL((*cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr)(&scratchBufferSizeParams), return -EFAULT);

    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Resize counterDataScratchBuffer to %ld, scratchBufferSizeParams.counterDataScratchBufferSize);
    tmp = realloc(eventSet->counterDataScratchBuffer, scratchBufferSizeParams.counterDataScratchBufferSize);
    if(!tmp)
    {
        return -ENOMEM;
    }
    eventSet->counterDataScratchBuffer = tmp;
    eventSet->counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Resized counterDataScratchBuffer to %ld, eventSet->counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = {CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
    initScratchBufferParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

    initScratchBufferParams.pCounterDataImage = initializeParams.pCounterDataImage;
    initScratchBufferParams.counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;
    initScratchBufferParams.pCounterDataScratchBuffer = eventSet->counterDataScratchBuffer;
    LIKWID_CUPTI_API_CALL((*cuptiProfilerCounterDataImageInitializeScratchBufferPtr)(&initScratchBufferParams), return -EFAULT);

    return 0;
}


typedef struct {
    int num_ranges;
    struct bstrList* names;
    double* values;
} PerfWorksMetricRanges;

static void freeCharList(int len, char** l)
{
    if (len >= 0 && l)
    {
        int i = 0;
        for (i = 0; i < len; i++)
        {
            free(l[i]);
        }
        free(l);
    }
}



static int nvmon_perfworks_getMetricValue(char* chip, uint8_t* counterDataImage, struct bstrList* events, double** values)
{
    if ((!chip) || (!counterDataImage) || (!events) || (!values))
        return -EINVAL;
    int i = 0;
    int j = 0;
    int ierr = 0;
    char** metricnames = NULL;
    double* gpuValues = malloc(events->qty * sizeof(double));
    if (!gpuValues)
        return -ENOMEM;


    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
    metricsContextCreateParams.pChipName = chip;
    LIKWID_NVPW_API_CALL((*NVPW_CUDA_MetricsContext_CreatePtr)(&metricsContextCreateParams), ierr = -1; goto nvmon_perfworks_getMetricValue_out);

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;

    NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
    getNumRangesParams.pCounterDataImage = counterDataImage;
    LIKWID_NVPW_API_CALL((*NVPW_CounterData_GetNumRangesPtr)(&getNumRangesParams), ierr = -1; goto nvmon_perfworks_getMetricValue_out;);


    int num_metricnames = bstrListToCharList(events, &metricnames);


    NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = { NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE };
    setCounterDataParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    setCounterDataParams.pCounterDataImage = counterDataImage;
    setCounterDataParams.isolated = 1;
    setCounterDataParams.rangeIndex = 0;
    LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_SetCounterDataPtr)(&setCounterDataParams), ierr = -1; goto nvmon_perfworks_getMetricValue_out;);

    //double* gpuValues = malloc(events->qty * sizeof(double));
    //memset(gpuValues, 0, events->qty * sizeof(double));
    NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = { NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE };
    evalToGpuParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    evalToGpuParams.numMetrics = num_metricnames;
    evalToGpuParams.ppMetricNames = (const char**)metricnames;
    memset(gpuValues, 0, events->qty * sizeof(double));
    evalToGpuParams.pMetricValues = gpuValues;
    LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_EvaluateToGpuValuesPtr)(&evalToGpuParams), ierr = -1; free(gpuValues); goto nvmon_perfworks_getMetricValue_out;);
    // for (j = 0; j < events->qty; j++)
    // {
    //     bstrListAdd(r->names, rname);
    //     r->values[j] += gpuValues[j];
    // }
    for (j = 0; j < events->qty; j++)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Final Eval %s: %f, bdata(events->entry[j]), gpuValues[j]);
    }
    *values = gpuValues;

nvmon_perfworks_getMetricValue_out:
    if (ierr != 0) free(gpuValues);
    freeCharList(num_metricnames, metricnames);
    LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_DestroyPtr)(&metricsContextDestroyParams), ierr = -1);
    return ierr;
}


int nvmon_perfworks_setupCounters(NvmonDevice_t device, NvmonEventSet* eventSet)
{
    int size = 0;
    int curDeviceId = 0;
    uint8_t *cimage = NULL;
    uint8_t *scratch = NULL;
    uint8_t *prefix = NULL;
    CUcontext curContext;
    LIKWID_CUDA_API_CALL((*cudaGetDevicePtr)(&curDeviceId), return -EFAULT);
    if (curDeviceId != device->deviceId)
    {
        LIKWID_CUDA_API_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
    }
    LIKWID_CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);
    int popContext = perfworks_check_nv_context(device, curContext);
    if (popContext < 0)
    {
        errno = -popContext;
        ERROR_PRINT(Failed to get context)
    }
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Setup Counters on device %d, device->deviceId);

    //cuptiProfiler_init();

    int ret = nvmon_perfworks_setupCounterImageData(eventSet);//&cimage, &scratch, eventSet->counterDataImagePrefix);
    device->activeEventSet = eventSet->id;
    nvGroupSet->activeGroup = eventSet->id;
    // if (ret > 0)
    // {
    //     eventSet->counterDataImage = cimage;
    //     eventSet->counterDataImageSize = ret;
    //     eventSet->counterDataScratchBuffer = scratch;
    //     // eventSet->counterDataImagePrefix = prefix;
    //     // eventSet->counterDataImageSize = ret;
    //     device->activeEventSet = eventSet->id;
    // }
    if(popContext > 0)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Pop Context %ld for device %d, device->context, device->deviceId);
        LIKWID_CU_CALL((*cuCtxPopCurrentPtr)(&device->context), return -EFAULT);
    }
    if (curDeviceId != device->deviceId)
    {
        LIKWID_CUDA_API_CALL((*cudaSetDevicePtr)(curDeviceId), return -EFAULT);
    }
    return ret;
}

int nvmon_perfworks_startCounters(NvmonDevice_t device)
{

    int numRanges = 1;
    //CUcontext cuContext;
    int curDeviceId = 0;
    CUcontext curContext;

    LIKWID_CUDA_API_CALL((*cudaGetDevicePtr)(&curDeviceId), return -EFAULT);
    if (curDeviceId != device->deviceId)
    {
        LIKWID_CUDA_API_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
    }
    //LIKWID_CU_CALL((*cuCtxGetCurrentPtr)(&cuContext), return -EFAULT);
    LIKWID_CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);
    int popContext = perfworks_check_nv_context(device, curContext);
    if (popContext < 0)
    {
        errno = -popContext;
        ERROR_PRINT(Failed to get context)
    }

    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Start Counters on device %d (Eventset %d), device->deviceId, device->activeEventSet);
    NvmonEventSet* eventSet = &device->nvEventSets[device->activeEventSet];

    CUpti_Profiler_BeginSession_Params beginSessionParams = {CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
    size_t CUpti_Profiler_SetConfig_Params_size = 3*sizeof(size_t) + sizeof(void*) + sizeof(CUcontext) + sizeof(const uint8_t*);
    if (cuda_runtime_version < 11000)
    {
        CUpti_Profiler_SetConfig_Params_size = 56;//2*sizeof(uint16_t); 
    }
    else
    {
        CUpti_Profiler_SetConfig_Params_size = 58;
    }
    CUpti_Profiler_SetConfig_Params setConfigParams = {CUpti_Profiler_SetConfig_Params_size};
    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
    CUpti_Profiler_PushRange_Params pushRangeParams = {CUpti_Profiler_PushRange_Params_STRUCT_SIZE};
    CUpti_Profiler_BeginPass_Params beginPassParams = {CUpti_Profiler_BeginPass_Params_STRUCT_SIZE};

    beginSessionParams.ctx = device->context;//cuContext;//;
    beginSessionParams.counterDataImageSize = eventSet->counterDataImageSize;
    beginSessionParams.pCounterDataImage = eventSet->counterDataImage;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, (START) counterDataImageSize %ld, eventSet->counterDataImageSize);
    beginSessionParams.counterDataScratchBufferSize = eventSet->counterDataScratchBufferSize;
    beginSessionParams.pCounterDataScratchBuffer = eventSet->counterDataScratchBuffer;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, (START) counterDataScratchBufferSize %ld, eventSet->counterDataScratchBufferSize);
    beginSessionParams.range = CUPTI_UserRange;
    beginSessionParams.replayMode = CUPTI_UserReplay;
    beginSessionParams.maxRangesPerPass = 1;
    beginSessionParams.maxLaunchesPerPass = 1;

    LIKWID_CUPTI_API_CALL((*cuptiProfilerBeginSessionPtr)(&beginSessionParams), return -1);

    setConfigParams.pConfig = eventSet->configImage;
    setConfigParams.configSize = eventSet->configImageSize;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, (START) configImage %ld, eventSet->configImageSize);

    setConfigParams.passIndex = 0;
    setConfigParams.ctx = device->context;
    setConfigParams.minNestingLevel = 1;
    setConfigParams.numNestingLevels = 1;
    setConfigParams.targetNestingLevel = 1;
    LIKWID_CUPTI_API_CALL((*cuptiProfilerSetConfigPtr)(&setConfigParams), return -1);

    LIKWID_CUPTI_API_CALL((*cuptiProfilerBeginPassPtr)(&beginPassParams), return -1;);
    LIKWID_CUPTI_API_CALL((*cuptiProfilerEnableProfilingPtr)(&enableProfilingParams), return -1);
    pushRangeParams.pRangeName = "nvmon_perfworks";
    LIKWID_CUPTI_API_CALL((*cuptiProfilerPushRangePtr)(&pushRangeParams), return -1);

    if(popContext > 0)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Pop Context %ld for device %d, device->context, device->deviceId);
        LIKWID_CU_CALL((*cuCtxPopCurrentPtr)(&device->context), return -EFAULT);
    }
    if (curDeviceId != device->deviceId)
    {
        LIKWID_CUDA_API_CALL((*cudaSetDevicePtr)(curDeviceId), return -EFAULT);
    }

    return 0;
}

int nvmon_perfworks_stopCounters(NvmonDevice_t device)
{
    double* values;
    int curDeviceId = 0;
    CUcontext curContext;
    LIKWID_CUDA_API_CALL((*cudaGetDevicePtr)(&curDeviceId), return -EFAULT);
    if (curDeviceId != device->deviceId)
    {
        LIKWID_CUDA_API_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
    }
    LIKWID_CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);
    int popContext = perfworks_check_nv_context(device, curContext);
    if (popContext < 0)
    {
        errno = -popContext;
        ERROR_PRINT(Failed to get context)
    }
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Stop Counters on device %d (Eventset %d), device->deviceId, device->activeEventSet);
    NvmonEventSet* eventSet = &device->nvEventSets[device->activeEventSet];

    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
    CUpti_Profiler_PopRange_Params popRangeParams = {CUpti_Profiler_PopRange_Params_STRUCT_SIZE};
    size_t CUpti_Profiler_EndPass_Params_size = 0;
    size_t CUpti_Profiler_FlushCounterData_Params_size = 0;

    if (cuda_runtime_version < 11000)
    {
        CUpti_Profiler_EndPass_Params_size = CUpti_Profiler_EndPass_Params_STRUCT_SIZE10;
        CUpti_Profiler_FlushCounterData_Params_size = CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE10;
    }
    else
    {
        CUpti_Profiler_EndPass_Params_size = CUpti_Profiler_EndPass_Params_STRUCT_SIZE11;
        CUpti_Profiler_FlushCounterData_Params_size = CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE11;
    }
    CUpti_Profiler_EndPass_Params endPassParams = {CUpti_Profiler_EndPass_Params_size};
    CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {CUpti_Profiler_FlushCounterData_Params_size};
    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
    CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};

    LIKWID_CUPTI_API_CALL((*cuptiProfilerPopRangePtr)(&popRangeParams), return -1);
    LIKWID_CUPTI_API_CALL((*cuptiProfilerDisableProfilingPtr)(&disableProfilingParams), return -1);
    LIKWID_CUPTI_API_CALL((*cuptiProfilerEndPassPtr)(&endPassParams), return -1);
    if (endPassParams.allPassesSubmitted != 1)
    {
        ERROR_PRINT(Events cannot be measured in a single pass and multi-pass/kernel replay is current not supported);
    }
    LIKWID_CUPTI_API_CALL((*cuptiProfilerFlushCounterDataPtr)(&flushCounterDataParams), return -1);
    LIKWID_CUPTI_API_CALL((*cuptiProfilerUnsetConfigPtr)(&unsetConfigParams), return -1);
    LIKWID_CUPTI_API_CALL((*cuptiProfilerEndSessionPtr)(&endSessionParams), return -1);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Get results on device %d (Eventset %d), device->deviceId, device->activeEventSet);

    nvmon_perfworks_getMetricValue(device->chip, eventSet->counterDataImage, eventSet->events, &values);

    int i = 0, j = 0;
    for (j = 0; j < eventSet->events->qty; j++)
    {
        double res = values[j];
        eventSet->results[j].lastValue = res;
        eventSet->results[j].fullValue += res;
        eventSet->results[j].stopValue = eventSet->results[j].fullValue;
        eventSet->results[j].overflows = 0;
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, %s Last %f Full %f, bdata(eventSet->events->entry[j]), eventSet->results[j].lastValue, eventSet->results[j].fullValue);
    }

    if(popContext > 0)
    {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Pop Context %ld for device %d, device->context, device->deviceId);
        LIKWID_CU_CALL((*cuCtxPopCurrentPtr)(&device->context), return -EFAULT);
    }
    if (curDeviceId != device->deviceId)
    {
        LIKWID_CUDA_API_CALL((*cudaSetDevicePtr)(curDeviceId), return -EFAULT);
    }


    return 0;

}

int nvmon_perfworks_readCounters(NvmonDevice_t device)
{
    nvmon_perfworks_stopCounters(device);
    nvmon_perfworks_startCounters(device);
}


NvmonFunctions nvmon_perfworks_functions = {
    .freeDevice = nvmon_perfworks_freeDevice,
    .createDevice = nvmon_perfworks_createDevice,
    .getEventList = nvmon_perfworks_getEventsOfGpu,
    .addEvents = nvmon_perfworks_addEventSet,
    .setupCounters = nvmon_perfworks_setupCounters,
    .startCounters = nvmon_perfworks_startCounters,
    .stopCounters = nvmon_perfworks_stopCounters,
    .readCounters = nvmon_perfworks_readCounters,
};
#else
NvmonFunctions nvmon_perfworks_functions = {
    .freeDevice = NULL,
    .createDevice = NULL,
    .getEventList = NULL,
    .addEvents = NULL,
    .setupCounters = NULL,
    .startCounters = NULL,
    .readCounters = NULL,
};

#endif

#endif /* LIKWID_NVMON_PERFWORKS_H */
