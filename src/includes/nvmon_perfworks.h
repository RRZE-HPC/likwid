/*
 * =======================================================================================
 *
 *      Filename:  nvmon_perfworks.h
 *
 *      Description:  Header File of nvmon module (PerfWorks backend).
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

#ifndef DECLARE_CUDAFUNC
#define CUDAAPIWEAK __attribute__( ( weak ) )
#define DECLARE_CUDAFUNC(funcname, funcsig) cudaError_t CUDAAPIWEAK funcname funcsig;  cudaError_t( *funcname##Ptr ) funcsig;
DECLARE_CUDAFUNC(cudaGetDevice, (int *));
DECLARE_CUDAFUNC(cudaSetDevice, (int));
DECLARE_CUDAFUNC(cudaFree, (void *));
#endif

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


#ifndef DECLARE_CUPTIFUNC
#define CUPTIAPIWEAK __attribute__( ( weak ) )
#define DECLARE_CUPTIFUNC(funcname, funcsig) CUptiResult CUPTIAPIWEAK funcname funcsig;  CUptiResult( *funcname##Ptr ) funcsig;
#endif
DECLARE_CUPTIFUNC(cuptiDeviceGetChipName, (CUpti_Device_GetChipName_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerInitialize, (CUpti_Profiler_Initialize_Params* params));
DECLARE_CUPTIFUNC(cuptiProfilerDeInitialize, (CUpti_Profiler_DeInitialize_Params* params))

#ifndef DLSYM_AND_CHECK
#define DLSYM_AND_CHECK( dllib, name ) dlsym( dllib, name ); if ( dlerror() != NULL ) { return -1; }
#endif


static int cuptiProfiler_initialized = 0;


static int cuptiProfiler_init()
{
    if (!cuptiProfiler_initialized)
    {
        LIKWID_CU_CALL((*cuInitPtr)(0), return -1);
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
        CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
        LIKWID_CUPTI_API_CALL((*cuptiProfilerDeInitializePtr)(&profilerDeInitializeParams), cuptiProfiler_initialized = 0; return;);
        cuptiProfiler_initialized = 0;
    }
}


static int
link_perfworks_libraries(void)
{
    /* Attempt to guess if we were statically linked to libc, if so bail */
    if(_dl_non_dynamic_init != NULL) {
        return -1;
    }
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

    dl_perfworks_libcudart = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if (!dl_perfworks_libcudart)
    {
        fprintf(stderr, "CUDA runtime library libcudart.so not found.");
        return -1;
    }
    cudaGetDevicePtr = DLSYM_AND_CHECK(dl_perfworks_libcudart, "cudaGetDevice");
    cudaSetDevicePtr = DLSYM_AND_CHECK(dl_perfworks_libcudart, "cudaSetDevice");
    cudaFreePtr = DLSYM_AND_CHECK(dl_perfworks_libcudart, "cudaFree");

    dl_libhost = dlopen("libnvperf_host.so", RTLD_NOW | RTLD_GLOBAL);
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

    dl_cupti = dlopen("libcupti.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_cupti || dlerror() != NULL)
    {
        fprintf(stderr, "CUpti library libcupti.so not found.");
        return -1;
    }
    cuptiProfilerInitializePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerInitialize");
    cuptiProfilerDeInitializePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerDeInitialize");
    cuptiDeviceGetChipNamePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceGetChipName");
    dlerror();
    return 0;
}

static void
release_perfworks_libraries(void)
{
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
        ierr = link_perfworks_libraries();
        if (ierr < 0)
        {
            return -ENODEV;
        }
    }

    cuptiProfiler_init();
    

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


    // Assign device ID and get cuDevice from CUDA
    CU_CALL((*cuDeviceGetPtr)(&dev->cuDevice, id), return -1);
    dev->deviceId = id;
    dev->context = 0UL;


    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipNameParams.deviceIndex = id;
    LIKWID_CUPTI_API_CALL((*cuptiDeviceGetChipNamePtr)(&getChipNameParams), return -1;);
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
    LIKWID_NVPW_API_CALL((*NVPW_CUDA_MetricsContext_CreatePtr)(&metricsContextCreateParams), return -1);

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;

    NVPW_MetricsContext_GetMetricNames_Begin_Params getMetricNameBeginParams = { NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE };
    getMetricNameBeginParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    getMetricNameBeginParams.hidePeakSubMetrics = 1;
    getMetricNameBeginParams.hidePerCycleSubMetrics = 1;
    getMetricNameBeginParams.hidePctOfPeakSubMetrics = 1;
    //getMetricNameBeginParams.hidePctOfPeakSubMetricsOnThroughputs = 1;
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
    LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_GetMetricNames_EndPtr)(&getMetricNameEndParams), return -1);
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


static int nvmon_perfworks_getMetricRequests(NVPA_MetricsContext* context, struct bstrList* events, NVPA_RawMetricRequest** requests)
{
    int i = 0;
    int isolated = 1;
    int keepInstances = 1;
    struct bstrList* temp = bstrListCreate();
    for (i = 0; i < events->qty; i++)
    {
        //nvmon_perfworks_parse_metric(events->entry[i], &isolated, &keepInstances);
        keepInstances = 1;
        NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = { NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE };
        getMetricPropertiesBeginParams.pMetricsContext = context;
        getMetricPropertiesBeginParams.pMetricName = bdata(events->entry[i]);
        LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_GetMetricProperties_BeginPtr)(&getMetricPropertiesBeginParams), bstrListDestroy(temp); return -EFAULT);

        for (const char** dep = getMetricPropertiesBeginParams.ppRawMetricDependencies; *dep ; ++dep)
        {
            bstrListAddChar(temp, (char*)*dep);
        }
        NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams = { NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE };
        getMetricPropertiesEndParams.pMetricsContext = context;
        LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_GetMetricProperties_EndPtr)(&getMetricPropertiesEndParams), bstrListDestroy(temp); return -EFAULT);

    }
    int num_reqs = 0;
    NVPA_RawMetricRequest* reqs = malloc(temp->qty * NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE);
    if (!reqs)
    {
        bstrListDestroy(temp);
        return -ENOMEM;
    }
    for (i = 0; i < temp->qty; i++)
    {
        NVPA_RawMetricRequest* req = &reqs[num_reqs];
        req->pMetricName = bdata(temp->entry[i]);
        req->isolated = isolated;
        req->keepInstances = keepInstances;
        num_reqs++;
    }
    bstrListDestroy(temp);
    *requests = reqs;
    return num_reqs;
}

static int nvmon_perfworks_createConfigImage(char* chip, struct bstrList* events, uint8_t **configImage)
{
    int ierr = 0;
    uint8_t* cimage = NULL;
    NVPA_RawMetricRequest* reqs = NULL;
    NVPA_RawMetricsConfig* pRawMetricsConfig = NULL;

    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
    metricsContextCreateParams.pChipName = chip;
    LIKWID_NVPW_API_CALL((*NVPW_CUDA_MetricsContext_CreatePtr)(&metricsContextCreateParams), return -1;);
    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;

    int num_reqs = nvmon_perfworks_getMetricRequests(metricsContextCreateParams.pMetricsContext, events, &reqs);

    NVPA_RawMetricsConfigOptions metricsConfigOptions = { NVPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE };
    metricsConfigOptions.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
    metricsConfigOptions.pChipName = chip;
    
    LIKWID_NVPW_API_CALL((*NVPA_RawMetricsConfig_CreatePtr)(&metricsConfigOptions, &pRawMetricsConfig), ierr = -1; goto nvmon_perfworks_createConfigImage_out);
    NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = { NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE };
    rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;

    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = { NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE };
    beginPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    LIKWID_NVPW_API_CALL((*NVPW_RawMetricsConfig_BeginPassGroupPtr)(&beginPassGroupParams), ierr = -1; goto nvmon_perfworks_createConfigImage_out;);


    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = { NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE };
    addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
    addMetricsParams.pRawMetricRequests = &reqs[0];
    addMetricsParams.numMetricRequests = num_reqs;
    LIKWID_NVPW_API_CALL((*NVPW_RawMetricsConfig_AddMetricsPtr)(&addMetricsParams), ierr = -1; goto nvmon_perfworks_createConfigImage_out;);

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = { NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE };
    endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    LIKWID_NVPW_API_CALL((*NVPW_RawMetricsConfig_EndPassGroupPtr)(&endPassGroupParams), ierr = -1; goto nvmon_perfworks_createConfigImage_out);

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

    getConfigImageParams.bytesAllocated = getConfigImageParams.bytesCopied;
    getConfigImageParams.pBuffer = &cimage[0];
    LIKWID_NVPW_API_CALL((*NVPW_RawMetricsConfig_GetConfigImagePtr)(&getConfigImageParams), free(cimage); ierr = -1; goto nvmon_perfworks_createConfigImage_out);

    *configImage = cimage;

nvmon_perfworks_createConfigImage_out:
    free(reqs);
    LIKWID_NVPW_API_CALL((*NVPW_RawMetricsConfig_DestroyPtr)(&rawMetricsConfigDestroyParams), return -1;);
    LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_DestroyPtr)(&metricsContextDestroyParams), return -1;);
    return ierr;
}

static int nvmon_perfworks_createCounterDataPrefixImage(char* chip, struct bstrList* events, uint8_t **cdpImage)
{
    int ierr = 0;
    NVPA_RawMetricRequest* reqs = NULL;
    uint8_t* cdp = NULL;

    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
    metricsContextCreateParams.pChipName = chip;
    LIKWID_NVPW_API_CALL((*NVPW_CUDA_MetricsContext_CreatePtr)(&metricsContextCreateParams), ierr = -1; goto nvmon_perfworks_createCounterDataPrefixImage_out);

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;

    int num_reqs = nvmon_perfworks_getMetricRequests(metricsContextCreateParams.pMetricsContext, events, &reqs);

    NVPW_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = { NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE };
    counterDataBuilderCreateParams.pChipName = chip;
    LIKWID_NVPW_API_CALL((*NVPW_CounterDataBuilder_CreatePtr)(&counterDataBuilderCreateParams), ierr = -1; goto nvmon_perfworks_createCounterDataPrefixImage_out);

    NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = { NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE };
    counterDataBuilderDestroyParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;

    NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = { NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE };
    addMetricsParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
    addMetricsParams.pRawMetricRequests = &reqs[0];
    addMetricsParams.numMetricRequests = num_reqs;
    LIKWID_NVPW_API_CALL((*NVPW_CounterDataBuilder_AddMetricsPtr)(&addMetricsParams), ierr = -1; goto nvmon_perfworks_createCounterDataPrefixImage_out);

    size_t counterDataPrefixSize = 0;
    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = { NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE };
    getCounterDataPrefixParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
    getCounterDataPrefixParams.bytesAllocated = 0;
    getCounterDataPrefixParams.pBuffer = NULL;
    LIKWID_NVPW_API_CALL((*NVPW_CounterDataBuilder_GetCounterDataPrefixPtr)(&getCounterDataPrefixParams), ierr = -1; goto nvmon_perfworks_createCounterDataPrefixImage_out);

    cdp = malloc(getCounterDataPrefixParams.bytesCopied);
    if (!cdp)
    {
        ierr = -ENOMEM;
        goto nvmon_perfworks_createCounterDataPrefixImage_out;
    }

    getCounterDataPrefixParams.bytesAllocated = getCounterDataPrefixParams.bytesCopied;
    getCounterDataPrefixParams.pBuffer = &cdp[0];
    LIKWID_NVPW_API_CALL((*NVPW_CounterDataBuilder_GetCounterDataPrefixPtr)(&getCounterDataPrefixParams), free(cdp); ierr = -1; goto nvmon_perfworks_createCounterDataPrefixImage_out);

    *cdpImage = cdp;

nvmon_perfworks_createCounterDataPrefixImage_out:
    LIKWID_NVPW_API_CALL((*NVPW_CounterDataBuilder_DestroyPtr)(&counterDataBuilderDestroyParams), ierr = -1;);
    LIKWID_NVPW_API_CALL((*NVPW_MetricsContext_DestroyPtr)(&metricsContextDestroyParams), ierr = -1);
    return ierr;
}


int
nvmon_perfworks_addEventSets(NvmonDevice_t device, const char* eventString)
{
    int i = 0, j = 0;
    int curDeviceId = -1;
    CUcontext curContext;
    struct bstrList* tmp, *eventtokens = NULL;
    int gid = 0;

    LIKWID_CUDA_API_CALL((*cudaGetDevicePtr)(&curDeviceId), return -EFAULT);
    LIKWID_CUDA_API_CALL((*cudaFreePtr)(NULL), return -EFAULT);
    LIKWID_CU_CALL((*cuCtxGetCurrentPtr)(&curContext), return -EFAULT);

    if (curDeviceId != device->deviceId)
    {
        LIKWID_CUDA_API_CALL((*cudaSetDevicePtr)(device->deviceId), return -EFAULT);
    }

    int popContext = check_nv_context(device, curContext);

    bstring eventBString = bfromcstr(eventString);
    tmp = bsplit(eventBString, ',');
    bdestroy(eventBString);

    eventtokens = bstrListCreate();

    for (i = 0; i < tmp->qty; i++)
    {
        for (j = 0; j < device->numAllEvents; j++)
        {
            bstring bname = bfromcstr(device->allevents[j]->name);
            if (bstrcmp(tmp->entry[i], bname) == BSTR_OK)
            {
                bstrListAddChar(eventtokens, device->allevents[j]->real);
            }
        }
    }
    bstrListDestroy(tmp);

    

    

    

    bstrListDestroy(eventtokens);
    

    if(popContext)
    {
        LIKWID_CU_CALL((*cuCtxPopCurrentPtr)(&device->context), return -EFAULT);
    }
}

NvmonFunctions nvmon_perfworks_functions = {
    .freeDevice = nvmon_perfworks_freeDevice,
    .createDevice = nvmon_perfworks_createDevice,
    .getEventList = nvmon_perfworks_getEventsOfGpu,
    .addEvents = nvmon_perfworks_addEventSets,
    .setupCounters = NULL,
    .startCounters = NULL,
    .readCounters = NULL,
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
