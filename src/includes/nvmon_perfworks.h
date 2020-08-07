/*
 * =======================================================================================
 *
 *      Filename:  nvmon_perfworks.h
 *
 *      Description:  Header File of nvmon module (PerfWorks backend).
 *
 *      Version:   5.0
 *      Released:  10.11.2019
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

#if CUDA_VERSION >= 10000

static void *perfw_dl_libnvperf = NULL;
static void *perfw_dl_libnvperf_t = NULL;
static void *perfw_dl_libcupti = NULL;
static void *perfw_dl_libcuda = NULL;
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <cuda_runtime_api.h>


#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include<nvperf_target.h>

#define CUPTI_API_CALL(apiFuncCall)                                            \
do {                                                                           \
    CUptiResult _status = apiFuncCall;                                         \
    if (_status != CUPTI_SUCCESS) {                                            \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)



#define NVPERFAPIWEAK __attribute__( ( weak ) )

#define DECLARENVPERFFUNC(funcname, funcsig) NVPA_Status NVPERFAPIWEAK funcname funcsig;  NVPA_Status( *funcname##Ptr ) funcsig;
//DECLARENVPERFFUNC(NVPW_MetricsContext_GetMetricNames_Begin, (NVPW_GetSupportedChipNames_Params *));

DECLARENVPERFFUNC(NVPW_CUDA_MetricsContext_Create, (NVPW_CUDA_MetricsContext_Create_Params *));
DECLARENVPERFFUNC(NVPW_MetricsContext_Destroy, (NVPW_MetricsContext_Destroy_Params*));
DECLARENVPERFFUNC(NVPW_MetricsContext_GetMetricNames_Begin, (NVPW_MetricsContext_GetMetricNames_Begin_Params*));
DECLARENVPERFFUNC(NVPW_MetricsContext_GetMetricNames_End, (NVPW_MetricsContext_GetMetricNames_End_Params*));
DECLARENVPERFFUNC(NVPW_GetSupportedChipNames, (NVPW_GetSupportedChipNames_Params*));
DECLARENVPERFFUNC(NVPW_Device_GetNames, (NVPW_Device_GetNames_Params*))

#ifndef DECLARECUPTIFUNC
#define CUPTIAPIWEAK __attribute__( ( weak ) )
#define DECLARECUPTIFUNC(funcname, funcsig) CUptiResult CUPTIAPIWEAK funcname funcsig;  CUptiResult( *funcname##Ptr ) funcsig;
#endif

DECLARECUPTIFUNC(cuptiDeviceGetChipName, (CUpti_Device_GetChipName_Params*));

#ifndef DECLARECUFUNC
#define CUAPIWEAK __attribute__( ( weak ) )
#define DECLARECUFUNC(funcname, funcsig) CUresult CUAPIWEAK funcname funcsig;  CUresult( *funcname##Ptr ) funcsig;
#endif

DECLARECUFUNC(cuInit, (unsigned int));

#ifndef DLSYM_AND_CHECK
#define DLSYM_AND_CHECK( dllib, name ) dlsym( dllib, name ); if ( dlerror() != NULL ) { return -1; }
#endif

static int
link_perfworks_libraries(void)
{
    if(_dl_non_dynamic_init != NULL) {
        return -1;
    }
    perfw_dl_libcuda = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!perfw_dl_libcuda)
    {
        fprintf(stderr, "CUDA library libcuda.so not found.\n");
        return -1;
    }
    perfw_dl_libnvperf = dlopen("libnvperf_host.so", RTLD_NOW | RTLD_GLOBAL);
    if (!perfw_dl_libnvperf)
    {
        fprintf(stderr, "CUDA library libnvperf_host.so not found.\n");
        return -1;
    }
    perfw_dl_libnvperf_t = dlopen("libnvperf_target.so", RTLD_NOW | RTLD_GLOBAL);
    if (!perfw_dl_libnvperf_t)
    {
        fprintf(stderr, "CUDA library libnvperf_target.so not found.\n");
        return -1;
    }
    perfw_dl_libcupti = dlopen("libcupti.so", RTLD_NOW | RTLD_GLOBAL);
    if (!perfw_dl_libcupti)
    {
        fprintf(stderr, "CUDA runtime library libcupti.so not found.\n");
        return -1;
    }
    cuInitPtr = DLSYM_AND_CHECK(perfw_dl_libcuda, "cuInit");
    cuptiDeviceGetChipNamePtr = DLSYM_AND_CHECK(perfw_dl_libcupti, "cuptiDeviceGetChipName");
    NVPW_CUDA_MetricsContext_CreatePtr = DLSYM_AND_CHECK(perfw_dl_libnvperf, "NVPW_CUDA_MetricsContext_Create");
    //NVPW_GetSupportedChipNamesPtr = DLSYM_AND_CHECK(perfw_dl_libnvperf, "NVPW_GetSupportedChipNames");
    NVPW_Device_GetNamesPtr = DLSYM_AND_CHECK(perfw_dl_libnvperf_t, "NVPW_Device_GetNames");


    CUresult cuErr = (*cuInitPtr)(0);
    if (cuErr != CUDA_SUCCESS)
    {
        fprintf(stderr, "CUDA cannot be found and initialized (cuInit failed).\n");
        return -ENODEV;
    }
    return 0;
}

static void
release_perfworks_libraries(void)
{
    if (perfw_dl_libnvperf)
    {
        dlclose(perfw_dl_libnvperf);
        perfw_dl_libnvperf = NULL;
    }
    if (perfw_dl_libcupti)
    {
        dlclose(perfw_dl_libcupti);
        perfw_dl_libcupti = NULL;
    }
}

int nvmon_perfworks_getEventsOfGpu(int gpuId, NvmonEventList_t* list)
{
    NVPA_Status err = 0;
    int ret = topology_gpu_init();
    if (ret != EXIT_SUCCESS)
    {
        return -ENODEV;
    }
    GpuTopology_t gtopo = get_gpuTopology();
    ret = link_perfworks_libraries();
    if (ret < 0)
    {
        return ret;
    }


    NVPW_Device_GetNames_Params getNamesParams = {NVPW_Device_GetNames_Params_STRUCT_SIZE};
    err = (*NVPW_Device_GetNamesPtr)(&getNamesParams);
    if (err != NVPA_STATUS_SUCCESS)
    {
        fprintf(stderr, "Cannot get name of GPU chip %d\n", err);
        return -ENODEV;
    }
    printf("GPU name %s\n", getNamesParams.pChipName);
    printf("GPU name %s\n", getNamesParams.pDeviceName);

/*    */
/*    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };*/
/*    getChipNameParams.pPriv = NULL;*/
/*    getChipNameParams.deviceIndex = 0;*/
/*    */
/*    */
/*    */
/*    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };*/
/*    metricsContextCreateParams.pChipName = getChipNameParams.pChipName;*/
/*    err = (*NVPW_CUDA_MetricsContext_CreatePtr)(&metricsContextCreateParams);*/
/*    printf("Here\n");*/
/*    if (err != NVPA_STATUS_SUCCESS)*/
/*    {*/
/*        printf("Failed to get event list of device %d\n", gpuId);*/
/*        return -ENODEV;*/
/*    }*/
/*    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };*/
/*    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;*/
/*    err = myNVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams);*/
/*    */
/*    NVPW_MetricsContext_GetMetricNames_Begin_Params getMetricNameBeginParams = { NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE };*/
/*    getMetricNameBeginParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;*/
/*    getMetricNameBeginParams.hidePeakSubMetrics = 0;*/
/*    getMetricNameBeginParams.hidePerCycleSubMetrics = 0;*/
/*    getMetricNameBeginParams.hidePctOfPeakSubMetrics = 0;*/
/*    err = myNVPW_MetricsContext_GetMetricNames_Begin(&getMetricNameBeginParams);*/
/*    if (err != NVPA_STATUS_SUCCESS)*/
/*    {*/
/*        printf("Failed to get event list of device %d\n", gpuId);*/
/*        return -ENODEV;*/
/*    }*/
/*    NVPW_MetricsContext_GetMetricNames_End_Params getMetricNameEndParams = { NVPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE };*/
/*    getMetricNameEndParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;*/
/*    err = myNVPW_MetricsContext_GetMetricNames_End((NVPW_MetricsContext_GetMetricNames_End_Params *)&getMetricNameEndParams);*/
/*    */
/*    */
/*    NvmonEventList_t l = malloc(sizeof(NvmonEventList));*/
/*    if (l)*/
/*    {*/
/*        */
/*        l->events = malloc(l->numEvents*sizeof(NvmonEventListEntry));*/
/*        if (l->events)*/
/*        {*/
/*            for (int i = 0; i < getMetricNameBeginParams.numMetrics; i++)*/
/*            {*/
/*                NvmonEventListEntry* out = &l->events[i];*/
/*                const char* mname = getMetricNameBeginParams.ppMetricNames[i];*/
/*                out->name = malloc(strlen(mname)+2);*/
/*                if (out->name)*/
/*                {*/
/*                    ret = snprintf(out->name, strlen(mname)+1, "%s", mname);*/
/*                    if (ret > 0)*/
/*                    {*/
/*                        out->name[ret] = '\0';*/
/*                    }*/
/*                }*/
/*                out->limit = malloc(10*sizeof(char));*/
/*                if (out->limit)*/
/*                {*/
/*                    ret = snprintf(out->limit, 9, "GPU");*/
/*                    if (ret > 0) out->limit[ret] = '\0';*/
/*                }*/
/*            }*/
/*            l->numEvents = getMetricNameBeginParams.numMetrics;*/
/*            *list = l;*/
/*            return 0;*/
/*        }*/
/*        free(l);*/
/*    }*/
/*    return -ENOMEM;*/
    return 0;
}



NvmonFunctions nvmon_perfworks_functions = {
    .freeDevice = NULL,
    .createDevice = NULL,
    .getEventList = nvmon_perfworks_getEventsOfGpu,
    .addEvents = NULL,
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
