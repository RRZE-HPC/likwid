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
 *                Michael Panzlaff, michael.panzlaff@fau.de
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
#ifndef LIKWID_NVMON_PERFWORKS_H
#define LIKWID_NVMON_PERFWORKS_H

#include <assert.h>

#if defined(CUDART_VERSION) && CUDART_VERSION > 10000

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>

#include <nvperf_cuda_host.h>
#include <nvperf_host.h>
#include <nvperf_target.h>

static void *dl_perfworks_libcuda = NULL;
static void *dl_libhost = NULL;
static void *dl_cupti = NULL;
static void *dl_perfworks_libcudart = NULL;

#define CUDA_CALL(func, ...)                            \
    do {                                                \
        cudaError_t s_ = (*func##_ptr)(__VA_ARGS__);    \
        if (s_ != cudaSuccess) {                        \
            ERROR_PRINT("Error: function %s failed with error: '%s' (cudaError_t=%d).", #func, cudaGetErrorString_ptr(s_), s_);   \
            return -EPERM;                              \
        }                                               \
    } while (0)

#ifndef CU_CALL
#define CU_CALL(func, ...)                            \
    do {                                                \
        CUresult s_ = (*func##_ptr)(__VA_ARGS__);    \
        if (s_ != CUDA_SUCCESS) {                        \
            const char *errstr = NULL;\
            cuGetErrorString_ptr(s_, &errstr);\
            ERROR_PRINT("Error: function %s failed with error: '%s' (CUresult=%d).", #func, errstr, s_);   \
            return -EPERM;                              \
        }                                               \
    } while (0)
#endif

#ifndef NVPW_CALL
#define NVPW_CALL(handleerror, func, ...)                 \
    do {                                                \
        NVPA_Status s_ = (*func##_ptr)(__VA_ARGS__);    \
        if (s_ != NVPA_STATUS_SUCCESS) {                        \
            ERROR_PRINT("Error: function %s failed with error: %d.", #func, s_);   \
            handleerror;                              \
        }                                               \
    } while (0)
#endif

#ifndef CUPTI_CALL
#define CUPTI_CALL(func, ...)                            \
    do {                                                \
        CUptiResult s_ = (*func##_ptr)(__VA_ARGS__);    \
        if (s_ != CUPTI_SUCCESS) {                        \
            const char *errstr = NULL; \
            cuptiGetResultString_ptr(s, &errstr); \
            ERROR_PRINT("Error: function %s failed with error: '%s' (CUptiResult=%d)", #func, errstr, s_);   \
            return -EPERM;                              \
        }                                               \
    } while (0)
#endif

/* This definitions are used for CUDA 10.1 */
#if defined(CUDART_VERSION) && CUDART_VERSION < 11000
typedef struct CUpti_Profiler_GetCounterAvailability_Params {
  size_t structSize;
  void *pPriv;
  CUcontext ctx;
  size_t counterAvailabilityImageSize;
  uint8_t *pCounterAvailabilityImage;
} CUpti_Profiler_GetCounterAvailability_Params;
#define CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE               \
  sizeof(CUpti_Profiler_GetCounterAvailability_Params)

CUptiResult cuptiProfilerGetCounterAvailability(
    CUpti_Profiler_GetCounterAvailability_Params *params) {
  return CUPTI_SUCCESS;
}

typedef struct {
  size_t structSize;
  void *pPriv;
  NVPA_RawMetricsConfig *pRawMetricsConfig;
  uint8_t *pCounterAvailabilityImage;
} NVPW_RawMetricsConfig_SetCounterAvailability_Params;
#define NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE        \
  sizeof(NVPW_RawMetricsConfig_SetCounterAvailability_Params)

NVPA_Status NVPW_RawMetricsConfig_SetCounterAvailability(
    NVPW_RawMetricsConfig_SetCounterAvailability_Params *params) {
  return NVPA_STATUS_SUCCESS;
}
#endif /* End of definitions for CUDA 10.1 */

/* struct size definitions for older CUDA versions (suffix = version). */

// We cannot define GetChipName_Params via last member. There is no way there
// parameters could fit. Perhaps the old STRUCT_SIZE definition was malformed.
#define CUpti_Device_GetChipName_Params_STRUCT_SIZE10 16
#define CUpti_Device_GetChipName_Params_STRUCT_SIZE11 32

#define CUpti_Profiler_SetConfig_Params_STRUCT_SIZE10 \
    NVPA_STRUCT_SIZE(CUpti_Profiler_SetConfig_Params, passIndex)
#define CUpti_Profiler_SetConfig_Params_STRUCT_SIZE11 \
    NVPA_STRUCT_SIZE(CUpti_Profiler_SetConfig_Params, targetNestingLevel)

#define CUpti_Profiler_EndPass_Params_STRUCT_SIZE10 \
    NVPA_STRUCT_SIZE(CUpti_Profiler_EndPass_Params, ctx)
#define CUpti_Profiler_EndPass_Params_STRUCT_SIZE11 \
    NVPA_STRUCT_SIZE(CUpti_Profiler_EndPass_Params, allPassesSubmitted)

#define CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE10 \
    NVPA_STRUCT_SIZE(CUpti_Profiler_FlushCounterData_Params, ctx)
#define CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE11 \
    NVPA_STRUCT_SIZE(CUpti_Profiler_FlushCounterData_Params, numTraceBytesDropped)

#if CUDART_VERSION >= 11040
typedef struct {
  size_t structSize;
  NVPA_ActivityKind activityKind;
  const char *pChipName;
  const uint8_t *pCounterAvailabilityImage;
} NVPA_RawMetricsConfigOptions;
#define NVPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE 1
#else
/*
 * Copies from CUDA 11.4
 */

typedef struct NVPW_MetricsEvaluator {
} NVPW_MetricsEvaluator;

typedef struct NVPW_MetricEvalRequest {
  size_t metricIndex;
  uint8_t metricType;
  uint8_t rollupOp;
  uint16_t submetric;
} NVPW_MetricEvalRequest;
#define NVPW_MetricEvalRequest_STRUCT_SIZE 1

typedef struct {
  size_t structSize;
  const char *pChipName;
  const uint8_t *pCounterAvailabilityImage;
  size_t scratchBufferSize;
} NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params;
#define NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE \
  1

typedef struct {
  size_t structSize;
  uint8_t *pScratchBuffer;
  size_t scratchBufferSize;
  const char *pChipName;
  const uint8_t *pCounterAvailabilityImage;
  const uint8_t *pCounterDataImage;
  size_t counterDataImageSize;
  struct NVPW_MetricsEvaluator *pMetricsEvaluator;
} NVPW_CUDA_MetricsEvaluator_Initialize_Params;
#define NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE 1

typedef struct {
  size_t structSize;
  struct NVPW_MetricsEvaluator *pMetricsEvaluator;
  const char *pMetricName;
  struct NVPW_MetricEvalRequest *pMetricEvalRequest;
  size_t metricEvalRequestStructSize;
} NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params;
#define NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE \
  1

typedef struct {
  size_t structSize;
  struct NVPW_MetricsEvaluator *pMetricsEvaluator;
  const struct NVPW_MetricEvalRequest *pMetricEvalRequests;
  size_t numMetricEvalRequests;
  size_t metricEvalRequestStructSize;
  size_t metricEvalRequestStrideSize;
  const char **ppRawDependencies;
  size_t numRawDependencies;
} NVPW_MetricsEvaluator_GetMetricRawDependencies_Params;
#define NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE 1

typedef struct {
  size_t structSize;
  struct NVPW_MetricsEvaluator *pMetricsEvaluator;
} NVPW_MetricsEvaluator_Destroy_Params;
#define NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE 1

typedef struct {
  size_t structSize;
  NVPA_ActivityKind activityKind;
  const char *pChipName;
  const uint8_t *pCounterAvailabilityImage;
  struct NVPA_RawMetricsConfig *pRawMetricsConfig;
} NVPW_CUDA_RawMetricsConfig_Create_V2_Params;

#define NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE 1

#endif

#if defined(CUDART_VERSION) && CUDART_VERSION < 11060
typedef struct NVPW_MetricsEvaluator_EvaluateToGpuValues_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        struct NVPW_MetricsEvaluator* pMetricsEvaluator;
        /// [in]
        const struct NVPW_MetricEvalRequest* pMetricEvalRequests;
        /// [in]
        size_t numMetricEvalRequests;
        /// [in] set to 'NVPW_MetricEvalRequest_STRUCT_SIZE'
        size_t metricEvalRequestStructSize;
        /// [in] set to sizeof('NVPW_MetricEvalRequest')
        size_t metricEvalRequestStrideSize;
        /// [in]
        const uint8_t* pCounterDataImage;
        /// [in]
        size_t counterDataImageSize;
        /// [in]
        size_t rangeIndex;
        /// [in]
        NVPA_Bool isolated;
        /// [inout] 'pMetricValues' is in, '*pMetricValues' is out
        double* pMetricValues;
    } NVPW_MetricsEvaluator_EvaluateToGpuValues_Params;
#define NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_MetricsEvaluator_EvaluateToGpuValues_Params, pMetricValues)
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12060
typedef struct NVPA_MetricsContext NVPA_MetricsContext;
typedef struct NVPW_CUDA_MetricsContext_Create_Params
{
    /// [in]
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    /// [in]
    const char* pChipName;
    /// [out]
    struct NVPA_MetricsContext* pMetricsContext;
} NVPW_CUDA_MetricsContext_Create_Params;
#define NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_CUDA_MetricsContext_Create_Params, pMetricsContext)
typedef struct NVPW_MetricsContext_Destroy_Params
{
    /// [in]
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    NVPA_MetricsContext* pMetricsContext;
} NVPW_MetricsContext_Destroy_Params;
#define NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_MetricsContext_Destroy_Params, pMetricsContext)

typedef struct NVPW_MetricsContext_GetMetricNames_Begin_Params
{
    /// [in]
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    NVPA_MetricsContext* pMetricsContext;
    /// out: number of elements in array ppMetricNames
    size_t numMetrics;
    /// out: pointer to array of 'const char* pMetricName'
    const char* const* ppMetricNames;
    /// in : if true, doesn't enumerate \<metric\>.peak_{burst, sustained}
    NVPA_Bool hidePeakSubMetrics;
    /// in : if true, doesn't enumerate \<metric\>.per_{active,elapsed,region,frame}_cycle
    NVPA_Bool hidePerCycleSubMetrics;
    /// in : if true, doesn't enumerate \<metric\>.pct_of_peak_{burst,sustained}_{active,elapsed,region,frame}
    NVPA_Bool hidePctOfPeakSubMetrics;
    /// in : if false, enumerate \<unit\>__throughput.pct_of_peak_sustained_elapsed even if hidePctOfPeakSubMetrics
    /// is true
    NVPA_Bool hidePctOfPeakSubMetricsOnThroughputs;
} NVPW_MetricsContext_GetMetricNames_Begin_Params;
#define NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetMetricNames_Begin_Params, hidePctOfPeakSubMetricsOnThroughputs)

typedef struct NVPW_MetricsContext_GetMetricNames_End_Params
{
    /// [in]
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    NVPA_MetricsContext* pMetricsContext;
} NVPW_MetricsContext_GetMetricNames_End_Params;
#define NVPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetMetricNames_End_Params, pMetricsContext)

typedef struct NVPW_MetricsContext_GetMetricProperties_Begin_Params
{
    /// [in]
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    NVPA_MetricsContext* pMetricsContext;
    const char* pMetricName;
    /// out
    const char* pDescription;
    /// out
    const char* pDimUnits;
    /// out: a NULL-terminated array of pointers to RawMetric names that can be passed to
    /// NVPW_RawMetricsConfig_AddMetrics()
    const char** ppRawMetricDependencies;
    /// out: metric.peak_burst.value.gpu
    double gpuBurstRate;
    /// out: metric.peak_sustained.value.gpu
    double gpuSustainedRate;
    /// out: a NULL-terminated array of pointers to RawMetric names that can be passed to
    /// NVPW_RawMetricsConfig_AddMetrics().
    const char** ppOptionalRawMetricDependencies;
} NVPW_MetricsContext_GetMetricProperties_Begin_Params;
#define NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetMetricProperties_Begin_Params, ppOptionalRawMetricDependencies)

typedef struct NVPW_MetricsContext_GetMetricProperties_End_Params
{
    /// [in]
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    NVPA_MetricsContext* pMetricsContext;
} NVPW_MetricsContext_GetMetricProperties_End_Params;
#define NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_MetricsContext_GetMetricProperties_End_Params, pMetricsContext)

typedef struct NVPW_MetricsContext_SetCounterData_Params
{
    /// [in]
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    NVPA_MetricsContext* pMetricsContext;
    const uint8_t* pCounterDataImage;
    size_t rangeIndex;
    NVPA_Bool isolated;
} NVPW_MetricsContext_SetCounterData_Params;
#define NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_MetricsContext_SetCounterData_Params, isolated)

typedef struct NVPW_MetricsContext_EvaluateToGpuValues_Params
{
    /// [in]
    size_t structSize;
    /// [in] assign to NULL
    void* pPriv;
    NVPA_MetricsContext* pMetricsContext;
    size_t numMetrics;
    const char* const* ppMetricNames;
    /// [out]
    double* pMetricValues;
} NVPW_MetricsContext_EvaluateToGpuValues_Params;
#define NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE NVPA_STRUCT_SIZE(NVPW_MetricsContext_EvaluateToGpuValues_Params, pMetricValues)
#endif

//we cannot use Nvidia's header at the moment, because of a bug in CUDA 12.6,
//which prevents cupti_profiler_host.h to be used in plain C. Caused by an
//#include <string> (outside of the __cplusplus guard)
//
//#if defined(CUDART_VERSION) && CUDART_VERSION < 12060
/*
 * Copies from CUDA 12.6
 */

typedef enum CUpti_MetricType
{
    CUPTI_METRIC_TYPE_COUNTER = 0,
    CUPTI_METRIC_TYPE_RATIO,
    CUPTI_METRIC_TYPE_THROUGHPUT,
    CUPTI_METRIC_TYPE__COUNT
} CUpti_MetricType;

typedef enum CUpti_ProfilerType
{
    CUPTI_PROFILER_TYPE_RANGE_PROFILER,
    CUPTI_PROFILER_TYPE_PM_SAMPLING,
    CUPTI_PROFILER_TYPE_PROFILER_INVALID
} CUpti_ProfilerType;

typedef struct CUpti_Profiler_Host_Object CUpti_Profiler_Host_Object;

typedef struct CUpti_Profiler_Host_Initialize_Params
{
    size_t structSize;
    void* pPriv;
    CUpti_ProfilerType profilerType;
    const char* pChipName;
    const uint8_t* pCounterAvailabilityImage;
    CUpti_Profiler_Host_Object* pHostObject;
} CUpti_Profiler_Host_Initialize_Params;
#define CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_Initialize_Params, pHostObject)

typedef struct CUpti_Profiler_Host_Deinitialize_Params
{
    size_t structSize;
    void* pPriv;
    struct CUpti_Profiler_Host_Object* pHostObject;
} CUpti_Profiler_Host_Deinitialize_Params;
#define CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_Deinitialize_Params, pHostObject)

typedef struct CUpti_Profiler_Host_GetBaseMetrics_Params
{
    size_t structSize;
    void* pPriv;
    struct CUpti_Profiler_Host_Object* pHostObject;
    CUpti_MetricType metricType;
    const char** ppMetricNames;
    size_t numMetrics;
} CUpti_Profiler_Host_GetBaseMetrics_Params;
#define CUpti_Profiler_Host_GetBaseMetrics_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetBaseMetrics_Params, numMetrics)

typedef struct CUpti_Profiler_Host_GetSubMetrics_Params
{
    size_t structSize;
    void* pPriv;
    CUpti_Profiler_Host_Object* pHostObject;
    CUpti_MetricType metricType;
    const char* pMetricName;
    size_t numOfSubmetrics;
    const char** ppSubMetrics;
} CUpti_Profiler_Host_GetSubMetrics_Params;
#define CUpti_Profiler_Host_GetSubMetrics_Params_STRUCT_SIZE CUPTI_PROFILER_STRUCT_SIZE(CUpti_Profiler_Host_GetSubMetrics_Params, ppSubMetrics)

//
//#else
//#include <cupti_profiler_host.h>
//#endif

#ifndef DECLARE_CUFUNC
#define DECLARE_CUFUNC(funcname, ...) CUresult __attribute__((weak)) funcname(__VA_ARGS__);  static CUresult (*funcname##_ptr)(__VA_ARGS__);
#endif

DECLARE_CUFUNC(cuCtxGetCurrent, CUcontext *);
DECLARE_CUFUNC(cuCtxSetCurrent, CUcontext);
DECLARE_CUFUNC(cuCtxDestroy, CUcontext);
DECLARE_CUFUNC(cuDeviceGet, CUdevice *, int);
DECLARE_CUFUNC(cuDeviceGetCount, int *);
DECLARE_CUFUNC(cuDeviceGetName, char *, int, CUdevice);
DECLARE_CUFUNC(cuInit, unsigned int);
DECLARE_CUFUNC(cuCtxPopCurrent, CUcontext * pctx);
DECLARE_CUFUNC(cuCtxPushCurrent, CUcontext pctx);
DECLARE_CUFUNC(cuCtxSynchronize);
DECLARE_CUFUNC(cuDeviceGetAttribute, int *, CUdevice_attribute, CUdevice);
DECLARE_CUFUNC(cuCtxCreate, CUcontext *, unsigned int, CUdevice);
DECLARE_CUFUNC(cuDevicePrimaryCtxRetain, CUcontext *, CUdevice);

#ifndef DECLARE_CUDAFUNC
#define DECLARE_CUDAFUNC(funcname, ...) cudaError_t __attribute__((weak)) funcname(__VA_ARGS__);  static cudaError_t (*funcname##_ptr)(__VA_ARGS__);
#endif
DECLARE_CUDAFUNC(cudaGetDevice, int *);
DECLARE_CUDAFUNC(cudaSetDevice, int);
DECLARE_CUDAFUNC(cudaFree, void *);
DECLARE_CUDAFUNC(cudaDriverGetVersion, int *);
DECLARE_CUDAFUNC(cudaRuntimeGetVersion, int *);

#ifndef DECLARE_NVPWFUNC
#define DECLARE_NVPWFUNC(funcname, ...) NVPA_Status __attribute__((weak)) funcname(__VA_ARGS__);  static NVPA_Status (*funcname##_ptr)(__VA_ARGS__);
#endif

DECLARE_NVPWFUNC(NVPW_GetSupportedChipNames,
                 NVPW_GetSupportedChipNames_Params * params);
DECLARE_NVPWFUNC(NVPW_CUDA_MetricsContext_Create,
                 NVPW_CUDA_MetricsContext_Create_Params * params);
DECLARE_NVPWFUNC(NVPW_MetricsContext_Destroy,
                 NVPW_MetricsContext_Destroy_Params * params);
DECLARE_NVPWFUNC(NVPW_MetricsContext_GetMetricNames_Begin,
                 NVPW_MetricsContext_GetMetricNames_Begin_Params * params);
DECLARE_NVPWFUNC(NVPW_MetricsContext_GetMetricNames_End,
                 NVPW_MetricsContext_GetMetricNames_End_Params * params);
DECLARE_NVPWFUNC(NVPW_InitializeHost, NVPW_InitializeHost_Params * params);
DECLARE_NVPWFUNC(NVPW_MetricsContext_GetMetricProperties_Begin,
                 NVPW_MetricsContext_GetMetricProperties_Begin_Params * p);
DECLARE_NVPWFUNC(NVPW_MetricsContext_GetMetricProperties_End,
                 NVPW_MetricsContext_GetMetricProperties_End_Params * p);
DECLARE_NVPWFUNC(NVPW_CUDA_RawMetricsConfig_Create,
                 NVPW_CUDA_RawMetricsConfig_Create_Params *);
DECLARE_NVPWFUNC(NVPW_RawMetricsConfig_Destroy,
                 NVPW_RawMetricsConfig_Destroy_Params * params);
DECLARE_NVPWFUNC(NVPW_RawMetricsConfig_BeginPassGroup,
                 NVPW_RawMetricsConfig_BeginPassGroup_Params * params);
DECLARE_NVPWFUNC(NVPW_RawMetricsConfig_EndPassGroup,
                 NVPW_RawMetricsConfig_EndPassGroup_Params * params);
DECLARE_NVPWFUNC(NVPW_RawMetricsConfig_AddMetrics,
                 NVPW_RawMetricsConfig_AddMetrics_Params * params);
DECLARE_NVPWFUNC(NVPW_RawMetricsConfig_GenerateConfigImage,
                 NVPW_RawMetricsConfig_GenerateConfigImage_Params * params);
DECLARE_NVPWFUNC(NVPW_RawMetricsConfig_GetConfigImage,
                 NVPW_RawMetricsConfig_GetConfigImage_Params * params);
DECLARE_NVPWFUNC(NVPW_CounterDataBuilder_Create,
                 NVPW_CounterDataBuilder_Create_Params * params);
DECLARE_NVPWFUNC(NVPW_CounterDataBuilder_Destroy,
                 NVPW_CounterDataBuilder_Destroy_Params * params);
DECLARE_NVPWFUNC(NVPW_CounterDataBuilder_AddMetrics,
                 NVPW_CounterDataBuilder_AddMetrics_Params * params);
DECLARE_NVPWFUNC(NVPW_CounterDataBuilder_GetCounterDataPrefix,
                 NVPW_CounterDataBuilder_GetCounterDataPrefix_Params *
                  params);
DECLARE_NVPWFUNC(NVPW_CounterData_GetNumRanges,
                 NVPW_CounterData_GetNumRanges_Params * params);
DECLARE_NVPWFUNC(NVPW_Profiler_CounterData_GetRangeDescriptions,
                 NVPW_Profiler_CounterData_GetRangeDescriptions_Params *
                  params);
DECLARE_NVPWFUNC(NVPW_MetricsContext_SetCounterData,
                 NVPW_MetricsContext_SetCounterData_Params * params);
DECLARE_NVPWFUNC(NVPW_MetricsContext_EvaluateToGpuValues,
                 NVPW_MetricsContext_EvaluateToGpuValues_Params * params);
DECLARE_NVPWFUNC(NVPW_RawMetricsConfig_GetNumPasses,
                 NVPW_RawMetricsConfig_GetNumPasses_Params * params);
DECLARE_NVPWFUNC(NVPW_RawMetricsConfig_SetCounterAvailability,
                 NVPW_RawMetricsConfig_SetCounterAvailability_Params *
                  params);

DECLARE_NVPWFUNC(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize,
                 NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params *
                  params);
DECLARE_NVPWFUNC(NVPW_CUDA_MetricsEvaluator_Initialize,
                 NVPW_CUDA_MetricsEvaluator_Initialize_Params * params);
DECLARE_NVPWFUNC(
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest,
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params *
     params);
DECLARE_NVPWFUNC(NVPW_MetricsEvaluator_GetMetricRawDependencies,
                 NVPW_MetricsEvaluator_GetMetricRawDependencies_Params *
                  params);
DECLARE_NVPWFUNC(NVPW_MetricsEvaluator_EvaluateToGpuValues,
                 NVPW_MetricsEvaluator_EvaluateToGpuValues_Params * params);
DECLARE_NVPWFUNC(NVPW_MetricsEvaluator_Destroy,
                 NVPW_MetricsEvaluator_Destroy_Params * params);
DECLARE_NVPWFUNC(NVPW_CUDA_RawMetricsConfig_Create_V2,
                 NVPW_CUDA_RawMetricsConfig_Create_V2_Params * params);
DECLARE_NVPWFUNC(NVPA_RawMetricsConfig_Create, const NVPA_RawMetricsConfigOptions*, NVPA_RawMetricsConfig**);

#ifndef DECLARE_CUPTIFUNC
#define DECLARE_CUPTIFUNC(funcname, ...) CUptiResult __attribute__((weak)) funcname(__VA_ARGS__);  static CUptiResult (*funcname##_ptr)(__VA_ARGS__);
#endif
DECLARE_CUPTIFUNC(cuptiDeviceGetChipName,
                  CUpti_Device_GetChipName_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerInitialize,
                  CUpti_Profiler_Initialize_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerDeInitialize,
                  CUpti_Profiler_DeInitialize_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerCounterDataImageCalculateSize,
                  CUpti_Profiler_CounterDataImage_CalculateSize_Params *
                   params);
DECLARE_CUPTIFUNC(cuptiProfilerCounterDataImageInitialize,
                  CUpti_Profiler_CounterDataImage_Initialize_Params * params);
DECLARE_CUPTIFUNC(
    cuptiProfilerCounterDataImageCalculateScratchBufferSize,
    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params *
     params);
DECLARE_CUPTIFUNC(
    cuptiProfilerCounterDataImageInitializeScratchBuffer,
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params * params);

DECLARE_CUPTIFUNC(cuptiProfilerBeginSession,
                  CUpti_Profiler_BeginSession_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerSetConfig,
                  CUpti_Profiler_SetConfig_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerBeginPass,
                  CUpti_Profiler_BeginPass_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerEnableProfiling,
                  CUpti_Profiler_EnableProfiling_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerPushRange,
                  CUpti_Profiler_PushRange_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerPopRange,
                  CUpti_Profiler_PopRange_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerDisableProfiling,
                  CUpti_Profiler_DisableProfiling_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerEndPass,
                  CUpti_Profiler_EndPass_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerFlushCounterData,
                  CUpti_Profiler_FlushCounterData_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerUnsetConfig,
                  CUpti_Profiler_UnsetConfig_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerEndSession,
                  CUpti_Profiler_EndSession_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerGetCounterAvailability,
                  CUpti_Profiler_GetCounterAvailability_Params * params);
DECLARE_CUPTIFUNC(cuptiGetResultString, CUptiResult result, const char **str);
DECLARE_CUPTIFUNC(cuptiProfilerHostInitialize,
                  CUpti_Profiler_Host_Initialize_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerHostDeinitialize,
                  CUpti_Profiler_Host_Deinitialize_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerHostGetBaseMetrics,
                  CUpti_Profiler_Host_GetBaseMetrics_Params * params);
DECLARE_CUPTIFUNC(cuptiProfilerHostGetSubMetrics,
                  CUpti_Profiler_Host_GetSubMetrics_Params * params);

#ifndef DLSYM_AND_CHECK
#define DLSYM_AND_CHECK(dllib, name)                                           \
  dlsym(dllib, name);                                                          \
  if (dlerror() != NULL) {                                                     \
    return -1;                                                                 \
  }
#endif

//static int cuptiProfiler_initialized = 0;
static int cuda_runtime_version = 0;
static int cuda_version = 0;

static int link_perfworks_libraries(void) {
    /* Attempt to guess if we were statically linked to libc, if so bail */
    if (_dl_non_dynamic_init != NULL) {
        return -1;
    }
    char libcudartpath[1024];
    char libnvperfpath[1024];
    char libcuptipath[1024];
    char *cudahome = getenv("CUDA_HOME");
    if (cudahome != NULL) {
        snprintf(libcudartpath, sizeof(libcudartpath), "%s/lib64/libcudart.so", cudahome);
        snprintf(libnvperfpath, sizeof(libnvperfpath),
                "%s/extras/CUPTI/lib64/libnvperf_host.so", cudahome);
        snprintf(libcuptipath, sizeof(libcuptipath), "%s/extras/CUPTI/lib64/libcupti.so",
                cudahome);
    } else {
        snprintf(libcudartpath, sizeof(libcudartpath), "libcudart.so");
        snprintf(libnvperfpath, sizeof(libnvperfpath), "libnvperf_host.so");
        snprintf(libcuptipath, sizeof(libcuptipath), "libcupti.so");
    }
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "LD_LIBRARY_PATH = %s",
            getenv("LD_LIBRARY_PATH"));
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "CUDA_HOME = %s", getenv("CUDA_HOME"));
    dl_perfworks_libcuda = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_perfworks_libcuda || dlerror() != NULL) {
        DEBUG_PRINT(DEBUGLEV_INFO, "CUDA library libcuda.so not found");
        return -1;
    }
    cuCtxGetCurrent_ptr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuCtxGetCurrent");
    cuCtxSetCurrent_ptr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuCtxSetCurrent");
    cuDeviceGet_ptr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuDeviceGet");
    cuDeviceGetCount_ptr =
        DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuDeviceGetCount");
    cuDeviceGetName_ptr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuDeviceGetName");
    cuInit_ptr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuInit");
    cuCtxPopCurrent_ptr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuCtxPopCurrent");
    cuCtxPushCurrent_ptr =
        DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuCtxPushCurrent");
    cuCtxSynchronize_ptr =
        DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuCtxSynchronize");
    cuCtxDestroy_ptr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuCtxDestroy");
    cuDeviceGetAttribute_ptr =
        DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuDeviceGetAttribute");
    cuCtxCreate_ptr = DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuCtxCreate");
    cuDevicePrimaryCtxRetain_ptr =
        DLSYM_AND_CHECK(dl_perfworks_libcuda, "cuDevicePrimaryCtxRetain");

    dl_perfworks_libcudart =
        dlopen(libcudartpath, RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if ((!dl_perfworks_libcudart) || (dlerror() != NULL)) {
        DEBUG_PRINT(DEBUGLEV_INFO, "CUDA library libcudart.so not found");
        return -1;
    }
    cudaGetDevice_ptr = DLSYM_AND_CHECK(dl_perfworks_libcudart, "cudaGetDevice");
    cudaSetDevice_ptr = DLSYM_AND_CHECK(dl_perfworks_libcudart, "cudaSetDevice");
    cudaFree_ptr = DLSYM_AND_CHECK(dl_perfworks_libcudart, "cudaFree");
    cudaDriverGetVersion_ptr =
        DLSYM_AND_CHECK(dl_perfworks_libcudart, "cudaDriverGetVersion");
    cudaRuntimeGetVersion_ptr =
        DLSYM_AND_CHECK(dl_perfworks_libcudart, "cudaRuntimeGetVersion");

    CUDA_CALL(cudaRuntimeGetVersion, &cuda_runtime_version);

    dl_libhost = dlopen(libnvperfpath, RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if ((!dl_libhost) || (dlerror() != NULL)) {
        dl_libhost =
            dlopen("libnvperf_host.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
        if ((!dl_libhost) || (dlerror() != NULL)) {
            DEBUG_PRINT(DEBUGLEV_INFO, "CUDA library libnvperf_host.so not found");
            return -1;
        }
    }
    NVPW_GetSupportedChipNames_ptr =
        DLSYM_AND_CHECK(dl_libhost, "NVPW_GetSupportedChipNames");
    if (cuda_runtime_version < 12060) {
        NVPW_CUDA_MetricsContext_Create_ptr =
            DLSYM_AND_CHECK(dl_libhost, "NVPW_CUDA_MetricsContext_Create");
        NVPW_MetricsContext_Destroy_ptr =
            DLSYM_AND_CHECK(dl_libhost, "NVPW_MetricsContext_Destroy");
        NVPW_MetricsContext_GetMetricNames_Begin_ptr =
            DLSYM_AND_CHECK(dl_libhost, "NVPW_MetricsContext_GetMetricNames_Begin");
        NVPW_MetricsContext_GetMetricNames_End_ptr =
            DLSYM_AND_CHECK(dl_libhost, "NVPW_MetricsContext_GetMetricNames_End");
    }
    NVPW_InitializeHost_ptr = DLSYM_AND_CHECK(dl_libhost, "NVPW_InitializeHost");

    if (cuda_runtime_version < 12060) {
        NVPW_MetricsContext_GetMetricProperties_Begin_ptr = DLSYM_AND_CHECK(
                dl_libhost, "NVPW_MetricsContext_GetMetricProperties_Begin");
        NVPW_MetricsContext_GetMetricProperties_End_ptr = DLSYM_AND_CHECK(
                dl_libhost, "NVPW_MetricsContext_GetMetricProperties_End");
    }

    NVPW_CUDA_RawMetricsConfig_Create_ptr =
        DLSYM_AND_CHECK(dl_libhost, "NVPW_CUDA_RawMetricsConfig_Create");
    NVPW_RawMetricsConfig_Destroy_ptr =
        DLSYM_AND_CHECK(dl_libhost, "NVPW_RawMetricsConfig_Destroy");
    NVPW_RawMetricsConfig_BeginPassGroup_ptr =
        DLSYM_AND_CHECK(dl_libhost, "NVPW_RawMetricsConfig_BeginPassGroup");
    NVPW_RawMetricsConfig_EndPassGroup_ptr =
        DLSYM_AND_CHECK(dl_libhost, "NVPW_RawMetricsConfig_EndPassGroup");
    NVPW_RawMetricsConfig_AddMetrics_ptr =
        DLSYM_AND_CHECK(dl_libhost, "NVPW_RawMetricsConfig_AddMetrics");
    NVPW_RawMetricsConfig_GenerateConfigImage_ptr =
        DLSYM_AND_CHECK(dl_libhost, "NVPW_RawMetricsConfig_GenerateConfigImage");
    NVPW_RawMetricsConfig_GetConfigImage_ptr =
        DLSYM_AND_CHECK(dl_libhost, "NVPW_RawMetricsConfig_GetConfigImage");
    if (cuda_runtime_version >= 11040) {
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_ptr = DLSYM_AND_CHECK(
                dl_libhost, "NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize");
        NVPW_CUDA_MetricsEvaluator_Initialize_ptr =
            DLSYM_AND_CHECK(dl_libhost, "NVPW_CUDA_MetricsEvaluator_Initialize");
        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_ptr =
            DLSYM_AND_CHECK(
                    dl_libhost,
                    "NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest");
        NVPW_MetricsEvaluator_GetMetricRawDependencies_ptr = DLSYM_AND_CHECK(
                dl_libhost, "NVPW_MetricsEvaluator_GetMetricRawDependencies");
        NVPW_MetricsEvaluator_EvaluateToGpuValues_ptr = DLSYM_AND_CHECK(
                dl_libhost, "NVPW_MetricsEvaluator_EvaluateToGpuValues");
        NVPW_MetricsEvaluator_Destroy_ptr =
            DLSYM_AND_CHECK(dl_libhost, "NVPW_MetricsEvaluator_Destroy");
        NVPW_CUDA_RawMetricsConfig_Create_V2_ptr =
            DLSYM_AND_CHECK(dl_libhost, "NVPW_CUDA_RawMetricsConfig_Create_V2");
    } else {
        NVPA_RawMetricsConfig_Create_ptr = DLSYM_AND_CHECK(dl_libhost, "NVPA_RawMetricsConfig_Create");
    }

    NVPW_CounterDataBuilder_Create_ptr =
        DLSYM_AND_CHECK(dl_libhost, "NVPW_CounterDataBuilder_Create");
    NVPW_CounterDataBuilder_Destroy_ptr =
        DLSYM_AND_CHECK(dl_libhost, "NVPW_CounterDataBuilder_Destroy");
    NVPW_CounterDataBuilder_AddMetrics_ptr =
        DLSYM_AND_CHECK(dl_libhost, "NVPW_CounterDataBuilder_AddMetrics");
    NVPW_CounterDataBuilder_GetCounterDataPrefix_ptr = DLSYM_AND_CHECK(
            dl_libhost, "NVPW_CounterDataBuilder_GetCounterDataPrefix");

    NVPW_CounterData_GetNumRanges_ptr =
        DLSYM_AND_CHECK(dl_libhost, "NVPW_CounterData_GetNumRanges");
    NVPW_Profiler_CounterData_GetRangeDescriptions_ptr = DLSYM_AND_CHECK(
            dl_libhost, "NVPW_Profiler_CounterData_GetRangeDescriptions");
    if (cuda_runtime_version < 12060) {
        NVPW_MetricsContext_SetCounterData_ptr =
            DLSYM_AND_CHECK(dl_libhost, "NVPW_MetricsContext_SetCounterData");
        NVPW_MetricsContext_EvaluateToGpuValues_ptr =
            DLSYM_AND_CHECK(dl_libhost, "NVPW_MetricsContext_EvaluateToGpuValues");
    }
    NVPW_RawMetricsConfig_GetNumPasses_ptr =
        DLSYM_AND_CHECK(dl_libhost, "NVPW_RawMetricsConfig_GetNumPasses");

    dl_cupti = dlopen(libcuptipath, RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if ((!dl_cupti) || (dlerror() != NULL)) {
        dl_cupti = dlopen("libcupti.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
        if ((!dl_cupti) || (dlerror() != NULL)) {
            DEBUG_PRINT(DEBUGLEV_INFO, "CUpti library libcupti.so not found");
            return -1;
        }
    }
    cuptiProfilerInitialize_ptr =
        DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerInitialize");
    cuptiProfilerDeInitialize_ptr =
        DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerDeInitialize");
    cuptiDeviceGetChipName_ptr =
        DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceGetChipName");
    cuptiProfilerCounterDataImageCalculateSize_ptr =
        DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageCalculateSize");
    cuptiProfilerCounterDataImageInitialize_ptr =
        DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageInitialize");
    cuptiProfilerCounterDataImageCalculateScratchBufferSize_ptr = DLSYM_AND_CHECK(
            dl_cupti, "cuptiProfilerCounterDataImageCalculateScratchBufferSize");
    cuptiProfilerCounterDataImageInitializeScratchBuffer_ptr = DLSYM_AND_CHECK(
            dl_cupti, "cuptiProfilerCounterDataImageInitializeScratchBuffer");
    cuptiProfilerBeginSession_ptr =
        DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerBeginSession");
    cuptiProfilerSetConfig_ptr =
        DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerSetConfig");
    cuptiProfilerBeginPass_ptr =
        DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerBeginPass");
    cuptiProfilerEnableProfiling_ptr =
        DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerEnableProfiling");
    cuptiProfilerPushRange_ptr =
        DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerPushRange");
    cuptiProfilerPopRange_ptr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerPopRange");
    cuptiProfilerDisableProfiling_ptr =
        DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerDisableProfiling");
    cuptiProfilerEndPass_ptr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerEndPass");
    cuptiProfilerFlushCounterData_ptr =
        DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerFlushCounterData");
    cuptiProfilerUnsetConfig_ptr =
        DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerUnsetConfig");
    cuptiProfilerEndSession_ptr =
        DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerEndSession");
    cuptiGetResultString_ptr = DLSYM_AND_CHECK(dl_cupti, "cuptiGetResultString");

    if (cuda_runtime_version >= 12060) {
        cuptiProfilerHostInitialize_ptr =
            DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerHostInitialize");
        cuptiProfilerHostGetBaseMetrics_ptr =
            DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerHostGetBaseMetrics");
        cuptiProfilerHostDeinitialize_ptr =
            DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerHostDeinitialize");
        cuptiProfilerHostGetSubMetrics_ptr =
            DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerHostGetSubMetrics");
    }

    dlerror();
    int curDeviceId = -1;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Run cuInit");
    CU_CALL(cuInit, 0);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Run cuDeviceGetCount");
    CU_CALL(cuDeviceGetCount, &curDeviceId);
    // GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Run cudaGetDevice");
    // CUDA_CALL(cudaGetDevice, &curDeviceId);
    CUdevice dev;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Run cuDeviceGet");
    CU_CALL(cuDeviceGet, &dev, 0);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Run cuDeviceGetAttribute for major CC");
    CU_CALL(
            cuDeviceGetAttribute,
                &curDeviceId, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Run cuDeviceGetAttribute for minor CC");
    CU_CALL(
            cuDeviceGetAttribute,
                &curDeviceId, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);

    CUDA_CALL(cudaDriverGetVersion, &cuda_version);
    CUDA_CALL(cudaRuntimeGetVersion, &cuda_runtime_version);

    if (cuda_version >= 11000 && cuda_runtime_version >= 11000) {
        cuptiProfilerGetCounterAvailability_ptr =
            DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerGetCounterAvailability");
        NVPW_RawMetricsConfig_SetCounterAvailability_ptr = DLSYM_AND_CHECK(
                dl_libhost, "NVPW_RawMetricsConfig_SetCounterAvailability");
    } else {
        cuptiProfilerGetCounterAvailability_ptr =
            &cuptiProfilerGetCounterAvailability;
        NVPW_RawMetricsConfig_SetCounterAvailability_ptr =
            &NVPW_RawMetricsConfig_SetCounterAvailability;
    }

    return 0;
}

static int perfworks_check_nv_context(NvmonDevice_t device,
                                      CUcontext currentContext) {
    int need_pop = 0;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Current context %p DevContext %p",
            currentContext, device->context);
    if (!device->context) {
        int context_of_dev = -1;
        for (int j = 0; j < nvGroupSet->numberOfGPUs; j++) {
            NvmonDevice_t dev = &nvGroupSet->gpus[j];
            if (dev->context == currentContext) {
                context_of_dev = j;
                break;
            }
        }
        if (context_of_dev < 0) // && !device->context)
        {
            device->context = currentContext;
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Reuse context %p for device %d", device->context, device->deviceId);
        } else {
            CUDA_CALL(cudaSetDevice, device->deviceId);
            CUDA_CALL(cudaFree, NULL);
            CU_CALL(
                    cuDevicePrimaryCtxRetain, &device->context, device->cuDevice);
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "New context %p for device %d", device->context, device->deviceId);
        }
    } else if (device->context != currentContext) {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Use context %p for device %d", device->context, device->deviceId);
        CU_CALL(cuCtxPushCurrent, device->context);
        need_pop = 1;
    } else {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Context %p fits for device %d", device->context, device->deviceId);
    }
    return need_pop;
}

// Do we still need it? Its references are all commented out.
//static int cuptiProfiler_init() {
//    if (!cuptiProfiler_initialized) {
//        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Init CUpti Profiler");
//        if (dl_perfworks_libcuda == NULL || dl_perfworks_libcudart == NULL ||
//                dl_libhost == NULL || dl_cupti == NULL) {
//            if (link_perfworks_libraries() < 0)
//                return -1;
//        }
//        // CU_CALL(cuInit, 0);
//        // CUdevice dev;
//        // CU_CALL(cuDeviceGet, &dev, 0);
//        CUpti_Profiler_Initialize_Params profilerInitializeParams = {
//            .structSize = CUpti_Profiler_Initialize_Params_STRUCT_SIZE,
//        };
//        CUPTI_CALL(
//                cuptiProfilerInitialize, &profilerInitializeParams);
//        NVPW_InitializeHost_Params initializeHostParams = {
//            .structSize = NVPW_InitializeHost_Params_STRUCT_SIZE
//        };
//        NVPW_CALL(return -1, NVPW_InitializeHost, &initializeHostParams);
//        cuptiProfiler_initialized = 1;
//    }
//    return 0;
//}

// Do we still need this function? It isn't used anymore / code that uses it is commented out.
//static int nvmon_perfworks_parse_metric(char *inoutmetric, int *isolated,
//                                        int *keepInstances) {
//    if (!inoutmetric)
//        return 0;
//    int len = strlen(inoutmetric);
//
//    bstring outmetric = bfromcstr(inoutmetric);
//    int newline = bstrchrp(outmetric, '\n', 0);
//    if (newline != BSTR_ERR) {
//        bdelete(outmetric, newline, 1);
//    }
//    btrimws(outmetric);
//    if (blength(outmetric) > 0) {
//        *keepInstances = 0;
//        if (bchar(outmetric, blength(outmetric) - 1) == '+') {
//            *keepInstances = 1;
//            bdelete(outmetric, blength(outmetric) - 1, 1);
//        }
//        if (blength(outmetric) > 0) {
//            *isolated = 1;
//            if (bchar(outmetric, blength(outmetric) - 1) == '$') {
//                bdelete(outmetric, blength(outmetric) - 1, 1);
//            } else if (bchar(outmetric, blength(outmetric) - 1) == '&') {
//                *isolated = 0;
//                bdelete(outmetric, blength(outmetric) - 1, 1);
//            }
//            if (blength(outmetric) > 0) {
//                snprintf(inoutmetric, len, "%s", bdata(outmetric));
//                bdestroy(outmetric);
//                return 1;
//            }
//        }
//    }
//    return 0;
//}

// static int expand_metric(NVPA_MetricsContext* context, char* inmetric, struct
// bstrList* events)
// {
//     int iso = 0;
//     int keep = 0;
//     nvmon_perfworks_parse_metric(inmetric, &iso, &keep);
//     keep = 1;
//     NVPW_MetricsContext_GetMetricProperties_Begin_Params
//     getMetricPropertiesBeginParams = {
//     NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE };
//     getMetricPropertiesBeginParams.pMetricsContext = context;
//     getMetricPropertiesBeginParams.pMetricName = inmetric;
//     NVPW_CALL(return -1, NVPW_MetricsContext_GetMetricProperties_Begin, &getMetricPropertiesBeginParams);

//     for (char** dep = getMetricPropertiesBeginParams.ppRawMetricDependencies;
//     *dep ; ++dep)
//     {
//         bstrListAddChar(events, *dep);
//     }
//     NVPW_MetricsContext_GetMetricProperties_End_Params
//     getMetricPropertiesEndParams = {
//     NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE };
//     getMetricPropertiesEndParams.pMetricsContext = context;
//     NVPW_CALL(return -1, NVPW_MetricsContext_GetMetricProperties_End, &getMetricPropertiesEndParams); return 0;
// }

static void dev_freeEventList(NvmonDevice *dev) {
    if (!dev->allevents) {
        dev->numAllEvents = 0;
        return;
    }

    for (int i = 0; i < dev->numAllEvents; i++)
        free(dev->allevents[i]);
    free(dev->allevents);

    dev->allevents = NULL;
    dev->numAllEvents = 0;
}

static void nvmon_perfworks_freeDevice(NvmonDevice_t dev) {
    if (dev) {
        free(dev->chip);
        dev->chip = NULL;

        if (dev->nvEventSets) {
            for (int i = 0; i < dev->numNvEventSets; i++) {
                NvmonEventSet *evset = &dev->nvEventSets[i];
                bstrListDestroy(evset->events);
                free(evset->nvEvents);
                evset->nvEvents = NULL;

                free(evset->results);
                evset->results = NULL;

                free(evset->configImage);
                evset->configImage = NULL;
                evset->configImageSize = 0;

                free(evset->counterDataImage);
                evset->counterDataImage = NULL;
                evset->counterDataImageSize = 0;

                free(evset->counterDataScratchBuffer);
                evset->counterDataScratchBuffer = NULL;
                evset->counterDataScratchBufferSize = 0;

                free(evset->counterDataImagePrefix);
                evset->counterDataImagePrefix = NULL;
                evset->counterDataImagePrefixSize = 0;

                free(evset->counterAvailabilityImage);
                evset->counterAvailabilityImage = NULL;
                evset->counterAvailabilityImageSize = 0;
            }
            free(dev->nvEventSets);
            dev->nvEventSets = NULL;
            dev->numNvEventSets = 0;
            dev->activeEventSet = -1;
        }

        if (dev->allevents) {
            int i = 0;
            if (dev->nvEventSets != NULL) {
                for (i = 0; i < dev->numNvEventSets; i++) {
                    NvmonEventSet *ev = &dev->nvEventSets[i];
                    free(ev->results);
                    free(ev->nvEvents);
                }
            }
            dev_freeEventList(dev);
        }
    }
}

static void prepare_metric_name(bstring metric) {
    static const struct tagbstring double_us = bsStatic("__");
    static const struct tagbstring us = bsStatic("_");
    static const struct tagbstring dot = bsStatic(".");

    btrimws(metric);
    int newline = bstrchrp(metric, '\n', 0);
    if (newline != BSTR_ERR)
        bdelete(metric, newline, 1);
    btoupper(metric);

    bfindreplace(metric, &double_us, &us, 0);
    bfindreplace(metric, &dot, &us, 0);
}

static int nvmon_perfworks_getChipName(int id, char **name) {
    /* CUDA 10.1 and CUDA 11.0 use different struct sizes, for cuptiDeviceGetChipName.
     * So use this hacky trick to get the name regardless. */
    size_t getChipNameParams_size;
    if (cuda_runtime_version < 11000)
        getChipNameParams_size = CUpti_Device_GetChipName_Params_STRUCT_SIZE10;
    else
        getChipNameParams_size = CUpti_Device_GetChipName_Params_STRUCT_SIZE11;

    CUpti_Device_GetChipName_Params getChipNameParams = {
        .structSize = getChipNameParams_size,
        .deviceIndex = id,
    };
    CUPTI_CALL(cuptiDeviceGetChipName, &getChipNameParams);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Current GPU chip %s",
                   getChipNameParams.pChipName);
    *name = strdup(getChipNameParams.pChipName);
    if (!name)
        return -errno;
    return 0;
}

static int nvmon_perfworks_appendEventList(NvmonDevice *dev, const char * const *metricNames, size_t numMetrics) {
    /* Increase list length */
    const size_t old_num_events = dev->numAllEvents;
    const size_t new_num_events = old_num_events + numMetrics;
    NvmonEvent **new_allevents = realloc(dev->allevents, new_num_events * sizeof(NvmonEvent *));
    if (new_allevents == NULL) {
        dev_freeEventList(dev);
        return -ENOMEM;
    }

    dev->allevents = new_allevents;
    dev->numAllEvents = new_num_events;

    /* Zero newly added entries so we can safely free them in case the for-loop below
     * does not complete due to error. */
    for (size_t i = old_num_events; i < new_num_events; i++)
        dev->allevents[i] = NULL;

    /* Add events to extended list */
    for (size_t i = 0; i < numMetrics; i++) {
        const size_t ev_idx = old_num_events + i;
        dev->allevents[ev_idx] = calloc(1, sizeof(NvmonEvent));
        if (!dev->allevents[ev_idx]) {
            /* Cleanup the entire list in case of error. */
            dev_freeEventList(dev);
            return -ENOMEM;
        }

        /* short helper variable */
        NvmonEvent_t e = dev->allevents[ev_idx];

        /* Create event name string. */
        bstring t = bfromcstr(metricNames[i]);
        prepare_metric_name(t);
        snprintf(e->name, sizeof(e->name), "%s", bdata(t));
        static const struct tagbstring sumtype = bsStatic(".sum");
        static const struct tagbstring mintype = bsStatic(".min");
        static const struct tagbstring maxtype = bsStatic(".max");
        if (binstrrcaseless(t, blength(t) - 1, &sumtype) != BSTR_OK) {
            e->rtype = ENTITY_TYPE_SUM;
        } else if (binstrrcaseless(t, blength(t) - 1, &mintype) != BSTR_OK) {
            e->rtype = ENTITY_TYPE_MIN;
        } else if (binstrrcaseless(t, blength(t) - 1, &maxtype) != BSTR_OK) {
            e->rtype = ENTITY_TYPE_MAX;
        } else {
            e->rtype = ENTITY_TYPE_INSTANT;
        }
        bdestroy(t);
        snprintf(e->real, sizeof(e->real), "%s", metricNames[i]);
        e->eventId = i;
        e->type = NVMON_PERFWORKS_EVENT;
    }

    return 0;
}

static int nvmon_perfworks_populateEvents_nvpw(NvmonDevice *dev) {
    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = {
      .structSize = NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE,
      .pChipName = dev->chip,
    };
    NVPW_CALL(
        return -1, NVPW_CUDA_MetricsContext_Create, &metricsContextCreateParams);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "NVPW Metrics Context created (deviceId=%d)", dev->deviceId);

    NVPW_MetricsContext_GetMetricNames_Begin_Params getMetricNameBeginParams = {
        .structSize = NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE,
        .pMetricsContext = metricsContextCreateParams.pMetricsContext,
        .hidePeakSubMetrics = 1,
        .hidePerCycleSubMetrics = 1,
        .hidePctOfPeakSubMetrics = 1,
        // .hidePctOfPeakSubMetricsOnThroughputs = 1,
    };
    int err = 0;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Create metric context getMetricNames");
    NVPW_CALL(err = -1; goto deinit, NVPW_MetricsContext_GetMetricNames_Begin, &getMetricNameBeginParams);

    err = nvmon_perfworks_appendEventList(dev,
            getMetricNameBeginParams.ppMetricNames,
            getMetricNameBeginParams.numMetrics);

    /* Cleanup NVPW stuff */
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Destroy metric context getMetricNames");
    NVPW_MetricsContext_GetMetricNames_End_Params getMetricNameEndParams = {
        .structSize = NVPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE,
        .pMetricsContext = metricsContextCreateParams.pMetricsContext,
    };
    NVPW_MetricsContext_GetMetricNames_End_ptr(&getMetricNameEndParams);

deinit:
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Destroy metric context");
    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = {
        .structSize = NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE,
        .pMetricsContext = metricsContextCreateParams.pMetricsContext,
    };
    NVPW_MetricsContext_Destroy_ptr(&metricsContextDestroyParams);
    return err;
}

static int nvmon_perfworks_appendSubEvents(NvmonDevice *dev, CUpti_Profiler_Host_Object *hobj, CUpti_MetricType type, const char * const *metricSuffixes, size_t numMetrics) {
    int err = 0;
    CUpti_Profiler_Host_GetBaseMetrics_Params getBaseMetricsParams = {
        .structSize = CUpti_Profiler_Host_GetBaseMetrics_Params_STRUCT_SIZE,
        .pHostObject = hobj,
        .metricType = type,
    };
    CUPTI_CALL(cuptiProfilerHostGetBaseMetrics, &getBaseMetricsParams);

    for (size_t i = 0; i < getBaseMetricsParams.numMetrics; i++) {
        const size_t METRIC_LEN = 256;
        char names[numMetrics][METRIC_LEN];
        const char *namesList[numMetrics];

        for (size_t j = 0; j < numMetrics; j++) {
            snprintf(names[j], sizeof(names[j]), "%s.%s", getBaseMetricsParams.ppMetricNames[i], metricSuffixes[j]);
            namesList[j] = names[j];
        }

        err = nvmon_perfworks_appendEventList(dev, namesList, numMetrics);
        if (err < 0)
            return err;
    }

    return 0;
}

static int nvmon_perfworks_populateEvents_cuptiProfilerHost(NvmonDevice *dev) {
    /* Init CUPTI host interface */
    CUpti_Profiler_Host_Initialize_Params initParams = {
        .structSize = CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE,
        .profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER,
        .pChipName = dev->chip,
    };
    CUPTI_CALL(cuptiProfilerHostInitialize, &initParams);
    CUpti_Profiler_Host_Object *hobj = initParams.pHostObject;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "CUDA12.6+ CUPTI Profiler Host created (deviceId=%d)", dev->deviceId);

    /* Get base metrics. */
    static const char *suffixesCnt[] = { "avg", "max", "min", "sum" };
    int err = nvmon_perfworks_appendSubEvents(dev, hobj, CUPTI_METRIC_TYPE_COUNTER, suffixesCnt, ARRAY_COUNT(suffixesCnt));
    if (err < 0)
        goto deinit;

    static const char *suffixesRatio[] = { "max_rate", "pct", "ratio" };
    err = nvmon_perfworks_appendSubEvents(dev, hobj, CUPTI_METRIC_TYPE_RATIO, suffixesRatio, ARRAY_COUNT(suffixesRatio));
    if (err < 0)
        goto deinit;

    static const char *suffixesThru[] = {
        "avg_pct_of_peak_sustained_active",
        "max_pct_of_peak_sustained_active",
        "min_pct_of_peak_sustained_active",
        "sum_pct_of_peak_sustained_active",
        "avg_pct_of_peak_sustained_elapsed",
        "max_pct_of_peak_sustained_elapsed",
        "min_pct_of_peak_sustained_elapsed",
        "sum_pct_of_peak_sustained_elapsed",
        "avg_pct_of_peak_sustained_region",
        "max_pct_of_peak_sustained_region",
        "min_pct_of_peak_sustained_region",
        "sum_pct_of_peak_sustained_region",
        "avg_pct_of_peak_sustained_frame",
        "max_pct_of_peak_sustained_frame",
        "min_pct_of_peak_sustained_frame",
        "sum_pct_of_peak_sustained_frame",
    };
    err = nvmon_perfworks_appendSubEvents(dev, hobj, CUPTI_METRIC_TYPE_THROUGHPUT, suffixesThru, ARRAY_COUNT(suffixesThru));
    if (err < 0)
        goto deinit;

    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Got supported metrics list (deviceId=%d)", dev->deviceId);

    /* hopefully CUPTI auto frees ppMetricNames on deinit, because it does not
     * appear to have an explicit free functions. */

deinit:
    err = 0;
    CUpti_Profiler_Host_Deinitialize_Params deinitParams = {
        .structSize = CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE,
        .pHostObject = hobj,
    };
    cuptiProfilerHostDeinitialize_ptr(&deinitParams);
    return err;
}

static int nvmon_perfworks_createDevice(int id, NvmonDevice *dev) {
    if (dl_perfworks_libcuda == NULL || dl_perfworks_libcudart == NULL ||
            dl_libhost == NULL || dl_cupti == NULL) {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "link_perfworks_libraries in createDevice");
        int err = link_perfworks_libraries();
        if (err < 0)
            return err;
    }

    int count = 0;
    CU_CALL(cuDeviceGetCount, &count);
    if (count == 0) {
        printf("No GPUs found\n");
        return -1;
    }
    if (id < 0 || id >= count) {
        printf("GPU %d not available\n", id);
        return -1;
    }
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Found %d GPUs", count);

    // Assign device ID and get cuDevice from CUDA
    CU_CALL(cuDeviceGet, &dev->cuDevice, id);
    dev->deviceId = id;
    dev->context = NULL;

    /* initialize */
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Current GPU %d", id);
    CUpti_Profiler_Initialize_Params profilerInitializeParams = {
        .structSize = CUpti_Profiler_Initialize_Params_STRUCT_SIZE,
    };
    CUPTI_CALL(
            cuptiProfilerInitialize, &profilerInitializeParams);

    NVPW_InitializeHost_Params initializeHostParams = {
        .structSize = NVPW_InitializeHost_Params_STRUCT_SIZE,
    };
    NVPW_CALL(return -1, NVPW_InitializeHost, &initializeHostParams);

    int err = nvmon_perfworks_getChipName(id, &dev->chip);
    if (err < 0)
        return err;

    /* Get events. Different implementation for different CUDA versions. */
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Get events for chip '%s'", dev->chip);
    dev->allevents = NULL;
    dev->numAllEvents = 0;
    if (cuda_runtime_version >= 12060)
        err = nvmon_perfworks_populateEvents_cuptiProfilerHost(dev);
    else
        err = nvmon_perfworks_populateEvents_nvpw(dev);

    dev->nvEventSets = NULL;
    dev->numNvEventSets = 0;
    dev->activeEventSet = -1;
    return err;
}

int nvmon_perfworks_getEventsOfGpu(int gpuId, NvmonEventList_t *list) {
    NvmonDevice device;
    int err = nvmon_perfworks_createDevice(gpuId, &device);
    if (err < 0) {
        ERROR_PRINT("No such device %d", gpuId);
        return err;
    }

    NvmonEventList_t l = malloc(sizeof(NvmonEventList));
    if (!l) {
        err = -errno;
        goto error;
    }

    l->events = calloc(device.numAllEvents, sizeof(NvmonEventListEntry));
    if (!l->events) {
        err = -errno;
        goto error;
    }

    for (int i = 0; i < device.numAllEvents; i++) {
        NvmonEventListEntry *out = &l->events[i];
        NvmonEvent_t event = device.allevents[i];
        out->name = strdup(event->name);
        if (!out->name) {
            err = -errno;
            goto error;
        }
        out->limit = strdup("GPU");
        if (!out->limit) {
            err = -errno;
            goto error;
        }
        out->desc = NULL;
    }
    l->numEvents = device.numAllEvents;
    *list = l;

    return 0;
    // TODO should we really not cleanup the nvmondevice in case of success?
    // This function should only retrieve the event list and not keep
    // a dangling device.
error:
    if (l) {
        if (l->events) {
            for (int i = 0; i < device.numAllEvents; i++) {
                free(l->events[i].name);
                free(l->events[i].limit);
            }
        }
        free(l->events);
    }
    free(l);
    // TODO: why are we not calling nvmon_perfworks_freeDevice?
    // It seems to do something different.
    nvmon_cupti_freeDevice(&device);
    return err;
}

static int nvmon_perfworks_getMetricRequests114(
        const char *chip, struct bstrList *events,
        uint8_t *availImage, NVPA_RawMetricRequest **requests) {
    int isolated = 1;
    int keepInstances = 1;

    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParam = {
        .structSize = NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE,
        .pChipName = chip,
        .pCounterAvailabilityImage = availImage,
    };
    NVPW_CALL(return -1,
            NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize,
                &calculateScratchBufferSizeParam);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Create scratch buffer for %s and %p", chip, availImage);
    uint8_t *scratch =
        malloc(calculateScratchBufferSizeParam.scratchBufferSize);
    if (!scratch)
        return -errno;

    NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {
        .structSize = NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE,
        .scratchBufferSize = calculateScratchBufferSizeParam.scratchBufferSize,
        .pScratchBuffer = scratch,
        .pChipName = chip,
        .pCounterAvailabilityImage = availImage,
    };
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Init Metric evaluator");
    NVPW_CALL(free(scratch); return -1, NVPW_CUDA_MetricsEvaluator_Initialize, &metricEvaluatorInitializeParams);

    NVPW_MetricsEvaluator *metricEvaluator =
        metricEvaluatorInitializeParams.pMetricsEvaluator;

    int raw_metrics = 0;
    size_t max_raw_deps = 0;
    for (int i = 0; i < events->qty; i++) {
        // TODO is this still used?
        // nvmon_perfworks_parse_metric(events->entry[i], &isolated,
        // &keepInstances);
        NVPW_MetricEvalRequest metricEvalRequest;
        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params
            convertMetricToEvalRequest = {
                .structSize = NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE,
                .pMetricsEvaluator = metricEvaluator,
                .pMetricName = bdata(events->entry[i]),
                .pMetricEvalRequest = &metricEvalRequest,
                .metricEvalRequestStructSize = sizeof(metricEvalRequest),
            };
        NVPW_CALL(free(scratch); return -1, NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest, &convertMetricToEvalRequest);

        NVPW_MetricsEvaluator_GetMetricRawDependencies_Params
            getMetricRawDependenciesParms = {
                .structSize = NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE,
                .pMetricsEvaluator = metricEvaluator,
                .pMetricEvalRequests = &metricEvalRequest,
                .numMetricEvalRequests = 1,
                .metricEvalRequestStructSize = sizeof(metricEvalRequest),
                .metricEvalRequestStrideSize = sizeof(metricEvalRequest),
            };
        NVPW_CALL(free(scratch); return -1, NVPW_MetricsEvaluator_GetMetricRawDependencies, &getMetricRawDependenciesParms);

        raw_metrics += getMetricRawDependenciesParms.numRawDependencies;
        max_raw_deps = MAX(max_raw_deps, getMetricRawDependenciesParms.numRawDependencies);
    }

    NVPA_RawMetricRequest *reqs = (NVPA_RawMetricRequest *)malloc(
            raw_metrics * sizeof(NVPA_RawMetricRequest));
    if (!reqs) {
        free(scratch);
        return -ENOMEM;
    }
    const char **rawDeps =
        (const char **)malloc(max_raw_deps * sizeof(const char *));
    if (!rawDeps) {
        free(scratch);
        free(reqs);
        return -ENOMEM;
    }

    raw_metrics = 0;
    for (int i = 0; i < events->qty; i++) {
        NVPW_MetricEvalRequest metricEvalRequest;
        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params
            convertMetricToEvalRequest = {
                .structSize = NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE,
                .pMetricsEvaluator = metricEvaluator,
                .pMetricName = bdata(events->entry[i]),
                .pMetricEvalRequest = &metricEvalRequest,
                .metricEvalRequestStructSize = sizeof(metricEvalRequest),
            };
        NVPW_CALL(free(scratch); return -1, NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest, &convertMetricToEvalRequest);

        NVPW_MetricsEvaluator_GetMetricRawDependencies_Params
            getMetricRawDependenciesParms = {
                .structSize = NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE,
                .pMetricsEvaluator = metricEvaluator,
                .pMetricEvalRequests = &metricEvalRequest,
                .numMetricEvalRequests = 1,
                .metricEvalRequestStructSize = sizeof(metricEvalRequest),
                .metricEvalRequestStrideSize = sizeof(metricEvalRequest),
            };
        NVPW_CALL(free(scratch); return -1, NVPW_MetricsEvaluator_GetMetricRawDependencies,
                    &getMetricRawDependenciesParms);
        getMetricRawDependenciesParms.ppRawDependencies = rawDeps;
        NVPW_CALL(free(scratch); return -1, NVPW_MetricsEvaluator_GetMetricRawDependencies,
                    &getMetricRawDependenciesParms);

        for (size_t j = 0; j < getMetricRawDependenciesParms.numRawDependencies; ++j) {
            reqs[raw_metrics].pMetricName = rawDeps[j];
            reqs[raw_metrics].isolated = isolated;
            reqs[raw_metrics].keepInstances = keepInstances;
            reqs[raw_metrics].structSize = NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE;
            raw_metrics++;
        }
    }
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Destroy Metric evaluator");
    NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams = {
        .structSize = NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE,
        .pMetricsEvaluator = metricEvaluator,
    };
    NVPW_CALL(
            free(scratch);
            free(rawDeps); free(reqs); return -1,
            NVPW_MetricsEvaluator_Destroy, &metricEvaluatorDestroyParams);

    free(scratch);
    free(rawDeps);
    *requests = reqs;
    return raw_metrics;
}

static int
nvmon_perfworks_getMetricRequests3(NVPA_MetricsContext *context,
                                   struct bstrList *events,
                                   NVPA_RawMetricRequest **requests) {
    int isolated = 1;
    int keepInstances = 1;

    int raw_metrics = 0;
    for (int i = 0; i < events->qty; i++) {
        // TODO is isolated still needed?
        // nvmon_perfworks_parse_metric(events->entry[i], &isolated,
        // &keepInstances);
        // keepInstances = 1; /* Bug in Nvidia API */
        NVPW_MetricsContext_GetMetricProperties_Begin_Params
            getMetricPropertiesBeginParams = {
                .structSize = NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE,
                .pMetricsContext = context,
                .pMetricName = bdata(events->entry[i]),
            };
        NVPW_MetricsContext_GetMetricProperties_End_Params
            getMetricPropertiesEndParams = {
                .structSize = NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE,
                .pMetricsContext = context,
            };
        NVPW_CALL(return -1, NVPW_MetricsContext_GetMetricProperties_Begin,
                    &getMetricPropertiesBeginParams);
        for (const char **dep =
                getMetricPropertiesBeginParams.ppRawMetricDependencies;
                *dep; ++dep)
            raw_metrics++;
        NVPW_CALL(return -1, NVPW_MetricsContext_GetMetricProperties_End,
                    &getMetricPropertiesEndParams);
    }

    NVPA_RawMetricRequest *reqs = (NVPA_RawMetricRequest *)malloc(
            raw_metrics * sizeof(NVPA_RawMetricRequest));
    if (!reqs)
        return -ENOMEM;

    raw_metrics = 0;

    for (int i = 0; i < events->qty; i++) {
        NVPW_MetricsContext_GetMetricProperties_Begin_Params
            getMetricPropertiesBeginParams = {
                .structSize = NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE,
                .pMetricsContext = context,
                .pMetricName = bdata(events->entry[i]),
            };
        NVPW_CALL(free(reqs); return -1, NVPW_MetricsContext_GetMetricProperties_Begin,
                    &getMetricPropertiesBeginParams);

        for (const char **dep =
                getMetricPropertiesBeginParams.ppRawMetricDependencies;
                *dep; ++dep) {
            NVPA_RawMetricRequest *req = &reqs[raw_metrics];
            req->pMetricName = strdup(*dep);
            req->isolated = isolated;
            req->keepInstances = keepInstances;
            raw_metrics++;
        }

        NVPW_MetricsContext_GetMetricProperties_End_Params
            getMetricPropertiesEndParams = {
                .structSize = NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE,
                .pMetricsContext = context,
            };
        NVPW_CALL(free(reqs); return -1, NVPW_MetricsContext_GetMetricProperties_End,
                    &getMetricPropertiesEndParams);
    }
    *requests = reqs;
    return raw_metrics;
}

static int nvmon_perfworks_createConfigImage(const char *chip,
        struct bstrList *events,
        uint8_t **configImage,
        uint8_t *availImage) {
    int ierr = 0;
    uint8_t *cimage = NULL;
    int num_reqs = 0;

    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = {
        .structSize = NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE,
        .pChipName = chip,
    };

    NVPA_RawMetricRequest *reqs = NULL;
    NVPA_RawMetricsConfig *pRawMetricsConfig = NULL;
    if (cuda_runtime_version < 11040 && NVPA_RawMetricsConfig_Create_ptr) {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Create Metrics Context");
        NVPW_CALL(return -1,
                NVPW_CUDA_MetricsContext_Create, &metricsContextCreateParams);
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Create config image for chip %s", chip);
        num_reqs = nvmon_perfworks_getMetricRequests3(metricsContextCreateParams.pMetricsContext, events, &reqs);
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Create config image for chip %s with %d metric requests", chip, num_reqs);

        NVPA_RawMetricsConfigOptions metricsConfigOptions = {
            .structSize = NVPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE,
            .activityKind = NVPA_ACTIVITY_KIND_PROFILER,
            .pChipName = chip
        };
        NVPW_CALL(ierr = -1; goto nvmon_perfworks_createConfigImage_out, NVPA_RawMetricsConfig_Create, &metricsConfigOptions, &pRawMetricsConfig);
    }
    else if (cuda_runtime_version >= 11040 && NVPW_CUDA_RawMetricsConfig_Create_V2_ptr) {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Create config image for chip %s", chip);
        num_reqs = nvmon_perfworks_getMetricRequests114(chip, events, availImage, &reqs);
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Create config image for chip %s with %d metric requests", chip, num_reqs);

        NVPW_CUDA_RawMetricsConfig_Create_V2_Params rawMetricsConfigCreateParams = {
            .structSize = NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE,
            .activityKind = NVPA_ACTIVITY_KIND_PROFILER,
            .pChipName = chip,
            .pCounterAvailabilityImage = availImage,
        };
        NVPW_CALL(free(reqs); return -1, NVPW_CUDA_RawMetricsConfig_Create_V2, &rawMetricsConfigCreateParams);
        pRawMetricsConfig = rawMetricsConfigCreateParams.pRawMetricsConfig;
    }

    if (availImage) {
        NVPW_RawMetricsConfig_SetCounterAvailability_Params
            setCounterAvailabilityParams = {
                .structSize = NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE,
                .pRawMetricsConfig = pRawMetricsConfig,
                .pCounterAvailabilityImage = availImage,
            };
        NVPW_CALL(
                ierr = -1;
                goto nvmon_perfworks_createConfigImage_out,
                NVPW_RawMetricsConfig_SetCounterAvailability,
                    &setCounterAvailabilityParams);
    }

    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = {
        .structSize = NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE,
        .pRawMetricsConfig = pRawMetricsConfig,
    };
    NVPW_CALL(
            ierr = -1;
            goto nvmon_perfworks_createConfigImage_out,
            NVPW_RawMetricsConfig_BeginPassGroup,&beginPassGroupParams);

    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = {
        .structSize = NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE,
        .pRawMetricsConfig = pRawMetricsConfig,
        .pRawMetricRequests = reqs,
        .numMetricRequests = num_reqs,
    };
    NVPW_CALL(
            ierr = -1;
            goto nvmon_perfworks_createConfigImage_out,
            NVPW_RawMetricsConfig_AddMetrics,&addMetricsParams);

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = {
        .structSize = NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE,
        .pRawMetricsConfig = pRawMetricsConfig,
    };
    NVPW_CALL(
            ierr = -1;
            goto nvmon_perfworks_createConfigImage_out,
            NVPW_RawMetricsConfig_EndPassGroup,&endPassGroupParams);

    NVPW_RawMetricsConfig_GetNumPasses_Params getNumPassesParams = {
        .structSize = NVPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE,
        .pRawMetricsConfig = pRawMetricsConfig,
    };
    NVPW_CALL(
            ierr = -1;
            goto nvmon_perfworks_createConfigImage_out,
            NVPW_RawMetricsConfig_GetNumPasses,&getNumPassesParams);
    if (getNumPassesParams.numPipelinedPasses +
            getNumPassesParams.numIsolatedPasses >
            1) {
        errno = 1; // why do we set errno here an nowhere else ???
        ierr = -errno;
        ERROR_PRINT("Given GPU eventset requires multiple passes. "
                "Currently not supported.");
        goto nvmon_perfworks_createConfigImage_out;
    }

    NVPW_RawMetricsConfig_GenerateConfigImage_Params
        generateConfigImageParams = {
            .structSize = NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE,
            .pRawMetricsConfig = pRawMetricsConfig,
        };
    NVPW_CALL(ierr = -1;
            goto nvmon_perfworks_createConfigImage_out, NVPW_RawMetricsConfig_GenerateConfigImage,
                &generateConfigImageParams);

    NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = {
        .structSize = NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE,
        .pRawMetricsConfig = pRawMetricsConfig,
        .bytesAllocated = 0,
        .pBuffer = NULL,
    };
    NVPW_CALL(
            ierr = -1;
            goto nvmon_perfworks_createConfigImage_out,
            NVPW_RawMetricsConfig_GetConfigImage,&getConfigImageParams);

    cimage = malloc(getConfigImageParams.bytesCopied);
    if (!cimage) {
        ierr = -ENOMEM;
        goto nvmon_perfworks_createConfigImage_out;
    }
    int ci_size = getConfigImageParams.bytesCopied;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Allocated %d byte for configImage", ci_size);

    getConfigImageParams.bytesAllocated = getConfigImageParams.bytesCopied;
    getConfigImageParams.pBuffer = cimage;
    NVPW_CALL(
            free(cimage);
            ierr = -1;
            goto nvmon_perfworks_createConfigImage_out,
            NVPW_RawMetricsConfig_GetConfigImage,&getConfigImageParams);

nvmon_perfworks_createConfigImage_out:
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP,
            "nvmon_perfworks_createConfigImage_out enter %d", ierr);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "NVPW_RawMetricsConfig_Destroy");
    NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {
        .structSize = NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE,
        .pRawMetricsConfig = pRawMetricsConfig,
    };
    NVPW_CALL(
            return -1, NVPW_RawMetricsConfig_Destroy,&rawMetricsConfigDestroyParams);

    if (metricsContextCreateParams.pMetricsContext) {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "NVPW_MetricsContext_Destroy");
        NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = {
            .structSize = NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE,
            .pMetricsContext = metricsContextCreateParams.pMetricsContext,
        };
        NVPW_CALL(return -1,
                NVPW_MetricsContext_Destroy,&metricsContextDestroyParams);
    }
    /*    for (i = 0; i < num_reqs; i++)*/
    /*    {*/
    /*        free((void*)reqs[i].pMetricName);*/
    /*    }*/
    free(reqs);
    if (ierr == 0) {
        ierr = ci_size;
        *configImage = cimage;
    } else {
        free(cimage);
    }

    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP,
            "nvmon_perfworks_createConfigImage returns %d", ierr);
    return ierr;
}

static int nvmon_perfworks_createCounterDataPrefixImage(
      char *chip, struct bstrList *events, uint8_t **cdpImage) {
    int err = 0;
    uint8_t *cdp = NULL;
    int pi_size = 0;
    NVPA_RawMetricRequest *reqs = NULL;

    NVPA_MetricsContext *mc = NULL;
    if (cuda_runtime_version < 12060) {
        NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = {
            .structSize = NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE,
            .pChipName = chip,
        };
        NVPW_CALL(
            err = -EPERM;
            goto nvmon_perfworks_createCounterDataPrefixImage_out,
            NVPW_CUDA_MetricsContext_Create,&metricsContextCreateParams);
        mc = metricsContextCreateParams.pMetricsContext;
    }

    int num_reqs;
    if (cuda_runtime_version < 11040)
        num_reqs = nvmon_perfworks_getMetricRequests3(mc, events, &reqs);
    else if (cuda_runtime_version >= 11040)
        num_reqs = nvmon_perfworks_getMetricRequests114(chip, events, NULL, &reqs);

    NVPW_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = {
        .structSize = NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE,
        .pChipName = chip,
    };
    NVPW_CALL(
        err = -EPERM;
        goto nvmon_perfworks_createCounterDataPrefixImage_out,
        NVPW_CounterDataBuilder_Create,&counterDataBuilderCreateParams);

    NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = {
        .structSize = NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE,
        .pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder,
        .pRawMetricRequests = reqs,
        .numMetricRequests = num_reqs,
    };
    NVPW_CALL(
        err = -EPERM;
        goto nvmon_perfworks_createCounterDataPrefixImage_out,
        NVPW_CounterDataBuilder_AddMetrics,&addMetricsParams);

    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = {
        .structSize = NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE,
        .pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder,
        .bytesAllocated = 0,
        .pBuffer = NULL,
    };
    NVPW_CALL(
            err = -1;
            goto nvmon_perfworks_createCounterDataPrefixImage_out,
            NVPW_CounterDataBuilder_GetCounterDataPrefix,
            &getCounterDataPrefixParams);

    cdp = malloc(getCounterDataPrefixParams.bytesCopied + 10); // why +10?
    if (!cdp) {
        err = -ENOMEM;
        goto nvmon_perfworks_createCounterDataPrefixImage_out;
    }
    pi_size = getCounterDataPrefixParams.bytesCopied;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Allocated %d byte for configPrefixImage", pi_size);

    getCounterDataPrefixParams.bytesAllocated =
        getCounterDataPrefixParams.bytesCopied + 10; // why +10?
    getCounterDataPrefixParams.pBuffer = cdp;
    NVPW_CALL(
            free(cdp);
            err = -1;
            goto nvmon_perfworks_createCounterDataPrefixImage_out,
            NVPW_CounterDataBuilder_GetCounterDataPrefix,
            &getCounterDataPrefixParams);

nvmon_perfworks_createCounterDataPrefixImage_out:
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP,
                   "nvmon_perfworks_createCounterDataPrefixImage_out enter %d",
                   err);
    // for (i = 0; i < num_reqs; i++)
    // {
    //     free((void*)reqs[i].pMetricName);
    // }
    // free(reqs);

    NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = {
        .structSize = NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE,
        .pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder,
    };
    NVPW_CALL(
        err = -1,
        NVPW_CounterDataBuilder_Destroy,&counterDataBuilderDestroyParams);

    if (mc) {
        NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = {
            .structSize = NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE,
            .pMetricsContext = mc,
        };
        NVPW_CALL(
            err = -1,
            NVPW_MetricsContext_Destroy,&metricsContextDestroyParams);
    }
    /*    for (i = 0; i < num_reqs; i++)*/
    /*    {*/
    /*        free((void*)reqs[i].pMetricName);*/
    /*    }*/
    free(reqs);
    if (err == 0) {
        err = pi_size;
        *cdpImage = cdp;
    } else {
        free(cdp);
    }
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP,
                   "nvmon_perfworks_createCounterDataPrefixImage returns %d",
                   err);
    return err;
}

static int nvmon_perfworks_addEventSet(NvmonDevice_t device,
        const char *eventString) {
    // cuptiProfiler_init(); // TODO why is this commented out?

    int curDeviceId = -1;
    CUDA_CALL(cudaGetDevice, &curDeviceId);

    /* implicitly create context via cudaFree */
    CUDA_CALL(cudaFree, NULL);

    CUcontext curContext;
    CU_CALL(cuCtxGetCurrent, &curContext);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP,
            "Add events to GPU device %d with context %p",
            device->deviceId, curContext);

    if (curDeviceId != device->deviceId) {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Switching to GPU device %d",
                device->deviceId);
        CUDA_CALL(cudaSetDevice,device->deviceId);
    }

    int popContext = perfworks_check_nv_context(device, curContext);
    if (popContext < 0) {
        errno = -popContext;
        ERROR_PRINT("Failed to get context");
    }

    bstring eventBString = bfromcstr(eventString);
    struct bstrList *tmp = bsplit(eventBString, ',');
    bdestroy(eventBString);

    NvmonEvent_t *nvEvents = malloc(tmp->qty * sizeof(NvmonEvent_t));
    if (!nvEvents) {
        bstrListDestroy(tmp);
        return -ENOMEM;
    }
    struct bstrList *eventtokens = bstrListCreate();

    for (int i = 0; i < tmp->qty; i++) {
        struct bstrList *parts = bsplit(tmp->entry[i], ':');
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "%s", bdata(parts->entry[0]));
        for (int j = 0; j < device->numAllEvents; j++) {
            bstring bname = bfromcstr(device->allevents[j]->name);
            if (bstrcmp(parts->entry[0], bname) == BSTR_OK) {
                bstrListAddChar(eventtokens, device->allevents[j]->real);
                nvEvents[i] = device->allevents[j];
                GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Adding real event %s",
                        device->allevents[j]->real);
            }
        }
        bstrListDestroy(parts);
    }
    bstrListDestroy(tmp);
    if (eventtokens->qty == 0) {
        ERROR_PRINT("No event in eventset");
        bstrListDestroy(eventtokens);
        if (popContext > 0) {
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Pop Context %p for device %d", device->context, device->deviceId);
            CU_CALL(cuCtxPopCurrent,&device->context);
        }
        if (curDeviceId != device->deviceId) {
            GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Switching to GPU device %d",
                    device->deviceId);
            CUDA_CALL(cudaSetDevice,device->deviceId);
        }
        return -EFAULT;
    }

    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP,
            "Increase size of eventSet space on device %d",
            device->deviceId);
    NvmonEventSet *tmpEventSet =
        realloc(device->nvEventSets,
                (device->numNvEventSets + 1) * sizeof(NvmonEventSet));
    if (!tmpEventSet) {
        ERROR_PRINT("Cannot enlarge GPU %d eventSet list", device->deviceId);
        bstrListDestroy(eventtokens);
        return -ENOMEM;
    }
    device->nvEventSets = tmpEventSet;
    NvmonEventSet *newEventSet = &device->nvEventSets[device->numNvEventSets];
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Filling eventset %d on device %d",
            device->numNvEventSets, device->deviceId);

    size_t availImageSize = 0;
    uint8_t *availImage = NULL;
    if (cuda_version >= 11000 && cuda_runtime_version >= 11000) {
        CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {
            .structSize = CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE,
            .ctx = device->context,
        };
        CUPTI_CALL(cuptiProfilerGetCounterAvailability,
                    &getCounterAvailabilityParams);

        availImage =
            malloc(getCounterAvailabilityParams.counterAvailabilityImageSize);
        if (!availImage) {
            bstrListDestroy(eventtokens);
            return -ENOMEM;
        }
        getCounterAvailabilityParams.ctx = device->context;
        getCounterAvailabilityParams.pCounterAvailabilityImage = availImage;
        CUPTI_CALL(cuptiProfilerGetCounterAvailability,
                    &getCounterAvailabilityParams);
        availImageSize =
            getCounterAvailabilityParams.counterAvailabilityImageSize;
    }

    uint8_t *configImage = NULL;
    int ci_size = nvmon_perfworks_createConfigImage(device->chip, eventtokens,
            &configImage, availImage);
    uint8_t *prefixImage = NULL;
    int pi_size = nvmon_perfworks_createCounterDataPrefixImage(
            device->chip, eventtokens, &prefixImage);

    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Filling eventset %d on device %d",
            device->numNvEventSets, device->deviceId);

    int gid = -1;
    if (configImage && prefixImage) {
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
        newEventSet->numberOfEvents = eventtokens->qty;
        newEventSet->id = device->numNvEventSets;
        gid = device->numNvEventSets;
        newEventSet->nvEvents = calloc(eventtokens->qty, sizeof(NvmonEvent_t));
        if (newEventSet->nvEvents == NULL) {
            ERROR_PRINT("Cannot allocate event list for group %d", gid);
            return -ENOMEM;
        }
        for (int i = 0; i < eventtokens->qty; i++) {
            for (int j = 0; j < device->numAllEvents; j++) {
                bstring brealname = bfromcstr(device->allevents[j]->real);
                if (bstrcmp(eventtokens->entry[i], brealname) == BSTR_OK) {
                    newEventSet->nvEvents[i] = device->allevents[j];
                }
            }
        }
        newEventSet->results = calloc(eventtokens->qty, sizeof(NvmonEventResult));
        if (newEventSet->results == NULL) {
            ERROR_PRINT("Cannot allocate result list for group %d", gid);
            return -ENOMEM;
        }
        newEventSet->nvEvents = nvEvents;
        device->numNvEventSets++;
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Adding eventset %d", gid);
    }

    if (popContext > 0) {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Pop Context %p for device %d", device->context, device->deviceId);
        CU_CALL(cuCtxPopCurrent,&device->context);
    }
    if (curDeviceId != device->deviceId) {
        CUDA_CALL(cudaSetDevice,curDeviceId);
    }
    return gid;
}

static int nvmon_perfworks_setupCounterImageData(NvmonEventSet *eventSet) {
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "counterDataPrefixSize %ld",
            eventSet->counterDataImagePrefixSize);

    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions = {
        .counterDataPrefixSize = eventSet->counterDataImagePrefixSize,
        .pCounterDataPrefix = eventSet->counterDataImagePrefix,
        .maxNumRanges = 1,
        .maxNumRangeTreeNodes = 1,
        .maxRangeNameLength = NVMON_DEFAULT_STR_LEN - 1,
    };
    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {
        .structSize = CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE,
        .pOptions = &counterDataImageOptions,
        .sizeofCounterDataImageOptions = sizeof(counterDataImageOptions),
    };
    CUPTI_CALL(
            cuptiProfilerCounterDataImageCalculateSize,&calculateSizeParams);

    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Resize counterDataImage to %ld",
            calculateSizeParams.counterDataImageSize);
    uint8_t *tmp = realloc(eventSet->counterDataImage,
            calculateSizeParams.counterDataImageSize);
    if (!tmp)
        return -ENOMEM;

    eventSet->counterDataImage = tmp;
    eventSet->counterDataImageSize = calculateSizeParams.counterDataImageSize;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Resized counterDataImage to %ld",
            eventSet->counterDataImageSize);

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {
        .structSize = CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE,
        .pOptions = &counterDataImageOptions,
        .sizeofCounterDataImageOptions = sizeof(counterDataImageOptions),
        .pCounterDataImage = eventSet->counterDataImage,
        .counterDataImageSize = calculateSizeParams.counterDataImageSize
    };
    CUPTI_CALL(
            cuptiProfilerCounterDataImageInitialize,&initializeParams);

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params
        scratchBufferSizeParams = {
            .structSize = CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE,
            .counterDataImageSize = calculateSizeParams.counterDataImageSize,
            .pCounterDataImage = eventSet->counterDataImage,
        };
    CUPTI_CALL(
            cuptiProfilerCounterDataImageCalculateScratchBufferSize,
                &scratchBufferSizeParams);

    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Resize counterDataScratchBuffer to %ld",
            scratchBufferSizeParams.counterDataScratchBufferSize);
    tmp = realloc(eventSet->counterDataScratchBuffer,
            scratchBufferSizeParams.counterDataScratchBufferSize);
    if (!tmp)
        return -ENOMEM;

    eventSet->counterDataScratchBuffer = tmp;
    eventSet->counterDataScratchBufferSize =
        scratchBufferSizeParams.counterDataScratchBufferSize;
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Resized counterDataScratchBuffer to %ld",
            eventSet->counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params
        initScratchBufferParams = {
            .structSize = CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE,
            .counterDataImageSize = calculateSizeParams.counterDataImageSize,
            .pCounterDataImage = initializeParams.pCounterDataImage,
            .counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize,
            .pCounterDataScratchBuffer = eventSet->counterDataScratchBuffer,
        };
    CUPTI_CALL(
            cuptiProfilerCounterDataImageInitializeScratchBuffer,
                &initScratchBufferParams);

    return 0;
}

typedef struct {
    int num_ranges;
    struct bstrList *names;
    double *values;
} PerfWorksMetricRanges;

static int nvmon_perfworks_getMetricValue12(
        char *chip, NvmonEventSet *eventSet, double **values) {
    assert(eventSet->events->qty == eventSet->numberOfEvents);

    uint8_t *scratchBuffer = NULL;
    int err = 0;
    double *gpuValues = calloc(eventSet->numberOfEvents, sizeof(double));
    if (!gpuValues)
        return -ENOMEM;

    NVPW_MetricEvalRequest *requests = calloc(eventSet->numberOfEvents, sizeof(NVPW_MetricEvalRequest));
    if (!requests) {
        free(gpuValues);
        return -errno;
    }

    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params scratchBufParams = {
        .structSize = NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE,
        .pChipName = chip,
        .pCounterAvailabilityImage = eventSet->counterAvailabilityImage,
    };
    NVPW_CALL(err = -EPERM; goto cleanup, NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize,&scratchBufParams);

    scratchBuffer = malloc(scratchBufParams.scratchBufferSize);
    if (!scratchBuffer) {
        err = -errno;
        goto cleanup;
    }

    NVPW_CUDA_MetricsEvaluator_Initialize_Params initParams = {
        .structSize = NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE,
        .pScratchBuffer = scratchBuffer,
        .scratchBufferSize = scratchBufParams.scratchBufferSize,
        .pChipName = chip,
        .pCounterAvailabilityImage = eventSet->counterAvailabilityImage,
        .pCounterDataImage = eventSet->counterDataImage,
        .counterDataImageSize = eventSet->counterDataImageSize,
    };
    NVPW_CALL(err = -EPERM; goto cleanup, NVPW_CUDA_MetricsEvaluator_Initialize,&initParams);

    for (int i = 0; i < eventSet->numberOfEvents; i++) {
        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convertParams = {
            .structSize = NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE,
            .pMetricsEvaluator = initParams.pMetricsEvaluator,
            .pMetricName = bdata(eventSet->events->entry[i]),
            .pMetricEvalRequest = &requests[i],
            .metricEvalRequestStructSize = sizeof(requests[i]),
        };
        NVPW_CALL(err = -EPERM; goto cleanup, NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest,&convertParams);
    }

    NVPW_MetricsEvaluator_EvaluateToGpuValues_Params evalParams = {
        .structSize = NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE,
        .pMetricsEvaluator = initParams.pMetricsEvaluator,
        .pMetricEvalRequests = requests,
        .numMetricEvalRequests = eventSet->numberOfEvents,
        .metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE,
        .metricEvalRequestStrideSize = sizeof(requests[0]),
        .pCounterDataImage = eventSet->counterDataImage,
        .counterDataImageSize = eventSet->counterDataImageSize,
        .rangeIndex = 0,
        .isolated = 1,
        .pMetricValues = gpuValues,
    };
    NVPW_CALL(err = -EPERM; goto cleanup, NVPW_MetricsEvaluator_EvaluateToGpuValues,&evalParams);
    for (int i = 0; i < eventSet->events->qty; i++) {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Final Eval %s: %f",
                bdata(eventSet->events->entry[i]), gpuValues[i]);
    }
    *values = gpuValues;
cleanup:
    free(scratchBuffer);
    free(requests);
    if (err < 0)
        free(gpuValues);

    if (initParams.pMetricsEvaluator) {
        NVPW_MetricsEvaluator_Destroy_Params destoryParams = {
            .structSize = NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE,
            .pMetricsEvaluator = initParams.pMetricsEvaluator,
        };
        NVPW_CALL(, NVPW_MetricsEvaluator_Destroy,&destoryParams);
    }
    return err;
}

static int nvmon_perfworks_getMetricValue11(
        char *chip, NvmonEventSet *eventSet, double **values) {
    assert(eventSet->events->qty == eventSet->numberOfEvents);

    int err = 0;
    char **metricnames = NULL;
    int num_metricnames = 0;
    double *gpuValues = calloc(eventSet->numberOfEvents, sizeof(double));
    if (!gpuValues)
        return -errno;

    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = {
        .structSize = NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE,
        .pChipName = chip,
    };
    NVPW_CALL(
            err = -1;
            goto nvmon_perfworks_getMetricValue_out,
            NVPW_CUDA_MetricsContext_Create,&metricsContextCreateParams);

    NVPW_CounterData_GetNumRanges_Params getNumRangesParams = {
        .structSize = NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE,
        .pCounterDataImage = eventSet->counterDataImage,
    };
    NVPW_CALL(
            err = -1; goto nvmon_perfworks_getMetricValue_out,
            NVPW_CounterData_GetNumRanges,&getNumRangesParams);

    NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = {
        .structSize = NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE,
        .pMetricsContext = metricsContextCreateParams.pMetricsContext,
        .pCounterDataImage = eventSet->counterDataImage,
        .isolated = 1,
        .rangeIndex = 0,
    };
    NVPW_CALL(
            err = -1;
            goto nvmon_perfworks_getMetricValue_out,
            NVPW_MetricsContext_SetCounterData,&setCounterDataParams);

    num_metricnames = bstrListToCharList(eventSet->events, &metricnames);

    NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = {
        .structSize = NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE,
        .pMetricsContext = metricsContextCreateParams.pMetricsContext,
        .numMetrics = num_metricnames,
        .ppMetricNames = (const char **)metricnames,
        .pMetricValues = &gpuValues[0],
    };
    NVPW_CALL(
            err = -1;
            free(gpuValues);
            goto nvmon_perfworks_getMetricValue_out,
            NVPW_MetricsContext_EvaluateToGpuValues,&evalToGpuParams);
    // for (j = 0; j < events->qty; j++)
    // {
    //     bstrListAdd(r->names, rname);
    //     r->values[j] += gpuValues[j];
    // }
    for (int j = 0; j < eventSet->events->qty; j++) {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Final Eval %s: %f",
                bdata(eventSet->events->entry[j]), gpuValues[j]);
    }
    *values = gpuValues;

nvmon_perfworks_getMetricValue_out:
    if (err != 0)
        free(gpuValues);

    if (metricnames) {
        for (int i = 0; i < num_metricnames; i++)
            free(metricnames[i]);
        free(metricnames);
    }

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = {
        .structSize = NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE,
        .pMetricsContext = metricsContextCreateParams.pMetricsContext,
    };
    NVPW_CALL(
            err = -1,
            NVPW_MetricsContext_Destroy,&metricsContextDestroyParams);
    return err;
}

static int nvmon_perfworks_getMetricValue(
        char *chip, NvmonEventSet *eventSet, double **values) {
    if (!chip || !eventSet || !values)
        return -EINVAL;

    if (cuda_runtime_version < 12060)
        return nvmon_perfworks_getMetricValue11(
                chip, eventSet, values);
    return nvmon_perfworks_getMetricValue12(
            chip, eventSet, values);
}

static int nvmon_perfworks_setupCounters(NvmonDevice_t device,
        NvmonEventSet * eventSet) {
    int curDeviceId = 0;
    CUDA_CALL(cudaGetDevice, &curDeviceId);
    if (curDeviceId != device->deviceId)
        CUDA_CALL(cudaSetDevice, device->deviceId);

    CUcontext curContext;
    CU_CALL(cuCtxGetCurrent,&curContext);
    int popContext = perfworks_check_nv_context(device, curContext);
    if (popContext < 0) {
        errno = -popContext;
        ERROR_PRINT("Failed to get context");
    }
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Setup Counters on device %d",
            device->deviceId);

    // cuptiProfiler_init();

    int ret = nvmon_perfworks_setupCounterImageData(eventSet);
    device->activeEventSet = eventSet->id;
    nvGroupSet->activeGroup = eventSet->id;
    if (popContext > 0) {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Pop Context %p for device %d", device->context, device->deviceId);
        CU_CALL(cuCtxPopCurrent,&device->context);
    }
    if (curDeviceId != device->deviceId)
        CUDA_CALL(cudaSetDevice, curDeviceId);

    return ret;
}

static int nvmon_perfworks_startCounters(NvmonDevice_t device) {
    int curDeviceId = 0;
    CUDA_CALL(cudaGetDevice,&curDeviceId);
    if (curDeviceId != device->deviceId)
        CUDA_CALL(cudaSetDevice,device->deviceId);

    CUcontext curContext;
    CU_CALL(cuCtxGetCurrent,&curContext);
    int popContext = perfworks_check_nv_context(device, curContext);
    if (popContext < 0) {
        errno = -popContext;
        ERROR_PRINT("Failed to get context");
    }

    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Start Counters on device %d (Eventset %d)",
            device->deviceId, device->activeEventSet);
    NvmonEventSet *eventSet = &device->nvEventSets[device->activeEventSet];

    CUpti_Profiler_BeginSession_Params beginSessionParams = {
        .structSize = CUpti_Profiler_BeginSession_Params_STRUCT_SIZE,
        .ctx = device->context,
        .counterDataImageSize = eventSet->counterDataImageSize,
        .pCounterDataImage = eventSet->counterDataImage,
        .counterDataScratchBufferSize = eventSet->counterDataScratchBufferSize,
        .pCounterDataScratchBuffer = eventSet->counterDataScratchBuffer,
        .range = CUPTI_UserRange,
        .replayMode = CUPTI_UserReplay,
        .maxRangesPerPass = 1,
        .maxLaunchesPerPass = 1,
    };
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "(START)counterDataImageSize %ld",
            eventSet->counterDataImageSize);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "(START)counterDataScratchBufferSize %ld",
            eventSet->counterDataScratchBufferSize);
    CUPTI_CALL(cuptiProfilerBeginSession,&beginSessionParams);

    /* Compute legacy struct size for backwards compatibility */
    const size_t params10_size = CUpti_Profiler_SetConfig_Params_STRUCT_SIZE10;
    const size_t params11_size = CUpti_Profiler_SetConfig_Params_STRUCT_SIZE11;
    CUpti_Profiler_SetConfig_Params setConfigParams = {
        .structSize = (cuda_runtime_version < 11000) ? params10_size : params11_size,
        .ctx = device->context,
        .pConfig = eventSet->configImage,
        .configSize = eventSet->configImageSize,
        .minNestingLevel = 1,
        .numNestingLevels = 1,
        .passIndex = 0,
        .targetNestingLevel = 1,
    };
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "(START)configImage %ld",
            eventSet->configImageSize);
    CUPTI_CALL(cuptiProfilerSetConfig,&setConfigParams);

    CUpti_Profiler_BeginPass_Params beginPassParams = {
        .structSize = CUpti_Profiler_BeginPass_Params_STRUCT_SIZE,
        .ctx = device->context,
    };
    CUPTI_CALL(cuptiProfilerBeginPass,&beginPassParams);

    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
        .structSize = CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE,
        .ctx = device->context,
    };
    CUPTI_CALL(
            cuptiProfilerEnableProfiling,&enableProfilingParams);
    CUpti_Profiler_PushRange_Params pushRangeParams = {
        .structSize = CUpti_Profiler_PushRange_Params_STRUCT_SIZE,
        .ctx = device->context,
        .pRangeName = "nvmon_perfworks",
    };
    CUPTI_CALL(cuptiProfilerPushRange,&pushRangeParams);

    if (popContext > 0) {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Pop Context %p for device %d", device->context, device->deviceId);
        CU_CALL(cuCtxPopCurrent,&device->context);
    }
    if (curDeviceId != device->deviceId)
        CUDA_CALL(cudaSetDevice,curDeviceId);

    return 0;
}

static int nvmon_perfworks_stopCounters(NvmonDevice_t device) {
    /* I have not gone through this function and made sure it does not leak
     * memory or illegal states when returning on error... */
    int curDeviceId = 0;
    CUDA_CALL(cudaGetDevice,&curDeviceId);
    if (curDeviceId != device->deviceId)
        CUDA_CALL(cudaSetDevice,device->deviceId);

    CUcontext curContext;
    CU_CALL(cuCtxGetCurrent,&curContext);
    int popContext = perfworks_check_nv_context(device, curContext);
    if (popContext < 0) {
        errno = -popContext;
        ERROR_PRINT("Failed to get context");
    }
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Stop Counters on device %d (Eventset %d)",
            device->deviceId, device->activeEventSet);
    NvmonEventSet *eventSet = &device->nvEventSets[device->activeEventSet];

    size_t CUpti_Profiler_EndPass_Params_size = 0;
    size_t CUpti_Profiler_FlushCounterData_Params_size = 0;
    if (cuda_runtime_version < 11000) {
        CUpti_Profiler_EndPass_Params_size =
            CUpti_Profiler_EndPass_Params_STRUCT_SIZE10;
        CUpti_Profiler_FlushCounterData_Params_size =
            CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE10;
    } else {
        CUpti_Profiler_EndPass_Params_size =
            CUpti_Profiler_EndPass_Params_STRUCT_SIZE11;
        CUpti_Profiler_FlushCounterData_Params_size =
            CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE11;
    }

    CUpti_Profiler_PopRange_Params popRangeParams = {
        .structSize = CUpti_Profiler_PopRange_Params_STRUCT_SIZE,
        .ctx = device->context,
    };
    CUPTI_CALL(cuptiProfilerPopRange,&popRangeParams);

    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
        .structSize = CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE,
        .ctx = device->context,
    };
    CUPTI_CALL(
            cuptiProfilerDisableProfiling,&disableProfilingParams);

    CUpti_Profiler_EndPass_Params endPassParams = {
        .structSize = CUpti_Profiler_EndPass_Params_size,
        .ctx = device->context,
    };
    CUPTI_CALL(cuptiProfilerEndPass,&endPassParams);
    if (endPassParams.allPassesSubmitted != 1) {
        ERROR_PRINT("Events cannot be measured in a single pass and "
                "multi - pass / kernel replay is current not supported");
    }
    CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {
        .structSize = CUpti_Profiler_FlushCounterData_Params_size,
        .ctx = device->context
    };
    CUPTI_CALL(
            cuptiProfilerFlushCounterData,&flushCounterDataParams);

    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {
        .structSize = CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE,
        .ctx = device->context,
    };
    CUPTI_CALL(cuptiProfilerUnsetConfig,&unsetConfigParams);

    CUpti_Profiler_EndSession_Params endSessionParams = {
        .structSize = CUpti_Profiler_EndSession_Params_STRUCT_SIZE,
        .ctx = device->context,
    };
    CUPTI_CALL(cuptiProfilerEndSession,&endSessionParams);
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Get results on device %d (Eventset %d)",
            device->deviceId, device->activeEventSet);

    double *values = NULL;
    int err = nvmon_perfworks_getMetricValue(device->chip, eventSet, &values);
    if (err < 0)
        return err;

    for (int j = 0; j < eventSet->numberOfEvents; j++) {
        double res = values[j];
        NvmonEvent_t nve = eventSet->nvEvents[j];
        eventSet->results[j].lastValue = res;
        switch (nve->rtype) {
        case ENTITY_TYPE_SUM:
            eventSet->results[j].fullValue += res;
            break;
        case ENTITY_TYPE_MIN:
            eventSet->results[j].fullValue = (res < eventSet->results[j].fullValue
                    ? res
                    : eventSet->results[j].fullValue);
            break;
        case ENTITY_TYPE_MAX:
            eventSet->results[j].fullValue = (res > eventSet->results[j].fullValue
                    ? res
                    : eventSet->results[j].fullValue);
            break;
        case ENTITY_TYPE_INSTANT:
            eventSet->results[j].fullValue = res;
            break;
        }
        eventSet->results[j].stopValue = eventSet->results[j].fullValue;
        eventSet->results[j].overflows = 0;
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "%s Last %f Full %f",
                bdata(eventSet->events->entry[j]),
                eventSet->results[j].lastValue,
                eventSet->results[j].fullValue);
    }

    if (popContext > 0) {
        GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, "Pop Context %p for device %d", device->context, device->deviceId);
        CU_CALL(cuCtxPopCurrent,&device->context);
    }
    if (curDeviceId != device->deviceId)
        CUDA_CALL(cudaSetDevice,curDeviceId);

    return 0;
}

int nvmon_perfworks_readCounters(NvmonDevice_t device) {
    // TODO should we report any errors in this function?
    nvmon_perfworks_stopCounters(device);
    nvmon_perfworks_startCounters(device);
    return 0;
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
