// SPDX-License-Identifier: GPL-3.0
#include "includes/nvmon_range.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <range_profiling.h>   // CUDA: extras/CUPTI/samples/range_profiling

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(call) do {                                     \
    CUresult _r = (call);                                         \
    if (_r != CUDA_SUCCESS) {                                     \
        const char* errstr = nullptr;                             \
        cuGetErrorString(_r, &errstr);                            \
        fprintf(stderr, "CUDA Driver error %d (%s) at %s:%d\n",   \
                (int)_r, errstr ? errstr : "unknown", __FILE__, __LINE__); \
        return -1;                                                \
    }                                                             \
} while(0)

struct RangeCtx {
    CUdevice device = 0;
    CUcontext ctx  = nullptr;
    RangeProfilerConfig    cfg;
    RangeProfilerTargetPtr pTgt;
    CuptiProfilerHostPtr   pHost;
    std::vector<uint8_t>   availabilityImage;
    std::vector<uint8_t>   configImage;
    std::vector<uint8_t>   counterDataImage;
    std::vector<std::string> metricsOwned;
    std::vector<const char*> metricsCStr;
    // CUpti_ProfilerReplayMode replayMode = CUPTI_KernelReplay;
    CUpti_ProfilerReplayMode  replayMode = CUPTI_UserReplay;
    bool enabled = false;
};
static std::unordered_map<int, RangeCtx> g_ctx;

static void fill_metrics(RangeCtx& C, const char** metricNames, int nMetrics)
{
    C.metricsOwned.clear(); C.metricsCStr.clear();
    C.metricsOwned.reserve(nMetrics); C.metricsCStr.reserve(nMetrics);
    for (int i = 0; i < nMetrics; ++i)
        C.metricsOwned.emplace_back(metricNames && metricNames[i] ? metricNames[i] : "");
    for (auto& s : C.metricsOwned) C.metricsCStr.push_back(s.c_str());
}

int nvmon_range_init_device(int devid, const char** metricNames, int nMetrics)
{
    RangeCtx& C = g_ctx[devid];
    CHECK_CUDA(cuInit(0));
    CHECK_CUDA(cuDeviceGet(&C.device, devid));
    CHECK_CUDA(cuDevicePrimaryCtxRetain(&C.ctx, C.device));
    CHECK_CUDA(cuCtxSetCurrent(C.ctx));

    if (metricNames && nMetrics > 0) fill_metrics(C, metricNames, nMetrics);
    else {
        static const char* kDefault[] = {"sm__ctas_launched.sum"};
        fill_metrics(C, kDefault, 1);
    }

    C.cfg.maxNumOfRanges = 4;
    C.cfg.minNestingLevel = 1;
    C.cfg.numOfNestingLevel = 1;

    C.pHost = std::make_shared<CuptiProfilerHost>();
    C.pTgt  = std::make_shared<RangeProfilerTarget>(C.ctx, C.cfg);

    std::string chipName;
    CUPTI_API_CALL(RangeProfilerTarget::GetChipName(C.device, chipName));
    CUPTI_API_CALL(RangeProfilerTarget::GetCounterAvailabilityImage(C.ctx, C.availabilityImage));

    C.pHost->SetUp(chipName, C.availabilityImage);
    CUPTI_API_CALL(C.pHost->CreateConfigImage(C.metricsCStr, C.configImage));
    CUPTI_API_CALL(C.pTgt->EnableRangeProfiler());
    CUPTI_API_CALL(C.pTgt->CreateCounterDataImage(C.metricsCStr, C.counterDataImage));

    // if (const char* rm = std::getenv("LIKWID_NVMON_REPLAY"))
    //     C.replayMode = (strcmp(rm, "user")==0) ? CUPTI_UserReplay : CUPTI_KernelReplay;

    // CUPTI_API_CALL(C.pTgt->SetConfig(
    //     CUPTI_UserRange, C.replayMode, C.configImage, C.counterDataImage
    // ));
    // C.enabled = true;
    // return 0;

    C.replayMode = CUPTI_UserReplay;

    if (const char* rm = std::getenv("LIKWID_NVMON_REPLAY")) {
        if (strcmp(rm, "kernel") == 0)
            C.replayMode = CUPTI_KernelReplay;
        else
            C.replayMode = CUPTI_UserReplay; 
    }

    CUPTI_API_CALL(C.pTgt->SetConfig(
        CUPTI_UserRange,
        C.replayMode,
        C.configImage,
        C.counterDataImage
    ));

    C.enabled = true;
    return 0;
}

int nvmon_range_region_start(int devid, const char* tag)
{
    auto it = g_ctx.find(devid);
    if (it == g_ctx.end() || !it->second.enabled) return -1;
    RangeCtx& C = it->second;

    /* The counter data image must be (re)initialized for every measurement window.
     * The CUDA samples do this in the "set up" phase; LIKWID may call start/stop
     * multiple times (marker mode, readCounters, etc.). Re-create + re-set config
     * so each measurement starts from a clean counter data image.
     */
    CUPTI_API_CALL(C.pTgt->CreateCounterDataImage(C.metricsCStr, C.counterDataImage));
    CUPTI_API_CALL(C.pTgt->SetConfig(
        CUPTI_UserRange,
        C.replayMode,
        C.configImage,
        C.counterDataImage
    ));

    CUPTI_API_CALL(C.pTgt->StartRangeProfiler());
    CUPTI_API_CALL(C.pTgt->PushRange(tag ? tag : "LIKWID"));
    return 0;
}

int nvmon_range_region_stop(int devid, const char*)
{
    auto it = g_ctx.find(devid);
    if (it == g_ctx.end() || !it->second.enabled) return -1;
    RangeCtx& C = it->second;
    CUPTI_API_CALL(C.pTgt->PopRange());
    CUPTI_API_CALL(C.pTgt->StopRangeProfiler());
    return 0;
}

int nvmon_range_decode_counter_data(int devid)
{
    auto it = g_ctx.find(devid);
    if (it == g_ctx.end() || !it->second.enabled) return -1;
    RangeCtx& C = it->second;
    CUPTI_API_CALL(C.pTgt->DecodeCounterData());
    return 0;
}

int nvmon_range_get_counter_data(int devid, const uint8_t** pData, size_t* pSize)
{
    if (!pData || !pSize) return -1;
    auto it = g_ctx.find(devid);
    if (it == g_ctx.end() || !it->second.enabled) return -1;
    RangeCtx& C = it->second;
    *pData = C.counterDataImage.data();
    *pSize = C.counterDataImage.size();
    return 0;
}

int nvmon_range_evaluate(int devid, const char*)
{
    auto it = g_ctx.find(devid);
    if (it == g_ctx.end() || !it->second.enabled) return -1;
    RangeCtx& C = it->second;

    // ここで Decode するように変更
    CUPTI_API_CALL(C.pTgt->DecodeCounterData());

    size_t numRanges = 0;
    CUPTI_API_CALL(C.pHost->GetNumOfRanges(C.counterDataImage, numRanges));
    for (size_t r = 0; r < numRanges; ++r)
        CUPTI_API_CALL(C.pHost->EvaluateCounterData(r, C.metricsCStr, C.counterDataImage));
    C.pHost->PrintProfilerRanges();
    return 0;
}

int nvmon_range_finalize_device(int devid)
{
    auto it = g_ctx.find(devid);
    if (it == g_ctx.end()) return 0;
    RangeCtx& C = it->second;
    if (C.pTgt) { (void)C.pTgt->DisableRangeProfiler(); C.pTgt.reset(); }
    if (C.pHost){ C.pHost->TearDown(); C.pHost.reset(); }
    if (C.ctx)  { cuDevicePrimaryCtxRelease(C.device); C.ctx = nullptr; }
    C.enabled = false;
    g_ctx.erase(it);
    return 0;
}
