// SPDX-License-Identifier: GPL-3.0
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int nvmon_range_init_device(int devid, const char** metricNames, int nMetrics);
int nvmon_range_region_start(int devid, const char* tag);
int nvmon_range_region_stop (int devid, const char* tag);
int nvmon_range_evaluate   (int devid, const char* tag);
int nvmon_range_finalize_device(int devid);

int nvmon_range_decode_counter_data(int devid);
int nvmon_range_get_counter_data(int devid, const uint8_t** pData, size_t* pSize);
#ifdef __cplusplus
} // extern "C"
#endif
