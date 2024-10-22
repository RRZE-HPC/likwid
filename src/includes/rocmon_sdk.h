/*
 * =======================================================================================
 *
 *      Filename:  rocmon_sdk.h
 *
 *      Description:  Header File of rocmon module for ROCm >= 6.2.
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
#ifndef LIKWID_ROCMON_SDK_H
#define LIKWID_ROCMON_SDK_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include <types.h>
#include <sys/types.h>
#include <inttypes.h>

#include <likwid.h>
#include <bstrlib.h>
#include <error.h>
#include <dlfcn.h>

#include <types.h>

#include <rocmon_common_types.h>
#include <rocmon_sdk_types.h>
#include <hsa.h>
#include <hsa/hsa_ext_amd.h>


static int rocmon_sdk_initialized = FALSE;

static void *rocmon_sdk_dl_profiler_lib = NULL;
static void *rocmon_sdk_dl_hsa_lib = NULL;
//static void *rocmon_sdk_dl_rsmi_lib = NULL;


// setup function for rocprofiler sdk
//rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*);

#ifndef ROCM_CALL
#define ROCM_CALL( call, args, handleerror )                                  \
    do {                                                                \
        hsa_status_t _status = (*call##_ptr)args;                                  \
        if (_status != HSA_STATUS_SUCCESS && _status != HSA_STATUS_INFO_BREAK) {           \
            const char* err = NULL; \
            fprintf(stderr, "Error: function %s failed with error %d\n", #call, _status); \
            rocprofiler_error_string(&err); \
            fprintf(stderr, "Error: %s\n", err); \
            handleerror;                                                \
        }                                                               \
    } while (0)
#endif

#ifndef ROCPROFILER_CALL
#define ROCPROFILER_CALL( call, args, handleerror )                                                              \
    do {                                                                                              \
        rocprofiler_status_t _status = (*call##_ptr)args;                                                 \
        if(_status != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            fprintf(stderr, "Error: function %s failed with error %d\n", #call, _status); \
            handleerror;                                               \
        }                                                              \
    } while (0);
#endif
//            fprintf(stderr, "Error: %s\n", (*rocprofiler_get_status_string_ptr)(_status)); \

#ifndef DECLARE_ROCPROFILER_SDK
#define DECLARE_ROCPROFILER_SDK(funcname, funcsig) rocprofiler_status_t ROCMWEAK funcname funcsig;  rocprofiler_status_t ( *funcname##_ptr ) funcsig;
#endif


DECLARE_ROCPROFILER_SDK(rocprofiler_create_context, (rocprofiler_context_id_t*))
DECLARE_ROCPROFILER_SDK(rocprofiler_create_buffer, (rocprofiler_context_id_t, size_t, size_t, rocprofiler_buffer_policy_t, rocprofiler_buffer_tracing_cb_t, void*, rocprofiler_buffer_id_t*)); 
DECLARE_ROCPROFILER_SDK(rocprofiler_query_available_agents, (rocprofiler_agent_version_t, rocprofiler_query_available_agents_cb_t, size_t, void*)); 
DECLARE_ROCPROFILER_SDK(rocprofiler_get_timestamp, (rocprofiler_timestamp_t* ts)); 
DECLARE_ROCPROFILER_SDK(rocprofiler_query_counter_info, (rocprofiler_counter_id_t, rocprofiler_counter_info_version_id_t, void*)); 
DECLARE_ROCPROFILER_SDK(rocprofiler_start_context, (rocprofiler_context_id_t)); 
DECLARE_ROCPROFILER_SDK(rocprofiler_stop_context, (rocprofiler_context_id_t)); 
DECLARE_ROCPROFILER_SDK(rocprofiler_create_profile_config, (rocprofiler_agent_id_t,  rocprofiler_counter_id_t *, size_t,  rocprofiler_profile_config_id_t *)); 
DECLARE_ROCPROFILER_SDK(rocprofiler_destroy_profile_config, (rocprofiler_profile_config_id_t)); 
DECLARE_ROCPROFILER_SDK(rocprofiler_configure_agent_profile_counting_service, (rocprofiler_context_id_t, rocprofiler_buffer_id_t, rocprofiler_agent_id_t, rocprofiler_agent_profile_callback_t, void*)); 
DECLARE_ROCPROFILER_SDK(rocprofiler_sample_agent_profile_counting_service, (rocprofiler_context_id_t, rocprofiler_user_data_t, rocprofiler_counter_flag_t));
DECLARE_ROCPROFILER_SDK(rocprofiler_iterate_agent_supported_counters, (rocprofiler_agent_id_t, rocprofiler_available_counters_cb_t, void*));
DECLARE_ROCPROFILER_SDK(rocprofiler_flush_buffer, (rocprofiler_buffer_id_t));
DECLARE_ROCPROFILER_SDK(rocprofiler_force_configure, (rocprofiler_configure_func_t));
DECLARE_ROCPROFILER_SDK(rocprofiler_destroy_buffer, (rocprofiler_buffer_id_t));
DECLARE_ROCPROFILER_SDK(rocprofiler_context_is_active, (rocprofiler_context_id_t, int*));
DECLARE_ROCPROFILER_SDK(rocprofiler_create_callback_thread, (rocprofiler_callback_thread_t*));
DECLARE_ROCPROFILER_SDK(rocprofiler_assign_callback_thread, (rocprofiler_buffer_id_t, rocprofiler_callback_thread_t));

const char *rocprofiler_get_status_string(rocprofiler_status_t);
const char * (*rocprofiler_get_status_string_ptr)(rocprofiler_status_t);

#ifndef DECLAREFUNC_HSA
#define DECLAREFUNC_HSA(funcname, funcsig) hsa_status_t ROCMWEAK funcname funcsig;  hsa_status_t ( *funcname##_ptr ) funcsig;
#endif
DECLAREFUNC_HSA(hsa_init, ());
DECLAREFUNC_HSA(hsa_shut_down, ());


typedef struct {
    rocprofiler_agent_t *agents;
    int num_agents;
} _rocmon_sdk_count_agents_cb_data;

rocprofiler_status_t _rocmon_sdk_count_agents_cb(rocprofiler_agent_version_t agents_ver,
                                const void**                agents_arr,
                                size_t                      num_agents,
                                void*                       udata)
{
    int gpu_agents = 0;
    RocmonContext **stat_context = (RocmonContext **)udata;
    RocmonContext* context = *stat_context;
    RocmonDevice* devices = malloc(num_agents * sizeof(RocmonDevice));
    if (!devices)
    {
        return ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    memset(devices, 0, num_agents * sizeof(RocmonDevice));

    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Found %d ROCm agents, num_agents);
    for(size_t i = 0; i < num_agents; ++i)
    {
        const rocprofiler_agent_t* in_agent = agents_arr[i];
        if (in_agent->type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Adding AMD GPU at index %d, gpu_agents);
            RocmonDevice* device = &devices[gpu_agents];
            device->agent = (rocprofiler_agent_t)*in_agent;
            device->deviceId = in_agent->logical_node_type_id;
            gpu_agents++;
        }
    }
    context->devices = devices;
    context->numDevices = gpu_agents;
    return ROCPROFILER_STATUS_SUCCESS;
}


typedef struct {
    rocprofiler_counter_info_v0_t *counters;
    int num_counters;
} _rocmon_sdk_fill_agent_counters_cb_data;

static void
_rocmon_sdk_free_agent_counters_internal(int num_counters, rocprofiler_counter_info_v0_t* counters)
{
    if ((num_counters < 0) || (!counters))
    {
        return;
    }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Freeing %d counters, num_counters);
    for (int i = 0; i < num_counters; i++)
    {
        rocprofiler_counter_info_v0_t* info = &counters[i];
        if (info)
        {
            if (info->name) free((char*)info->name);
            if (info->description) free((char*)info->description);
            if (info->block) free((char*)info->block);
            if (info->expression) free((char*)info->expression);
        }
    }
    free(counters);
}


rocprofiler_status_t
_rocmon_sdk_fill_agent_counters_cb(rocprofiler_agent_id_t agent,
                                    rocprofiler_counter_id_t* counters,
                                    size_t                    num_counters,
                                    void*                     udata)
{
    _rocmon_sdk_fill_agent_counters_cb_data *data = (_rocmon_sdk_fill_agent_counters_cb_data*)udata;

    rocprofiler_counter_info_v0_t* out = malloc(num_counters * sizeof(rocprofiler_counter_info_v0_t));
    if (!out)
    {
        return -ENOMEM;
    }
    for (int i = 0; i < num_counters; i++)
    {
        rocprofiler_counter_info_v0_t info;
        rocprofiler_status_t stat = (*rocprofiler_query_counter_info_ptr)(counters[i], (rocprofiler_counter_info_version_id_t)ROCPROFILER_COUNTER_INFO_VERSION_0, &info);
        if (stat != ROCPROFILER_STATUS_SUCCESS)
        {
            ERROR_PRINT(Failed to query counter info for %d, i);
            for (int j = 0; j < i; j++)
            {
                free((char*)out[j].name);
                free((char*)out[j].description);
            }
            free(out);
            return -EFAULT;
        }
        //ROCPROFILER_CALL(rocprofiler_query_counter_info, (counters[i], ROCPROFILER_COUNTER_INFO_VERSION_0, &info),
        /*{
            free(out);
            return -EFAULT;
        });*/
        int namelen = strlen(info.name)+1;
        int desclen = strlen(info.description)+1;
        out[i].name = malloc(namelen * sizeof(char));
        if (!out[i].name)
        {
            _rocmon_sdk_free_agent_counters_internal(i, out);
            return -ENOMEM;
        }
        out[i].description = malloc(desclen * sizeof(char));
        if (!out[i].description)
        {
            free((char*)out[i].name);
            _rocmon_sdk_free_agent_counters_internal(i, out);
            return -ENOMEM;
        }
        out[i].block = malloc((strlen(info.block)+1) * sizeof(char));
        if (!out[i].block)
        {
            free((char*)out[i].name);
            free((char*)out[i].description);
            _rocmon_sdk_free_agent_counters_internal(i, out);
            return -ENOMEM;
        }
        out[i].expression = malloc((strlen(info.expression)+1) * sizeof(char));
        if (!out[i].expression)
        {
            free((char*)out[i].name);
            free((char*)out[i].description);
            free((char*)out[i].block);
            _rocmon_sdk_free_agent_counters_internal(i, out);
            return -ENOMEM;
        }
        int ret = 0;
        ret = snprintf((char*)out[i].name, namelen-1, "%s", info.name);
        ret = snprintf((char*)out[i].description, desclen-1, "%s", info.description);
        out[i].id = info.id;
        out[i].is_constant = info.is_constant;
        out[i].is_derived = info.is_derived;
    }
    data->counters = out;
    data->num_counters = num_counters;
    return ROCPROFILER_STATUS_SUCCESS;
}

int _rocmon_sdk_fill_agent_counters(RocmonDevice *device)
{
    _rocmon_sdk_fill_agent_counters_cb_data fill_data = {
        .counters = NULL,
        .num_counters = 0,
    };

    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Getting counters for agent %d, device->deviceId);
    rocprofiler_status_t _status = (rocprofiler_iterate_agent_supported_counters_ptr)(device->agent.id, _rocmon_sdk_fill_agent_counters_cb, &fill_data);
    if (_status != ROCPROFILER_STATUS_SUCCESS)
    {
        return -EFAULT;
    }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Agent %d provides %d counters, device->deviceId, fill_data.num_counters);
    device->sdk_rocMetrics = fill_data.counters;
    device->numRocMetrics = fill_data.num_counters;

    return ROCPROFILER_STATUS_SUCCESS;
}


static void
_rocmon_sdk_free_agent_counters(RocmonDevice *device)
{
    if (!device->sdk_rocMetrics)
    {
        return;
    }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Freeing counters for agent %d, device->deviceId);
    _rocmon_sdk_free_agent_counters_internal(device->numRocMetrics, device->sdk_rocMetrics);
    device->sdk_rocMetrics = NULL;
    device->numRocMetrics = 0;
}


typedef struct {
    rocprofiler_context_id_t* context;
    rocprofiler_agent_t agent;
    RocmonEventResultList* result;
} rocmon_sdk_read_buffers_cb;

static void
_rocmon_sdk_read_buffers(rocprofiler_context_id_t context,
                  rocprofiler_buffer_id_t buffer,
                  rocprofiler_record_header_t** headers,
                  size_t                        num_headers,
                  void*                         udata,
                  uint64_t)
{
    rocmon_sdk_read_buffers_cb* cbdata = (rocmon_sdk_read_buffers_cb*)udata;

/*    if (cbdata->result->numResults == 0)*/
/*    {*/
/*        cbdata->result->results = malloc(sizeof(RocmonEventResult))*/
/*    }*/
    printf("_rocmon_sdk_read_buffers\n");
    for (int i = 0; i < num_headers; i++)
    {
        rocprofiler_record_header_t* h = headers[i];
        if(h->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS && h->kind == ROCPROFILER_COUNTER_RECORD_VALUE)
        {
            rocprofiler_record_counter_t* r = h->payload;
            printf("Counter ID %d Value %f Dispatch %ld\n", r->id, r->counter_value, r->dispatch_id);
        }
    }


/*    RocmonContext* mycontext = *cbdata->context;*/
/*    for (int i = 0; i < mycontext->numDevices; i++)*/
/*    {*/
/*        RocmonDevice* device = &mycontext->devices[i];*/
/*        if (device->agent.id.handle == cbdata->agent.id.handle)*/
/*        {*/
/*            RocmonEventResultList* groupResults = &device->groupResults[device->activeGroup];*/

/*            for(int i = 0; i < num_headers; ++i)*/
/*            {*/
/*                rocprofiler_record_header_t* h = headers[i];*/
/*                if(h->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS && h->kind == ROCPROFILER_COUNTER_RECORD_VALUE)*/
/*                {*/
/*                    rocprofiler_record_counter_t* r = h->payload;*/
/*                    if (r->id >= 0 && r->id < groupResults->numResults)*/
/*                    {*/
/*                        RocmonEventResult* eventResult = &cbdata->result->results[r->id];*/
/*                        double diff = r->counter_value - eventResult->fullValue;*/
/*                        eventResult->lastValue = eventResult->fullValue;*/
/*                        eventResult->fullValue += diff;*/
/*                    }*/
/*                }*/
/*            }*/
/*        }*/
/*    }*/

    return;
}


int
tool_init(rocprofiler_client_finalize_t fini, void* udata)
{
    rocprofiler_status_t stat = ROCPROFILER_STATUS_SUCCESS;
    RocmonContext** stat_context = (RocmonContext**)udata;
    RocmonContext* context = *stat_context;
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Running tool_init);

    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Initialize HSA);
    hsa_status_t hstat = (*hsa_init_ptr)();
    if (hstat != HSA_STATUS_SUCCESS)
    {
        return -EFAULT;
    }

    //ROCPROFILER_CALL(rocprofiler_query_available_agents, (ROCPROFILER_AGENT_INFO_VERSION_0, _rocmon_sdk_count_agents_cb, sizeof(rocprofiler_agent_t), &agent_count), return -EFAULT;);
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Querying available agents);
    stat = (*rocprofiler_query_available_agents_ptr)(ROCPROFILER_AGENT_INFO_VERSION_0, _rocmon_sdk_count_agents_cb, sizeof(rocprofiler_agent_t), udata);
    if (stat != ROCPROFILER_STATUS_SUCCESS)
    {
        return -EFAULT;
    }
    if (context->numDevices == 0)
    {
        FREE_IF_NOT_NULL(context->devices);
        return -1;
    }

    for (int i = 0; i < context->numDevices; i++)
    {
        rocprofiler_context_id_t device_context;
        rocprofiler_buffer_id_t buffer;
        rocprofiler_callback_thread_t thread;
        RocmonDevice* device = &context->devices[i];
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Creating context for device %d, device->deviceId);
        stat = (*rocprofiler_create_context_ptr)(&device_context);
        if (stat != ROCPROFILER_STATUS_SUCCESS)
        {
            errno = EFAULT;
            ERROR_PRINT(Failed to create context for device %d: %s, device->deviceId, (*rocprofiler_get_status_string_ptr)(stat));
            FREE_IF_NOT_NULL(context->devices);
            return -EFAULT;
        }
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Creating buffer for device %d, device->deviceId);
        stat = (*rocprofiler_create_buffer_ptr)(device_context, 100, 50, ROCPROFILER_BUFFER_POLICY_LOSSLESS, _rocmon_sdk_read_buffers, udata, &buffer);
        if (stat != ROCPROFILER_STATUS_SUCCESS)
        {
            errno = EFAULT;
            ERROR_PRINT(Failed to create buffer for device %d: %s, device->deviceId, (*rocprofiler_get_status_string_ptr)(stat));
            FREE_IF_NOT_NULL(context->devices);
            return -EFAULT;
        }
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Creating callback thread for device %d, device->deviceId);
        stat = (*rocprofiler_create_callback_thread_ptr)(&thread);
        if (stat != ROCPROFILER_STATUS_SUCCESS)
        {
            errno = EFAULT;
            ERROR_PRINT(Failed to create callback thread for device %d: %s, device->deviceId, (*rocprofiler_get_status_string_ptr)(stat));
            FREE_IF_NOT_NULL(context->devices);
            return -EFAULT;
        }
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Assign callback thread to buffer for device %d, device->deviceId);
        stat = (*rocprofiler_assign_callback_thread_ptr)(buffer, thread);
        if (stat != ROCPROFILER_STATUS_SUCCESS)
        {
            errno = EFAULT;
            ERROR_PRINT(Failed to create callback thread for device %d: %s, device->deviceId, (*rocprofiler_get_status_string_ptr)(stat));
            FREE_IF_NOT_NULL(context->devices);
            return -EFAULT;
        }
        
        device->sdk_context = device_context;
        device->buffer = buffer;
        device->thread = thread;
    }
    return 0;
}


void
tool_fini(void* udata)
{
    rocprofiler_status_t stat = ROCPROFILER_STATUS_SUCCESS;
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Running tool_fini);
    RocmonContext** stat_context = (RocmonContext**)udata;
    RocmonContext* context = *stat_context;
    if ((!context) || (!context->devices) || (context->numDevices == 0))
    {
        return;
    }
    for (int i = 0; i < context->numDevices; i++)
    {
        RocmonDevice* device = &context->devices[i];
        int active = 0;
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Checking context for device %d, device->deviceId);
        stat = (*rocprofiler_context_is_active_ptr)(device->sdk_context, &active);
        if (active)
        {
            ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Stopping context for device %d, device->deviceId);
            stat = (*rocprofiler_stop_context_ptr)(device->sdk_context);
            if (stat != ROCPROFILER_STATUS_SUCCESS)
            {
                ERROR_PRINT(Failed to stop context for device %d: %s, device->deviceId, (*rocprofiler_get_status_string_ptr)(stat));
            }
        }
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Flushing buffer for device %d, device->deviceId);
        stat = (*rocprofiler_flush_buffer_ptr)(device->buffer);
        if (stat != ROCPROFILER_STATUS_SUCCESS)
        {
            ERROR_PRINT(Failed to flush buffer for device %d: %s, device->deviceId, (*rocprofiler_get_status_string_ptr)(stat));
        }
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Destroying buffer for device %d, device->deviceId);
        stat = (*rocprofiler_destroy_buffer_ptr)(device->buffer);
        if (stat != ROCPROFILER_STATUS_SUCCESS)
        {
            ERROR_PRINT(Failed to destroy buffer for device %d: %s, device->deviceId, (*rocprofiler_get_status_string_ptr)(stat));
        }
        _rocmon_sdk_free_agent_counters(device);
    }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Shutdown HSA);
    (*hsa_shut_down_ptr)();
}

void
_rocmon_sdk_set_profile(rocprofiler_context_id_t                 context_id,
                        rocprofiler_agent_id_t                   agent,
                        rocprofiler_agent_set_profile_callback_t set_config,
                        void* udata)
{
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, _rocmon_sdk_set_profile);
    RocmonDevice* device = (RocmonDevice*) udata;
    if (device->agent.id.handle == agent.handle)
    {
        if (device->activeGroup >= 0 && device->activeGroup < device->numProfiles)
        {
            ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Setting profile %d for device %d, device->activeGroup, device->deviceId);
            set_config(context_id, device->profiles[device->activeGroup]);
        }
        else
        {
            ERROR_PRINT(Invalid active group for device %d, device->deviceId);
        }
    }
    else
    {
        ERROR_PRINT(Mismatch between device %s agent and given agent, device->deviceId);
    }
    return;
}



static int
_rocmon_sdk_link_libraries()
{
    #define DLSYM_AND_CHECK( dllib, name ) name##_ptr = dlsym( dllib, #name ); if ( dlerror() != NULL ) { ERROR_PRINT(Failed to link  #name); return -1; }
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Linking AMD ROCMm SDK libraries);
    dlerror();
    // Need to link in the ROCm HSA libraries
    rocmon_sdk_dl_hsa_lib = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!rocmon_sdk_dl_hsa_lib)
    {
        ERROR_PRINT(ROCm HSA library libhsa-runtime64.so not found: %s, dlerror());
        return -1;
    }

    // Need to link in the Rocprofiler libraries
    rocmon_sdk_dl_profiler_lib = dlopen("librocprofiler-sdk.so", RTLD_NOW | RTLD_GLOBAL);
    if (!rocmon_sdk_dl_profiler_lib)
    {
        // Delete last error
        dlerror();
        rocmon_sdk_dl_profiler_lib = dlopen("librocprofiler-sdk.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!rocmon_sdk_dl_profiler_lib)
        {
            ERROR_PRINT(Rocprofiler library librocprofiler-sdk.so not found: %s, dlerror());
            return -1;
        }
    }

    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_create_context);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_get_status_string);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_create_buffer);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_query_available_agents);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_get_timestamp);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_start_context);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_stop_context);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_create_profile_config);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_destroy_profile_config);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_configure_agent_profile_counting_service);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_iterate_agent_supported_counters);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_flush_buffer);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_query_counter_info);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_sample_agent_profile_counting_service);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_force_configure);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_destroy_buffer);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_context_is_active);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_create_callback_thread);
    DLSYM_AND_CHECK(rocmon_sdk_dl_profiler_lib, rocprofiler_assign_callback_thread);

    DLSYM_AND_CHECK(rocmon_sdk_dl_hsa_lib, hsa_init);
    DLSYM_AND_CHECK(rocmon_sdk_dl_hsa_lib, hsa_shut_down);

    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Linking AMD ROCMm SDK libraries done);
    return 0;
}


rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* client_id)
{
    client_id->name = "LIKWID";
    static rocprofiler_tool_configure_result_t config_result = {
        .size = sizeof(rocprofiler_tool_configure_result_t),
        .initialize = tool_init,
        .finalize = tool_fini,
        .tool_data = &rocmon_context,
    };
    return &config_result;
}

int
rocmon_sdk_init(RocmonContext* context, int numGpus, const int* gpuIds)
{
    int ret = 0;
    rocprofiler_status_t stat = ROCPROFILER_STATUS_SUCCESS;
    if ((numGpus < 0) || (!gpuIds) || (!context))
    {
        return -EINVAL;
    }
    if (rocmon_sdk_initialized)
    {
        return 0;
    }

    // initialize libraries
    ret = _rocmon_sdk_link_libraries();
    if (ret < 0)
    {
        //ERROR_PLAIN_PRINT(Failed to initialize libraries);
        return ret;
    }

    stat = (*rocprofiler_force_configure_ptr)(rocprofiler_configure);
    if (stat != ROCPROFILER_STATUS_SUCCESS)
    {
        return -EFAULT;
    }

    if (context->numDevices == 0)
    {
        errno = ENODEV;
        ERROR_PRINT(Cannot ROCm GPUs);
        return -ENODEV;
    }

    RocmonDevice* devices = malloc(numGpus * sizeof(RocmonDevice));
    if (!devices)
    {
        return -ENOMEM;
    }
    memset(devices, 0, numGpus * sizeof(RocmonDevice));

    for (int i = 0; i < numGpus; i++)
    {
        int idx = -1;
        for (int j = 0; j < context->numDevices; j++)
        {
            RocmonDevice* device = &context->devices[j];
            if (gpuIds[i] == device->deviceId)
            {
                idx = j;
                break;
            }
        }
        if (idx >= 0)
        {
            memcpy(&devices[i], &context->devices[idx], sizeof(RocmonDevice));
            RocmonDevice* out = &devices[i];
/*            RocmonDevice* in = &context->devices[idx];*/
/*            out->agent = in->agent;*/
/*            printf("%d -> %d\n", in->agent.id.handle, out->agent.id.handle);*/
/*            out->thread = in->thread;*/
/*            out->buffer = in->buffer;*/
/*            printf("%d -> %d\n", in->buffer.handle, out->buffer.handle);*/
/*            out->sdk_context = in->sdk_context;*/
/*            printf("%d -> %d\n", in->sdk_context.handle, out->sdk_context.handle);*/
            ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Fill agent counters for device %d, out->deviceId);
            ret = _rocmon_sdk_fill_agent_counters(out);
            if (ret < 0)
            {
                errno = -ret;
                ERROR_PRINT(Failed to fill events for device %d: %s, out->deviceId, (*rocprofiler_get_status_string_ptr)(stat));
            }
        }
        else
        {
            errno = ENODEV;
            ERROR_PRINT(Cannot find ROCm GPU %d, gpuIds[i]);
            free(devices);
            return -ENODEV;
        }
    }
    free(context->devices);
    context->devices = devices;
    context->numDevices = numGpus;

    rocmon_sdk_initialized = TRUE;
    return 0;
}


void
rocmon_sdk_finalize(RocmonContext* context)
{
    if (context)
    {
        if (context->devices)
        {
            for (int i = 0; i < context->numDevices; i++)
            {
                //free device i
                RocmonDevice* dev = &context->devices[i];
                if (dev->sdk_activeRocEvents)
                {
                    free(dev->sdk_activeRocEvents);
                    dev->sdk_activeRocEvents = NULL;
                    dev->numActiveRocEvents = 0;
                }
                if (dev->sdk_rocMetrics)
                {
                    _rocmon_sdk_free_agent_counters_internal(dev->numRocMetrics, dev->sdk_rocMetrics);
                    dev->sdk_rocMetrics = NULL;
                    dev->numRocMetrics = 0;
                }
                if (dev->profiles)
                {
                    for (int i = 0; i < dev->numProfiles; i++)
                    {
                        (*rocprofiler_destroy_profile_config_ptr)(dev->profiles[i]);
                    }
                }
            }
        }
/*        if (context->sdk_agents)*/
/*        {*/
/*            free(context->sdk_agents);*/
/*            context->sdk_agents = NULL;*/
/*            free(context->sdk_agent_buffers);*/
/*            context->sdk_agent_buffers = NULL;*/
/*            context->num_sdk_agents = 0;*/
/*        }*/
    }
    rocmon_sdk_initialized = 0;
    return;
}



static int
_rocmon_setupCounters_rocprofiler_sdk(RocmonDevice* device, const char** events, int numEvents)
{
    rocprofiler_profile_config_id_t profile;
    rocprofiler_status_t stat = ROCPROFILER_STATUS_SUCCESS;
    if ((!device) || (!events) || (numEvents <= 0))
    {
        return -EINVAL;
    }

    int num_counters = 0;
    rocprofiler_counter_id_t* counters = malloc(numEvents * sizeof(rocprofiler_counter_id_t));
    if (!counters)
    {
        return -ENOMEM;
    }

    for (int i = 0; i < numEvents; i++)
    {
        int found = -1;
        for (int j = 0; j < device->numRocMetrics; j++)
        {
            rocprofiler_counter_info_v0_t* m = &device->sdk_rocMetrics[j];
            if (strncmp(events[i], m->name, strlen(m->name)) == 0)
            {
                found = j;
                break;
            }
        }
        if (found >= 0)
        {
            counters[num_counters++] = device->sdk_rocMetrics[found].id;
        }
        else
        {
            ERROR_PRINT(Unknown ROCm event %s, events[i]);
        }
    }

    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Creating profile for %d event(s) for device %d, num_counters, device->deviceId);
    stat = (*rocprofiler_create_profile_config_ptr)(device->agent.id, counters, num_counters * sizeof(rocprofiler_counter_id_t), &profile);
    if (stat != ROCPROFILER_STATUS_SUCCESS)
    {
        ERROR_PRINT(Failed to create profile: %s, (*rocprofiler_get_status_string_ptr)(stat));
        FREE_IF_NOT_NULL(counters);
        return -ENOMEM;
    }

    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Increasing profile space to %d for device %d, device->numProfiles + 1, device->deviceId);
    rocprofiler_profile_config_id_t* profiles = realloc(device->profiles, (device->numProfiles+1) * sizeof(rocprofiler_profile_config_id_t));
    if (!profiles)
    {
        (*rocprofiler_destroy_profile_config_ptr)(profile);
        FREE_IF_NOT_NULL(counters);
        return -ENOMEM;
    }
    device->profiles = profiles;

    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Adding profile %d at idx %d for device %d, device->numProfiles, device->numProfiles, device->deviceId);
    device->profiles[device->numProfiles++] = profile;
    FREE_IF_NOT_NULL(counters);
    return 0;
}

int
rocmon_sdk_setupCounters(RocmonContext* context, int gid)
{
    int ret = 0;
    int numRocEvents = 0;
    const char **rocEvents = NULL;
    // Check arguments
    if (gid < 0 || gid >= context->numActiveGroups)
    {
        return -EINVAL;
    }
    
    // Ensure rocmon is initialized
    if (!rocmon_sdk_initialized)
    {
        ERROR_PRINT(Rocmon SDK not initialized);
        return -EFAULT;
    }

    // Get group info
    GroupInfo* group = &context->groups[gid];

    // Allocate memory for string arrays
    rocEvents = (const char**) malloc(group->nevents * sizeof(const char*));
    if (rocEvents == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate rocEvent name array);
        return -ENOMEM;
    }

    // Go through each event and sort it
    for (int i = 0; i < group->nevents; i++)
    {
        const char* name = group->events[i];
        if (strncmp(name, "ROCP_", 5) == 0)
        {
            // Rocprofiler event
            rocEvents[numRocEvents] = name + 5; // +5 removes 'ROCP_' prefix
            numRocEvents++;
        }
    }
    if (numRocEvents == 0)
    {
        free(rocEvents);
        return 0;
    }

    // Add events to each device
    //rocmon_context->activeGroup = gid;
    for (int i = 0; i < context->numDevices; i++)
    {
        RocmonDevice* device = &context->devices[i];

        // Add rocprofiler events
        ROCMON_DEBUG_PRINT(DEBUGLEV_INFO, SETUP ROCPROFILER WITH %d events, numRocEvents);
        ret = _rocmon_setupCounters_rocprofiler_sdk(device, rocEvents, numRocEvents);
        if (ret < 0)
        {
            if (rocEvents) free(rocEvents);
            return ret;
        }

    }
    // Cleanup
    free(rocEvents);

    return 0;
}

static int _rocmon_sdk_get_timestamp(uint64_t* timestamp)
{
    rocprofiler_timestamp_t ts;
    rocprofiler_status_t stat = (*rocprofiler_get_timestamp_ptr)(&ts);
    if (stat != ROCPROFILER_STATUS_SUCCESS)
    {
        ERROR_PRINT(Failed to get timestamp: %s, (*rocprofiler_get_status_string_ptr)(stat));
        return -EFAULT;
    }


    *timestamp = (uint64_t) ts;
    return 0;
}

static int
_rocmon_startCounters_rocprofiler_sdk(RocmonDevice* device)
{
    int active = 0;
    rocprofiler_status_t stat = ROCPROFILER_STATUS_SUCCESS;
    //ROCPROFILER_CALL(rocprofiler_configure_agent_profile_counting_service, (device->sdk_context, device->buffer, device->agent.id, _rocmon_sdk_set_profile, NULL), \
        //ROCPROFILER_CALL(rocprofiler_destroy_profile_config, (profile), free(counters); return -EFAULT;); \
        free(counters); return -ENOMEM);

    // if not running
    stat = (*rocprofiler_context_is_active)(device->sdk_context, &active);
    if (stat != ROCPROFILER_STATUS_SUCCESS)
    {
        ERROR_PRINT(Failed to check ROCm context for device %d: %s, device->deviceId, (*rocprofiler_get_status_string_ptr)(stat));
    }
    if (!active)
    {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Configuring counting service for device %d, device->deviceId);
        stat = (*rocprofiler_configure_agent_profile_counting_service_ptr)(device->sdk_context, device->buffer, device->agent.id, _rocmon_sdk_set_profile, device);
        if (stat != ROCPROFILER_STATUS_SUCCESS)
        {
            ERROR_PRINT(Failed to configure counting service for device %d: %s, device->deviceId, (*rocprofiler_get_status_string_ptr)(stat));
            return -EFAULT;
        }
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Starting context for device %d, device->deviceId);
        stat = (*rocprofiler_start_context_ptr)(device->sdk_context);
        if (stat != ROCPROFILER_STATUS_SUCCESS)
        {
            ERROR_PRINT(Failed to start ROCm context for device %d: %s, device->deviceId, (*rocprofiler_get_status_string_ptr)(stat));
            return -EFAULT;
        }
    }
    return 0;
}

int
rocmon_sdk_startCounters(RocmonContext* context)
{
    int ret = 0;
    uint64_t timestamp = 0;
    // Ensure rocmon is initialized
    if (!rocmon_sdk_initialized)
    {
        ERROR_PRINT(Rocmon SDK not initialized);
        return -EFAULT;
    }

    // Get timestamp
    if (ret = _rocmon_sdk_get_timestamp(&timestamp))
    {
        return ret;
    }

    // Start counters on each device
    for (int i = 0; i < context->numDevices; i++)
    {
        RocmonDevice* device = &context->devices[i];
        device->time.start = timestamp;
        device->time.read = timestamp;

        // Start rocprofiler events
        ret = _rocmon_startCounters_rocprofiler_sdk(device);
        if (ret < 0) return ret;

    }

    return 0;
}


static int
_rocmon_stopCounters_rocprofiler_sdk(RocmonDevice* device)
{
    int active = 0;
    rocprofiler_status_t stat = ROCPROFILER_STATUS_SUCCESS;
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Checking context for device %d, device->deviceId);
    stat = (*rocprofiler_context_is_active)(device->sdk_context, &active);
    if (stat != ROCPROFILER_STATUS_SUCCESS)
    {
        ERROR_PRINT(Failed to check ROCm context for device %d: %s, device->deviceId, (*rocprofiler_get_status_string_ptr)(stat));
    }
    if (active)
    {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Stopping context for device %d, device->deviceId);
        stat = (*rocprofiler_stop_context_ptr)(device->sdk_context);
        if (stat != ROCPROFILER_STATUS_SUCCESS)
        {
            ERROR_PRINT(Failed to stop ROCm context for device %d: %s, device->deviceId, (*rocprofiler_get_status_string_ptr)(stat));
        }
/*        stat = (*rocprofiler_flush_buffer_ptr)(device->buffer);*/
/*        if (stat != ROCPROFILER_STATUS_SUCCESS)*/
/*        {*/
/*            ERROR_PRINT(Failed to flush buffer for device %d: %s, device->deviceId, (*rocprofiler_get_status_string_ptr)(stat));*/
/*        }*/
    }
    return 0;
}

int
rocmon_sdk_stopCounters(RocmonContext* context)
{
    int ret = 0;
    uint64_t t = 0;
    // Ensure rocmon is initialized
    if (!rocmon_sdk_initialized)
    {
        ERROR_PRINT(Rocmon SDK not initialized);
        return -EFAULT;
    }
    // Read counters
    ret = _rocmon_sdk_get_timestamp(&t);
    if (ret < 0)
    {
        return ret;
    }
    for (int i = 0; i < context->numDevices; i++)
    {
        RocmonDevice* device = &context->devices[i];

        // Stop rocprofiler events
        ret = _rocmon_stopCounters_rocprofiler_sdk(device);
        if (ret < 0) return ret;
        device->time.stop = t;
    }

    return 0;
}

static int
_rocmon_readCounters_rocprofiler_sdk(RocmonDevice* device)
{
    int active = 0;
    rocprofiler_status_t stat = ROCPROFILER_STATUS_SUCCESS;
    // do read
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Checking context for device %d, device->deviceId);
    stat = (*rocprofiler_context_is_active)(device->sdk_context, &active);
    if (stat != ROCPROFILER_STATUS_SUCCESS)
    {
        ERROR_PRINT(Failed to check ROCm context for device %d: %s, device->deviceId, (*rocprofiler_get_status_string_ptr)(stat));
    }
    if (active)
    {
        rocprofiler_user_data_t udata = {
            .value = 0,
            .ptr = NULL,
        };
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Sampling counting service for device %d, device->deviceId);
        stat = (*rocprofiler_sample_agent_profile_counting_service_ptr)(device->sdk_context, udata, ROCPROFILER_COUNTER_FLAG_NONE);
        if (stat != ROCPROFILER_STATUS_SUCCESS)
        {
            ERROR_PRINT(Failed to sample counting service for device %d: %s, device->deviceId, (*rocprofiler_get_status_string_ptr)(stat));
            return -EFAULT;
        }
    }
/*    stat = (*rocprofiler_flush_buffer_ptr)(device->buffer);*/
/*    if (stat != ROCPROFILER_STATUS_SUCCESS)*/
/*    {*/
/*        ERROR_PRINT(Failed to flush buffer for device %d: %s, device->deviceId, (*rocprofiler_get_status_string_ptr)(stat));*/
/*        return -EFAULT;*/
/*    }*/
    return 0;
}


int
rocmon_sdk_readCounters(RocmonContext* context)
{
    int ret = 0;
    uint64_t t = 0;
    // Ensure rocmon is initialized
    if (!rocmon_sdk_initialized)
    {
        return -EFAULT;
    }
    ret = _rocmon_sdk_get_timestamp(&t);
    if (ret < 0)
    {
        return ret;
    }

    for (int i = 0; i < context->numDevices; i++)
    {
        RocmonDevice* device = &context->devices[i];
        // Read counters
        ret = _rocmon_readCounters_rocprofiler_sdk(device);
        if (ret < 0) return ret;
        device->time.read = t;
    }

    return 0;
}




int
rocmon_sdk_getEventsOfGpu(RocmonContext* context, int gpuIdx, EventList_rocm_t* list)
{
    EventList_rocm_t tmpList = NULL;
    Event_rocm_t* tmpEventList = NULL;
    // Ensure rocmon is initialized
    if (!rocmon_sdk_initialized)
    {
        return -EFAULT;
    }
    // Validate args
    if ((gpuIdx < 0) || (gpuIdx > context->numDevices) || (!list))
    {
        return -EINVAL;
    }

    RocmonDevice* device = &context->devices[gpuIdx];

    if (*list)
    {
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Reusing existing event list);
        tmpList = *list;
    }
    else
    {
        // Allocate list structure
        ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Allocate new event list);
        EventList_rocm_t tmpList = (EventList_rocm_t) malloc(sizeof(EventList_rocm));
        if (tmpList == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate event list);
            return -ENOMEM;
        }
        tmpList->numEvents = 0;
        tmpList->events = NULL;
    }

    // Get number of events
    ROCMON_DEBUG_PRINT(DEBUGLEV_DEVELOP, Add %d RocProfiler SDK events, device->numRocMetrics);
    if (device->numRocMetrics == 0)
    {
        // No events -> return list
        *list = tmpList;
        return 0;
    }
    // (Re-)Allocate event array
    tmpEventList = realloc(tmpList->events, (tmpList->numEvents + device->numRocMetrics) * sizeof(Event_rocm_t));
    if (!tmpEventList)
    {
        if (!*list) free(tmpList);
        ERROR_PLAIN_PRINT(Cannot allocate events for event list);
        return -ENOMEM;
    }
    tmpList->events = tmpEventList;
    int startindex = tmpList->numEvents;

    // Copy rocprofiler event information
    for (int i = 0; i < device->numRocMetrics; i++)
    {
        rocprofiler_counter_info_v0_t* event = &device->sdk_rocMetrics[i];
        Event_rocm_t* out = &tmpList->events[startindex + i];
        int len;

        // Copy name
        len = strlen(event->name) + 5 /* Prefix */ + 1 /* NULL byte */;
        out->name = (char*) malloc(len);
        if (out->name)
        {
            snprintf(out->name, len, "ROCP_%s", event->name);
        }

        // Copy description
        len = strlen(event->description) + 1 /* NULL byte */;
        out->description = (char*) malloc(len);
        if (out->description)
        {
            snprintf(out->description, len, "%s", event->description);
        }
        tmpList->numEvents++;
    }
    *list = tmpList;
    return 0;
}




int
rocmon_sdk_switchActiveGroup(RocmonContext* context, int newGroupId)
{
    int ret;

    ret = rocmon_sdk_stopCounters(context);
    if (ret < 0)
    {
        return ret;
    }

    ret = rocmon_sdk_setupCounters(context, newGroupId);
    if (ret < 0)
    {
        return ret;
    }

    ret = rocmon_sdk_startCounters(context);
    if (ret < 0)
    {
        return ret;
    }

    return 0;
}





#endif /* LIKWID_ROCMON_SDK_H */

