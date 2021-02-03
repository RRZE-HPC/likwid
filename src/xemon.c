/*
 * =======================================================================================
 *
 *      Filename:  xemon.c
 *
 *      Description:  Main implementation of the performance monitoring module
 *                    for Intel GPUs
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tg), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 RRZE, University Erlangen-Nuremberg
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

#include <likwid.h>
#include <topology.h>
#include <error.h>

#define LIKWID_XE_CALL( call, handleerror )                                    \
    do {                                                                \
        ze_result_t _status = (call);                                      \
        if (_status != ZE_RESULT_SUCCESS) {                                  \
            fprintf(stderr, "Error: function %s failed with error %d.\n", #call, _status); \
            handleerror;                                                \
        }                                                               \
    } while (0)
    
#ifndef DECLARE_XEFUNC
#define XEAPIWEAK __attribute__( ( weak ) )
#define DECLARE_XEFUNC(funcname, funcsig) ze_result_t XEAPIWEAK funcname funcsig;  ze_result_t( *funcname##Ptr ) funcsig;
#endif

#ifndef DLSYM_AND_CHECK
#define DLSYM_AND_CHECK( dllib, name ) dlsym( dllib, name ); if ( dlerror() != NULL ) { return -1; }
#endif

DECLARE_XEFUNC(zetMetricGroupGet, (zet_device_handle_t, uint32_t *, zet_metric_group_handle_t *));

static int xemon_initialized = 0;
XemonGroupSet* xeGroupSet = NULL;
int likwid_xemon_verbosity = DEBUGLEV_ONLY_ERROR;

LikwidXeResults* xeMarkerResults = NULL;
int xeMarkerRegions = 0;

static void *dl_xet_lib = NULL;

static int
link_xet_libraries(void)
{
    /* Attempt to guess if we were statically linked to libc, if so bail */
    if(_dl_non_dynamic_init != NULL) {
        return -1;
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Init Intel XET Libaries);
    dl_xet_lib = dlopen("libpti.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_xet_lib || dlerror() != NULL)
    {
        fprintf(stderr, "Intel XET library libpti.so not found.");
        return -1;
    }
    zetMetricGroupGetPtr = DLSYM_AND_CHECK(dl_xet_lib, "zetMetricGroupGet");
    return 0;
}


static void
release_xet_libraries(void)
{
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Finalize Intel XET Libaries);
    if (dl_xet_lib)
    {
        dlclose(dl_xet_lib);
        dl_xet_lib = NULL;
    }
}
