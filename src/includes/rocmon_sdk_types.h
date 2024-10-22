/*
 * =======================================================================================
 *
 *      Filename:  rocmon_sdk_types.h
 *
 *      Description:  Header File of rocmon sdk module for ROCM >= 6.2
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
#ifndef LIKWID_ROCMON_SDK_TYPES_H
#define LIKWID_ROCMON_SDK_TYPES_H

#include <likwid.h>
/*#ifdef ROCPROFILER_EXPORT*/
/*#undef ROCPROFILER_EXPORT*/
/*#endif*/
/*#ifdef ROCPROFILER_IMPORT*/
/*#undef ROCPROFILER_IMPORT*/
/*#endif*/
/*#ifdef ROCPROFILER_VERSION_MAJOR*/
/*#undef ROCPROFILER_VERSION_MAJOR*/
/*#endif*/
/*#ifdef ROCPROFILER_VERSION_MINOR*/
/*#undef ROCPROFILER_VERSION_MINOR*/
/*#endif*/
/*#ifdef ROCPROFILER_API*/
/*#undef ROCPROFILER_API*/
/*#endif*/
#include <rocprofiler-sdk/rocprofiler.h>
/*#ifdef ROCPROFILER_API*/
/*#undef ROCPROFILER_API*/
/*#endif*/
#include <rocprofiler-sdk/registration.h>


typedef struct {
    rocprofiler_agent_t* agent;
    rocprofiler_buffer_id_t buffer;
    rocprofiler_context_id_t context;
    RocmonEventResultList *result;
} RocprofilerSdkAgentData;

typedef struct {
    int num_agents;
    RocprofilerSdkAgentData* agents;
} RocprofilerSdkData;



#endif /* LIKWID_ROCMON_SDK_TYPES_H */
