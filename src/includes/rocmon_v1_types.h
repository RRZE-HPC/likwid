/*
 * =======================================================================================
 *
 *      Filename:  rocmon_v1_types.h
 *
 *      Description:  Header File of rocmon v1 module.
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
#ifndef LIKWID_ROCMON_V1_TYPES_H
#define LIKWID_ROCMON_V1_TYPES_H

#include <likwid.h>
// #include <hsa.h>
#ifdef HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE
#undef HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE
#endif
#ifdef ROCPROFILER_EXPORT
#undef ROCPROFILER_EXPORT
#endif
#ifdef ROCPROFILER_IMPORT
#undef ROCPROFILER_IMPORT
#endif
#ifdef ROCPROFILER_VERSION_MAJOR
#undef ROCPROFILER_VERSION_MAJOR
#endif
#ifdef ROCPROFILER_VERSION_MINOR
#undef ROCPROFILER_VERSION_MINOR
#endif
#ifdef ROCPROFILER_API
#undef ROCPROFILER_API
#endif
#include <rocprofiler/rocprofiler.h>


#include <rocmon_common_types.h>


#endif /* LIKWID_ROCMON_V1_TYPES_H */
