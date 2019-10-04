/*
 * =======================================================================================
 *
 *      Filename:  likwid-gpumarker.h
 *
 *      Description:  Header File of likwid GPU Marker API
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Authors:  Thomas Gruber (tg), thomas.roehl@googlemail.com
 *
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
#ifndef LIKWID_GPUMARKER_H
#define LIKWID_GPUMARKER_H

#include <likwid.h>

/** \addtogroup MarkerAPI Marker API module
*  @{
*/
/*!
\def LIKWID_GPUMARKER_INIT
Shortcut for likwid_gpuMarkerInit() if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_GPUMARKER_THREADINIT
Shortcut for likwid_gpuMarkerThreadInit() if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_GPUMARKER_REGISTER(regionTag)
Shortcut for likwid_gpuMarkerRegisterRegion() with \a regionTag if compiled with -DLIKWID_NVMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_GPUMARKER_START(regionTag)
Shortcut for likwid_gpuMarkerStartRegion() with \a regionTag if compiled with -DLIKWID_NVMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_GPUMARKER_STOP(regionTag)
Shortcut for likwid_gpuMarkerStopRegion() with \a regionTag if compiled with -DLIKWID_NVMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_GPUMARKER_GET(regionTag, ngpus, nevents, events, time, count)
Shortcut for likwid_gpuMarkerGetRegion() for \a regionTag if compiled with -DLIKWID_NVMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_GPUMARKER_SWITCH
Shortcut for likwid_gpuMarkerNextGroup() if compiled with -DLIKWID_NVMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_GPUMARKER_RESET(regionTag)
Shortcut for likwid_gpuMarkerResetRegion() if compiled with -DLIKWID_NVMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_GPUMARKER_CLOSE
Shortcut for likwid_gpuMarkerClose() if compiled with -DLIKWID_NVMON. Otherwise no operation is performed
*/
/** @}*/

#ifdef LIKWID_NVMON
#define LIKWID_GPUMARKER_INIT likwid_gpuMarkerInit()
#define LIKWID_GPUMARKER_THREADINIT likwid_gpuMarkerThreadInit()
#define LIKWID_GPUMARKER_SWITCH likwid_gpuMarkerNextGroup()
#define LIKWID_GPUMARKER_REGISTER(regionTag) likwid_gpuMarkerGetRegion(regionTag)
#define LIKWID_GPUMARKER_START(regionTag) likwid_gpuMarkerStartRegion(regionTag)
#define LIKWID_GPUMARKER_STOP(regionTag) likwid_gpuMarkerStopRegion(regionTag)
#define LIKWID_GPUMARKER_CLOSE likwid_gpuMarkerClose()
#define LIKWID_GPUMARKER_RESET(regionTag) likwid_gpuMarkerResetRegion(regionTag)
#define LIKWID_GPUMARKER_GET(regionTag, ngpus, nevents, events, time, count) \
    likwid_gpuMarkerGetRegion(regionTag, ngpus, nevents, events, time, count)
#else
#define LIKWID_GPUMARKER_INIT
#define LIKWID_GPUMARKER_THREADINIT
#define LIKWID_GPUMARKER_SWITCH
#define LIKWID_GPUMARKER_REGISTER(regionTag)
#define LIKWID_GPUMARKER_START(regionTag)
#define LIKWID_GPUMARKER_STOP(regionTag)
#define LIKWID_GPUMARKER_CLOSE
#define LIKWID_GPUMARKER_GET(regionTag, nevents, events, time, count)
#define LIKWID_GPUMARKER_RESET(regionTag)
#endif


#endif /* LIKWID_GPUMARKER_H */
