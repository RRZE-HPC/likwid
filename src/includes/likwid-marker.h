/*
 * =======================================================================================
 *
 *      Filename:  likwid-marker.h
 *
 *      Description:  Header File of likwid Marker API
 *
 *      Version:   5.4.1
 *      Released:  09.12.2024
 *
 *      Authors:  Thomas Gruber (tg), thomas.roehl@googlemail.com
 *
 *      Project:  likwid
 *
 *      Copyright (C) 2024 RRZE, University Erlangen-Nuremberg
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
#ifndef LIKWID_MARKER_H
#define LIKWID_MARKER_H


/** \addtogroup MarkerAPI Marker API module
*  @{
*/
/*!
\def LIKWID_MARKER_INIT
Shortcut for likwid_markerInit() if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_MARKER_THREADINIT
Shortcut for likwid_markerThreadInit() if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_MARKER_REGISTER(regionTag)
Shortcut for likwid_markerRegisterRegion() with \a regionTag if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_MARKER_START(regionTag)
Shortcut for likwid_markerStartRegion() with \a regionTag if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_MARKER_STOP(regionTag)
Shortcut for likwid_markerStopRegion() with \a regionTag if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
Shortcut for likwid_markerGetResults() for \a regionTag if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_MARKER_SWITCH
Shortcut for likwid_markerNextGroup() if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_MARKER_RESET(regionTag)
Shortcut for likwid_markerResetRegion() if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_MARKER_CLOSE
Shortcut for likwid_markerClose() if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def LIKWID_MARKER_WRITE_FILE(markerfile)
Shortcut for likwid_markerWriteFile() if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/** @}*/

#ifdef LIKWID_PERFMON
#include <likwid.h>
#define LIKWID_MARKER_INIT likwid_markerInit()
#define LIKWID_MARKER_THREADINIT likwid_markerThreadInit()
#define LIKWID_MARKER_SWITCH likwid_markerNextGroup()
#define LIKWID_MARKER_REGISTER(regionTag) likwid_markerRegisterRegion(regionTag)
#define LIKWID_MARKER_START(regionTag) likwid_markerStartRegion(regionTag)
#define LIKWID_MARKER_STOP(regionTag) likwid_markerStopRegion(regionTag)
#define LIKWID_MARKER_CLOSE likwid_markerClose()
#define LIKWID_MARKER_WRITE_FILE(markerfile) likwid_markerWriteFile(markerfile)
#define LIKWID_MARKER_RESET(regionTag) likwid_markerResetRegion(regionTag)
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count) likwid_markerGetRegion(regionTag, nevents, events, time, count)
#else  /* LIKWID_PERFMON */
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_WRITE_FILE(markerfile)
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#define LIKWID_MARKER_RESET(regionTag)
#endif /* LIKWID_PERFMON */


/** \addtogroup NvMarkerAPI NvMarker API module (MarkerAPI for Nvidia GPUs)
*  @{
*/
/*!
\def NVMON_MARKER_INIT
Shortcut for nvmon_markerInit() if compiled with -DLIKWID_PERFMON. Otherwise no operation is performed
*/
/*!
\def NVMON_MARKER_THREADINIT
No operation is performed, this macro exists only to be similar as CPU MarkerAPI
*/
/*!
\def NVMON_MARKER_REGISTER(regionTag)
Shortcut for nvmon_markerRegisterRegion() with \a regionTag if compiled with -DLIKWID_NVMON. Otherwise no operation is performed
*/
/*!
\def NVMON_MARKER_START(regionTag)
Shortcut for nvmon_markerStartRegion() with \a regionTag if compiled with -DLIKWID_NVMON. Otherwise no operation is performed
*/
/*!
\def NVMON_MARKER_STOP(regionTag)
Shortcut for nvmon_markerStopRegion() with \a regionTag if compiled with -DLIKWID_NVMON. Otherwise no operation is performed
*/
/*!
\def NVMON_MARKER_GET(regionTag, ngpus, nevents, events, time, count)
Shortcut for nvmon_markerGetRegion() for \a regionTag if compiled with -DLIKWID_NVMON. Otherwise no operation is performed
*/
/*!
\def NVMON_MARKER_SWITCH
Shortcut for nvmon_markerNextGroup() if compiled with -DLIKWID_NVMON. Otherwise no operation is performed
*/
/*!
\def NVMON_MARKER_RESET(regionTag)
Shortcut for nvmon_markerResetRegion() if compiled with -DLIKWID_NVMON. Otherwise no operation is performed
*/
/*!
\def NVMON_MARKER_CLOSE
Shortcut for nvmon_markerClose() if compiled with -DLIKWID_NVMON. Otherwise no operation is performed
*/
/*!
\def NVMON_MARKER_WRITE_FILE
Shortcut for nvmon_markerWriteFile() with \a filename if compiled with -DLIKWID_NVMON. Otherwise no operation is performed
*/
/** @}*/

#ifdef LIKWID_NVMON
#ifndef LIKWID_WITH_NVMON
#define LIKWID_WITH_NVMON
#endif
#include <likwid.h>
#define NVMON_MARKER_INIT nvmon_markerInit()
#define NVMON_MARKER_THREADINIT
#define NVMON_MARKER_SWITCH nvmon_markerNextGroup()
#define NVMON_MARKER_REGISTER(regionTag) nvmon_markerRegisterRegion(regionTag)
#define NVMON_MARKER_START(regionTag) nvmon_markerStartRegion(regionTag)
#define NVMON_MARKER_STOP(regionTag) nvmon_markerStopRegion(regionTag)
#define NVMON_MARKER_CLOSE nvmon_markerClose()
#define NVMON_MARKER_RESET(regionTag) nvmon_markerResetRegion(regionTag)
#define NVMON_MARKER_GET(regionTag, ngpus, nevents, events, time, count) \
    nvmon_markerGetRegion(regionTag, ngpus, nevents, events, time, count)
#define NVMON_MARKER_WRITE_FILE(markerfile) \
    nvmon_markerWriteFile(markerfile)
#else /* LIKWID_NVMON */
#define NVMON_MARKER_INIT
#define NVMON_MARKER_THREADINIT
#define NVMON_MARKER_SWITCH
#define NVMON_MARKER_REGISTER(regionTag)
#define NVMON_MARKER_START(regionTag)
#define NVMON_MARKER_STOP(regionTag)
#define NVMON_MARKER_CLOSE
#define NVMON_MARKER_GET(regionTag, nevents, events, time, count)
#define NVMON_MARKER_RESET(regionTag)
#define NVMON_MARKER_WRITE_FILE(markerfile)
#endif /* LIKWID_NVMON */


/** \addtogroup RocMarkerAPI RocMarker API module (MarkerAPI for AMD GPUs)
*  @{
*/
/*!
\def ROCMON_MARKER_INIT
Shortcut for rocmon_markerInit() if compiled with -DLIKWID_ROCMON. Otherwise no operation is performed
*/
/*!
\def ROCMON_MARKER_THREADINIT
Shortcut for rocmon_markerThreadInit() if compiled with -DLIKWID_ROCMON. Otherwise no operation is performed
*/
/*!
\def ROCMON_MARKER_REGISTER(regionTag)
Shortcut for rocmon_markerRegisterRegion() with \a regionTag if compiled with -DLIKWID_ROCMON. Otherwise no operation is performed
*/
/*!
\def ROCMON_MARKER_START(regionTag)
Shortcut for rocmon_markerStartRegion() with \a regionTag if compiled with -DLIKWID_ROCMON. Otherwise no operation is performed
*/
/*!
\def ROCMON_MARKER_STOP(regionTag)
Shortcut for rocmon_markerStopRegion() with \a regionTag if compiled with -DLIKWID_ROCMON. Otherwise no operation is performed
*/
/*!
\def ROCMON_MARKER_GET(regionTag, ngpus, nevents, events, time, count)
Shortcut for rocmon_markerGetRegion() for \a regionTag if compiled with -DLIKWID_ROCMON. Otherwise no operation is performed
*/
/*!
\def ROCMON_MARKER_SWITCH
Shortcut for rocmon_markerNextGroup() if compiled with -DLIKWID_ROCMON. Otherwise no operation is performed
*/
/*!
\def ROCMON_MARKER_RESET(regionTag)
Shortcut for rocmon_markerResetRegion() if compiled with -DLIKWID_ROCMON. Otherwise no operation is performed
*/
/*!
\def ROCMON_MARKER_CLOSE
Shortcut for rocmon_markerClose() if compiled with -DLIKWID_ROCMON. Otherwise no operation is performed
*/
/*!
\def ROCMON_MARKER_WRITE_FILE
Shortcut for rocmon_markerWriteFile() with \a filename if compiled with -DLIKWID_ROCMON. Otherwise no operation is performed
*/
/** @}*/

#ifdef LIKWID_ROCMON
#ifndef LIKWID_WITH_ROCMON
#define LIKWID_WITH_ROCMON
#endif
#include <likwid.h>
#define ROCMON_MARKER_INIT rocmon_markerInit()
#define ROCMON_MARKER_THREADINIT rocmon_markerThreadInit()
#define ROCMON_MARKER_SWITCH rocmon_markerNextGroup()
#define ROCMON_MARKER_REGISTER(regionTag) rocmon_markerRegisterRegion(regionTag)
#define ROCMON_MARKER_START(regionTag) rocmon_markerStartRegion(regionTag)
#define ROCMON_MARKER_STOP(regionTag) rocmon_markerStopRegion(regionTag)
#define ROCMON_MARKER_CLOSE rocmon_markerClose()
#define ROCMON_MARKER_RESET(regionTag) rocmon_markerResetRegion(regionTag)
#define ROCMON_MARKER_GET(regionTag, ngpus, nevents, events, time, count) rocmon_markerGetRegion(regionTag, ngpus, nevents, events, time, count)
#define ROCMON_MARKER_WRITE_FILE(filename) rocmon_markerWriteFile(filename)
#else /* LIKWID_ROCMON */
#define ROCMON_MARKER_INIT
#define ROCMON_MARKER_THREADINIT
#define ROCMON_MARKER_SWITCH
#define ROCMON_MARKER_REGISTER(regionTag)
#define ROCMON_MARKER_START(regionTag)
#define ROCMON_MARKER_STOP(regionTag)
#define ROCMON_MARKER_CLOSE
#define ROCMON_MARKER_GET(regionTag, nevents, events, time, count)
#define ROCMON_MARKER_RESET(regionTag)
#define ROCMON_MARKER_WRITE_FILE(filename)
#endif /* LIKWID_ROCMON */


#endif /* LIKWID_MARKER_H */
