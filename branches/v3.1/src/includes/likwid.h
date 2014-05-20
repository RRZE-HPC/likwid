/*
 * =======================================================================================
 *
 *      Filename:  likwid.h
 *
 *      Description:  Header File of likwid marker API
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
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

#ifndef LIKWID_H
#define LIKWID_H

#ifdef LIKWID_PERFMON
#define LIKWID_MARKER_INIT likwid_markerInit()
#define LIKWID_MARKER_THREADINIT likwid_markerThreadInit()
#define LIKWID_MARKER_START(reg) likwid_markerStartRegion(reg)
#define LIKWID_MARKER_STOP(reg) likwid_markerStopRegion(reg)
#define LIKWID_MARKER_CLOSE likwid_markerClose()
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_START(reg)
#define LIKWID_MARKER_STOP(reg)
#define LIKWID_MARKER_CLOSE
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* marker API routines */
extern void likwid_markerInit(void);
extern void likwid_markerThreadInit(void);
extern void likwid_markerClose(void);
extern void likwid_markerStartRegion(const char* regionTag);
extern void likwid_markerStopRegion(const char* regionTag);

/* utility routines */
extern int  likwid_getProcessorId();
extern int  likwid_pinProcess(int processorId);
extern int  likwid_pinThread(int processorId);

#ifdef __cplusplus
}
#endif

#endif /*LIKWID_H*/
