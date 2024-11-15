/*
 * =======================================================================================
 *
 *      Filename:  meta.c
 *
 *      Description:  Get information what is supported by the library
 *
 *      Version:   5.4.0
 *      Released:  15.11.2024
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
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

#include <stdlib.h>
#include <stdio.h>

#include <likwid.h>

int likwid_getMajorVersion(void)
{
    return (int) VERSION ;
}

int likwid_getMinorVersion(void)
{
    return (int) RELEASE ;
}

int likwid_getBugfixVersion(void)
{
    return (int) MINORVERSION ;
}

int likwid_getNvidiaSupport(void)
{
#ifdef LIKWID_WITH_NVMON
    return 1;
#else
    return 0;
#endif
}

int likwid_getRocmSupport(void)
{
#ifdef LIKWID_WITH_ROCMON
    return 1;
#else
    return 0;
#endif
}

int likwid_getMaxSupportedThreads(void)
{
    return (int) MAX_NUM_THREADS;
}

int likwid_getMaxSupportedSockets(void)
{
    return (int) MAX_NUM_NODES;
}

int likwid_getSysFeaturesSupport(void)
{
#ifdef LIKWID_WITH_SYSFEATURES
    return 1;
#else
    return 0;
#endif
}
