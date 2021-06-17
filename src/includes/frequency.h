/*
 * =======================================================================================
 *
 *      Filename:  frequency.h
 *
 *      Description:  Header File frequency module.
 *
 *      Version:   5.2
 *      Released:  17.6.2021
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 NHR@FAU, University Erlangen-Nuremberg
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

#ifndef FREQUENCY_H
#define FREQUENCY_H

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)


extern char* daemon_path;

#if !defined(__ARM_ARCH_7A__) && !defined(__ARM_ARCH_8A) && !defined(_ARCH_PPC)
#include <cpuid.h>
#endif

#if !defined(__ARM_ARCH_7A__) && !defined(__ARM_ARCH_8A) && !defined(_ARCH_PPC)
static int isAMD()
{
    unsigned int eax,ebx,ecx,edx;
    eax = 0x0;
    CPUID(eax,ebx,ecx,edx);
    if (ecx == 0x444d4163)
        return 1;
    return 0;
}
#else
static int isAMD()
{
    return 0;
}
#endif


#endif /* FREQUENCY_H */
