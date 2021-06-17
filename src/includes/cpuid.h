/*
 * =======================================================================================
 *
 *      Filename:  cpuid.h
 *
 *      Description:  Common macro definition for CPUID instruction
 *
 *      Version:   5.2
 *      Released:  17.6.2021
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@gmail.com
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
#ifndef LIKWID_CPUID_H
#define LIKWID_CPUID_H

/* This was taken from the linux kernel
 * Kernel version 3.19
 * File: arch/x86/boot/cpuflags.c
*/
#if !defined(__ARM_ARCH_7A__) && !defined(__ARM_ARCH_8A) && !defined(_ARCH_PPC)


#if defined(__i386__) && defined(__PIC__)
# define EBX_REG "=r"
#else
# define EBX_REG "=b"
#endif

#ifndef __clang__
#define CPUID(eax,ebx,ecx,edx)                            \
    __asm__ volatile(".ifnc %%ebx,%3 ; movl  %%ebx,%3 ; .endif  \n\t" \
                     "cpuid                                     \n\t" \
                     ".ifnc %%ebx,%3 ; xchgl %%ebx,%3 ; .endif  \n\t" \
                     : "=a" (eax), "=c" (ecx), "=d" (edx), EBX_REG (ebx) \
                     : "a" (eax), "c" (ecx) \
                     )
#else
#define CPUID(eax,ebx,ecx,edx)         \
    __asm__ volatile("cpuid" : "=a" (eax), "=c" (ecx), "=d" (edx), EBX_REG (ebx) : "a" (eax), "c" (ecx) );
#endif

#else /* ARCH Filters */
#define CPUID(eax,ebx,ecx,edx)
#endif /* ARCH Filters */

#endif /* LIKWID_CPUID_H */
