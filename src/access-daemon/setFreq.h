/*
 * =======================================================================================
 *
 *      Filename:  setFreq.h
 *
 *      Description:  Header for frequency daemon
 *
 *      Version:   4.3.4
 *      Released:  01.04.2019
 *
 *      Authors:  Thomas Roehl (tr), thomas.roehl@googlemail.com
 *
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

#ifndef SETFREQ_DAEMON
#define SETFREQ_DAEMON

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

extern int do_pstate (int argn, char** argv);
extern int do_cpufreq (int argn, char** argv);

#endif
