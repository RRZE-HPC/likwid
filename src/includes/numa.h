/*
 * =======================================================================================
 *
 *      Filename:  numa.h
 *
 *      Description:  Header File NUMA module for internal use. External functions are
 *                    defined in likwid.h
 *
 *      Version:   5.1
 *      Released:  16.11.2020
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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
#ifndef LIKWID_NUMA
#define LIKWID_NUMA

#include <stdlib.h>
#include <stdio.h>

#include <types.h>
#include <likwid.h>
#include <numa_hwloc.h>
#include <numa_proc.h>
#include <numa_virtual.h>

extern int numaInitialized;

extern int str2int(const char* str);

extern uint64_t proc_getFreeSysMem(void);

extern uint64_t proc_getTotalSysMem(void);

struct numa_functions {
    int (*numa_init) (void);
    void (*numa_setInterleaved) (const int*, int);
    void (*numa_setMembind) (const int*, int);
    void (*numa_membind) (void*, size_t, int);
};

#endif
