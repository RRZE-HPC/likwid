/*
 * =======================================================================================
 *
 *      Filename:  affinity.h
 *
 *      Description:  Header File affinity Module
 *
 *      Version:   4.3.2
 *      Released:  12.04.2018
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Roehl (tr), thomas.roehl@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2018 RRZE, University Erlangen-Nuremberg
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
#ifndef AFFINITY_H
#define AFFINITY_H

#include <types.h>
#include <likwid.h>

int socket_lock[MAX_NUM_NODES];
int core_lock[MAX_NUM_THREADS];
int tile_lock[MAX_NUM_THREADS];
int numa_lock[MAX_NUM_THREADS];
int sharedl2_lock[MAX_NUM_THREADS];
int sharedl3_lock[MAX_NUM_THREADS];

extern AffinityDomains affinityDomains;

extern int *affinity_thread2core_lookup;
extern int *affinity_thread2socket_lookup;
extern int *affinity_thread2numa_lookup;
extern int *affinity_thread2sharedl3_lookup;
extern int affinity_processGetProcessorId();
extern int affinity_threadGetProcessorId();
extern const AffinityDomain* affinity_getDomain(bstring domain);

#endif /*AFFINITY_H*/
