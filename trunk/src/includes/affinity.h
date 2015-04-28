/*
 * =======================================================================================
 *
 *      Filename:  affinity.h
 *
 *      Description:  Header File affinity Module
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2013 Jan Treibig 
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
int tile_lock[MAX_NUM_THREADS];
extern AffinityDomains affinityDomains;

extern int affinity_core2node_lookup[MAX_NUM_THREADS];

extern int affinity_processGetProcessorId();
extern int affinity_threadGetProcessorId();
extern const AffinityDomain* affinity_getDomain(bstring domain);

/* DEPRECATED
extern void affinity_printDomains();*/


#endif /*AFFINITY_H*/

