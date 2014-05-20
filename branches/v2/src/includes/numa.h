/*
 * =======================================================================================
 *
 *      Filename:  numa.h
 *
 *      Description:  Header File numa Module. 
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2012 Jan Treibig 
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

#ifndef NUMA_H
#define NUMA_H

#include <types.h>

/** Structure holding numa information
 *
 */
extern NumaTopology numa_info;

extern int numa_init (void);
extern void numa_setInterleaved(int* processorList, int numberOfProcessors);
extern void numa_membind(void* ptr, size_t size, int domainId);

#endif /*NUMA_H*/
