/*
 * =======================================================================================
 *
 *      Filename:  numa_types.h
 *
 *      Description:  Types file for numa module.
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

#ifndef NUMA_TYPES_H
#define NUMA_TYPES_H


typedef struct {
    int id;
    uint64_t totalMemory;
    uint64_t freeMemory;
    int numberOfProcessors;
    uint32_t* processors;
    uint32_t* processorsCompact;
    int numberOfDistances;
    uint32_t* distances;
} NumaNode;

typedef struct {
    uint32_t numberOfNodes;
    NumaNode* nodes;
} NumaTopology;


#endif /*NUMA_TYPES_H*/
