/*
 * =======================================================================================
 *
 *      Filename:  numa_virtual.c
 *
 *      Description:  Virtual/Fake NUMA backend
 *
 *      Version:   5.1.0
 *      Released:  20.11.2020
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *                Tobias Auerochs, tobi291019@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2020 RRZE, University Erlangen-Nuremberg
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>

#include <error.h>

#include <numa.h>
#include <topology.h>

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
virtual_numa_init()
{
    int i;

    NumaNode* nodes = (NumaNode*) malloc(sizeof(NumaNode));
    if (!nodes)
    {
        fprintf(stderr,"No memory to allocate %ld byte for nodes array\n",sizeof(NumaNode));
        return -1;
    }
    nodes[0].processors = (uint32_t*) malloc(cpuid_topology.numHWThreads * sizeof(uint32_t));
    if (!nodes[0].processors)
    {
        fprintf(stderr,"No memory to allocate %ld byte for processors array of NUMA node %d\n",
                cpuid_topology.numHWThreads * sizeof(uint32_t),0);
        free(nodes);
        return -1;
    }
    nodes[0].distances = (uint32_t*) malloc(sizeof(uint32_t));
    if (!nodes[0].distances)
    {
        fprintf(stderr,"No memory to allocate %ld byte for distances array of NUMA node %d\n",
                sizeof(uint32_t),0);
        free(nodes);
        free(nodes[0].processors);
        return -1;
    }

    nodes[0].id = 0;
    nodes[0].numberOfProcessors = cpuid_topology.numHWThreads;
    nodes[0].totalMemory = proc_getTotalSysMem();
    nodes[0].freeMemory = proc_getFreeSysMem();
    for (i = 0; i < cpuid_topology.numHWThreads; i++)
    {
        nodes[0].processors[i] = cpuid_topology.threadPool[i].apicId;
    }
    nodes[0].distances[0] = 10;
    nodes[0].numberOfDistances = 1;
    numa_info.numberOfNodes = 1;
    numa_info.nodes = nodes;

    numaInitialized = 1;
    return 0;
}
