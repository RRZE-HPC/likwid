/*
 * =======================================================================================
 *
 *      Filename:  numa_hwloc.c
 *
 *      Description:  Interface to hwloc for NUMA topology
 *
 *      Version:   4.3.0
 *      Released:  22.12.2017
 *
 *      Author:  Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2017 RRZE, University Erlangen-Nuremberg
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
#ifdef LIKWID_USE_HWLOC
#include <hwloc.h>
#include <topology_hwloc.h>
#endif

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

#ifdef LIKWID_USE_HWLOC
uint64_t
getFreeNodeMem(int nodeId)
{
    FILE *fp;
    bstring filename;
    uint64_t free = 0;
    bstring freeString  = bformat("MemFree:");
    int i;

    filename = bformat("/sys/devices/system/node/node%d/meminfo", nodeId);

    if (NULL != (fp = fopen (bdata(filename), "r")))
    {
        bstring src = bread ((bNread) fread, fp);
        struct bstrList* tokens = bsplit(src,(char) '\n');

        for (i=0;i<tokens->qty;i++)
        {
            if (binstr(tokens->entry[i],0,freeString) != BSTR_ERR)
            {
                 bstring tmp = bmidstr (tokens->entry[i], 18, blength(tokens->entry[i])-18  );
                 bltrimws(tmp);
                 struct bstrList* subtokens = bsplit(tmp,(char) ' ');
                 free = str2int(bdata(subtokens->entry[0]));
                 bdestroy(tmp);
                 bstrListDestroy(subtokens);
            }
        }
        bstrListDestroy(tokens);
        bdestroy(src);
        fclose(fp);
    }
    else if (!access("/proc/meminfo", R_OK))
    {
        bdestroy(filename);
        filename = bfromcstr("/proc/meminfo");
        if (NULL != (fp = fopen (bdata(filename), "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            struct bstrList* tokens = bsplit(src,(char) '\n');
            for (i=0;i<tokens->qty;i++)
            {
                if (binstr(tokens->entry[i],0,freeString) != BSTR_ERR)
                {
                     bstring tmp = bmidstr (tokens->entry[i], 10, blength(tokens->entry[i])-10  );
                     bltrimws(tmp);
                     struct bstrList* subtokens = bsplit(tmp,(char) ' ');
                     free = str2int(bdata(subtokens->entry[0]));
                     bdestroy(tmp);
                     bstrListDestroy(subtokens);
                }
            }
            bstrListDestroy(tokens);
            bdestroy(src);
            fclose(fp);
        }
    }
    else
    {
        bdestroy(freeString);
        bdestroy(filename);
        ERROR;
    }
    bdestroy(freeString);
    bdestroy(filename);
    return free;
}

uint64_t
getTotalNodeMem(int nodeId)
{
    int i;
    FILE *fp;
    uint64_t total = 0;
    bstring totalString  = bformat("MemTotal:");
    bstring sysfilename = bformat("/sys/devices/system/node/node%d/meminfo", nodeId);
    bstring procfilename = bformat("/proc/meminfo");
    char *sptr = bdata(procfilename);

    if (NULL != (fp = fopen (bdata(sysfilename), "r")))
    {
        bstring src = bread ((bNread) fread, fp);
        struct bstrList* tokens = bsplit(src,(char) '\n');

        for (i=0;i<tokens->qty;i++)
        {
            if (binstr(tokens->entry[i],0,totalString) != BSTR_ERR)
            {
                 bstring tmp = bmidstr (tokens->entry[i], 18, blength(tokens->entry[i])-18  );
                 bltrimws(tmp);
                 struct bstrList* subtokens = bsplit(tmp,(char) ' ');
                 total = str2int(bdata(subtokens->entry[0]));
                 bdestroy(tmp);
                 bstrListDestroy(subtokens);
            }
        }
        bstrListDestroy(tokens);
        bdestroy(src);
        fclose(fp);
    }
    else if (!access(sptr, R_OK))
    {
        if (NULL != (fp = fopen (bdata(procfilename), "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            struct bstrList* tokens = bsplit(src,(char) '\n');
            for (i=0;i<tokens->qty;i++)
            {
                if (binstr(tokens->entry[i],0,totalString) != BSTR_ERR)
                {
                     bstring tmp = bmidstr (tokens->entry[i], 10, blength(tokens->entry[i])-10  );
                     bltrimws(tmp);
                     struct bstrList* subtokens = bsplit(tmp,(char) ' ');
                     total = str2int(bdata(subtokens->entry[0]));
                     bdestroy(tmp);
                     bstrListDestroy(subtokens);
                }
            }
            bstrListDestroy(tokens);
            bdestroy(src);
            fclose(fp);
        }
    }
    else
    {
        bdestroy(totalString);
        bdestroy(sysfilename);
        bdestroy(procfilename);
        ERROR;
    }

    bdestroy(totalString);
    bdestroy(sysfilename);
    bdestroy(procfilename);
    return total;
}

int
likwid_hwloc_findProcessor(int nodeID, int cpuID)
{
    hwloc_obj_t obj;
    int i;
    int pu_count = likwid_hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_PU);

    for (i=0; i<pu_count; i++)
    {
        obj = likwid_hwloc_get_obj_by_type(hwloc_topology, HWLOC_OBJ_PU, i);
        if (!obj)
        {
            continue;
        }
        else
        {
            if (obj->os_index == cpuID)
            {
                return 1;
            }
        }
    }
    return 0;

}


int
hwloc_numa_init(void)
{
    int errno;
    uint32_t i;
    int d;
    int depth;
    int cores_per_socket;
    int numPUs = 0;
    hwloc_obj_t obj;
    const struct hwloc_distances_s* distances;
    hwloc_obj_type_t hwloc_type = HWLOC_OBJ_NODE;
    if (numaInitialized > 0 || numa_info.numberOfNodes > 0)
        return 0;

    if (!hwloc_topology)
    {
        likwid_hwloc_topology_init(&hwloc_topology);
        likwid_hwloc_topology_load(hwloc_topology);
    }

    numa_info.numberOfNodes = likwid_hwloc_get_nbobjs_by_type(hwloc_topology, hwloc_type);
    numPUs = likwid_hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_PU);

    /* If the amount of NUMA nodes == 0, there is actually no NUMA node, hence
       aggregate all sockets in the system into the single virtually created NUMA node */
    if (numa_info.numberOfNodes == 0)
    {
        hwloc_type = HWLOC_OBJ_SOCKET;
        numa_info.numberOfNodes = 1;

        numa_info.nodes = (NumaNode*) malloc(sizeof(NumaNode));
        if (!numa_info.nodes)
        {
            fprintf(stderr,"No memory to allocate %ld byte for nodes array\n",sizeof(NumaNode));
            return -1;
        }
        numa_info.nodes[0].id = 0;
        numa_info.nodes[0].numberOfProcessors = 0;
        numa_info.nodes[0].totalMemory = getTotalNodeMem(0);
        numa_info.nodes[0].freeMemory = getFreeNodeMem(0);
        numa_info.nodes[0].processors = (uint32_t*) malloc(numPUs * sizeof(uint32_t));
        if (!numa_info.nodes[0].processors)
        {
            fprintf(stderr,"No memory to allocate %ld byte for processors array of NUMA node %d\n",
                    numPUs * sizeof(uint32_t),0);
            return -1;
        }
        numa_info.nodes[0].distances = (uint32_t*) malloc(sizeof(uint32_t));
        if (!numa_info.nodes[0].distances)
        {
            fprintf(stderr,"No memory to allocate %ld byte for distances array of NUMA node %d\n",
                    sizeof(uint32_t),0);
            return -1;
        }
        numa_info.nodes[0].distances[0] = 10;
        numa_info.nodes[0].numberOfDistances = 1;
        cores_per_socket = cpuid_topology.numHWThreads/cpuid_topology.numSockets;

        for (d=0; d<likwid_hwloc_get_nbobjs_by_type(hwloc_topology, hwloc_type); d++)
        {
            obj = likwid_hwloc_get_obj_by_type(hwloc_topology, hwloc_type, d);
            /* depth is here used as index in the processors array */
            depth = d * cores_per_socket;
            numa_info.nodes[0].numberOfProcessors += likwid_hwloc_record_objs_of_type_below_obj(
                    likwid_hwloc_topology, obj, HWLOC_OBJ_PU, &depth, &numa_info.nodes[0].processors);
        }
    }
    else
    {
        numa_info.nodes = (NumaNode*) malloc(numa_info.numberOfNodes * sizeof(NumaNode));
        if (!numa_info.nodes)
        {
            fprintf(stderr,"No memory to allocate %ld byte for nodes array\n",
                    numa_info.numberOfNodes * sizeof(NumaNode));
            return -1;
        }
        depth = likwid_hwloc_get_type_depth(hwloc_topology, hwloc_type);
        distances = likwid_hwloc_get_whole_distance_matrix_by_type(hwloc_topology, hwloc_type);
        for (i=0; i<numa_info.numberOfNodes; i++)
        {
            obj = likwid_hwloc_get_obj_by_depth(hwloc_topology, depth, i);

            numa_info.nodes[i].id = obj->os_index;

            if (obj->memory.local_memory != 0)
            {
                numa_info.nodes[i].totalMemory = (uint64_t)(obj->memory.local_memory/1024);
            }
            else if (obj->memory.total_memory != 0)
            {
                numa_info.nodes[i].totalMemory = (uint64_t)(obj->memory.total_memory/1024);
            }
            else
            {
                numa_info.nodes[i].totalMemory = getTotalNodeMem(numa_info.nodes[i].id);
            }
            /* freeMemory not detected by hwloc, do it the native way */
            numa_info.nodes[i].freeMemory = getFreeNodeMem(numa_info.nodes[i].id);
            numa_info.nodes[i].processors = (uint32_t*) malloc(numPUs * sizeof(uint32_t));
            if (!numa_info.nodes[i].processors)
            {
                fprintf(stderr,"No memory to allocate %ld byte for processors array of NUMA node %d\n",
                        numPUs * sizeof(uint32_t), i);
                return -1;
            }
            d = 0;
            numa_info.nodes[i].numberOfProcessors = likwid_hwloc_record_objs_of_type_below_obj(
                    hwloc_topology, obj, HWLOC_OBJ_PU, &d, &numa_info.nodes[i].processors);
            numa_info.nodes[i].distances = (uint32_t*) malloc(numa_info.numberOfNodes * sizeof(uint32_t));
            if (!numa_info.nodes[i].distances)
            {
                fprintf(stderr,"No memory to allocate %ld byte for distances array of NUMA node %d\n",
                        numa_info.numberOfNodes*sizeof(uint32_t),i);
                return -1;
            }
            if (distances)
            {
                numa_info.nodes[i].numberOfDistances = distances->nbobjs;
                for(d=0;d<distances->nbobjs;d++)
                {
                    numa_info.nodes[i].distances[d] = distances->latency[i*distances->nbobjs + d] * distances->latency_base;
                }
            }
            else
            {
                numa_info.nodes[i].numberOfDistances = numa_info.numberOfNodes;
                for(d=0;d<numa_info.numberOfNodes;d++)
                {
                    numa_info.nodes[i].distances[d] = 10;
                }
            }

        }
    }

    if (numa_info.nodes[0].numberOfProcessors == 0)
    {
        return -1;
    }
    else
    {
        numaInitialized = 1;
        return 0;
    }
}

void
hwloc_numa_membind(void* ptr, size_t size, int domainId)
{
    int ret = 0;
    hwloc_membind_flags_t flags = HWLOC_MEMBIND_STRICT|HWLOC_MEMBIND_PROCESS;
    hwloc_nodeset_t nodeset = likwid_hwloc_bitmap_alloc();
    likwid_hwloc_bitmap_zero(nodeset);
    likwid_hwloc_bitmap_set(nodeset, domainId);
    ret = likwid_hwloc_set_area_membind_nodeset(hwloc_topology, ptr, size, nodeset, HWLOC_MEMBIND_BIND, flags);
    likwid_hwloc_bitmap_free(nodeset);

    if (ret < 0)
    {
        ERROR;
    }
}

void
hwloc_numa_setInterleaved(int* processorList, int numberOfProcessors)
{
    int i,j;
    int ret = 0;
    likwid_hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    likwid_hwloc_membind_flags_t flags = HWLOC_MEMBIND_STRICT|HWLOC_MEMBIND_PROCESS;
    likwid_hwloc_bitmap_zero(cpuset);
    for (i=0; i<numa_info.numberOfNodes; i++)
    {
        for (j=0; j<numberOfProcessors; j++)
        {
            if (likwid_hwloc_findProcessor(i,processorList[j]))
            {
                likwid_hwloc_bitmap_set(cpuset, i);
            }
        }
    }
    ret = likwid_hwloc_set_membind(hwloc_topology, cpuset, HWLOC_MEMBIND_INTERLEAVE, flags);
    likwid_hwloc_bitmap_free(cpuset);
    if (ret < 0)
    {
        ERROR;
    }
}
#else
int
hwloc_numa_init(void)
{
    return 1;
}

void
hwloc_numa_membind(void* ptr, size_t size, int domainId)
{
    return;
}

void hwloc_numa_setInterleaved(int* processorList, int numberOfProcessors)
{
    return;
}

#endif
