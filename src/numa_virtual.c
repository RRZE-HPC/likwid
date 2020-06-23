/*
 * =======================================================================================
 *
 *      Filename:  numa_virtual.c
 *
 *      Description:  Virtual/Fake NUMA backend
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *                Tobias Auerochs, tobi291019@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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

static uint64_t
getFreeNodeMem(void)
{
    FILE *fp;
    uint64_t free = 0;
    bstring freeString  = bformat("MemFree:");
    int i;

    if (!access("/proc/meminfo", R_OK))
    {
        if (NULL != (fp = fopen ("/proc/meminfo", "r")))
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
        ERROR;
    }
    bdestroy(freeString);
    return free;
}

static uint64_t
getTotalNodeMem(void)
{
    int i;
    FILE *fp;
    uint64_t total = 0;
    bstring totalString  = bformat("MemTotal:");

    if (!access("/proc/meminfo", R_OK))
    {
        if (NULL != (fp = fopen ("/proc/meminfo", "r")))
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
        ERROR;
    }

    bdestroy(totalString);
    return total;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
virtual_numa_init()
{
    int i;

    numa_info.numberOfNodes = 1;

    numa_info.nodes = (NumaNode*) malloc(sizeof(NumaNode));
    if (!numa_info.nodes)
    {
        fprintf(stderr,"No memory to allocate %ld byte for nodes array\n",sizeof(NumaNode));
        return -1;
    }
    numa_info.nodes[0].id = 0;
    numa_info.nodes[0].numberOfProcessors = cpuid_topology.numHWThreads;
    numa_info.nodes[0].totalMemory = getTotalNodeMem();
    numa_info.nodes[0].freeMemory = getFreeNodeMem();
    numa_info.nodes[0].processors = (uint32_t*) malloc(cpuid_topology.numHWThreads * sizeof(uint32_t));
    if (!numa_info.nodes[0].processors)
    {
        fprintf(stderr,"No memory to allocate %ld byte for processors array of NUMA node %d\n",
                cpuid_topology.numHWThreads * sizeof(uint32_t),0);
        return -1;
    }
    for (i = 0; i < cpuid_topology.numHWThreads; i++)
    {
        numa_info.nodes[0].processors[i] = i;
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

    return 0;
}
