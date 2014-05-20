/*
 * ===========================================================================
 *
 *      Filename:  numa.c
 *
 *      Description:  Implementation of Linux NUMA interface
 *
 *      Version:  <VERSION>
 *      Created:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Company:  RRZE Erlangen
 *      Project:  likwid
 *      Copyright:  Copyright (c) 2010, Jan Treibig
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License, v2, as
 *      published by the Free Software Foundation
 *     
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *     
 *      You should have received a copy of the GNU General Public License
 *      along with this program; if not, write to the Free Software
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * ===========================================================================
 */

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sched.h>
#include <bstrlib.h>
#include <sys/types.h>
#include <dirent.h>
#ifdef HAS_MEMPOLICY
#include <linux/mempolicy.h>
#endif

#include <error.h>
#include <numa.h>
#include <strUtil.h>

/* #####   EXPORTED VARIABLES   ########################################### */

NumaTopology numa_info;

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#ifdef HAS_MEMPOLICY
#define get_mempolicy(policy,nmask,maxnode,addr,flags) syscall(SYS_get_mempolicy,policy,nmask,maxnode,addr,flags)
#define set_mempolicy(mode,nmask,maxnode) syscall(SYS_set_mempolicy,mode,nmask,maxnode)
#endif

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int maxIdConfiguredNode = 0;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static void
setConfiguredNodes(void)
{
	DIR *dir;
	struct dirent *de;

	dir = opendir("/sys/devices/system/node");

	if (!dir) 
    {
		maxIdConfiguredNode = 0;
	}
    else
    {
		while ((de = readdir(dir)) != NULL) 
        {
            int nd;
            if (strncmp(de->d_name, "node", 4))
            {
                continue;
            }

            nd = str2int(de->d_name+4);

            if (maxIdConfiguredNode < nd)
            {
                maxIdConfiguredNode = nd;
            }
        }
		closedir(dir);
	}
}


static void
nodeMeminfo(int node, uint32_t* totalMemory, uint32_t* freeMemory)
{ 
	FILE *fp; 
    bstring filename;
    bstring totalString = bformat("MemTotal:");
    bstring freeString  = bformat("MemFree:");
    int i;

	filename = bformat("/sys/devices/system/node/node%d/meminfo", node); 

	if (NULL != (fp = fopen (bdata(filename), "r"))) 
	{
		bstring src = bread ((bNread) fread, fp);
        struct bstrList* tokens = bsplit(src,(char) '\n');
        
        for (i=0;i<tokens->qty;i++)
        {
            if (binstr(tokens->entry[i],0,totalString) != BSTR_ERR)
            {
                 bstring tmp = bmidstr (tokens->entry[i], 18, blength(tokens->entry[i])-18 );
                 bltrimws(tmp);
                 struct bstrList* subtokens = bsplit(tmp,(char) ' ');
                 *totalMemory = str2int(bdata(subtokens->entry[0]));
            }
            else if (binstr(tokens->entry[i],0,freeString) != BSTR_ERR)
            {
                 bstring tmp = bmidstr (tokens->entry[i], 18, blength(tokens->entry[i])-18  );
                 bltrimws(tmp);
                 struct bstrList* subtokens = bsplit(tmp,(char) ' ');
                 *freeMemory = str2int(bdata(subtokens->entry[0]));
            }
        }
	} 
    else
    {
        ERROR;
    }

	fclose(fp); 
}

static int
nodeProcessorList(int node, uint32_t** list)
{
	FILE *fp; 
    bstring filename;
    int count = 0;
    bstring src;

    *list = (uint32_t*) malloc(MAX_NUM_THREADS * sizeof(uint32_t));

	filename = bformat("/sys/devices/system/node/node%d/cpulist", node); 
    /* The format in cpumap is not really documented, at least I
     * found no documentation.*/

	if (NULL != (fp = fopen (bdata(filename), "r"))) 
	{
		src = bread ((bNread) fread, fp);
    }
    else
    {
        return -1;
    }

	fclose(fp); 
    count = (int)  bstr_to_cpuset_physical(*list,  src);
    bdestroy(src);

    return count;
}

static int
findProcessor(uint32_t nodeId, uint32_t coreId)
{
    int i;

    for (i=0; i<numa_info.nodes[nodeId].numberOfProcessors; i++)
    {
        if (numa_info.nodes[nodeId].processors[i] == coreId)
        {
            return 1;
        }
    }
    return 0;
}


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

#ifdef HAS_MEMPOLICY
int
numa_init()
{
    int errno;
    uint32_t i;

    if (get_mempolicy(NULL, NULL, 0, 0, 0) < 0 && errno == ENOSYS)
    {
        return -1; 
    }

    /* First determine maximum number of nodes */
    setConfiguredNodes();
    numa_info.numberOfNodes = maxIdConfiguredNode+1;
    numa_info.nodes = (NumaNode*) malloc(numa_info.numberOfNodes * sizeof(NumaNode));

    for (i=0; i<numa_info.numberOfNodes; i++)
    {
        nodeMeminfo(i, &numa_info.nodes[i].totalMemory, &numa_info.nodes[i].freeMemory);
        numa_info.nodes[i].numberOfProcessors = nodeProcessorList(i,&numa_info.nodes[i].processors);
    }

    if (numa_info.nodes[0].numberOfProcessors < 0)
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

void 
numa_setInterleaved(int* processorList, int numberOfProcessors)
{
    uint32_t i;
    int j;
    int ret=0;
    int numberOfNodes = 0;
    unsigned long mask = 0UL;

    for (i=0; i<numa_info.numberOfNodes; i++)
    {
        for (j=0; j<numberOfProcessors; j++)
        {
            if (findProcessor(i,processorList[j]))
            {
                mask |= (1UL<<i);
                numberOfNodes++;
                continue;
            }
        }
    }

    ret = set_mempolicy(MPOL_INTERLEAVE,&mask,numberOfNodes);

    if (ret < 0)
    {
        ERROR;
    }
}
#else
int
numa_init()
{
    printf("MEMPOLICY NOT supported in kernel!\n");
}

void 
numa_setInterleaved(int* processorList, int numberOfProcessors)
{
    printf("MEMPOLICY NOT supported in kernel!\n");
}
#endif


