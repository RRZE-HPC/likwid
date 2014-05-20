/*
 * =======================================================================================
 *
 *      Filename:  numa.c
 *
 *      Description:  Implementation of Linux NUMA interface
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
/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <dirent.h>
#ifdef HAS_MEMPOLICY
#include <linux/mempolicy.h>
#endif

#include <error.h>
#include <bstrlib.h>
#include <numa.h>
#include <strUtil.h>

/* #####   EXPORTED VARIABLES   ########################################### */


NumaTopology numa_info;

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#ifdef HAS_MEMPOLICY
#define get_mempolicy(policy,nmask,maxnode,addr,flags) syscall(SYS_get_mempolicy,policy,nmask,maxnode,addr,flags)
#define set_mempolicy(mode,nmask,maxnode) syscall(SYS_set_mempolicy,mode,nmask,maxnode)
#define mbind(start, len, nmask, maxnode, flags) syscall(SYS_mbind,(start),len,MPOL_BIND,(nmask),maxnode,flags)
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
nodeMeminfo(int node, uint64_t* totalMemory, uint64_t* freeMemory)
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
    int i,j;
    struct bstrList* tokens;
    unsigned long val;
    char* endptr;
    int cursor=0;
//    int unitSize = (int) (sizeof(unsigned long)*8);
    int unitSize = (int) 32; /* 8 nibbles */

    *list = (uint32_t*) malloc(MAX_NUM_THREADS * sizeof(uint32_t));

    /* the cpumap interface should be always there */
    filename = bformat("/sys/devices/system/node/node%d/cpumap", node); 

    if (NULL != (fp = fopen (bdata(filename), "r"))) 
    {

        src = bread ((bNread) fread, fp);
        tokens = bsplit(src,',');

        for (i=(tokens->qty-1); i>=0 ;i--)
        {
            val = strtoul((char*) tokens->entry[i]->data, &endptr, 16);

            if ((errno != 0 && val == LONG_MAX )
                    || (errno != 0 && val == 0)) 
            {
                ERROR;
            }

            if (endptr == (char*) tokens->entry[i]->data) 
            {
                ERROR_PLAIN_PRINT(No digits were found);
            }

            if (val != 0UL)
            {
                for (j=0; j<unitSize; j++)
                {
                    if (val&(1UL<<j))
                    {
                        if (count < MAX_NUM_THREADS)
                        {
                            (*list)[count] = (j+cursor);
                        }
                        else
                        {
                            ERROR_PRINT(Number Of threads %d too large,count);
                        }
                        count++;
                    }
                }
            }
            cursor += unitSize;
        }

        bstrListDestroy(tokens);
        bdestroy(src);
        bdestroy(filename);
        fclose(fp); 

        /* FIXME: CPU list here is not physical cores first but numerical sorted */


        return count;
    }

    /* something went wrong */
    return -1;
}
 
static int
nodeDistanceList(int node, int numberOfNodes, uint32_t** list)
{
    FILE *fp;
    bstring filename;
    int count = 0;
    bstring src;
    struct bstrList* tokens;

    *list = (uint32_t*) malloc(numberOfNodes * sizeof(uint32_t));

    /* the distance interface should be always there */
    filename = bformat("/sys/devices/system/node/node%d/distance", node);

    if (NULL != (fp = fopen (bdata(filename), "r")))
    {

        src = bread ((bNread) fread, fp);
        tokens = bsplit(src,' ');

        for (int i=0; i<(tokens->qty); i++)
        {
            if (count < numberOfNodes)
            {
                (*list)[count] = (uint32_t)strtoul((char*) (tokens->entry[i]->data), NULL, 10);
            }
            else
            {
                ERROR_PRINT(Number Of nodes %d too large,count);
            }
            count++;
        }

        bstrListDestroy(tokens);
        bdestroy(src);
        bdestroy(filename);
        fclose(fp);
        return count;
    }

    /* something went wrong */
    return -1;
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
        numa_info.nodes[i].numberOfDistances = nodeDistanceList(i, numa_info.numberOfNodes, &numa_info.nodes[i].distances);
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
    long i;
    int j;
    int ret=0;
    unsigned long numberOfNodes = 65;
    unsigned long mask = 0UL;

    for (i=0; i<numa_info.numberOfNodes; i++)
    {
        for (j=0; j<numberOfProcessors; j++)
        {
            if (findProcessor(i,processorList[j]))
            {
                mask |= (1UL<<i);
                break;
            }
        }
    }

    ret = set_mempolicy(MPOL_INTERLEAVE,&mask,numberOfNodes);

    if (ret < 0)
    {
        ERROR;
    }
}

void
numa_membind(void* ptr, size_t size, int domainId)
{
    int ret=0;
    unsigned long mask = 0UL;
    unsigned int flags = 0U;

    flags |= MPOL_MF_STRICT;
    mask |= (1UL<<domainId);

    ret = mbind(ptr, size, &mask, numa_info.numberOfNodes+1, flags);

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

void
numa_membind(void* ptr, size_t size, int domainId)
{
    printf("MBIND NOT supported in kernel!\n");
}

#endif


