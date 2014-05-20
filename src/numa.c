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
#ifdef LIKWID_USE_HWLOC
#include <cpuid-hwloc.h>
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

#ifdef LIKWID_USE_HWLOC
uint64_t getFreeNodeMem(int nodeId)
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
            }
        }
        fclose(fp);
	}
    else
    {
        ERROR;
    }

	
	return free;
    
}

uint64_t getTotalNodeMem(int nodeId)
{
	FILE *fp;
    bstring filename;
    uint64_t free = 0;
    bstring freeString  = bformat("MemTotal:");
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
            }
        }
        fclose(fp);
	}
    else
    {
        ERROR;
    }

	
	return free;
    
}
#endif


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

#ifdef HAS_MEMPOLICY
int
numa_init()
{
    int errno;
    uint32_t i;
#ifdef LIKWID_USE_HWLOC
	int use_native = 0;
	int d;
    int depth;
    hwloc_obj_t obj, tmp_obj;
    const struct hwloc_distances_s* distances;
    hwloc_obj_type_t numa_type = HWLOC_OBJ_NODE;
    int cores_per_socket;
#else
	int use_native = 1;
#endif


    if (get_mempolicy(NULL, NULL, 0, 0, 0) < 0 && errno == ENOSYS)
    {
        return -1; 
    }
#ifdef LIKWID_USE_HWLOC    
    if (!use_native)
    {
		if (!hwloc_topology)
		{
		    hwloc_topology_init(&hwloc_topology);
		    hwloc_topology_load(hwloc_topology);
		}
		/* First determine maximum number of nodes */
		numa_info.numberOfNodes = hwloc_get_nbobjs_by_type(hwloc_topology, numa_type);
		if (numa_info.numberOfNodes == 0)
		{
			numa_type = HWLOC_OBJ_SOCKET;
			numa_info.numberOfNodes = 1;
			hwloc_get_nbobjs_by_type(hwloc_topology, numa_type);
			maxIdConfiguredNode = 1;
			i = 0;
			
			numa_info.nodes = (NumaNode*) malloc(sizeof(NumaNode));
			if (!numa_info.nodes)
			{
				fprintf(stderr,"No memory to allocate %d byte for nodes array\n",sizeof(NumaNode));
				return -1;
			}
			numa_info.nodes[i].id = i;
			numa_info.nodes[i].numberOfProcessors = 0;
			numa_info.nodes[i].totalMemory = getTotalNodeMem(i);
			numa_info.nodes[i].freeMemory = getFreeNodeMem(i);
			numa_info.nodes[i].processors = (uint32_t*) malloc(MAX_NUM_THREADS * sizeof(uint32_t));
			if (!numa_info.nodes[i].processors)
			{
				fprintf(stderr,"No memory to allocate %d byte for processors array of NUMA node %d\n",
								MAX_NUM_THREADS * sizeof(uint32_t),i);
				return -1;
			}
			numa_info.nodes[i].distances = (uint32_t*) malloc(sizeof(uint32_t));
			if (!numa_info.nodes[i].distances)
			{
				fprintf(stderr,"No memory to allocate %d byte for distances array of NUMA node %d\n",sizeof(uint32_t),i);
				return -1;
			}
			numa_info.nodes[i].distances[i] = 10;
			numa_info.nodes[i].numberOfDistances = 1;
			cores_per_socket = cpuid_topology.numHWThreads/cpuid_topology.numSockets;
			for (d=0; d<hwloc_get_nbobjs_by_type(hwloc_topology, hwloc_type); d++)
			{
				obj = hwloc_get_obj_by_type(hwloc_topology, hwloc_type, d);
				/* depth is here used as index in the processors array */		
				depth = d * cores_per_socket;
				numa_info.nodes[i].numberOfProcessors += hwloc_record_objs_of_type_below_obj(
					    hwloc_topology, obj, HWLOC_OBJ_PU, &depth, &numa_info.nodes[i].processors);
			}
		}
		else
		{
			numa_info.nodes = (NumaNode*) malloc(numa_info.numberOfNodes * sizeof(NumaNode));
			/* Get depth of NUMA nodes in hwloc topology */
			depth = hwloc_get_type_depth(hwloc_topology, numa_type);
			/* Get distance matrix, contains all node to node latencies*/
			distances = hwloc_get_whole_distance_matrix_by_type(hwloc_topology, numa_type);
			for (i=0; i<numa_info.numberOfNodes; i++)
			{
				/* Get NUMA node object */
				obj = hwloc_get_obj_by_depth(hwloc_topology, depth, i);
				
				/* Use the physical ID of the NUMA node as id */
				numa_info.nodes[i].id = obj->os_index;
				/* Retrieve maximum ID */
				if (obj->os_index > maxIdConfiguredNode)
				    maxIdConfiguredNode = obj->os_index;
				/* Save memory amount of the current NUMA node */
				
				if (obj->memory.local_memory == 0)
				{
					numa_info.nodes[i].totalMemory = getTotalNodeMem(i);
				}
				else
				{
					numa_info.nodes[i].totalMemory = obj->memory.local_memory;
				}
				/* freeMemory not detected by hwloc, do it the native way */
				numa_info.nodes[i].freeMemory = getFreeNodeMem(i);
				/* Create and fill list of processors of the current NUMA node */
				numa_info.nodes[i].processors = (uint32_t*) malloc(MAX_NUM_THREADS * sizeof(uint32_t));
				d = 0;
				numa_info.nodes[i].numberOfProcessors = hwloc_record_objs_of_type_below_obj(
				        hwloc_topology, obj, HWLOC_OBJ_PU, &d, &numa_info.nodes[i].processors);
				
				/* Determine the distances of the current NUMA node to all others in the system */ 
				numa_info.nodes[i].distances = (uint32_t*) malloc(numa_info.numberOfNodes * sizeof(uint32_t));
				if (distances)
				{
					for(d=0;d<distances->nbobjs;d++)
					{
						numa_info.nodes[i].distances[d] = distances->latency[i*distances->nbobjs +d] * distances->latency_base;
					}
					numa_info.nodes[i].numberOfDistances = distances->nbobjs;
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
	}
#endif	
	if (use_native)
	{
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


