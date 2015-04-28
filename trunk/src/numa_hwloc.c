#include <stdlib.h>
#include <stdio.h>

#include <error.h>
//#include <strUtil.h>

#include <numa.h>
#include <topology.h>
#ifdef LIKWID_USE_HWLOC
#include <hwloc.h>
#include <topology_hwloc.h>
#endif



/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */
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

int hwloc_findProcessor(int nodeID, int cpuID)
{
    hwloc_obj_t obj;
    int i;
    int pu_count = hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_PU);
    
    for (i=0; i<pu_count; i++)
    {
        obj = hwloc_get_obj_by_type(hwloc_topology, HWLOC_OBJ_PU, i);
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

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */
int hwloc_numa_init(void)
{
    int errno;
    uint32_t i;
    int d;
    int depth;
    int cores_per_socket;
    hwloc_obj_t obj;
    const struct hwloc_distances_s* distances;
    hwloc_obj_type_t hwloc_type = HWLOC_OBJ_NODE;

    if (!hwloc_topology)
    {
        hwloc_topology_init(&hwloc_topology);
        hwloc_topology_load(hwloc_topology);
    }

    numa_info.numberOfNodes = hwloc_get_nbobjs_by_type(hwloc_topology, hwloc_type);

    /* If the amount of NUMA nodes == 0, there is actually no NUMA node, hence
       aggregate all sockets in the system into the single virtually created NUMA node */
    if (numa_info.numberOfNodes == 0)
    {
        hwloc_type = HWLOC_OBJ_SOCKET;
        numa_info.numberOfNodes = 1;
        maxIdConfiguredNode = 1;
        i = 0;
    
        numa_info.nodes = (NumaNode*) malloc(sizeof(NumaNode));
        if (!numa_info.nodes)
        {
            fprintf(stderr,"No memory to allocate %ld byte for nodes array\n",sizeof(NumaNode));
            return -1;
        }
        
        numa_info.nodes[i].id = i;
        numa_info.nodes[i].numberOfProcessors = 0;
        numa_info.nodes[i].totalMemory = getTotalNodeMem(i);
        numa_info.nodes[i].freeMemory = getFreeNodeMem(i);
        numa_info.nodes[i].processors = (uint32_t*) malloc(MAX_NUM_THREADS * sizeof(uint32_t));
        if (!numa_info.nodes[i].processors)
        {
            fprintf(stderr,"No memory to allocate %ld byte for processors array of NUMA node %d\n",MAX_NUM_THREADS * sizeof(uint32_t),i);
            return -1;
        }
        numa_info.nodes[i].distances = (uint32_t*) malloc(sizeof(uint32_t));
        if (!numa_info.nodes[i].distances)
        {
            fprintf(stderr,"No memory to allocate %ld byte for distances array of NUMA node %d\n",sizeof(uint32_t),i);
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
        if (!numa_info.nodes)
        {
            fprintf(stderr,"No memory to allocate %ld byte for nodes array\n",numa_info.numberOfNodes * sizeof(NumaNode));
            return -1;
        }
        depth = hwloc_get_type_depth(hwloc_topology, hwloc_type);
        distances = hwloc_get_whole_distance_matrix_by_type(hwloc_topology, hwloc_type);
        for (i=0; i<numa_info.numberOfNodes; i++)
        {
            obj = hwloc_get_obj_by_depth(hwloc_topology, depth, i);
            
            numa_info.nodes[i].id = obj->os_index;
            if (obj->os_index > maxIdConfiguredNode)
                maxIdConfiguredNode = obj->os_index;
            if (obj->memory.local_memory == 0)
            {
                numa_info.nodes[i].totalMemory = getTotalNodeMem(i);
            }
            else
            {
                numa_info.nodes[i].totalMemory = (uint64_t)(obj->memory.local_memory/1024);
            }
            
            /* freeMemory not detected by hwloc, do it the native way */
            numa_info.nodes[i].freeMemory = getFreeNodeMem(i);
            numa_info.nodes[i].processors = (uint32_t*) malloc(MAX_NUM_THREADS * sizeof(uint32_t));
            if (!numa_info.nodes[i].processors)
            {
                fprintf(stderr,"No memory to allocate %ld byte for processors array of NUMA node %d\n",MAX_NUM_THREADS * sizeof(uint32_t), i);
                return -1;
            }
            d = 0;
            numa_info.nodes[i].numberOfProcessors = hwloc_record_objs_of_type_below_obj(
                    hwloc_topology, obj, HWLOC_OBJ_PU, &d, &numa_info.nodes[i].processors);
            
            numa_info.nodes[i].distances = (uint32_t*) malloc(numa_info.numberOfNodes * sizeof(uint32_t));
            if (!numa_info.nodes[i].distances)
            {
                fprintf(stderr,"No memory to allocate %ld byte for distances array of NUMA node %d\n",numa_info.numberOfNodes*sizeof(uint32_t),i);
                return -1;
            }
            if (distances)
            {
                numa_info.nodes[i].numberOfDistances = distances->nbobjs;
                for(d=0;d<distances->nbobjs;d++)
                {
                    numa_info.nodes[i].distances[d] = distances->latency[i*distances->nbobjs +d] * distances->latency_base;
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
        return 0;
    }
}

void hwloc_numa_membind(void* ptr, size_t size, int domainId)
{
    int ret = 0;
    hwloc_membind_flags_t flags = HWLOC_MEMBIND_STRICT|HWLOC_MEMBIND_PROCESS;
    hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
    
    hwloc_bitmap_zero(nodeset);
    hwloc_bitmap_set(nodeset, domainId);
    
    ret = hwloc_set_area_membind_nodeset(hwloc_topology, ptr, size, nodeset, HWLOC_MEMBIND_BIND, flags);
    
    hwloc_bitmap_free(nodeset);

    if (ret < 0)
    {
        ERROR;
    }
}



void hwloc_numa_setInterleaved(int* processorList, int numberOfProcessors)
{
    int i,j;
    int ret = 0;
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    hwloc_membind_flags_t flags = HWLOC_MEMBIND_STRICT|HWLOC_MEMBIND_PROCESS;
    
    hwloc_bitmap_zero(cpuset);
    
    for (i=0; i<numa_info.numberOfNodes; i++)
    {
        for (j=0; j<numberOfProcessors; j++)
        {
            if (hwloc_findProcessor(i,processorList[j]))
            {
                hwloc_bitmap_set(cpuset, i);
            }
        }
    }
    
    
    ret = hwloc_set_membind(hwloc_topology, cpuset, HWLOC_MEMBIND_INTERLEAVE, flags);
    
    hwloc_bitmap_free(cpuset);
    
    if (ret < 0)
    {
        ERROR;
    }
}
#else
int hwloc_numa_init(void)
{
    return 1;
}

void hwloc_numa_membind(void* ptr, size_t size, int domainId)
{
    return;
}

void hwloc_numa_setInterleaved(int* processorList, int numberOfProcessors)
{
    return;
}

#endif
