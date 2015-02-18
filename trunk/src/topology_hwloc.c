
#include <stdlib.h>
#include <stdio.h>
#include <error.h>
#include <strUtil.h>

#include <topology.h>
#ifdef LIKWID_USE_HWLOC
#include <hwloc.h>
#include <topology_hwloc.h>
#endif



/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */
/* this was taken from the linux kernel */
#define CPUID                              \
    __asm__ volatile ("cpuid"                             \
            : "=a" (eax),     \
            "=b" (ebx),     \
            "=c" (ecx),     \
            "=d" (edx)      \
            : "0" (eax), "2" (ecx))

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */



/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int readCacheInclusive(int level)
{
    uint32_t eax, ebx, ecx, edx;
    eax = 0x04;
    ecx = level;
    CPUID;
    return edx & 0x2;
}

static int get_stepping(void)
{
    uint32_t eax, ebx, ecx, edx;
    eax = 0x01;
    CPUID;
    return (eax&0xFU);
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */
#ifdef LIKWID_USE_HWLOC
int hwloc_record_objs_of_type_below_obj(hwloc_topology_t t, hwloc_obj_t obj, hwloc_obj_type_t type, int* index, uint32_t **list)
{
    int i;
    int count = 0;
    hwloc_obj_t walker;
    if (!obj->arity) return 0;
    for (i=0;i<obj->arity;i++)
    {
        walker = obj->children[i];
        if (walker->type == type)
        {
            if (list && *list && index)
            {
                (*list)[(*index)++] = walker->os_index;
            }
            count++;
        }
        count += hwloc_record_objs_of_type_below_obj(t, walker, type, index, list);
    }
    return count;
}

void hwloc_init_cpuInfo(cpu_set_t cpuSet)
{
    int i;
    hwloc_obj_t obj;

    hwloc_topology_init(&hwloc_topology);

    hwloc_topology_load(hwloc_topology);
    obj = hwloc_get_obj_by_type(hwloc_topology, HWLOC_OBJ_SOCKET, 0);

    cpuid_info.model = 0;
    cpuid_info.family = 0;
    cpuid_info.isIntel = 0;
    cpuid_info.osname = malloc(MAX_MODEL_STRING_LENGTH * sizeof(char));
    for(i=0;i<obj->infos_count;i++)
    {
        if (strcmp(obj->infos[i].name ,"CPUModelNumber") == 0)
        {
            cpuid_info.model = atoi(hwloc_obj_get_info_by_name(obj, "CPUModelNumber"));
        }
        else if (strcmp(obj->infos[i].name, "CPUFamilyNumber") == 0)
        {
            cpuid_info.family = atoi(hwloc_obj_get_info_by_name(obj, "CPUFamilyNumber"));
        }
        else if (strcmp(obj->infos[i].name, "CPUVendor") == 0 &&
                strcmp(hwloc_obj_get_info_by_name(obj, "CPUVendor"), "GenuineIntel") == 0)
        {
            cpuid_info.isIntel = 1;
        }
        else if (strcmp(obj->infos[i].name ,"CPUModel") == 0)
        {
            strcpy(cpuid_info.osname, obj->infos[i].value);
        }
    }
    cpuid_topology.numHWThreads = hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_PU);
    cpuid_info.stepping = get_stepping();
    DEBUG_PRINT(DEBUGLEV_DEVELOP, HWLOC CpuInfo Family %d Model %d Stepping %d isIntel %d numHWThreads %d activeHWThreads %d,
                            cpuid_info.family,
                            cpuid_info.model,
                            cpuid_info.stepping,
                            cpuid_info.isIntel,
                            cpuid_topology.numHWThreads,
                            cpuid_topology.activeHWThreads)
    return;
}

void hwloc_init_nodeTopology(cpu_set_t cpuSet)
{
    uint32_t apicId;
    uint32_t bitField;
    int level;
    int prevOffset = 0;
    int currOffset = 0;
    cpu_set_t set;
    HWThread* hwThreadPool;
    int maxNumLogicalProcs;
    int maxNumLogicalProcsPerCore;
    int maxNumCores;
    int width;
    hwloc_obj_t obj;
    int realThreadId;
    int sibling;
    int poolsize = 0;
    int threadIdx = 0;
    hwloc_obj_type_t socket_type = HWLOC_OBJ_SOCKET;
    for (uint32_t i=0;i<cpuid_topology.numHWThreads;i++)
    {
        if (CPU_ISSET(i, &cpuSet))
        {
            poolsize = i+1;
        }
    }
    hwThreadPool = (HWThread*) malloc(cpuid_topology.numHWThreads * sizeof(HWThread));
    for (uint32_t i=0;i<cpuid_topology.numHWThreads;i++)
    {
        hwThreadPool[i].apicId = -1;
        hwThreadPool[i].threadId = -1;
        hwThreadPool[i].coreId = -1;
        hwThreadPool[i].packageId = -1;
        hwThreadPool[i].inCpuSet = 0;
    }

    maxNumLogicalProcs = hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_PU);
    maxNumCores = hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_CORE);
    if (hwloc_get_nbobjs_by_type(hwloc_topology, socket_type) == 0)
    {
        socket_type = HWLOC_OBJ_NODE;
    }
    maxNumLogicalProcsPerCore = maxNumLogicalProcs/maxNumCores;
    for (uint32_t i=0; i< cpuid_topology.numHWThreads; i++)
    {
        obj = hwloc_get_obj_by_type(hwloc_topology, HWLOC_OBJ_PU, i);
        if (!obj)
        {
            printf("No obj for CPU %d\n", i);
            continue;
        }
        hwThreadPool[threadIdx].inCpuSet = 1;
        hwThreadPool[threadIdx].apicId = obj->os_index;
        hwThreadPool[threadIdx].threadId = obj->sibling_rank;
        while (obj->type != HWLOC_OBJ_CORE) {
            obj = obj->parent;
        }
        hwThreadPool[threadIdx].coreId = obj->os_index;
        while (obj->type != socket_type) {
            obj = obj->parent;
        }
        hwThreadPool[threadIdx].packageId = obj->os_index;
        DEBUG_PRINT(DEBUGLEV_DEVELOP, HWLOC Thread Pool PU %d Thread %d Core %d Socket %d,
                            hwThreadPool[threadIdx].apicId,
                            hwThreadPool[threadIdx].threadId,
                            hwThreadPool[threadIdx].coreId,
                            hwThreadPool[threadIdx].packageId)
        threadIdx++;
    }

    cpuid_topology.threadPool = hwThreadPool;

    return;
}


void hwloc_init_cacheTopology(void)
{
    int maxNumLevels=0;
    int id=0;
    CacheLevel* cachePool = NULL;
    CacheType type = DATACACHE;
    hwloc_obj_t obj;
    int depth;
    int d;

    /* Sum up all depths with caches */
    depth = hwloc_topology_get_depth(hwloc_topology);
    for (d = 0; d < depth; d++)
    {
        if (hwloc_get_depth_type(hwloc_topology, d) == HWLOC_OBJ_CACHE)
            maxNumLevels++;
    }
    cachePool = (CacheLevel*) malloc(maxNumLevels * sizeof(CacheLevel));
    /* Start at the bottom of the tree to get all cache levels in order */
    depth = hwloc_topology_get_depth(hwloc_topology);
    id = 0;
    for(d=depth-1;d >= 0; d--)
    {
        /* We only need caches, so skip other levels */
        if (hwloc_get_depth_type(hwloc_topology, d) != HWLOC_OBJ_CACHE)
        {
            continue;
        }
        /* Get the cache object */
        obj = hwloc_get_obj_by_depth(hwloc_topology, d, 0);
        /* All caches have this attribute, so safe to access */
        switch (obj->attr->cache.type)
        {
            case HWLOC_OBJ_CACHE_DATA:
                cachePool[id].type = DATACACHE;
                break;
            case HWLOC_OBJ_CACHE_INSTRUCTION:
                cachePool[id].type = INSTRUCTIONCACHE;
                break;
            case HWLOC_OBJ_CACHE_UNIFIED:
                cachePool[id].type = UNIFIEDCACHE;
                break;
            default:
                cachePool[id].type = NOCACHE;
                break;
        }

        cachePool[id].associativity = obj->attr->cache.associativity;
        cachePool[id].level = obj->attr->cache.depth;
        cachePool[id].lineSize = obj->attr->cache.linesize;
        cachePool[id].size = obj->attr->cache.size;
        cachePool[id].sets = 0;
        if ((cachePool[id].associativity * cachePool[id].lineSize) != 0)
        {
            cachePool[id].sets = cachePool[id].size /
                (cachePool[id].associativity * cachePool[id].lineSize);
        }
        /* Count all HWThreads below the current cache */
        cachePool[id].threads = hwloc_record_objs_of_type_below_obj(
                        hwloc_topology, obj, HWLOC_OBJ_PU, NULL, NULL);
        /* We need to read the inclusiveness from CPUID, no possibility in hwloc */
        switch ( cpuid_info.family )
        {
            case MIC_FAMILY:
            case P6_FAMILY:
            case K16_FAMILY:
            case K15_FAMILY:
                cachePool[id].inclusive = readCacheInclusive(cachePool[id].level);
                break;
            /* For K8 and K10 it is known that they are inclusive */
            case K8_FAMILY:
            case K10_FAMILY:
                cachePool[id].inclusive = 1;
                break;
            default:
                ERROR_PLAIN_PRINT(Processor is not supported);
                break;
        }
        id++;
    }

    cpuid_topology.numCacheLevels = maxNumLevels;
    cpuid_topology.cacheLevels = cachePool;
    return;
}

#else

void hwloc_init_cpuInfo(void)
{
    return;
}

void hwloc_init_cpuFeatures(void)
{
    return;
}

void hwloc_init_nodeTopology(void)
{
    return;
}

void hwloc_init_cacheTopology(void)
{
    return;
}
#endif
