
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
    cpuid_info.stepping = (eax&0xFU);
}

static int get_cpu_perf_data(void)
{
    uint32_t eax, ebx, ecx, edx;
    int largest_function = 0;
    eax = 0x00;
    CPUID;
    largest_function = eax;
    if (cpuid_info.family == P6_FAMILY && 0x0A <= largest_function)
    {
        eax = 0x0A;
        CPUID;
        cpuid_info.perf_version   =  (eax&0xFFU);
        cpuid_info.perf_num_ctr   =   ((eax>>8)&0xFFU);
        cpuid_info.perf_width_ctr =  ((eax>>16)&0xFFU);
        cpuid_info.perf_num_fixed_ctr =  (edx&0xFU);

        eax = 0x06;
        CPUID;
        if (eax & (1<<1))
        {
            cpuid_info.turbo = 1;
        }
        else
        {
            cpuid_info.turbo = 0;
        }
    }
    return 0;
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

void hwloc_init_cpuInfo(void)
{
    int i;
    hwloc_obj_t obj;
    
    hwloc_topology_init(&hwloc_topology);
    
    hwloc_topology_load(hwloc_topology);
    obj = hwloc_get_obj_by_type(hwloc_topology, HWLOC_OBJ_SOCKET, 0);

    cpuid_info.model = 0;
    cpuid_info.family = 0;
    cpuid_info.isIntel = 0;
    for(i=0;i<obj->infos_count;i++)
    {
        if (strcmp(obj->infos[i].name ,"CPUModelNumber") == 0)
            cpuid_info.model = atoi(hwloc_obj_get_info_by_name(obj, "CPUModelNumber"));
        if (strcmp(obj->infos[i].name, "CPUFamilyNumber") == 0)
            cpuid_info.family = atoi(hwloc_obj_get_info_by_name(obj, "CPUFamilyNumber"));
        if (strcmp(obj->infos[i].name, "CPUVendor") == 0 && 
                strcmp(hwloc_obj_get_info_by_name(obj, "CPUVendor"), "GenuineIntel") == 0)
            cpuid_info.isIntel = 1;
    }
    cpuid_topology.numHWThreads = hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_PU);
    get_stepping();
    DEBUG_PRINT(DEBUGLEV_DEVELOP, HWLOC CpuInfo Family %d Model %d Stepping %d isIntel %d numHWThreads %d,
                            cpuid_info.family,
                            cpuid_info.model,
                            cpuid_info.stepping,
                            cpuid_info.isIntel,
                            cpuid_topology.numHWThreads)
    return;
}

void hwloc_init_cpuFeatures(void)
{
    int ret;
    FILE* file;
    char buf[1024];
    char ident[30];
    char delimiter[] = " ";
    char* cptr;

    if ( (file = fopen( "/proc/cpuinfo", "r")) == NULL )
    {
        fprintf(stderr, "Cannot open /proc/cpuinfo\n");
        return;
    }
    ret = 0;
    while( fgets(buf, sizeof(buf)-1, file) )
    {
        ret = sscanf(buf, "%s\t:", &(ident[0]));
        if (ret != 1 || strcmp(ident,"flags") != 0)
        {
            continue;
        }
        else
        {
            ret = 1;
            break;
        }
    }
    fclose(file);
    if (ret == 0)
    {
        return;
    }
    
    cpuid_info.featureFlags = 0;
    cpuid_info.features = (char*) malloc(MAX_FEATURE_STRING_LENGTH*sizeof(char));
    cpuid_info.features[0] = '\0';

    cptr = strtok(&(buf[6]),delimiter);
    
    while (cptr != NULL)
    {
        if (strcmp(cptr,"ssse3") == 0)
        {
            cpuid_info.featureFlags |= (1<<SSSE3);
            strcat(cpuid_info.features, "SSSE3 ");
        }
        else if (strcmp(cptr,"sse3") == 0)
        {
            cpuid_info.featureFlags |= (1<<SSE3);
            strcat(cpuid_info.features, "SSE3 ");
        }
        else if (strcmp(cptr,"vsx") == 0)
        {
            cpuid_info.featureFlags |= (1<<VSX);
            strcat(cpuid_info.features, "VSX ");
        }
        else if (strcmp(cptr,"monitor") == 0)
        {
            cpuid_info.featureFlags |= (1<<MONITOR);
            strcat(cpuid_info.features, "MONITOR ");
        }
        else if (strcmp(cptr,"mmx") == 0)
        {
            cpuid_info.featureFlags |= (1<<MMX);
            strcat(cpuid_info.features, "MMX ");
        }
        else if (strcmp(cptr,"sse") == 0)
        {
            cpuid_info.featureFlags |= (1<<SSE);
            strcat(cpuid_info.features, "SSE ");
        }
        else if (strcmp(cptr,"sse2") == 0)
        {
            cpuid_info.featureFlags |= (1<<SSE2);
            strcat(cpuid_info.features, "SSE2 ");
        }
        else if (strcmp(cptr,"acpi") == 0)
        {
            cpuid_info.featureFlags |= (1<<ACPI);
            strcat(cpuid_info.features, "ACPI ");
        }
        else if (strcmp(cptr,"rdtscp") == 0)
        {
            cpuid_info.featureFlags |= (1<<RDTSCP);
            strcat(cpuid_info.features, "RDTSCP ");
        }
        else if (strcmp(cptr,"vmx") == 0)
        {
            cpuid_info.featureFlags |= (1<<VMX);
            strcat(cpuid_info.features, "VMX ");
        }
        else if (strcmp(cptr,"eist") == 0)
        {
            cpuid_info.featureFlags |= (1<<EIST);
            strcat(cpuid_info.features, "EIST ");
        }
        else if (strcmp(cptr,"tm") == 0)
        {
            cpuid_info.featureFlags |= (1<<TM);
            strcat(cpuid_info.features, "TM ");
        }
        else if (strcmp(cptr,"tm2") == 0)
        {
            cpuid_info.featureFlags |= (1<<TM2);
            strcat(cpuid_info.features, "TM2 ");
        }
        else if (strcmp(cptr,"aes") == 0)
        {
            cpuid_info.featureFlags |= (1<<AES);
            strcat(cpuid_info.features, "AES ");
        }
        else if (strcmp(cptr,"rdrand") == 0)
        {
            cpuid_info.featureFlags |= (1<<RDRAND);
            strcat(cpuid_info.features, "RDRAND ");
        }
        else if (strcmp(cptr,"sse4_1") == 0)
        {
            cpuid_info.featureFlags |= (1<<SSE41);
            strcat(cpuid_info.features, "SSE41 ");
        }
        else if (strcmp(cptr,"sse4_2") == 0)
        {
            cpuid_info.featureFlags |= (1<<SSE42);
            strcat(cpuid_info.features, "SSE42 ");
        }
        else if (strcmp(cptr,"avx") == 0)
        {
            cpuid_info.featureFlags |= (1<<AVX);
            strcat(cpuid_info.features, "AVX ");
        }
        else if (strcmp(cptr,"fma") == 0)
        {
            cpuid_info.featureFlags |= (1<<FMA);
            strcat(cpuid_info.features, "FMA ");
        }
        cptr = strtok(NULL, delimiter);
    
    }

    get_cpu_perf_data();

    return;
}

void hwloc_init_nodeTopology(void)
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
    hwloc_obj_type_t socket_type = HWLOC_OBJ_NODE;

    hwThreadPool = (HWThread*) malloc(cpuid_topology.numHWThreads * sizeof(HWThread));

    maxNumLogicalProcs = hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_PU);
    maxNumCores = hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_CORE);
    if (hwloc_get_nbobjs_by_type(hwloc_topology, socket_type) == 0)
    {
        socket_type = HWLOC_OBJ_SOCKET;
    }
    maxNumLogicalProcsPerCore = maxNumLogicalProcs/maxNumCores;
    for (uint32_t i=0; i< maxNumLogicalProcs; i++)
    {
        obj = hwloc_get_obj_by_type(hwloc_topology, HWLOC_OBJ_PU, i);
        realThreadId = obj->os_index;
        hwThreadPool[realThreadId].apicId = obj->os_index;
        hwThreadPool[realThreadId].threadId = obj->sibling_rank;
        while (obj->type != HWLOC_OBJ_CORE) {
            obj = obj->parent;
        }
        hwThreadPool[realThreadId].coreId = obj->os_index;
        while (obj->type != socket_type) {
            obj = obj->parent;
        }
        hwThreadPool[realThreadId].packageId = obj->os_index;
        DEBUG_PRINT(DEBUGLEV_DEVELOP, HWLOC Thread Pool PU %d Thread %d Core %d Socket %d, 
                            realThreadId,
                            hwThreadPool[realThreadId].threadId,
                            hwThreadPool[realThreadId].coreId,
                            hwThreadPool[realThreadId].packageId)
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
