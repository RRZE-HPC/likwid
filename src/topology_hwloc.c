/*
 * =======================================================================================
 *
 *      Filename:  topology_hwloc.c
 *
 *      Description:  Interface to the hwloc based topology backend
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Authors:  Thomas Roehl (tr), thomas.roehl@googlemail.com
 *
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

#include <topology.h>
#include <affinity.h>
#if !defined(__ARM_ARCH_7A__) && !defined(__ARM_ARCH_8A)
#include <cpuid.h>
#endif
#ifdef LIKWID_USE_HWLOC
#include <hwloc.h>
#include <topology_hwloc.h>
#endif

/* #####  EXPORTED VARIABLESE   ########################################### */

hwloc_topology_t hwloc_topology = NULL;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */
#if defined(__ARM_ARCH_8A) || defined(__ARM_ARCH_7A__)
int parse_cpuinfo(uint32_t* family, uint32_t* variant, uint32_t *stepping, uint32_t *part, uint32_t *vendor)
{
    int i = 0;
    FILE *fp = NULL;
    uint32_t f = 0;
    uint32_t v = 0;
    uint32_t s = 0;
    uint32_t p = 0;
    uint32_t vend = 0;
    int (*ownatoi)(const char*);
    ownatoi = &atoi;

    if (NULL != (fp = fopen ("/proc/cpuinfo", "r")))
    {
        const_bstring familyString = bformat("CPU architecture");
        const_bstring variantString = bformat("CPU variant");
        const_bstring steppingString = bformat("CPU revision");
        const_bstring partString = bformat("CPU part");
        const_bstring vendString = bformat("CPU implementer");
        bstring src = bread ((bNread) fread, fp);
        struct bstrList* tokens = bsplit(src,(char) '\n');
        bdestroy(src);
        fclose(fp);
        for (i=0;i<tokens->qty;i++)
        {
            if ((f == 0) && (binstr(tokens->entry[i],0,familyString) != BSTR_ERR))
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                f = ownatoi(bdata(subtokens->entry[1]));
                bstrListDestroy(subtokens);
            }
            else if ((s == 0) && (binstr(tokens->entry[i],0,steppingString) != BSTR_ERR))
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                s = ownatoi(bdata(subtokens->entry[1]));
                bstrListDestroy(subtokens);
            }
            else if ((v == 0) && (binstr(tokens->entry[i],0,variantString) != BSTR_ERR))
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                v = strtol(bdata(subtokens->entry[1]), NULL, 0);
                bstrListDestroy(subtokens);
            }
            else if ((p == 0) && (binstr(tokens->entry[i],0,partString) != BSTR_ERR))
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                p = strtol(bdata(subtokens->entry[1]), NULL, 0);
                bstrListDestroy(subtokens);
            }
            else if ((p == 0) && (binstr(tokens->entry[i],0,vendString) != BSTR_ERR))
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                vend = strtol(bdata(subtokens->entry[1]), NULL, 0);
                bstrListDestroy(subtokens);
            }
        }
        bstrListDestroy(tokens);
        /*bdestroy(familyString);
        bdestroy(variantString);
        bdestroy(steppingString);*/
    }
    else
    {
        return -1;
    }
    *family = f;
    *variant = v;
    *stepping = s;
    *part = p;
    *vendor = vend;
    return 0;
}

int parse_cpuname(char *name)
{
    FILE *fp = NULL;
    if (NULL != (fp = fopen ("/proc/cpuinfo", "r")))
    {
        int found = 0;
        const_bstring nameString = bformat("Hardware");
        const_bstring nameString2 = bformat("model name");
        bstring src = bread ((bNread) fread, fp);
        struct bstrList* tokens = bsplit(src,(char) '\n');
        bdestroy(src);
        fclose(fp);
        for (int i = 0; i < tokens->qty; i++)
        {
            if ((binstr(tokens->entry[i],0,nameString) != BSTR_ERR))
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                strncpy(name, bdata(subtokens->entry[1]), MAX_MODEL_STRING_LENGTH-1);
                bstrListDestroy(subtokens);
                found = 1;
                break;
            }
        }
        if (!found)
        {
            for (int i = 0; i < tokens->qty; i++)
            {
                if ((binstr(tokens->entry[i],0,nameString2) != BSTR_ERR))
                {
                    struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                    bltrimws(subtokens->entry[1]);
                    strncpy(name, bdata(subtokens->entry[1]), MAX_MODEL_STRING_LENGTH-1);
                    bstrListDestroy(subtokens);
                    break;
                }
            }
        }
        bstrListDestroy(tokens);
    }
    return 0;
}
#endif

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */
#if !defined(__ARM_ARCH_7A__) && !defined(__ARM_ARCH_8A)
static int
readCacheInclusiveIntel(int level)
{
    uint32_t eax = 0x0U, ebx = 0x0U, ecx = 0x0U, edx = 0x0U;
    eax = 0x04;
    ecx = level;
    CPUID(eax, ebx, ecx, edx);
    return edx & 0x2;
}

static int readCacheInclusiveAMD(int level)
{
    uint32_t eax = 0x0U, ebx = 0x0U, ecx = 0x0U, edx = 0x0U;
    eax = 0x8000001D;
    ecx = level;
    CPUID(eax, ebx, ecx, edx);
    return (edx & (0x1<<1));
}
#endif

#ifdef LIKWID_USE_HWLOC
int
likwid_hwloc_record_objs_of_type_below_obj(
        hwloc_topology_t t,
        hwloc_obj_t obj,
        hwloc_obj_type_t type,
        int* index,
        uint32_t **list)
{
    int i;
    int count = 0;
    hwloc_obj_t walker;
    if (!obj) return 0;
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
        count += likwid_hwloc_record_objs_of_type_below_obj(t, walker, type, index, list);
    }
    return count;
}

void
hwloc_init_cpuInfo(cpu_set_t cpuSet)
{
    int i;
    hwloc_obj_t obj;
    if (perfmon_verbosity <= 1)
    {
        setenv("HWLOC_HIDE_ERRORS", "1", 1);
    }
    if (!hwloc_topology)
    {
        likwid_hwloc_topology_init(&hwloc_topology);
        likwid_hwloc_topology_set_flags(hwloc_topology, HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM|HWLOC_TOPOLOGY_FLAG_WHOLE_IO );
        likwid_hwloc_topology_load(hwloc_topology);
    }
    obj = likwid_hwloc_get_obj_by_type(hwloc_topology, HWLOC_OBJ_SOCKET, 0);

    cpuid_info.model = 0;
    cpuid_info.family = 0;
    cpuid_info.isIntel = 0;
    cpuid_info.stepping = 0;
    cpuid_info.vendor = 0;
    cpuid_info.part = 0;
    cpuid_info.osname = malloc(MAX_MODEL_STRING_LENGTH * sizeof(char));
    cpuid_info.osname[0] = '\0';
    if (!obj)
    {
        return;
    }

    const char * info;
#ifdef __x86_64
    if ((info = hwloc_obj_get_info_by_name(obj, "CPUModelNumber")))
        cpuid_info.model = atoi(info);
    if ((info = likwid_hwloc_obj_get_info_by_name(obj, "CPUFamilyNumber")))
       cpuid_info.family = atoi(info);
    if ((info = likwid_hwloc_obj_get_info_by_name(obj, "CPUVendor")))
        cpuid_info.isIntel = strcmp(info, "GenuineIntel") == 0;
    if ((info = likwid_hwloc_obj_get_info_by_name(obj, "CPUStepping")))
        cpuid_info.stepping = atoi(info);
#endif
#ifdef __ARM_ARCH_7A__
    if ((info = hwloc_obj_get_info_by_name(obj, "CPUArchitecture")))
       cpuid_info.family = atoi(info);
    if ((info = hwloc_obj_get_info_by_name(obj, "CPURevision")))
        cpuid_info.model = atoi(info);
    if (cpuid_info.family == 0 || cpuid_info.model == 0)
    {
        uint32_t part = 0;
        parse_cpuinfo(&cpuid_info.family, &cpuid_info.model, &cpuid_info.stepping, &cpuid_info.part, &cpuid_info.vendor);
        parse_cpuname(cpuid_info.osname);
    }
#endif
#ifdef __ARM_ARCH_8A
    uint32_t part = 0;
    parse_cpuinfo(&cpuid_info.family, &cpuid_info.model, &cpuid_info.stepping, &cpuid_info.part, &cpuid_info.vendor);
    parse_cpuname(cpuid_info.osname);
#endif
    if ((info = hwloc_obj_get_info_by_name(obj, "CPUModel")))
        strcpy(cpuid_info.osname, info);

    cpuid_topology.numHWThreads = likwid_hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_PU);
    if (cpuid_topology.activeHWThreads > cpuid_topology.numHWThreads)
        cpuid_topology.numHWThreads = cpuid_topology.activeHWThreads;
    DEBUG_PRINT(DEBUGLEV_DEVELOP, HWLOC CpuInfo Family %d Model %d Stepping %d Vendor 0x%X Part 0x%X isIntel %d numHWThreads %d activeHWThreads %d,
                            cpuid_info.family,
                            cpuid_info.model,
                            cpuid_info.stepping,
                            cpuid_info.vendor,
                            cpuid_info.part,
                            cpuid_info.isIntel,
                            cpuid_topology.numHWThreads,
                            cpuid_topology.activeHWThreads)
    return;
}

void
hwloc_init_nodeTopology(cpu_set_t cpuSet)
{
    HWThread* hwThreadPool = NULL;
    int maxNumLogicalProcs = 0;
    int maxNumLogicalProcsPerCore = 0;
    int maxNumCores = 0;
    int maxNumSockets = 0;
    int maxNumCoresPerSocket = 0;
    hwloc_obj_t obj = NULL;
    int poolsize = 0;
    int nr_sockets = 1;
    int id = 0;
    int consecutive_cores = -1;
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

    maxNumLogicalProcs = likwid_hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_PU);
    maxNumCores = likwid_hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_CORE);
    if (likwid_hwloc_get_nbobjs_by_type(hwloc_topology, socket_type) == 0)
    {
        socket_type = HWLOC_OBJ_NODE;
    }
    maxNumSockets = likwid_hwloc_get_nbobjs_by_type(hwloc_topology, socket_type);
    obj = likwid_hwloc_get_obj_by_type(hwloc_topology, socket_type, 0);
    if (obj)
    {
        maxNumCoresPerSocket = likwid_hwloc_record_objs_of_type_below_obj(hwloc_topology, obj, HWLOC_OBJ_CORE, NULL, NULL);
    }
    obj = likwid_hwloc_get_obj_by_type(hwloc_topology, HWLOC_OBJ_CORE, 0);
    if (obj)
    {
        maxNumLogicalProcsPerCore = likwid_hwloc_record_objs_of_type_below_obj(hwloc_topology, obj, HWLOC_OBJ_PU, NULL, NULL);
    }
    else
    {
        maxNumLogicalProcsPerCore = maxNumLogicalProcs/maxNumCores;
    }
    for (uint32_t i=0; i< cpuid_topology.numHWThreads; i++)
    {
        int skip = 0;
        obj = likwid_hwloc_get_obj_by_type(hwloc_topology, HWLOC_OBJ_PU, i);
        if (!obj)
        {
            continue;
        }
        id = obj->os_index;
        if (CPU_ISSET(id, &cpuSet))
            hwThreadPool[id].inCpuSet = 1;
        hwThreadPool[id].apicId = obj->os_index;
        hwThreadPool[id].threadId = obj->sibling_rank;
        if (maxNumLogicalProcsPerCore > 1)
        {
            while (obj->type != HWLOC_OBJ_CORE) {
                obj = obj->parent;
                if (!obj)
                {
                    skip = 1;
                    break;
                }
            }
            if (skip)
            {
                hwThreadPool[id].coreId = 0;
                hwThreadPool[id].packageId = 0;
                continue;
            }
        }
        if (skip)
        {
            hwThreadPool[id].coreId = 0;
            hwThreadPool[id].packageId = 0;
            continue;
        }
#ifdef __ARM_ARCH_8A
        hwThreadPool[id].coreId = hwThreadPool[id].apicId;
#else
        hwThreadPool[id].coreId = obj->logical_index;
#endif
#if defined(__x86_64) || defined(__i386__)
        if (maxNumLogicalProcsPerCore == 1 && cpuid_info.isIntel == 0)
        {
            if (id == 0)
            {
                hwThreadPool[id].coreId = hwThreadPool[id].apicId;
            }
            else
            {
                if (hwThreadPool[id].apicId == hwThreadPool[id-1].apicId + 1 &&
                    hwThreadPool[id].packageId == hwThreadPool[id-1].packageId)
                {
                    hwThreadPool[id].coreId = hwThreadPool[id].apicId % maxNumCoresPerSocket;
                }
                else
                {
                    hwThreadPool[id].coreId = hwThreadPool[id].apicId;
                }
            }
        }
#endif
        while (obj->type != socket_type) {
            obj = obj->parent;
            if (!obj)
            {
                skip = 1;
                break;
            }
        }
        if (skip)
        {
            hwThreadPool[id].packageId = 0;
            continue;
        }
#ifdef __ARM_ARCH_8A
        if (obj->type == HWLOC_OBJ_SOCKET)
        {
            hwThreadPool[id].packageId = obj->os_index;
        }
        else
        {
            hwThreadPool[id].packageId = 0;
        }
#else
        hwThreadPool[id].packageId = obj->os_index;
#endif
        DEBUG_PRINT(DEBUGLEV_DEVELOP, HWLOC Thread Pool PU %d Thread %d Core %d Socket %d inCpuSet %d,
                            hwThreadPool[id].apicId,
                            hwThreadPool[id].threadId,
                            hwThreadPool[id].coreId,
                            hwThreadPool[id].packageId,
                            hwThreadPool[id].inCpuSet)
    }

    int socket_nums[16];
    int num_sockets = 0;
    for (uint32_t i=0; i< cpuid_topology.numHWThreads; i++)
    {
        int found = 0;
        for (uint32_t j=0; j < num_sockets; j++)
        {
            if (hwThreadPool[i].packageId == socket_nums[j])
            {
                found = 1;
                break;
            }
        }
        if (!found)
        {
            socket_nums[num_sockets] = hwThreadPool[i].packageId;
            num_sockets++;
        }
    }
    for (uint32_t i=0; i< cpuid_topology.numHWThreads; i++)
    {
        for (uint32_t j=0; j < num_sockets; j++)
        {
            if (hwThreadPool[i].packageId == socket_nums[j])
            {
                hwThreadPool[i].packageId = j;
            }
        }
    }
    cpuid_topology.threadPool = hwThreadPool;
    cpuid_topology.numThreadsPerCore = maxNumLogicalProcsPerCore;
    cpuid_topology.numCoresPerSocket = maxNumCoresPerSocket;
    cpuid_topology.numSockets = maxNumSockets;
    return;
}

void hwloc_split_llc_check(CacheLevel* llc_cache)
{
    hwloc_obj_t obj = NULL;
    int num_sockets = likwid_hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_SOCKET);
    int num_nodes = likwid_hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_NODE);
    if (num_sockets == num_nodes)
    {
        return;
    }
    obj = likwid_hwloc_get_obj_by_type(hwloc_topology, HWLOC_OBJ_SOCKET, 0);
    int num_threads_per_socket = likwid_hwloc_record_objs_of_type_below_obj(hwloc_topology, obj, HWLOC_OBJ_PU, NULL, NULL);
    obj = likwid_hwloc_get_obj_by_type(hwloc_topology, HWLOC_OBJ_NODE, 0);
    int num_threads_per_node = likwid_hwloc_record_objs_of_type_below_obj(hwloc_topology, obj, HWLOC_OBJ_PU, NULL, NULL);
    if (num_threads_per_node < num_threads_per_socket)
    {
        llc_cache->threads = num_threads_per_node;
        uint32_t size = llc_cache->size;
        double factor = (((double)num_threads_per_node)/((double)num_threads_per_socket));
        llc_cache->size = (uint32_t)(size*factor);
    }
    return;
}

void
hwloc_init_cacheTopology(void)
{
    int maxNumLevels=0;
    int id=0;
    CacheLevel* cachePool = NULL;
    hwloc_obj_t obj;
    int depth;
    int d;
    const char* info;

    /* Sum up all depths with caches */
    depth = likwid_hwloc_topology_get_depth(hwloc_topology);
    for (d = 0; d < depth; d++)
    {
        if (likwid_hwloc_get_depth_type(hwloc_topology, d) == HWLOC_OBJ_CACHE)
            maxNumLevels++;
    }
    cachePool = (CacheLevel*) malloc(maxNumLevels * sizeof(CacheLevel));
    /* Start at the bottom of the tree to get all cache levels in order */
    depth = likwid_hwloc_topology_get_depth(hwloc_topology);
    id = 0;
    for(d=depth-1;d >= 0; d--)
    {
        /* We only need caches, so skip other levels */
        if (likwid_hwloc_get_depth_type(hwloc_topology, d) != HWLOC_OBJ_CACHE)
        {
            continue;
        }
        cachePool[id].level = 0;
        cachePool[id].type = NOCACHE;
        cachePool[id].associativity = 0;
        cachePool[id].lineSize = 0;
        cachePool[id].size = 0;
        cachePool[id].sets = 0;
        cachePool[id].inclusive = 0;
        cachePool[id].threads = 0;
        /* Get the cache object */
        obj = likwid_hwloc_get_obj_by_depth(hwloc_topology, d, 0);
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
        cachePool[id].threads = likwid_hwloc_record_objs_of_type_below_obj(
                        hwloc_topology, obj, HWLOC_OBJ_PU, NULL, NULL);
#if defined(__x86_64) || defined(__i386__)
        if (obj->infos_count > 0)
        {
            while (!(info = likwid_hwloc_obj_get_info_by_name(obj, "inclusiveness")) && obj->next_cousin)
            {
                // If some PU/core are not bindable because of cgroup, hwloc may
                // not know the inclusiveness of some of their cache.
                obj = obj->next_cousin;
            }
            if(info)
            {
                cachePool[id].inclusive = info[0]=='t';
            }
            else if (cpuid_info.isIntel)
            {
                cachePool[id].inclusive = readCacheInclusiveIntel(obj->attr->cache.depth);
            }
        }
        else
        {
            // If a CPU is already pinned, it might not get the cache infos,
            // so we get the inclusiveness through CPUID
            if (cpuid_info.isIntel)
                cachePool[id].inclusive = readCacheInclusiveIntel(obj->attr->cache.depth);
            else
                cachePool[id].inclusive = readCacheInclusiveAMD(obj->attr->cache.depth);
        }
#endif
#if defined(_ARCH_PPC) || defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_8A)
        cachePool[id].inclusive = 0;
#endif
        DEBUG_PRINT(DEBUGLEV_DEVELOP, HWLOC Cache Pool ID %d Level %d Size %d,
                                      id, cachePool[id].level,
                                      cachePool[id].size);
        id++;
    }

    if (cpuid_info.family == P6_FAMILY)
    {
        switch (cpuid_info.model)
        {
            case HASWELL_EP:
            case BROADWELL_E:
            case BROADWELL_D:
            case SKYLAKEX:
                hwloc_split_llc_check(&cachePool[maxNumLevels-1]);
                break;
            default:
                break;
        }
    }

    cpuid_topology.numCacheLevels = maxNumLevels;
    cpuid_topology.cacheLevels = cachePool;
    return;
}

void
hwloc_close(void)
{
    if (hwloc_topology)
    {
        likwid_hwloc_topology_destroy(hwloc_topology);
        hwloc_topology = NULL;
    }
}
#else
void
hwloc_init_cpuInfo(void)
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
