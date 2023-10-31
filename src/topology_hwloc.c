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
 *      Authors:  Thomas Gruber (tr), thomas.roehl@googlemail.com
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
int parse_cpuinfo(uint32_t* count, uint32_t* family, uint32_t* variant, uint32_t *stepping, uint32_t *part, uint32_t *vendor)
{
    int i = 0;
    FILE *fp = NULL;
    uint32_t f = 0;
    uint32_t v = 0;
    uint32_t s = 0;
    uint32_t p = 0;
    uint32_t vend = 0;
    uint32_t c = 0;
    int (*ownatoi)(const char*);
    ownatoi = &atoi;
    struct tagbstring familyString = bsStatic("CPU architecture");
    struct tagbstring variantString = bsStatic("CPU variant");
    struct tagbstring steppingString = bsStatic("CPU revision");
    struct tagbstring partString = bsStatic("CPU part");
    struct tagbstring vendString = bsStatic("CPU implementer");
    struct tagbstring procString = bsStatic("processor");

    if (NULL != (fp = fopen ("/proc/cpuinfo", "r")))
    {
        bstring src = bread ((bNread) fread, fp);
        struct bstrList* tokens = bsplit(src,(char) '\n');
        bdestroy(src);
        fclose(fp);
        for (i=0;i<tokens->qty;i++)
        {
            if ((binstr(tokens->entry[i],0,&procString) != BSTR_ERR))
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                uint32_t tmp = ownatoi(bdata(subtokens->entry[1]));
                if (tmp > c)
                {
                    c = tmp;
                }
                bstrListDestroy(subtokens);
            }
            else if ((f == 0) && (binstr(tokens->entry[i],0,&familyString) != BSTR_ERR))
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                f = ownatoi(bdata(subtokens->entry[1]));
                bstrListDestroy(subtokens);
            }
            else if ((s == 0) && (binstr(tokens->entry[i],0,&steppingString) != BSTR_ERR))
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                s = ownatoi(bdata(subtokens->entry[1]));
                bstrListDestroy(subtokens);
            }
            else if ((v == 0) && (binstr(tokens->entry[i],0,&variantString) != BSTR_ERR))
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                v = strtol(bdata(subtokens->entry[1]), NULL, 0);
                bstrListDestroy(subtokens);
            }
            else if ((p == 0) && (binstr(tokens->entry[i],0,&partString) != BSTR_ERR))
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                p = strtol(bdata(subtokens->entry[1]), NULL, 0);
                bstrListDestroy(subtokens);
            }
            else if ((vend == 0) && (binstr(tokens->entry[i],0,&vendString) != BSTR_ERR))
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                vend = strtol(bdata(subtokens->entry[1]), NULL, 0);
                bstrListDestroy(subtokens);
            }
        }
        bstrListDestroy(tokens);
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
    *count = c + 1;
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
#if !defined(__ARM_ARCH_7A__) && !defined(__ARM_ARCH_8A) && !defined(_ARCH_PPC)
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
    uint32_t count = 0;
    hwloc_obj_t obj;
    if (perfmon_verbosity <= 1)
    {
        setenv("HWLOC_HIDE_ERRORS", "1", 1);
    }
    if (!hwloc_topology)
    {
        //HWLOC_PREFIX#hwloc_topology_init(&hwloc_topology);
        LIKWID_HWLOC_NAME(topology_init)(&hwloc_topology);
#if HWLOC_API_VERSION > 0x00020000
        LIKWID_HWLOC_NAME(topology_set_flags)(hwloc_topology, HWLOC_TOPOLOGY_FLAG_INCLUDE_DISALLOWED );
        LIKWID_HWLOC_NAME(topology_set_type_filter)(hwloc_topology, HWLOC_OBJ_PCI_DEVICE, HWLOC_TYPE_FILTER_KEEP_ALL);
#else
        LIKWID_HWLOC_NAME(topology_set_flags)(hwloc_topology, HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM|HWLOC_TOPOLOGY_FLAG_WHOLE_IO );
#endif
        LIKWID_HWLOC_NAME(topology_load)(hwloc_topology);
    }
    obj = LIKWID_HWLOC_NAME(get_obj_by_type)(hwloc_topology, HWLOC_OBJ_SOCKET, 0);

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
    if ((info = LIKWID_HWLOC_NAME(obj_get_info_by_name)(obj, "CPUModelNumber")))
        cpuid_info.model = atoi(info);
    if ((info = LIKWID_HWLOC_NAME(obj_get_info_by_name)(obj, "CPUFamilyNumber")))
       cpuid_info.family = atoi(info);
    if ((info = LIKWID_HWLOC_NAME(obj_get_info_by_name)(obj, "CPUVendor")))
        cpuid_info.isIntel = strcmp(info, "GenuineIntel") == 0;
    if ((info = LIKWID_HWLOC_NAME(obj_get_info_by_name)(obj, "CPUStepping")))
        cpuid_info.stepping = atoi(info);
    snprintf(cpuid_info.architecture, 19, "x86_64");
#endif
#if defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_8A)
    parse_cpuinfo(&count, &cpuid_info.family, &cpuid_info.model, &cpuid_info.stepping, &cpuid_info.part, &cpuid_info.vendor);
    parse_cpuname(cpuid_info.osname);
#ifdef __ARM_ARCH_7A__
    snprintf(cpuid_info.architecture, 19, "armv7");
#endif
#ifdef __ARM_ARCH_8A
    snprintf(cpuid_info.architecture, 19, "armv8");
#endif
#endif

#ifndef _ARCH_PPC
    if ((info = LIKWID_HWLOC_NAME(obj_get_info_by_name)(obj, "CPUModel")))
        strcpy(cpuid_info.osname, info);
#else
    if ((info = LIKWID_HWLOC_NAME(obj_get_info_by_name)(obj, "CPUModel")))
    {
        if (strstr(info, "POWER7") != NULL)
        {
            cpuid_info.model = POWER7;
            cpuid_info.family = PPC_FAMILY;
            cpuid_info.isIntel = 0;
            strcpy(cpuid_info.osname, info);
            cpuid_info.stepping = 0;
            snprintf(cpuid_info.architecture, 19, "power7");
        }
        if (strstr(info, "POWER8") != NULL)
        {
            cpuid_info.model = POWER8;
            cpuid_info.family = PPC_FAMILY;
            cpuid_info.isIntel = 0;
            strcpy(cpuid_info.osname, info);
            cpuid_info.stepping = 0;
            snprintf(cpuid_info.architecture, 19, "power8");
        }
        if (strstr(info, "POWER9") != NULL)
        {
            cpuid_info.model = POWER9;
            cpuid_info.family = PPC_FAMILY;
            cpuid_info.isIntel = 0;
            strcpy(cpuid_info.osname, info);
            cpuid_info.stepping = 0;
            snprintf(cpuid_info.architecture, 19, "power9");
        }

    }
#endif


    cpuid_topology.numHWThreads = LIKWID_HWLOC_NAME(get_nbobjs_by_type)(hwloc_topology, HWLOC_OBJ_PU);
#if defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_8A)
    if (count > cpuid_topology.numHWThreads)
    {
        cpuid_topology.numHWThreads = count;
    }
#endif
    if (!getenv("HWLOC_FSROOT"))
    {
        count = likwid_sysfs_list_len("/sys/devices/system/cpu/online");
        if (count > cpuid_topology.numHWThreads)
        {
            cpuid_topology.numHWThreads = count;
        }
        if (cpuid_topology.activeHWThreads > cpuid_topology.numHWThreads)
        {
            cpuid_topology.numHWThreads = cpuid_topology.activeHWThreads;
        }
    }
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
    int maxNumDies = 0;
    int maxNumCoresPerSocket = 0;
    hwloc_obj_t obj = NULL;
    int poolsize = 0;
    int nr_sockets = 1;
    int id = 0;
    int consecutive_cores = -1;
    int from_file = (getenv("HWLOC_FSROOT") != NULL);
    hwloc_obj_type_t socket_type = HWLOC_OBJ_SOCKET;
    if (!from_file)
    {
        for (uint32_t i=0;i<cpuid_topology.numHWThreads;i++)
        {
            if (CPU_ISSET(i, &cpuSet))
            {
                poolsize = i+1;
            }
        }
    }
    else
    {
        poolsize = cpuid_topology.numHWThreads;
    }
    hwThreadPool = (HWThread*) malloc(cpuid_topology.numHWThreads * sizeof(HWThread));
    for (uint32_t i=0;i<cpuid_topology.numHWThreads;i++)
    {
        hwThreadPool[i].apicId = -1;
        hwThreadPool[i].threadId = -1;
        hwThreadPool[i].coreId = -1;
        hwThreadPool[i].packageId = -1;
        hwThreadPool[i].dieId = -1;
        hwThreadPool[i].inCpuSet = 0;
    }

    maxNumLogicalProcs = LIKWID_HWLOC_NAME(get_nbobjs_by_type)(hwloc_topology, HWLOC_OBJ_PU);
    maxNumCores = LIKWID_HWLOC_NAME(get_nbobjs_by_type)(hwloc_topology, HWLOC_OBJ_CORE);
    if (LIKWID_HWLOC_NAME(get_nbobjs_by_type)(hwloc_topology, socket_type) == 0)
    {
        socket_type = HWLOC_OBJ_NODE;
    }

    maxNumSockets = LIKWID_HWLOC_NAME(get_nbobjs_by_type)(hwloc_topology, socket_type);
    maxNumDies = LIKWID_HWLOC_NAME(get_nbobjs_by_type)(hwloc_topology, HWLOC_OBJ_DIE);
    obj = LIKWID_HWLOC_NAME(get_obj_by_type)(hwloc_topology, socket_type, 0);

    if (obj)
    {
        maxNumCoresPerSocket = likwid_hwloc_record_objs_of_type_below_obj(hwloc_topology, obj, HWLOC_OBJ_CORE, NULL, NULL);
    }
    obj = LIKWID_HWLOC_NAME(get_obj_by_type)(hwloc_topology, HWLOC_OBJ_CORE, 0);
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
        obj = LIKWID_HWLOC_NAME(get_obj_by_type)(hwloc_topology, HWLOC_OBJ_PU, i);
        if (!obj)
        {
            continue;
        }
        id = obj->os_index;

        if (id < 0 || id >= cpuid_topology.numHWThreads)
            continue;

        if (CPU_ISSET(id, &cpuSet))
        {
            hwThreadPool[id].inCpuSet = 1;
        }
        else if (from_file)
        {
            hwThreadPool[id].inCpuSet = 1;
        }

        if (!likwid_cpu_online(obj->os_index))
        {
            hwThreadPool[id].inCpuSet = 0;
        }

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
        if (maxNumSockets < maxNumDies)
        {
            while (obj->type != HWLOC_OBJ_DIE) {
                obj = obj->parent;
                if (!obj)
                {
                    skip = 1;
                    break;
                }
            }
            if (skip)
            {
                hwThreadPool[id].dieId = 0;
                continue;
            }
            if (obj->type == HWLOC_OBJ_DIE)
            {
                hwThreadPool[id].dieId = obj->os_index;
            }
            else
            {
                hwThreadPool[id].dieId = 0;
            }
        }
        else
        {
            hwThreadPool[id].dieId = 0;
        }
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
        DEBUG_PRINT(DEBUGLEV_DEVELOP, HWLOC Thread Pool PU %d Thread %d Core %d Die %d Socket %d inCpuSet %d,
                            hwThreadPool[id].apicId,
                            hwThreadPool[id].threadId,
                            hwThreadPool[id].coreId,
                            hwThreadPool[id].dieId,
                            hwThreadPool[id].packageId,
                            hwThreadPool[id].inCpuSet)
    }

    int socket_nums[MAX_NUM_NODES];
    int num_sockets = 0;
    int has_zero_id = 0;
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
        if (hwThreadPool[i].packageId == 0)
        {
            has_zero_id = 1;
        }
    }
    if (!has_zero_id)
    {
        for (uint32_t i=0; i< cpuid_topology.numHWThreads; i++)
        {
            HWThread* t = &hwThreadPool[i];
            for (uint32_t j=0; j < num_sockets; j++)
            {
                if (hwThreadPool[i].packageId == socket_nums[j])
                {
                    hwThreadPool[i].packageId = j;
                }
            }
        }
    }
    cpuid_topology.threadPool = hwThreadPool;
    cpuid_topology.numThreadsPerCore = maxNumLogicalProcsPerCore;
    cpuid_topology.numCoresPerSocket = maxNumCoresPerSocket;
    cpuid_topology.numSockets = maxNumSockets;
    cpuid_topology.numDies = maxNumDies;
    return;
}

void hwloc_split_llc_check(CacheLevel* llc_cache)
{
    int i = 0;
    hwloc_obj_t obj = NULL;
    int num_sockets = LIKWID_HWLOC_NAME(get_nbobjs_by_type)(hwloc_topology, HWLOC_OBJ_SOCKET);
    int num_nodes = LIKWID_HWLOC_NAME(get_nbobjs_by_type)(hwloc_topology, HWLOC_OBJ_NODE);
    if (num_sockets == num_nodes)
    {
        return;
    }
    obj = LIKWID_HWLOC_NAME(get_obj_by_type)(hwloc_topology, HWLOC_OBJ_SOCKET, 0);
    int num_threads_per_socket = likwid_hwloc_record_objs_of_type_below_obj(hwloc_topology, obj, HWLOC_OBJ_PU, NULL, NULL);
    if (num_threads_per_socket == 0)
    {
        for (i = 0; i < LIKWID_HWLOC_NAME(get_nbobjs_by_type)(hwloc_topology, HWLOC_OBJ_PU); i++)
        {
            if (LIKWID_HWLOC_NAME(bitmap_isset)(obj->cpuset, i))
                num_threads_per_socket++;
        }
    }
    obj = LIKWID_HWLOC_NAME(get_obj_by_type)(hwloc_topology, HWLOC_OBJ_NODE, 0);
    int num_threads_per_node = likwid_hwloc_record_objs_of_type_below_obj(hwloc_topology, obj, HWLOC_OBJ_PU, NULL, NULL);
    if (num_threads_per_node == 0)
    {
        for (i = 0; i < LIKWID_HWLOC_NAME(get_nbobjs_by_type)(hwloc_topology, HWLOC_OBJ_PU); i++)
        {
            if (LIKWID_HWLOC_NAME(bitmap_isset)(obj->cpuset, i))
                num_threads_per_node++;
        }
    }
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
    depth = LIKWID_HWLOC_NAME(topology_get_depth)(hwloc_topology);
    for (d = 0; d < depth; d++)
    {
#if HWLOC_API_VERSION > 0x00020000
        hwloc_obj_type_t depth_type = LIKWID_HWLOC_NAME(get_depth_type)(hwloc_topology, d);
        if (depth_type == HWLOC_OBJ_L1CACHE ||
            depth_type == HWLOC_OBJ_L2CACHE ||
            depth_type == HWLOC_OBJ_L3CACHE ||
            depth_type == HWLOC_OBJ_L4CACHE ||
            depth_type == HWLOC_OBJ_L5CACHE)
            maxNumLevels++;
#else
        if (LIKWID_HWLOC_NAME(get_depth_type)(hwloc_topology, d) == HWLOC_OBJ_CACHE)
            maxNumLevels++;
#endif
    }
    cachePool = (CacheLevel*) malloc(maxNumLevels * sizeof(CacheLevel));
    if (!cachePool)
    {
        cpuid_topology.numCacheLevels = 0;
        cpuid_topology.cacheLevels = NULL;
        return;
    }
    /* Start at the bottom of the tree to get all cache levels in order */
    depth = LIKWID_HWLOC_NAME(topology_get_depth)(hwloc_topology);
    id = 0;
    for(d=depth-1;d >= 0; d--)
    {
        /* We only need caches, so skip other levels */
#if HWLOC_API_VERSION > 0x00020000
        hwloc_obj_type_t depth_type = LIKWID_HWLOC_NAME(get_depth_type)(hwloc_topology, d);
        if (depth_type != HWLOC_OBJ_L1CACHE &&
            depth_type != HWLOC_OBJ_L2CACHE &&
            depth_type != HWLOC_OBJ_L3CACHE &&
            depth_type != HWLOC_OBJ_L4CACHE &&
            depth_type != HWLOC_OBJ_L5CACHE)
#else
        if (LIKWID_HWLOC_NAME(get_depth_type)(hwloc_topology, d) < HWLOC_OBJ_CACHE)
#endif
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
        obj = LIKWID_HWLOC_NAME(get_obj_by_depth)(hwloc_topology, d, 0);
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
#ifdef _ARCH_PPC
        if ((cpuid_info.family == PPC_FAMILY) && ((cpuid_info.model == POWER8) || (cpuid_info.model == POWER9)))
        {
            if (cachePool[id].lineSize == 0)
                cachePool[id].lineSize = 128;
            if (cachePool[id].associativity == 0)
                cachePool[id].associativity = 8;
        }
#endif
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
            while (!(info = LIKWID_HWLOC_NAME(obj_get_info_by_name)(obj, "inclusiveness")) && obj->next_cousin)
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
        DEBUG_PRINT(DEBUGLEV_DEVELOP, HWLOC Cache Pool ID %d Level %d Size %d Threads %d,
                                      id, cachePool[id].level,
                                      cachePool[id].size,
                                      cachePool[id].threads);
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
        LIKWID_HWLOC_NAME(topology_destroy)(hwloc_topology);
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
