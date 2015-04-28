/*
 * =======================================================================================
 *
 *      Filename:  topology.c
 *
 *      Description:  Interface to the topology backends
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Authors:  Jan Treibig (jt), jan.treibig@gmail.com,
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 Jan Treibig, Thomas Roehl
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
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>

#include <likwid.h>

#include <topology.h>
#include <error.h>
#include <tree.h>
#include <bitUtil.h>
//#include <strUtil.h>
#include <configuration.h>


static volatile int init = 0;
CpuInfo cpuid_info;
CpuTopology cpuid_topology;

int affinity_thread2tile_lookup[MAX_NUM_THREADS];

static char* pentium_m_b_str = "Intel Pentium M Banias processor";
static char* pentium_m_d_str = "Intel Pentium M Dothan processor";
static char* core_duo_str = "Intel Core Duo processor";
static char* core_2a_str = "Intel Core 2 65nm processor";
static char* core_2b_str = "Intel Core 2 45nm processor";
static char* atom_45_str = "Intel Atom 45nm processor";
static char* atom_32_str = "Intel Atom 32nm processor";
static char* atom_22_str = "Intel Atom 22nm processor";
static char* atom_silvermont_str = "Intel Atom (Silvermont) processor";
static char* atom_airmont_str = "Intel Atom (Airmont) processor";
static char* nehalem_bloom_str = "Intel Core Bloomfield processor";
static char* nehalem_lynn_str = "Intel Core Lynnfield processor";
static char* nehalem_west_str = "Intel Core Westmere processor";
static char* sandybridge_str = "Intel Core SandyBridge processor";
static char* ivybridge_str = "Intel Core IvyBridge processor";
static char* ivybridge_ep_str = "Intel Xeon IvyBridge EN/EP/EX processor";
static char* sandybridge_ep_str = "Intel Xeon SandyBridge EN/EP processor";
static char* haswell_str = "Intel Core Haswell processor";
static char* haswell_ep_str = "Intel Xeon Haswell EN/EP/EX processor";
static char* broadwell_str = "Intel Core Broadwell processor";
static char* broadwell_d_str = "Intel Xeon D Broadwell processor";
static char* broadwell_ep_str = "Intel Xeon Broadwell EN/EP/EX processor";
static char* nehalem_ex_str = "Intel Nehalem EX processor";
static char* westmere_ex_str = "Intel Westmere EX processor";
static char* xeon_mp_string = "Intel Xeon MP processor";
static char* xeon_phi_string = "Intel Xeon Phi Coprocessor";
static char* barcelona_str = "AMD Barcelona processor";
static char* shanghai_str = "AMD Shanghai processor";
static char* istanbul_str = "AMD Istanbul processor";
static char* magnycours_str = "AMD Magny Cours processor";
static char* interlagos_str = "AMD Interlagos processor";
static char* kabini_str = "AMD Family 16 model - Kabini processor";
static char* opteron_sc_str = "AMD Opteron single core 130nm processor";
static char* opteron_dc_e_str = "AMD Opteron Dual Core Rev E 90nm processor";
static char* opteron_dc_f_str = "AMD Opteron Dual Core Rev F 90nm processor";
static char* athlon64_str = "AMD Athlon64 X2 (AM2) Rev F 90nm processor";
static char* athlon64_f_str = "AMD Athlon64 (AM2) Rev F 90nm processor";
static char* athlon64_X2_g_str = "AMD Athlon64 X2 (AM2) Rev G 65nm processor";
static char* athlon64_g_str = "AMD Athlon64 (AM2) Rev G 65nm processor";
static char* amd_k8_str = "AMD K8 architecture";
static char* unknown_intel_str = "Unknown Intel Processor";
static char* unknown_amd_str = "Unknown AMD Processor";

static char* short_core2 = "core2";
static char* short_atom = "atom";
static char* short_pm = "pentiumm";
static char* short_silvermont = "silvermont";
static char* short_nehalem = "nehalem";
static char* short_nehalemEX = "nehalemEX";
static char* short_westmere = "westmere";
static char* short_westmereEX = "westmereEX";
static char* short_haswell = "haswell";
static char* short_haswell_ep = "haswellEP";
static char* short_broadwell = "broadwell";
static char* short_broadwell_ep = "broadwellEP";
static char* short_ivybridge = "ivybridge";
static char* short_sandybridge = "sandybridge";
static char* short_phi = "phi";
static char* short_k8 = "k8";
static char* short_k10 = "k10";
static char* short_k15 = "interlagos";
static char* short_k16 = "kabini";
static char* short_unknown = "unknown";


int cpu_count(cpu_set_t* set)
{
    uint32_t i;
    int s = 0;
    const __cpu_mask *p = set->__bits;
    const __cpu_mask *end = &set->__bits[sizeof(cpu_set_t) / sizeof (__cpu_mask)];

    while (p < end)
    {
        __cpu_mask l = *p++;

        if (l == 0)
        {
            continue;
        }

        for (i=0; i< (sizeof(__cpu_mask)*8); i++)
        {
            if (l&(1UL<<i))
            {
                s++;
            }
        }
    }

    return s;
}

static void initTopologyFile(FILE* file)
{
    size_t items;
    HWThread* hwThreadPool;
    CacheLevel* cacheLevels;
    TreeNode* currentNode;

    items = fread((void*) &cpuid_topology, sizeof(CpuTopology), 1, file);

    hwThreadPool = (HWThread*) malloc(cpuid_topology.numHWThreads * sizeof(HWThread));
    items = fread((void*) hwThreadPool, sizeof(HWThread), cpuid_topology.numHWThreads, file);
    cpuid_topology.threadPool = hwThreadPool;

    cacheLevels = (CacheLevel*) malloc(cpuid_topology.numCacheLevels * sizeof(CacheLevel));
    items = fread((void*) cacheLevels, sizeof(CacheLevel), cpuid_topology.numCacheLevels, file);
    cpuid_topology.cacheLevels = cacheLevels;
    cpuid_topology.topologyTree = NULL;

    tree_init(&cpuid_topology.topologyTree, 0);

    for (uint32_t i=0; i<  cpuid_topology.numHWThreads; i++)
    {
        if (!tree_nodeExists(cpuid_topology.topologyTree,
                    hwThreadPool[i].packageId))
        {
            tree_insertNode(cpuid_topology.topologyTree,
                    hwThreadPool[i].packageId);
        }
        currentNode = tree_getNode(cpuid_topology.topologyTree,
                hwThreadPool[i].packageId);

        if (!tree_nodeExists(currentNode, hwThreadPool[i].coreId))
        {
            tree_insertNode(currentNode, hwThreadPool[i].coreId);
        }
        currentNode = tree_getNode(currentNode, hwThreadPool[i].coreId);

        if (!tree_nodeExists(currentNode, i))
        {
            tree_insertNode(currentNode, i);
            affinity_thread2tile_lookup[hwThreadPool[i].apicId] = hwThreadPool[i].coreId;
        }
    }
}


static int readTopologyFile(const char* filename)
{
    FILE* fp;
    char structure[256];
    char field[256];
    char value[256];
    char line[512];
    int numHWThreads = -1;
    int numCacheLevels = -1;
    int numberOfNodes = -1;
    int* tmpNumberOfProcessors;
    int counter;
    int i;
    uint32_t tmp, tmp1;

    fp = fopen(filename, "r");

    while (fgets(line, 512, fp) != NULL) {
        sscanf(line,"%s %s", structure, field);
        if ((strcmp(structure, "cpuid_topology") == 0) && (strcmp(field, "numHWThreads") == 0))
        {
            sscanf(line,"%s %s = %d", structure, field, &numHWThreads);
        }
        else if ((strcmp(structure, "cpuid_topology") == 0) && (strcmp(field, "numCacheLevels") == 0))
        {
            sscanf(line,"%s %s = %d", structure, field, &numCacheLevels);
        }
        else if ((strcmp(structure, "numa_info") == 0) && (strcmp(field, "numberOfNodes") == 0))
        {
            sscanf(line,"%s %s = %d", structure, field, &numberOfNodes);
        }
        if ((numHWThreads >= 0) && (numCacheLevels >= 0) && (numberOfNodes >= 0))
        {
            break;
        }
    }

    tmpNumberOfProcessors = (int*) malloc(numberOfNodes *sizeof(int));
    fseek(fp, 0, SEEK_SET);
    counter = 0;
    while (fgets(line, 512, fp) != NULL) {
        sscanf(line,"%s %s %d %s = %d", structure, field, &tmp, value, &tmp1);
        if ((strcmp(structure, "numa_info") == 0) && (strcmp(value, "numberOfProcessors") == 0))
        {
            tmpNumberOfProcessors[tmp-1] = tmp1;
            counter++;
        }
        if (counter == numberOfNodes)
        {
            break;
        }
    }

    cpuid_topology.threadPool = (HWThread*)malloc(numHWThreads * sizeof(HWThread));
    cpuid_topology.cacheLevels = (CacheLevel*)malloc(numCacheLevels * sizeof(CacheLevel));
    cpuid_topology.numHWThreads = numHWThreads;
    cpuid_topology.numCacheLevels = numCacheLevels;

    numa_info.nodes = (NumaNode*) malloc(numberOfNodes * sizeof(NumaNode));
    numa_info.numberOfNodes = numberOfNodes;

    for(i=0;i<numberOfNodes;i++)
    {
        numa_info.nodes[i].processors = (uint32_t*) malloc (tmpNumberOfProcessors[i] * sizeof(int));
        numa_info.nodes[i].distances = (uint32_t*) malloc (numberOfNodes * sizeof(int));
    }
    free(tmpNumberOfProcessors);

    fseek(fp, 0, SEEK_SET);

    while (fgets(line, 512, fp) != NULL) {
        sscanf(line,"%s %s", structure, field);
        if (strcmp(structure, "cpuid_topology") == 0)
        {
            if (strcmp(field, "numSockets") == 0)
            {
                sscanf(line, "%s %s = %d", structure, field, &tmp);
                cpuid_topology.numSockets = tmp;
            }
            else if (strcmp(field, "numCoresPerSocket") == 0)
            {
                sscanf(line, "%s %s = %d", structure, field, &tmp);
                cpuid_topology.numCoresPerSocket = tmp;
            }
            else if (strcmp(field, "numThreadsPerCore") == 0)
            {
                sscanf(line, "%s %s = %d", structure, field, &tmp);
                cpuid_topology.numThreadsPerCore = tmp;
            }
            else if (strcmp(field, "threadPool") == 0)
            {
                int thread;

                sscanf(line, "%s %s %d %s = %d", structure, field, &thread, value, &tmp);

                if (strcmp(value, "threadId") == 0)
                {
                    cpuid_topology.threadPool[thread-1].threadId = tmp;
                }
                else if (strcmp(value, "coreId") == 0)
                {
                    cpuid_topology.threadPool[thread-1].coreId = tmp;
                }
                else if (strcmp(value, "packageId") == 0)
                {
                    cpuid_topology.threadPool[thread-1].packageId = tmp;
                }
                else if (strcmp(value, "apicId") == 0)
                {
                    cpuid_topology.threadPool[thread-1].apicId = tmp;
                }
                
            }
            else if (strcmp(field, "cacheLevels") == 0)
            {
                int level;
                char type[128];
                sscanf(line, "%s %s %d %s", structure, field, &level, value);
                
                cpuid_topology.cacheLevels[level-1].level = level-1;
                if (strcmp(value, "type") == 0)
                {
                    sscanf(line, "%s %s %d %s = %s", structure, field, &level, value, type);
                    if (strcmp(type, "UNIFIEDCACHE") == 0)
                    {
                        cpuid_topology.cacheLevels[level-1].type = UNIFIEDCACHE;
                    } 
                    else if (strcmp(type, "DATACACHE") == 0)
                    {
                        cpuid_topology.cacheLevels[level-1].type = DATACACHE;
                    } 
                    else if (strcmp(type, "INSTRUCTIONCACHE") == 0)
                    {
                        cpuid_topology.cacheLevels[level-1].type = INSTRUCTIONCACHE;
                    } 
                    else if (strcmp(type, "ITLB") == 0)
                    {
                        cpuid_topology.cacheLevels[level-1].type = ITLB;
                    } 
                    else if (strcmp(type, "DTLB") == 0)
                    {
                        cpuid_topology.cacheLevels[level-1].type = DTLB;
                    }
                    else if (strcmp(type, "NOCACHE") == 0)
                    {
                        cpuid_topology.cacheLevels[level-1].type = NOCACHE;
                    }
                }
                else
                {
                    sscanf(line, "%s %s %d %s = %d", structure, field, &level, value, &tmp);
                    if (strcmp(value, "associativity") == 0)
                    {
                        cpuid_topology.cacheLevels[level-1].associativity = tmp;
                    }
                    else if (strcmp(value, "sets") == 0)
                    {
                        cpuid_topology.cacheLevels[level-1].sets = tmp;
                    }
                    else if (strcmp(value, "lineSize") == 0)
                    {
                        cpuid_topology.cacheLevels[level-1].lineSize = tmp;
                    }
                    else if (strcmp(value, "size") == 0)
                    {
                        cpuid_topology.cacheLevels[level-1].size = tmp;
                    }
                    else if (strcmp(value, "threads") == 0)
                    {
                        cpuid_topology.cacheLevels[level-1].threads = tmp;
                    }
                    else if (strcmp(value, "inclusive") == 0)
                    {
                        cpuid_topology.cacheLevels[level-1].inclusive = tmp;
                    }
                }
                
            }
        }
        else if (strcmp(structure, "cpuid_info") == 0)
        {
            if (strcmp(field, "family") == 0)
            {
                sscanf(line, "%s %s = %d", structure, field, &tmp);
                cpuid_info.family = tmp;
                
            }
            else if (strcmp(field, "model") == 0)
            {
                sscanf(line, "%s %s = %d", structure, field, &tmp);
                cpuid_info.model = tmp;
                
            }
            else if (strcmp(field, "stepping") == 0)
            {
                sscanf(line, "%s %s = %d", structure, field, &tmp);
                cpuid_info.stepping = tmp;
                
            }
            else if (strcmp(field, "clock") == 0)
            {
                sscanf(line, "%s %s = %d", structure, field, &tmp);
                cpuid_info.clock = tmp;
                
            }
            else if (strcmp(field, "turbo") == 0)
            {
                sscanf(line, "%s %s = %d", structure, field, &tmp);
                cpuid_info.turbo = tmp;
                
            }
            else if (strcmp(field, "isIntel") == 0)
            {
                sscanf(line, "%s %s = %d", structure, field, &tmp);
                cpuid_info.isIntel = tmp;
                
            }
            else if (strcmp(field, "featureFlags") == 0)
            {
                sscanf(line, "%s %s = %d", structure, field, &tmp);
                cpuid_info.featureFlags = tmp;
                
            }
            else if (strcmp(field, "perf_version") == 0)
            {
                sscanf(line, "%s %s = %d", structure, field, &tmp);
                cpuid_info.perf_version = tmp;
                
            }
            else if (strcmp(field, "perf_num_ctr") == 0)
            {
                sscanf(line, "%s %s = %d", structure, field, &tmp);
                cpuid_info.perf_num_ctr = tmp;
                
            }
            else if (strcmp(field, "perf_width_ctr") == 0)
            {
                sscanf(line, "%s %s = %d", structure, field, &tmp);
                cpuid_info.perf_width_ctr = tmp;
                
            }
            else if (strcmp(field, "perf_num_fixed_ctr") == 0)
            {
                sscanf(line, "%s %s = %d", structure, field, &tmp);
                cpuid_info.perf_num_fixed_ctr = tmp;
                
            }
            else if (strcmp(field, "features") == 0)
            {
                strcpy(value,&(line[strlen(structure)+strlen(field)+4]));
                cpuid_info.features = (char*) malloc((strlen(value)+1) * sizeof(char));
                strncpy(cpuid_info.features, value, strlen(value));
                cpuid_info.features[strlen(value)-1] = '\0';
            }
        }
        else if (strcmp(structure, "numa_info") == 0)
        {
            if (strcmp(field, "nodes") == 0)
            {
                int id;
                sscanf(line, "%s %s %d %s", structure, field, &id, value);
                    
                if (strcmp(value,"numberOfProcessors") == 0)
                {
                    sscanf(line, "%s %s %d %s = %d", structure, field, &id, value, &tmp);
                    numa_info.nodes[id-1].numberOfProcessors = tmp;
                }
                else if (strcmp(value, "freeMemory") == 0)
                {
                    sscanf(line, "%s %s %d %s = %d", structure, field, &id, value, &tmp);
                    numa_info.nodes[id-1].freeMemory = tmp;
                }
                else if (strcmp(value, "id") == 0)
                {
                    sscanf(line, "%s %s %d %s = %d", structure, field, &id, value, &tmp);
                    numa_info.nodes[id-1].id = tmp;
                }
                else if (strcmp(value, "totalMemory") == 0)
                {
                    sscanf(line, "%s %s %d %s = %d", structure, field, &id, value, &tmp);
                    numa_info.nodes[id-1].totalMemory = tmp;
                }
                else if (strcmp(value, "numberOfDistances") == 0)
                {
                    sscanf(line, "%s %s %d %s = %d", structure, field, &id, value, &tmp);
                    numa_info.nodes[id-1].numberOfDistances = tmp;
                }
                if (strcmp(value, "processors") == 0)
                {
                    sscanf(line, "%s %s %d %s %d = %d", structure, field, &id, value, &tmp, &tmp1);
                    numa_info.nodes[id-1].processors[tmp-1] = tmp1;
                }
                else if (strcmp(value,"distances") == 0)
                {
                    sscanf(line, "%s %s %d %s %d = %d", structure, field, &id, value, &tmp, &tmp1);
                    numa_info.nodes[id-1].distances[tmp] = tmp1;
                }
            }
        }
    }

    fclose(fp);

    return 0;
}

int topology_setName(void)
{
    switch ( cpuid_info.family )
    {
        case P6_FAMILY:
            switch ( cpuid_info.model )
            {
                case PENTIUM_M_BANIAS:
                    cpuid_info.name = pentium_m_b_str;
                    cpuid_info.short_name = short_pm;
                    break;

                case PENTIUM_M_DOTHAN:
                    cpuid_info.name = pentium_m_d_str;
                    cpuid_info.short_name = short_pm;
                    break;

                case CORE_DUO:
                    cpuid_info.name = core_duo_str;
                    cpuid_info.short_name = short_core2;
                    break;

                case CORE2_65:
                    cpuid_info.name = core_2a_str;
                    cpuid_info.short_name = short_core2;
                    break;

                case CORE2_45:
                    cpuid_info.name = core_2b_str;
                    cpuid_info.short_name = short_core2;
                    break;

                case NEHALEM_BLOOMFIELD:
                    cpuid_info.name = nehalem_bloom_str;
                    cpuid_info.short_name = short_nehalem;
                    break;

                case NEHALEM_LYNNFIELD:
                    cpuid_info.name = nehalem_lynn_str;
                    cpuid_info.short_name = short_nehalem;
                    break;

                case NEHALEM_WESTMERE_M:

                case NEHALEM_WESTMERE:
                    cpuid_info.name = nehalem_west_str;
                    cpuid_info.short_name = short_westmere;
                    break;

                case SANDYBRIDGE:
                    cpuid_info.name = sandybridge_str;
                    cpuid_info.short_name = short_sandybridge;
                    break;

                case SANDYBRIDGE_EP:
                    cpuid_info.supportUncore = 1;
                    cpuid_info.name = sandybridge_ep_str;
                    cpuid_info.short_name = short_sandybridge;
                    break;

                case IVYBRIDGE:
                    cpuid_info.name = ivybridge_str;
                    cpuid_info.short_name = short_ivybridge;
                    break;

                case IVYBRIDGE_EP:
                    cpuid_info.supportUncore = 1;
                    cpuid_info.name = ivybridge_ep_str;
                    cpuid_info.short_name = short_ivybridge;
                    break;

                case HASWELL_EP:
                    cpuid_info.supportUncore = 1;
                    cpuid_info.name = haswell_ep_str;
                    cpuid_info.short_name = short_haswell_ep;
                    break;
                case HASWELL:
                case HASWELL_M1:
                case HASWELL_M2:
                    cpuid_info.name = haswell_str;
                    cpuid_info.short_name = short_haswell;
                    break;

                case BROADWELL:
                    cpuid_info.name = broadwell_str;
                    cpuid_info.short_name = short_broadwell;
                    break;
                case BROADWELL_D:
                    cpuid_info.name = broadwell_d_str;
                    cpuid_info.short_name = short_broadwell;
                    break;
                case BROADWELL_E:
                    cpuid_info.name = broadwell_ep_str;
                    cpuid_info.short_name = short_broadwell_ep;
                    break;

                case NEHALEM_EX:
                    cpuid_info.name = nehalem_ex_str;
                    cpuid_info.short_name = short_nehalemEX;
                    break;

                case WESTMERE_EX:
                    cpuid_info.name = westmere_ex_str;
                    cpuid_info.short_name = short_westmereEX;
                    break;

                case XEON_MP:
                    cpuid_info.name = xeon_mp_string;
                    cpuid_info.short_name = short_unknown;
                    break;

                case ATOM_45:

                case ATOM:
                    cpuid_info.name = atom_45_str;
                    cpuid_info.short_name = short_atom;
                    break;

                case ATOM_32:
                    cpuid_info.name = atom_32_str;
                    cpuid_info.short_name = short_atom;
                    break;

                case ATOM_22:
                    cpuid_info.name = atom_22_str;
                    cpuid_info.short_name = short_atom;
                    break;

                case ATOM_SILVERMONT_E:
                case ATOM_SILVERMONT_C:
                case ATOM_SILVERMONT_Z1:
                case ATOM_SILVERMONT_Z2:
                case ATOM_SILVERMONT_F:
                    cpuid_info.name = atom_silvermont_str;
                    cpuid_info.short_name = short_silvermont;
                    break;
                case ATOM_SILVERMONT_AIR:
                    cpuid_info.name = atom_airmont_str;
                    cpuid_info.short_name = short_silvermont;
                    break;

                default:
                    cpuid_info.name = unknown_intel_str;
                    cpuid_info.short_name = short_unknown;
                    break;
            }
            break;

        case MIC_FAMILY:
            switch ( cpuid_info.model ) 
            {
                case XEON_PHI:
                    cpuid_info.name = xeon_phi_string;
                    cpuid_info.short_name = short_phi;
                    break;

            }
            break;

        case K8_FAMILY:

            if (cpuid_info.isIntel)
            {
                ERROR_PLAIN_PRINT(Netburst architecture is not supported);
            }

            switch ( cpuid_info.model )
            {
                case OPTERON_DC_E:
                    cpuid_info.name = opteron_dc_e_str;
                    break;

                case OPTERON_DC_F:
                    cpuid_info.name = opteron_dc_f_str;
                    break;

                case ATHLON64_X2:

                case ATHLON64_X2_F:
                    cpuid_info.name = athlon64_str;
                    break;

                case ATHLON64_F1:

                case ATHLON64_F2:
                    cpuid_info.name = athlon64_f_str;
                    break;

                case ATHLON64_X2_G:
                    cpuid_info.name = athlon64_X2_g_str;
                    break;

                case ATHLON64_G1:

                case ATHLON64_G2:
                    cpuid_info.name = athlon64_g_str;
                    break;

                case OPTERON_SC_1MB:
                    cpuid_info.name = opteron_sc_str;
                    break;

                default:
                    cpuid_info.name = amd_k8_str;
                    break;
            }
            cpuid_info.short_name = short_k8;
            break;

        case K10_FAMILY:
            switch ( cpuid_info.model )
            {
                case BARCELONA:
                    cpuid_info.name = barcelona_str;
                    break;

                case SHANGHAI:
                    cpuid_info.name = shanghai_str;
                    break;

                case ISTANBUL:
                    cpuid_info.name = istanbul_str;
                    break;

                case MAGNYCOURS:
                    cpuid_info.name = magnycours_str;
                    break;

                default:
                    cpuid_info.name = unknown_amd_str;
                    break;
            }
            cpuid_info.short_name = short_k10;
            break;

        case K15_FAMILY:
            cpuid_info.name = interlagos_str;
            cpuid_info.short_name = short_k15;
            break;

        case K16_FAMILY:
            cpuid_info.name = kabini_str;
            cpuid_info.short_name = short_k16;
            break;
            
        default:
            return EXIT_FAILURE;
            break;
    }
    return EXIT_SUCCESS;
}

const struct topology_functions topology_funcs = {
#ifndef LIKWID_USE_HWLOC
    .init_cpuInfo = cpuid_init_cpuInfo,
    .init_cpuFeatures = cpuid_init_cpuFeatures,
    .init_nodeTopology = cpuid_init_nodeTopology,
    .init_cacheTopology = cpuid_init_cacheTopology,
#else
    .init_cpuInfo = hwloc_init_cpuInfo,
    .init_nodeTopology = hwloc_init_nodeTopology,
    .init_cacheTopology = hwloc_init_cacheTopology,
    .init_cpuFeatures = proc_init_cpuFeatures,
#endif
    .init_fileTopology = initTopologyFile,
};


void topology_setupTree(void)
{
    uint32_t i;
    TreeNode* currentNode;
    HWThread* hwThreadPool = cpuid_topology.threadPool;
    
    tree_init(&cpuid_topology.topologyTree, 0);
    for (i=0; i<  cpuid_topology.numHWThreads; i++)
    {
        /* Add node to Topology tree */
        if (!tree_nodeExists(cpuid_topology.topologyTree,
                    hwThreadPool[i].packageId))
        {
            tree_insertNode(cpuid_topology.topologyTree,
                    hwThreadPool[i].packageId);
        }
        currentNode = tree_getNode(cpuid_topology.topologyTree,
                hwThreadPool[i].packageId);
        if (!tree_nodeExists(currentNode, hwThreadPool[i].coreId))
        {
            tree_insertNode(currentNode, hwThreadPool[i].coreId);
        }
        currentNode = tree_getNode(currentNode, hwThreadPool[i].coreId);
        if (!tree_nodeExists(currentNode, hwThreadPool[i].apicId))
        {
            /*
               printf("WARNING: Thread already exists!\n");
               */
            tree_insertNode(currentNode, hwThreadPool[i].apicId);
            affinity_thread2tile_lookup[hwThreadPool[i].apicId] = hwThreadPool[i].coreId;
        }

    }
    cpuid_topology.numSockets = tree_countChildren(cpuid_topology.topologyTree);
    currentNode = tree_getChildNode(cpuid_topology.topologyTree);
    cpuid_topology.numCoresPerSocket = tree_countChildren(currentNode);
    currentNode = tree_getChildNode(currentNode);
    cpuid_topology.numThreadsPerCore = tree_countChildren(currentNode);
    return;
}

int topology_init(void)
{
    struct topology_functions funcs = topology_funcs;

    if (init)
    {
        return EXIT_SUCCESS;
    }
    init = 1;

    init_configuration();

    if (access(config.topologyCfgFileName, R_OK))
    {
        cpu_set_t cpuSet;
        CPU_ZERO(&cpuSet);
        sched_getaffinity(0,sizeof(cpu_set_t), &cpuSet);
        if (cpu_count(&cpuSet) < sysconf(_SC_NPROCESSORS_CONF))
        {
            funcs.init_cpuInfo = proc_init_cpuInfo;
            funcs.init_cpuFeatures = proc_init_cpuFeatures;
            funcs.init_nodeTopology = proc_init_nodeTopology;
            funcs.init_cacheTopology = proc_init_cacheTopology;
            cpuid_topology.activeHWThreads =
                ((cpu_count(&cpuSet) < sysconf(_SC_NPROCESSORS_CONF)) ?
                cpu_count(&cpuSet) :
                sysconf(_SC_NPROCESSORS_CONF));
        }
        else
        {
            cpuid_topology.activeHWThreads = sysconf(_SC_NPROCESSORS_CONF);
        }
        funcs.init_cpuInfo(cpuSet);
        topology_setName();
        funcs.init_cpuFeatures();
        funcs.init_nodeTopology(cpuSet);
        topology_setupTree();
        funcs.init_cacheTopology();
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuSet);
    }
    else
    {
        readTopologyFile(config.topologyCfgFileName);
        topology_setName();
        topology_setupTree();
    }


    return EXIT_SUCCESS;
}


void topology_finalize(void)
{
    free(cpuid_info.features);
    free(cpuid_info.osname);
    free(cpuid_topology.cacheLevels);
    free(cpuid_topology.threadPool);
    tree_destroy(cpuid_topology.topologyTree);
}





void print_supportedCPUs (void)
{
    printf("\nSupported Intel processors:\n");
    printf("\t%s\n",core_2a_str);
    printf("\t%s\n",core_2b_str);
    printf("\t%s\n",xeon_mp_string);
    printf("\t%s\n",atom_45_str);
    printf("\t%s\n",atom_32_str);
    printf("\t%s\n",atom_22_str);
    printf("\t%s\n",nehalem_bloom_str);
    printf("\t%s\n",nehalem_lynn_str);
    printf("\t%s\n",nehalem_west_str);
    printf("\t%s\n",nehalem_ex_str);
    printf("\t%s\n",westmere_ex_str);
    printf("\t%s\n",sandybridge_str);
    printf("\t%s\n",sandybridge_ep_str);
    printf("\t%s\n",ivybridge_str);
    printf("\t%s\n",ivybridge_ep_str);
    printf("\t%s\n",haswell_str);
    printf("\t%s\n",haswell_ep_str);
    printf("\t%s\n",atom_silvermont_str);
    printf("\t%s\n",atom_airmont_str);
    printf("\t%s\n\n",xeon_phi_string);
    printf("\t%s\n",broadwell_str);

    printf("Supported AMD processors:\n");
    printf("\t%s\n",opteron_sc_str);
    printf("\t%s\n",opteron_dc_e_str);
    printf("\t%s\n",opteron_dc_f_str);
    printf("\t%s\n",barcelona_str);
    printf("\t%s\n",shanghai_str);
    printf("\t%s\n",istanbul_str);
    printf("\t%s\n",magnycours_str);
    printf("\t%s\n",interlagos_str);
    printf("\t%s\n\n",kabini_str);
}



CpuTopology_t get_cpuTopology(void)
{
    return &cpuid_topology;
}

CpuInfo_t get_cpuInfo(void)
{
    return &cpuid_info;
}
NumaTopology_t get_numaTopology(void)
{
    return &numa_info;
}

