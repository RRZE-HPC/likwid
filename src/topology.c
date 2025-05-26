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
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
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
#include <topology_static.h>

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int topology_initialized = 0;
static const char* pentium_m_b_str = "Intel Pentium M Banias processor";
static const char* pentium_m_d_str = "Intel Pentium M Dothan processor";
static const char* core_duo_str = "Intel Core Duo processor";
static const char* core_2a_str = "Intel Core 2 65nm processor";
static const char* core_2b_str = "Intel Core 2 45nm processor";
static const char* atom_45_str = "Intel Atom 45nm processor";
static const char* atom_32_str = "Intel Atom 32nm processor";
static const char* atom_22_str = "Intel Atom 22nm processor";
static const char* atom_silvermont_str = "Intel Atom (Silvermont) processor";
static const char* atom_airmont_str = "Intel Atom (Airmont) processor";
static const char* atom_goldmont_str = "Intel Atom (Goldmont) processor";
static const char* atom_goldmontplus_str = "Intel Atom (Goldmont Plus) processor";
static const char* atom_tremont_str = "Intel Atom (Tremont) processor";
static const char* nehalem_bloom_str = "Intel Core Bloomfield processor";
static const char* nehalem_lynn_str = "Intel Core Lynnfield processor";
static const char* nehalem_west_str = "Intel Core Westmere processor";
static const char* sandybridge_str = "Intel Core SandyBridge processor";
static const char* ivybridge_str = "Intel Core IvyBridge processor";
static const char* ivybridge_ep_str = "Intel Xeon IvyBridge EN/EP/EX processor";
static const char* sandybridge_ep_str = "Intel Xeon SandyBridge EN/EP processor";
static const char* haswell_str = "Intel Core Haswell processor";
static const char* haswell_ep_str = "Intel Xeon Haswell EN/EP/EX processor";
static const char* broadwell_str = "Intel Core Broadwell processor";
static const char* broadwell_d_str = "Intel Xeon D Broadwell processor";
static const char* broadwell_e3_str = "Intel Xeon E3 Broadwell processor";
static const char* broadwell_ep_str = "Intel Xeon Broadwell EN/EP/EX processor";
static const char* skylake_str = "Intel Skylake processor";
static const char* skylakeX_str = "Intel Skylake SP processor";
static const char* cascadelakeX_str = "Intel Cascadelake SP processor";
static const char* kabylake_str = "Intel Kabylake processor";
static const char* cannonlake_str = "Intel Cannonlake processor";
static const char* coffeelake_str = "Intel Coffeelake processor";
static const char* cometlake_str = "Intel Cometlake processor";
static const char* nehalem_ex_str = "Intel Nehalem EX processor";
static const char* westmere_ex_str = "Intel Westmere EX processor";
static const char* xeon_mp_string = "Intel Xeon MP processor";
static const char* xeon_phi_string = "Intel Xeon Phi (Knights Corner) Coprocessor";
static const char* xeon_phi2_string = "Intel Xeon Phi (Knights Landing) (Co)Processor";
static const char* xeon_phi3_string = "Intel Xeon Phi (Knights Mill) (Co)Processor";
static const char* icelake_str = "Intel Icelake processor";
static const char* tigerlake_str = "Intel Tigerlake processor";
static const char* icelakesp_str = "Intel Icelake SP processor";
static const char* rocketlake_str = "Intel Rocketlake processor";
static const char* sapphire_rapids_str = "Intel SapphireRapids processor";
static const char* emerald_rapids_str = "Intel EmeraldRapids processor";
static const char* granite_rapids_str = "Intel GraniteRapids processor";
static const char* sierra_forrest_str = "Intel SierraForrest processor";
//static const char* snowridgex_str = "Intel SnowridgeX processor";

static const char* barcelona_str = "AMD K10 (Barcelona) processor";
static const char* shanghai_str = "AMD K10 (Shanghai) processor";
static const char* istanbul_str = "AMD K10 (Istanbul) processor";
static const char* magnycours_str = "AMD K10 (Magny Cours) processor";
static const char* thuban_str = "AMD K10 (Thuban) processor";
static const char* interlagos_str = "AMD Interlagos processor";
static const char* kabini_str = "AMD Family 16 model - Kabini processor";
static const char* opteron_sc_str = "AMD Opteron single core 130nm processor";
static const char* opteron_dc_e_str = "AMD Opteron Dual Core Rev E 90nm processor";
static const char* opteron_dc_f_str = "AMD Opteron Dual Core Rev F 90nm processor";
static const char* athlon64_str = "AMD Athlon64 X2 (AM2) Rev F 90nm processor";
static const char* athlon64_f_str = "AMD Athlon64 (AM2) Rev F 90nm processor";
static const char* athlon64_X2_g_str = "AMD Athlon64 X2 (AM2) Rev G 65nm processor";
static const char* athlon64_g_str = "AMD Athlon64 (AM2) Rev G 65nm processor";
static const char* amd_k8_str = "AMD K8 architecture";
static const char* amd_zen_str = "AMD K17 (Zen) architecture";
static const char* amd_zenplus_str = "AMD K17 (Zen+) architecture";
static const char* amd_zen2_str = "AMD K17 (Zen2) architecture";
static const char* amd_zen3_str = "AMD K19 (Zen3) architecture";
static const char* amd_zen4_str = "AMD K19 (Zen4) architecture";
static const char* amd_zen4c_str = "AMD K19 (Zen4c) architecture";
static const char* armv7l_str = "ARM 7l architecture";
//static const char* armv8_str = "ARM 8 architecture";
static const char* cavium_thunderx2t99_str = "Cavium Thunder X2 (ARMv8)";
static const char* cavium_thunderx_str = "Cavium Thunder X (ARMv8)";
static const char* arm_cortex_a57 = "ARM Cortex A57";
static const char* arm_cortex_a53 = "ARM Cortex A53";
static const char* arm_cortex_a72 = "ARM Cortex A72";
static const char* arm_cortex_a73 = "ARM Cortex A73";
static const char* arm_cortex_a76 = "ARM Cortex A76";
static const char* arm_neoverse_n1 = "ARM Neoverse N1";
static const char* arm_neoverse_v1 = "ARM Neoverse V1";
static const char* arm_huawei_tsv110 = "Huawei TSV110 (ARMv8)";
static const char* arm_nvidia_grace = "Nvidia Grace";
static const char* fujitsu_a64fx = "Fujitsu A64FX";
static const char* apple_m1_studio = "Apple M1";
static const char* power7_str = "POWER7 architecture";
static const char* power8_str = "POWER8 architecture";
static const char* power9_str = "POWER9 architecture";

//static const char* unknown_intel_str = "Unknown Intel Processor";
static const char* unknown_amd_str = "Unknown AMD Processor";
static const char* unknown_power_str = "Unknown POWER Processor";

static const char* short_core2 = "core2";
static const char* short_atom = "atom";
static const char* short_pm = "pentiumm";
static const char* short_silvermont = "silvermont";
static const char* short_goldmont = "goldmont";
static const char* short_goldmontplus = "goldmontplus";
static const char* short_nehalem = "nehalem";
static const char* short_nehalemEX = "nehalemEX";
static const char* short_westmere = "westmere";
static const char* short_westmereEX = "westmereEX";
static const char* short_haswell = "haswell";
static const char* short_haswell_ep = "haswellEP";
static const char* short_broadwell = "broadwell";
static const char* short_broadwell_d = "broadwellD";
static const char* short_broadwell_ep = "broadwellEP";
static const char* short_ivybridge = "ivybridge";
static const char* short_ivybridge_ep = "ivybridgeEP";
static const char* short_sandybridge = "sandybridge";
static const char* short_sandybridge_ep = "sandybridgeEP";
static const char* short_skylake = "skylake";
static const char* short_skylakeX = "skylakeX";
static const char* short_kabylake = "skylake";
static const char* short_cascadelakeX = "CLX";
static const char* short_cannonlake = "cannonlake";
static const char* short_coffeelake = "coffeelake";
static const char* short_tigerlake = "TGL";
static const char* short_sapphire_rapids = "SPR";
static const char* short_emerald_rapids = "EMR";
static const char* short_phi = "phi";
static const char* short_phi2 = "knl";
static const char* short_icelake = "ICL";
static const char* short_rocketlake = "RKL";
static const char* short_icelakesp = "ICX";
static const char* short_granite_rapids = "GNR";
static const char* short_sierra_forrest = "SRF";
//static const char* short_snowridgex = "SNR";

static const char* short_k8 = "k8";
static const char* short_k10 = "k10";
static const char* short_k15 = "interlagos";
static const char* short_k16 = "kabini";
static const char* short_zen = "zen";
static const char* short_zen2 = "zen2";
static const char* short_zen3 = "zen3";
static const char* short_zen4 = "zen4";
static const char* short_zen4c = "zen4c";

static const char* short_arm7 = "arm7";
static const char* short_arm8 = "arm8";
static const char* short_arm8_cav_tx2 = "arm8_tx2";
static const char* short_arm8_cav_tx = "arm8_tx";
static const char* short_arm8_neo_n1 = "arm8_n1";
static const char* short_arm8_neo_v1 = "arm8_v1";
static const char* short_a64fx = "arm64fx";
static const char* short_apple_m1 = "apple_m1";
static const char* short_nvidia_grace = "nvidia_grace";

static const char* short_power7 = "power7";
static const char* short_power8 = "power8";
static const char* short_power9 = "power9";

static const char* short_unknown = "unknown";

/* #####  EXPORTED VARIABLES  ########################################## */

CpuInfo cpuid_info;
CpuTopology cpuid_topology;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static void
initTopologyFile(FILE* file)
{
    HWThread* hwThreadPool;
    CacheLevel* cacheLevels;
    TreeNode* currentNode;

    fread((void*) &cpuid_topology, sizeof(CpuTopology), 1, file);

    hwThreadPool = (HWThread*) malloc(cpuid_topology.numHWThreads * sizeof(HWThread));
    fread((void*) hwThreadPool, sizeof(HWThread), cpuid_topology.numHWThreads, file);
    cpuid_topology.threadPool = hwThreadPool;

    cacheLevels = (CacheLevel*) malloc(cpuid_topology.numCacheLevels * sizeof(CacheLevel));
    fread((void*) cacheLevels, sizeof(CacheLevel), cpuid_topology.numCacheLevels, file);
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
        }
    }
}

static int
readTopologyFile(const char* filename, cpu_set_t cpuSet)
{
    int err = 0;
    char structure[512];
    char field[512];
    char value[512];
    char line[512];
    int *numProcessorsPerNumaNode = NULL;

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        ERROR_PRINT("Failed to open topology file %s", filename);
        err = -errno;
        goto error;
    }

    int numHWThreads = -1;
    int numCacheLevels = -1;
    int numNumaNodes = -1;

    while (fgets(line, sizeof(line), fp) != NULL) {
        if (sscanf(line,"%s %s", structure, field) != 2)
            continue;

        if ((strcmp(structure, "cpuid_topology") == 0) && (strcmp(field, "numHWThreads") == 0)) {
            if (sscanf(line,"%s %s = %d", structure, field, &numHWThreads) != 3)
                continue;
        } else if ((strcmp(structure, "cpuid_topology") == 0) && (strcmp(field, "numCacheLevels") == 0)) {
            if (sscanf(line,"%s %s = %d", structure, field, &numCacheLevels) != 3)
                continue;
        } else if ((strcmp(structure, "numa_info") == 0) && (strcmp(field, "numberOfNodes") == 0)) {
            if (sscanf(line,"%s %s = %d", structure, field, &numNumaNodes) != 3)
                continue;
        }
        if (numHWThreads >= 0 && numCacheLevels >= 0 && numNumaNodes >= 0) {
            break;
        }
    }

    if (numHWThreads < 0 || numCacheLevels < 0 || numNumaNodes < 0) {
        ERROR_PRINT("Topology file error: %s\n  numHWThreads=%s numCacheLevels=%s numberOfNodes=%s",
                filename,
                (numHWThreads < 0) ? "missing" : "ok",
                (numCacheLevels < 0) ? "missing" : "ok",
                (numNumaNodes < 0) ? "missing" : "ok");
        err = -EINVAL;
        goto error;
    }

    cpuid_topology.numHWThreads = numHWThreads;
    cpuid_topology.numCacheLevels = numCacheLevels;
    numa_info.numberOfNodes = numNumaNodes;

    if (fseek(fp, 0, SEEK_SET) < 0) {
        err = -errno;
        goto error;
    }

    numProcessorsPerNumaNode = calloc(numa_info.numberOfNodes, sizeof(*numProcessorsPerNumaNode));
    int curNumaNode = 0;
    while (fgets(line, sizeof(line), fp) != NULL) {
        int numaNode = -1; // starting from 1, not from 0
        int numberOfProcessors = -1;
        if (sscanf(line,"%s %s %d %s = %d", structure, field, &numaNode, value, &numberOfProcessors) != 5)
            continue;
        if (strcmp(structure, "numa_info") != 0 || strcmp(value, "numberOfProcessors") != 0)
            continue;

        numaNode -= 1;

        if (numaNode < 0 || numaNode >= numNumaNodes) {
            ERROR_PRINT("Topology file contains numa_info out of range: %s", filename);
            err = -EINVAL;
            goto error;
        }

        numProcessorsPerNumaNode[numaNode] = numberOfProcessors;
        curNumaNode++;

        if (curNumaNode == numNumaNodes)
            break;
    }

    if (curNumaNode != numNumaNodes) {
        ERROR_PRINT("Topology file only contains %d of claimed %d numa_info lines: %s",
                curNumaNode, numNumaNodes, filename);
        err = -EINVAL;
        goto error;
    }

    cpuid_topology.threadPool = calloc(numHWThreads, sizeof(*cpuid_topology.threadPool));
    if (!cpuid_topology.threadPool) {
        err = -errno;
        goto error;
    }

    cpuid_topology.cacheLevels = calloc(numCacheLevels, sizeof(*cpuid_topology.cacheLevels));
    if (!cpuid_topology.cacheLevels) {
        err = -errno;
        goto error;
    }

    cpuid_topology.numHWThreads = numHWThreads;
    cpuid_topology.numCacheLevels = numCacheLevels;

    numa_info.nodes = calloc(numNumaNodes, sizeof(*numa_info.nodes));
    if (!numa_info.nodes) {
        err = -errno;
        goto error;
    }

    for (size_t i = 0; i< numa_info.numberOfNodes; i++) {
        numa_info.nodes[i].processors = calloc(numProcessorsPerNumaNode[i], sizeof(*numa_info.nodes[i].processors));
        if (!numa_info.nodes[i].processors) {
            err = -errno;
            goto error;
        }
        numa_info.nodes[i].distances = calloc(numNumaNodes, sizeof(*numa_info.nodes[i].distances));
        if (!numa_info.nodes[i].distances) {
            err = -errno;
            goto error;
        }
    }

    if (fseek(fp, 0, SEEK_SET) < 0) {
        err = -errno;
        goto error;
    }

    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "%s %s", structure, field) != 2)
            continue;

        if (strcmp(structure, "cpuid_topology") == 0) {
            unsigned tmp;
            if (strcmp(field, "numSockets") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_topology.numSockets = tmp;
            } else if (strcmp(field, "numCoresPerSocket") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_topology.numCoresPerSocket = tmp;
            } else if (strcmp(field, "numThreadsPerCore") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_topology.numThreadsPerCore = tmp;
            } else if (strcmp(field, "numDies") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_topology.numDies = tmp;
            } else if (strcmp(field, "threadPool") == 0) {
                unsigned hwthread;
                if (sscanf(line, "%s %s %u %s = %u", structure, field, &hwthread, value, &tmp) != 5)
                    continue;

                if (strcmp(value, "threadId") == 0) {
                    cpuid_topology.threadPool[hwthread].threadId = tmp;
                } else if (strcmp(value, "coreId") == 0) {
                    cpuid_topology.threadPool[hwthread].coreId = tmp;
                } else if (strcmp(value, "packageId") == 0) {
                    cpuid_topology.threadPool[hwthread].packageId = tmp;
                } else if (strcmp(value, "dieId") == 0) {
                    cpuid_topology.threadPool[hwthread].dieId = tmp;
                } else if (strcmp(value, "apicId") == 0) {
                    cpuid_topology.threadPool[hwthread].apicId = tmp;
                    if (CPU_ISSET(tmp, &cpuSet)) {
                        cpuid_topology.threadPool[hwthread].inCpuSet = 1;
                    } else {
                        cpuid_topology.threadPool[hwthread].inCpuSet = 0;
                    }
                }
            } else if (strcmp(field, "cacheLevels") == 0) {
                int cacheLevel = 0, dummy;
                char type[128];
                if (sscanf(line, "%s %s %d %s", structure, field, &cacheLevel, value) != 4)
                    continue;

                if (cacheLevel <= 0 || cacheLevel > numCacheLevels) {
                    ERROR_PRINT("Topology file contains cacheLevels line with level out of range: %d", cacheLevel);
                    err = -EINVAL;
                    goto error;
                }

                cacheLevel -= 1;

                cpuid_topology.cacheLevels[cacheLevel].level = cacheLevel;
                if (strcmp(value, "type") == 0) {
                    if (sscanf(line, "%s %s %d %s = %s", structure, field, &dummy, value, type) != 5)
                        continue;

                    if (strcmp(type, "UNIFIEDCACHE") == 0)
                        cpuid_topology.cacheLevels[cacheLevel].type = UNIFIEDCACHE;
                    else if (strcmp(type, "DATACACHE") == 0)
                        cpuid_topology.cacheLevels[cacheLevel].type = DATACACHE;
                    else if (strcmp(type, "INSTRUCTIONCACHE") == 0)
                        cpuid_topology.cacheLevels[cacheLevel].type = INSTRUCTIONCACHE;
                    else if (strcmp(type, "ITLB") == 0)
                        cpuid_topology.cacheLevels[cacheLevel].type = ITLB;
                    else if (strcmp(type, "DTLB") == 0)
                        cpuid_topology.cacheLevels[cacheLevel].type = DTLB;
                    else if (strcmp(type, "NOCACHE") == 0)
                        cpuid_topology.cacheLevels[cacheLevel].type = NOCACHE;
                } else {
                    if (sscanf(line, "%s %s %d %s = %u", structure, field, &dummy, value, &tmp) != 5)
                        continue;

                    if (strcmp(value, "associativity") == 0)
                        cpuid_topology.cacheLevels[cacheLevel].associativity = tmp;
                    else if (strcmp(value, "sets") == 0)
                        cpuid_topology.cacheLevels[cacheLevel].sets = tmp;
                    else if (strcmp(value, "lineSize") == 0)
                        cpuid_topology.cacheLevels[cacheLevel].lineSize = tmp;
                    else if (strcmp(value, "size") == 0)
                        cpuid_topology.cacheLevels[cacheLevel].size = tmp;
                    else if (strcmp(value, "threads") == 0)
                        cpuid_topology.cacheLevels[cacheLevel].threads = tmp;
                    else if (strcmp(value, "inclusive") == 0)
                        cpuid_topology.cacheLevels[cacheLevel].inclusive = tmp;
                }
            }
        } else if (strcmp(structure, "cpuid_info") == 0) {
            unsigned tmp;
            if (strcmp(field, "family") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_info.family = tmp;
            } else if (strcmp(field, "model") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_info.model = tmp;
            } else if (strcmp(field, "osname") == 0) {
                // line example:
                // cpuid_info osname = Intel(R) Code(TM) ...
                const size_t val_index = strlen(structure) + strlen(field) + 4;
                if (strlen(line) < val_index)
                    continue;

                snprintf(value, sizeof(value), "%s", &line[val_index]);

                cpuid_info.osname = strdup(value);
                if (!cpuid_info.osname) {
                    err = -errno;
                    goto error;
                }
            } else if (strcmp(field, "stepping") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_info.stepping = tmp;
            } else if (strcmp(field, "vendor") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_info.vendor = tmp;
            } else if (strcmp(field, "part") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_info.part = tmp;
            } else if (strcmp(field, "clock") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_info.clock = tmp;
            } else if (strcmp(field, "turbo") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_info.turbo = tmp;
            } else if (strcmp(field, "isIntel") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_info.isIntel = tmp;
            } else if (strcmp(field, "featureFlags") == 0) {
                uint64_t flags;
                if (sscanf(line, "%s %s = %ld", structure, field, &flags) != 3)
                    continue;
                cpuid_info.featureFlags = flags;
            } else if (strcmp(field, "perf_version") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_info.perf_version = tmp;
            } else if (strcmp(field, "perf_num_ctr") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_info.perf_num_ctr = tmp;
            } else if (strcmp(field, "perf_width_ctr") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_info.perf_width_ctr = tmp;
            } else if (strcmp(field, "perf_num_fixed_ctr") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_info.perf_num_fixed_ctr = tmp;
            } else if (strcmp(field, "supportClientmem") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_info.supportClientmem = tmp;
            } else if (strcmp(field, "supportUncore") == 0) {
                if (sscanf(line, "%s %s = %u", structure, field, &tmp) != 3)
                    continue;
                cpuid_info.supportUncore = tmp;
            } else if (strcmp(field, "features") == 0) {
                // line example
                // cpuid_info features = TP ACPI MMX ...
                const size_t val_index = strlen(structure) + strlen(field) + 4;
                if (strlen(line) < val_index)
                    continue;

                cpuid_info.features = strdup(&line[val_index]);
                if (!cpuid_info.features) {
                    err = -errno;
                    goto error;
                }
            } else if (strcmp(field, "architecture") == 0) {
                // line example
                // cpuid_info architecture = x86_64
                const size_t val_index = strlen(structure) + strlen(field) + 4;
                if (strlen(line) < val_index)
                    continue;
                snprintf(cpuid_info.architecture, sizeof(cpuid_info.architecture), "%s", &line[val_index]);
            }
        } else if (strcmp(structure, "numa_info") == 0) {
            if (strcmp(field, "nodes") == 0) {
                unsigned tmp, tmp1;
                int id = 0, dummy;
                if (sscanf(line, "%s %s %d %s", structure, field, &id, value) != 4)
                    continue;

                id -= 1;

                if (id < 0 || id >= numNumaNodes)
                    continue;

                if (strcmp(value,"numberOfProcessors") == 0) {
                    if (sscanf(line, "%s %s %d %s = %u", structure, field, &dummy, value, &tmp) != 5)
                        continue;
                    numa_info.nodes[id].numberOfProcessors = tmp;
                } else if (strcmp(value, "freeMemory") == 0) {
                    if (sscanf(line, "%s %s %d %s = %u", structure, field, &dummy, value, &tmp) != 5)
                        continue;
                    numa_info.nodes[id].freeMemory = tmp;
                } else if (strcmp(value, "id") == 0) {
                    if (sscanf(line, "%s %s %d %s = %u", structure, field, &dummy, value, &tmp) != 5)
                        continue;
                    numa_info.nodes[id].id = tmp;
                } else if (strcmp(value, "totalMemory") == 0) {
                    if (sscanf(line, "%s %s %d %s = %u", structure, field, &dummy, value, &tmp) != 5)
                        continue;
                    numa_info.nodes[id].totalMemory = tmp;
                } else if (strcmp(value, "numberOfDistances") == 0) {
                    if (sscanf(line, "%s %s %d %s = %u", structure, field, &dummy, value, &tmp) != 5)
                        continue;
                    numa_info.nodes[id].numberOfDistances = tmp;
                } else if (strcmp(value, "processors") == 0) {
                    if (sscanf(line, "%s %s %d %s %u = %u", structure, field, &dummy, value, &tmp, &tmp1) != 5)
                        continue;
                    numa_info.nodes[id].processors[tmp-1] = tmp1;
                } else if (strcmp(value,"distances") == 0) {
                    if (sscanf(line, "%s %s %d %s %u = %u", structure, field, &dummy, value, &tmp, &tmp1) != 5)
                        continue;
                    numa_info.nodes[id].distances[tmp] = tmp1;
                }
            }
        }
    }

    free(numProcessorsPerNumaNode);

    return 0;

error:
    if (fp)
        fclose(fp);

    free(numProcessorsPerNumaNode);

    free(cpuid_topology.cacheLevels);
    free(cpuid_topology.threadPool);
    memset(&cpuid_topology, 0, sizeof(cpuid_topology));

    free(cpuid_info.features);
    free(cpuid_info.osname);
    memset(&cpuid_info, 0, sizeof(cpuid_info));

    if (numa_info.nodes) {
        for (size_t i = 0; i < numa_info.numberOfNodes; i++) {
            free(numa_info.nodes[i].processors);
            free(numa_info.nodes[i].distances);
        }
    }

    free(numa_info.nodes);
    memset(&numa_info, 0, sizeof(numa_info));
    return err;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
cpu_count(cpu_set_t* set)
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

int
likwid_cpu_online(int cpu_id)
{
    if (cpu_id < 0)
        return 0;
    int state = 0;
    bstring bpath = bformat("/sys/devices/system/cpu/cpu%d/online", cpu_id);
    FILE* fp = fopen(bdata(bpath), "r");
    if (fp)
    {
        char buf[10];
        int ret = fread(buf, sizeof(char), 9, fp);
        fclose(fp);
        if (ret > 0)
        {
            state = atoi(buf);
        }
    }
    else
    {
        fp = fopen("/sys/devices/system/cpu/online", "r");
        if (fp)
        {
            char buf[100];
            int ret = fread(buf, sizeof(char), 99, fp);
            fclose(fp);
            if (ret > 0)
            {
                bdestroy(bpath);
                buf[ret] = '\0';
                bpath = bfromcstr(buf);
                struct bstrList* first = bsplit(bpath, ',');
                for (int i = 0; i < first->qty; i++)
                {
                    struct bstrList* second = bsplit(first->entry[i], '-');
                    if (second->qty == 1)
                    {
#pragma GCC diagnostic ignored "-Wnonnull"
                        int core = atoi(bdata(second->entry[0]));
                        if (core == cpu_id)
                            state = 1;
                    }
                    else if (second->qty == 2)
                    {
#pragma GCC diagnostic ignored "-Wnonnull"
                        int s = atoi(bdata(second->entry[0]));
                        int e = atoi(bdata(second->entry[1]));
                        if (cpu_id >= s && cpu_id <= e)
                            state = 1;
                    }
                    bstrListDestroy(second);
                }
                bstrListDestroy(first);

            }
        }
    }
    bdestroy(bpath);
    return state;
}

size_t likwid_sysfs_list_len(char* sysfsfile)
{
    size_t len = 0;
    FILE* fp;
    if (NULL != (fp = fopen (sysfsfile, "r")))
    {
        bstring src = bread ((bNread) fread, fp);
        struct bstrList* first = bsplit(src, ',');
        for (int i = 0; i < first->qty; i++)
        {
            struct bstrList* second = bsplit(first->entry[i], '-');
            if (second->qty == 1)
            {
                len++;
            }
            else
            {
                #pragma GCC diagnostic ignored "-Wnonnull"
                int s = atoi(bdata(second->entry[0]));
                int e = atoi(bdata(second->entry[1]));
                len += e - s + 1;
            }
            bstrListDestroy(second);
        }
        bstrListDestroy(first);
        bdestroy(src);
        fclose(fp);
    }
    return len;
}



int
topology_setName(void)
{
    int err = 0;
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
                    cpuid_info.supportClientmem = 1;
                    cpuid_info.name = sandybridge_str;
                    cpuid_info.short_name = short_sandybridge;
                    break;

                case SANDYBRIDGE_EP:
                    cpuid_info.supportUncore = 1;
                    cpuid_info.name = sandybridge_ep_str;
                    cpuid_info.short_name = short_sandybridge_ep;
                    break;

                case IVYBRIDGE:
                    cpuid_info.supportClientmem = 1;
                    cpuid_info.name = ivybridge_str;
                    cpuid_info.short_name = short_ivybridge;
                    break;

                case IVYBRIDGE_EP:
                    cpuid_info.supportUncore = 1;
                    cpuid_info.name = ivybridge_ep_str;
                    cpuid_info.short_name = short_ivybridge_ep;
                    break;

                case HASWELL_EP:
                    cpuid_info.supportUncore = 1;
                    cpuid_info.name = haswell_ep_str;
                    cpuid_info.short_name = short_haswell_ep;
                    break;
                case HASWELL:
                case HASWELL_M1:
                case HASWELL_M2:
                    cpuid_info.supportClientmem = 1;
                    cpuid_info.name = haswell_str;
                    cpuid_info.short_name = short_haswell;
                    break;

                case BROADWELL:
                    cpuid_info.supportClientmem = 1;
                    cpuid_info.name = broadwell_str;
                    cpuid_info.short_name = short_broadwell;
                    break;
                case BROADWELL_E3:
                    cpuid_info.name = broadwell_e3_str;
                    cpuid_info.short_name = short_broadwell;
                    break;
                case BROADWELL_D:
                    cpuid_info.supportUncore = 1;
                    cpuid_info.name = broadwell_d_str;
                    cpuid_info.short_name = short_broadwell_d;
                    break;
                case BROADWELL_E:
                    cpuid_info.supportUncore = 1;
                    cpuid_info.name = broadwell_ep_str;
                    cpuid_info.short_name = short_broadwell_ep;
                    break;

                case SKYLAKE1:
                case SKYLAKE2:
                    cpuid_info.supportClientmem = 1;
                    cpuid_info.name = skylake_str;
                    cpuid_info.short_name = short_skylake;
                    break;

                case SKYLAKEX:
                    if (cpuid_info.stepping < 5)
                    {
                        cpuid_info.name = skylakeX_str;
                        cpuid_info.short_name = short_skylakeX;
                    }
                    else
                    {
                        cpuid_info.name = cascadelakeX_str;
                        cpuid_info.short_name = short_cascadelakeX;
                    }
                    cpuid_info.supportUncore = 1;
                    break;

                case KABYLAKE1:
                    cpuid_info.supportClientmem = 1;
                    cpuid_info.name = kabylake_str;
                    cpuid_info.short_name = short_kabylake;
                    break;

                case KABYLAKE2:
                    cpuid_info.supportClientmem = 1;
                    cpuid_info.name = coffeelake_str;
                    cpuid_info.short_name = short_coffeelake;
                    break;

                case CANNONLAKE:
                    cpuid_info.supportClientmem = 1;
                    cpuid_info.name = cannonlake_str;
                    cpuid_info.short_name = short_cannonlake;
                    break;

                case COMETLAKE1:
                case COMETLAKE2:
                    cpuid_info.supportClientmem = 1;
                    cpuid_info.name = cometlake_str;
                    cpuid_info.short_name = short_skylake;
                    break;

                case XEON_PHI_KNL:
                    cpuid_info.supportUncore = 1;
                    cpuid_info.name = xeon_phi2_string;
                    cpuid_info.short_name = short_phi2;
                    break;

                case XEON_PHI_KML:
                    cpuid_info.supportUncore = 1;
                    cpuid_info.name = xeon_phi3_string;
                    cpuid_info.short_name = short_phi2;
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
                    cpuid_info.short_name = short_core2;
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
                case ATOM_SILVERMONT_GOLD:
                case ATOM_DENVERTON:
                    cpuid_info.name = atom_goldmont_str;
                    cpuid_info.short_name = short_goldmont;
                    break;
                case ATOM_GOLDMONT_PLUS:
                    cpuid_info.name = atom_goldmontplus_str;
                    cpuid_info.short_name = short_goldmontplus;
                    break;
                case ATOM_TREMONT:
                    cpuid_info.name = atom_tremont_str;
                    cpuid_info.short_name = short_goldmontplus;
                    break;

                case ICELAKE1:
                case ICELAKE2:
                    cpuid_info.supportClientmem = 1;
                    cpuid_info.name = icelake_str;
                    cpuid_info.short_name = short_icelake;
                    break;

                case ROCKETLAKE:
                    cpuid_info.supportClientmem = 1;
                    cpuid_info.name = rocketlake_str;
                    cpuid_info.short_name = short_rocketlake;
                    break;

                case ICELAKEX1:
                case ICELAKEX2:
                    cpuid_info.supportUncore = 1;
                    cpuid_info.name = icelakesp_str;
                    cpuid_info.short_name = short_icelakesp;
                    break;

/*                case SNOWRIDGEX:*/
/*                    cpuid_info.name = snowridgex_str;*/
/*                    cpuid_info.short_name = short_snowridgex;*/
/*                    break;*/

                case TIGERLAKE1:
                case TIGERLAKE2:
                    //cpuid_info.supportClientmem = 1;
                    cpuid_info.name = tigerlake_str;
                    cpuid_info.short_name = short_tigerlake;
                    break;

                case SAPPHIRERAPIDS:
                    cpuid_info.supportUncore = 1;
                    cpuid_info.supportClientmem = 0;
                    cpuid_info.name = sapphire_rapids_str;
                    cpuid_info.short_name = short_sapphire_rapids;
                    break;
                
                case EMERALDRAPIDS:
                    cpuid_info.supportUncore = 1;
                    cpuid_info.supportClientmem = 0;
                    cpuid_info.name = emerald_rapids_str;
                    cpuid_info.short_name = short_emerald_rapids;
                    break;

                case GRANITERAPIDS:
                    cpuid_info.supportUncore = 1;
                    cpuid_info.supportClientmem = 0;
                    cpuid_info.name = granite_rapids_str;
                    cpuid_info.short_name = short_granite_rapids;
                    break;

                case SIERRAFORREST:
                    cpuid_info.supportUncore = 0;
                    cpuid_info.supportClientmem = 0;
                    cpuid_info.name = sierra_forrest_str;
                    cpuid_info.short_name = short_sierra_forrest;
                    break;

                default:
                    err = -EFAULT;
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
                default:
                    err = -EFAULT;
                    break;
            }
            break;

        case K8_FAMILY:

            if (cpuid_info.isIntel)
            {
                ERROR_PRINT("Netburst architecture is not supported");
                err = -EFAULT;
                break;
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

                case THUBAN:
                    cpuid_info.name = thuban_str;
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

        case PPC_FAMILY:
            switch(cpuid_info.model)
            {
                case POWER7:
                    cpuid_info.name = power7_str;
                    cpuid_info.short_name = short_power7;
                    break;
                case POWER8:
                    cpuid_info.name = power8_str;
                    cpuid_info.short_name = short_power8;
                    break;
                case POWER9:
                    cpuid_info.name = power9_str;
                    cpuid_info.short_name = short_power9;
                    break;
                default:
                    cpuid_info.name = unknown_power_str;
                    cpuid_info.short_name = short_unknown;
                    err = -EFAULT;
                    break;
           }
           break;


        case ZEN_FAMILY:
            switch (cpuid_info.model)
            {
                case ZEN_RYZEN:
                    cpuid_info.name = amd_zen_str;
                    cpuid_info.short_name = short_zen;
                    cpuid_info.supportUncore = 1;
                    break;
                case ZENPLUS_RYZEN:
                case ZENPLUS_RYZEN2:
                    cpuid_info.name = amd_zenplus_str;
                    cpuid_info.short_name = short_zen;
                    cpuid_info.supportUncore = 1;
                    break;
                case ZEN2_RYZEN:
                case ZEN2_RYZEN2:
                case ZEN2_RYZEN3:
                    cpuid_info.name = amd_zen2_str;
                    cpuid_info.short_name = short_zen2;
                    cpuid_info.supportUncore = 1;
                    break;
                default:
                    err = -EFAULT;
                    break;
            }
            break;
        case ZEN3_FAMILY:
            switch (cpuid_info.model)
            {
                case ZEN3_RYZEN:
                case ZEN3_RYZEN2:
                case ZEN3_RYZEN3:
                case ZEN3_EPYC_TRENTO:
                    cpuid_info.name = amd_zen3_str;
                    cpuid_info.short_name = short_zen3;
                    cpuid_info.supportUncore = 1;
                    break;
                case ZEN4_RYZEN:
                case ZEN4_RYZEN2:
                case ZEN4_RYZEN3:
                case ZEN4_EPYC:
                case ZEN4_RYZEN_PRO:
                    cpuid_info.name = amd_zen4_str;
                    cpuid_info.short_name = short_zen4;
                    cpuid_info.supportUncore = 1;
                    break;
                case ZEN4_EPYC_BERGAMO:
                    cpuid_info.name = amd_zen4c_str;
                    cpuid_info.short_name = short_zen4c;
                    cpuid_info.supportUncore = 1;
                    break;
                default:
                    err = -EFAULT;
                    break;
            }
            break;

        case ARMV7_FAMILY:
            switch (cpuid_info.part)
            {
                case ARM7L:
                case ARMV7L:
                    cpuid_info.name = armv7l_str;
                    cpuid_info.short_name = short_arm7;
                    break;
                case ARM_CORTEX_A57:
                    cpuid_info.name = arm_cortex_a57;
                    cpuid_info.short_name = short_arm7;
                    break;
                case ARM_CORTEX_A53:
                    cpuid_info.name = arm_cortex_a53;
                    cpuid_info.short_name = short_arm7;
                    break;
                case ARM_CORTEX_A72:
                    cpuid_info.name = arm_cortex_a72;
                    cpuid_info.short_name = short_arm7;
                    break;
                case ARM_CORTEX_A73:
                    cpuid_info.name = arm_cortex_a73;
                    cpuid_info.short_name = short_arm7;
                    break;
                default:
                    err = -EFAULT;
                    break;
            }
            break;
        case ARMV8_FAMILY:
            switch (cpuid_info.vendor)
            {
                case DEFAULT_ARM:
                    switch (cpuid_info.part)
                    {
                        case ARM_CORTEX_A57:
                            cpuid_info.name = arm_cortex_a57;
                            cpuid_info.short_name = short_arm8;
                            break;
                        case ARM_CORTEX_A53:
                            cpuid_info.name = arm_cortex_a53;
                            cpuid_info.short_name = short_arm8;
                            break;
                        case ARM_CORTEX_A72:
                            cpuid_info.name = arm_cortex_a72;
                            cpuid_info.short_name = short_arm8;
                            break;
                        case ARM_CORTEX_A73:
                            cpuid_info.name = arm_cortex_a73;
                            cpuid_info.short_name = short_arm8;
                            break;
                        case ARM_CORTEX_A76:
                            cpuid_info.name = arm_cortex_a76;
                            cpuid_info.short_name = short_arm8;
                            break;
                        case ARM_NEOVERSE_N1:
                            cpuid_info.name = arm_neoverse_n1;
                            cpuid_info.short_name = short_arm8_neo_n1;
                            break;
                        case AWS_GRAVITON3:
                            cpuid_info.name = arm_neoverse_v1;
                            cpuid_info.short_name = short_arm8_neo_v1;
                            break;
                        case NVIDIA_GRACE:
                            cpuid_info.name = arm_nvidia_grace;
                            cpuid_info.short_name = short_nvidia_grace;
                            cpuid_info.supportUncore = 1;
                            break;
                        default:
                            err = -EFAULT;
                            break;
                    }
                    break;
                case CAVIUM1:
                case CAVIUM2:
                    switch (cpuid_info.part)
                    {
                        case CAV_THUNDERX2T99:
                        case CAV_THUNDERX2T99P1:
                            cpuid_info.name = cavium_thunderx2t99_str;
                            cpuid_info.short_name = short_arm8_cav_tx2;
                            break;
                        case CAV_THUNDERX:
                        case CAV_THUNDERX88:
                        case CAV_THUNDERX81:
                        case CAV_THUNDERX82:
                            cpuid_info.name = cavium_thunderx_str;
                            cpuid_info.short_name = short_arm8_cav_tx;
                            break;
                        default:
                            err = -EFAULT;
                            break;
                    }
                    break;
                case FUJITSU_ARM:
                    switch (cpuid_info.part)
                    {
                        case FUJITSU_A64FX:
                            cpuid_info.name = fujitsu_a64fx;
                            cpuid_info.short_name = short_a64fx;
                            break;
                        default:
                            err = -EFAULT;
                            break;
                    }
                    break;
                case APPLE_M1:
                case APPLE:
                    switch (cpuid_info.model)
                    {
                        case APPLE_M1_STUDIO:
                            cpuid_info.name = apple_m1_studio;
                            cpuid_info.short_name = short_apple_m1;
                            break;
                        default:
                            err = -EFAULT;
                            break;
                    }
                    break;
                case HUAWEI_ARM:
                    switch (cpuid_info.part)
                    {
                        case HUAWEI_TSV110:
                            cpuid_info.name = arm_huawei_tsv110;
                            cpuid_info.short_name = short_arm8;
                            break;
                        default:
                            err = -EFAULT;
                            break;

                    }
                    break;
                default:
                    err = -EFAULT;
                    break;
            }
            break;
        default:
            err = -EFAULT;
            break;
    }
    return err;
}

const struct
topology_functions topology_funcs = {
#ifndef LIKWID_USE_HWLOC
    .init_cpuInfo = proc_init_cpuInfo,
    .init_cpuFeatures = proc_init_cpuFeatures,
    .init_nodeTopology = proc_init_nodeTopology,
    .init_cacheTopology = proc_init_cacheTopology,
    .close_topology = NULL,
#else
#if !defined(__ARM_ARCH_8A)
    .init_cpuInfo = hwloc_init_cpuInfo,
    .init_nodeTopology = hwloc_init_nodeTopology,
    .init_cacheTopology = hwloc_init_cacheTopology,
    .init_cpuFeatures = proc_init_cpuFeatures,
    .close_topology = hwloc_close,
#else
    .init_cpuInfo = hwloc_init_cpuInfo,
    .init_cpuFeatures = proc_init_cpuFeatures,
    .init_nodeTopology = proc_init_nodeTopology,
    .init_cacheTopology = proc_init_cacheTopology,
    .close_topology = NULL,
#endif
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
            DEBUG_PRINT(DEBUGLEV_DEVELOP, "Adding socket %d", hwThreadPool[i].packageId);
            tree_insertNode(cpuid_topology.topologyTree,
                    hwThreadPool[i].packageId);
        }
        currentNode = tree_getNode(cpuid_topology.topologyTree,
                hwThreadPool[i].packageId);
        /*if (!tree_nodeExists(currentNode, hwThreadPool[i].dieId))
        {
            printf("Insert Die %d at Socket %d\n", hwThreadPool[i].dieId, hwThreadPool[i].packageId);
            tree_insertNode(currentNode, hwThreadPool[i].dieId);
        }
        currentNode = tree_getNode(currentNode, hwThreadPool[i].dieId);*/
        if (!tree_nodeExists(currentNode, hwThreadPool[i].coreId))
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, "Adding core %d to socket %d", hwThreadPool[i].coreId, hwThreadPool[i].packageId);
            tree_insertNode(currentNode, hwThreadPool[i].coreId);
        }
        currentNode = tree_getNode(currentNode, hwThreadPool[i].coreId);
        if (!tree_nodeExists(currentNode, hwThreadPool[i].apicId))
        {
            /*
               printf("WARNING: Thread already exists!\n");
               */
            DEBUG_PRINT(DEBUGLEV_DEVELOP, "Adding hwthread %d at core %d on socket %d", hwThreadPool[i].apicId, hwThreadPool[i].coreId, hwThreadPool[i].packageId);
            tree_insertNode(currentNode, hwThreadPool[i].apicId);
        }

    }
    i = tree_countChildren(cpuid_topology.topologyTree);
    DEBUG_PRINT(DEBUGLEV_DEVELOP, "Determine number of sockets. tree tells %d", i);
    if (cpuid_topology.numSockets == 0)
        cpuid_topology.numSockets = i;
    currentNode = tree_getChildNode(cpuid_topology.topologyTree);
    i = tree_countChildren(currentNode);
    DEBUG_PRINT(DEBUGLEV_DEVELOP, "Determine number of cores per socket. tree tells %d", i);
    if (cpuid_topology.numCoresPerSocket == 0)
        cpuid_topology.numCoresPerSocket = i;
    currentNode = tree_getChildNode(currentNode);
    i = tree_countChildren(currentNode);
    DEBUG_PRINT(DEBUGLEV_DEVELOP, "Determine number of hwthreads per cores. tree tells %d", i);
    if (cpuid_topology.numThreadsPerCore == 0)
        cpuid_topology.numThreadsPerCore = i;
    return;
}

int
topology_init(void)
{
    int ret = 0;
    cpu_set_t cpuSet;
    struct topology_functions funcs = topology_funcs;
    if (topology_initialized)
    {
        return EXIT_SUCCESS;
    }

    if (init_configuration())
    {
        ERROR_PRINT("Cannot initialize configuration module to check for topology file name");
        return EXIT_FAILURE;
    }

    if ((config.topologyCfgFileName == NULL) || access(config.topologyCfgFileName, R_OK))
    {
standard_init:
        CPU_ZERO(&cpuSet);
        if (getenv("LIKWID_IGNORE_CPUSET") == NULL)
        {
            sched_getaffinity(0,sizeof(cpu_set_t), &cpuSet);
        }
        else
        {
            for (int i = 0; i < sysconf(_SC_NPROCESSORS_CONF); i++)
            {
                if (likwid_cpu_online(i))
                {
                    CPU_SET(i, &cpuSet);
                }
            }
        }
        if (cpu_count(&cpuSet) < sysconf(_SC_NPROCESSORS_CONF))
        {
#if !defined(__ARM_ARCH_7A__) && !defined(__ARM_ARCH_8A)
            cpuid_topology.activeHWThreads = cpu_count(&cpuSet);
#else
            cpuid_topology.activeHWThreads = sysconf(_SC_NPROCESSORS_ONLN);
#endif
        }
        else
        {
            cpuid_topology.activeHWThreads = sysconf(_SC_NPROCESSORS_CONF);
        }
        ret = funcs.init_cpuInfo(cpuSet);
        if (ret < 0)
        {
            errno = ret;
            ERROR_PRINT("Failed to read cpuinfo");
            cpuid_topology.activeHWThreads = 0;
            return ret;
        }
        ret = topology_setName();
        if (ret < 0)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, "Cannot use machine-given CPU name");
        }
        ret = funcs.init_cpuFeatures();
        if (ret < 0)
        {
            errno = ret;
            ERROR_PRINT("Failed to detect CPU features");
            free(cpuid_info.osname);
            memset(&cpuid_info, 0, sizeof(CpuInfo));
            memset(&cpuid_topology, 0, sizeof(CpuTopology));
            return ret;
        }
        ret = funcs.init_nodeTopology(cpuSet);
        if (ret < 0)
        {
            errno = ret;
            ERROR_PRINT("Failed to setup system topology");
            free(cpuid_info.osname);
            free(cpuid_info.features);
            free(cpuid_topology.threadPool);
            memset(&cpuid_info, 0, sizeof(CpuInfo));
            memset(&cpuid_topology, 0, sizeof(CpuTopology));
            return ret;
        }
        uint32_t  activeCount = 0;
        for (size_t i = 0; i < cpuid_topology.numHWThreads; i++)
        {
            if (cpuid_topology.threadPool[i].inCpuSet)
                activeCount++;
        }
        if (activeCount > cpuid_topology.activeHWThreads)
            cpuid_topology.activeHWThreads = activeCount;
        ret = funcs.init_cacheTopology();
        if (ret < 0)
        {
            errno = ret;
            ERROR_PRINT("Failed to setup cache topology");
            free(cpuid_info.osname);
            free(cpuid_topology.threadPool);
            memset(&cpuid_info, 0, sizeof(CpuInfo));
            memset(&cpuid_topology, 0, sizeof(CpuTopology));
            return ret;
        }
        if (cpuid_topology.numCacheLevels == 0)
        {
            CacheLevel* cachePool = NULL;
            switch(cpuid_info.family)
            {
                case ARMV8_FAMILY:
                    switch (cpuid_info.vendor)
                    {
                        case CAVIUM2:
                            switch (cpuid_info.part) {
                                case CAV_THUNDERX2T99:
                                    cachePool = (CacheLevel*) malloc(3 * sizeof(CacheLevel));
                                    for(int i=0;i < 3; i++)
                                    {
                                        cachePool[i].level = caviumTX2_caches[i].level;
                                        cachePool[i].size = caviumTX2_caches[i].size;
                                        cachePool[i].lineSize = caviumTX2_caches[i].lineSize;
                                        cachePool[i].threads = caviumTX2_caches[i].threads;
                                        cachePool[i].inclusive = caviumTX2_caches[i].inclusive;
                                        cachePool[i].sets = caviumTX2_caches[i].sets;
                                        cachePool[i].associativity = caviumTX2_caches[i].associativity;
                                    }
                                    cpuid_topology.cacheLevels = cachePool;
                                    cpuid_topology.numCacheLevels = 3;
                                    break;
                                default:
                                    break;
                            }
                            break;
                        case CAVIUM1:
                            switch (cpuid_info.part) {
                                case CAV_THUNDERX2T99P1:
                                    cachePool = (CacheLevel*) malloc(3 * sizeof(CacheLevel));
                                    for(int i=0;i < 3; i++)
                                    {
                                        cachePool[i].level = caviumTX2_caches[i].level;
                                        cachePool[i].size = caviumTX2_caches[i].size;
                                        cachePool[i].lineSize = caviumTX2_caches[i].lineSize;
                                        cachePool[i].threads = caviumTX2_caches[i].threads;
                                        cachePool[i].inclusive = caviumTX2_caches[i].inclusive;
                                        cachePool[i].sets = caviumTX2_caches[i].sets;
                                        cachePool[i].associativity = caviumTX2_caches[i].associativity;
                                    }
                                    cpuid_topology.cacheLevels = cachePool;
                                    cpuid_topology.numCacheLevels = 3;
                                    break;
                                default:
                                    break;
                            }
                            break;
                        case FUJITSU_ARM:
                            switch(cpuid_info.part) {
                                case FUJITSU_A64FX:
                                    cachePool = (CacheLevel*) malloc(2 * sizeof(CacheLevel));
                                    for(int i=0;i < 2; i++)
                                    {
                                        cachePool[i].level = a64fx_caches[i].level;
                                        cachePool[i].size = a64fx_caches[i].size;
                                        cachePool[i].lineSize = a64fx_caches[i].lineSize;
                                        cachePool[i].threads = a64fx_caches[i].threads;
                                        cachePool[i].inclusive = a64fx_caches[i].inclusive;
                                        cachePool[i].sets = a64fx_caches[i].sets;
                                        cachePool[i].associativity = a64fx_caches[i].associativity;
                                    }
                                    cpuid_topology.cacheLevels = cachePool;
                                    cpuid_topology.numCacheLevels = 2;
                                    break;
                                default:
                                    break;
                            }
                            break;
/*                        case APPLE_M1:*/
/*                            switch(cpuid_info.model) {*/
/*                                case APPLE_M1_STUDIO:*/
/*                                    cachePool = (CacheLevel*) malloc(2 * sizeof(CacheLevel));*/
/*                                    for(int i=0;i < 2; i++)*/
/*                                    {*/
/*                                        cachePool[i].level = apple_m1_caches[i].level;*/
/*                                        cachePool[i].size = apple_m1_caches[i].size;*/
/*                                        cachePool[i].lineSize = apple_m1_caches[i].lineSize;*/
/*                                        cachePool[i].threads = apple_m1_caches[i].threads;*/
/*                                        cachePool[i].inclusive = apple_m1_caches[i].inclusive;*/
/*                                        cachePool[i].sets = apple_m1_caches[i].sets;*/
/*                                        cachePool[i].associativity = apple_m1_caches[i].associativity;*/
/*                                    }*/
/*                                    cpuid_topology.cacheLevels = cachePool;*/
/*                                    cpuid_topology.numCacheLevels = 2;*/
/*                                    break;*/
/*                                default:*/
/*                                    break;*/
/*                            }*/
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
        }
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "Setting up tree");
        topology_setupTree();
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuSet);
    }
    else
    {
        CPU_ZERO(&cpuSet);
        if (getenv("LIKWID_IGNORE_CPUSET") == NULL)
        {
            sched_getaffinity(0, sizeof(cpu_set_t), &cpuSet);
        }
        else
        {
            for (int i = 0; i < sysconf(_SC_NPROCESSORS_CONF); i++)
            {
                if (likwid_cpu_online(i))
                {
                    CPU_SET(i, &cpuSet);
                }
            }
        }
        DEBUG_PRINT(DEBUGLEV_INFO, "Reading topology information from %s", config.topologyCfgFileName);
        ret = readTopologyFile(config.topologyCfgFileName, cpuSet);
        if (ret < 0)
            goto standard_init;
        cpuid_topology.activeHWThreads = 0;
        for (size_t i=0;i<cpuid_topology.numHWThreads;i++)
        {
            if (CPU_ISSET(cpuid_topology.threadPool[i].apicId, &cpuSet))
            {
                cpuid_topology.activeHWThreads++;
                cpuid_topology.threadPool[i].inCpuSet = 1;
            }
        }
        topology_setName();
        topology_setupTree();
    }


    topology_initialized = 1;
    return EXIT_SUCCESS;
}


void
topology_finalize(void)
{
    if (!topology_initialized)
    {
        return;
    }
    if (cpuid_info.features != NULL)
    {
        free(cpuid_info.features);
        cpuid_info.features = NULL;
    }
    if (cpuid_info.osname != NULL)
    {
        free(cpuid_info.osname);
        cpuid_info.osname = NULL;
    }
    if (cpuid_topology.cacheLevels != NULL)
    {
        free(cpuid_topology.cacheLevels);
        cpuid_topology.cacheLevels = NULL;
    }
    if (cpuid_topology.threadPool != NULL)
    {
        free(cpuid_topology.threadPool);
        cpuid_topology.threadPool = NULL;
    }
    if (cpuid_topology.topologyTree != NULL)
    {
        tree_destroy(cpuid_topology.topologyTree);
        cpuid_topology.topologyTree = NULL;
    }
    if (topology_funcs.close_topology != NULL)
    {
        topology_funcs.close_topology();
    }
    cpuid_info.family = 0;
    cpuid_info.model = 0;
    cpuid_info.stepping = 0;
    cpuid_info.clock = 0;
    cpuid_info.turbo = 0;
    cpuid_info.name = NULL;
    cpuid_info.short_name = NULL;
    cpuid_info.isIntel = 0;
    cpuid_info.supportUncore = 0;
    cpuid_info.featureFlags = 0;
    cpuid_info.perf_version = 0;
    cpuid_info.perf_num_ctr = 0;
    cpuid_info.perf_width_ctr = 0;
    cpuid_info.perf_num_fixed_ctr = 0;

    cpuid_topology.numHWThreads = 0;
    cpuid_topology.activeHWThreads = 0;
    cpuid_topology.numSockets = 0;
    cpuid_topology.numCoresPerSocket = 0;
    cpuid_topology.numThreadsPerCore = 0;
    cpuid_topology.numCacheLevels = 0;

    topology_initialized = 0;
}

void
print_supportedCPUs (void)
{
    printf("Supported Intel processors:\n");
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
    printf("\t%s\n",xeon_phi_string);
    printf("\t%s\n",broadwell_str);
    printf("\t%s\n",broadwell_d_str);
    printf("\t%s\n",broadwell_ep_str);
    printf("\t%s\n",atom_goldmont_str);
    printf("\t%s\n",xeon_phi2_string);
    printf("\t%s\n",skylake_str);
    printf("\t%s\n",skylakeX_str);
    printf("\t%s\n",xeon_phi3_string);
    printf("\t%s\n",kabylake_str);
    printf("\t%s\n",coffeelake_str);
    printf("\t%s\n",cascadelakeX_str);
    printf("\t%s\n",tigerlake_str);
    printf("\t%s\n",icelake_str);
    printf("\t%s\n",rocketlake_str);
    printf("\t%s\n",icelakesp_str);
    printf("\t%s\n",sapphire_rapids_str);
    printf("\t%s\n",emerald_rapids_str);
    printf("\t%s\n",granite_rapids_str);
    printf("\t%s\n",sierra_forrest_str);
    printf("\n");
    printf("Supported AMD processors:\n");
    printf("\t%s\n",opteron_sc_str);
    printf("\t%s\n",opteron_dc_e_str);
    printf("\t%s\n",opteron_dc_f_str);
    printf("\t%s\n",barcelona_str);
    printf("\t%s\n",shanghai_str);
    printf("\t%s\n",istanbul_str);
    printf("\t%s\n",magnycours_str);
    printf("\t%s\n",interlagos_str);
    printf("\t%s\n",kabini_str);
    printf("\t%s\n",amd_zen_str);
    printf("\t%s\n",amd_zen2_str);
    printf("\t%s\n",amd_zen3_str);
    printf("\t%s\n",amd_zen4_str);
    printf("\t%s\n",amd_zen4c_str);
    printf("\n");
    printf("Supported ARMv8 processors:\n");
    printf("\t%s\n",arm_cortex_a53);
    printf("\t%s\n",arm_cortex_a57);
    printf("\t%s\n",arm_cortex_a76);
    printf("\t%s\n",cavium_thunderx_str);
    printf("\t%s\n",cavium_thunderx2t99_str);
    printf("\t%s\n",fujitsu_a64fx);
    printf("\t%s\n",arm_neoverse_n1);
    printf("\t%s\n",arm_neoverse_v1);
    printf("\t%s\n",arm_huawei_tsv110);
    printf("\t%s\n",apple_m1_studio);
    printf("\t%s\n",arm_nvidia_grace);
    printf("\n");
    printf("Supported ARMv7 processors:\n");
    printf("\t%s\n",armv7l_str);
    printf("\n");
    printf("Supported POWER processors:\n");
    printf("\t%s\n",power8_str);
    printf("\t%s\n",power9_str);
    printf("\n");
}

CpuTopology_t
get_cpuTopology(void)
{
    return &cpuid_topology;
}

CpuInfo_t
get_cpuInfo(void)
{
    return &cpuid_info;
}

NumaTopology_t
get_numaTopology(void)
{
    return &numa_info;
}
