#include <sys/types.h>
#include <unistd.h>
#include <sched.h>

#include <topology.h>
#include <error.h>
#include <tree.h>
#include <bitUtil.h>
#include <strUtil.h>


static volatile int init = 0;
CpuInfo cpuid_info;
CpuTopology cpuid_topology;



static char* pentium_m_b_str = "Intel Pentium M Banias processor";
static char* pentium_m_d_str = "Intel Pentium M Dothan processor";
static char* core_duo_str = "Intel Core Duo processor";
static char* core_2a_str = "Intel Core 2 65nm processor";
static char* core_2b_str = "Intel Core 2 45nm processor";
static char* atom_45_str = "Intel Atom 45nm processor";
static char* atom_32_str = "Intel Atom 32nm processor";
static char* atom_22_str = "Intel Atom 22nm processor";
static char* nehalem_bloom_str = "Intel Core Bloomfield processor";
static char* nehalem_lynn_str = "Intel Core Lynnfield processor";
static char* nehalem_west_str = "Intel Core Westmere processor";
static char* sandybridge_str = "Intel Core SandyBridge processor";
static char* ivybridge_str = "Intel Core IvyBridge processor";
static char* ivybridge_ep_str = "Intel Core IvyBridge EP processor";
static char* sandybridge_ep_str = "Intel Core SandyBridge EP processor";
static char* haswell_str = "Intel Core Haswell processor";
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
        }
    }
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
                    break;

                case PENTIUM_M_DOTHAN:
                    cpuid_info.name = pentium_m_d_str;
                    break;

                case CORE_DUO:
                    cpuid_info.name = core_duo_str;
                    break;

                case CORE2_65:
                    cpuid_info.name = core_2a_str;
                    break;

                case CORE2_45:
                    cpuid_info.name = core_2b_str;
                    break;

                case NEHALEM_BLOOMFIELD:
                    cpuid_info.name = nehalem_bloom_str;
                    break;

                case NEHALEM_LYNNFIELD:
                    cpuid_info.name = nehalem_lynn_str;
                    break;

                case NEHALEM_WESTMERE_M:

                case NEHALEM_WESTMERE:
                    cpuid_info.name = nehalem_west_str;
                    break;

                case SANDYBRIDGE:
                    cpuid_info.name = sandybridge_str;
                    break;

                case SANDYBRIDGE_EP:
                    cpuid_info.name = sandybridge_ep_str;
                    break;

                case IVYBRIDGE:
                    cpuid_info.name = ivybridge_str;
                    break;

                case IVYBRIDGE_EP:
                    cpuid_info.name = ivybridge_ep_str;
                    break;

                case HASWELL:

                case HASWELL_EX:

                case HASWELL_M1:

                case HASWELL_M2:
                    cpuid_info.name = haswell_str;
                    break;

                case NEHALEM_EX:
                    cpuid_info.name = nehalem_ex_str;
                    break;

                case WESTMERE_EX:
                    cpuid_info.name = westmere_ex_str;
                    break;

                case XEON_MP:
                    cpuid_info.name = xeon_mp_string;
                    break;

                case ATOM_45:

                case ATOM:
                    cpuid_info.name = atom_45_str;
                    break;

                case ATOM_32:
                    cpuid_info.name = atom_32_str;
                    break;

                case ATOM_22:
                    cpuid_info.name = atom_22_str;
                    break;

                default:
                    cpuid_info.name = unknown_intel_str;
                    break;
            }
            break;

        case MIC_FAMILY:
            switch ( cpuid_info.model ) 
            {
                case XEON_PHI:
                    cpuid_info.name = xeon_phi_string;
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
            break;

        case K15_FAMILY:
            cpuid_info.name = interlagos_str;
            break;

        case K16_FAMILY:
            cpuid_info.name = kabini_str;
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
    .init_cpuFeatures = hwloc_init_cpuFeatures,
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

        if (!tree_nodeExists(currentNode, i))
        {
            /*
               printf("WARNING: Thread already exists!\n");
               */
            tree_insertNode(currentNode, i);
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
    FILE *file;
    char *filepath = TOSTRING(CFGFILE);
    const struct topology_functions funcs = topology_funcs;
    
    if (init)
    {
        return EXIT_SUCCESS;
    }
    init = 1;
    
    
    funcs.init_cpuInfo();
    topology_setName();
    funcs.init_cpuFeatures();
    if (access(filepath, R_OK))
    {
        cpu_set_t cpuSet;
        CPU_ZERO(&cpuSet);
        sched_getaffinity(0,sizeof(cpu_set_t), &cpuSet);
        funcs.init_nodeTopology();
        topology_setupTree();
        funcs.init_cacheTopology();
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuSet);
    }
    else
    {
        file = fopen(filepath, "rb");
        funcs.init_fileTopology(file);
        fclose(file);
    }
    
    return EXIT_SUCCESS;
}


void topology_finalize(void)
{
    free(cpuid_info.features);
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
    printf("\t%s\n\n",xeon_phi_string);

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


int cpuid_isInCpuset(void)
{
    int pos = 0;
    int ret = 0;
    bstring  cpulist;
    bstring grepString = bformat("Cpus_allowed_list:");
    bstring filename = bformat("/proc/%d/status",getpid());
    FILE* fp = fopen(bdata(filename),"r");

    if (fp == NULL)
    {
        bdestroy(filename);
        bdestroy(grepString);
        return 0;
    } 
    else
    {
        uint32_t tmpThreads[MAX_NUM_THREADS];
        bstring src = bread ((bNread) fread, fp);
        if ((pos = binstr(src,0,grepString)) != BSTR_ERR)
        {
            int end = bstrchrp(src, 10, pos);
            pos = pos+blength(grepString);
            cpulist = bmidstr(src,pos, end-pos);
            btrimws(cpulist);

            if (bstr_to_cpuset_physical(tmpThreads, cpulist) < cpuid_topology.numHWThreads)
            {
                ret = 1;
                goto freeStrings;
            }
            else
            {
                ret = 0;
                goto freeStrings;
            }
        }
        return 0;
    }
freeStrings:
    bdestroy(filename);  
    bdestroy(grepString);
    bdestroy(cpulist);
    return ret;
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

