/*
 * =======================================================================================
 *
 *      Filename:  cpuid.c
 *
 *      Description:  Implementation of cpuid module.
 *                  Provides API to extract cpuid info on x86 processors.
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
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>

#include <error.h>
#include <cpuid.h>
#include <tree.h>
#include <bitUtil.h>
#include <strUtil.h>
#ifdef LIKWID_USE_HWLOC
#include "cpuid-hwloc.h"
#define CPU_FEATURES 1
#endif
#include <tlb-info.h>

/* #####   EXPORTED VARIABLES   ########################################### */

CpuInfo cpuid_info;
CpuTopology cpuid_topology;


/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int largest_function = 0;

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

static volatile int init = 0;
static uint32_t eax, ebx, ecx, edx;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static void initTopology(FILE* file)
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

static uint32_t amdGetAssociativity(uint32_t flag)
{
    uint32_t asso= 0;

    switch ( flag )
    {
        case 0x0:
            asso = 0;
            break;

        case 0x1:
            asso = 1;
            break;

        case 0x2:
            asso = 2;
            break;

        case 0x4:
            asso = 4;
            break;

        case 0x6:
            asso = 8;
            break;

        case 0x8:
            asso = 16;
            break;

        case 0xA:
            asso = 32;
            break;

        case 0xB:
            asso = 48;
            break;

        case 0xC:
            asso = 64;
            break;

        case 0xD:
            asso = 96;
            break;

        case 0xE:
            asso = 128;
            break;

        case 0xF:
            asso = 0;
            break;

        default:
            break;
    }
    return asso;

}

static int intelCpuidFunc_4(CacheLevel** cachePool)
{
    int i;
    int level=0;
    int maxNumLevels=0;
    uint32_t valid=1;
    CacheLevel* pool;

    while (valid)
    {
        eax = 0x04;
        ecx = level;
        CPUID;
        valid = extractBitField(eax,5,0);
        if (!valid)
        {
            break;
        }
        level++;
    }

    maxNumLevels = level;
    *cachePool = (CacheLevel*) malloc(maxNumLevels * sizeof(CacheLevel));
    pool = *cachePool;

    for (i=0; i < maxNumLevels; i++) 
    {
        eax = 0x04;
        ecx = i;
        CPUID;

        pool[i].level = extractBitField(eax,3,5);
        pool[i].type = (CacheType) extractBitField(eax,5,0);
        pool[i].associativity = extractBitField(ebx,8,22)+1;
        pool[i].sets = ecx+1;
        pool[i].lineSize = extractBitField(ebx,12,0)+1;
        pool[i].size = pool[i].sets *
            pool[i].associativity *
            pool[i].lineSize;
        pool[i].threads = extractBitField(eax,10,14)+1;

        /* WORKAROUND cpuid reports wrong number of threads on SMT processor with SMT
         * turned off */
        if (i < 3)
        {
            if ((cpuid_info.model == NEHALEM_BLOOMFIELD) ||
                    (cpuid_info.model == NEHALEM_LYNNFIELD) ||
                    (cpuid_info.model == NEHALEM_WESTMERE) ||
                    (cpuid_info.model == NEHALEM_WESTMERE_M) ||
                    (cpuid_info.model == SANDYBRIDGE) ||
                    (cpuid_info.model == SANDYBRIDGE_EP) ||
                    (cpuid_info.model == IVYBRIDGE) ||
                    (cpuid_info.model == IVYBRIDGE_EP) ||
                    (cpuid_info.model == HASWELL) ||
                    (cpuid_info.model == HASWELL_EX) ||
                    (cpuid_info.model == HASWELL_M1) ||
                    (cpuid_info.model == HASWELL_M2) ||
                    (cpuid_info.model == WESTMERE_EX) ||
                    (cpuid_info.model == NEHALEM_EX))
            {
                if (cpuid_topology.numThreadsPerCore == 1)
                {
                    pool[i].threads = 1;
                }
            }
        }

        /* :WORKAROUND:08/13/2009 08:34:15 AM:jt: For L3 caches the value is sometimes 
         * too large in here. Ask Intel what is wrong here!
         * Limit threads per Socket then to the maximum possible value.*/
        if(pool[i].threads > (int)
                (cpuid_topology.numCoresPerSocket*
                 cpuid_topology.numThreadsPerCore))
        {
            pool[i].threads = cpuid_topology.numCoresPerSocket*
                cpuid_topology.numThreadsPerCore;
        }
        pool[i].inclusive = edx&0x2;
    }

    return maxNumLevels;
}

static int readCacheInclusive(int level)
{
    eax = 0x04;
    ecx = level;
    CPUID;
    return edx & 0x2;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */
void cpuid_printTlbTopology()
{
	int i;
	uint32_t loop = 1;
	int isIntel = 1;
	
	eax = 0x00;
	CPUID;

	largest_function = eax;
	if (ebx == 0x68747541U)
	{
	    isIntel = 0;
	}
	
	if (isIntel)
	{
		eax = 0x02;
		CPUID;
	
	
		loop = extractBitField(eax,8,0);
		for(i=1;i<loop;i++)
		{
			eax = 0x02;
			CPUID;
		}

		for(i=8;i<32;i+=8)
		{
			if (extractBitField(eax,8,i) != 0x0)
			{
				if (intel_tlb_info[extractBitField(eax,8,i)])
					printf("%s\n",intel_tlb_info[extractBitField(eax,8,i)]);
			}
		}
		for(i=0;i<32;i+=8)
		{
			if (extractBitField(eax,8,i) != 0x0)
			{
				if (intel_tlb_info[extractBitField(ebx,8,i)])
					printf("%s\n",intel_tlb_info[extractBitField(ebx,8,i)]);
			}
		}
		for(i=0;i<32;i+=8)
		{
			if (extractBitField(eax,8,i) != 0x0)
			{
				if (intel_tlb_info[extractBitField(ecx,8,i)])
					printf("%s\n",intel_tlb_info[extractBitField(ecx,8,i)]);
			}
		}
		for(i=0;i<32;i+=8)
		{
			if (extractBitField(eax,8,i) != 0x0)
			{
				if (intel_tlb_info[extractBitField(edx,8,i)])
					printf("%s\n",intel_tlb_info[extractBitField(edx,8,i)]);
			}
		}
	}
	else
	{
		eax = 0x80000005;
		CPUID;
		printf("L1DTlb2and4MAssoc: 0x%x\n",extractBitField(eax,8,24));
		printf("L1DTlb2and4MSize: %d entries for 2MB pages\n",(uint32_t)extractBitField(eax,8,16));
		printf("L1ITlb2and4MAssoc: 0x%x\n",extractBitField(eax,8,8));
		printf("L1ITlb2and4MSize: %d entries for 2MB pages\n",(uint32_t)extractBitField(eax,8,0));
		ebx = 0x80000005;
		CPUID;
		printf("L1DTlb4KAssoc: 0x%x\n",extractBitField(ebx,8,24));
		printf("L1DTlb4KSize: 0x%x\n",extractBitField(ebx,8,16));
		printf("L1ITlb4KAssoc: 0x%x\n",extractBitField(ebx,8,8));
		printf("L1ITlb4KSize: 0x%x\n",extractBitField(ebx,8,0));
		eax = 0x80000006;
		CPUID;
		printf("L2DTlb2and4MAssoc: 0x%x\n",extractBitField(eax,4,24));
		printf("L2DTlb2and4MAssoc_c: %d\n",amdGetAssociativity(extractBitField(eax,4,24)));
		printf("L2DTlb2and4MSize: 0x%x\n",extractBitField(eax,12,16));
		printf("L2ITlb2and4MAssoc: 0x%x\n",extractBitField(eax,4,12));
		printf("L2ITlb2and4MAssoc_c: %d\n",amdGetAssociativity(extractBitField(eax,4,12)));
		printf("L2ITlb2and4MSize: 0x%x\n",extractBitField(eax,12,0));
		ebx = 0x80000006;
		CPUID;
		printf("L2DTlb4KAssoc: 0x%x\n",extractBitField(eax,4,24));
		printf("L2DTlb4KAssoc_c: %d\n",amdGetAssociativity(extractBitField(eax,4,24)));
		printf("L2DTlb4KSize: 0x%x\n",extractBitField(eax,12,16));
		printf("L2ITlb4KAssoc: 0x%x\n",extractBitField(eax,4,12));
		printf("L2ITlb4KAssoc_c: %d\n",amdGetAssociativity(extractBitField(eax,4,12)));
		printf("L2ITlb4KSize: 0x%x\n",extractBitField(eax,12,0));
	}		
	return;
}

int cpuid_init (void)
{
    int isIntel = 1;
#ifdef LIKWID_USE_HWLOC
    int use_cpuid = 0;
    int i;
    hwloc_obj_t obj;
    /*int flags;*/
#else
	int use_cpuid = 1;
#endif
    /* FIXME: Race condition??? */
    if (init) return EXIT_SUCCESS;
    init =1;
#ifdef LIKWID_USE_HWLOC	
	if (!use_cpuid) 
    {
        hwloc_topology_init(&hwloc_topology);
        
        /*flags = hwloc_topology_get_flags(hwloc_topology);
        hwloc_topology_set_flags(hwloc_topology, flags | HWLOC_TOPOLOGY_FLAG_ICACHES);*/
        
        hwloc_topology_load(hwloc_topology);
        obj = hwloc_get_obj_by_type(hwloc_topology, HWLOC_OBJ_SOCKET, 0);
        if (!obj)
        	goto model_family_cpuid;
	    cpuid_info.model = 0;
	    cpuid_info.family = 0;
	    
	    for(i=0;i<obj->infos_count;i++)
	    {
	        if (strcmp(obj->infos[i].name ,"CPUModelNumber") == 0)
	            cpuid_info.model = atoi(hwloc_obj_get_info_by_name(obj, "CPUModelNumber"));
	        if (strcmp(obj->infos[i].name, "CPUFamilyNumber") == 0)
	            cpuid_info.family = atoi(hwloc_obj_get_info_by_name(obj, "CPUFamilyNumber"));
	        if (strcmp(obj->infos[i].name, "CPUVendor") == 0 && 
	                strcmp(hwloc_obj_get_info_by_name(obj, "CPUVendor"), "AuthenticAMD") == 0)
	            isIntel = 0;
	        if (strcmp(obj->infos[i].name, "CPUVendor") == 0 && 
	                strcmp(hwloc_obj_get_info_by_name(obj, "CPUVendor"), "GenuineIntel") == 0)
	            isIntel = 1;
	        /* Currently no CPU stepping */
	    }
	
        if (!cpuid_info.model && !cpuid_info.family)
            use_cpuid = 1;
    }
#endif    
    if (use_cpuid)
    {
model_family_cpuid:
		eax = 0x00;
		CPUID;

		largest_function = eax;
		if (ebx == 0x68747541U)
		{
		    isIntel = 0;
		}

		eax = 0x01;
		CPUID;
		cpuid_info.family = ((eax>>8)&0xFU) + ((eax>>20)&0xFFU);
		cpuid_info.model = (((eax>>16)&0xFU)<<4) + ((eax>>4)&0xFU);
		cpuid_info.stepping =  (eax&0xFU);
	}

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

            if (isIntel)
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
#ifdef CPU_FEATURES
    cpuid_info.featureFlags = 0;
    cpuid_info.features = (char*) malloc(200*sizeof(char));
    if (ecx & (1<<0))
    {
        strcat(cpuid_info.features, "SSE3 ");
        cpuid_info.featureFlags |= (1<<SSE3);
    }
    if (ecx & (1<<3))
    {
        strcat(cpuid_info.features, "MONITOR ");
        cpuid_info.featureFlags |= (1<<MONITOR);
    }
    if (ecx & (1<<5))
    {
        strcat(cpuid_info.features, "VMX ");
        cpuid_info.featureFlags |= (1<<VMX);
    }
    if (ecx & (1<<7))
    {
        strcat(cpuid_info.features, "EIST ");
        cpuid_info.featureFlags |= (1<<EIST);
    }
    if (ecx & (1<<8))
    {
        strcat(cpuid_info.features, "TM2 ");
        cpuid_info.featureFlags |= (1<<TM2);
    }
    if (ecx & (1<<9))
    {
        strcat(cpuid_info.features, "SSSE3 ");
        cpuid_info.featureFlags |= (1<<SSSE3);
    }
    if (ecx & (1<<12))
    {
        strcat(cpuid_info.features, "FMA ");
        cpuid_info.featureFlags |= (1<<FMA);
    }
    if (ecx & (1<<19))
    {
        strcat(cpuid_info.features, "SSE4.1 ");
        cpuid_info.featureFlags |= (1<<SSE41);
    }
    if (ecx & (1<<20))
    {
        strcat(cpuid_info.features, "SSE4.2 ");
        cpuid_info.featureFlags |= (1<<SSE42);
    }
    if (ecx & (1<<25))
    {
        strcat(cpuid_info.features, "AES ");
        cpuid_info.featureFlags |= (1<<AES);
    }
    if (ecx & (1<<28))
    {
        strcat(cpuid_info.features, "AVX ");
        cpuid_info.featureFlags |= (1<<AVX);
    }
    if (ecx & (1<<30))
    {
        strcat(cpuid_info.features, "RDRAND ");
        cpuid_info.featureFlags |= (1<<RDRAND);
    }

    if (edx & (1<<22))
    {
        strcpy(cpuid_info.features, "ACPI ");
        cpuid_info.featureFlags |= (1<<ACPI);
    }
    if (edx & (1<<23))
    {
        strcpy(cpuid_info.features, "MMX ");
        cpuid_info.featureFlags |= (1<<MMX);
    }
    if (edx & (1<<25))
    {
        strcpy(cpuid_info.features, "SSE ");
        cpuid_info.featureFlags |= (1<<SSE);
    }
    if (edx & (1<<26))
    {
        strcat(cpuid_info.features, "SSE2 ");
        cpuid_info.featureFlags |= (1<<SSE2);
    }
    if (edx & (1<<29))
    {
        strcat(cpuid_info.features, "TM ");
        cpuid_info.featureFlags |= (1<<TM);
    }

    eax = 0x80000001;
    CPUID;
    if (edx & (1<<27))
    {
        strcat(cpuid_info.features, "RDTSCP ");
        cpuid_info.featureFlags |= (1<<RDTSCP);
    }

    cpuid_info.perf_version   =  0;
    if( cpuid_info.family == P6_FAMILY && 0x0A <= largest_function)
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
#endif
    FILE *file;
    char *filepath = TOSTRING(CFGFILE);

    if ((file = fopen(filepath, "rb")) != NULL) 
    {
        printf("Read config from file\n");
        initTopology(file);
        fclose(file);
    }
    else
    {
        cpuid_topology.numHWThreads = sysconf(_SC_NPROCESSORS_CONF);
        cpu_set_t cpuSet;
        CPU_ZERO(&cpuSet);
        sched_getaffinity(0,sizeof(cpu_set_t), &cpuSet);
        cpuid_initTopology();
        cpuid_initCacheTopology();

        /* restore affinity mask of process */
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuSet);
    }

    return EXIT_SUCCESS;
}

void cpuid_print (void)
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





#define freeStrings  \
    bdestroy(filename);  \
bdestroy(grepString); \
bdestroy(cpulist)


int cpuid_isInCpuset(void)
{
    int pos = 0;
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
        bstring  cpulist;
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
                freeStrings;
                return 1;
            }
            else
            {
                freeStrings;
                return 0;
            }
        }
        return 0;
    }
}

void cpuid_initTopology(void)
{
    uint32_t apicId;
    uint32_t bitField;
    int level;
    int prevOffset = 0;
    int currOffset = 0;
    cpu_set_t set;
    HWThread* hwThreadPool;
    int hasBLeaf = 0;
    int maxNumLogicalProcs;
    int maxNumLogicalProcsPerCore;
    int maxNumCores;
    TreeNode* currentNode;
    int width;
#ifdef LIKWID_USE_HWLOC
	int use_cpuid = 0;
	hwloc_obj_t obj;
	int realThreadId;
	int sibling;
#else
	int use_cpuid = 1;
#endif

	hwThreadPool = (HWThread*) malloc(cpuid_topology.numHWThreads * sizeof(HWThread));
    tree_init(&cpuid_topology.topologyTree, 0);
#ifdef LIKWID_USE_HWLOC    
    if (!hwloc_topology)
    {
        printf("hwloc_topology not initialized, use CPUID\n");
        use_cpuid = 1;
    }
    
    if (!use_cpuid)
    {
        maxNumLogicalProcs = hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_PU);
        maxNumCores = hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_CORE);
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
            while (obj->type != HWLOC_OBJ_SOCKET) {
                obj = obj->parent;
            }
            hwThreadPool[realThreadId].packageId = obj->logical_index;
            
        }
    }
#endif
	if (use_cpuid)
    {
		/* check if 0x0B cpuid leaf is supported */
		if (largest_function >= 0x0B)
		{
		    eax = 0x0B;
		    ecx = 0;
		    CPUID;

		    if (ebx)
		    {
		        hasBLeaf = 1;
		    }
		}

		if (hasBLeaf)
		{
		    for (uint32_t i=0; i < cpuid_topology.numHWThreads; i++)
		    {

		        CPU_ZERO(&set);
		        CPU_SET(i,&set);
		        sched_setaffinity(0, sizeof(cpu_set_t), &set);
		        eax = 0x0B;
		        ecx = 0;
		        CPUID;
		        apicId = edx;
		        hwThreadPool[i].apicId = apicId;

		        for (level=0; level < 3; level++)
		        {
		            eax = 0x0B;
		            ecx = level;
		            CPUID;
		            currOffset = eax&0xFU;

		            switch ( level ) {
		                case 0:  /* SMT thread */
		                    bitField = extractBitField(apicId,
		                            currOffset,
		                            0);
		                    hwThreadPool[i].threadId = bitField;
		                    break;

		                case 1:  /* Core */
		                    bitField = extractBitField(apicId,
		                            currOffset-prevOffset,
		                            prevOffset);
		                    hwThreadPool[i].coreId = bitField;
		                    break;

		                case 2:  /* Package */
		                    bitField = extractBitField(apicId,
		                            32-prevOffset,
		                            prevOffset);
		                    hwThreadPool[i].packageId = bitField;
		                    break;

		            }
		            prevOffset = currOffset;
		        }
		    }
		}
		else
		{
		    switch ( cpuid_info.family )
		    {

		        case MIC_FAMILY:

		        case P6_FAMILY:
		            eax = 0x01;
		            CPUID;
		            maxNumLogicalProcs = extractBitField(ebx,8,16);

		            /* Check number of cores per package */
		            eax = 0x04;
		            ecx = 0;
		            CPUID;
		            maxNumCores = extractBitField(eax,6,26)+1;

		            maxNumLogicalProcsPerCore = maxNumLogicalProcs/maxNumCores;

		            for (uint32_t i=0; i<  cpuid_topology.numHWThreads; i++)
		            {
		                CPU_ZERO(&set);
		                CPU_SET(i,&set);
		                sched_setaffinity(0, sizeof(cpu_set_t), &set);

		                eax = 0x01;
		                CPUID;
		                hwThreadPool[i].apicId = extractBitField(ebx,8,24);

		                /* ThreadId is extracted from th apicId using the bit width
		                 * of the number of logical processors
		                 * */
		                hwThreadPool[i].threadId =
		                    extractBitField(hwThreadPool[i].apicId,
		                            getBitFieldWidth(maxNumLogicalProcsPerCore),0); 

		                /* CoreId is extracted from th apicId using the bitWidth 
		                 * of the number of logical processors as offset and the
		                 * bit width of the number of cores as width
		                 * */
		                hwThreadPool[i].coreId =
		                    extractBitField(hwThreadPool[i].apicId,
		                            getBitFieldWidth(maxNumCores),
		                            getBitFieldWidth(maxNumLogicalProcsPerCore)); 

		                hwThreadPool[i].packageId =
		                    extractBitField(hwThreadPool[i].apicId,
		                            8-getBitFieldWidth(maxNumLogicalProcs),
		                            getBitFieldWidth(maxNumLogicalProcs)); 
		            }
		            break;

		        case K8_FAMILY:
		            /* AMD Bios manual Rev. 2.28 section 3.1
		             * Legacy method */
		            /*FIXME: This is a bit of a hack */

		            maxNumLogicalProcsPerCore = 1;
		            maxNumLogicalProcs = 1;

		            eax = 0x80000008;
		            CPUID;

		            maxNumCores =  extractBitField(ecx,8,0)+1;

		            for (uint32_t i=0; i<  cpuid_topology.numHWThreads; i++)
		            {
		                CPU_ZERO(&set);
		                CPU_SET(i,&set);
		                sched_setaffinity(0, sizeof(cpu_set_t), &set);

		                eax = 0x01;
		                CPUID;
		                hwThreadPool[i].apicId = extractBitField(ebx,8,24);

		                /* ThreadId is extracted from th apicId using the bit width
		                 * of the number of logical processors
		                 * */
		                hwThreadPool[i].threadId =
		                    extractBitField(hwThreadPool[i].apicId,
		                            getBitFieldWidth(maxNumLogicalProcsPerCore),0); 

		                /* CoreId is extracted from th apicId using the bitWidth 
		                 * of the number of logical processors as offset and the
		                 * bit width of the number of cores as width
		                 * */
		                hwThreadPool[i].coreId =
		                    extractBitField(hwThreadPool[i].apicId,
		                            getBitFieldWidth(maxNumCores),
		                            0); 

		                hwThreadPool[i].packageId =
		                    extractBitField(hwThreadPool[i].apicId,
		                            8-getBitFieldWidth(maxNumCores),
		                            getBitFieldWidth(maxNumCores)); 
		            }
		            break;

		        case K16_FAMILY:

		        case K15_FAMILY:

		        case K10_FAMILY:
		            /* AMD Bios manual Rev. 2.28 section 3.2
		             * Extended method */
		            eax = 0x80000008;
		            CPUID;

		            width =  extractBitField(ecx,4,12);

		            if (width == 0)
		            {
		                width =  extractBitField(ecx,8,0)+1;
		            }

		            eax = 0x01;
		            CPUID;
		            maxNumLogicalProcs =  extractBitField(ebx,8,16);
		            maxNumCores = extractBitField(ecx,8,0)+1;


		            for (uint32_t i=0; i<  cpuid_topology.numHWThreads; i++)
		            {
		                CPU_ZERO(&set);
		                CPU_SET(i,&set);
		                sched_setaffinity(0, sizeof(cpu_set_t), &set);

		                eax = 0x01;
		                CPUID;
		                hwThreadPool[i].apicId = extractBitField(ebx,8,24);
		                /* AMD only knows cores */
		                hwThreadPool[i].threadId = 0;

		                hwThreadPool[i].coreId =
		                    extractBitField(hwThreadPool[i].apicId,
		                            width, 0); 
		                hwThreadPool[i].packageId =
		                    extractBitField(hwThreadPool[i].apicId,
		                            (8-width), width); 
		            }

		            break;
		    }
		}
	}

    for (uint32_t i=0; i<  cpuid_topology.numHWThreads; i++)
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


    cpuid_topology.threadPool = hwThreadPool;
    cpuid_topology.numSockets = tree_countChildren(cpuid_topology.topologyTree);
    currentNode = tree_getChildNode(cpuid_topology.topologyTree);
    cpuid_topology.numCoresPerSocket = tree_countChildren(currentNode);
    currentNode = tree_getChildNode(currentNode);
    cpuid_topology.numThreadsPerCore = tree_countChildren(currentNode);
    
    
}

void cpuid_initCacheTopology()
{
    int maxNumLevels=0;
    int id=0;
    CacheLevel* cachePool = NULL;
    CacheType type = DATACACHE;
#ifdef LIKWID_USE_HWLOC
	hwloc_obj_t obj;
    int use_cpuid = 0;
    int depth;
    int d;
#else
	int use_cpuid = 1;
#endif

#ifdef LIKWID_USE_HWLOC	
	if (!use_cpuid && !hwloc_topology)
    {
        printf("hwloc_topology not initialized, use CPUID\n");
        use_cpuid = 1;
    }
	if (!use_cpuid && hwloc_topology)
    {
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
    }
#endif    
    if (use_cpuid)
    {
		switch ( cpuid_info.family ) 
		{
		    case MIC_FAMILY:

		    case P6_FAMILY:

		        if (largest_function >= 4)
		        {
		            maxNumLevels = intelCpuidFunc_4(&cachePool);
		        }
		        else
		        {
		            //				intelCpuidFunc_2(&cachePool);
		        }

		        break;

		    case K8_FAMILY:
		        maxNumLevels = 2;
		        cachePool = (CacheLevel*) malloc(maxNumLevels * sizeof(CacheLevel));

		        eax = 0x80000005;
		        CPUID;
		        cachePool[0].level = 1;
		        cachePool[0].type = DATACACHE;
		        cachePool[0].associativity = extractBitField(ecx,8,16);
		        cachePool[0].lineSize = extractBitField(ecx,8,0);
		        cachePool[0].size =  extractBitField(ecx,8,24) * 1024;
		        if ((cachePool[0].associativity * cachePool[0].lineSize) != 0)
		        {
		            cachePool[0].sets = cachePool[0].size/
		                (cachePool[0].associativity * cachePool[0].lineSize);
		        }
		        cachePool[0].threads = 1;
		        cachePool[0].inclusive = 1;

		        eax = 0x80000006;
		        CPUID;
		        cachePool[1].level = 2;
		        cachePool[1].type = UNIFIEDCACHE;
		        cachePool[1].associativity = 
		            amdGetAssociativity(extractBitField(ecx,4,12));
		        cachePool[1].lineSize = extractBitField(ecx,8,0);
		        cachePool[1].size =  extractBitField(ecx,16,16) * 1024;
		        if ((cachePool[0].associativity * cachePool[0].lineSize) != 0)
		        {
		            cachePool[1].sets = cachePool[1].size/
		                (cachePool[1].associativity * cachePool[1].lineSize);
		        }
		        cachePool[1].threads = 1;
		        cachePool[1].inclusive = 1;

		        break;


		    case K10_FAMILY:
		        /* FIXME: Adds one level for the instruction cache on Intel
		         * This fixes the level for the cores
		         */
		        maxNumLevels = 3;
		        cachePool = (CacheLevel*) malloc(maxNumLevels * sizeof(CacheLevel));

		        eax = 0x80000005;
		        CPUID;
		        cachePool[0].level = 1;
		        cachePool[0].type = DATACACHE;
		        cachePool[0].associativity = extractBitField(ecx,8,16);
		        cachePool[0].lineSize = extractBitField(ecx,8,0);
		        cachePool[0].size =  extractBitField(ecx,8,24) * 1024;
		        if ((cachePool[0].associativity * cachePool[0].lineSize) != 0)
		        {
		            cachePool[0].sets = cachePool[0].size/
		                (cachePool[0].associativity * cachePool[0].lineSize);
		        }
		        cachePool[0].threads = 1;
		        cachePool[0].inclusive = 1;

		        eax = 0x80000006;
		        CPUID;
		        cachePool[1].level = 2;
		        cachePool[1].type = UNIFIEDCACHE;
		        cachePool[1].associativity = 
		            amdGetAssociativity(extractBitField(ecx,4,12));
		        cachePool[1].lineSize = extractBitField(ecx,8,0);
		        cachePool[1].size =  extractBitField(ecx,16,16) * 1024;
		        if ((cachePool[0].associativity * cachePool[0].lineSize) != 0)
		        {
		            cachePool[1].sets = cachePool[1].size/
		                (cachePool[1].associativity * cachePool[1].lineSize);
		        }
		        cachePool[1].threads = 1;
		        cachePool[1].inclusive = 1;

		        cachePool[2].level = 3;
		        cachePool[2].type = UNIFIEDCACHE;
		        cachePool[2].associativity =
		            amdGetAssociativity(extractBitField(edx,4,12));
		        cachePool[2].lineSize = extractBitField(edx,8,0);
		        cachePool[2].size =  (extractBitField(edx,14,18)+1) * 524288;
		        if ((cachePool[0].associativity * cachePool[0].lineSize) != 0)
		        {
		            cachePool[2].sets = cachePool[1].size/
		                (cachePool[1].associativity * cachePool[1].lineSize);
		        }

		        if (cpuid_info.model != MAGNYCOURS)
		        {
		            cachePool[2].threads = cpuid_topology.numCoresPerSocket;
		        }
		        else
		        {
		            cachePool[2].threads = cpuid_topology.numCoresPerSocket/2;
		            cachePool[2].size /= 2 ;
		        }

		        cachePool[2].inclusive = 1;

		        break;

		    case K16_FAMILY:

		    case K15_FAMILY:

		        maxNumLevels = 0;
		        cachePool = (CacheLevel*) malloc(3 * sizeof(CacheLevel));

		        while (type)
		        {
		            ecx = id;
		            eax = 0x8000001D;
		            CPUID;
		            type = (CacheType) extractBitField(eax,4,0);

		            if ((type == DATACACHE) || (type == UNIFIEDCACHE))
		            {
		                cachePool[maxNumLevels].level =   extractBitField(eax,3,5);
		                cachePool[maxNumLevels].type = type;
		                cachePool[maxNumLevels].associativity = extractBitField(ebx,10,22)+1;
		                cachePool[maxNumLevels].lineSize = extractBitField(ebx,12,0)+1;
		                cachePool[maxNumLevels].sets =  extractBitField(ecx,32,0)+1;
		                cachePool[maxNumLevels].size = cachePool[maxNumLevels].associativity *
		                    cachePool[maxNumLevels].lineSize * cachePool[maxNumLevels].sets;
		                cachePool[maxNumLevels].threads =  extractBitField(eax,12,14)+1;
		                cachePool[maxNumLevels].inclusive =  (edx & (0x1<<1));
		                maxNumLevels++;
		            }
		            id++;
		        }
		        break;

		    default:
		        ERROR_PLAIN_PRINT(Processor is not supported);
		        break;
		}
	}

    cpuid_topology.numCacheLevels = maxNumLevels;
    cpuid_topology.cacheLevels = cachePool;
}



