/*
 * ===========================================================================
 *
 *      Filename:  cpuid.c
 *
 *      Description:  Implementation of cpuid module.
 *                  Provides API to extract cpuid info on x86 processors.
 *
 *      Version:  <VERSION>
 *      Created:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Company:  RRZE Erlangen
 *      Project:  likwid
 *      Copyright:  Copyright (c) 2010, Jan Treibig
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License, v2, as
 *      published by the Free Software Foundation
 *     
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *     
 *      You should have received a copy of the GNU General Public License
 *      along with this program; if not, write to the Free Software
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * ===========================================================================
 */


/* #####   HEADER FILE INCLUDES   ######################################### */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <error.h>
#include <cpuid.h>
#include <affinity.h>
#include <tree.h>

#include <osdep/asm.h>
#include <osdep/cpuid.h>

/* #####   EXPORTED VARIABLES   ########################################### */

CpuInfo cpuid_info;
CpuTopology cpuid_topology;


/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static uint32_t eax, ebx, ecx, edx;
static int largest_function=0;

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ###################### */

#define STORE 0
#define RESTORE 1

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static char* pentium_m_b_str = "Intel Pentium M Banias processor";
static char* pentium_m_d_str = "Intel Pentium M Dothan processor";
static char* core_duo_str = "Intel Core Duo processor";
static char* core_2a_str = "Intel Core 2 65nm processor";
static char* core_2b_str = "Intel Core 2 45nm processor";
static char* nehalem_bloom_str = "Intel Core Bloomfield processor";
static char* nehalem_lynn_str = "Intel Core Lynnfield processor";
static char* nehalem_west_str = "Intel Core Westmere processor";
static char* nehalem_ex_str = "Intel Nehalem EX processor";
static char* xeon_mp_string = "Intel Xeon MP processor";
static char* barcelona_str = "AMD Barcelona processor";
static char* shanghai_str = "AMD Shanghai processor";
static char* istanbul_str = "AMD Istanbul processor";
static char* magnycours_str = "AMD Magny Cours processor";
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

static int init = 0;
static uint32_t eax, ebx, ecx, edx;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static uint32_t 
extractBitField(uint32_t inField, uint32_t width, uint32_t offset)
{
    uint32_t bitMask;
    uint32_t outField;

    if ((offset+width) == 32) 
    {
        bitMask = (0xFFFFFFFF<<offset);
    }
    else 
    {
        bitMask = (0xFFFFFFFF<<offset) ^ (0xFFFFFFFF<<(offset+width));

    }

    outField = (inField & bitMask) >> offset;
    return outField;
}

static uint32_t
getBitFieldWidth(uint32_t number)
{
    uint32_t fieldWidth;

    number--;
    if (number == 0)
    {
        return 0;
    }

    ASM_BSR(fieldWidth,number);

    return fieldWidth+1;  /* bsr returns the position, we want the width */
}

static uint32_t
amdGetAssociativity(uint32_t flag)
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

static int
intelCpuidFunc_4(CacheLevel** cachePool)
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
		 * Limit threads per Socket than to the maximum possible value.*/
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

 /* :TODO:04/19/2010 09:06:36 AM:jt: Add cache parameter from old cpuid leaf 0x2 */
#if 0
static int
intelCpuidFunc_2(CacheLevel** cachePool)
{
    eax = 0x02;
    CPUID;

    return 0;

}
#endif



/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
cpuid_init (void)
{
    int isIntel = 1;

    if (init) return;
    init =1;

    eax = 0x00;
    CPUID;

//    printf("CPUID Largest supported basic function 0x%X \n",eax);
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

                case NEHALEM_EX:
                    cpuid_info.name = nehalem_ex_str;
                    break;

                case XEON_MP:
                    cpuid_info.name = xeon_mp_string;
                    break;

                default:
                    cpuid_info.name = unknown_intel_str;
                    break;
            }
            break;

        case K8_FAMILY:

            if (isIntel)
            {
                ERROR_MSG(Netburst architecture is not supported);
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

        default:
            ERROR_MSG(Processor is not supported);
            break;
    }

    cpuid_info.features = (char*) malloc(100*sizeof(char));
    if (edx & (1<<25)) strcpy(cpuid_info.features, "SSE ");
    if (edx & (1<<26)) strcat(cpuid_info.features, "SSE2 ");
    if (ecx & (1<<0))  strcat(cpuid_info.features, "SSE3 ");
    if (ecx & (1<<19)) strcat(cpuid_info.features, "SSE4.1 ");
    if (ecx & (1<<20)) strcat(cpuid_info.features, "SSE4.2 ");

    cpuid_info.perf_version   =  0;
    if( cpuid_info.family == P6_FAMILY && 0x0A <= largest_function) 
    {
        eax = 0x0A;
        CPUID;
        cpuid_info.perf_version   =  (eax&0xFFU);
        cpuid_info.perf_num_ctr   =   ((eax>>8)&0xFFU);
        cpuid_info.perf_width_ctr =  ((eax>>16)&0xFFU);
        cpuid_info.perf_num_fixed_ctr =  (edx&0xFU);
    }

    /* Determine the number of processors accessible */
    cpuid_topology.numHWThreads = numberOfProcessors();

    affinity_state(STORE);
    cpuid_initTopology();
    cpuid_initCacheTopology();
    affinity_state(RESTORE);

    return;
}

void
cpuid_initTopology(void)
{
    uint32_t apicId;
    uint32_t bitField;
    uint32_t i;
    int level;
    int prevOffset = 0;
    int currOffset = 0;
    HWThread* hwThreadPool;
    int hasBLeaf = 0;
    int maxNumLogicalProcs;
    int maxNumLogicalProcsPerCore;
    int maxNumCores;
    TreeNode* currentNode;
    int width;


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

    if (cpuid_topology.numHWThreads == 1)
    {
        ERROR_MSG(This is a single core processor);
    }

    hwThreadPool = (HWThread*) malloc(cpuid_topology.numHWThreads * sizeof(HWThread));
    tree_init(&cpuid_topology.topologyTree, 0);

    if (hasBLeaf)
    {
        for (i=0; i < cpuid_topology.numHWThreads; i++)
        {
            affinity_pinProcess(i);
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

                for (i=0; i<  cpuid_topology.numHWThreads; i++)
                {
                    affinity_pinProcess(i);

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

                for (i=0; i<  cpuid_topology.numHWThreads; i++)
                {
                    affinity_pinProcess(i);

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


                for (i=0; i<  cpuid_topology.numHWThreads; i++)
                {
                    affinity_pinProcess(i);

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

    cpuid_topology.threadPool = hwThreadPool;
    cpuid_topology.numSockets = tree_countChildren(cpuid_topology.topologyTree);
    currentNode = tree_getNode(cpuid_topology.topologyTree, 0);
    cpuid_topology.numCoresPerSocket = tree_countChildren(currentNode);
    currentNode = tree_getNode(currentNode, 0);
    cpuid_topology.numThreadsPerCore = tree_countChildren(currentNode);
}

void 
cpuid_initCacheTopology()
{
    int maxNumLevels=0;
    CacheLevel* cachePool = NULL;

    switch ( cpuid_info.family ) 
    {
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

            if (! (cpuid_info.model == MAGNYCOURS))
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

        default:
            ERROR_MSG(Processor is not supported);
            break;
    }

    cpuid_topology.numCacheLevels = maxNumLevels;
    cpuid_topology.cacheLevels = cachePool;
}



