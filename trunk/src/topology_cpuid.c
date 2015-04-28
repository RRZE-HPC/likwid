#include <stdlib.h>
#include <stdio.h>
#include <sched.h>
#include <unistd.h>

#include <error.h>

#include <tree.h>
#include <bitUtil.h>
//#include <strUtil.h>
#include <tlb-info.h>
#include <topology.h>

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
static int largest_function = 0;        
static uint32_t eax, ebx, ecx, edx;

/* Dirty hack to avoid nonull warnings */
char* (*ownstrcpy)(char *__restrict __dest, const char *__restrict __src);

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */
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
                    (cpuid_info.model == HASWELL_EP) ||
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


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void cpuid_printTlbTopology()
{
    int i;
    uint32_t loop = 1;

    if (cpuid_info.isIntel)
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

static void
cpuid_set_osname(void)
{
    FILE *fp;
    bstring nameString = bformat("model name");
    cpuid_info.osname = malloc(MAX_MODEL_STRING_LENGTH * sizeof(char));
    ownstrcpy = strcpy;
    int i;

    if (NULL != (fp = fopen ("/proc/cpuinfo", "r"))) 
    {
        bstring src = bread ((bNread) fread, fp);
        struct bstrList* tokens = bsplit(src,(char) '\n');

        for (i=0;i<tokens->qty;i++)
        {
            if (binstr(tokens->entry[i],0,nameString) != BSTR_ERR)
            {
                 struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                 bltrimws(subtokens->entry[1]);
                 ownstrcpy(cpuid_info.osname, bdata(subtokens->entry[1]));
            }
        }
    }
    else
    {
        ERROR;
    }

    fclose(fp);
}


void cpuid_init_cpuInfo(cpu_set_t cpuSet)
{
    int cpus_in_set = 0;
    cpuid_info.isIntel = 1;

    eax = 0x00;
    CPUID;

    largest_function = eax;
    if (ebx == 0x68747541U)
    {
        cpuid_info.isIntel = 0;
    }

    eax = 0x01;
    CPUID;
    cpuid_info.family = ((eax>>8)&0xFU) + ((eax>>20)&0xFFU);
    cpuid_info.model = (((eax>>16)&0xFU)<<4) + ((eax>>4)&0xFU);
    cpuid_info.stepping =  (eax&0xFU);
    cpuid_set_osname();
    cpuid_topology.numHWThreads = sysconf(_SC_NPROCESSORS_CONF);
    cpus_in_set = cpu_count(&cpuSet);
    if (cpus_in_set < cpuid_topology.numHWThreads)
    {
        cpuid_topology.numHWThreads = cpus_in_set;
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, CPU-ID CpuInfo Family %d Model %d Stepping %d isIntel %d numHWThreads %d activeHWThreads %d,
                            cpuid_info.family,
                            cpuid_info.model,
                            cpuid_info.stepping,
                            cpuid_info.isIntel,
                            cpuid_topology.numHWThreads,
                            cpuid_topology.activeHWThreads)
    return;
}

void cpuid_init_cpuFeatures(void)
{
    eax = 0x01;
    CPUID;

    cpuid_info.featureFlags = 0;
    cpuid_info.features = (char*) malloc(MAX_FEATURE_STRING_LENGTH*sizeof(char));
    cpuid_info.features[0] = '\0';
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
        strcat(cpuid_info.features, "ACPI ");
        cpuid_info.featureFlags |= (1<<ACPI);
    }
    if (edx & (1<<23))
    {
        strcat(cpuid_info.features, "MMX ");
        cpuid_info.featureFlags |= (1<<MMX);
    }
    if (edx & (1<<25))
    {
        strcat(cpuid_info.features, "SSE ");
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

    return;
}

void cpuid_init_nodeTopology(cpu_set_t cpuSet)
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
    int width;
    
    hwThreadPool = (HWThread*) malloc(cpuid_topology.numHWThreads * sizeof(HWThread));
    
    
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
            int id;
            CPU_ZERO(&set);
            CPU_SET(i,&set);
            sched_setaffinity(0, sizeof(cpu_set_t), &set);
            eax = 0x0B;
            ecx = 0;
            CPUID;
            apicId = edx;
            id = i;
            hwThreadPool[id].apicId = i;
            hwThreadPool[id].inCpuSet = 0;
            if (CPU_ISSET(id, &cpuSet))
            {
                hwThreadPool[id].inCpuSet = 1;
            }

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
                        hwThreadPool[id].threadId = bitField;
                        break;

                    case 1:  /* Core */
                        bitField = extractBitField(apicId,
                                currOffset-prevOffset,
                                prevOffset);
                        hwThreadPool[id].coreId = bitField;
                        break;

                    case 2:  /* Package */
                        bitField = extractBitField(apicId,
                                32-prevOffset,
                                prevOffset);
                        hwThreadPool[id].packageId = bitField;
                        break;

                }
                prevOffset = currOffset;
            }
            DEBUG_PRINT(DEBUGLEV_DEVELOP, I[%d] ID[%d] APIC[%d] T[%d] C[%d] P [%d], i, id,
                                    hwThreadPool[id].apicId, hwThreadPool[id].threadId,
                                    hwThreadPool[id].coreId, hwThreadPool[id].packageId);
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
                    int id;
                    CPU_ZERO(&set);
                    CPU_SET(i,&set);
                    sched_setaffinity(0, sizeof(cpu_set_t), &set);

                    eax = 0x01;
                    CPUID;
                    id = i;
                    hwThreadPool[id].apicId = i;//extractBitField(ebx,8,24);

                    /* ThreadId is extracted from th apicId using the bit width
                     * of the number of logical processors
                     * */
                    hwThreadPool[id].threadId =
                        extractBitField(hwThreadPool[id].apicId,
                                getBitFieldWidth(maxNumLogicalProcsPerCore),0); 

                    /* CoreId is extracted from th apicId using the bitWidth 
                     * of the number of logical processors as offset and the
                     * bit width of the number of cores as width
                     * */
                    hwThreadPool[id].coreId =
                        extractBitField(hwThreadPool[id].apicId,
                                getBitFieldWidth(maxNumCores),
                                getBitFieldWidth(maxNumLogicalProcsPerCore)); 

                    hwThreadPool[id].packageId =
                        extractBitField(hwThreadPool[id].apicId,
                                8-getBitFieldWidth(maxNumLogicalProcs),
                                getBitFieldWidth(maxNumLogicalProcs));
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, I[%d] ID[%d] APIC[%d] T[%d] C[%d] P [%d], i, id,
                                    hwThreadPool[id].apicId, hwThreadPool[id].threadId,
                                    hwThreadPool[id].coreId, hwThreadPool[id].packageId);
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
                    int id;
                    CPU_ZERO(&set);
                    CPU_SET(i,&set);
                    sched_setaffinity(0, sizeof(cpu_set_t), &set);

                    eax = 0x01;
                    CPUID;
                    id = extractBitField(ebx,8,24);
                    hwThreadPool[id].apicId = extractBitField(ebx,8,24);

                    /* ThreadId is extracted from th apicId using the bit width
                     * of the number of logical processors
                     * */
                    hwThreadPool[id].threadId =
                        extractBitField(hwThreadPool[i].apicId,
                                getBitFieldWidth(maxNumLogicalProcsPerCore),0); 

                    /* CoreId is extracted from th apicId using the bitWidth 
                     * of the number of logical processors as offset and the
                     * bit width of the number of cores as width
                     * */
                    hwThreadPool[id].coreId =
                        extractBitField(hwThreadPool[i].apicId,
                                getBitFieldWidth(maxNumCores),
                                0); 

                    hwThreadPool[id].packageId =
                        extractBitField(hwThreadPool[i].apicId,
                                8-getBitFieldWidth(maxNumCores),
                                getBitFieldWidth(maxNumCores));
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, I[%d] ID[%d] APIC[%d] T[%d] C[%d] P [%d], i, id,
                                    hwThreadPool[id].apicId, hwThreadPool[id].threadId,
                                    hwThreadPool[id].coreId, hwThreadPool[id].packageId);
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
                    int id;
                    CPU_ZERO(&set);
                    CPU_SET(i,&set);
                    sched_setaffinity(0, sizeof(cpu_set_t), &set);

                    eax = 0x01;
                    CPUID;
                    id = extractBitField(ebx,8,24);
                    hwThreadPool[id].apicId = extractBitField(ebx,8,24);
                    /* AMD only knows cores */
                    hwThreadPool[id].threadId = 0;

                    hwThreadPool[id].coreId =
                        extractBitField(hwThreadPool[i].apicId,
                                width, 0); 
                    hwThreadPool[id].packageId =
                        extractBitField(hwThreadPool[i].apicId,
                                (8-width), width);
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, I[%d] ID[%d] APIC[%d] T[%d] C[%d] P [%d], i, id,
                                    hwThreadPool[id].apicId, hwThreadPool[id].threadId,
                                    hwThreadPool[id].coreId, hwThreadPool[id].packageId);
                }

                break;
        }
    }
    cpuid_topology.threadPool = hwThreadPool;
    
    return;
}


void cpuid_init_cacheTopology(void)
{
    int maxNumLevels=0;
    int id=0;
    CacheLevel* cachePool = NULL;
    CacheType type = DATACACHE;

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
                //                intelCpuidFunc_2(&cachePool);
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
    

    cpuid_topology.numCacheLevels = maxNumLevels;
    cpuid_topology.cacheLevels = cachePool;
    
    return;
}
