#include <topology_proc.h>


/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */
/* this was taken from the linux kernel */
#define CPUID                              \
    __asm__ volatile ("cpuid"                             \
            : "=a" (eax),     \
            "=b" (ebx),     \
            "=c" (ecx),     \
            "=d" (edx)      \
            : "0" (eax), "2" (ecx))
/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */
static int get_cpu_perf_data(void)
{
    uint32_t eax = 0x0U, ebx = 0x0U, ecx = 0x0U, edx = 0x0U;
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

int get_listPosition(int ownid, bstring list)
{
    bstring ownStr = bformat("%d",ownid);
    struct bstrList* tokens = bsplit(list,(char) ',');
    for(int i=0;i<tokens->qty;i++)
    {
        btrimws(tokens->entry[i]);
        if (bstrcmp(ownStr, tokens->entry[i]) == BSTR_OK)
        {
            return i;
        }
    }
    return -1;
}

int fillList(int* outList, int outOffset, bstring list)
{
    int current = 0;
    int (*ownatoi)(const char*);
    struct bstrList* tokens = bsplit(list,',');
    ownatoi = &atoi;
    for(int i=0;i<tokens->qty;i++)
    {
        btrimws(tokens->entry[i]);
        if (bstrchrp(tokens->entry[i],'-',0) == BSTR_ERR)
        {
            if (outList)
            {
                outList[outOffset+current] = ownatoi(bdata(tokens->entry[i]));
            }
            current++;
        }
        else
        {
            struct bstrList* range = bsplit(tokens->entry[i],'-');
            if (range->qty == 2)
            {
                for (int j=ownatoi(bdata(range->entry[0]));j<=ownatoi(bdata(range->entry[1]));j++)
                {
                    if (outList)
                    {
                        outList[outOffset+current] = j;
                    }
                    
                    current++;
                }
            }
        }
    }
    return current;
}

static int readCacheInclusive(int level)
{
    uint32_t eax = 0x0U, ebx = 0x0U, ecx = 0x0U, edx = 0x0U;
    eax = 0x04;
    ecx = level;
    CPUID;
    return edx & 0x2;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */
void proc_init_cpuInfo(cpu_set_t cpuSet)
{
    int i;
    int HWthreads = 0;
    FILE *fp;

    int (*ownatoi)(const char*);
    char* (*ownstrcpy)(char*,const char*);
    ownatoi = &atoi;
    ownstrcpy = &strcpy;

    const_bstring countString = bformat("processor\t:");
    const_bstring modelString = bformat("model\t\t:");
    const_bstring familyString = bformat("cpu family\t:");
    const_bstring steppingString = bformat("stepping\t:");
    const_bstring vendorString = bformat("vendor_id\t:");
    const_bstring vendorIntelString = bformat("GenuineIntel");
    const_bstring nameString = bformat("model name\t:");

    cpuid_info.isIntel = 0;
    cpuid_info.model = 0;
    cpuid_info.family = 0;
    cpuid_info.stepping = 0;
    cpuid_topology.numHWThreads = 0;
    cpuid_info.osname = malloc(MAX_MODEL_STRING_LENGTH * sizeof(char));

    if (NULL != (fp = fopen ("/proc/cpuinfo", "r"))) 
    {
        bstring src = bread ((bNread) fread, fp);
        struct bstrList* tokens = bsplit(src,(char) '\n');
        for (i=0;i<tokens->qty;i++)
        {
            if (binstr(tokens->entry[i],0,countString) != BSTR_ERR)
            {
                HWthreads++;
            }
            else if ((cpuid_info.model == 0) && (binstr(tokens->entry[i],0,modelString) != BSTR_ERR))
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                cpuid_info.model = ownatoi(bdata(subtokens->entry[1]));
            }
            else if ((cpuid_info.family == 0) && (binstr(tokens->entry[i],0,familyString) != BSTR_ERR))
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                cpuid_info.family = ownatoi(bdata(subtokens->entry[1]));
            }
            else if (binstr(tokens->entry[i],0,steppingString) != BSTR_ERR)
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                cpuid_info.stepping = ownatoi(bdata(subtokens->entry[1]));
            }
            else if (binstr(tokens->entry[i],0,nameString) != BSTR_ERR)
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                ownstrcpy(cpuid_info.osname, bdata(subtokens->entry[1]));
            }
            else if (binstr(tokens->entry[i],0,vendorString) != BSTR_ERR)
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                if (bstrcmp(subtokens->entry[1], vendorIntelString) == BSTR_OK)
                {
                    cpuid_info.isIntel = 1;
                }
            }
        }
        cpuid_topology.numHWThreads = HWthreads;
        DEBUG_PRINT(DEBUGLEV_DEVELOP, PROC CpuInfo Family %d Model %d Stepping %d isIntel %d numHWThreads %d,
                            cpuid_info.family,
                            cpuid_info.model,
                            cpuid_info.stepping,
                            cpuid_info.isIntel,
                            cpuid_topology.numHWThreads)
    }
    return;
}

void proc_init_cpuFeatures(void)
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



void proc_init_nodeTopology(cpu_set_t cpuSet)
{
    HWThread* hwThreadPool;
    FILE *fp;
    bstring cpudir;
    bstring file;
    int (*ownatoi)(const char*);
    ownatoi = &atoi;

    hwThreadPool = (HWThread*) malloc(cpuid_topology.numHWThreads * sizeof(HWThread));
    for (uint32_t i=0;i<cpuid_topology.numHWThreads;i++)
    {
        hwThreadPool[i].apicId = i;
        hwThreadPool[i].threadId = -1;
        hwThreadPool[i].coreId = -1;
        hwThreadPool[i].packageId = -1;
        hwThreadPool[i].inCpuSet = 1;
        if (!CPU_ISSET(i, &cpuSet))
        {
            hwThreadPool[i].inCpuSet = 0;
        }
        cpudir = bformat("/sys/devices/system/cpu/cpu%d/topology",i);
        file = bformat("%s/core_id", bdata(cpudir));
        if (NULL != (fp = fopen (bdata(file), "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            hwThreadPool[i].coreId = ownatoi(bdata(src));
            fclose(fp);
        }
        bdestroy(file);
        file = bformat("%s/physical_package_id", bdata(cpudir));
        if (NULL != (fp = fopen (bdata(file), "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            hwThreadPool[i].packageId = ownatoi(bdata(src));
            fclose(fp);
        }
        bdestroy(file);
        file = bformat("%s/thread_siblings_list", bdata(cpudir));
        if (NULL != (fp = fopen (bdata(file), "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            hwThreadPool[i].threadId = get_listPosition(i, src);
            fclose(fp);
        }
        bdestroy(file);
        DEBUG_PRINT(DEBUGLEV_DEVELOP, PROC Thread Pool PU %d Thread %d Core %d Socket %d,
                            hwThreadPool[i].apicId,
                            hwThreadPool[i].threadId,
                            hwThreadPool[i].coreId,
                            hwThreadPool[i].packageId)
    }
    cpuid_topology.threadPool = hwThreadPool;
    return;
}

void proc_init_cacheTopology(void)
{
    FILE *fp;
    CacheLevel* cachePool = NULL;
    int maxNumLevels = 0;
    int nrCaches = 0;
    bstring cpudir = bformat("/sys/devices/system/cpu/cpu0/cache");
    bstring levelStr;
    int (*ownatoi)(const char*);
    ownatoi = &atoi;
    for (int i=0;i<10;i++)
    {
        levelStr = bformat("%s/index%d/level",bdata(cpudir),i);
        if (NULL != (fp = fopen (bdata(levelStr), "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            int tmp = 0;
            tmp = ownatoi(bdata(src));
            if (tmp > maxNumLevels)
            {
                maxNumLevels = tmp;
            }
            nrCaches++;
            fclose(fp);
        }
        else
        {
            bdestroy(levelStr);
            break;
        }
        bdestroy(levelStr);
    }

    cachePool = (CacheLevel*) malloc(nrCaches * sizeof(CacheLevel));
    for (int i=0;i<nrCaches;i++)
    {
        levelStr = bformat("%s/index%d/level",bdata(cpudir),i);
        if (NULL != (fp = fopen (bdata(levelStr), "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            cachePool[i].level = ownatoi(bdata(src));
            fclose(fp);
            bdestroy(src);
        }
        bdestroy(levelStr);
        levelStr = bformat("%s/index%d/type",bdata(cpudir),i);
        if (NULL != (fp = fopen (bdata(levelStr), "r")))
        {
            bstring unifiedStr = bformat("Unified");
            bstring dataStr = bformat("Data");
            bstring intrStr = bformat("Instruction");
            bstring src = bread ((bNread) fread, fp);
            btrimws(src);
            if (bstrcmp(dataStr, src) == BSTR_OK)
            {
                cachePool[i].type = DATACACHE;
            }
            else if (bstrcmp(intrStr, src) == BSTR_OK)
            {
                cachePool[i].type = INSTRUCTIONCACHE;
            }
            else if (bstrcmp(unifiedStr, src) == BSTR_OK)
            {
                cachePool[i].type = UNIFIEDCACHE;
            }
            else
            {
                cachePool[i].type = NOCACHE;
            }
            fclose(fp);
            bdestroy(unifiedStr);
            bdestroy(dataStr);
            bdestroy(intrStr);
            bdestroy(src);
        }
        bdestroy(levelStr);
        levelStr = bformat("%s/index%d/size",bdata(cpudir),i);
        if (NULL != (fp = fopen (bdata(levelStr), "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            btrimws(src);
            bdelete(src, blength(src)-1, 1);
            cachePool[i].size = ownatoi(bdata(src)) * 1024;
            fclose(fp);
            bdestroy(src);
        }
        else
        {
            cachePool[i].size = 0;
        }
        bdestroy(levelStr);
        levelStr = bformat("%s/index%d/ways_of_associativity",bdata(cpudir),i);
        if (NULL != (fp = fopen (bdata(levelStr), "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            btrimws(src);
            cachePool[i].associativity = ownatoi(bdata(src));
            fclose(fp);
            bdestroy(src);
        }
        else
        {
            cachePool[i].associativity = 0;
        }
        bdestroy(levelStr);
        levelStr = bformat("%s/index%d/coherency_line_size",bdata(cpudir),i);
        if (NULL != (fp = fopen (bdata(levelStr), "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            btrimws(src);
            cachePool[i].lineSize = ownatoi(bdata(src));
            fclose(fp);
            bdestroy(src);
        }
        else
        {
            cachePool[i].lineSize = 0;
        }
        bdestroy(levelStr);
        levelStr = bformat("%s/index%d/number_of_sets",bdata(cpudir),i);
        if (NULL != (fp = fopen (bdata(levelStr), "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            btrimws(src);
            cachePool[i].sets = ownatoi(bdata(src));
            fclose(fp);
            bdestroy(src);
        }
        else
        {
            if ((cachePool[i].associativity * cachePool[i].lineSize) != 0)
            {
                cachePool[i].sets = cachePool[i].size /
                    (cachePool[i].associativity * cachePool[i].lineSize);
            }
        }
        bdestroy(levelStr);
        levelStr = bformat("%s/index%d/shared_cpu_list",bdata(cpudir),i);
        if (NULL != (fp = fopen (bdata(levelStr), "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            btrimws(src);
            cachePool[i].threads = fillList(NULL, 0, src);
            fclose(fp);
            bdestroy(src);
        }
        bdestroy(levelStr);

        switch ( cpuid_info.family )
        {
            case MIC_FAMILY:
            case P6_FAMILY:
            case K16_FAMILY:
            case K15_FAMILY:
                cachePool[i].inclusive = readCacheInclusive(cachePool[i].level);
                break;
            /* For K8 and K10 it is known that they are inclusive */
            case K8_FAMILY:
            case K10_FAMILY:
                cachePool[i].inclusive = 1;
                break;
            default:
                ERROR_PLAIN_PRINT(Processor is not supported);
                break;
        }
    }
    cpuid_topology.numCacheLevels = nrCaches;
    cpuid_topology.cacheLevels = cachePool;
    return;
}

