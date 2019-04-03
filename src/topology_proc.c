/*
 * =======================================================================================
 *
 *      Filename:  topology_proc.c
 *
 *      Description:  Interface to the procfs/sysfs based topology backend
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Authors:  Jan Treibig (jt), jan.treibig@gmail.com,
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
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

#include <topology_proc.h>
#include <affinity.h>
#if !defined(__ARM_ARCH_7A__) && !defined(__ARM_ARCH_8A)
#include <cpuid.h>
#endif
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int
get_cpu_perf_data(void)
{
#if defined(__x86_64) || defined(__i386__)
    uint32_t eax = 0x0U, ebx = 0x0U, ecx = 0x0U, edx = 0x0U;
    int largest_function = 0;
    eax = 0x00;
    CPUID(eax, ebx, ecx, edx);
    largest_function = eax;
    cpuid_info.perf_version = 0;
    cpuid_info.perf_num_ctr = 0;
    cpuid_info.perf_width_ctr = 0;
    cpuid_info.perf_num_fixed_ctr = 0;
    cpuid_info.turbo = 0;
    if (cpuid_info.family == P6_FAMILY && 0x0A <= largest_function)
    {
        eax = 0x0A;
        CPUID(eax, ebx, ecx, edx);
        cpuid_info.perf_version   =  (eax&0xFFU);
        cpuid_info.perf_num_ctr   =   ((eax>>8)&0xFFU);
        cpuid_info.perf_width_ctr =  ((eax>>16)&0xFFU);
        cpuid_info.perf_num_fixed_ctr =  (edx&0xFU);

        eax = 0x06;
        CPUID(eax, ebx, ecx, edx);
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
    return 0;
}

static int
get_listPosition(int ownid, bstring list)
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
    bstrListDestroy(tokens);
    return -1;
}

static int
get_listLength(bstring list)
{
    struct bstrList* tokens = bsplit(list,(char) ',');
    int len = tokens->qty;
    bstrListDestroy(tokens);
    return len;
}

static int
fillList(int* outList, int outOffset, bstring list)
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
            bstrListDestroy(range);
        }
    }
    bstrListDestroy(tokens);
    return current;
}
#if defined(__x86_64) || defined(__i386__)
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
#else
static int readCacheInclusiveIntel(int level)
{
    return 0;
}
static int readCacheInclusiveAMD(int level)
{
    return 0;
}
#endif

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
proc_init_cpuInfo(cpu_set_t cpuSet)
{
    int i = 0;
    int HWthreads = 0;
    FILE *fp = NULL;

    int (*ownatoi)(const char*);
    char* (*ownstrcpy)(char*,const char*);
    ownatoi = &atoi;
    ownstrcpy = &strcpy;

#if defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(__x86_64)
    const_bstring modelString = bformat("model\t\t:");
    const_bstring steppingString = bformat("stepping\t:");
    const_bstring nameString = bformat("model name\t:");
#endif
#ifdef _ARCH_PPC
    const_bstring modelString = bformat("cpu\t\t:");
    const_bstring steppingString = bformat("revision\t:");
    const_bstring nameString = bformat("machine\t\t:");
#endif
    const_bstring familyString = bformat("cpu family\t:");
    const_bstring countString = bformat("processor\t:");
    const_bstring vendorString = bformat("vendor_id\t:");
    const_bstring vendorIntelString = bformat("GenuineIntel");

    cpuid_info.isIntel = 0;
    cpuid_info.model = 0;
    cpuid_info.family = 0;
#ifdef _ARCH_PPC
    cpuid_info.family = PPC_FAMILY;
#endif
    cpuid_info.stepping = 0;
    cpuid_topology.numHWThreads = 0;
    cpuid_info.osname = malloc(MAX_MODEL_STRING_LENGTH * sizeof(char));

    if (NULL != (fp = fopen ("/proc/cpuinfo", "r")))
    {
        bstring src = bread ((bNread) fread, fp);
        struct bstrList* tokens = bsplit(src,(char) '\n');
        bdestroy(src);
        fclose(fp);
        for (i=0;i<tokens->qty;i++)
        {
            printf("%d\n", binstr(tokens->entry[i],0,modelString));
            if (binstr(tokens->entry[i],0,countString) != BSTR_ERR)
            {
                HWthreads++;
            }
            else if ((cpuid_info.model == 0) && (binstr(tokens->entry[i],0,modelString) != BSTR_ERR))
            {
#ifndef _ARCH_PPC
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                cpuid_info.model = ownatoi(bdata(subtokens->entry[1]));
                bstrListDestroy(subtokens);
#else
		const_bstring power7str = bformat("POWER7");
		const_bstring power8str = bformat("POWER8");
		const_bstring power9str = bformat("POWER9");
		if (binstr(tokens->entry[i],0, power7str) != BSTR_ERR)
		{
			cpuid_info.model = POWER7;
		}
		else if (binstr(tokens->entry[i],0, power8str) != BSTR_ERR)
                {
                        cpuid_info.model = POWER8;
                }
		else if (binstr(tokens->entry[i],0, power9str) != BSTR_ERR)
                {
                        cpuid_info.model = POWER9;
                }
#endif
            }
            else if ((cpuid_info.family == 0) && (binstr(tokens->entry[i],0,familyString) != BSTR_ERR))
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                cpuid_info.family = ownatoi(bdata(subtokens->entry[1]));
                bstrListDestroy(subtokens);
            }
            else if (binstr(tokens->entry[i],0,steppingString) != BSTR_ERR)
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                cpuid_info.stepping = ownatoi(bdata(subtokens->entry[1]));
                bstrListDestroy(subtokens);
            }
            else if (binstr(tokens->entry[i],0,nameString) != BSTR_ERR)
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                ownstrcpy(cpuid_info.osname, bdata(subtokens->entry[1]));
                bstrListDestroy(subtokens);
            }
            else if (binstr(tokens->entry[i],0,vendorString) != BSTR_ERR)
            {
                struct bstrList* subtokens = bsplit(tokens->entry[i],(char) ':');
                bltrimws(subtokens->entry[1]);
                if (bstrcmp(subtokens->entry[1], vendorIntelString) == BSTR_OK)
                {
                    cpuid_info.isIntel = 1;
                }
                bstrListDestroy(subtokens);
            }
        }
        bstrListDestroy(tokens);
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

void
proc_init_cpuFeatures(void)
{
    int ret;
    FILE* file;
    char buf[1024];
    char ident[30];
    char delimiter[] = " ";
    char* cptr;
#ifdef _ARCH_PPC
    return;
#endif

    if ( (file = fopen( "/proc/cpuinfo", "r")) == NULL )
    {
        fprintf(stderr, "Cannot open /proc/cpuinfo\n");
        return;
    }
    ret = 0;
    while( fgets(buf, sizeof(buf)-1, file) )
    {
        ret = sscanf(buf, "%s\t:", &(ident[0]));
#ifdef __x86_64
        if (ret != 1 || strcmp(ident,"flags") != 0 || strcmp(ident, "Features") != 0)
#endif
#if defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_8A__) || defined(__ARM_ARCH_8A)
        if (ret != 1 || strcmp(ident, "Features") != 0)
#endif
#ifdef _ARCH_PPC
	if (ret != 1)
#endif
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
    buf[strcspn(buf, "\n")] = '\0';
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
        else if (strcmp(cptr,"est") == 0)
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
            strcat(cpuid_info.features, "SSE4.1 ");
        }
        else if (strcmp(cptr,"sse4_2") == 0)
        {
            cpuid_info.featureFlags |= (1<<SSE42);
            strcat(cpuid_info.features, "SSE4.2 ");
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
        else if (strcmp(cptr,"avx2") == 0)
        {
            cpuid_info.featureFlags |= (1<<AVX2);
            strcat(cpuid_info.features, "AVX2 ");
        }
        else if (strcmp(cptr,"rtm") == 0)
        {
            cpuid_info.featureFlags |= (1<<RTM);
            strcat(cpuid_info.features, "RTM ");
        }
        else if (strcmp(cptr,"hle") == 0)
        {
            cpuid_info.featureFlags |= (1<<HLE);
            strcat(cpuid_info.features, "HLE ");
        }
        else if (strcmp(cptr,"rdseed") == 0)
        {
            cpuid_info.featureFlags |= (1<<RDSEED);
            strcat(cpuid_info.features, "RDSEED ");
        }
        else if (strcmp(cptr,"ht") == 0)
        {
            cpuid_info.featureFlags |= (1<<HTT);
            strcat(cpuid_info.features, "HTT ");
        }
        else if (strncmp(cptr,"avx512", 6) == 0 && !(cpuid_info.featureFlags & (1<<AVX512)))
        {
            cpuid_info.featureFlags |= (1<<AVX512);
            strcat(cpuid_info.features, "AVX512 ");
        }
        else if (strcmp(cptr,"swp") == 0)
        {
            cpuid_info.featureFlags |= (1<<SWP);
            strcat(cpuid_info.features, "SWP ");
        }
        else if (strcmp(cptr,"neon") == 0)
        {
            cpuid_info.featureFlags |= (1<<NEON);
            strcat(cpuid_info.features, "NEON ");
        }
        else if (strcmp(cptr,"vfp") == 0)
        {
            cpuid_info.featureFlags |= (1<<VFP);
            strcat(cpuid_info.features, "VFP ");
        }
        else if (strcmp(cptr,"vfpv3") == 0)
        {
            cpuid_info.featureFlags |= (1<<VFPV3);
            strcat(cpuid_info.features, "VFPv3 ");
        }
        else if (strcmp(cptr,"vfpv4") == 0)
        {
            cpuid_info.featureFlags |= (1<<VFPV4);
            strcat(cpuid_info.features, "VFPv4 ");
        }
        else if (strcmp(cptr,"edsp") == 0)
        {
            cpuid_info.featureFlags |= (1<<EDSP);
            strcat(cpuid_info.features, "EDSP ");
        }
        else if (strcmp(cptr,"tls") == 0)
        {
            cpuid_info.featureFlags |= (1<<TLS);
            strcat(cpuid_info.features, "TLS ");
        }
        cptr = strtok(NULL, delimiter);
    }

    if ((cpuid_info.featureFlags & (1<<SSSE3)) && !((cpuid_info.featureFlags) & (1<<SSE3)))
    {
        cpuid_info.featureFlags |= (1<<SSE3);
        strcat(cpuid_info.features, "SSE3 ");
    }

    get_cpu_perf_data();
    return;
}

void
proc_init_nodeTopology(cpu_set_t cpuSet)
{
    HWThread* hwThreadPool;
    FILE *fp;
    bstring cpudir;
    bstring file;
    int (*ownatoi)(const char*);
    ownatoi = &atoi;
    int last_socket = -1;
    int num_sockets = 0;
    int num_cores_per_socket = 0;
    int num_threads_per_core = 0;

    hwThreadPool = (HWThread*) malloc(cpuid_topology.numHWThreads * sizeof(HWThread));
    for (uint32_t i=0;i<cpuid_topology.numHWThreads;i++)
    {
        hwThreadPool[i].apicId = i;
        cpudir = bformat("/sys/devices/system/cpu/cpu%d/topology",i);
        hwThreadPool[i].threadId = -1;
        hwThreadPool[i].coreId = -1;
        hwThreadPool[i].packageId = -1;
        hwThreadPool[i].inCpuSet = 0;
        if (CPU_ISSET(i, &cpuSet))
        {
            hwThreadPool[i].inCpuSet = 1;
        }
        file = bformat("%s/physical_package_id", bdata(cpudir));
        if (NULL != (fp = fopen (bdata(file), "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            int packageId = ownatoi(bdata(src));
            hwThreadPool[i].packageId = packageId;
            if (packageId > last_socket)
            {
                num_sockets++;
                last_socket = packageId;
            }
            fclose(fp);
        }
        bdestroy(file);
        file = bformat("%s/core_id", bdata(cpudir));
        if (NULL != (fp = fopen (bdata(file), "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            hwThreadPool[i].coreId = ownatoi(bdata(src));
            if (hwThreadPool[i].packageId == 0)
            {
                num_cores_per_socket++;
            }
            fclose(fp);
        }
        bdestroy(file);
        file = bformat("%s/thread_siblings_list", bdata(cpudir));
        if (NULL != (fp = fopen (bdata(file), "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            hwThreadPool[i].threadId = get_listPosition(i, src);
            if (hwThreadPool[i].packageId == 0 &&
                hwThreadPool[i].coreId == 0)
            {
                num_threads_per_core++;
            }
            fclose(fp);
        }
        bdestroy(file);
#ifdef __ARM_ARCH_8A
        if (hwThreadPool[i].packageId == -1)
            hwThreadPool[i].packageId = 0;
#endif
        DEBUG_PRINT(DEBUGLEV_DEVELOP, PROC Thread Pool PU %d Thread %d Core %d Socket %d inCpuSet %d,
                            hwThreadPool[i].apicId,
                            hwThreadPool[i].threadId,
                            hwThreadPool[i].coreId,
                            hwThreadPool[i].packageId,
                            hwThreadPool[i].inCpuSet)
        bdestroy(cpudir);
    }
    cpuid_topology.threadPool = hwThreadPool;
    cpuid_topology.numSockets = num_sockets;
    cpuid_topology.numCoresPerSocket = num_cores_per_socket;
    cpuid_topology.numThreadsPerCore = num_threads_per_core;
    return;
}

void proc_split_llc_check(CacheLevel* llc_cache)
{
    int num_sockets = cpuid_topology.numSockets;
    int num_nodes = 0;
    int num_threads_per_node = 0;
    int num_threads_per_socket = (cpuid_topology.numCoresPerSocket * cpuid_topology.numThreadsPerCore) / num_sockets;
    struct dirent *ep = NULL;
    DIR *dp = NULL;
    dp = opendir("/sys/devices/system/node");
    if (dp == NULL)
    {
        fprintf(stderr, "No NUMA support (no folder %s)\n", "/sys/devices/system/node");
        return;
    }
    while (ep = readdir(dp))
    {
        if (strncmp(ep->d_name, "node", 4) == 0)
        {
            num_nodes++;
        }
    }
    closedir(dp);
    dp = opendir("/sys/devices/system/node/node0/");
    if (dp == NULL)
    {
        fprintf(stderr, "No NUMA support (no folder %s)\n", "/sys/devices/system/node/node0/");
        return;
    }
    while (ep = readdir(dp))
    {
        if (strncmp(ep->d_name, "cpu", 3) == 0 &&
            ep->d_name[strlen(ep->d_name)-1] >= '0' &&
            ep->d_name[strlen(ep->d_name)-1] <= '9')
        {
            num_threads_per_node++;
        }
    }
    closedir(dp);
    if (num_sockets == num_nodes)
    {
        return;
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
proc_init_cacheTopology(void)
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
                cachePool[i].inclusive = readCacheInclusiveIntel(cachePool[i].level);
                break;
            case K16_FAMILY:
            case K15_FAMILY:
            case ZEN_FAMILY:
                cachePool[i].inclusive = readCacheInclusiveAMD(cachePool[i].level);
                break;
            /* For K8 and K10 it is known that they are inclusive */
            case K8_FAMILY:
            case K10_FAMILY:
                cachePool[i].inclusive = 1;
                break;
            case ARMV8_FAMILY:
            case ARMV7_FAMILY:
	    case PPC_FAMILY:
                cachePool[i].inclusive = 0;
                break;
            default:
                ERROR_PLAIN_PRINT(Processor is not supported);
                break;
        }
    }
    bdestroy(cpudir);
    cpuid_topology.numCacheLevels = nrCaches;
    cpuid_topology.cacheLevels = cachePool;
    return;
}

