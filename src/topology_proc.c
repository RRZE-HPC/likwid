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
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2019 RRZE, University Erlangen-Nuremberg
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

#include <bstrlib.h>
#include <bstrlib_helper.h>

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
        if (eax & (1ULL<<1))
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
    return (edx & (0x1ULL<<1));
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
    const_bstring vendorString = bformat("vendor_id\t:");
    const_bstring familyString = bformat("cpu family\t:");
#endif
#ifdef _ARCH_PPC
    const_bstring modelString = bformat("cpu\t\t:");
    const_bstring steppingString = bformat("revision\t:");
    const_bstring nameString = bformat("machine\t\t:");
    const_bstring vendorString = bformat("vendor_id\t:");
    const_bstring familyString = bformat("cpu family\t:");
#endif
#if defined(__ARM_ARCH_8A) || defined(__ARM_ARCH_7A__)
    const_bstring modelString = bformat("CPU variant\t:");
    const_bstring steppingString = bformat("CPU revision\t:");
    const_bstring nameString = bformat("machine\t\t:");
    const_bstring vendorString = bformat("CPU implementer\t:");
    const_bstring familyString = bformat("CPU architecture");
#endif

    const_bstring countString = bformat("processor\t:");

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
    cpuid_info.osname[0] = '\0';

    if (NULL != (fp = fopen ("/proc/cpuinfo", "r")))
    {
        bstring src = bread ((bNread) fread, fp);
        struct bstrList* tokens = bsplit(src,(char) '\n');
        bdestroy(src);
        fclose(fp);
        for (i=0;i<tokens->qty;i++)
        {
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
        HWthreads = likwid_sysfs_list_len("/sys/devices/system/cpu/present");
        if (HWthreads > cpuid_topology.numHWThreads)
            cpuid_topology.numHWThreads = HWthreads;
#ifdef __x86_64
        snprintf(cpuid_info.architecture, 19, "x86_64");
#endif
#ifdef __ARM_ARCH_7A__
        snprintf(cpuid_info.architecture, 19, "armv7");
#endif
#ifdef __ARM_ARCH_8A
        snprintf(cpuid_info.architecture, 19, "armv8");
#endif
#ifdef _ARCH_PPC
        switch (cpuid_info.model)
        {
            case POWER7:
                snprintf(cpuid_info.architecture, 19, "power7");
                break;
            case POWER8:
                snprintf(cpuid_info.architecture, 19, "power8");
                break;
            case POWER9:
                snprintf(cpuid_info.architecture, 19, "power9");
                break;
        }

#endif
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
    int ret = 0;
    FILE* file;
    char buf[1024];
    char ident[30];
    char delimiter[] = " ";
    char* cptr;
#ifdef _ARCH_PPC
    return;
#endif
    const_bstring flagString = bformat("flags");
    const_bstring featString = bformat("Features");
    bstring flagline = bfromcstr("");

    bstring cpuinfo = read_file("/proc/cpuinfo");
    struct bstrList* cpulines = bsplit(cpuinfo, '\n');
    bdestroy(cpuinfo);
    for (int i = 0; i < cpulines->qty; i++)
    {
#if defined(__x86_64__) || defined(__i386__)
        if (bstrncmp(cpulines->entry[i], flagString, 5) == BSTR_OK ||
            bstrncmp(cpulines->entry[i], featString, 8) == BSTR_OK)
#endif
#if defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_8A__) || defined(__ARM_ARCH_8A)
        if (bstrncmp(cpulines->entry[i], featString, 8) == BSTR_OK)
#endif
#ifdef _ARCH_PPC
	if (ret != 1)
#endif
        {
            bdestroy(flagline);
            flagline = bstrcpy(cpulines->entry[i]);
            break;
        }
    }
    bstrListDestroy(cpulines);

    struct bstrList* flaglist = bsplit(flagline, ' ');
    bstring bfeatures = bfromcstr("");

    cpuid_info.featureFlags = 0;

    for (int i = 1; i < flaglist->qty; i++)
    {
        if (bisstemeqblk(flaglist->entry[i], "sse4_1", 6) == 1)
        {
            setBit(cpuid_info.featureFlags, SSE41);
            bcatcstr(bfeatures, "SSE4.1 ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "sse4_2", 6) == 1)
        {
            setBit(cpuid_info.featureFlags, SSE42);
            bcatcstr(bfeatures, "SSE4.2 ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "sse4a", 6) == 1)
        {
            setBit(cpuid_info.featureFlags, SSE4A);
            bcatcstr(bfeatures, "SSE4a ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "ssse3", 5) == 1 && (!(testBit(cpuid_info.featureFlags, SSSE3))))
        {
            setBit(cpuid_info.featureFlags, SSSE3);
            bcatcstr(bfeatures, "SSSE ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "sse3", 4) == 1 && (!(testBit(cpuid_info.featureFlags, SSE3))))
        {
            setBit(cpuid_info.featureFlags, SSE3);
            bcatcstr(bfeatures, "SSE3 ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "sse2", 4) == 1 && (!(testBit(cpuid_info.featureFlags, SSE2))))
        {
            setBit(cpuid_info.featureFlags, SSE2);
            bcatcstr(bfeatures, "SSE2 ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "monitor", 7) == 1)
        {
            setBit(cpuid_info.featureFlags, MONITOR);
            bcatcstr(bfeatures, "MONITOR ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "mmx", 3) == 1)
        {
            setBit(cpuid_info.featureFlags, MMX);
            bcatcstr(bfeatures, "MMX ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "sse", 3) == 1 && (!(testBit(cpuid_info.featureFlags, SSE))))
        {
            setBit(cpuid_info.featureFlags, SSE);
            bcatcstr(bfeatures, "SSE ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "acpi", 4) == 1)
        {
            setBit(cpuid_info.featureFlags, ACPI);
            bcatcstr(bfeatures, "ACPI ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "rdtscp", 6) == 1)
        {
            setBit(cpuid_info.featureFlags, RDTSCP);
            bcatcstr(bfeatures, "RDTSCP ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "vmx", 3) == 1)
        {
            setBit(cpuid_info.featureFlags, VMX);
            bcatcstr(bfeatures, "VMX ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "est", 3) == 1)
        {
            setBit(cpuid_info.featureFlags, EIST);
            bcatcstr(bfeatures, "EIST ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "tm2", 3) == 1 && (!(testBit(cpuid_info.featureFlags, TM2))))
        {
            setBit(cpuid_info.featureFlags, TM2);
            bcatcstr(bfeatures, "TM2 ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "tm", 2) == 1 && (!(testBit(cpuid_info.featureFlags, TM))))
        {
            setBit(cpuid_info.featureFlags, TM);
            bcatcstr(bfeatures, "TM ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "aes", 3) == 1)
        {
            setBit(cpuid_info.featureFlags, AES);
            bcatcstr(bfeatures, "AES ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "rdrand", 6) == 1)
        {
            setBit(cpuid_info.featureFlags, RDRAND);
            bcatcstr(bfeatures, "RDRAND ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "rdseed", 6) == 1)
        {
            setBit(cpuid_info.featureFlags, RDSEED);
            bcatcstr(bfeatures, "RDSEED ");
        }
        else if ((bisstemeqblk(flaglist->entry[i], "avx512", 6) == 1) && (!(testBit(cpuid_info.featureFlags, AVX512))))
        {
            setBit(cpuid_info.featureFlags, AVX512);
            bcatcstr(bfeatures, "AVX512 ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "avx2", 4) == 1 && (!testBit(cpuid_info.featureFlags, AVX2)))
        {
            setBit(cpuid_info.featureFlags, AVX2);
            bcatcstr(bfeatures, "AVX2 ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "avx", 3) == 1 && (!testBit(cpuid_info.featureFlags, AVX)))
        {
            setBit(cpuid_info.featureFlags, AVX);
            bcatcstr(bfeatures, "AVX ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "fma", 3) == 1)
        {
            setBit(cpuid_info.featureFlags, FMA);
            bcatcstr(bfeatures, "FMA ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "rtm", 3) == 1)
        {
            setBit(cpuid_info.featureFlags, RTM);
            bcatcstr(bfeatures, "RTM ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "hle", 3) == 1)
        {
            setBit(cpuid_info.featureFlags, HLE);
            bcatcstr(bfeatures, "HLE ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "ht", 2) == 1 && (!testBit(cpuid_info.featureFlags, HTT)))
        {
            setBit(cpuid_info.featureFlags, HTT);
            bcatcstr(bfeatures, "HTT ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "fp", 2) == 1 && (!testBit(cpuid_info.featureFlags, FP)))
        {
            setBit(cpuid_info.featureFlags, FP);
            bcatcstr(bfeatures, "FP ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "swp", 3) == 1)
        {
            setBit(cpuid_info.featureFlags, SWP);
            bcatcstr(bfeatures, "SWP ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "vfpv3", 5) == 1)
        {
            setBit(cpuid_info.featureFlags, VFPV3);
            bcatcstr(bfeatures, "VFPV3 ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "vfpv4", 5) == 1)
        {
            setBit(cpuid_info.featureFlags, VFPV4);
            bcatcstr(bfeatures, "VFPV4 ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "vfp", 3) == 1 && (!testBit(cpuid_info.featureFlags, VFP)))
        {
            setBit(cpuid_info.featureFlags, VFP);
            bcatcstr(bfeatures, "VFP ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "neon", 4) == 1)
        {
            setBit(cpuid_info.featureFlags, NEON);
            bcatcstr(bfeatures, "NEON ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "edsp", 4) == 1)
        {
            setBit(cpuid_info.featureFlags, EDSP);
            bcatcstr(bfeatures, "EDSP ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "tls", 3) == 1)
        {
            setBit(cpuid_info.featureFlags, TLS);
            bcatcstr(bfeatures, "TLS ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "asimdrdm", 8) == 1 && (!testBit(cpuid_info.featureFlags, ASIMDRDM)))
        {
            setBit(cpuid_info.featureFlags, ASIMDRDM);
            bcatcstr(bfeatures, "ASIMDRDM ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "asimd", 5) == 1 && (!testBit(cpuid_info.featureFlags, ASIMD)))
        {
            setBit(cpuid_info.featureFlags, ASIMD);
            bcatcstr(bfeatures, "ASIMD ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "pmull", 5) == 1)
        {
            setBit(cpuid_info.featureFlags, PMULL);
            bcatcstr(bfeatures, "PMULL ");
        }
        else if (bisstemeqblk(flaglist->entry[i], "sve", 3) == 1)
        {
            setBit(cpuid_info.featureFlags, SVE);
            bcatcstr(bfeatures, "SVE ");
        }
    }

    if (testBit(cpuid_info.featureFlags, SSSE3) && (!(testBit(cpuid_info.featureFlags, SSE3))))
    {
        setBit(cpuid_info.featureFlags, SSE3);
        bcatcstr(bfeatures, "SSE3 ");
    }

    cpuid_info.features = (char*) malloc((blength(bfeatures)+2)*sizeof(char));
    ret = snprintf(cpuid_info.features, blength(bfeatures)+1, "%s", bdata(bfeatures));
    if (ret > 0)
    {
        cpuid_info.features[ret] = '\0';
    }
    bdestroy(bfeatures);
    bstrListDestroy(flaglist);

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
    int num_cores_per_die = 0;

    hwThreadPool = (HWThread*) malloc(cpuid_topology.numHWThreads * sizeof(HWThread));
    for (uint32_t i=0;i<cpuid_topology.numHWThreads;i++)
    {
        hwThreadPool[i].apicId = i;
        cpudir = bformat("/sys/devices/system/cpu/cpu%d/topology",i);
        hwThreadPool[i].threadId = -1;
        hwThreadPool[i].coreId = -1;
        hwThreadPool[i].packageId = -1;
        hwThreadPool[i].dieId = -1;
        hwThreadPool[i].inCpuSet = 0;
        if (CPU_ISSET(i, &cpuSet) && likwid_cpu_online(i))
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
        file = bformat("%s/die_id", bdata(cpudir));
        if (NULL != (fp = fopen (bdata(file), "r")))
        {
            bstring src = bread ((bNread) fread, fp);
            hwThreadPool[i].dieId = ownatoi(bdata(src));
            if (hwThreadPool[i].packageId == 0 && hwThreadPool[i].dieId == 0)
            {
                num_cores_per_die++;
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
        {
            hwThreadPool[i].packageId = 0;
            for (int j = 0; j < num_sockets*4; j++)
            {
                bstring nodestr = bformat("/sys/devices/system/cpu/cpu%d/node%d", i, j);
                if (!access(bdata(nodestr), F_OK))
                {
                    hwThreadPool[i].packageId = j;
                    int offset = 0;
                    for (int k = 0; k < i; k++)
                    {
                        bstring teststr = bformat("/sys/devices/system/cpu/cpu%d/node%d/cpu%d", i, j, k);
                        if (!access(bdata(teststr), F_OK))
                            offset++;
                        bdestroy(teststr);
                    }
                    if (hwThreadPool[i].coreId == -1)
                        hwThreadPool[i].coreId = offset;
                    break;
                }
                bdestroy(nodestr);
            }
        }
        if (hwThreadPool[i].coreId == -1)
            hwThreadPool[i].coreId = 0;
        if (hwThreadPool[i].threadId == -1)
            hwThreadPool[i].threadId = 0;
        if (hwThreadPool[i].dieId == -1)
            hwThreadPool[i].dieId = 0;
#endif
        DEBUG_PRINT(DEBUGLEV_DEVELOP, PROC Thread Pool PU %d Thread %d Core %d Die %d Socket %d inCpuSet %d,
                            hwThreadPool[i].apicId,
                            hwThreadPool[i].threadId,
                            hwThreadPool[i].coreId,
                            hwThreadPool[i].dieId,
                            hwThreadPool[i].packageId,
                            hwThreadPool[i].inCpuSet)
        bdestroy(cpudir);
    }
    int* helper = malloc(cpuid_topology.numHWThreads * sizeof(int));
    if (!helper)
    {
	    return;
    }
    cpuid_topology.threadPool = hwThreadPool;
    int hidx = 0;
    for (int i = 0; i < cpuid_topology.numHWThreads; i++)
    {
        int pid = hwThreadPool[i].packageId;
        int found = 0;
        for (int j = 0; j < hidx; j++)
        {
            if (pid == helper[j])
            {
                found = 1;
                break;
            }
        }
        if (!found)
        {
            helper[hidx++] = pid;
        }
    }
    cpuid_topology.numSockets = hidx;
    /* Traverse all sockets to get maximal thread count per socket.
     * This should fix the code for architectures with "empty" sockets.
     */
    int num_threads_per_socket = 0;
    for (int i = 0; i < cpuid_topology.numSockets; i++)
    {
        int threadCount = 0;
        for (int j = 0; j < cpuid_topology.numHWThreads; j++)
        {
            if (helper[i] == hwThreadPool[j].packageId)
            {
                threadCount++;
            }
        }
        if (threadCount > num_threads_per_socket)
        {
            num_threads_per_socket = threadCount;
        }
    }

    int first_socket_id = helper[0];
    hidx = 0;
    for (int i = 0; i < cpuid_topology.numHWThreads; i++)
    {
        int did = hwThreadPool[i].dieId;
        int pid = hwThreadPool[i].packageId;
        if (pid != first_socket_id) continue;
        int found = 0;
        for (int j = 0; j < hidx; j++)
        {
            if (did == helper[j])
            {
                found = 1;
                break;
            }
        }
        if (!found)
        {
            helper[hidx++] = did;
        }
    }

    cpuid_topology.numDies = hidx * cpuid_topology.numSockets;
    if (cpuid_topology.numDies == cpuid_topology.numSockets)
    {
        cpuid_topology.numDies = 0;
    }
    int max_thread_sibling_id = 0;
    for (int i = 0; i < cpuid_topology.numHWThreads; i++)
    {
        if (hwThreadPool[i].threadId > max_thread_sibling_id)
        {
            max_thread_sibling_id = hwThreadPool[i].threadId;
        }
    }
    num_threads_per_core = max_thread_sibling_id + 1;
    cpuid_topology.numCoresPerSocket = num_threads_per_socket/num_threads_per_core;
    cpuid_topology.numThreadsPerCore = num_threads_per_core;
    free(helper);
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
    while ((ep = readdir(dp)))
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
    while ((ep = readdir(dp)))
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
