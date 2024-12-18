/*
 * =======================================================================================
 *
 *      Filename:  numa_proc.c
 *
 *      Description:  Get NUMA topology from procfs and sysfs
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
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

#include <dirent.h>
#include <error.h>
//#include <strUtil.h>
#include <unistd.h>
#include <sys/syscall.h>
#ifdef HAS_MEMPOLICY
#include <linux/mempolicy.h>
#endif

#include <numa.h>
#include <topology.h>

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#ifdef HAS_MEMPOLICY
#define get_mempolicy(policy,nmask,maxnode,addr,flags) syscall(SYS_get_mempolicy,policy,nmask,maxnode,addr,flags)
#define set_mempolicy(mode,nmask,maxnode) syscall(SYS_set_mempolicy,mode,nmask,maxnode)
#define mbind(start, len, nmask, maxnode, flags) syscall(SYS_mbind,(start),len,MPOL_BIND,(nmask),maxnode,flags)
#endif

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int
proc_findProcessor(uint32_t nodeId, uint32_t coreId)
{
    int i = 0;

    for (i=0; i < (int)numa_info.nodes[nodeId].numberOfProcessors; i++)
    {
        if (numa_info.nodes[nodeId].processors[i] == coreId)
        {
            return 1;
        }
    }
    return 0;
}

static int
setConfiguredNodes(void)
{
    DIR *dir = NULL;
    struct dirent *de = NULL;
    int maxIdConfiguredNode = 0;

    dir = opendir("/sys/devices/system/node");

    if (!dir)
    {
        maxIdConfiguredNode = 0;
    }
    else
    {
        while ((de = readdir(dir)) != NULL)
        {
            int nd;
            if (strncmp(de->d_name, "node", 4))
            {
                continue;
            }

            nd = str2int(de->d_name+4);

            if (maxIdConfiguredNode < nd)
            {
                maxIdConfiguredNode = nd;
            }
        }
        closedir(dir);
    }
    return maxIdConfiguredNode;
}

static int
get_numaNodes(int* array, int maxlen)
{
    DIR *dir = NULL;
    struct dirent *de = NULL;
    int count = 0;

    dir = opendir("/sys/devices/system/node");

    if (!dir)
    {
        count = 0;
    }
    else
    {
        while ((de = readdir(dir)) != NULL)
        {
            if (strncmp(de->d_name, "node", 4))
            {
                continue;
            }
            if (array && count < maxlen)
            {
                int nd = str2int(de->d_name+4);
                array[count] = nd;
            }
            count++;
        }
    }
    if (array && count > 0)
    {
        int i = 0;
        int j = 0;
        while (i < count)
        {
            j = i;
            while (j > 0 && array[j-1] > array[j])
            {
                int tmp = array[j];
                array[j] = array[j-1];
                array[j-1] = tmp;
                j--;
            }
            i++;
        }
    }
    return count;
}

static void
nodeMeminfo(int node, uint64_t* totalMemory, uint64_t* freeMemory)
{
    FILE *fp;
    bstring filename;
    bstring totalString = bformat("MemTotal:");
    bstring freeString  = bformat("MemFree:");
    int i;

    filename = bformat("/sys/devices/system/node/node%d/meminfo", node);
    if (NULL != (fp = fopen (bdata(filename), "r")))
    {
        bstring src = bread ((bNread) fread, fp);
        struct bstrList* tokens = bsplit(src,(char) '\n');

        for (i=0;i<tokens->qty;i++)
        {
            if (binstr(tokens->entry[i],0,totalString) != BSTR_ERR)
            {
                 bstring tmp = bmidstr (tokens->entry[i], 18, blength(tokens->entry[i])-18 );
                 bltrimws(tmp);
                 struct bstrList* subtokens = bsplit(tmp,(char) ' ');
                 *totalMemory = str2int(bdata(subtokens->entry[0]));
                 bstrListDestroy(subtokens);
                 bdestroy(tmp);
            }
            else if (binstr(tokens->entry[i],0,freeString) != BSTR_ERR)
            {
                 bstring tmp = bmidstr (tokens->entry[i], 18, blength(tokens->entry[i])-18  );
                 bltrimws(tmp);
                 struct bstrList* subtokens = bsplit(tmp,(char) ' ');
                 *freeMemory = str2int(bdata(subtokens->entry[0]));
                 bstrListDestroy(subtokens);
                 bdestroy(tmp);
            }
        }
        bdestroy(src);
        bstrListDestroy(tokens);
    }
    else
    {
        bdestroy(filename);
        bdestroy(totalString);
        bdestroy(freeString);
        ERROR;
    }
    bdestroy(filename);
    bdestroy(totalString);
    bdestroy(freeString);
    fclose(fp);
}

static int
nodeProcessorList(int node, uint32_t** list)
{
    FILE *fp;
    bstring filename;
    uint32_t count = 0;
    bstring src;
    int i = 0, j = 0;
    struct bstrList* tokens;
    unsigned long val;
    char* endptr;
    int cursor=0;
//    int unitSize = (int) (sizeof(unsigned long)*8);
    int unitSize = (int) 32; /* 8 nibbles */

    *list = (uint32_t*) malloc(cpuid_topology.numHWThreads * sizeof(uint32_t));
    if (!(*list))
    {
        return -ENOMEM;
    }

    /* the cpumap interface should be always there */
    filename = bformat("/sys/devices/system/node/node%d/cpumap", node);

    if (NULL != (fp = fopen (bdata(filename), "r")))
    {
        src = bread ((bNread) fread, fp);
        tokens = bsplit(src,',');

        for (i=(int)(tokens->qty-1); i >= 0 ; i--)
        {
            val = strtoul((char*) tokens->entry[i]->data, &endptr, 16);

            if ((errno != 0 && val == LONG_MAX )
                    || (errno != 0 && val == 0))
            {
                return -EFAULT;
            }

            if (endptr == (char*) tokens->entry[i]->data)
            {
                ERROR_PRINT("No digits were found");
                return -EFAULT;
            }

            if (val != 0UL)
            {
                for (j=0; j<unitSize; j++)
                {
                    if (val&(1UL<<j))
                    {
                        if (count < cpuid_topology.numHWThreads)
                        {
                            (*list)[count] = (j+cursor);
                        }
                        else
                        {
                            ERROR_PRINT("Number Of threads %d too large for NUMA node %d", count, node);
                            return -EFAULT;
                        }
                        count++;
                    }
                }
            }
            cursor += unitSize;
        }

        bstrListDestroy(tokens);
        bdestroy(src);
        bdestroy(filename);
        fclose(fp);

        /* FIXME: CPU list here is not physical cores first but numerical sorted */
        return count;
    }

    /* something went wrong */
    return -1;
}

static int
nodeDistanceList(int node, int numberOfNodes, uint32_t** list)
{
    FILE *fp;
    bstring filename;
    int count = 0;
    bstring src;
    struct bstrList* tokens;

    *list = (uint32_t*) malloc(numberOfNodes * sizeof(uint32_t));
    if (!(*list))
    {
        return -ENOMEM;
    }

    /* the distance interface should be always there */
    filename = bformat("/sys/devices/system/node/node%d/distance", node);

    if (NULL != (fp = fopen (bdata(filename), "r")))
    {

        src = bread ((bNread) fread, fp);
        tokens = bsplit(src,' ');

        for (int i=0; i<(tokens->qty); i++)
        {
            if (count < numberOfNodes)
            {
                (*list)[count] = (uint32_t)strtoul((char*) (tokens->entry[i]->data), NULL, 10);
            }
            else
            {
                ERROR_PRINT("Number Of nodes %d too large", count);
                return -EFAULT;
            }
            count++;
        }

        bstrListDestroy(tokens);
        bdestroy(src);
        bdestroy(filename);
        fclose(fp);
        return count;
    }

    /* something went wrong */
    return -1;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int proc_numa_init(void)
{
    int err = 0;
    uint32_t i;
    uint64_t nrCPUs = 0;
    if (numaInitialized > 0 || numa_info.numberOfNodes > 0)
        return 0;

    if (get_mempolicy(NULL, NULL, 0, 0, 0) < 0 && errno == ENOSYS)
    {
        /* Allocate a virtual node instead and bail out */
        return virtual_numa_init();
    }
    /* First determine maximum number of nodes */
    //numa_info.numberOfNodes = setConfiguredNodes()+1;
    numa_info.numberOfNodes= get_numaNodes(NULL, 10000);
    int* nodes = malloc(numa_info.numberOfNodes*sizeof(int));
    if (!nodes)
    {
        return -ENOMEM;
    }
    numa_info.numberOfNodes = get_numaNodes(nodes, numa_info.numberOfNodes);
    numa_info.nodes = (NumaNode*) malloc(numa_info.numberOfNodes * sizeof(NumaNode));
    if (!numa_info.nodes)
    {
        return -ENOMEM;
    }

    for (i=0; i<numa_info.numberOfNodes; i++)
    {
        int id = nodes[i];
        numa_info.nodes[i].id = id;
        nodeMeminfo(id, &numa_info.nodes[i].totalMemory, &numa_info.nodes[i].freeMemory);
        numa_info.nodes[i].numberOfProcessors = nodeProcessorList(id, &numa_info.nodes[i].processors);
/*        int check = 0;*/
/*        for (int j = 0; j < numa_info.nodes[i].numberOfProcessors; j++)*/
/*        {*/
/*            int id = numa_info.nodes[i].processors[j];*/
/*            if (cpuid_topology.threadPool[id].inCpuSet == 1)*/
/*            {*/
/*                numa_info.nodes[i].processors[check++] = numa_info.nodes[i].processors[j];*/
/*            }*/
/*        }*/
/*        if (check < numa_info.nodes[i].numberOfProcessors)*/
/*            numa_info.nodes[i].numberOfProcessors = check;*/
        numa_info.nodes[i].numberOfDistances = nodeDistanceList(id, numa_info.numberOfNodes, &numa_info.nodes[i].distances);
        if (numa_info.nodes[i].numberOfDistances == 0)
        {
            err = -EFAULT;
            break;
        }
    }
    for (; i<numa_info.numberOfNodes; i++)
    {
        int id = nodes[i];
        numa_info.nodes[i].numberOfProcessors = 0;
        numa_info.nodes[i].numberOfDistances = nodeDistanceList(id, numa_info.numberOfNodes, &numa_info.nodes[i].distances);
    }
    if (err == 0)
        numaInitialized = 1;
    return err;
}

void
proc_numa_setInterleaved(const int* processorList, int numberOfProcessors)
{
    long i;
    int j;
    int ret=0;
    unsigned long numberOfNodes = 65;
    unsigned long mask = 0UL;

    for (i=0; i<numa_info.numberOfNodes; i++)
    {
        for (j=0; j<numberOfProcessors; j++)
        {
            if (proc_findProcessor(i,processorList[j]))
            {
                mask |= (1UL<<i);
                break;
            }
        }
    }

    ret = set_mempolicy(MPOL_INTERLEAVE,&mask,numberOfNodes);

    if (ret < 0)
    {
        ERROR;
    }
}

void
proc_numa_setMembind(const int* processorList, int numberOfProcessors)
{
    long i;
    int j;
    int ret=0;
    unsigned long numberOfNodes = 65;
    unsigned long mask = 0UL;

    for (i=0; i<numa_info.numberOfNodes; i++)
    {
        for (j=0; j<numberOfProcessors; j++)
        {
            if (proc_findProcessor(i,processorList[j]))
            {
                mask |= (1UL<<i);
                break;
            }
        }
    }

    ret = set_mempolicy(MPOL_BIND, &mask, numberOfNodes);

    if (ret < 0)
    {
        ERROR;
    }
}

void
proc_numa_membind(void* ptr, size_t size, int domainId)
{
    int ret=0;
    unsigned long mask = 0UL;
    unsigned int flags = 0U;

    flags |= MPOL_MF_STRICT;
    mask |= (1UL<<domainId);

    ret = mbind(ptr, size, &mask, numa_info.numberOfNodes+1, flags);

    if (ret < 0)
    {
        ERROR;
    }
}

