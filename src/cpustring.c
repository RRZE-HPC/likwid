/*
 * =======================================================================================
 *
 *      Filename:  cpustring.c
 *
 *      Description:  Parser for CPU selection strings
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
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
#include <math.h>

#include <likwid.h>

#define MAX(a, b)  (((a) > (b)) ? (a) : (b))
#define MIN(a, b)  (((a) < (b)) ? (a) : (b))


/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int
check_and_atoi(char* s)
{
    int i = 0;
    int len = strlen(s);
    for (i = 0; i < len ; i++)
    {
        if (s[i] < '0' || s[i] > '9')
            return -1;
    }
    return atoi(s);
}

static int
cpulist_sort(int* incpus, int* outcpus, int length)
{
    int insert = 0;
    topology_init();
    CpuTopology_t cpuid_topology = get_cpuTopology();
    if (length <= 0)
    {
        return -1;
    }
    for (int k = 0; k < cpuid_topology->numThreadsPerCore; k++)
    {
        for (int i = 0; i < length; i++)
        {
            int idx = -1;
            for (int j = 0; j < cpuid_topology->numHWThreads; j++)
            {
                if (cpuid_topology->threadPool[j].apicId == incpus[i])
                {
                    idx = j;
                    break;
                }
            }
            if (idx >= 0 && cpuid_topology->threadPool[idx].threadId == k)
            {
                outcpus[insert] = incpus[i];
                insert++;
            }
            if (insert == length) break;
        }
        if (insert == length) break;
    }
    return insert;
}

static int
cpulist_concat(int* cpulist, int startidx, int* addlist, int addlength)
{
    int count = 0;
    if (addlength <= 0)
    {
        return 0;
    }
    for (int i=startidx;i<(startidx+addlength);i++)
    {
        cpulist[i] = addlist[i-startidx];
        count++;
    }
    return count;
}

static int
cpu_in_domain(int domainidx, int cpu)
{
    affinity_init();
    AffinityDomains_t affinity = get_affinityDomains();
    for (int i=0;i<affinity->domains[domainidx].numberOfProcessors; i++)
    {
        if (cpu == affinity->domains[domainidx].processorList[i])
        {
            return 1;
        }
    }
    return 0;
}

static int
get_domain_idx(bstring bdomain)
{
    affinity_init();
    AffinityDomains_t affinity = get_affinityDomains();
    for (int i=0;i<affinity->numberOfAffinityDomains; i++)
    {
        if (bstrcmp(affinity->domains[i].tag, bdomain) == BSTR_OK)
        {
            return i;
        }
    }
    return -1;
}

static int
cpuexpr_to_list(bstring bcpustr, bstring prefix, int* list, int length)
{
    topology_init();
    CpuTopology_t cpuid_topology = get_cpuTopology();
    affinity_init();
    AffinityDomains_t affinity = get_affinityDomains();
    struct bstrList* strlist;
    strlist = bsplit(bcpustr, ',');
    int oldinsert = 0;
    int insert = 0;
    int tmp = 0;
    for (int i=0;i < strlist->qty; i++)
    {
        bstring newstr = bstrcpy(prefix);
        bconcat(newstr, strlist->entry[i]);
        oldinsert = insert;
        for (int j = 0; j < affinity->numberOfAffinityDomains; j++)
        {
            if (bstrcmp(affinity->domains[j].tag, newstr) == 0)
            {
                tmp = check_and_atoi(bdata(strlist->entry[i]));
                if (tmp >= 0)
                {
                    list[insert] = tmp;
                    insert++;
                }
                if (insert == length)
                    goto list_done;
                break;
            }
        }
        if (insert == oldinsert)
        {
            fprintf(stderr,"Domain %s cannot be found\n", bdata(newstr));
        }
        bdestroy(newstr);
    }
list_done:
    bstrListDestroy(strlist);
    return insert;
}


static int
cpustr_to_cpulist_method(bstring bcpustr, int* cpulist, int length)
{
    int max_procs = 0;
    int given_procs = 0;
    topology_init();
    CpuTopology_t cpuid_topology = get_cpuTopology();
    affinity_init();
    AffinityDomains_t affinity = get_affinityDomains();
    bstring scattercheck = bformat("scatter");
    bstring balancedcheck = bformat("balanced");
    bstring cbalancedcheck = bformat("cbalanced");
    char* cpustring = bstr2cstr(bcpustr, '\0');
    if (bstrchrp(bcpustr, ':', 0) != BSTR_ERR)
    {
        int insert = 0;
        int suitidx = 0;
        struct bstrList* parts = bstrListCreate();
        parts = bsplit(bcpustr, ':');
        if (parts->qty == 3)
        {
            int tmp = check_and_atoi(bdata(parts->entry[2]));
            if (tmp > 0)
                given_procs = tmp;
        }
        int* suitable = (int*)malloc(affinity->numberOfAffinityDomains*sizeof(int));
        if (!suitable)
        {
            bcstrfree(cpustring);
            return -ENOMEM;
        }
        for (int i=0; i<affinity->numberOfAffinityDomains; i++)
        {
            if (binstr(affinity->domains[i].tag, 0, parts->entry[0]) != BSTR_ERR &&
                affinity->domains[i].numberOfProcessors > 0)
            {
                suitable[suitidx] = i;
                suitidx++;
                if (affinity->domains[i].numberOfProcessors > max_procs)
                    max_procs = affinity->domains[i].numberOfProcessors;
            }
        }
        int** sLists = (int**) malloc(suitidx * sizeof(int*));
        if (!sLists)
        {
            free(suitable);
            bcstrfree(cpustring);
            return -ENOMEM;
        }
        for (int i = 0; i< suitidx; i++)
        {
            sLists[i] = (int*) malloc(max_procs * sizeof(int));
            if (!sLists[i])
            {
                free(suitable);
                for (int j=0; j<i; j++)
                {
                    free(sLists[j]);
                }
                bcstrfree(cpustring);
                return -ENOMEM;
            }
            cpulist_sort(affinity->domains[suitable[i]].processorList, sLists[i], affinity->domains[suitable[i]].numberOfProcessors);
        }
        if (binstr(bcpustr, 0, scattercheck) != BSTR_ERR)
        {
            if (given_procs > 0)
                length = given_procs;
            for (int off=0;off<max_procs;off++)
            {
                for(int i=0;i < suitidx; i++)
                {
                    cpulist[insert] = sLists[i][off];
                    insert++;
                    if (insert == length)
                        goto method_done;
                }
            }
        }
        else if (binstr(bcpustr, 0, cbalancedcheck) != BSTR_ERR)
        {
            if (given_procs > 0)
                length = given_procs;
            else
                length = max_procs * suitidx;
            int per_domain = length/suitidx;
            int remain = length % suitidx;
            int coresAllDomains = (max_procs*suitidx)/cpuid_topology->numThreadsPerCore;
            int coresPerDomain = coresAllDomains/suitidx;
            int threadsPerCore = cpuid_topology->numThreadsPerCore;
            if ((per_domain+remain) > coresPerDomain)
            {
                for(int i=0;i < suitidx; i++)
                {
                    int with_ht = ((per_domain+remain)-coresPerDomain)*threadsPerCore;
                    for (int j = 0; j < with_ht; j++)
                    {
                        int cpu = affinity->domains[suitable[i]].processorList[j];
                        cpulist[insert] = cpu;
                        insert++;
                        for (int k=0; k< max_procs;k++)
                        {
                            if (sLists[i][k] == cpu)
                            {
                                sLists[i][k] = -1;
                                break;
                            }
                        }
                    }
                    int wo_ht = (per_domain+remain) - with_ht;
                    for (int j = 0; j < wo_ht; j++)
                    {
                        if (sLists[i][j] >= 0)
                        {
                            if (remain > 0)
                            {
                                remain--;
                            }
                            int cpu = sLists[i][j];
                            cpulist[insert] = cpu;
                            insert++;
                            for (int k=0; k< max_procs;k++)
                            {
                                if (sLists[i][k] == cpu)
                                {
                                    sLists[i][k] = -1;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            wo_ht++;
                        }
                    }
                }
            }
            else
            {
                for(int i=0;i < suitidx; i++)
                {
                    int new_domain = per_domain;
                    if (remain > 0)
                    {
                        new_domain++;
                        remain--;
                    }
                    for (int j = 0; j < new_domain; j++)
                    {
                        cpulist[insert] = sLists[i][j];
                        insert++;
                        if (insert == length)
                            goto method_done;
                    }
                }
            }
        }
        else if (binstr(bcpustr, 0, balancedcheck) != BSTR_ERR)
        {
            if (given_procs > 0)
                length = given_procs;
            else
                length = max_procs * suitidx;
            int per_domain = length/suitidx;
            int remain = length % suitidx;
            for(int i=0;i < suitidx; i++)
            {
                int new_domain = per_domain;
                if (remain > 0)
                {
                    new_domain++;
                    remain--;
                }
                for (int j = 0; j < new_domain; j++)
                {
                    cpulist[insert] = affinity->domains[suitable[i]].processorList[j];
                    insert++;
                    if (insert == length)
                        goto method_done;
                }
            }
        }
method_done:
        bcstrfree(cpustring);
        for (int i = 0; i< suitidx; i++)
        {
            free(sLists[i]);
        }
        free(sLists);
        free(suitable);
        return insert;
    }
    else
    {
        fprintf(stderr, "Not a valid CPU expression\n");
    }
    bcstrfree(cpustring);
    return 0;
}

static int
cpustr_to_cpulist_expression(bstring bcpustr, int* cpulist, int length)
{
    topology_init();
    CpuTopology_t cpuid_topology = get_cpuTopology();
    affinity_init();
    AffinityDomains_t affinity = get_affinityDomains();
    bstring bdomain;
    int domainidx = -1;
    int count = 0;
    int stride = 0;
    int chunk = 0;
    int off = 0;
    if (bstrchrp(bcpustr, 'E', 0) != 0)
    {
        fprintf(stderr, "Not a valid CPU expression\n");
        return 0;
    }
    struct bstrList* strlist;
    strlist = bsplit(bcpustr, ':');
    if (strlist->qty == 2)
    {
        bdomain = bstrcpy(strlist->entry[1]);
        count = cpuid_topology->activeHWThreads;
        stride = 1;
        chunk = 1;
    }
    else if (strlist->qty == 3)
    {
        bdomain = bstrcpy(strlist->entry[1]);
        count = check_and_atoi(bdata(strlist->entry[2]));
        stride = 1;
        chunk = 1;
    }
    else if (strlist->qty == 5)
    {
        bdomain = bstrcpy(strlist->entry[1]);
        count = check_and_atoi(bdata(strlist->entry[2]));
        chunk = check_and_atoi(bdata(strlist->entry[3]));
        stride = check_and_atoi(bdata(strlist->entry[4]));
    }
    else if (strlist->qty == 6)
    {
        bdomain = bstrcpy(strlist->entry[1]);
        count = check_and_atoi(bdata(strlist->entry[2]));
        chunk = check_and_atoi(bdata(strlist->entry[3]));
        stride = check_and_atoi(bdata(strlist->entry[4]));
        off = check_and_atoi(bdata(strlist->entry[5]));
    }
    if (count < 0 || chunk < 0 || stride < 0 || off < 0)
    {
        fprintf(stderr, "CPU expression contains non-numerical characters\n");
        bdestroy(bdomain);
        bstrListDestroy(strlist);
        return 0;
    }
    domainidx = get_domain_idx(bdomain);
    if (domainidx < 0)
    {
        fprintf(stderr, "Cannot find domain %s\n", bdata(bdomain));
        bdestroy(bdomain);
        bstrListDestroy(strlist);
        return 0;
    }
    int offset = 0;
    int insert = 0;
    for (int i=0;i<count;i++)
    {
        for (int j=0; j<chunk && offset+j<affinity->domains[domainidx].numberOfProcessors;j++)
        {
            cpulist[insert] = affinity->domains[domainidx].processorList[off + offset + j];
            insert++;
            if (insert == length || insert == count)
                goto expression_done;
        }
        offset += stride;
        if (off+offset >= affinity->domains[domainidx].numberOfProcessors)
        {
            offset = 0;
        }
        if (insert >= count)
            goto expression_done;
    }
    bdestroy(bdomain);
    bstrListDestroy(strlist);
    return 0;
expression_done:
    bdestroy(bdomain);
    bstrListDestroy(strlist);
    return insert;
}

static int
cpustr_to_cpulist_logical(bstring bcpustr, int* cpulist, int length)
{
    topology_init();
    CpuTopology_t cpuid_topology = get_cpuTopology();
    affinity_init();
    AffinityDomains_t affinity = get_affinityDomains();
    int domainidx = -1;
    bstring bdomain;
    bstring blist;
    struct bstrList* strlist;
    if (bstrchrp(bcpustr, 'L', 0) != 0)
    {
        fprintf(stderr, "ERROR: Not a valid CPU expression\n");
        return 0;
    }

    strlist = bsplit(bcpustr, ':');
    if (strlist->qty != 3)
    {
        fprintf(stderr, "ERROR: Invalid expression, should look like <domain>:<indexlist> or L:<domain>:<indexlist>\n");
        bstrListDestroy(strlist);
        return 0;
    }
    if (blength(strlist->entry[2]) == 0)
    {
        fprintf(stderr, "ERROR: Invalid expression, should look like <domain>:<indexlist> or L:<domain>:<indexlist>\n");
        bstrListDestroy(strlist);
        return 0;
    }
    bdomain = bstrcpy(strlist->entry[1]);
    blist = bstrcpy(strlist->entry[2]);
    bstrListDestroy(strlist);
    domainidx = get_domain_idx(bdomain);
    if (domainidx < 0)
    {
        fprintf(stderr, "ERROR: Cannot find domain %s\n", bdata(bdomain));
        bdestroy(bdomain);
        bdestroy(blist);
        return 0;
    }

    int *inlist = malloc(affinity->domains[domainidx].numberOfProcessors * sizeof(int));
    if (inlist == NULL)
    {
        bdestroy(bdomain);
        bdestroy(blist);
        return -ENOMEM;
    }

    int ret = cpulist_sort(affinity->domains[domainidx].processorList,
            inlist, affinity->domains[domainidx].numberOfProcessors);

    strlist = bsplit(blist, ',');
    int insert = 0;
    int insert_offset = 0;
    int inlist_offset = 0;
    int inlist_idx = 0;
    int require = 0;
    for (int i=0; i< strlist->qty; i++)
    {
        if (bstrchrp(strlist->entry[i], '-', 0) != BSTR_ERR)
        {
            struct bstrList* indexlist;
            indexlist = bsplit(strlist->entry[i], '-');
            int start = check_and_atoi(bdata(indexlist->entry[0]));
            int end = check_and_atoi(bdata(indexlist->entry[1]));
            if (start < 0 || end < 0)
            {
                fprintf(stderr, "CPU expression %s contains non-numerical characters\n",
                                 bdata(strlist->entry[i]));
                bstrListDestroy(indexlist);
                continue;
            }
            if (start <= end)
            {
                require += end - start + 1;
            }
            else
            {
                require += start - end + 1;
            }
            bstrListDestroy(indexlist);
        }
        else
        {
            require++;
        }
    }
    if (require > ret && getenv("LIKWID_SILENT") == NULL)
    {
        fprintf(stderr,
                "WARN: Selected affinity domain %s has only %d hardware threads, but selection string evaluates to %d threads.\n",
                bdata(affinity->domains[domainidx].tag), ret, require);
        fprintf(stderr, "      This results in multiple threads on the same hardware thread.\n");
        return 0;
    }
logical_redo:
    for (int i=0; i< strlist->qty; i++)
    {
        if (bstrchrp(strlist->entry[i], '-', 0) != BSTR_ERR)
        {
            struct bstrList* indexlist;
            indexlist = bsplit(strlist->entry[i], '-');
            int start = check_and_atoi(bdata(indexlist->entry[0]));
            int end = check_and_atoi(bdata(indexlist->entry[1]));
            if (start < 0 || end < 0)
            {
                fprintf(stderr, "CPU expression %s contains non-numerical characters\n",
                                 bdata(strlist->entry[i]));
                bstrListDestroy(indexlist);
                continue;
            }
            if (start <= end)
            {
                for (int j=start; j<=end && (insert_offset+insert < require); j++)
                {
                    inlist_idx = j;
                    cpulist[insert_offset + insert] = inlist[inlist_idx % ret];
                    insert++;
                    if (insert == ret)
                    {
                        bstrListDestroy(indexlist);
                        if (insert == require)
                            goto logical_done;
                        else
                            goto logical_redo;
                    }
                }
            }
            else
            {
                int start = check_and_atoi(bdata(indexlist->entry[0]));
                int end = check_and_atoi(bdata(indexlist->entry[1]));
                if (start < 0 || end < 0)
                {
                    fprintf(stderr, "CPU expression %s contains non-numerical characters\n",
                                     bdata(strlist->entry[i]));
                    bstrListDestroy(indexlist);
                    continue;
                }
                for (int j=start; j>=end && (insert_offset+insert < require); j--)
                {
                    inlist_idx = j;
                    cpulist[insert_offset + insert] = inlist[inlist_idx % ret];
                    insert++;
                    if (insert == ret)
                    {
                        bstrListDestroy(indexlist);
                        if (insert == require)
                            goto logical_done;
                        else
                            goto logical_redo;
                    }
                }
            }
            bstrListDestroy(indexlist);
        }
        else
        {
            int cpu = check_and_atoi(bdata(strlist->entry[i]));
            if (cpu < 0)
            {
                fprintf(stderr, "CPU expression %s contains non-numerical characters\n",
                                 bdata(strlist->entry[i]));
            }
            cpulist[insert_offset + insert] = inlist[cpu % ret];
            insert++;
            if (insert == ret)
            {
                if (insert == require)
                    goto logical_done;
                else
                    goto logical_redo;
            }
        }
    }
logical_done:
    bdestroy(bdomain);
    bdestroy(blist);
    bstrListDestroy(strlist);
    free(inlist);
    return require;
}

static int
cpustr_to_cpulist_physical(bstring bcpustr, int* cpulist, int length)
{
    topology_init();
    CpuTopology_t cpuid_topology = get_cpuTopology();
    affinity_init();
    AffinityDomains_t affinity = get_affinityDomains();
    bstring bdomain;
    bstring blist;
    int domainidx = -1;
    struct bstrList* strlist;
    if (bstrchrp(bcpustr, ':', 0) != BSTR_ERR)
    {
        strlist = bsplit(bcpustr, ':');
        bdomain = bstrcpy(strlist->entry[0]);
        blist = bstrcpy(strlist->entry[1]);
        bstrListDestroy(strlist);
    }
    else
    {
        bdomain = bformat("N");
        blist = bstrcpy(bcpustr);
    }
    domainidx = get_domain_idx(bdomain);
    if (domainidx < 0)
    {
        fprintf(stderr, "Cannot find domain %s\n", bdata(bdomain));
        bdestroy(bdomain);
        bdestroy(blist);
        return 0;
    }
    bstring domtag = affinity->domains[domainidx].tag;

    strlist = bsplit(blist, ',');
    int insert = 0;
    for (int i=0;i< strlist->qty; i++)
    {
        if (bstrchrp(strlist->entry[i], '-', 0) != BSTR_ERR)
        {
            struct bstrList* indexlist;
            indexlist = bsplit(strlist->entry[i], '-');
            int start = check_and_atoi(bdata(indexlist->entry[0]));
            int end = check_and_atoi(bdata(indexlist->entry[1]));
            if (start < 0 || end < 0)
            {
                fprintf(stderr, "CPU expression %s contains non-numerical characters\n",
                                 bdata(strlist->entry[i]));
                bstrListDestroy(indexlist);
                continue;
            }
            if (start <= end)
            {
                for (int j = start; j <= end; j++)
                {
                    if (cpu_in_domain(domainidx, j))
                    {
                        cpulist[insert] = j;
                        insert++;
                        if (insert == length)
                        {
                            bstrListDestroy(indexlist);
                            goto physical_done;
                        }
                    }
                    else
                    {
                        int notInCpuSet = 0;
                        for (int k=0;k<cpuid_topology->numHWThreads;k++)
                        {
                            if (cpuid_topology->threadPool[k].apicId == j && !cpuid_topology->threadPool[k].inCpuSet)
                            {
                                notInCpuSet = 1;
                            }
                        }
                        fprintf(stderr, "CPU %d not in domain %s.", j, bdata(domtag));
                        if (notInCpuSet)
                        {
                            fprintf(stderr, " It is not in the given cpuset\n");
                        }
                        else
                        {
                            fprintf(stderr, "\n");
                        }
                    }
                }
            }
            else
            {
                for (int j = start; j >= end; j--)
                {
                    if (cpu_in_domain(domainidx, j))
                    {
                        cpulist[insert] = j;
                        insert++;
                        if (insert == length)
                        {
                            bstrListDestroy(indexlist);
                            goto physical_done;
                        }
                    }
                    else
                    {
                        int notInCpuSet = 0;
                        for (int k=0;k<cpuid_topology->numHWThreads;k++)
                        {
                            if (cpuid_topology->threadPool[k].apicId == j && !cpuid_topology->threadPool[k].inCpuSet)
                            {
                                notInCpuSet = 1;
                            }
                        }
                        fprintf(stderr, "CPU %d not in domain %s.", j, bdata(domtag));
                        if (notInCpuSet)
                        {
                            fprintf(stderr, " It is not in the given cpuset\n");
                        }
                        else
                        {
                            fprintf(stderr, "\n");
                        }
                    }
                }
            }
            bstrListDestroy(indexlist);
        }
        else
        {
            int cpu = check_and_atoi(bdata(strlist->entry[i]));
            if (cpu < 0)
            {
                fprintf(stderr, "CPU expression %s contains non-numerical characters\n",
                                 bdata(strlist->entry[i]));
            }
            else if (cpu_in_domain(domainidx, cpu))
            {
                cpulist[insert] = cpu;
                insert++;
                if (insert == length)
                {
                    goto physical_done;
                }
            }
            else
            {
                int notInCpuSet = 0;
                for (int k=0;k<cpuid_topology->numHWThreads;k++)
                {
                    if (cpuid_topology->threadPool[k].apicId == cpu && !cpuid_topology->threadPool[k].inCpuSet)
                    {
                        notInCpuSet = 1;
                    }
                }
                fprintf(stderr, "CPU %d not in domain %s.", cpu, bdata(domtag));
                if (notInCpuSet)
                {
                    fprintf(stderr, " It is not in the given cpuset\n");
                }
                else
                {
                    fprintf(stderr, "\n");
                }
            }
        }
    }
physical_done:
    bstrListDestroy(strlist);
    bdestroy(bdomain);
    bdestroy(blist);
    return insert;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
cpustr_to_cpulist(const char* cpustring, int* cpulist, int length)
{
    int insert = 0;
    int len = 0;
    int ret = 0;
    bstring bcpustr = bfromcstr(cpustring);
    struct bstrList* strlist;
    bstring scattercheck = bformat("scatter");
    bstring balancedcheck = bformat("balanced");
    topology_init();
    CpuTopology_t cpuid_topology = get_cpuTopology();
    strlist = bsplit(bcpustr, '@');

    int* tmpList = (int*)malloc(length * sizeof(int));
    if (tmpList == NULL)
    {
        bstrListDestroy(strlist);
        bdestroy(scattercheck);
        bdestroy(balancedcheck);
        bdestroy(bcpustr);
        return -ENOMEM;
    }
    memset(tmpList, 0, length * sizeof(int));
    for (int i=0; i< strlist->qty; i++)
    {
        if (binstr(strlist->entry[i], 0, scattercheck) != BSTR_ERR ||
            binstr(strlist->entry[i], 0, balancedcheck) != BSTR_ERR)
        {
            ret = cpustr_to_cpulist_method(strlist->entry[i], tmpList, length);
            insert += cpulist_concat(cpulist, insert, tmpList, ret);
        }
        else if (bstrchrp(strlist->entry[i], 'E', 0) == 0)
        {
            ret = cpustr_to_cpulist_expression(strlist->entry[i], tmpList, length);
            insert += cpulist_concat(cpulist, insert, tmpList, ret);
        }
        else if (bstrchrp(strlist->entry[i], 'L', 0) == 0)
        {
            ret = cpustr_to_cpulist_logical(strlist->entry[i], tmpList, length);
            insert += cpulist_concat(cpulist, insert, tmpList, ret);
        }

        else if ((bstrchrp(strlist->entry[i], 'N', 0) == 0) ||
                 (bstrchrp(strlist->entry[i], 'S', 0) == 0) ||
                 (bstrchrp(strlist->entry[i], 'C', 0) == 0) ||
                 (bstrchrp(strlist->entry[i], 'M', 0) == 0) ||
                 (bstrchrp(strlist->entry[i], 'D', 0) == 0))
        {
            if (bstrchrp(strlist->entry[i], ':', 0) != BSTR_ERR)
            {
                bstring newstr = bformat("L:");
                bconcat(newstr, strlist->entry[i]);
                ret = cpustr_to_cpulist_logical(newstr, tmpList, length);
                insert += cpulist_concat(cpulist, insert, tmpList, ret);
                bdestroy(newstr);
            }
            else
            {
                int dom = get_domain_idx(strlist->entry[i]);
                if (dom >= 0)
                {
                    AffinityDomains_t affinity = get_affinityDomains();
                    bstring newstr = bformat("E:");
                    bconcat(newstr, strlist->entry[i]);
                    length = MIN(length, affinity->domains[dom].numberOfProcessors);
                    ret = cpustr_to_cpulist_expression(newstr, tmpList, length);
                    insert += cpulist_concat(cpulist, insert, tmpList, ret);
                    bdestroy(newstr);
                }
                else
                {
                    fprintf(stderr, "Cannot find domain %s\n", bdata(strlist->entry[i]));
                    continue;
                }
            }
        }
        else
        {
            if (cpuid_topology->activeHWThreads < cpuid_topology->numHWThreads)
            {
                if (getenv("LIKWID_SILENT") == NULL)
                {
                    fprintf(stdout,
                        "INFO: You are running LIKWID in a cpuset with %d CPUs. Taking given IDs as logical ID in cpuset\n",
                        cpuid_topology->activeHWThreads);
                }
                bstring newstr = bformat("L:N:");
                bconcat(newstr, strlist->entry[i]);
                ret = cpustr_to_cpulist_logical(newstr, tmpList, length);
                insert += cpulist_concat(cpulist, insert, tmpList, ret);
                bdestroy(newstr);
            }
            else
            {
                ret = cpustr_to_cpulist_physical(strlist->entry[i], tmpList, length);
                insert += cpulist_concat(cpulist, insert, tmpList, ret);
            }
        }
    }
    free(tmpList);
    bdestroy(bcpustr);
    bdestroy(scattercheck);
    bdestroy(balancedcheck);
    bstrListDestroy(strlist);
    return insert;
}

int
nodestr_to_nodelist(const char* nodestr, int* nodes, int length)
{
    int ret = 0;
    bstring prefix = bformat("M");
    bstring bnodestr = bfromcstr(nodestr);
    ret = cpuexpr_to_list(bnodestr, prefix, nodes, length);
    bdestroy(bnodestr);
    bdestroy(prefix);
    return ret;
}

int
sockstr_to_socklist(const char* sockstr, int* sockets, int length)
{
    int ret = 0;
    bstring prefix = bformat("S");
    bstring bsockstr = bfromcstr(sockstr);
    ret = cpuexpr_to_list(bsockstr, prefix, sockets, length);
    bdestroy(bsockstr);
    bdestroy(prefix);
    return ret;
}

#ifdef LIKWID_WITH_NVMON

static int valid_gpu_nvmon(GpuTopology_t topo, int id)
{
    for (int i = 0; i < topo->numDevices; i++)
    {
        if (topo->devices[i].devid == id)
        {
            return 1;
        }
    }
    return 0;
}

int
gpustr_to_gpulist(const char* gpustr, int* gpulist, int length)
{
    int insert = 0;
    topology_gpu_init();
    GpuTopology_t gpu_topology = get_gpuTopology();
    bstring bgpustr = bfromcstr(gpustr);
    struct bstrList* commalist = bsplit(bgpustr, ',');
    for (int i = 0; i < commalist->qty; i++)
    {
        if (bstrchrp(commalist->entry[i], '-', 0) != BSTR_ERR)
        {
            struct bstrList* indexlist = bsplit(commalist->entry[i], '-');
            int start = check_and_atoi(bdata(indexlist->entry[0]));
            int end = check_and_atoi(bdata(indexlist->entry[1]));
            if (start <= end)
            {
                for (int k = start; k <= end; k++)
                {
                    if (valid_gpu_nvmon(gpu_topology, k) && insert < length)
                    {
                        gpulist[insert] = k;
                        insert++;
                    }
                }
            }
        }
        else
        {
            int id = check_and_atoi(bdata(commalist->entry[i]));
            if (valid_gpu_nvmon(gpu_topology, id) && insert < length)
            {
                gpulist[insert] = id;
                insert++;
            }
        }
    }
    return insert;
}

#endif /* LIKWID_WITH_NVMON */

#ifdef LIKWID_WITH_ROCMON

static int valid_gpu_rocmon(GpuTopology_rocm_t topo, int id)
{
    for (int i = 0; i < topo->numDevices; i++)
    {
        if (topo->devices[i].devid == id)
        {
            return 1;
        }
    }
    return 0;
}

int
gpustr_to_gpulist_rocm(const char* gpustr, int* gpulist, int length)
{
    int insert = 0;
    topology_gpu_init_rocm();
    GpuTopology_rocm_t gpu_topology = get_gpuTopology_rocm();
    bstring bgpustr = bfromcstr(gpustr);
    struct bstrList* commalist = bsplit(bgpustr, ',');
    for (int i = 0; i < commalist->qty; i++)
    {
        if (bstrchrp(commalist->entry[i], '-', 0) != BSTR_ERR)
        {
            struct bstrList* indexlist = bsplit(commalist->entry[i], '-');
            int start = check_and_atoi(bdata(indexlist->entry[0]));
            int end = check_and_atoi(bdata(indexlist->entry[1]));
            if (start <= end)
            {
                for (int k = start; k <= end; k++)
                {
                    if (valid_gpu_rocmon(gpu_topology, k) && insert < length)
                    {
                        gpulist[insert] = k;
                        insert++;
                    }
                }
            }
        }
        else
        {
            int id = check_and_atoi(bdata(commalist->entry[i]));
            if (valid_gpu_rocmon(gpu_topology, id) && insert < length)
            {
                gpulist[insert] = id;
                insert++;
            }
        }
    }
    return insert;
}

#endif /* LIKWID_WITH_ROCMON */
