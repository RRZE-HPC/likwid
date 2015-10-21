
#include <stdlib.h>
#include <stdio.h>

#include <likwid.h>


static int cpulist_sort(int* incpus, int* outcpus, int length)
{
    int insert = 0;
    topology_init();
    CpuTopology_t cpuid_topology = get_cpuTopology();
    if (length <= 0)
    {
        return -1;
    }
    for (int off=0;off < cpuid_topology->numThreadsPerCore;off++)
    {
        for (int i=0; i<length/cpuid_topology->numThreadsPerCore;i++)
        {
            outcpus[insert] = incpus[(i*cpuid_topology->numThreadsPerCore)+off];
            insert++;
        }
    }
    return insert;
}

static int cpulist_concat(int* cpulist, int startidx, int* addlist, int addlength)
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

static int cpu_in_domain(int domainidx, int cpu)
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

static int cpuexpr_to_list(bstring bcpustr, bstring prefix, int* list, int length)
{
    topology_init();
    CpuTopology_t cpuid_topology = get_cpuTopology();
    affinity_init();
    AffinityDomains_t affinity = get_affinityDomains();
    struct bstrList* strlist = bstrListCreate();
    strlist = bsplit(bcpustr, ',');
    int oldinsert = 0;
    int insert = 0;
    for (int i=0;i < strlist->qty; i++)
    {
        bstring newstr = bstrcpy(prefix);
        bconcat(newstr, strlist->entry[i]);
        oldinsert = insert;
        for (int j = 0; j < affinity->numberOfAffinityDomains; j++)
        {
            if (bstrcmp(affinity->domains[j].tag, newstr) == 0)
            {
                list[insert] = atoi(bdata(strlist->entry[i]));
                insert++;
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

static int cpustr_to_cpulist_scatter(bstring bcpustr, int* cpulist, int length)
{
    topology_init();
    CpuTopology_t cpuid_topology = get_cpuTopology();
    affinity_init();
    AffinityDomains_t affinity = get_affinityDomains();
    char* cpustring = bstr2cstr(bcpustr, '\0');
    if (bstrchrp(bcpustr, ':', 0) != BSTR_ERR)
    {
        int insert = 0;
        int suitidx = 0;
        int* suitable = (int*)malloc(affinity->numberOfAffinityDomains*sizeof(int));
        if (!suitable)
        {
            bcstrfree(cpustring);
            return -ENOMEM;
        }
        for (int i=0; i<affinity->numberOfAffinityDomains; i++)
        {
            if (bstrchrp(affinity->domains[i].tag, cpustring[0], 0) != BSTR_ERR)
            {
                suitable[suitidx] = i;
                suitidx++;
            }
        }
        int* sortedList = (int*) malloc(affinity->domains[suitable[0]].numberOfProcessors * sizeof(int));
        if (!sortedList)
        {
            free(suitable);
            bcstrfree(cpustring);
            return -ENOMEM;
        }
        for (int off=0;off<affinity->domains[suitable[0]].numberOfProcessors;off++)
        {
            for(int i=0;i < suitidx; i++)
            {
                cpulist_sort(affinity->domains[suitable[i]].processorList, sortedList, affinity->domains[suitable[i]].numberOfProcessors);
                cpulist[insert] = sortedList[off];
                insert++;
                if (insert == length)
                    goto scatter_done;
            }
        }
scatter_done:
        bcstrfree(cpustring);
        free(sortedList);
        free(suitable);
        return insert;
    }
    bcstrfree(cpustring);
    return 0;
}

static int cpustr_to_cpulist_expression(bstring bcpustr, int* cpulist, int length)
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
    if (bstrchrp(bcpustr, 'E', 0) != 0)
    {
        fprintf(stderr, "Not a valid CPU expression\n");
        return 0;
    }
    struct bstrList* strlist = bstrListCreate();
    strlist = bsplit(bcpustr, ':');
    if (strlist->qty == 3)
    {
        bdomain = bstrcpy(strlist->entry[1]);
        count = atoi(bdata(strlist->entry[2]));
        stride = 1;
        chunk = 1;
    }
    else if (strlist->qty == 5)
    {
        bdomain = bstrcpy(strlist->entry[1]);
        count = atoi(bdata(strlist->entry[2]));
        chunk = atoi(bdata(strlist->entry[3]));
        stride = atoi(bdata(strlist->entry[4]));
    }
    for (int i=0; i<affinity->numberOfAffinityDomains; i++)
    {
        if (bstrcmp(bdomain, affinity->domains[i].tag) == 0)
        {
            domainidx = i;
            break;
        }
    }
    if (domainidx < 0)
    {
        fprintf(stderr, "Cannot find domain %s\n", bdata(bdomain));
        bstrListDestroy(strlist);
        return 0;
    }
    int offset = 0;
    int insert = 0;
    for (int i=0;i<count;i++)
    {
        for (int j=0;j<chunk && offset+j<affinity->domains[domainidx].numberOfProcessors;j++)
        {
            cpulist[insert] = affinity->domains[domainidx].processorList[offset + j];
            insert++;
            if (insert == length)
                goto expression_done;
        }
        offset += stride;
        if (offset >= affinity->domains[domainidx].numberOfProcessors)
        {
            offset = 0;
        }
        if (insert >= count)
            goto expression_done;
    }
    bstrListDestroy(strlist);
    return 0;
expression_done:
    bstrListDestroy(strlist);
    return insert;
}

static int cpustr_to_cpulist_logical(bstring bcpustr, int* cpulist, int length)
{
    topology_init();
    CpuTopology_t cpuid_topology = get_cpuTopology();
    affinity_init();
    AffinityDomains_t affinity = get_affinityDomains();
    int domainidx = -1;
    bstring bdomain;
    bstring blist;
    if (bstrchrp(bcpustr, 'L', 0) != 0)
    {
        fprintf(stderr, "Not a valid CPU expression\n");
        return 0;
    }
    struct bstrList* strlist = bstrListCreate();
    strlist = bsplit(bcpustr, ':');
    if (strlist->qty != 3)
    {
        fprintf(stderr, "ERROR: Invalid expression, should look like L:<domain>:<indexlist> or be in a cpuset\n");
        bstrListDestroy(strlist);
        return 0;
    }
    bdomain = bstrcpy(strlist->entry[1]);
    blist = bstrcpy(strlist->entry[2]);
    bstrListDestroy(strlist);
    for (int i=0; i<affinity->numberOfAffinityDomains; i++)
    {
        if (bstrcmp(bdomain, affinity->domains[i].tag) == 0)
        {
            domainidx = i;
            break;
        }
    }
    if (domainidx < 0)
    {
        printf("Cannot find domain %s\n", bdata(bdomain));
        return 0;
    }
    int *inlist = malloc(affinity->domains[domainidx].numberOfProcessors * sizeof(int));
    if (inlist == NULL)
    {
        return -ENOMEM;
    }
    int ret = cpulist_sort(affinity->domains[domainidx].processorList, inlist, affinity->domains[domainidx].numberOfProcessors);

    strlist = bstrListCreate();
    strlist = bsplit(blist, ',');
    int insert = 0;
    for (int i=0; i< strlist->qty; i++)
    {
        if (bstrchrp(strlist->entry[i], '-', 0) != BSTR_ERR)
        {
            struct bstrList* indexlist = bstrListCreate();
            indexlist = bsplit(strlist->entry[i], '-');
            for (int j=atoi(bdata(indexlist->entry[0])); j<=atoi(bdata(indexlist->entry[1]));j++)
            {
                cpulist[insert] = inlist[j];
                insert++;
                if (insert == length)
                {
                    bstrListDestroy(indexlist);
                    goto logical_done;
                }
            }
            bstrListDestroy(indexlist);
        }
        else
        {
            cpulist[insert] = inlist[atoi(bdata(strlist->entry[i]))];
            insert++;
            if (insert == length)
            {
                goto logical_done;
            }
        }
    }
logical_done:
    free(inlist);
    bstrListDestroy(strlist);
    return insert;
}



static int cpustr_to_cpulist_physical(bstring bcpustr, int* cpulist, int length)
{
    topology_init();
    CpuTopology_t cpuid_topology = get_cpuTopology();
    affinity_init();
    AffinityDomains_t affinity = get_affinityDomains();
    bstring bdomain;
    bstring blist;
    int domainidx = -1;
    if (bstrchrp(bcpustr, ':', 0) != BSTR_ERR)
    {
        struct bstrList* strlist = bstrListCreate();
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
    for (int i=0; i<affinity->numberOfAffinityDomains; i++)
    {
        if (bstrcmp(bdomain, affinity->domains[i].tag) == 0)
        {
            domainidx = i;
            break;
        }
    }
    if (domainidx < 0)
    {
        fprintf(stderr, "Cannot find domain %s\n", bdata(bdomain));
        bdestroy(bdomain);
        bdestroy(blist);
        return 0;
    }
    struct bstrList* strlist = bstrListCreate();
    strlist = bsplit(blist, ',');
    int insert = 0;
    for (int i=0;i< strlist->qty; i++)
    {
        if (bstrchrp(strlist->entry[i], '-', 0) != BSTR_ERR)
        {
            struct bstrList* indexlist = bstrListCreate();
            indexlist = bsplit(strlist->entry[i], '-');
            for (int j=atoi(bdata(indexlist->entry[0])); j<=atoi(bdata(indexlist->entry[1]));j++)
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
                    fprintf(stderr, "CPU %d not in domain %s\n", j, bdata(affinity->domains[domainidx].tag));
                }
            }
            bstrListDestroy(indexlist);
        }
        else
        {
            int cpu = atoi(bdata(strlist->entry[i]));
            if (cpu_in_domain(domainidx, cpu))
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
                fprintf(stderr, "CPU %d not in domain %s\n", cpu, bdata(affinity->domains[domainidx].tag));
            }
        }
    }
physical_done:
    bstrListDestroy(strlist);
    bdestroy(bdomain);
    bdestroy(blist);
    return insert;
}

int cpustr_to_cpulist(char* cpustring, int* cpulist, int length)
{
    int insert = 0;
    int len = 0;
    int ret = 0;
    bstring bcpustr = bfromcstr(cpustring);
    struct bstrList* strlist = bstrListCreate();
    bstring scattercheck = bformat("scatter");
    topology_init();
    CpuTopology_t cpuid_topology = get_cpuTopology();
    strlist = bsplit(bcpustr, '@');

    int* tmpList = (int*)malloc(length * sizeof(int));
    if (tmpList == NULL)
    {
        bstrListDestroy(strlist);
        bdestroy(scattercheck);
        bdestroy(bcpustr);
        return -ENOMEM;
    }
    for (int i=0; i< strlist->qty; i++)
    {
        if (binstr(strlist->entry[i], 0, scattercheck) != BSTR_ERR)
        {
            ret = cpustr_to_cpulist_scatter(strlist->entry[i], tmpList, length);
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
        else if (cpuid_topology->activeHWThreads < cpuid_topology->numHWThreads)
        {
            fprintf(stdout, "INFO: You are running LIKWID in a cpuset with %d CPUs, only logical numbering allowed", cpuid_topology->activeHWThreads);
            if (((bstrchrp(strlist->entry[i], 'N', 0) == 0) ||
                (bstrchrp(strlist->entry[i], 'S', 0) == 0) ||
                (bstrchrp(strlist->entry[i], 'C', 0) == 0) ||
                (bstrchrp(strlist->entry[i], 'M', 0) == 0)) &&
                (bstrchrp(strlist->entry[i], ':', 0) != BSTR_ERR))
            {
                bstring newstr = bformat("L:");
                bconcat(newstr, strlist->entry[i]);
                ret = cpustr_to_cpulist_logical(newstr, tmpList, length);
                insert += cpulist_concat(cpulist, insert, tmpList, ret);
                bdestroy(newstr);
            }
            else
            {
                bstring newstr = bformat("L:N:");
                bconcat(newstr, strlist->entry[i]);
                ret = cpustr_to_cpulist_logical(newstr, tmpList, length);
                insert += cpulist_concat(cpulist, insert, tmpList, ret);
                bdestroy(newstr);
            }
        }
        else if (((bstrchrp(strlist->entry[i], 'N', 0) == 0) ||
            (bstrchrp(strlist->entry[i], 'S', 0) == 0) ||
            (bstrchrp(strlist->entry[i], 'C', 0) == 0) ||
            (bstrchrp(strlist->entry[i], 'M', 0) == 0)) &&
            (bstrchrp(strlist->entry[i], ':', 0) != BSTR_ERR))
        {
            bstring newstr = bformat("L:");
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
    free(tmpList);
    bstrListDestroy(strlist);
    return insert;
}

int nodestr_to_nodelist(char* nodestr, int* nodes, int length)
{
    int ret = 0;
    bstring prefix = bformat("M");
    bstring bnodestr = bfromcstr(nodestr);
    ret = cpuexpr_to_list(bnodestr, prefix, nodes, length);
    bdestroy(bnodestr);
    return ret;
}

int sockstr_to_socklist(char* sockstr, int* sockets, int length)
{
    int ret = 0;
    bstring prefix = bformat("S");
    bstring bsockstr = bfromcstr(sockstr);
    ret = cpuexpr_to_list(bsockstr, prefix, sockets, length);
    bdestroy(bsockstr);
    return ret;
}
