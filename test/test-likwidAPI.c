#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <likwid.h>
//#include <configuration.h>
//#include <access.h>
//#include <types.h>
//#include <perfmon.h>

typedef struct {
    char* testname;
    int(*testfunc)(void);
    int result;
} test;

static int verbose = 0;

static char eventset_ok_intel[] = "INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,CPU_CLK_UNHALTED_REF:FIXC2";
static char eventset_ok_amd[] = "RETIRED_INSTRUCTIONS:PMC0,RETIRED_BRANCH_INSTR:PMC1,RETIRED_MISP_BRANCH_INSTR:PMC2";
static char event1_ok_intel[] = "INSTR_RETIRED_ANY";
static char event2_ok_intel[] = "CPU_CLK_UNHALTED_CORE";
static char event3_ok_intel[] = "CPU_CLK_UNHALTED_REF";
static char event1_ok_amd[] = "RETIRED_INSTRUCTIONS";
static char event2_ok_amd[] = "RETIRED_BRANCH_INSTR";
static char event3_ok_amd[] = "RETIRED_MISP_BRANCH_INSTR";
static char ctr1_ok_intel[] = "FIXC0";
static char ctr2_ok_intel[] = "FIXC1";
static char ctr3_ok_intel[] = "FIXC2";
static char ctr1_ok_amd[] = "PMC0";
static char ctr2_ok_amd[] = "PMC1";
static char ctr3_ok_amd[] = "PMC2";
static char eventset_option_intel[] = "INSTR_RETIRED_ANY:FIXC0:ANYTHREAD,CPU_CLK_UNHALTED_CORE:FIXC1:ANYTHREAD,CPU_CLK_UNHALTED_REF:FIXC2:ANYTHREAD";
static char eventset_option_amd[] = "RETIRED_INSTRUCTIONS:PMC0:EDGEDETECT,RETIRED_BRANCH_INSTR:PMC1:EDGEDETECT,RETIRED_MISP_BRANCH_INSTR:PMC2:EDGEDETECT";
static int isIntel = 0;
static char perfgroup_ok[] = "BRANCH";
static char perfgroup_fail[] = "BRAN";





int test_initconfig()
{
    int ret;
    ret = init_configuration();
    if (ret != 0)
        goto fail;
    Configuration_t config = get_configuration();
    if (config == NULL)
        goto fail;
    if ((config->daemonMode != ACCESSMODE_DIRECT) && (config->daemonMode != ACCESSMODE_DAEMON))
        goto fail;
    if ((config->daemonMode == ACCESSMODE_DAEMON) && (config->daemonPath == NULL))
        goto fail;
    destroy_configuration();
    return 1;
fail:
    destroy_configuration();
    return 0;
}

int enable_configuration()
{
    init_configuration();
    return 1;
}

int disable_configuration()
{
    destroy_configuration();
    return 1;
}

int test_hpmmode()
{
    Configuration_t config;
    config = get_configuration();
    int def = config->daemonMode;
    HPMmode(ACCESSMODE_DIRECT);
    if (config->daemonMode != ACCESSMODE_DIRECT)
        goto fail;
    HPMmode(ACCESSMODE_DAEMON);
    if (config->daemonMode != ACCESSMODE_DAEMON)
        goto fail;
    HPMmode(def);
    HPMmode(ACCESSMODE_DAEMON+1);
    if (config->daemonMode != def)
        goto fail;
    return 1;
fail:
    return 0;
}

int test_hpminit()
{
    int ret = HPMinit();
    if (ret != 0)
        return 0;
    HPMfinalize();
    return 1;
}

int test_hpmaddthread()
{
    HPMinit();
    int ret = HPMaddThread(0);
    if (ret != 0)
        return 0;
    HPMfinalize();
    return 1;
}

int enable_hpm()
{
    HPMinit();
    HPMaddThread(0);
    return 1;
}

int disable_hpm()
{
    HPMfinalize();
    return 1;
}

int test_topologyinit()
{
    int i, j;
    int ret = topology_init();
    if (ret != 0)
        goto fail;
    CpuInfo_t cpuinfo = get_cpuInfo();
    if (cpuinfo == NULL)
        goto fail;
    if (cpuinfo->family == 0)
        goto fail;
    if (cpuinfo->model == 0)
        goto fail;
    if (cpuinfo->osname == NULL)
        goto fail;
    if (cpuinfo->name == NULL)
        goto fail;
    if (cpuinfo->features == NULL)
        goto fail;
    CpuTopology_t cputopo = get_cpuTopology();
    if (cputopo->threadPool == NULL)
        goto fail;
    if (cputopo->cacheLevels == NULL)
        goto fail;
    if (cputopo->numHWThreads == 0)
        goto fail;
    if (cputopo->activeHWThreads == 0)
        goto fail;
    if (cputopo->numSockets == 0)
        goto fail;
    if (cputopo->numCoresPerSocket < 1)
        goto fail;
    if (cputopo->numThreadsPerCore < 1)
        goto fail;
    if (cputopo->numHWThreads > 0)
    {
        for (i = 0; i < cputopo->numHWThreads; i++)
        {
            for (j=0;j< cputopo->numHWThreads; j++)
            {
                if ((i != j) && (cputopo->threadPool[i].apicId == cputopo->threadPool[j].apicId))
                    goto fail;
            }
            if (cputopo->threadPool[i].threadId >= cputopo->numThreadsPerCore)
            {
                goto fail;
            }
            if (cputopo->threadPool[i].packageId >= cputopo->numSockets)
            {
                goto fail;
            }
        }
    }
    if (cputopo->numCacheLevels > 0)
    {
        for (i=0;i<cputopo->numCacheLevels;i++)
        {
            if (cputopo->cacheLevels[i].level > cputopo->numCacheLevels)
            {
                goto fail;
            }

        }
    }
    isIntel = cpuinfo->isIntel;
    topology_finalize();
    return 1;
fail:
    topology_finalize();
    return 0;
}

int enable_topology()
{
    topology_init();
    return 1;
}

int disable_topology()
{
    topology_finalize();
    return 1;
}

int test_numainit()
{
    int i = 0;
    topology_init();
    CpuInfo_t cpuinfo = get_cpuInfo();
    numa_init();
    int valid = 0, filled_domains = 0;
    NumaTopology_t numainfo = get_numaTopology();
    if (numainfo == NULL)
        goto fail;
    if (numainfo->numberOfNodes <= 0)
        goto fail;
    if (likwid_getNumberOfNodes() <= 0)
        goto fail;
    for (i = 0; i < likwid_getNumberOfNodes(); i++)
    {
        valid = 1;
        if (numainfo->nodes[i].totalMemory == 0)
        {
            valid = 0;
            fprintf(stderr, "WARNING: NUMA domain %d: totalMemory = 0\n", numainfo->nodes[i].id);
        }
        if (numainfo->nodes[i].freeMemory == 0)
        {
            valid = 0;
            fprintf(stderr, "WARNING: NUMA domain %d: freeMemory = 0\n", numainfo->nodes[i].id);
        }
        if (numainfo->nodes[i].numberOfProcessors == 0)
        {
            valid = 0;
            fprintf(stderr, "WARNING: NUMA domain %d: numberOfProcessors = 0\n", numainfo->nodes[i].id);
        }
        if (numainfo->nodes[i].numberOfDistances == 0)
        {
            valid = 0;
            fprintf(stderr, "WARNING: NUMA domain %d: numberOfDistances = 0\n", numainfo->nodes[i].id);
        }
        if (numainfo->nodes[i].numberOfDistances != likwid_getNumberOfNodes())
        {
            valid = 0;
        }
        if (valid)
            filled_domains++;
        else if (strcmp(cpuinfo->short_name, "knl") != 0)
            goto fail;
    }
    if (strcmp(cpuinfo->short_name, "knl") != 0 && likwid_getNumberOfNodes() % filled_domains != 0)
        goto fail;
    numa_finalize();
    topology_finalize();
    return 1;
fail:
    numa_finalize();
    topology_finalize();
    return 0;
}

int test_affinityinit()
{
    int i = 0;
    int valid = 0, filled_domains = 0;
    topology_init();
    CpuInfo_t cpuinfo = get_cpuInfo();
    CpuTopology_t cputopo = get_cpuTopology();
    numa_init();
    affinity_init();
    AffinityDomains_t doms = get_affinityDomains();
    if (doms == NULL)
        goto fail;
    if (doms->numberOfSocketDomains != cputopo->numSockets)
        goto fail;
    if (doms->numberOfNumaDomains == 0)
        goto fail;
    if (doms->numberOfProcessorsPerSocket == 0)
        goto fail;
    if (doms->numberOfAffinityDomains == 0)
        goto fail;
    if (doms->numberOfCacheDomains == 0)
        goto fail;
    if (doms->numberOfCoresPerCache == 0)
        goto fail;
    if (doms->numberOfProcessorsPerCache == 0)
        goto fail;
    if (doms->numberOfProcessorsPerCache < doms->numberOfCoresPerCache)
        goto fail;
    if (doms->domains == NULL)
        goto fail;
    for (i = 0; i < doms->numberOfAffinityDomains; i++)
    {
        valid = 1;
        if (doms->domains[i].numberOfProcessors == 0)
        {
            valid = 0;
            fprintf(stderr, "WARNING: Affinity domain %d: numberOfProcessors = 0\n", i);
        }
        if (doms->domains[i].numberOfCores == 0)
        {
            valid = 0;
            fprintf(stderr, "WARNING: Affinity domain %d: numberOfCores = 0\n", i);
        }
        if (doms->domains[i].numberOfProcessors < doms->domains[i].numberOfCores)
        {
            valid = 0;
            fprintf(stderr, "WARNING: Affinity domain %d: numberOfProcessors < doms->domains[i].numberOfCores\n", i);
        }
        if (doms->domains[i].processorList == NULL)
        {
            valid = 0;
            fprintf(stderr, "WARNING: Affinity domain %d: processorList == NULL\n", i);
        }
        if (valid)
            filled_domains++;
        else if (strcmp(cpuinfo->short_name, "knl") != 0)
        {
            fprintf(stderr, "Domain %s failed\n", bdata(doms->domains[i].tag));
            goto fail;
        }
    }
    affinity_finalize();
    topology_finalize();
    return 1;
fail:
    affinity_finalize();
    topology_finalize();
    return 0;
}

int test_cpustring_logical()
{
    int test[5];
    int len = 5;
    int ret = cpustr_to_cpulist("S0:0-3", test, len);
    if (ret < 0)
    {
        if (verbose) printf("Returned %d\n", ret);
        return 0;
    }
    if (ret != 4)
    {
        if (verbose) printf("Returned with %d not enough CPUs\n", ret);
        return 0;
    }
    return 1;
}

int test_cpustring_physical()
{
    int test[5];
    int len = 5;
    int ret = cpustr_to_cpulist("0,1,2,3", test, len);
    if (ret < 0)
    {
        if (verbose) printf("Returned %d\n", ret);
        return 0;
    }
    if (ret != 4)
    {
        if (verbose) printf("Returned with %d not enough CPUs\n", ret);
        return 0;
    }
    return 1;
}

int test_cpustring_expression()
{
    int test[5];
    int len = 5;
    int ret = cpustr_to_cpulist("E:S0:4:1:2", test, len);
    if (ret < 0)
    {
        if (verbose) printf("Returned %d\n", ret);
        return 0;
    }
    if (ret != 4)
    {
        if (verbose) printf("Returned with %d not enough CPUs\n", ret);
        return 0;
    }
    return 1;
}

int test_cpustring_scatter()
{
    CpuTopology_t cputopo = get_cpuTopology();
    int len = cputopo->numHWThreads;
    int *test = (int*) malloc(len * sizeof(int));
    if (!test)
    {
        return 0;
    }
    int ret = cpustr_to_cpulist("S:scatter", test, len);
    if (ret < 0)
    {
        if (verbose) printf("Returned %d\n", ret);
        free(test);
        return 0;
    }
    if (ret != cputopo->numHWThreads)
    {
        if (verbose) printf("Returned with %d not enough CPUs (%d)\n", ret, cputopo->numHWThreads);
        free(test);
        return 0;
    }
    free(test);
    return 1;
}

int test_cpustring_combined()
{
    int test[100];
    int len = 100;
    int ret = cpustr_to_cpulist("N:0-3@S0:0-3", test, len);
    if (ret < 0)
    {
        if (verbose) printf("Returned %d\n", ret);
        return 0;
    }
    if (ret != 8)
    {
        if (verbose) printf("Returned with %d not enough CPUs\n", ret);
        return 0;
    }
    return 1;
}

int test_perfmoninit_faulty()
{
    int cpu = 0;
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
    {
        if (verbose) fprintf(stderr, "perfmon_init failed with %d: %s\n", ret, strerror(errno));
        goto fail;
    }
    perfmon_finalize();
    return 1;
fail:
    perfmon_finalize();
    return 0;
}

int test_perfmoninit_valid()
{
    int cpu = 0;
    init_configuration();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
    {
        if (verbose) fprintf(stderr, "perfmon_init failed with %d: %s\n", ret, strerror(errno));
        goto fail;
    }
    if (perfmon_getNumberOfGroups() != 0)
    {
        if (verbose) fprintf(stderr, "perfmon_getNumberOfGroups() unequal to zero\n");
        goto fail;
    }
    if (perfmon_getNumberOfThreads() != 1)
    {
        if (verbose) fprintf(stderr, "perfmon_getNumberOfThreads() unequal to 1\n");
        goto fail;
    }
    perfmon_finalize();

    destroy_configuration();
    return 1;
fail:
    perfmon_finalize();
    affinity_finalize();
    topology_finalize();
    destroy_configuration();
    return 0;
}

int test_perfmoninit()
{
    int cpu = 0;
    int i;
    topology_init();
    affinity_init();
    for(i=0;i<10;i++)
    {
        perfmon_init(1, &cpu);
        perfmon_finalize();
    }
    affinity_finalize();
    topology_finalize();
    return 1;
}

int test_perfmonfinalize()
{
    perfmon_finalize();
    return 1;
}

int test_perfmonaddeventset()
{
    char eventset_fail1[] = "INSTR_RETIRED.ANY:FIXC0";
    char eventset_fail2[] = "INSTR_RETIRED-ANY:FIXC0";
    CpuInfo_t cpuinfo;
    int cpu = 0;
    topology_init();
    cpuinfo = get_cpuInfo();
    int ret = perfmon_init(1, &cpu);
    if (ret != 0) {
        if (verbose > 0) printf("Perfmon init failed\n");
        goto fail;
    }
    if (perfmon_getNumberOfGroups() != 0) {
        if (verbose > 0) printf("Perfmon number of groups != 0\n");
        goto fail;
    }
    if (perfmon_getNumberOfThreads() != 1) {
        if (verbose > 0) printf("Perfmon number of threads != 1\n");
        goto fail;
    }
    if (perfmon_getIdOfActiveGroup() != -1) {
        if (verbose > 0) printf("Perfmon id of active group != -1\n");
        goto fail;
    }
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0) {
        if (verbose > 0) printf("Perfmon addEventSet(ok) failed\n");
        goto fail;
    }
    if (perfmon_getNumberOfGroups() != 1) {
        if (verbose > 0) printf("Perfmon number of groups != 1\n");
        goto fail;
    }
    if (perfmon_getNumberOfEvents(ret) != 3) {
        if (verbose > 0) printf("Perfmon number of events != 3\n");
        goto fail;
    }
    if (perfmon_getIdOfActiveGroup() != -1) {
        if (verbose > 0) printf("Perfmon id of active group != -1\n");
        goto fail;
    }
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_option_intel);
    else
        ret = perfmon_addEventSet(eventset_option_amd);

    if (ret != 1) {
        if (verbose > 0) printf("Perfmon addEventSet(options) failed\n");
        goto fail;
    }
    if (perfmon_getNumberOfGroups() != 2) {
        if (verbose > 0) printf("Perfmon number of groups != 2\n");
        goto fail;
    }
    if (perfmon_getNumberOfEvents(ret) != 3) {
        if (verbose > 0) printf("Perfmon number of events != 3\n");
        goto fail;
    }
    if (perfmon_getIdOfActiveGroup() != -1) {
        if (verbose > 0) printf("Perfmon id of active group != -1\n");
        goto fail;
    }
    ret = perfmon_addEventSet(eventset_fail1);
    if (ret >= 0) {
        if (verbose > 0) printf("Perfmon addEventSet(fail1) failed\n");
        goto fail;
    }
    if (perfmon_getNumberOfGroups() != 2) {
        if (verbose > 0) printf("Perfmon number of groups != 2\n");
        goto fail;
    }
    ret = perfmon_addEventSet(eventset_fail2);
    if (ret >= 0) {
        if (verbose > 0) printf("Perfmon addEventSet(fail2) failed\n");
        goto fail;
    }
    if (perfmon_getNumberOfGroups() != 2) {
        if (verbose > 0) printf("Perfmon number of groups != 2\n");
        goto fail;
    }
    if (perfmon_getIdOfActiveGroup() != -1) {
        if (verbose > 0) printf("Perfmon id of active group != -1\n");
        goto fail;
    }
    perfmon_finalize();
    affinity_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    affinity_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonaddeventset_noinit()
{
    int ret = 0;
    topology_init();
    CpuInfo_t cpuinfo = get_cpuInfo();
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret == 0)
        goto fail;
    return 1;
fail:
    topology_finalize();
    return 0;
}

int test_perfmoncustomgroup()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    topology_init();
    cpuinfo = get_cpuInfo();
    int ret = perfmon_init(1, &cpu);
    if (ret != 0) {
        if (verbose > 0) printf("Perfmon init failed\n");
        goto fail;
    }

    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0) {
        if (verbose > 0) printf("Perfmon addEventSet(ok) failed\n");
        goto fail;
    }
    if (perfmon_getNumberOfEvents(ret) != 3) {
        if (verbose > 0) printf("Perfmon number of events != 3\n");
        goto fail;
    }
    if (perfmon_getNumberOfMetrics(ret) != 0) {
        if (verbose > 0) printf("Perfmon number of metrics != 0\n");
        goto fail;
    }
    if (cpuinfo->isIntel)
    {
        if (strcmp(perfmon_getEventName(ret, 0), event1_ok_intel) != 0)
        {
            goto fail;
        }
        if (strcmp(perfmon_getEventName(ret, 1), event2_ok_intel) != 0)
        {
            goto fail;
        }
        if (strcmp(perfmon_getEventName(ret, 2), event3_ok_intel) != 0)
        {
            goto fail;
        }
        if (strcmp(perfmon_getCounterName(ret, 0), ctr1_ok_intel) != 0)
        {
            goto fail;
        }
        if (strcmp(perfmon_getCounterName(ret, 1), ctr2_ok_intel) != 0)
        {
            goto fail;
        }
        if (strcmp(perfmon_getCounterName(ret, 2), ctr3_ok_intel) != 0)
        {
            goto fail;
        }
    }
    else
    {
        if (strcmp(perfmon_getEventName(ret, 0), event1_ok_amd) != 0)
        {
            goto fail;
        }
        if (strcmp(perfmon_getEventName(ret, 1), event2_ok_amd) != 0)
        {
            goto fail;
        }
        if (strcmp(perfmon_getEventName(ret, 2), event3_ok_amd) != 0)
        {
            goto fail;
        }
        if (strcmp(perfmon_getCounterName(ret, 0), ctr1_ok_amd) != 0)
        {
            goto fail;
        }
        if (strcmp(perfmon_getCounterName(ret, 1), ctr2_ok_amd) != 0)
        {
            goto fail;
        }
        if (strcmp(perfmon_getCounterName(ret, 2), ctr3_ok_amd) != 0)
        {
            goto fail;
        }
    }

    if (strcmp(perfmon_getGroupName(ret), "Custom") != 0)
    {
        goto fail;
    }
    if (strcmp(perfmon_getGroupInfoShort(ret), "Custom") != 0)
    {
        goto fail;
    }
    if (strcmp(perfmon_getGroupInfoLong(ret), "Custom") != 0)
    {
        goto fail;
    }
    if (perfmon_getLastTimeOfGroup(ret) != 0)
    {
        goto fail;
    }
    
    perfmon_finalize();
    affinity_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    affinity_finalize();
    topology_finalize();
    return 0;
}

int test_perfmongetgroups()
{
    int i;
    topology_init();
    char** glist = NULL;
    char** slist = NULL;
    char** llist = NULL;
    int ret = perfmon_getGroups(&glist, &slist, &llist);
    fprintf(stderr, "perfmon_getGroups() returned %d groups\n", ret);
    if (ret <= 0)
    {
        goto fail;
    }

    for (i=0; i< ret; i++)
    {
        if (!glist[i])
            fprintf(stderr, "No group name");
        if (!slist[i])
            fprintf(stderr, "No short info for group name %s\n", glist[i]);
        if (!llist[i])
            fprintf(stderr, "No long info for group name %s\n", glist[i]);
        if (!llist[i] || !glist[i] || !slist[i])
            goto fail;
        if (strcmp(glist[i], "") == 0)
        {
            fprintf(stderr, "Empty group name\n");
            goto fail;
        }
        if (strcmp(slist[i], "") == 0)
        {
            fprintf(stderr, "Empty short info in group %s\n", glist[i]);
            goto fail;
        }
        if (strcmp(llist[i], "") == 0)
        {
            fprintf(stderr, "Empty long info in group %s\n", glist[i]);
            goto fail;
        }
    }
    perfmon_returnGroups(ret, glist, slist, llist);
    topology_finalize();
    return 1;
fail:
    perfmon_returnGroups(ret, glist, slist, llist);
    topology_finalize();
    return 0;
}

int _test_perfmonperfgroup(char* perfgroup)
{
    CpuInfo_t cpuinfo;
    int i;
    int cpu = 0;
    topology_init();
    cpuinfo = get_cpuInfo();
    int ret = perfmon_init(1, &cpu);
    if (ret != 0) {
        if (verbose > 0) printf("Perfmon init failed\n");
        goto fail;
    }
    ret = perfmon_addEventSet(perfgroup);
    if (ret != 0) {
        if (verbose > 0) printf("Perfmon addEventSet(%s) failed\n", perfgroup);
        goto fail;
    }
    if (perfmon_getNumberOfEvents(ret) == 0) {
        if (verbose > 0) printf("Perfmon number of events == 0\n");
        goto fail;
    }
    if (perfmon_getNumberOfMetrics(ret) == 0) {
        if (verbose > 0) printf("Perfmon number of metrics == 0\n");
        goto fail;
    }
    for (i=0; i<perfmon_getNumberOfEvents(ret); i++) {
        if (strcmp(perfmon_getEventName(ret, i), "") == 0)
        {
            if (verbose > 0) printf("Perfmon event name zero\n");
            goto fail;
        }
        if (strcmp(perfmon_getCounterName(ret, i), "") == 0)
        {
            if (verbose > 0) printf("Perfmon counter name zero\n");
            goto fail;
        }
    }
    if (strcmp(perfmon_getGroupName(ret), "Custom") == 0)
    {
        if (verbose > 0) if (verbose > 0) printf("Perfmon groupName %s == %s\n", perfgroup, perfmon_getGroupName(ret));
        goto fail;
    }
    if (strcmp(perfmon_getGroupInfoShort(ret), "Custom") == 0)
    {
        printf("Perfmon shortInfo %s == %s\n", perfgroup, perfmon_getGroupInfoShort(ret));
        goto fail;
    }
    if (strcmp(perfmon_getGroupInfoLong(ret), "Custom") == 0)
    {
        if (verbose > 0) printf("Perfmon longInfo %s == %s\n", perfgroup, perfmon_getGroupInfoShort(ret));
        goto fail;
    }
    if (perfmon_getLastTimeOfGroup(ret) != 0)
    {
        if (verbose > 0) printf("Perfmon last time of %s: %f\n", perfgroup, perfmon_getLastTimeOfGroup(ret));
        goto fail;
    }
    if (perfmon_getTimeOfGroup(ret) != 0)
    {
        if (verbose > 0) printf("Perfmon time of %s: %f\n", perfgroup, perfmon_getTimeOfGroup(ret));
        goto fail;
    }
    perfmon_setupCounters(ret);
    perfmon_startCounters();
    sleep(1);
    perfmon_stopCounters();
    for (i=0; i<perfmon_getNumberOfMetrics(ret); i++) {
        if (strcmp(perfmon_getMetricName(ret, i), "") == 0)
        {
            if (verbose > 0) printf("Perfmon metric name zero\n");
            goto fail;
        }
        double res = perfmon_getMetric(ret, i, 0);
        if ((res != 0.0) && (res < 0))
        {
            if (verbose > 0) printf("Perfmon metric %s result %f\n", perfmon_getMetricName(ret, i), res );
            goto fail;
        }
        double lastres = perfmon_getLastMetric(ret, i, 0);
        if  ((ret >= 0) &&
            (lastres >= 0) &&
            (res != lastres))
        {
            if (verbose > 0) printf("Perfmon metric %s result %f not equal to last %f\n", perfmon_getMetricName(ret, i), res, lastres);
            goto fail;
        }
    }
    perfmon_finalize();
    affinity_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    affinity_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonperfgroup_ok()
{
    return _test_perfmonperfgroup(perfgroup_ok);
}

int test_perfmonperfgroup_fail()
{
    return !_test_perfmonperfgroup(perfgroup_fail);
}

int test_perfmonsetup()
{
    CpuInfo_t cpuinfo;
    int group1, group2;
    int cpu = 0;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (perfmon_getNumberOfGroups() != 0)
        goto fail;
    if (perfmon_getNumberOfThreads() != 1)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group1 = ret;
    if (perfmon_getNumberOfGroups() != 1)
        goto fail;
    if (perfmon_getNumberOfEvents(group1) != 3)
        goto fail;
    ret = perfmon_setupCounters(group1);
    if (ret != 0)
        goto fail;
    if (perfmon_getIdOfActiveGroup() != group1)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_option_intel);
    else
        ret = perfmon_addEventSet(eventset_option_amd);
    if (ret != 1)
        goto fail;
    group2 = ret;
    if (perfmon_getIdOfActiveGroup() != group1)
        goto fail;
    if (perfmon_getNumberOfGroups() != 2)
        goto fail;
    if (perfmon_getNumberOfEvents(group1) != 3)
        goto fail;
    if (perfmon_getNumberOfEvents(group2) != 3)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonswitch()
{
    CpuInfo_t cpuinfo;
    int group1, group2;
    int cpu = 0;
    topology_init();
    cpuinfo = get_cpuInfo();
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
    {
        ret = perfmon_addEventSet(eventset_ok_intel);
        if (ret != 0)
            goto fail;
        group1 = ret;
        ret = perfmon_addEventSet(eventset_option_intel);
        if (ret != 1)
            goto fail;
    }
    else
    {
        ret = perfmon_addEventSet(eventset_ok_amd);
        if (ret != 0)
            goto fail;
        group1 = ret;
        ret = perfmon_addEventSet(eventset_option_amd);
        if (ret != 1)
            goto fail;
    }
    group2 = ret;
    ret = perfmon_setupCounters(group1);
    if (ret != 0)
        goto fail;
    if (perfmon_getIdOfActiveGroup() != group1)
        goto fail;
    ret = perfmon_switchActiveGroup(group2);
    if (perfmon_getIdOfActiveGroup() != group2)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonstart()
{
    CpuInfo_t cpuinfo;
    int group1, group2;
    int cpu = 0;
    topology_init();
    cpuinfo = get_cpuInfo();
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group1 = ret;
    ret = perfmon_setupCounters(group1);
    if (ret != 0)
        goto fail;
    if (perfmon_getIdOfActiveGroup() != group1)
        goto fail;
    ret = perfmon_startCounters();
    if (ret != 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonsetup_noinit()
{
    int ret = perfmon_setupCounters(0);
    if (ret == 0)
        goto fail;
    return 1;
fail:
    return 0;
}

int test_perfmonsetup_noadd()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    topology_init();
    cpuinfo = get_cpuInfo();
    if (cpuinfo->isIntel == 0)
    {
        topology_finalize();
        return 1;
    }
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    ret = perfmon_setupCounters(0);
    if (ret == 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:

    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonstart_noinit()
{
    int ret = perfmon_startCounters();
    if (ret == 0)
        goto fail;
    return 1;
fail:
    return 0;
}

int test_perfmonstart_noadd()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    topology_init();
    cpuinfo = get_cpuInfo();
    if (cpuinfo->isIntel == 0)
    {
        topology_finalize();
        return 1;
    }
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    ret = perfmon_startCounters();
    if (ret == 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonstop()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    ret = perfmon_setupCounters(group);
    if (ret != 0)
        goto fail;
    ret = perfmon_startCounters();
    if (ret != 0)
        goto fail;
    ret = perfmon_stopCounters();
    if (ret != 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonstop_noinit()
{
    int ret = perfmon_stopCounters();
    if (ret == 0)
        goto fail;
    return 1;
fail:
    return 0;
}

int test_perfmonstop_noadd()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();
    if (cpuinfo->isIntel == 0)
    {
        topology_finalize();
        return 1;
    }
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    ret = perfmon_stopCounters();
    if (ret == 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonstop_nosetup()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    ret = perfmon_stopCounters();
    if (ret == 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonstop_nostart()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    ret = perfmon_setupCounters(group);
    if (ret != 0)
        goto fail;
    ret = perfmon_stopCounters();
    if (ret == 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonresult_noinit()
{
    double result = perfmon_getResult(0,0,0);
    if (isnan(result));
    {
        return 1;
    }
fail:
    return 0;
}

int test_perfmonresult_noadd()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    double result = perfmon_getResult(0,0,0);
    if (!(isnan(result)))
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonresult_nosetup()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    double result = perfmon_getResult(group,0,0);
    if (result != 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonresult_nostart()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    ret = perfmon_setupCounters(group);
    if (ret != 0)
        goto fail;
    double result = perfmon_getResult(group,0,0);
    if (result != 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonresult_nostop()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    ret = perfmon_setupCounters(group);
    if (ret != 0)
        goto fail;
    ret = perfmon_startCounters();
    if (ret != 0)
        goto fail;
    double result = perfmon_getResult(group,0,0);
    if (result != 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonresult()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    ret = perfmon_setupCounters(group);
    if (ret != 0)
        goto fail;

    ret = perfmon_startCounters();
    if (ret != 0)
        goto fail;
    sleep(1);
    ret = perfmon_stopCounters();
    if (ret != 0)
        goto fail;
    if ((perfmon_getResult(group,0,0) == 0)||(perfmon_getResult(group,1,0) == 0))
        goto fail;
    if (perfmon_getTimeOfGroup(group) == 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonlastresult_noinit()
{
    double result = perfmon_getLastResult(0,0,0);
    if (result != 0)
        goto fail;
    return 1;
fail:
    return 0;
}

int test_perfmonlastresult_noadd()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    double result = perfmon_getLastResult(0,0,0);
    if (result != 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonlastresult_nosetup()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    double result = perfmon_getLastResult(group,0,0);
    if (result != 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonlastresult_nostart()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    ret = perfmon_setupCounters(group);
    if (ret != 0)
        goto fail;
    double result = perfmon_getLastResult(group,0,0);
    if (result != 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonlastresult_nostop()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    ret = perfmon_setupCounters(group);
    if (ret != 0)
        goto fail;
    ret = perfmon_startCounters();
    if (ret != 0)
        goto fail;
    double result = perfmon_getLastResult(group,0,0);
    if (result != 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonlastresult()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    ret = perfmon_setupCounters(group);
    if (ret != 0)
        goto fail;

    ret = perfmon_startCounters();
    if (ret != 0)
        goto fail;
    sleep(1);
    ret = perfmon_stopCounters();
    if (ret != 0)
        goto fail;
    if ((perfmon_getLastResult(group,0,0) == 0)||(perfmon_getLastResult(group,1,0) == 0))
        goto fail;
    if (perfmon_getLastResult(group,0,0) != perfmon_getResult(group,0,0))
        goto fail;
    if (perfmon_getTimeOfGroup(group) == 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonmetric_noinit()
{
    double result = perfmon_getMetric(0,0,0);
    if (!(isnan(result)))
        goto fail;
    return 1;
fail:
    return 0;
}

int test_perfmonmetric_noadd()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    double result = perfmon_getMetric(0,0,0);
    if (!(isnan(result)))
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonmetric_nosetup()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    double result = perfmon_getMetric(group,0,0);
    if (!(isnan(result)))
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonmetric_nostart()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    ret = perfmon_setupCounters(group);
    if (ret != 0)
        goto fail;
    double result = perfmon_getMetric(group,0,0);
    if (!(isnan(result)))
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonmetric_nostop()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    ret = perfmon_setupCounters(group);
    if (ret != 0)
        goto fail;
    ret = perfmon_startCounters();
    if (ret != 0)
        goto fail;
    double result = perfmon_getMetric(group,0,0);
    if (!(isnan(result)))
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonmetric_ok()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    ret = perfmon_setupCounters(group);
    if (ret != 0)
        goto fail;

    ret = perfmon_startCounters();
    if (ret != 0)
        goto fail;
    sleep(1);
    ret = perfmon_stopCounters();
    if (ret != 0)
        goto fail;
    if ((perfmon_getMetric(group,0,0) == 0)||(perfmon_getMetric(group,1,0) == 0))
        goto fail;
    if (perfmon_getTimeOfGroup(group) == 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonlastmetric_noinit()
{
    double result = perfmon_getLastMetric(0,0,0);
    if (!(isnan(result)))
        goto fail;
    return 1;
fail:
    return 0;
}

int test_perfmonlastmetric_noadd()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    double result = perfmon_getLastMetric(0,0,0);
    if (!(isnan(result)))
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonlastmetric_nosetup()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    double result = perfmon_getLastMetric(group,0,0);
    if (!(isnan(result)))
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonlastmetric_nostart()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    ret = perfmon_setupCounters(group);
    if (ret != 0)
        goto fail;
    double result = perfmon_getLastMetric(group,0,0);
    if (!(isnan(result)))
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonlastmetric_nostop()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(eventset_ok_intel);
    else
        ret = perfmon_addEventSet(eventset_ok_amd);
    if (ret != 0)
        goto fail;
    group = ret;
    ret = perfmon_setupCounters(group);
    if (ret != 0)
        goto fail;
    ret = perfmon_startCounters();
    if (ret != 0)
        goto fail;
    double result = perfmon_getLastMetric(group,0,0);
    if (!(isnan(result)))
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}

int test_perfmonlastmetric_ok()
{
    CpuInfo_t cpuinfo;
    int cpu = 0;
    int group;
    topology_init();
    cpuinfo = get_cpuInfo();

    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (cpuinfo->isIntel)
        ret = perfmon_addEventSet(perfgroup_ok);
    else
        ret = perfmon_addEventSet(perfgroup_ok);
    if (ret != 0)
        goto fail;
    group = ret;
    ret = perfmon_setupCounters(group);
    if (ret != 0)
        goto fail;

    ret = perfmon_startCounters();
    if (ret != 0)
        goto fail;
    sleep(1);
    ret = perfmon_stopCounters();
    if (ret != 0)
        goto fail;
    if ((perfmon_getLastMetric(group,0,0) == 0)||(perfmon_getLastMetric(group,1,0) == 0))
        goto fail;
    if (perfmon_getLastMetric(group,0,0) != perfmon_getMetric(group,0,0))
        goto fail;
    if (perfmon_getLastMetric(group,1,0) != perfmon_getMetric(group,1,0))
        goto fail;
    if (perfmon_getTimeOfGroup(group) == 0)
        goto fail;
    perfmon_finalize();
    topology_finalize();
    return 1;
fail:
    perfmon_finalize();
    topology_finalize();
    return 0;
}



int test_timerinit()
{
    timer_init();
    uint64_t clock = timer_getCpuClock();
    if (clock == 0)
        goto fail;
    timer_finalize();
    return 1;
fail:
    timer_finalize();
    return 0;
}

int test_timerfinalize()
{
    timer_finalize();
    return 1;
}

int test_timerprint_noinit()
{
    TimerData timer;
    timer_reset(&timer);
    double time = timer_print(&timer);
    if (time != 0)
        goto fail;
    return 1;
fail:
    return 0;
}

int test_timerprint()
{
    TimerData timer;
    timer_reset(&timer);
    timer_init();
    double time = timer_print(&timer);
    if (time != 0)
        goto fail;
    uint64_t cycles = timer_printCycles(&timer);
    if (cycles != 0)
        goto fail;
    timer_finalize();
    return 1;
fail:
    timer_finalize();
    return 0;
}

int test_timerprint_start()
{
    TimerData timer;
    timer_reset(&timer);
    timer_init();
    timer_start(&timer);
    double time = timer_print(&timer);
    if (time == 0)
        goto fail;
    uint64_t cycles = timer_printCycles(&timer);
    if (cycles == 0)
        goto fail;
    timer_finalize();
    return 1;
fail:
    timer_finalize();
    return 0;
}

int test_timerprint_stop()
{
    TimerData timer;
    timer_init();
    timer_reset(&timer);
    timer_start(&timer);
    timer_stop(&timer);
    double time = timer_print(&timer);
    if (time > 1)
        goto fail;
    if (time == 0)
        goto fail;
    uint64_t cycles = timer_printCycles(&timer);
    if (cycles == 0)
        goto fail;
    if (cycles > timer_getCpuClock())
        goto fail;
    timer_finalize();
    return 1;
fail:
    timer_finalize();
    return 0;
}

int test_timercpuclock_noinit()
{
    uint64_t cyc = timer_getCpuClock();
    if (cyc != 0)
        return 0;
    return 1;
}

int test_timercpuclock()
{
    timer_init();
    uint64_t cyc = timer_getCpuClock();
    if (cyc == 0)
        return 0;
    timer_finalize();
    return 1;
fail:
    timer_finalize();
    return 0;
}

int test_timerbaseline_noinit()
{
    uint64_t cyc = timer_getBaseline();
    if (cyc != 0)
        return 0;
    return 1;
}

int test_timerbaseline()
{
    timer_init();
    uint64_t cyc = timer_getBaseline();
    if (cyc == 0)
        return 0;
    timer_finalize();
    return 1;
fail:
    timer_finalize();
    return 0;
}

int test_timersleep_noinit()
{
    timer_sleep(1E4);
    return 1;
}

int test_timersleep()
{
    timer_init();
    TimerData timer;
    timer_start(&timer);
    timer_sleep(1E6);
    timer_stop(&timer);
    if (timer_print(&timer) < 0.9E6*1E-6)
    {
        printf("Sleeping too short. timer is %f instead of 1 s\n", timer_print(&timer));
        goto fail;
    }
    if (timer_print(&timer) > 1.1E6*1E-6)
    {
        printf("Sleeping too long. timer is %f instead of 1 s\n", timer_print(&timer));
        goto fail;
    }
    timer_finalize();
    return 1;
fail:
    timer_finalize();
    return 0;
}

static test testlist[] = {
    {"Test configuration initialization", test_initconfig, 1},
    {"Enable configuration for following tests", enable_configuration, 1},
    {"Test setting of access mode", test_hpmmode, 1},
    {"Test access initialization", test_hpminit, 1},
    {"Test adding CPU to access module", test_hpmaddthread, 1},
    {"Disable configuration", disable_configuration, 1},
    {"Test perfmon initialization without topology information", test_perfmoninit_faulty, 1},
    {"Test topology module initialization", test_topologyinit, 1},
    {"Test NUMA module initialization", test_numainit, 1},
    {"Test affinity module initialization", test_affinityinit, 1},
    {"Test perfmon initialization with topology information", test_perfmoninit_valid, 1},
    {"Test adding event sets to perfmon module", test_perfmonaddeventset, 1},
    {"Test adding event sets to perfmon module without initialization of perfmon", test_perfmonaddeventset_noinit, 1},
    {"Test setting up an event set", test_perfmonsetup, 1},
    {"Test switching between event sets", test_perfmonswitch, 1},
    {"Test starting an event set", test_perfmonstart, 1},
    {"Test setting up an event set without initialization", test_perfmonsetup_noinit, 1},
    {"Test starting an event set without initialization", test_perfmonstart_noinit, 1},
    {"Test setting up an event set without adding one", test_perfmonsetup_noadd, 1},
    {"Test getting all performance groups", test_perfmongetgroups, 1},
    {"Test setting up a custom event set and test group handling", test_perfmoncustomgroup, 1},
    {"Test setting up a valid performance group and test group handling", test_perfmonperfgroup_ok, 1},
    {"Test setting up a invalid performance group and test group handling", test_perfmonperfgroup_fail, 1},
    {"Test starting an event set without adding one", test_perfmonstart_noadd, 1},
    {"Test stopping an event set", test_perfmonstop, 1},
    {"Test stopping an event set without initialization", test_perfmonstop_noinit, 1},
    {"Test stopping an event set without adding one", test_perfmonstop_noadd, 1},
    {"Test stopping an event set without setting one up", test_perfmonstop_nosetup, 1},
    {"Test stopping an event set without starting one", test_perfmonstop_nostart, 1},
    {"Test perfmon finalization", test_perfmonfinalize, 1},
    {"Test perfmon result without initialization", test_perfmonresult_noinit, 1},
    {"Test perfmon result without adding one", test_perfmonresult_noadd, 1},
    {"Test perfmon result without setting up one", test_perfmonresult_nosetup, 1},
    {"Test perfmon result without starting", test_perfmonresult_nostart, 1},
    {"Test perfmon result without stopping", test_perfmonresult_nostop, 1},
    {"Test perfmon result", test_perfmonresult, 1},
    {"Test perfmon last result without initialization", test_perfmonlastresult_noinit, 1},
    {"Test perfmon last result without adding one", test_perfmonlastresult_noadd, 1},
    {"Test perfmon last result without setting up one", test_perfmonlastresult_nosetup, 1},
    {"Test perfmon last result without starting", test_perfmonlastresult_nostart, 1},
    {"Test perfmon last result without stopping", test_perfmonlastresult_nostop, 1},
    {"Test perfmon last result", test_perfmonlastresult, 1},
    {"Test initialization of timer module", test_timerinit, 1},
    {"Test printing time without initialization", test_timerprint_noinit, 1},
    {"Test printing time", test_timerprint, 1},
    {"Test timer module finalization", test_timerfinalize, 1},
    {"Test printing time for started clock", test_timerprint_start, 1},
    {"Test printing time for started/stopped clock", test_timerprint_stop, 1},
    {"Test reading cpu clock without initialization", test_timercpuclock_noinit, 1},
    {"Test reading cpu clock", test_timercpuclock, 1},
    {"Test reading baseline without initialization", test_timerbaseline_noinit, 1},
    {"Test reading baseline", test_timerbaseline, 1},
    {"Test sleeping with timer module without initialization", test_timersleep_noinit, 1},
    {"Test sleeping with timer module", test_timersleep, 1},
    {"Test perfmon metric without initialization", test_perfmonmetric_noinit, 1},
    {"Test perfmon metric without adding one", test_perfmonmetric_noadd, 1},
    {"Test perfmon metric without setting up one", test_perfmonmetric_nosetup, 1},
    {"Test perfmon metric without starting", test_perfmonmetric_nostart, 1},
    {"Test perfmon metric without stopping", test_perfmonmetric_nostop, 1},
    {"Test perfmon metric", test_perfmonmetric_ok, 1},
    {"Test perfmon last metric without initialization", test_perfmonlastmetric_noinit, 1},
    {"Test perfmon last metric without adding one", test_perfmonlastmetric_noadd, 1},
    {"Test perfmon last metric without setting up one", test_perfmonlastmetric_nosetup, 1},
    {"Test perfmon last metric without starting", test_perfmonlastmetric_nostart, 1},
    {"Test perfmon last metric without stopping", test_perfmonlastmetric_nostop, 1},
    {"Test perfmon last metric", test_perfmonlastmetric_ok, 1},
    {"Test cpustring with logical input", test_cpustring_logical, 1},
    {"Test cpustring with physical input", test_cpustring_physical, 1},
    {"Test cpustring with expression input", test_cpustring_expression, 1},
    {"Test cpustring with scatter input", test_cpustring_scatter, 1},
    {"Test cpustring with combined input", test_cpustring_combined, 1},
    {NULL, NULL, 0},
};

int main()
{
    int i = 0;
    int ret = 0;
    //fclose(stderr);
    if (verbose > 0) perfmon_setVerbosity(3);
    while (testlist[i].testfunc != NULL)
    {
        printf("%s:\t", testlist[i].testname);
        if (verbose > 0) printf("\n");
        ret = testlist[i].testfunc();
        if (ret != testlist[i].result)
        {
            printf("FAILED with return code %d instead of expected %d\n", ret, testlist[i].result);
            return 1;
        }
        printf("OK\n");
        i++;
    }
    printf("All tests completed successfully.\n");
    return 0;
}
