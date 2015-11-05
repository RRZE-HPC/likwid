#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include <likwid.h>
#include <configuration.h>
#include <access.h>
#include <types.h>
#include <perfmon.h>

typedef struct {
    char* testname;
    int(*testfunc)(void);
    int result;
} test;

static char eventset_ok[] = "INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1";
static char eventset_option[] = "CPU_CLK_UNHALTED_CORE:FIXC1:ANYTHREAD";

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
    numa_init();
    NumaTopology_t numainfo = get_numaTopology();
    if (numainfo == NULL)
        goto fail;
    if (numainfo->numberOfNodes <= 0)
        goto fail;
    if (likwid_getNumberOfNodes() <= 0)
        goto fail;
    for (i = 0; i < likwid_getNumberOfNodes(); i++)
    {
        if (numainfo->nodes[i].totalMemory == 0)
            goto fail;
        if (numainfo->nodes[i].freeMemory == 0)
            goto fail;
        if (numainfo->nodes[i].numberOfProcessors == 0)
            goto fail;
        if (numainfo->nodes[i].numberOfDistances == 0)
            goto fail;
        if (numainfo->nodes[i].numberOfDistances != likwid_getNumberOfNodes())
            goto fail;
    }
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
    topology_init();
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
        if (doms->domains[i].numberOfProcessors == 0)
            goto fail;
        if (doms->domains[i].numberOfCores == 0)
            goto fail;
        if (doms->domains[i].numberOfProcessors < doms->domains[i].numberOfCores)
            goto fail;
        if (doms->domains[i].processorList == NULL)
            goto fail;
    }
    affinity_finalize();
    topology_finalize();
    return 1;
fail:
    affinity_finalize();
    topology_finalize();
    return 0;
}

int test_perfmoninit_faulty()
{
    int cpu = 0;
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    perfmon_finalize();
    return 0;
fail:
    perfmon_finalize();
    return 1;
}

int test_perfmoninit_valid()
{
    int cpu = 0;
    topology_init();
    affinity_init();
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (counter_map == NULL)
        goto fail;
    if (box_map == NULL)
        goto fail;
    if (eventHash == NULL)
        goto fail;
    if (perfmon_getNumberOfGroups() != 0)
        goto fail;
    if (perfmon_getNumberOfThreads() != 1)
        goto fail;
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
    if (cpuinfo->isIntel == 0)
    {
        topology_finalize();
        return 1;
    }
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    if (perfmon_getNumberOfGroups() != 0)
        goto fail;
    if (perfmon_getNumberOfThreads() != 1)
        goto fail;
    if (perfmon_getIdOfActiveGroup() != -1)
        goto fail;
    ret = perfmon_addEventSet(eventset_ok);
    if (ret != 0)
        goto fail;
    if (perfmon_getNumberOfGroups() != 1)
        goto fail;
    if (perfmon_getNumberOfEvents(ret) != 2)
        goto fail;
    if (perfmon_getIdOfActiveGroup() != -1)
        goto fail;
    ret = perfmon_addEventSet(eventset_option);
    if (ret != 1)
        goto fail;
    if (perfmon_getNumberOfGroups() != 2)
        goto fail;
    if (perfmon_getNumberOfEvents(ret) != 1)
        goto fail;
    if (perfmon_getIdOfActiveGroup() != -1)
        goto fail;
    ret = perfmon_addEventSet(eventset_fail1);
    if (ret >= 0)
        goto fail;
    if (perfmon_getNumberOfGroups() != 2)
        goto fail;
    ret = perfmon_addEventSet(eventset_fail2);
    if (ret >= 0)
        goto fail;
    if (perfmon_getNumberOfGroups() != 2)
        goto fail;
    if (perfmon_getIdOfActiveGroup() != -1)
        goto fail;
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
    int ret = perfmon_addEventSet(eventset_ok);
    if (ret == 0)
        goto fail;
    return 1;
fail:
    return 0;
}

int test_perfmonsetup()
{
    CpuInfo_t cpuinfo;
    int group1, group2;
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
    if (perfmon_getNumberOfGroups() != 0)
        goto fail;
    if (perfmon_getNumberOfThreads() != 1)
        goto fail;
    ret = perfmon_addEventSet(eventset_ok);
    if (ret != 0)
        goto fail;
    group1 = ret;
    if (perfmon_getNumberOfGroups() != 1)
        goto fail;
    if (perfmon_getNumberOfEvents(group1) != 2)
        goto fail;
    ret = perfmon_setupCounters(group1);
    if (ret != 0)
        goto fail;
    if (perfmon_getIdOfActiveGroup() != group1)
        goto fail;
    ret = perfmon_addEventSet(eventset_option);
    if (ret != 1)
        goto fail;
    group2 = ret;
    if (perfmon_getIdOfActiveGroup() != group1)
        goto fail;
    if (perfmon_getNumberOfGroups() != 2)
        goto fail;
    if (perfmon_getNumberOfEvents(group1) != 2)
        goto fail;
    if (perfmon_getNumberOfEvents(group2) != 1)
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
    if (cpuinfo->isIntel == 0)
    {
        topology_finalize();
        return 1;
    }
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    ret = perfmon_addEventSet(eventset_ok);
    if (ret != 0)
        goto fail;
    group1 = ret;
    ret = perfmon_addEventSet(eventset_option);
    if (ret != 1)
        goto fail;
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
    if (cpuinfo->isIntel == 0)
    {
        topology_finalize();
        return 1;
    }
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    ret = perfmon_addEventSet(eventset_ok);
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
    if (cpuinfo->isIntel == 0)
    {
        topology_finalize();
        return 1;
    }
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    ret = perfmon_addEventSet(eventset_ok);
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
    if (cpuinfo->isIntel == 0)
    {
        topology_finalize();
        return 1;
    }
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    ret = perfmon_addEventSet(eventset_ok);
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
    if (cpuinfo->isIntel == 0)
    {
        topology_finalize();
        return 1;
    }
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    ret = perfmon_addEventSet(eventset_ok);
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
    if (result != 0)
        goto fail;
    return 1;
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
    if (cpuinfo->isIntel == 0)
    {
        topology_finalize();
        return 1;
    }
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    double result = perfmon_getResult(0,0,0);
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

int test_perfmonresult_nosetup()
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
    ret = perfmon_addEventSet(eventset_ok);
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
    if (cpuinfo->isIntel == 0)
    {
        topology_finalize();
        return 1;
    }
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    ret = perfmon_addEventSet(eventset_ok);
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
    if (cpuinfo->isIntel == 0)
    {
        topology_finalize();
        return 1;
    }
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    ret = perfmon_addEventSet(eventset_ok);
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
    if (cpuinfo->isIntel == 0)
    {
        topology_finalize();
        return 1;
    }
    int ret = perfmon_init(1, &cpu);
    if (ret != 0)
        goto fail;
    ret = perfmon_addEventSet(eventset_ok);
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
    timer_sleep(1E4);
    timer_stop(&timer);
    if (timer_print(&timer) < 0.01)
        goto fail;
    if (timer_print(&timer) > 0.015)
        goto fail;
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
    {NULL, NULL, 0},
};

int main()
{
    int i = 0;
    while (testlist[i].testfunc != NULL)
    {
        printf("%s:\t", testlist[i].testname);
        if (testlist[i].testfunc() != testlist[i].result)
        {
            printf("FAILED\n");
            return 1;
        }
        printf("OK\n");
        i++;
    }
    printf("All tests completed successfully.\n");
    return 0;
}
