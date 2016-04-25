#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <signal.h>
#include <likwid.h>


static int sleeptime = 1;

static int run = 1;

void  INThandler(int sig)
{
    signal(sig, SIG_IGN);
    run = 0;
}


int main (int argc, char* argv[])
{
    int i, c, err = 0;
    double timer = 0.0;
    topology_init();
    numa_init();
    affinity_init();
    timer_init();
    CpuInfo_t cpuinfo = get_cpuInfo();
    CpuTopology_t cputopo = get_cpuTopology();
    int numCPUs = cputopo->activeHWThreads;
    int* cpus = malloc(numCPUs * sizeof(int));
    if (!cpus)
    {
        affinity_finalize();
        numa_finalize();
        topology_finalize();
        return 1;
    }
    c = 0;
    for (i=0;i<cputopo->numHWThreads;i++)
    {
        if (cputopo->threadPool[i].inCpuSet)
        {
            cpus[c] = cputopo->threadPool[i].apicId;
            c++;
        }
    }
    NumaTopology_t numa = get_numaTopology();
    AffinityDomains_t affi = get_affinityDomains();
    timer = timer_getCpuClock();
    perfmon_init(numCPUs, cpus);
    int gid1 = perfmon_addEventSet("L2");
    if (gid1 < 0)
    {
        printf("Failed to add performance group L2\n");
        err = 1;
        goto monitor_exit;
    }
    int gid2 = perfmon_addEventSet("L3");
    if (gid2 < 0)
    {
        printf("Failed to add performance group L3\n");
        err = 1;
        goto monitor_exit;
    }
    int gid3 = perfmon_addEventSet("ENERGY");
    if (gid3 < 0)
    {
        printf("Failed to add performance group ENERGY\n");
        err = 1;
        goto monitor_exit;
    }
    signal(SIGINT, INThandler);

    while (run)
    {
        perfmon_setupCounters(gid1);
        perfmon_startCounters();
        sleep(sleeptime);
        perfmon_stopCounters();
        for (c = 0; c < 8; c++)
        {
            for (i = 0; i< perfmon_getNumberOfMetrics(gid1); i++)
            {
                printf("%s,cpu=%d %f\n", perfmon_getMetricName(gid1, i), cpus[c], perfmon_getLastMetric(gid1, i, c));
            }
        }
        perfmon_setupCounters(gid2);
        perfmon_startCounters();
        sleep(sleeptime);
        perfmon_stopCounters();
        for (c = 0; c < 8; c++)
        {
            for (i = 0; i< perfmon_getNumberOfMetrics(gid2); i++)
            {
                printf("%s,cpu=%d %f\n", perfmon_getMetricName(gid2, i), cpus[c], perfmon_getLastMetric(gid2, i, c));
            }
        }
        perfmon_setupCounters(gid3);
        perfmon_startCounters();
        sleep(sleeptime);
        perfmon_stopCounters();
        for (c = 0; c < 8; c++)
        {
            for (i = 0; i< perfmon_getNumberOfMetrics(gid3); i++)
            {
                printf("%s,cpu=%d %f\n", perfmon_getMetricName(gid3, i), cpus[c], perfmon_getLastMetric(gid3, i, c));
            }
        }
    }
monitor_exit:
    free(cpus);
    perfmon_finalize();
    affinity_finalize();
    numa_finalize();
    topology_finalize();
    return 0;
}
