#ifndef LIKWID_TOPOLOGY_CPUID
#define LIKWID_TOPOLOGY_CPUID

#include <sched.h>

void cpuid_init_cpuInfo(cpu_set_t cpuSet);
void cpuid_init_cpuFeatures(void);
void cpuid_init_nodeTopology(cpu_set_t cpuSet);
void cpuid_init_cacheTopology(void);


#endif
