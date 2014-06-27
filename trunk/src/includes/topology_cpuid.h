#ifndef LIKWID_TOPOLOGY_CPUID
#define LIKWID_TOPOLOGY_CPUID

void cpuid_init_cpuInfo(void);
void cpuid_init_cpuFeatures(void);
void cpuid_init_nodeTopology(void);
void cpuid_init_cacheTopology(void);


#endif
