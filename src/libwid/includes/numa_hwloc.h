#ifndef LIKWID_NUMA_HWLOC
#define LIKWID_NUMA_HWLOC

extern int hwloc_numa_init(void);
extern void hwloc_numa_membind(void* ptr, size_t size, int domainId);
extern void hwloc_numa_setInterleaved(int* processorList, int numberOfProcessors);


#endif
