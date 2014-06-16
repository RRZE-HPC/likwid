#ifndef LIKWID_NUMA_PROC
#define LIKWID_NUMA_PROC

extern int proc_numa_init(void);
extern void proc_numa_membind(void* ptr, size_t size, int domainId);
extern void proc_numa_setInterleaved(int* processorList, int numberOfProcessors);


#endif
