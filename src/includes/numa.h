#ifndef LIKWID_NUMA
#define LIKWID_NUMA

#include <stdlib.h>
#include <stdio.h>

#include <types.h>

#include <numa_hwloc.h>
#include <numa_proc.h>



extern int maxIdConfiguredNode;

struct numa_functions {
    int (*numa_init) (void);
    void (*numa_setInterleaved) (int*, int);
    void (*numa_membind) (void*, size_t, int);
};



void numa_setInterleaved(int* processorList, int numberOfProcessors);
void numa_membind(void* ptr, size_t size, int domainId);
void numa_finalize(void);

int likwid_getNumberOfNodes(void);

#endif
