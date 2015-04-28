#ifndef LIKWID_NUMA
#define LIKWID_NUMA

#include <stdlib.h>
#include <stdio.h>

#include <types.h>
#include <likwid.h>
#include <numa_hwloc.h>
#include <numa_proc.h>



extern int maxIdConfiguredNode;

extern int str2int(const char* str);

struct numa_functions {
    int (*numa_init) (void);
    void (*numa_setInterleaved) (int*, int);
    void (*numa_membind) (void*, size_t, int);
};





#endif
