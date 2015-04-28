#ifndef LIKWID_TOPOLOGY_PROC
#define LIKWID_TOPOLOGY_PROC

#include <stdlib.h>
#include <stdio.h>
#include <sched.h>
#include <unistd.h>
#include <sched.h>

#include <error.h>
#include <tree.h>
#include <bitUtil.h>
//#include <strUtil.h>
//#include <tlb-info.h>
#include <topology.h>

void proc_init_cpuInfo(cpu_set_t cpuSet);
void proc_init_cpuFeatures(void);
void proc_init_nodeTopology(cpu_set_t cpuSet);
void proc_init_cacheTopology(void);


#endif
