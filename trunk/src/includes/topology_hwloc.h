#ifndef LIKWID_TOPOLOGY_HWLOC
#define LIKWID_TOPOLOGY_HWLOC


#include <hwloc.h>
#include <sched.h>


extern hwloc_topology_t hwloc_topology;

int hwloc_record_objs_of_type_below_obj(hwloc_topology_t t, hwloc_obj_t obj, hwloc_obj_type_t type, int* index, uint32_t **list);



void hwloc_init_cpuInfo(cpu_set_t cpuSet);
void hwloc_init_cpuFeatures(void);
void hwloc_init_nodeTopology(cpu_set_t cpuSet);
void hwloc_init_cacheTopology(void);


#endif
