#ifndef LIKWID_TOPOLOGY_HWLOC
#define LIKWID_TOPOLOGY_HWLOC

#ifdef LIKWID_USE_HWLOC
#include <hwloc.h>


static hwloc_topology_t hwloc_topology = NULL;

int hwloc_record_objs_of_type_below_obj(hwloc_topology_t t, hwloc_obj_t obj, hwloc_obj_type_t type, int* index, uint32_t **list);

#endif

void hwloc_init_cpuInfo(void);
void hwloc_init_cpuFeatures(void);
void hwloc_init_nodeTopology(void);
void hwloc_init_cacheTopology(void);


#endif
