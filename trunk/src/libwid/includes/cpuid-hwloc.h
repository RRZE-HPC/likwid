#ifndef LIKWID_HWLOC_H
#define LIKWID_HWLOC_H

#ifdef LIKWID_USE_HWLOC
#include <hwloc.h>

static hwloc_topology_t hwloc_topology = NULL;

extern int hwloc_record_objs_of_type_below_obj(hwloc_topology_t t, hwloc_obj_t obj, hwloc_obj_type_t type, int* index, uint32_t **list);
#endif

#endif
