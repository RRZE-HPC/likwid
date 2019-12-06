/*
 * =======================================================================================
 *
 *      Filename:  topology_hwloc.h
 *
 *      Description:  Header File of topology backend using the hwloc library
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */
#ifndef TOPOLOGY_HWLOC_H
#define TOPOLOGY_HWLOC_H

#include <hwloc.h>
#include <sched.h>

extern hwloc_topology_t hwloc_topology;

int likwid_hwloc_record_objs_of_type_below_obj(hwloc_topology_t t, hwloc_obj_t obj, hwloc_obj_type_t type, int* index, uint32_t **list);

void hwloc_init_cpuInfo(cpu_set_t cpuSet);
void hwloc_init_cpuFeatures(void);
void hwloc_init_nodeTopology(cpu_set_t cpuSet);
void hwloc_init_cacheTopology(void);
void hwloc_close(void);

#endif /* TOPOLOGY_HWLOC_H */
