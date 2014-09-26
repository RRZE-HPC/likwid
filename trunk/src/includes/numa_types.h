/*
 * =======================================================================================
 *
 *      Filename:  numa_types.h
 *
 *      Description:  Types file for numa module.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2013 Jan Treibig 
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

#ifndef NUMA_TYPES_H
#define NUMA_TYPES_H


/** \addtogroup NumaTopology NUMA memory topology
*  @{
*/
/*! \brief CPUs in NUMA node and general information about a NUMA domain

The NumaNode structure describes the topology and holds general information of a
NUMA node. The structure is filled by calling numa_init() by either the HWLOC 
library or by evaluating the /proc filesystem.
\extends NumaTopology
*/
typedef struct {
    uint32_t id; /*!< \brief ID of the NUMA node */
    uint64_t totalMemory; /*!< \brief Amount of memory in the NUMA node */
    uint64_t freeMemory; /*!< \brief Amount of free memory in the NUMA node */
    uint32_t numberOfProcessors; /*!< \brief umber of processors covered by the NUMA node and length of \a processors */
    uint32_t*  processors; /*!< \brief List of HW threads in the NUMA node */
    uint32_t*  processorsCompact; /*!< \brief Currently unused */
    uint32_t numberOfDistances; /*!< \brief Amount of distances to the other NUMA nodes in the system and self  */
    uint32_t*  distances; /*!< \brief List of distances to the other NUMA nodes and self */
} NumaNode;


/*! \brief  The NumaTopology structure describes all NUMA nodes in the current system.
*/
typedef struct {
    uint32_t numberOfNodes; /*!< \brief Number of NUMA nodes in the system and length of \a nodes  */
    NumaNode* nodes; /*!< \brief List of NUMA nodes */
} NumaTopology;

/** \brief Pointer for exporting the NumaTopology data structure */
typedef NumaTopology* NumaTopology_t;
/** @}*/
#endif /*NUMA_TYPES_H*/
