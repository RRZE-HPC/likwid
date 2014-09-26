/*
 * =======================================================================================
 *
 *      Filename:  affinity_types.h
 *
 *      Description:  Type Definitions for affinity Module
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

#ifndef AFFINITY_TYPES_H
#define AFFINITY_TYPES_H

/**
 * \addtogroup AffinityDomains Affinity domain information
 * @{
 */

/*! \brief The AffinityDomain data structure describes a single domain in the current system

The AffinityDomain data structure describes a single domain in the current system. Example domains are NUMA nodes, CPU sockets/packages or LLC (Last Level Cache) cache domains.
\extends AffinityDomains
*/
typedef struct {
    bstring tag; /*!< \brief Bstring with the ID for the affinity domain. Currently possible values: N (node), SX (socket/package X), CX (LLC cache domain X) and MX (memory domain X) */
    uint32_t numberOfProcessors; /*!< \brief Number of HW threads in the domain and length of \a processorList */
    uint32_t numberOfCores; /*!< \brief Number of CPU cores in the domain */
    int*  processorList; /*!< \brief List of HW thread IDs in the domain */
} AffinityDomain;

/*! \brief The AffinityDomains data structure holds different count variables describing the
various system layers

Affinity domains are for example the amount of NUMA domains, CPU sockets/packages or LLC 
(Last Level Cache) cache domains of the current machine. Moreover a list of
\a domains holds the processor lists for each domain that are used for
scheduling processes to domain specific HW threads. Some amounts are duplicates
or derivation of values in \a CpuInfo, \a CpuTopology and \a NumaTopology.
*/
typedef struct {
    uint32_t numberOfSocketDomains; /*!< \brief Number of CPU sockets/packages in the system */
    uint32_t numberOfNumaDomains; /*!< \brief Number of NUMA nodes in the system */
    uint32_t numberOfProcessorsPerSocket; /*!< \brief Number of HW threads per socket/package in the system */
    uint32_t numberOfCacheDomains; /*!< \brief Number of LLC caches in the system */
    uint32_t numberOfCoresPerCache; /*!< \brief Number of HW threads per LLC cache in the system */
    uint32_t numberOfProcessorsPerCache; /*!< \brief Number of CPU cores per LLC cache in the system */
    uint32_t numberOfAffinityDomains; /*!< \brief Number of affinity domains in the current system  and length of \a domains array */
    AffinityDomain* domains; /*!< \brief List of all domains in the system */
} AffinityDomains;

/** \brief Pointer for exporting the AffinityDomains data structure */
typedef AffinityDomains* AffinityDomains_t;

/**@}*/
#endif /*AFFINITY_TYPES_H*/
