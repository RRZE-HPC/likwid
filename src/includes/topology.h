/*
 * =======================================================================================
 *
 *      Filename:  topology.h
 *
 *      Description:  Header File of topology module.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
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
#ifndef TOPOLOGY_H
#define TOPOLOGY_H

#include <stdlib.h>
#include <stdio.h>

#if !defined(__ARM_ARCH_7A__) && !defined(__ARM_ARCH_8A__)
#include <topology_cpuid.h>
#endif
#include <topology_proc.h>
#ifdef LIKWID_USE_HWLOC
#include <topology_hwloc.h>
#endif
#include <types.h>
#include <tree.h>

#define MAX_FEATURE_STRING_LENGTH 512
#define MAX_MODEL_STRING_LENGTH 512

struct topology_functions {
    void (*init_cpuInfo) (cpu_set_t cpuSet);
    void (*init_cpuFeatures) (void);
    void (*init_nodeTopology) (cpu_set_t cpuSet);
    void (*init_cacheTopology) (void);
    void (*init_fileTopology) (FILE*);
    void (*close_topology) (void);
};

/* Intel P6 */
#define PENTIUM_M_BANIAS     0x09U
#define PENTIUM_M_DOTHAN     0x0DU
#define CORE_DUO             0x0EU
#define CORE2_65             0x0FU
#define CORE2_45             0x17U
#define ATOM                 0x1CU
#define ATOM_45              0x26U
#define ATOM_32              0x36U
#define ATOM_22              0x27U
#define ATOM_SILVERMONT_E    0x37U
#define ATOM_SILVERMONT_C    0x4DU
#define ATOM_SILVERMONT_Z1   0x4AU
#define ATOM_SILVERMONT_Z2   0x5AU
#define ATOM_SILVERMONT_F    0x5DU
#define ATOM_SILVERMONT_AIR  0x4CU
#define ATOM_SILVERMONT_GOLD 0x5CU
#define ATOM_DENVERTON       0x5FU
#define ATOM_GOLDMONT_PLUS   0x7AU
#define NEHALEM              0x1AU
#define NEHALEM_BLOOMFIELD   0x1AU
#define NEHALEM_LYNNFIELD    0x1EU
#define NEHALEM_LYNNFIELD_M  0x1FU
#define NEHALEM_WESTMERE     0x2CU
#define NEHALEM_WESTMERE_M   0x25U
#define SANDYBRIDGE          0x2AU
#define SANDYBRIDGE_EP       0x2DU
#define HASWELL              0x3CU
#define HASWELL_EP           0x3FU
#define HASWELL_M1           0x45U
#define HASWELL_M2           0x46U
#define IVYBRIDGE            0x3AU
#define IVYBRIDGE_EP         0x3EU
#define NEHALEM_EX           0x2EU
#define WESTMERE_EX          0x2FU
#define XEON_MP              0x1DU
#define BROADWELL            0x3DU
#define BROADWELL_E          0x4FU
#define BROADWELL_D          0x56U
#define BROADWELL_E3         0x47U
#define SKYLAKE1             0x4EU
#define SKYLAKE2             0x5EU
#define SKYLAKEX             0x55U
#define KABYLAKE1            0x8EU
#define KABYLAKE2            0x9EU
#define CANNONLAKE           0x66U

/* Intel MIC */
#define XEON_PHI           0x01U
#define XEON_PHI_KNL       0x57U
#define XEON_PHI_KML       0x85U

/* AMD K10 */
#define BARCELONA      0x02U
#define SHANGHAI       0x04U
#define ISTANBUL       0x08U
#define MAGNYCOURS     0x09U
#define THUBAN         0x0AU

/* AMD K8 */
#define OPTERON_SC_1MB  0x05U
#define OPTERON_DC_E    0x21U
#define OPTERON_DC_F    0x41U
#define ATHLON64_X2     0x43U
#define ATHLON64_X2_F   0x4BU
#define ATHLON64_F1     0x4FU
#define ATHLON64_F2     0x5FU
#define ATHLON64_X2_G   0x6BU
#define ATHLON64_G1     0x6FU
#define ATHLON64_G2     0x7FU
#define ZEN_RYZEN       0x01

/* ARM */
#define  ARM7L          0x3U
#define  CORTEX_A57_1   0x1U
#define  CORTEX_A53_1   0xD03U

#define  P6_FAMILY        0x6U
#define  MIC_FAMILY       0xBU
#define  NETBURST_FAMILY  0xFFU
#define  ZEN_FAMILY       0x17
#define  K15_FAMILY       0x15U
#define  K16_FAMILY       0x16U
#define  K10_FAMILY       0x10U
#define  K8_FAMILY        0xFU
#define  ARMV7_FAMILY     0x7U
#define  ARMV8_FAMILY     0x8U

extern int cpu_count(cpu_set_t* set);

static inline int cpuid_hasFeature(FeatureBit bit)
{
      return (cpuid_info.featureFlags & (1<<bit));
}

#endif /* TOPOLOGY_H */
