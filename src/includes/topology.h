#ifndef LIKWID_TOPOLOGY
#define LIKWID_TOPOLOGY

#include <stdlib.h>
#include <stdio.h>

#include <topology_cpuid.h>
#include <topology_proc.h>
#ifdef LIKWID_USE_HWLOC
#include <topology_hwloc.h>
#endif
#include <types.h>
#include <tree.h>


#define MAX_FEATURE_STRING_LENGTH 512
#define MAX_MODEL_STRING_LENGTH 512

extern int affinity_thread2tile_lookup[MAX_NUM_THREADS];
extern int affinity_thread2tile_lookup[MAX_NUM_THREADS];
struct topology_functions {
    void (*init_cpuInfo) (cpu_set_t cpuSet);
    void (*init_cpuFeatures) (void);
    void (*init_nodeTopology) (cpu_set_t cpuSet);
    void (*init_cacheTopology) (void);
    void (*init_fileTopology) (FILE*);
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

/* Intel MIC */
#define XEON_PHI           0x01U

/* AMD K10 */
#define BARCELONA      0x02U
#define SHANGHAI       0x04U
#define ISTANBUL       0x08U
#define MAGNYCOURS     0x09U

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


#define  P6_FAMILY        0x6U
#define  MIC_FAMILY       0xBU
#define  NETBURST_FAMILY  0xFFU
#define  K15_FAMILY       0x15U
#define  K16_FAMILY       0x16U
#define  K10_FAMILY       0x10U
#define  K8_FAMILY        0xFU





extern int cpu_count(cpu_set_t* set);

static inline int cpuid_hasFeature(FeatureBit bit)
{
      return (cpuid_info.featureFlags & (1<<bit));
}


#endif
