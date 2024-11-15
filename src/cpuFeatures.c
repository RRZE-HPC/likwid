/*
 * =======================================================================================
 *
 *     Filename:  cpuFeatures.c
 *
 *     Description:  Implementation of cpuFeatures Module.
 *                  Provides an API to read out and print the IA32_MISC_ENABLE
 *                  model specific register on Intel x86 processors.
 *                  Allows to turn on and off the Hardware prefetcher
 *                  available.
 *
 *      Version:   5.4.0
 *      Released:  15.11.2024
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2024 RRZE, University Erlangen-Nuremberg
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <types.h>
#include <access.h>
#include <topology.h>
#include <registers.h>
#include <textcolor.h>
#include <likwid.h>
#include <lock.h>

/* #####   EXPORTED VARIABLES   ########################################### */

static uint64_t *cpuFeatureMask = NULL;
static int features_initialized = 0;

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define PRINT_VALUE(color,string)  \
    color_on(BRIGHT,(color));      \
    printf(#string"\n");            \
    color_reset()

#define TEST_FLAG(feature,flag)  \
    if (flags & (1ULL<<(flag)))   \
    { \
        cpuFeatureMask[cpu] |= (1ULL<<feature); \
    } \
    else \
    { \
        cpuFeatureMask[cpu] &= ~(1ULL<<feature); \
    }

#define TEST_FLAG_INV(feature,flag)  \
    if (flags & (1ULL<<(flag)))   \
    { \
        cpuFeatureMask[cpu] &= ~(1ULL<<feature); \
    } \
    else \
    { \
        cpuFeatureMask[cpu] |= (1ULL<<feature); \
    }

#define IF_FLAG(feature) (cpuFeatureMask[cpu] & (1ULL<<feature))

/* #####   FUNCTIONS  -  LOCAL TO THIS SOURCE FILE   ######################### */

static void
cpuFeatures_update(int cpu)
{
    int ret;
    uint64_t flags = 0x0ULL;
    ret = HPMread(cpu, MSR_DEV, MSR_IA32_MISC_ENABLE, &flags);
    if (ret != 0)
    {
        fprintf(stderr, "Cannot read register 0x%X on cpu %d: err %d\n", MSR_IA32_MISC_ENABLE, cpu, ret);
    }

    /*cpuFeatureFlags.fastStrings = 0;
    cpuFeatureFlags.thermalControl = 0;
    cpuFeatureFlags.perfMonitoring = 0;
    cpuFeatureFlags.hardwarePrefetcher = 0;
    cpuFeatureFlags.ferrMultiplex = 0;
    cpuFeatureFlags.branchTraceStorage = 0;
    cpuFeatureFlags.pebs = 0;
    cpuFeatureFlags.speedstep = 0;
    cpuFeatureFlags.monitor = 0;
    cpuFeatureFlags.clPrefetcher = 0;
    cpuFeatureFlags.speedstepLock = 0;
    cpuFeatureFlags.cpuidMaxVal = 0;
    cpuFeatureFlags.xdBit = 0;
    cpuFeatureFlags.dcuPrefetcher = 0;
    cpuFeatureFlags.dynamicAcceleration = 0;
    cpuFeatureFlags.turboMode = 0;
    cpuFeatureFlags.ipPrefetcher = 0;*/

    TEST_FLAG(FEAT_FAST_STRINGS,0);
    TEST_FLAG(FEAT_THERMAL_CONTROL,3);
    TEST_FLAG(FEAT_PERF_MON,7);
    TEST_FLAG_INV(FEAT_BRANCH_TRACE_STORAGE,11);
    TEST_FLAG_INV(FEAT_PEBS,12);
    TEST_FLAG(FEAT_SPEEDSTEP,16);
    TEST_FLAG(FEAT_MONITOR,18);
    TEST_FLAG(FEAT_CPUID_MAX_VAL,22);
    TEST_FLAG_INV(FEAT_XTPR_MESSAGE, 23);
    TEST_FLAG_INV(FEAT_XD_BIT,34);

    if ((cpuid_info.model == CORE2_45) ||
        (cpuid_info.model == CORE2_65))
    {
        TEST_FLAG_INV(FEAT_HW_PREFETCHER,9);
        TEST_FLAG(FEAT_FERR_MULTIPLEX,10);
        TEST_FLAG(FEAT_TM2,13);
        TEST_FLAG_INV(FEAT_CL_PREFETCHER,19);
        TEST_FLAG(FEAT_SPEEDSTEP_LOCK,20);
        TEST_FLAG_INV(FEAT_DCU_PREFETCHER,37);
        TEST_FLAG_INV(FEAT_DYN_ACCEL,38);
        TEST_FLAG_INV(FEAT_IP_PREFETCHER,39);
    }
    else if ((cpuid_info.model == NEHALEM) ||
             (cpuid_info.model == NEHALEM_BLOOMFIELD) ||
             (cpuid_info.model == NEHALEM_LYNNFIELD) ||
             (cpuid_info.model == NEHALEM_WESTMERE) ||
             (cpuid_info.model == NEHALEM_WESTMERE_M) ||
             (cpuid_info.model == NEHALEM_EX) ||
             (cpuid_info.model == WESTMERE_EX) ||
             (cpuid_info.model == ATOM_SILVERMONT_E) ||
             (cpuid_info.model == ATOM_SILVERMONT_C) ||
             (cpuid_info.model == ATOM_SILVERMONT_Z1) ||
             (cpuid_info.model == ATOM_SILVERMONT_Z2) ||
             (cpuid_info.model == ATOM_SILVERMONT_F) ||
             (cpuid_info.model == ATOM_SILVERMONT_AIR) ||
             (cpuid_info.model == ATOM_SILVERMONT_GOLD) ||
             (cpuid_info.model == SANDYBRIDGE) ||
             (cpuid_info.model == SANDYBRIDGE_EP) ||
             (cpuid_info.model == IVYBRIDGE) ||
             (cpuid_info.model == IVYBRIDGE_EP) ||
             (cpuid_info.model == HASWELL) ||
             (cpuid_info.model == HASWELL_M1) ||
             (cpuid_info.model == HASWELL_M2) ||
             (cpuid_info.model == HASWELL_EP) ||
             (cpuid_info.model == BROADWELL) ||
             (cpuid_info.model == BROADWELL_E3) ||
             (cpuid_info.model == BROADWELL_D) ||
             (cpuid_info.model == BROADWELL_E) ||
             (cpuid_info.model == SKYLAKE1) ||
             (cpuid_info.model == SKYLAKE2) ||
             (cpuid_info.model == SKYLAKEX) ||
             (cpuid_info.model == KABYLAKE1) ||
             (cpuid_info.model == KABYLAKE2) ||
             (cpuid_info.model == CANNONLAKE) ||
             (cpuid_info.model == COMETLAKE1) ||
             (cpuid_info.model == COMETLAKE2))
    {
        TEST_FLAG_INV(FEAT_TURBO_MODE,38);
    }

    if ((cpuid_info.model == NEHALEM) ||
            (cpuid_info.model == NEHALEM_BLOOMFIELD) ||
            (cpuid_info.model == NEHALEM_LYNNFIELD) ||
            (cpuid_info.model == NEHALEM_WESTMERE) ||
            (cpuid_info.model == NEHALEM_WESTMERE_M) ||
            (cpuid_info.model == NEHALEM_EX) ||
            (cpuid_info.model == WESTMERE_EX) ||
            (cpuid_info.model == SANDYBRIDGE) ||
            (cpuid_info.model == SANDYBRIDGE_EP) ||
            (cpuid_info.model == IVYBRIDGE) ||
            (cpuid_info.model == IVYBRIDGE_EP) ||
            (cpuid_info.model == HASWELL) ||
            (cpuid_info.model == HASWELL_M1) ||
            (cpuid_info.model == HASWELL_M2) ||
            (cpuid_info.model == HASWELL_EP) ||
            (cpuid_info.model == BROADWELL) ||
            (cpuid_info.model == BROADWELL_E3) ||
            (cpuid_info.model == BROADWELL_D) ||
            (cpuid_info.model == BROADWELL_E) ||
            (cpuid_info.model == SKYLAKE1) ||
            (cpuid_info.model == SKYLAKE2) ||
            (cpuid_info.model == SKYLAKEX) ||
            (cpuid_info.model == KABYLAKE1) ||
            (cpuid_info.model == KABYLAKE2) ||
            (cpuid_info.model == ICELAKE1) ||
            (cpuid_info.model == ICELAKE2) ||
            (cpuid_info.model == ROCKETLAKE) ||
            (cpuid_info.model == ICELAKEX1) ||
            (cpuid_info.model == ICELAKEX2) ||
            (cpuid_info.model == SAPPHIRERAPIDS) ||
            (cpuid_info.model == ATOM_SILVERMONT_GOLD) ||
            (cpuid_info.model == CANNONLAKE) ||
            (cpuid_info.model == COMETLAKE1) ||
            (cpuid_info.model == COMETLAKE2))
    {
        ret = HPMread(cpu, MSR_DEV, MSR_PREFETCH_ENABLE, &flags);
        if (ret != 0)
        {
            fprintf(stderr,
                    "Cannot read register 0x%X on cpu %d: err %d\n",
                    MSR_PREFETCH_ENABLE, cpu, ret);
        }
        TEST_FLAG_INV(FEAT_IP_PREFETCHER,3);
        TEST_FLAG_INV(FEAT_DCU_PREFETCHER,2);
        TEST_FLAG_INV(FEAT_CL_PREFETCHER,1);
        TEST_FLAG_INV(FEAT_HW_PREFETCHER,0);
    }

    if ((cpuid_info.model == XEON_PHI_KNL) ||
        (cpuid_info.model == XEON_PHI_KML))
    {
        ret = HPMread(cpu, MSR_DEV, MSR_PREFETCH_ENABLE, &flags);
        if (ret != 0)
        {
            fprintf(stderr,
                    "Cannot read register 0x%X on cpu %d: err %d\n",
                    MSR_PREFETCH_ENABLE, cpu, ret);
        }
        TEST_FLAG_INV(FEAT_DCU_PREFETCHER,0);
        TEST_FLAG_INV(FEAT_HW_PREFETCHER,1);
    }
}

static char*
cpuFeatureNames[CPUFEATURES_MAX] = {
    [FEAT_HW_PREFETCHER] = "Hardware Prefetcher",
    [FEAT_IP_PREFETCHER] = "IP Prefetcher",
    [FEAT_DCU_PREFETCHER] = "DCU Pretecher",
    [FEAT_CL_PREFETCHER] = "Adjacent Cache Line Prefetcher",
    [FEAT_FAST_STRINGS] = "Fast-Strings",
    [FEAT_THERMAL_CONTROL] = "Automatic Thermal Control Circuit",
    [FEAT_PERF_MON] = "Performance Monitoring",
    [FEAT_BRANCH_TRACE_STORAGE] = "Branch Trace Storage",
    [FEAT_PEBS] = "Precise Event Based Sampling (PEBS)",
    [FEAT_SPEEDSTEP] = "Enhanced Intel SpeedStep Technology",
    [FEAT_MONITOR] = "MONITOR/MWAIT",
    [FEAT_CPUID_MAX_VAL] = "Limit CPUID Maxval",
    [FEAT_XD_BIT] = "Execute Disable Bit",
    [FEAT_TURBO_MODE] = "Intel Turbo Mode",
    [FEAT_DYN_ACCEL] = "Intel Dynamic Acceleration",
    [FEAT_FERR_MULTIPLEX] = "FERR# Multiplexing",
    [FEAT_XTPR_MESSAGE] = "xTPR Message",
    [FEAT_TM2] = "Thermal Monitoring 2",
    [FEAT_SPEEDSTEP_LOCK] = "Enhanced Intel SpeedStep Technology Select Lock",
};

static char*
cpuFeatureNamesFixed[CPUFEATURES_MAX] = {};

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
cpuFeatures_init()
{
    if (features_initialized)
    {
        return;
    }
    if (!lock_check())
    {
        fprintf(stderr,"Access to CPU feature backend is locked.\n");
        return;
    }

    topology_init();
    if (!cpuFeatureMask)
    {
        cpuFeatureMask = malloc(cpuid_topology.numHWThreads*sizeof(uint64_t));
        memset(cpuFeatureMask, 0, cpuid_topology.numHWThreads*sizeof(uint64_t));
    }

    if (!HPMinitialized())
    {
        HPMinit();
    }
    for (int i = 0; i < cpuid_topology.numHWThreads; i++)
    {
        int ret = HPMaddThread(cpuid_topology.threadPool[i].apicId);
        if (ret != 0)
        {
            ERROR_PRINT(Cannot get access to register CPU feature register on CPU %d, cpuid_topology.threadPool[i].apicId);
            return;
        }
    }
    for (int i = 0; i < cpuid_topology.numHWThreads; i++)
    {
        cpuFeatures_update(cpuid_topology.threadPool[i].apicId);
    }

    features_initialized = 1;
}

void
cpuFeatures_print(int cpu)
{
    uint64_t flags = 0x0ULL;
    if (!features_initialized)
    {
        return;
    }
    cpuFeatures_update(cpu);

    printf(HLINE);
    for (int i=0; i<CPUFEATURES_MAX; i++)
    {
        if ((cpuid_info.model != CORE2_45) &&
            (cpuid_info.model != CORE2_65) &&
            ((i == FEAT_FERR_MULTIPLEX) ||
             (i == FEAT_DYN_ACCEL) ||
             (i == FEAT_SPEEDSTEP_LOCK) ||
             (i == FEAT_TM2)))
        {
            continue;
        }
        printf("%-48s: ",cpuFeatureNames[i]);
        if (IF_FLAG(i))
        {
            PRINT_VALUE(GREEN, enabled);
        }
        else
        {
            PRINT_VALUE(RED,disabled);
        }
    }
    printf(HLINE);
}

int
cpuFeatures_enable(int cpu, CpuFeature type, int print)
{
    int ret;
    uint64_t flags;
    uint32_t reg = MSR_IA32_MISC_ENABLE;
    int newOffsets = 0;
    int knlOffsets = 0;
    if (!features_initialized)
    {
        return -1;
    }
    if (IF_FLAG(type))
    {
        return 0;
    }
    if ((cpuid_info.model == NEHALEM) ||
            (cpuid_info.model == NEHALEM_BLOOMFIELD) ||
            (cpuid_info.model == NEHALEM_LYNNFIELD) ||
            (cpuid_info.model == NEHALEM_WESTMERE) ||
            (cpuid_info.model == NEHALEM_WESTMERE_M) ||
            (cpuid_info.model == NEHALEM_EX) ||
            (cpuid_info.model == WESTMERE_EX) ||
            (cpuid_info.model == SANDYBRIDGE) ||
            (cpuid_info.model == SANDYBRIDGE_EP) ||
            (cpuid_info.model == IVYBRIDGE) ||
            (cpuid_info.model == IVYBRIDGE_EP) ||
            (cpuid_info.model == HASWELL) ||
            (cpuid_info.model == HASWELL_M1) ||
            (cpuid_info.model == HASWELL_M2) ||
            (cpuid_info.model == HASWELL_EP) ||
            (cpuid_info.model == BROADWELL) ||
            (cpuid_info.model == BROADWELL_E3) ||
            (cpuid_info.model == BROADWELL_D) ||
            (cpuid_info.model == BROADWELL_E) ||
            (cpuid_info.model == SKYLAKE1) ||
            (cpuid_info.model == SKYLAKE2) ||
            (cpuid_info.model == SKYLAKEX) ||
            (cpuid_info.model == KABYLAKE1) ||
            (cpuid_info.model == KABYLAKE2) ||
            (cpuid_info.model == CANNONLAKE) ||
            (cpuid_info.model == COMETLAKE1) ||
            (cpuid_info.model == COMETLAKE2) ||
            (cpuid_info.model == ICELAKE1) ||
            (cpuid_info.model == ICELAKE2) ||
            (cpuid_info.model == SAPPHIRERAPIDS) ||
            (cpuid_info.model == GRANITERAPIDS) ||
            (cpuid_info.model == SIERRAFORREST) ||
            (cpuid_info.model == ROCKETLAKE) ||
            (cpuid_info.model == ICELAKEX1) ||
            (cpuid_info.model == ICELAKEX2) ||
            (cpuid_info.model == ATOM_SILVERMONT_GOLD))
    {
        reg = MSR_PREFETCH_ENABLE;
        newOffsets = 1;
    }
    if ((cpuid_info.model == XEON_PHI_KNL) ||
        (cpuid_info.model == XEON_PHI_KML))
    {
        reg = MSR_PREFETCH_ENABLE;
        knlOffsets = 1;
        if (type == FEAT_CL_PREFETCHER ||
            type == FEAT_IP_PREFETCHER)
        {
            fprintf(stderr, "CL_PREFETCHER and IP_PREFETCHER not available on Intel Xeon Phi (KNL)");
            return 0;
        }
    }

    ret = HPMread(cpu, MSR_DEV, reg, &flags);
    if (ret != 0)
    {
        fprintf(stderr, "Cannot read register 0x%X for CPU %d to activate feature %s\n", reg, cpu, cpuFeatureNames[type]);
        return ret;
    }
    ret = 0;
    switch ( type )
    {
        case FEAT_HW_PREFETCHER:
            if (print)
                printf("HW_PREFETCHER:\t");
            if (newOffsets)
            {
                flags &= ~(1ULL<<0);
            }
            else if (knlOffsets)
            {
                flags &= ~(1ULL<<1);
            }
            else
            {
                flags &= ~(1ULL<<9);
            }
            break;

        case FEAT_CL_PREFETCHER:
            if (print)
                printf("CL_PREFETCHER:\t");
            if (newOffsets)
            {
                flags &= ~(1ULL<<1);
            }
            else
            {
                flags &= ~(1ULL<<19);
            }
            break;

        case FEAT_DCU_PREFETCHER:
            if (print)
                printf("DCU_PREFETCHER:\t");
            if (newOffsets)
            {
                flags &= ~(1ULL<<2);
            }
            else if (knlOffsets)
            {
                flags &= ~(1ULL<<0);
            }
            else
            {
                flags &= ~(1ULL<<37);
            }
            break;

        case FEAT_IP_PREFETCHER:
            if (print)
                printf("IP_PREFETCHER:\t");
            if (newOffsets)
            {
                flags &= ~(1ULL<<3);
            }
            else
            {
                flags &= ~(1ULL<<39);
            }
            break;

        default:
            printf("\nERROR: Processor feature '%s' cannot be enabled!\n", cpuFeatureNames[type]);
            ret = -EINVAL;
            break;
    }
    if (ret != 0)
    {
        return ret;
    }

    ret = HPMwrite(cpu, MSR_DEV, reg, flags);
    if (ret == 0)
    {
        if (print)
        {
            PRINT_VALUE(GREEN,enabled);
        }
    }
    else
    {
        if (print)
        {
            PRINT_VALUE(RED,failed);
        }
    }
    cpuFeatures_update(cpu);
    return 0;
}

int
cpuFeatures_disable(int cpu, CpuFeature type, int print)
{
    int ret;
    uint64_t flags;
    uint32_t reg = MSR_IA32_MISC_ENABLE;
    int newOffsets = 0;
    int knlOffsets = 1;
    if (!features_initialized)
    {
        return -1;
    }
    if (!IF_FLAG(type))
    {
        return 0;
    }
    if ((cpuid_info.model == NEHALEM) ||
            (cpuid_info.model == NEHALEM_BLOOMFIELD) ||
            (cpuid_info.model == NEHALEM_LYNNFIELD) ||
            (cpuid_info.model == NEHALEM_WESTMERE) ||
            (cpuid_info.model == NEHALEM_WESTMERE_M) ||
            (cpuid_info.model == NEHALEM_EX) ||
            (cpuid_info.model == WESTMERE_EX) ||
            (cpuid_info.model == SANDYBRIDGE) ||
            (cpuid_info.model == SANDYBRIDGE_EP) ||
            (cpuid_info.model == IVYBRIDGE) ||
            (cpuid_info.model == IVYBRIDGE_EP) ||
            (cpuid_info.model == HASWELL) ||
            (cpuid_info.model == HASWELL_M1) ||
            (cpuid_info.model == HASWELL_M2) ||
            (cpuid_info.model == HASWELL_EP) ||
            (cpuid_info.model == BROADWELL) ||
            (cpuid_info.model == BROADWELL_E3) ||
            (cpuid_info.model == BROADWELL_D) ||
            (cpuid_info.model == BROADWELL_E) ||
            (cpuid_info.model == SKYLAKE1) ||
            (cpuid_info.model == SKYLAKE2) ||
            (cpuid_info.model == SKYLAKEX) ||
            (cpuid_info.model == KABYLAKE1) ||
            (cpuid_info.model == KABYLAKE2) ||
            (cpuid_info.model == CANNONLAKE) ||
            (cpuid_info.model == COMETLAKE1) ||
            (cpuid_info.model == COMETLAKE2) ||
            (cpuid_info.model == ICELAKE1) ||
            (cpuid_info.model == ICELAKE2) ||
            (cpuid_info.model == SAPPHIRERAPIDS) ||
            (cpuid_info.model == GRANITERAPIDS) ||
            (cpuid_info.model == SIERRAFORREST) ||
            (cpuid_info.model == ROCKETLAKE) ||
            (cpuid_info.model == ICELAKEX1) ||
            (cpuid_info.model == ICELAKEX2) ||
            (cpuid_info.model == ATOM_SILVERMONT_GOLD))
    {
        reg = MSR_PREFETCH_ENABLE;
        newOffsets = 1;
    }
    if ((cpuid_info.model == XEON_PHI_KNL) ||
        (cpuid_info.model == XEON_PHI_KML))
    {
        reg = MSR_PREFETCH_ENABLE;
        knlOffsets = 1;
        if (type == FEAT_CL_PREFETCHER ||
            type == FEAT_IP_PREFETCHER)
        {
            fprintf(stderr, "CL_PREFETCHER and IP_PREFETCHER not available on Intel Xeon Phi (KNL)");
            return 0;
        }
    }
    ret = HPMread(cpu, MSR_DEV, reg, &flags);
    if (ret != 0)
    {
        fprintf(stderr, "Reading register 0x%X on CPU %d failed\n", reg, cpu);
        return ret;
    }
    ret = 0;
    switch ( type )
    {
        case FEAT_HW_PREFETCHER:
            if (print)
                printf("HW_PREFETCHER:\t");
            if (newOffsets)
            {
                flags |= (1ULL<<0);
            }
            else if (knlOffsets)
            {
                flags |= (1ULL<<1);
            }
            else
            {
                flags |= (1ULL<<9);
            }
            break;

        case FEAT_CL_PREFETCHER:
            if (print)
                printf("CL_PREFETCHER:\t");
            if (newOffsets)
            {
                flags |= (1ULL<<1);
            }
            else
            {
                flags |= (1ULL<<19);
            }
            break;

        case FEAT_DCU_PREFETCHER:
            if (print)
                printf("DCU_PREFETCHER:\t");
            if (newOffsets)
            {
                flags |= (1ULL<<2);
            }
            else if (knlOffsets)
            {
                flags |= (1ULL<<0);
            }
            else
            {
                flags |= (1ULL<<37);
            }
            break;

        case FEAT_IP_PREFETCHER:
            if (print)
                printf("IP_PREFETCHER:\t");
            if (newOffsets)
            {
                flags |= (1ULL<<3);
            }
            else
            {
                flags |= (1ULL<<39);
            }
            break;

        default:
            printf("ERROR: Processor feature '%s' cannot be disabled!\n", cpuFeatureNames[type]);
            ret = -EINVAL;
            break;
    }
    if (ret != 0)
    {
        return ret;
    }

    ret = HPMwrite(cpu, MSR_DEV, reg, flags);
    if (ret != 0)
    {
        if (print)
        {
            PRINT_VALUE(RED,failed);
        }
        ret = -EFAULT;
    }
    else
    {
        if (print)
        {
            PRINT_VALUE(RED,disabled);
        }
        ret = 0;
    }
    cpuFeatures_update(cpu);
    return ret;
}

int
cpuFeatures_get(int cpu, CpuFeature type)
{
    if (!features_initialized)
    {
        return -EINVAL;
    }
    if ((type >= FEAT_HW_PREFETCHER) && (type < CPUFEATURES_MAX))
    {
        if (IF_FLAG(type))
        {
            return TRUE;
        }
        else
        {
            return FALSE;
        }
    }
    return -EINVAL;
}

char*
cpuFeatures_name(CpuFeature type)
{
    if ((type >= FEAT_HW_PREFETCHER) && (type < CPUFEATURES_MAX))
    {
        return cpuFeatureNames[type];
    }
    return NULL;
}


void __attribute__((destructor (104))) cpuFeatures_finalizeDestruct(void)
{
    if (cpuFeatureMask)
    {
        free(cpuFeatureMask);
        cpuFeatureMask = NULL;
    }
}
