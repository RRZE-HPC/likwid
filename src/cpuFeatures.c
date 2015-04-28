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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <types.h>
#include <access.h>
#include <topology.h>
#include <registers.h>
#include <textcolor.h>
#include <cpuFeatures.h>

/* #####   EXPORTED VARIABLES   ########################################### */

CpuFeatureFlags cpuFeatureFlags;

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */


#define PRINT_VALUE(color,string)  \
    color_on(BRIGHT,(color));      \
    printf(#string"\n");            \
    color_reset()

#define TEST_FLAG(feature,flag)  \
    if (flags & (1ULL<<(flag)))   \
    {                    \
        cpuFeatureFlags.feature = 1; \
    }                    \
    else                \
    {                \
        cpuFeatureFlags.feature = 0; \
    }


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
cpuFeatures_init(int cpu)
{
    int ret;
    uint64_t flags;
    ret = HPMread(cpu, MSR_DEV, MSR_IA32_MISC_ENABLE, &flags);

    TEST_FLAG(fastStrings,0);
    TEST_FLAG(thermalControl,3);
    TEST_FLAG(perfMonitoring,7);
    TEST_FLAG(branchTraceStorage,11);
    TEST_FLAG(pebs,12);
    TEST_FLAG(speedstep,16);
    TEST_FLAG(monitor,18);
    TEST_FLAG(cpuidMaxVal,22);
    TEST_FLAG(xdBit,34);

    if ((cpuid_info.model == NEHALEM) ||
            (cpuid_info.model == NEHALEM_BLOOMFIELD) ||
            (cpuid_info.model == NEHALEM_LYNNFIELD) ||
            (cpuid_info.model == NEHALEM_WESTMERE) ||
            (cpuid_info.model == NEHALEM_WESTMERE_M) ||
            (cpuid_info.model == NEHALEM_EX))
    {
        /*Nehalem */
        TEST_FLAG(turboMode,38);
        TEST_FLAG(hardwarePrefetcher,9);
        TEST_FLAG(clPrefetcher,19);
        TEST_FLAG(dcuPrefetcher,37);
        TEST_FLAG(ipPrefetcher,39);
    }
    else if ((cpuid_info.model == CORE2_45) ||
            (cpuid_info.model == CORE2_65))
    {
        /*Core 2*/
        TEST_FLAG(hardwarePrefetcher,9);
        TEST_FLAG(ferrMultiplex,10);
        TEST_FLAG(clPrefetcher,19);
        TEST_FLAG(speedstepLock,20);
        TEST_FLAG(dcuPrefetcher,37);
        TEST_FLAG(dynamicAcceleration,38);
        TEST_FLAG(ipPrefetcher,39);
    }

    /*
    printf("FLAGS: 0x%llX \n",flags);
    */
}

void
cpuFeatures_print(int cpu)
{
    int ret;
    uint64_t flags;
    ret = HPMread(cpu, MSR_DEV, MSR_IA32_MISC_ENABLE, &flags);

    printf(HLINE);
    printf("Fast-Strings: \t\t\t");
    if (flags & 1)
    {
        PRINT_VALUE(GREEN,enabled);
    }
    else
    {
        PRINT_VALUE(RED,disabled);
    }

    printf("Automatic Thermal Control: \t");
    if (flags & (1ULL<<3))
    {
        PRINT_VALUE(GREEN,enabled);
    }
    else
    {
        PRINT_VALUE(RED,disabled);
    }

    printf("Performance monitoring: \t");
    if (flags & (1ULL<<7))
    {
        PRINT_VALUE(GREEN,enabled);
    }
    else
    {
        PRINT_VALUE(RED,disabled);
    }
    printf("Branch Trace Storage: \t\t");

    if (flags & (1ULL<<11)) 
    {
        PRINT_VALUE(RED,notsupported);
    }
    else
    {
        PRINT_VALUE(GREEN,supported);
    }

    printf("PEBS: \t\t\t\t");
    if (flags & (1ULL<<12)) 
    {
        PRINT_VALUE(RED,notsupported);
    }
    else
    {
        PRINT_VALUE(GREEN,supported);
    }

    printf("Intel Enhanced SpeedStep: \t");
    if (flags & (1ULL<<16)) 
    {
        PRINT_VALUE(GREEN,enabled);
    }
    else
    {
        PRINT_VALUE(RED,disabled);
    }

    printf("MONITOR/MWAIT: \t\t\t");
    if (flags & (1ULL<<18)) 
    {
        PRINT_VALUE(GREEN,supported);
    }
    else
    {
        PRINT_VALUE(RED,notsupported);
    }

    printf("Limit CPUID Maxval: \t\t");
    if (flags & (1ULL<<22)) 
    {
        PRINT_VALUE(RED,enabled);
    }
    else
    {
        PRINT_VALUE(GREEN,disabled);
    }

    printf("XD Bit Disable: \t\t");
    if (flags & (1ULL<<34)) 
    {
        PRINT_VALUE(RED,disabled);
    }
    else
    {
        PRINT_VALUE(GREEN,enabled);
    }

    printf("IP Prefetcher: \t\t\t");
    if (flags & (1ULL<<39)) 
    {
        PRINT_VALUE(RED,disabled);
    }
    else
    {
        PRINT_VALUE(GREEN,enabled);
    }

    printf("Hardware Prefetcher: \t\t");
    if (flags & (1ULL<<9)) 
    {
        PRINT_VALUE(RED,disabled);
    }
    else
    {
        PRINT_VALUE(GREEN,enabled);
    }

    printf("Adjacent Cache Line Prefetch: \t");
    if (flags & (1ULL<<19)) 
    {
        PRINT_VALUE(RED,disabled);
    }
    else
    {
        PRINT_VALUE(GREEN,enabled);
    }

    printf("DCU Prefetcher: \t\t");
    if (flags & (1ULL<<37)) 
    {
        PRINT_VALUE(RED,disabled);
    }
    else
    {
        PRINT_VALUE(GREEN,enabled);
    }

    if ((cpuid_info.model == NEHALEM) ||
            (cpuid_info.model == NEHALEM_BLOOMFIELD) ||
            (cpuid_info.model == NEHALEM_LYNNFIELD) ||
            (cpuid_info.model == NEHALEM_WESTMERE) ||
            (cpuid_info.model == NEHALEM_WESTMERE_M) ||
            (cpuid_info.model == NEHALEM_EX))
    {
        printf("Intel Turbo Mode: \t");
        if (flags & (1ULL<<38)) 
        {
            PRINT_VALUE(RED,disabled);
        }
        else 
        {
            PRINT_VALUE(GREEN,enabled);
        }
    }
    else if ((cpuid_info.model == CORE2_45) ||
            (cpuid_info.model == CORE2_65))
    {

        printf("Intel Dynamic Acceleration: \t");
        if (flags & (1ULL<<38)) 
        {
            PRINT_VALUE(RED,disabled);
        }
        else 
        {
            PRINT_VALUE(GREEN,enabled);
        }
    }

    printf(HLINE);
}

void 
cpuFeatures_enable(int cpu, CpuFeature type)
{
    int ret;
    uint64_t flags; 
    ret = HPMread(cpu, MSR_DEV, MSR_IA32_MISC_ENABLE, &flags);

    switch ( type )
    {
        case HW_PREFETCHER:
            printf("HW_PREFETCHER:\t");
            flags &= ~(1ULL<<9);
            break;

        case CL_PREFETCHER:
            printf("CL_PREFETCHER:\t");
            flags &= ~(1ULL<<19);
            break;

        case DCU_PREFETCHER:
            printf("DCU_PREFETCHER:\t");
            flags &= ~(1ULL<<37);
            break;

        case IP_PREFETCHER:
            printf("IP_PREFETCHER:\t");
            flags &= ~(1ULL<<39);
            break;

        default:
            printf("ERROR: CpuFeature not supported!\n");
            break;
    }
    PRINT_VALUE(GREEN,enabled);
    printf("\n");

    HPMwrite(cpu, MSR_DEV, MSR_IA32_MISC_ENABLE, flags);
}


void
cpuFeatures_disable(int cpu, CpuFeature type)
{
    int ret;
    uint64_t flags;
    ret = HPMread(cpu, MSR_DEV, MSR_IA32_MISC_ENABLE, &flags);

    switch ( type ) 
    {
        case HW_PREFETCHER:
            printf("HW_PREFETCHER:\t");
            flags |= (1ULL<<9);
            break;

        case CL_PREFETCHER:
            printf("CL_PREFETCHER:\t");
            flags |= (1ULL<<19);
            break;

        case DCU_PREFETCHER:
            printf("DCU_PREFETCHER:\t");
            flags |= (1ULL<<37);
            break;

        case IP_PREFETCHER:
            printf("IP_PREFETCHER:\t");
            flags |= (1ULL<<39);
            break;

        default:
            printf("ERROR: CpuFeature not supported!\n");
            break;
    }
    PRINT_VALUE(RED,disabled);
    printf("\n");

    HPMwrite(cpu, MSR_DEV, MSR_IA32_MISC_ENABLE, flags);
}

