/*
 * =======================================================================================
 *
 *      Filename:  power.c
 *
 *      Description:  Module implementing Intel RAPL interface
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2012 Jan Treibig 
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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <types.h>
#include <power.h>
#include <cpuid.h>

/* #####   EXPORTED VARIABLES   ########################################### */

PowerInfo power_info;
const uint32_t power_regs[4] = {MSR_PKG_ENERGY_STATUS,
                                MSR_PP0_ENERGY_STATUS,
                                MSR_PP1_ENERGY_STATUS,
                                MSR_DRAM_ENERGY_STATUS};

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */



/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
power_init(int cpuId)
{
    uint64_t flags;
    uint32_t tmp;
    int hasRAPL = 0;

    /* determine Turbo Mode features */
    double busSpeed;

    if ((cpuid_info.model == SANDYBRIDGE_EP) ||
            (cpuid_info.model == SANDYBRIDGE) ||
            (cpuid_info.model == IVYBRIDGE))
    {
        hasRAPL = 1;
    }

    if (cpuid_info.turbo)
    {
        flags = msr_read(cpuId, MSR_PLATFORM_INFO);

        if ( hasRAPL )
        {
            busSpeed = 100.0;
        }
        else 
        {
            busSpeed = 133.33;
        }

        power_info.baseFrequency = busSpeed * (double) extractBitField(flags,8,8);
        power_info.minFrequency  = busSpeed * (double) extractBitField((flags>>(32)),8,8);

        power_info.turbo.numSteps = cpuid_topology.numCoresPerSocket;
        power_info.turbo.steps = (double*) malloc(power_info.turbo.numSteps * sizeof(double));

        flags = msr_read(cpuId, MSR_TURBO_RATIO_LIMIT);

        for (int i=0; i < power_info.turbo.numSteps; i++)
        {
            if ( i < 4 )
            {
                power_info.turbo.steps[i] = busSpeed * (double) extractBitField(flags,8,i*8);
            }
            else
            {
                power_info.turbo.steps[i] = busSpeed * (double) extractBitField(flags>>(32),8,i*8);
            }
        }
    }
    else
    {
        power_info.turbo.numSteps = 0;
    }

    /* determine RAPL parameters */
    if ( hasRAPL )
    {
        flags = msr_read(cpuId, MSR_RAPL_POWER_UNIT);

        power_info.powerUnit = pow(0.5,(double) extractBitField(flags,4,0));
        power_info.energyUnit = pow(0.5,(double) extractBitField(flags,5,8));
        power_info.timeUnit = pow(0.5,(double) extractBitField(flags,4,16));

        flags = msr_read(cpuId, MSR_PKG_POWER_INFO);
        power_info.tdp = (double) extractBitField(flags,15,0) * power_info.powerUnit;
        power_info.minPower =  (double) extractBitField(flags,15,16) * power_info.powerUnit;
        power_info.maxPower = (double) extractBitField(flags,15,32) * power_info.powerUnit;
        power_info.maxTimeWindow = (double) extractBitField(flags,15,48) * power_info.timeUnit;
    }
    else
    {
        power_info.powerUnit = 0.0;
    }
}

