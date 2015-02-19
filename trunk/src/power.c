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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <types.h>
#include <power.h>
#include <topology.h>

/* #####   EXPORTED VARIABLES   ########################################### */

PowerInfo power_info;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */



/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
power_init(int cpuId)
{
    uint64_t flags;
    uint32_t info_register = MSR_PKG_POWER_INFO;
    int i;
    int err;

    /* determine Turbo Mode features */
    double busSpeed;

    power_info.baseFrequency = 0;
    power_info.minFrequency = 0;
    power_info.turbo.numSteps = 0;
    power_info.powerUnit = 0;
    power_info.energyUnits = NULL;
    power_info.timeUnit = 0;
    power_info.minPower = 0;
    power_info.maxPower = 0;
    power_info.maxTimeWindow = 0;
    power_info.tdp = 0;
    power_info.hasRAPL = 0;
    power_info.supportedTypes = 0x0U;

    if ( power_info.hasRAPL )
    {
        busSpeed = 100.0;
    }
    else
    {
        busSpeed = 133.33;
    }

    if (!HPMinitialized())
    {
        HPMaddThread(cpuId);
    }

    if (cpuid_info.turbo)
    {
        err = HPMread(cpuId, MSR_DEV, MSR_PLATFORM_INFO, &flags);
        if (err == 0)
        {
            power_info.baseFrequency = busSpeed * (double) extractBitField(flags,8,8);
            power_info.minFrequency  = busSpeed * (double) extractBitField((flags>>(32)),8,8);

            power_info.turbo.numSteps = cpuid_topology.numCoresPerSocket;
            if (cpuid_info.model == WESTMERE_EX)
            {
                power_info.turbo.numSteps = 4;
            }
            power_info.turbo.steps = (double*) malloc(power_info.turbo.numSteps * sizeof(double));
            if (!power_info.turbo.steps)
            {
                return -ENOMEM;
            }

            err = HPMread(cpuId, MSR_DEV, MSR_TURBO_RATIO_LIMIT, &flags);
            if (err)
            {
                fprintf(stderr,"Cannot gather values from MSR_TURBO_RATIO_LIMIT,\n");
            }
            else
            {
                for (int i=0; i < power_info.turbo.numSteps; i++)
                {
                    if (i < 8)
                    {
                        power_info.turbo.steps[i] = busSpeed * (double) field64(flags,i*8, 8);
                    }
                    else
                    {
                        power_info.turbo.steps[i] = power_info.turbo.steps[7];
                    }
                }
            }
        }
        else
        {
            fprintf(stderr,"Cannot gather values from MSR_PLATFORM_INFO,\n");
        }
    }

    switch (cpuid_info.model)
    {
        case SANDYBRIDGE:
        case IVYBRIDGE:
        case HASWELL:
        case SANDYBRIDGE_EP:
        case IVYBRIDGE_EP:
        case HASWELL_EP:
        case ATOM_SILVERMONT_E:
        case ATOM_SILVERMONT_Z1:
        case ATOM_SILVERMONT_Z2:
        case ATOM_SILVERMONT_F:
            power_info.hasRAPL = 1;
            break;
        case ATOM_SILVERMONT_C:
            power_info.hasRAPL = 1;
            info_register = MSR_PKG_POWER_INFO_SILVERMONT;
            break;
        default:
            DEBUG_PLAIN_PRINT(DEBUGLEV_INFO, NO RAPL SUPPORT);
            return 0;
            break;
    }

    /* determine RAPL parameters */
    if ( power_info.hasRAPL )
    {
        power_info.energyUnits = (double*) malloc(NUM_POWER_DOMAINS * sizeof(double));
        if (!power_info.energyUnits)
        {
            return -ENOMEM;
        }
        err = HPMread(cpuId, MSR_DEV, MSR_RAPL_POWER_UNIT, &flags);
        if (err == 0)
        {
            double energyUnit;
            power_info.powerUnit = pow(0.5,(double) extractBitField(flags,4,0));
            energyUnit = pow(0.5,(double) extractBitField(flags,5,8));
            power_info.timeUnit = pow(0.5,(double) extractBitField(flags,4,16));
            for (i = 0; i < NUM_POWER_DOMAINS; i++)
            {
                power_info.energyUnits[i] = energyUnit;
            }
            if ((cpuid_info.model == HASWELL_EP) ||
                (cpuid_info.model == HASWELL_M1) ||
                (cpuid_info.model == HASWELL_M2))
            {
                power_info.energyUnits[3] = 15.3E-6;
            }

            /* info_register set in the switch-case-statement at the beginning
               because Atom Silvermont C uses another register */
            err = HPMread(cpuId, MSR_DEV, info_register, &flags);
            if (err == 0)
            {
                power_info.tdp = (double) extractBitField(flags,15,0) * power_info.powerUnit;
                if (cpuid_info.model != ATOM_SILVERMONT_C)
                {
                    power_info.minPower =  (double) extractBitField(flags,15,16) * power_info.powerUnit;
                    power_info.maxPower = (double) extractBitField(flags,15,32) * power_info.powerUnit;
                    power_info.maxTimeWindow = (double) extractBitField(flags,7,48) * power_info.timeUnit;
                }
            }
            for(i = 0; i < 4; i++)
            {
                err = HPMread(cpuId, MSR_DEV, power_regs[i], &flags);
                if (err == 0)
                {
                    power_info.supportedTypes |= (1<<i);
                }
                else
                {
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, RAPL domain %s not supported, power_names[i]);
                }
            }
        }
        else
        {
            fprintf(stderr,"Cannot gather values from MSR_RAPL_POWER_UNIT, deactivating RAPL support\n");
            power_info.hasRAPL =  0;
        }
        return power_info.hasRAPL;
    }
    else
    {
        return power_info.hasRAPL;
    }
    return 0;
}

void power_finalize(void)
{
    if (power_info.energyUnits)
    {
        free(power_info.energyUnits);
    }
}

PowerInfo_t get_powerInfo(void)
{
    return &power_info;
}
