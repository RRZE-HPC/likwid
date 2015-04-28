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
 *      Authors:  Jan Treibig (jt), jan.treibig@gmail.com,
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 Jan Treibig, Thomas Roehl
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
static int power_initialized = 0;


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
power_init(int cpuId)
{
    uint64_t flags;
    int i;
    int err;

    /* determine Turbo Mode features */
    double busSpeed;

    power_info.baseFrequency = 0;
    power_info.minFrequency = 0;
    power_info.turbo.numSteps = 0;
    power_info.powerUnit = 0;
    power_info.timeUnit = 0;
    power_info.hasRAPL = 0;

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
        case BROADWELL:
        case BROADWELL_E:
        case BROADWELL_D:
            power_info.hasRAPL = 1;
            break;
        case ATOM_SILVERMONT_C:
            power_info.hasRAPL = 1;
            /* The info_regs list needs an update for Silvermont Type C
               because it uses another info register */
            info_regs[PKG] = MSR_PKG_POWER_INFO_SILVERMONT;
            break;
        default:
            DEBUG_PLAIN_PRINT(DEBUGLEV_INFO, NO RAPL SUPPORT);
            return 0;
            break;
    }

    if ( power_info.hasRAPL )
    {
        busSpeed = 100.0;
    }
    else
    {
        busSpeed = 133.33;
    }
    perfmon_init_maps();
    if (!HPMinitialized())
    {
        HPMaddThread(cpuId);
    }
    if (power_initialized)
    {
        return 0;
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
            //TODO: Haswell EP and possibly Broadwell EP support multiple turbo 
            //      registers besides MSR_TURBO_RATIO_LIMIT:
            //      MSR_TURBO_RATIO_LIMIT1 and MSR_TURBO_RATIO_LIMIT2
        }
        else
        {
            fprintf(stderr,"Cannot gather values from MSR_PLATFORM_INFO,\n");
        }
    }

    /* determine RAPL parameters */
    if ( power_info.hasRAPL )
    {
        err = HPMread(cpuId, MSR_DEV, MSR_RAPL_POWER_UNIT, &flags);
        if (err == 0)
        {
            double energyUnit;
            power_info.powerUnit = pow(0.5,(double) extractBitField(flags,4,0));
            if (cpuid_info.model != ATOM_SILVERMONT_E)
            {
                energyUnit = 1.0 / (1 << ((flags >> 8) & 0x1F));
            }
            else
            {
                energyUnit = 1.0 * (1 << ((flags >> 8) & 0x1F)) / 1000000;
            }
            power_info.timeUnit = pow(0.5,(double) extractBitField(flags,4,16));
            for (i = 0; i < NUM_POWER_DOMAINS; i++)
            {
                power_info.domains[i].energyUnit = energyUnit;
                power_info.domains[i].type = i;
                power_info.domains[i].supportFlags = 0x0U;
                power_info.domains[i].tdp = 0.0;
                power_info.domains[i].minPower = 0.0;
                power_info.domains[i].maxPower = 0.0;
                power_info.domains[i].maxTimeWindow = 0.0;
            }
            
            if ((cpuid_info.model == HASWELL_EP) ||
                (cpuid_info.model == HASWELL_M1) ||
                (cpuid_info.model == HASWELL_M2))
            {
                power_info.domains[DRAM].energyUnit = 15.3E-6;
            }

            for(i = 0; i < NUM_POWER_DOMAINS; i++)
            {
                err = HPMread(cpuId, MSR_DEV, power_regs[i], &flags);
                if (err == 0)
                {
                    power_info.domains[i].supportFlags |= POWER_DOMAIN_SUPPORT_STATUS;
                }
                else
                {
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, RAPL domain %s not supported, power_names[i]);
                    continue;
                }
                if (limit_regs[i] != 0x0)
                {
                    err = HPMread(cpuId, MSR_DEV, limit_regs[i], &flags);
                    if (err == 0)
                    {
                        power_info.domains[i].supportFlags |= POWER_DOMAIN_SUPPORT_LIMIT;
                    }
                    else
                    {
                        DEBUG_PRINT(DEBUGLEV_INFO, Deactivating limit register for RAPL domain %s, power_names[i]);
                        limit_regs[i] = 0x0;
                    }
                }
                if (info_regs[i] != 0x0)
                {
                    err = HPMread(cpuId, MSR_DEV, info_regs[i], &flags);
                    if (err == 0)
                    {
                        power_info.domains[i].supportFlags |= POWER_DOMAIN_SUPPORT_INFO;
                        power_info.domains[i].tdp = (double) extractBitField(flags,15,0) * power_info.powerUnit;
                        if (cpuid_info.model != ATOM_SILVERMONT_C)
                        {
                            power_info.domains[i].minPower = (double) extractBitField(flags,15,16) * power_info.powerUnit;
                            power_info.domains[i].maxPower = (double) extractBitField(flags,15,32) * power_info.powerUnit;
                            power_info.domains[i].maxTimeWindow = (double) extractBitField(flags,7,48) * power_info.timeUnit;
                        }
                    }
                    /*else
                    {
                        DEBUG_PRINT(DEBUGLEV_INFO, Deactivating info register for RAPL domain %s, power_names[i]);
                        info_regs[i] = 0x0;
                    }*/
                }
                if (policy_regs[i] != 0x0)
                {
                    err = HPMread(cpuId, MSR_DEV, policy_regs[i], &flags);
                    if (err == 0)
                    {
                        power_info.domains[i].supportFlags |= POWER_DOMAIN_SUPPORT_POLICY;
                    }
                    /*else
                    {
                        DEBUG_PRINT(DEBUGLEV_INFO, Deactivating policy register for RAPL domain %s, power_names[i]);
                        policy_regs[i] = 0x0;
                    }*/
                }
                if (perf_regs[i] != 0x0)
                {
                    err = HPMread(cpuId, MSR_DEV, perf_regs[i], &flags);
                    if (err == 0)
                    {
                        power_info.domains[i].supportFlags |= POWER_DOMAIN_SUPPORT_PERF;
                    }
                    /*else
                    {
                        DEBUG_PRINT(DEBUGLEV_INFO, Deactivating perf register for RAPL domain %s, power_names[i]);
                        perf_regs[i] = 0x0;
                    }*/
                }
            }
        }
        else
        {
            fprintf(stderr,"Cannot gather values from MSR_RAPL_POWER_UNIT, deactivating RAPL support\n");
            power_info.hasRAPL =  0;
        }
        power_initialized = 1;
        return power_info.hasRAPL;
    }
    else
    {
        return power_info.hasRAPL;
    }
    return 0;
}


int power_perfGet(int cpuId, PowerType domain, uint32_t* status)
{
    int err = 0;
    *status = 0x0U;
    if (domain >= NUM_POWER_DOMAINS)
    {
        return -EINVAL;
    }
    if (power_info.domains[domain].supportFlags & POWER_DOMAIN_SUPPORT_PERF)
    {
        err = HPMread(cpuId, MSR_DEV, perf_regs[domain], (uint64_t*)status);
        if (err)
        {
            ERROR_PRINT(Failed to get power perf value for domain %s on CPU %d,power_names[domain], cpuId);
            return -EFAULT;
        }
    }
    return 0;
}

int power_limitSet(int cpuId, PowerType domain, double power, double time, int doClamping)
{
    int err = 0;
    if (domain >= NUM_POWER_DOMAINS)
    {
        return -EINVAL;
    }
    fprintf(stderr, "Not implemented\n");
    return 0;

    uint32_t X = (log(time) - log(power_info.timeUnit))/log(2);
    uint32_t powerField = (uint32_t)(power/(power_info.domains[domain].energyUnit));
    uint64_t flags = (powerField & 0xFFFF)|((X & (0x1F))<<17);
    // Construct flags missing. How is timeField calculated?
    if (doClamping)
    {
        flags |= (1ULL<<16);
    }
    if (power_info.domains[domain].supportFlags & POWER_DOMAIN_SUPPORT_LIMIT)
    {
        err = HPMwrite(cpuId, MSR_DEV, limit_regs[domain], flags);
        if (err)
        {
            fprintf(stderr, "Failed to set power limit for domain %s on CPU %d\n",power_names[domain], cpuId);
            return -EFAULT;
        }
    }
    return 0;
}

int power_limitGet(int cpuId, PowerType domain, double* power, double* time)
{
    int err = 0;
    *power = 0;
    *time = 0;
    unsigned int Y,Z;
    if (domain >= NUM_POWER_DOMAINS)
    {
        return -EINVAL;
    }
    uint64_t flags = 0x0ULL;
    if (power_info.domains[domain].supportFlags & POWER_DOMAIN_SUPPORT_LIMIT)
    {
        err = HPMread(cpuId, MSR_DEV, limit_regs[domain], &flags);
        if (err)
        {
            fprintf(stderr, "Failed to set power limit for domain %s on CPU %d\n",power_names[domain], cpuId);
            return -EFAULT;
        }
        *power = ((double)extractBitField(flags, 15, 0)) * power_info.domains[domain].energyUnit;
        Y = extractBitField(flags, 5, 17);
        Z = extractBitField(flags, 2, 22);
        *time = pow(2,((double)Y)) * (1.0 + (((double)Z)/4.0)) * power_info.timeUnit;
    }
    return 0;
}

int power_limitState(int cpuId, PowerType domain)
{
    int err = 0;
    if (domain >= NUM_POWER_DOMAINS)
    {
        return -EINVAL;
    }
    uint64_t flags = 0x0ULL;

    if (power_info.domains[domain].supportFlags & POWER_DOMAIN_SUPPORT_LIMIT)
    {
        err = HPMread(cpuId, MSR_DEV, limit_regs[domain], &flags);
        if (err)
        {
            ERROR_PRINT(Failed to activate power limit for domain %s on CPU %d,power_names[domain], cpuId);
            return -EFAULT;
        }
    }
    if (flags & (1ULL<<15))
    {
        return 1;
    }
    return 0;
}

int power_limitActivate(int cpuId, PowerType domain)
{
    int err = 0;
    if (domain >= NUM_POWER_DOMAINS)
    {
        return -EINVAL;
    }
    uint64_t flags = 0x0ULL;

    if (power_info.domains[domain].supportFlags & POWER_DOMAIN_SUPPORT_LIMIT)
    {
        err = HPMread(cpuId, MSR_DEV, limit_regs[domain], &flags);
        if (err)
        {
            ERROR_PRINT(Failed to activate power limit for domain %s on CPU %d,power_names[domain], cpuId);
            return -EFAULT;
        }
        flags |= (1ULL<<15);
        err = HPMwrite(cpuId, MSR_DEV, limit_regs[domain], flags);
        if (err)
        {
            ERROR_PRINT(Failed to activate power limit for domain %s on CPU %d,power_names[domain], cpuId);
            return -EFAULT;
        }
    }
    return 0;
}

int power_limitDectivate(int cpuId, PowerType domain)
{
    int err = 0;
    uint64_t flags = 0x0ULL;

    if (power_info.domains[domain].supportFlags & POWER_DOMAIN_SUPPORT_LIMIT)
    {
        err = HPMread(cpuId, MSR_DEV, limit_regs[domain], &flags);
        if (err)
        {
            ERROR_PRINT(Failed to deactivate power limit for domain %s on CPU %d,power_names[domain], cpuId);
            return -EFAULT;
        }
        flags &= ~(1ULL<<15);
        err = HPMwrite(cpuId, MSR_DEV, limit_regs[domain], flags);
        if (err)
        {
            ERROR_PRINT(Failed to deactivate power limit for domain %s on CPU %d,power_names[domain], cpuId);
            return -EFAULT;
        }
    }
    return 0;
}

int power_policySet(int cpuId, PowerType domain, uint32_t priority)
{
    int err = 0;
    if (domain >= NUM_POWER_DOMAINS)
    {
        return -EINVAL;
    }
    priority = extractBitField(priority, 5, 0);
    if (power_info.domains[domain].supportFlags & POWER_DOMAIN_SUPPORT_POLICY)
    {
        err = HPMwrite(cpuId, MSR_DEV, policy_regs[domain], priority);
        if (err)
        {
            ERROR_PRINT(Failed to set power policy for domain %s on CPU %d,power_names[domain], cpuId);
            return -EFAULT;
        }
    }
    return 0;
}

int power_policyGet(int cpuId, PowerType domain, uint32_t* priority)
{
    int err = 0;
    *priority = 0x0U;
    if (domain >= NUM_POWER_DOMAINS)
    {
        return -EINVAL;
    }
    if (power_info.domains[domain].supportFlags & POWER_DOMAIN_SUPPORT_POLICY)
    {
        err = HPMread(cpuId, MSR_DEV, policy_regs[domain], (uint64_t*)priority);
        if (err)
        {
            ERROR_PRINT(Failed to get power policy for domain %s on CPU %d,power_names[domain], cpuId);
            return -EFAULT;
        }
    }
    return 0;
}


void power_finalize(void)
{
    if (power_info.turbo.steps)
    {
        free(power_info.turbo.steps);
    }
}

PowerInfo_t get_powerInfo(void)
{
    return &power_info;
}
