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
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <types.h>
#include <power.h>
#include <topology.h>
#include <lock.h>

/* #####   EXPORTED VARIABLES   ########################################### */

PowerInfo power_info;

/* #####   LOCAL VARIABLES   ############################################## */

static int power_initialized = 0;

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
power_init(int cpuId)
{
    uint64_t flags;
    int i;
    int err;
    int core_limits = 0;
    uint32_t unit_reg = MSR_RAPL_POWER_UNIT;
    int numDomains = NUM_POWER_DOMAINS;
    Configuration_t config;

    /* determine Turbo Mode features */
    double busSpeed;
    if (power_initialized)
    {
        return 0;
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Init power);
    if (!lock_check())
    {
        ERROR_PRINT(Access to performance monitoring registers locked);
        return -ENOLCK;
    }
    init_configuration();
    config = get_configuration();


    power_info.baseFrequency = 0;
    power_info.minFrequency = 0;
    power_info.turbo.numSteps = 0;
    power_info.turbo.steps = NULL;
    power_info.powerUnit = 0;
    power_info.timeUnit = 0;
    power_info.hasRAPL = 0;
    power_info.uncoreMinFreq = 0;
    power_info.uncoreMaxFreq = 0;
    power_info.perfBias = 0;
    power_info.statusRegWidth = 32;
    if (config->daemonMode == ACCESSMODE_PERF)
    {
        ERROR_PRINT(RAPL in access mode 'perf_event' only available with perfmon);
        return 0;
    }

    switch (cpuid_info.family)
    {
        case P6_FAMILY:
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
                case ATOM_SILVERMONT_AIR:
                case ATOM_SILVERMONT_GOLD:
                case BROADWELL:
                case BROADWELL_E:
                case BROADWELL_D:
                case BROADWELL_E3:
                case HASWELL_M1:
                case HASWELL_M2:
                case XEON_PHI_KNL:
                case XEON_PHI_KML:
                    power_info.hasRAPL = 1;
                    numDomains = NUM_POWER_DOMAINS - 1;
                    break;
                case SKYLAKE1:
                case SKYLAKE2:
                case KABYLAKE1:
                case KABYLAKE2:
                case CANNONLAKE:
                case COMETLAKE1:
                case COMETLAKE2:
                case TIGERLAKE1:
                case TIGERLAKE2:
                case ICELAKE1:
                case ICELAKE2:
                case ROCKETLAKE:
                    power_info.hasRAPL = 1;
                    numDomains = NUM_POWER_DOMAINS;
                    break;
                case SKYLAKEX:
                case ICELAKEX1:
                case ICELAKEX2:
                case SAPPHIRERAPIDS:
                case GRANITERAPIDS:
                case SIERRAFORREST:
                    core_limits = 1;
                    power_info.hasRAPL = 1;
                    numDomains = NUM_POWER_DOMAINS;
                    break;
                case ATOM_SILVERMONT_C:
                    power_info.hasRAPL = 1;
                    /* The info_regs list needs an update for Silvermont Type C
                       because it uses another info register */
                    info_regs[PKG] = MSR_PKG_POWER_INFO_SILVERMONT;
                    numDomains = NUM_POWER_DOMAINS - 1;
                    break;

                default:
                    DEBUG_PLAIN_PRINT(DEBUGLEV_INFO, NO RAPL SUPPORT);
                    return 0;
                    break;
            }
            break;
        case ZEN_FAMILY:
            if (cpuid_info.model == ZEN_RYZEN ||
                cpuid_info.model == ZENPLUS_RYZEN ||
                cpuid_info.model == ZENPLUS_RYZEN2 ||
                cpuid_info.model == ZEN2_RYZEN ||
                cpuid_info.model == ZEN2_RYZEN2 ||
                cpuid_info.model == ZEN2_RYZEN3 ||
                cpuid_info.model == ZEN3_RYZEN ||
                cpuid_info.model == ZEN3_RYZEN2 ||
                cpuid_info.model == ZEN3_RYZEN3 ||
                cpuid_info.model == ZEN3_EPYC_TRENTO)
            {
                cpuid_info.turbo = 0;
                power_info.hasRAPL = 1;
                numDomains = 2;
                unit_reg = MSR_AMD17_RAPL_POWER_UNIT;
                power_names[0] = "CORE";
                power_names[1] = "PKG";
                power_regs[0] = MSR_AMD17_RAPL_CORE_STATUS;
                power_regs[1] = MSR_AMD17_RAPL_PKG_STATUS;

                for (i = 0; i< NUM_POWER_DOMAINS; i++)
                {
                    limit_regs[i] = 0x0;
                    policy_regs[i] = 0x0;
                    perf_regs[i] = 0x0;
                    info_regs[i] = 0x0;
                }
            }
            break;
        case ZEN3_FAMILY:
            switch (cpuid_info.model)
            {
                case ZEN3_RYZEN:
                case ZEN3_RYZEN2:
                case ZEN3_RYZEN3:
                    cpuid_info.turbo = 0;
                    power_info.hasRAPL = 1;
                    numDomains = 2;
                    unit_reg = MSR_AMD17_RAPL_POWER_UNIT;
                    power_names[0] = "CORE";
                    power_names[1] = "PKG";
                    power_regs[0] = MSR_AMD17_RAPL_CORE_STATUS;
                    power_regs[1] = MSR_AMD17_RAPL_PKG_STATUS;

                    for (i = 0; i< NUM_POWER_DOMAINS; i++)
                    {
                        limit_regs[i] = 0x0;
                        policy_regs[i] = 0x0;
                        perf_regs[i] = 0x0;
                        info_regs[i] = 0x0;
                    }
                    break;
                case ZEN4_RYZEN:
                case ZEN4_RYZEN2:
                case ZEN4_EPYC:
                case ZEN4_RYZEN_PRO:
                    cpuid_info.turbo = 0;
                    power_info.hasRAPL = 1;
                    power_info.statusRegWidth = 64;
                    numDomains = 2;
                    unit_reg = MSR_AMD17_RAPL_POWER_UNIT;
                    power_names[0] = "CORE";
                    power_names[1] = "L3";
                    power_regs[0] = MSR_AMD17_RAPL_CORE_STATUS;
                    power_regs[1] = MSR_AMD19_RAPL_L3_STATUS;

                    for (i = 0; i< NUM_POWER_DOMAINS; i++)
                    {
                        limit_regs[i] = 0x0;
                        policy_regs[i] = 0x0;
                        perf_regs[i] = 0x0;
                        info_regs[i] = 0x0;
                    }
                    break;
            }
            break;
    }

    err = perfmon_init_maps();
    if (err != 0)
    {
        return err;
    }
    if (!HPMinitialized())
    {
        HPMinit();
        err = HPMaddThread(cpuId);
        if (err != 0)
        {
            ERROR_PRINT(Cannot get access to RAPL counters)
            return err;
        }
    }


    if ( power_info.hasRAPL )
    {
        busSpeed = 100.0;
    }
    else
    {
        busSpeed = 133.33;
    }

    if (cpuid_info.isIntel)
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
            memset(power_info.turbo.steps, 0, power_info.turbo.numSteps * sizeof(double));
            err = HPMread(cpuId, MSR_DEV, MSR_TURBO_RATIO_LIMIT, &flags);
            if (err)
            {
                ERROR_PRINT(Cannot gather values from %s, "MSR_TURBO_RATIO_LIMIT");
            }
            else
            {
                if (!core_limits)
                {
                    uint64_t flag_vals[4];
                    int flag_idx = 0;
                    int valid_idx = 0;
                    flag_vals[0] = flags;
                    if (power_info.turbo.numSteps > 8)
                    {
                        err = HPMread(cpuId, MSR_DEV, MSR_TURBO_RATIO_LIMIT1, &flag_vals[1]);
                        if (err)
                        {
                            ERROR_PRINT(Cannot read core limits from %s, "MSR_TURBO_RATIO_LIMIT1");
                            flag_vals[1] = 0;
                        }
                    }
                    if (power_info.turbo.numSteps > 16)
                    {
                        err = HPMread(cpuId, MSR_DEV, MSR_TURBO_RATIO_LIMIT2, &flag_vals[2]);
                        if (err)
                        {
                            ERROR_PRINT(Cannot read core limits from %s, "MSR_TURBO_RATIO_LIMIT2");
                            flag_vals[2] = 0;
                        }
                    }
                    if (power_info.turbo.numSteps > 24)
                    {
                        err = HPMread(cpuId, MSR_DEV, MSR_TURBO_RATIO_LIMIT3, &flag_vals[3]);
                        if (err)
                        {
                            ERROR_PRINT(Cannot read core limits from %s, "MSR_TURBO_RATIO_LIMIT3");
                            flag_vals[3] = 0;
                        }
                    }
                    power_info.turbo.steps[0] = busSpeed * (double) field64(flags,0, 8);
                    for (int i=1; i < power_info.turbo.numSteps; i++)
                    {
                        if (i % 8 == 0)
                        {
                            flag_idx++;
                        }
                        power_info.turbo.steps[i] = busSpeed * (double) field64(flag_vals[flag_idx],(i%8)*8, 8);
                        if (power_info.turbo.steps[i] > 0)
                            valid_idx = i;
                        else
                            power_info.turbo.steps[i] = power_info.turbo.steps[valid_idx];
                    }
                }
                else
                {
                    uint64_t flags_cores = 0;
                    int insert = 0;
                    err = HPMread(cpuId, MSR_DEV, MSR_TURBO_RATIO_LIMIT_CORES, &flags_cores);
                    if (err)
                    {
                        ERROR_PRINT(Cannot read core limits from %s, "MSR_TURBO_RATIO_LIMIT_CORES");
                        flags_cores = 0;
                    }
                    for (int i = 0; i < 8; i++)
                    {
                        int num_cores_befores = 0;
                        if (i > 0)
                            num_cores_befores = field64(flags_cores,(i-1)*8, 8);
                        int num_cores = field64(flags_cores,i*8, 8);
                        double freq = busSpeed * (double) field64(flags, i*8, 8);
                        for (int j = num_cores_befores; j < num_cores && insert < power_info.turbo.numSteps; j++)
                        {
                            power_info.turbo.steps[insert] = freq;
                            insert++;
                        }
                    }
                }
            }
        }
        else
        {
            ERROR_PRINT(Cannot gather values from %s, "MSR_PLATFORM_INFO");
        }
    }

    /* determine RAPL parameters */
    if ( power_info.hasRAPL )
    {
        err = HPMread(cpuId, MSR_DEV, unit_reg, &flags);
        if (err == 0)
        {
            double energyUnit;
            power_info.powerUnit = 1000000 / (1<<(flags & 0xF));
            power_info.timeUnit = 1000000 / (1 << ((flags>>16) & 0xF));
            if (cpuid_info.model != ATOM_SILVERMONT_E)
            {
                energyUnit = 1.0 / (1 << ((flags >> 8) & 0x1F));
            }
            else
            {
                energyUnit = 1.0 * (1 << ((flags >> 8) & 0x1F)) / 1000000;
            }
            for (i = 0; i < numDomains; i++)
            {
                power_info.domains[i].energyUnit = energyUnit;
                power_info.domains[i].type = i;
                power_info.domains[i].supportFlags = 0x0U;
                power_info.domains[i].tdp = 0.0;
                power_info.domains[i].minPower = 0.0;
                power_info.domains[i].maxPower = 0.0;
                power_info.domains[i].maxTimeWindow = 0.0;
            }
            if (cpuid_info.family == P6_FAMILY && ((cpuid_info.model == HASWELL_EP) ||
                (cpuid_info.model == HASWELL_M1) ||
                (cpuid_info.model == HASWELL_M2) ||
                (cpuid_info.model == BROADWELL_D) ||
                (cpuid_info.model == BROADWELL_E) ||
                (cpuid_info.model == SKYLAKEX) ||
                (cpuid_info.model == ICELAKEX1) ||
                (cpuid_info.model == ICELAKEX2) ||
                (cpuid_info.model == SAPPHIRERAPIDS) ||
                (cpuid_info.model == XEON_PHI_KNL) ||
                (cpuid_info.model == XEON_PHI_KML) ||
                (cpuid_info.model == GRANITERAPIDS)))
            {
                power_info.domains[DRAM].energyUnit = 15.3E-6;
            }

            for(i = 0; i < numDomains; i++)
            {
                err = HPMread(cpuId, MSR_DEV, power_regs[i], &flags);
                if (err == 0)
                {
                    power_info.domains[i].supportFlags |= POWER_DOMAIN_SUPPORT_STATUS;
                }
                else
                {
                    DEBUG_PRINT(DEBUGLEV_DETAIL, RAPL domain %s not supported, power_names[i]);
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
                        DEBUG_PRINT(DEBUGLEV_DETAIL, Deactivating limit register for RAPL domain %s, power_names[i]);
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
                            if (power_info.domains[i].minPower > power_info.domains[i].maxPower)
                                power_info.domains[i].minPower = 0;
                            power_info.domains[i].maxTimeWindow = (double) extractBitField(flags,7,48) * power_info.timeUnit;
                        }
                    }
                    else
                    {
                        DEBUG_PRINT(DEBUGLEV_DETAIL, Deactivating info register for RAPL domain %s, power_names[i]);
                        info_regs[i] = 0x0;
                    }
                }
                if (policy_regs[i] != 0x0)
                {
                    err = HPMread(cpuId, MSR_DEV, policy_regs[i], &flags);
                    if (err == 0)
                    {
                        power_info.domains[i].supportFlags |= POWER_DOMAIN_SUPPORT_POLICY;
                    }
                    else
                    {
                        DEBUG_PRINT(DEBUGLEV_DETAIL, Deactivating policy register for RAPL domain %s, power_names[i]);
                        policy_regs[i] = 0x0;
                    }
                }
                if (perf_regs[i] != 0x0)
                {
                    err = HPMread(cpuId, MSR_DEV, perf_regs[i], &flags);
                    if (err == 0)
                    {
                        power_info.domains[i].supportFlags |= POWER_DOMAIN_SUPPORT_PERF;
                    }
                    else
                    {
                        DEBUG_PRINT(DEBUGLEV_DETAIL, Deactivating perf register for RAPL domain %s, power_names[i]);
                        perf_regs[i] = 0x0;
                    }
                }
            }
        }
        else
        {
            DEBUG_PRINT(DEBUGLEV_INFO, Cannot gather values from unit register 0x%X. deactivating RAPL support, unit_reg);
            power_info.hasRAPL =  0;
        }

        if (cpuid_info.isIntel)
        {
            err = HPMread(cpuId, MSR_DEV, MSR_UNCORE_FREQ, &flags);
            if (err == 0)
            {
                power_info.uncoreMinFreq = ((double)((flags >> 8) & 0x3FULL)) * busSpeed;
                power_info.uncoreMaxFreq = ((double)(flags & 0x3F)) * busSpeed;
            }
            err = HPMread(cpuId, MSR_DEV, MSR_ENERGY_PERF_BIAS, &flags);
            if (err == 0)
            {
                power_info.perfBias = flags & 0xF;
            }
        }

        if (cpuid_info.family == ZEN3_FAMILY && (cpuid_info.model == ZEN4_RYZEN || cpuid_info.model == ZEN4_RYZEN2 || cpuid_info.model == ZEN4_EPYC))
        {
            err = HPMread(cpuId, MSR_DEV, MSR_AMD19_RAPL_L3_UNIT, &flags);
            if (err == 0)
            {
                DEBUG_PRINT(DEBUGLEV_DETAIL, Reading energy unit for Zen4 L3 RAPL domain);
                power_info.domains[1].energyUnit = 1.0 / (1 << ((flags >> 8) & 0x1F));
            }
        }
        power_info.numDomains = numDomains;
        power_initialized = 1;
        return power_info.hasRAPL;
    }
    else
    {
        return power_info.hasRAPL;
    }
    return 0;
}

/* All functions below are experimental and probably don't work */
int
power_perfGet(int cpuId, PowerType domain, uint32_t* status)
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

int
power_limitSet(int cpuId, PowerType domain, double power, double time, int doClamping)
{
    int err = 0;
    if (domain >= NUM_POWER_DOMAINS)
    {
        return -EINVAL;
    }


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
            ERROR_PRINT(Failed to set power limit for domain %s on CPU %d,power_names[domain], cpuId);
            return -EFAULT;
        }
    }
    return 0;
}

int
power_limitGet(int cpuId, PowerType domain, double* power, double* time)
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
            ERROR_PRINT(Failed to set power limit for domain %s on CPU %d,power_names[domain], cpuId);
            return -EFAULT;
        }
        *power = ((double)extractBitField(flags, 15, 0)) * power_info.domains[domain].energyUnit;
        Y = extractBitField(flags, 5, 17);
        Z = extractBitField(flags, 2, 22);
        *time = pow(2,((double)Y)) * (1.0 + (((double)Z)/4.0)) * power_info.timeUnit;
    }
    return 0;
}

int
power_limitState(int cpuId, PowerType domain)
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

int
power_limitActivate(int cpuId, PowerType domain)
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

int
power_limitDectivate(int cpuId, PowerType domain)
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

int
power_policySet(int cpuId, PowerType domain, uint32_t priority)
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

int
power_policyGet(int cpuId, PowerType domain, uint32_t* priority)
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

void
power_finalize(void)
{
    if (power_initialized == 0)
    {
        return;
    }
    if (power_info.turbo.steps != NULL)
    {
        free(power_info.turbo.steps);
    }
    power_info.turbo.steps = NULL;
    power_info.baseFrequency = 0;
    power_info.minFrequency = 0;
    power_info.turbo.numSteps = 0;
    power_info.powerUnit = 0;
    power_info.timeUnit = 0;
    power_info.hasRAPL = 0;
    power_info.uncoreMinFreq = 0;
    power_info.uncoreMaxFreq = 0;
    memset(power_info.domains, 0, NUM_POWER_DOMAINS*sizeof(PowerDomain));
    power_initialized = 0;
}

PowerInfo_t
get_powerInfo(void)
{
    return &power_info;
}
