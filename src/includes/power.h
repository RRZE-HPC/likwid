/*
 * =======================================================================================
 *
 *      Filename:  power.h
 *
 *      Description:  Header File Power Module
 *                    Implements Intel RAPL Interface.
 *
 *      Version:   4.1
 *      Released:  13.6.2016
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

#ifndef POWER_H
#define POWER_H

#include <types.h>
#include <registers.h>
#include <bitUtil.h>
#include <error.h>
#include <access.h>

const char* power_names[NUM_POWER_DOMAINS] = {"PKG", "PP0", "PP1", "DRAM"};

uint32_t power_regs[NUM_POWER_DOMAINS] = {MSR_PKG_ENERGY_STATUS,
                                MSR_PP0_ENERGY_STATUS,
                                MSR_PP1_ENERGY_STATUS,
                                MSR_DRAM_ENERGY_STATUS};

uint32_t limit_regs[NUM_POWER_DOMAINS] = {MSR_PKG_RAPL_POWER_LIMIT,
                                MSR_PP0_RAPL_POWER_LIMIT,
                                MSR_PP1_RAPL_POWER_LIMIT,
                                MSR_DRAM_RAPL_POWER_LIMIT};

uint32_t policy_regs[NUM_POWER_DOMAINS] = {0,
                                MSR_PP0_ENERGY_POLICY,
                                MSR_PP1_ENERGY_POLICY,
                                0};

uint32_t perf_regs[NUM_POWER_DOMAINS] = {MSR_PKG_PERF_STATUS,
                                MSR_PP0_PERF_STATUS,
                                0,
                                MSR_DRAM_PERF_STATUS};

uint32_t info_regs[NUM_POWER_DOMAINS] = {MSR_PKG_POWER_INFO,
                                0,
                                0,
                                MSR_DRAM_POWER_INFO};


double
power_printEnergy(PowerData* data)
{
    return  (double) ((data->after - data->before) * power_info.domains[data->domain].energyUnit);
}

int
power_start(PowerData* data, int cpuId, PowerType type)
{
    if (power_info.hasRAPL)
    {
        if (power_info.domains[type].supportFlags & POWER_DOMAIN_SUPPORT_STATUS)
        {
            uint64_t result = 0;
            data->before = 0;
            CHECK_MSR_READ_ERROR(HPMread(cpuId, MSR_DEV, power_regs[type], &result))
            data->before = field64(result, 0, 32);
            data->domain = type;
            return 0;
        }
        else
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, RAPL domain %s not supported, power_names[type]);
            return -EFAULT;
        }
    }
    else
    {
        DEBUG_PLAIN_PRINT(DEBUGLEV_DEVELOP, No RAPL support);
        return -EIO;
    }
}

int
power_stop(PowerData* data, int cpuId, PowerType type)
{
    if (power_info.hasRAPL)
    {
        if (power_info.domains[type].supportFlags & POWER_DOMAIN_SUPPORT_STATUS)
        {
            uint64_t result = 0;
            data->after = 0;
            CHECK_MSR_READ_ERROR(HPMread(cpuId, MSR_DEV, power_regs[type], &result))
            data->after = field64(result, 0, 32);
            data->domain = type;
            return 0;
        }
        else
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, RAPL domain %s not supported, power_names[type]);
            return -EFAULT;
        }
    }
    else
    {
        DEBUG_PLAIN_PRINT(DEBUGLEV_DEVELOP, No RAPL support);
        return -EIO;
    }
}

int
power_read(int cpuId, uint64_t reg, uint32_t *data)
{
    int i;
    PowerType type = -1;

    if (power_info.hasRAPL)
    {
        for (i = 0; i < NUM_POWER_DOMAINS; i++)
        {
            if (reg == power_regs[i])
            {
                type = i;
                break;
            }
        }
        if (power_info.domains[type].supportFlags & POWER_DOMAIN_SUPPORT_STATUS)
        {
            uint64_t result = 0;
            *data = 0;
            CHECK_MSR_READ_ERROR(HPMread(cpuId, MSR_DEV, reg, &result))
            *data = field64(result, 0, 32);
            return 0;
        }
        else
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, RAPL domain %s not supported, power_names[type]);
            return -EFAULT;
        }
    }
    else
    {
        DEBUG_PLAIN_PRINT(DEBUGLEV_DEVELOP, No RAPL support);
        return -EIO;
    }
}

int
power_tread(int socket_fd, int cpuId, uint64_t reg, uint32_t *data)
{
    int i;
    PowerType type;
    if (power_info.hasRAPL)
    {
        for (i = 0; i < NUM_POWER_DOMAINS; i++)
        {
            if (reg == power_regs[i])
            {
                type = i;
                break;
            }
        }
        if (power_info.domains[type].supportFlags & POWER_DOMAIN_SUPPORT_STATUS)
        {
            uint64_t result = 0;
            *data = 0;
            CHECK_MSR_READ_ERROR(HPMread(cpuId, MSR_DEV, reg, &result))
            *data = field64(result, 0, 32);
            return 0;
        }
        else
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, RAPL domain %s not supported, power_names[type]);
            return -EFAULT;
        }
    }
    else
    {
        DEBUG_PLAIN_PRINT(DEBUGLEV_DEVELOP, No RAPL support);
        return -EIO;
    }
}

double
power_getEnergyUnit(int domain)
{
    return power_info.domains[domain].energyUnit;
}

#endif /*POWER_H*/
