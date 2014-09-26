/*
 * =======================================================================================
 *
 *      Filename:  power.h
 *
 *      Description:  Header File Power Module
 *                    Implements Intel RAPL Interface.
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

#ifndef POWER_H
#define POWER_H

#include <types.h>
#include <registers.h>
#include <bitUtil.h>
#include <msr.h>
#include <error.h>





double
power_printEnergy(PowerData* data)
{
    return  (double) ((data->after - data->before) * power_info.energyUnit);
}

int
power_start(PowerData* data, int cpuId, PowerType type)
{
    uint64_t result = 0;
    data->before = 0;
    CHECK_MSR_READ_ERROR(msr_read(cpuId, power_regs[type], &result))
    data->before = extractBitField(result,32,0);
}

int
power_stop(PowerData* data, int cpuId, PowerType type)
{
    uint64_t result = 0;
    data->after = 0;
    CHECK_MSR_READ_ERROR(msr_read(cpuId, power_regs[type], &result))
    data->after = extractBitField(result,32,0);
}

int
power_read(int cpuId, uint64_t reg, uint32_t *data)
{
    uint64_t result = 0;
    *data = 0;
    CHECK_MSR_READ_ERROR(msr_read(cpuId, reg, &result))
    *data = extractBitField(result,32,0);
    return 0;
}

int
power_tread(int socket_fd, int cpuId, uint64_t reg, uint32_t *data)
{
    uint64_t result = 0;
    *data = 0;
    CHECK_MSR_READ_ERROR(msr_tread(socket_fd, cpuId, reg, &result))
    *data = extractBitField(result,32,0);
    return 0;
}



#endif /*POWER_H*/
