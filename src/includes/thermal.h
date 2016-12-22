/*
 * =======================================================================================
 *
 *      Filename:  thermal.h
 *
 *      Description:  Header File Thermal Module.
 *                    Implements Intel TM/TM2 Interface.
 *
 *      Version:   4.2
 *      Released:  22.12.2016
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
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
#ifndef THERMAL_H
#define THERMAL_H

#include <types.h>
#include <registers.h>
#include <bitUtil.h>
#include <error.h>
#include <access.h>

int
thermal_read(int cpuId, uint32_t *data)
{
    uint64_t result = 0;
    uint32_t readout = 0;
    if (HPMread(cpuId, MSR_DEV, IA32_THERM_STATUS, &result))
    {
        *data = 0;
        return -EIO;
    }
    readout = extractBitField(result,7,16);
    *data = (readout == 0 ?
                thermal_info.activationT - thermal_info.offset :
                (thermal_info.activationT - thermal_info.offset) - readout );
    return 0;
}

int
thermal_tread(int socket_fd, int cpuId, uint32_t *data)
{
    uint64_t result = 0;
    uint32_t readout = 0;
    if (HPMread(cpuId, MSR_DEV, IA32_THERM_STATUS, &result))
    {
        *data = 0;
        return -EIO;
    }
    readout = extractBitField(result,7,16);
    *data = (readout == 0 ?
                thermal_info.activationT - thermal_info.offset :
                (thermal_info.activationT - thermal_info.offset) - readout );
    return 0;
}

#endif /*THERMAL_H*/
