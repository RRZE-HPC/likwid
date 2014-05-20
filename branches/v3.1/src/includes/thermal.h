/*
 * =======================================================================================
 *
 *      Filename:  thermal.h
 *
 *      Description:  Header File Thermal Module.
 *                    Implements Intel TM/TM2 Interface.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
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
#include <msr.h>

extern ThermalInfo thermal_info;

extern void thermal_init(int cpuId);
static inline uint32_t thermal_read(int cpuId);

static uint32_t
thermal_read(int cpuId)
{
    return (thermal_info.activationT - extractBitField(msr_read(cpuId, IA32_THERM_STATUS),7,16));
}

#endif /*THERMAL_H*/
