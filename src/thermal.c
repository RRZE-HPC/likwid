/*
 * =======================================================================================
 *
 *      Filename:  thermal.c
 *
 *      Description:  Module implementing Intel TM/TM2 interface
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <lock.h>
#include <thermal.h>
#include <topology.h>
#include <types.h>

/* #####   EXPORTED VARIABLES   ########################################### */

ThermalInfo thermal_info;

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void thermal_init(int cpuId)
{
    uint64_t flags = 0ULL;
    if (!lock_check()) {
        fprintf(stderr, "Access to thermal backend is locked.\n");
        return;
    }
    HPMinit();
    if (HPMaddThread(cpuId) < 0)
        fprintf(stderr, "Cannot initialize access to registers on CPU %d\n", cpuId);

    if (cpuid_hasFeature(TM2)) {
        if (HPMread(cpuId, MSR_DEV, IA32_THERM_STATUS, &flags)) {
            return;
        }

        if (flags & 0x1) {
            thermal_info.highT = 1;
        } else {
            thermal_info.highT = 0;
        }

        thermal_info.resolution = extractBitField(flags, 4, 27);

        flags                   = 0ULL;
        if (HPMread(cpuId, MSR_DEV, MSR_TEMPERATURE_TARGET, &flags)) {
            return;
        }
        thermal_info.activationT = extractBitField(flags, 8, 16);
        thermal_info.offset      = extractBitField(flags, 6, 24);
    }
}
