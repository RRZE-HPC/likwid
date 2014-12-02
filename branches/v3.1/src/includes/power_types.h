/*
 * =======================================================================================
 *
 *      Filename:  power_types.h
 *
 *      Description:  Types file for power module.
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

#ifndef POWER_TYPES_H
#define POWER_TYPES_H

#include <stdint.h>

typedef enum {
    PKG = 0,
    PP0,
    PP1,
    DRAM
} PowerType;

typedef struct {
    int numSteps;
    double* steps;
} TurboBoost;

typedef struct {
    double baseFrequency;
    double minFrequency;
    TurboBoost turbo;
    double powerUnit;
    double energyUnit;
    double energyUnitDRAM;
    double timeUnit;
    double tdp;
    double minPower;
    double maxPower;
    double maxTimeWindow;
} PowerInfo;

typedef struct {
    uint32_t before;
    uint32_t after;
    PowerType type;
} PowerData;


#endif /*POWER_TYPES_H*/
