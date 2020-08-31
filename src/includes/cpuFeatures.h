/*
 * =======================================================================================
 *
 *      Filename:  cpuFeatures.h
 *
 *      Description:  Header File of Module cpuFeatures.
 *
 *      Version:   5.0.2
 *      Released:  31.08.2020
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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

#ifndef CPUFEATURES_H
#define CPUFEATURES_H

#include <types.h>

extern CpuFeatureFlags cpuFeatureFlags;


extern void cpuFeatures_init(int cpu);
extern void cpuFeatures_print(int cpu);
extern void cpuFeatures_enable(int cpu, CpuFeature type);
extern void cpuFeatures_disable(int cpu, CpuFeature type);

#endif /*CPUFEATURES_H*/
