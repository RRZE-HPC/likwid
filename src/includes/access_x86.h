/*
 * =======================================================================================
 *
 *      Filename:  access_x86.h
 *
 *      Description:  Header file for the interface to x86 related functions for the access module.
 *
 *      Version:   5.2
 *      Released:  17.6.2021
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 NHR@FAU, University Erlangen-Nuremberg
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
#ifndef ACCESS_X86_H
#define ACCESS_X86_H

#include <types.h>

int access_x86_init(int cpu_id);
int access_x86_read(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t *data);
int access_x86_write(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t data);
void access_x86_finalize(int cpu_id);
int access_x86_check(PciDeviceIndex dev, int cpu_id);

#endif /* ACCESS_X86_H */
