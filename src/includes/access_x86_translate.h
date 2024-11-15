/*
 * =======================================================================================
 *
 *      Filename:  access_x86_translate.h
 *
 *      Description:  Interface to x86 related functions for the access module.
 *
 *      Version:   5.4.0
 *      Released:  15.11.2024
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2024 RRZE, University Erlangen-Nuremberg
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

#ifndef ACCESS_X86_TRANSLATE_H
#define ACCESS_X86_TRANSLATE_H


int access_x86_translate_init(const int cpu_id);
int access_x86_translate_check(PciDeviceIndex dev, int cpu_id);
int access_x86_translate_read(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t *data);
int access_x86_translate_write(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t data);
int access_x86_translate_finalize(int cpu_id);


#endif /* ACCESS_X86_TRANSLATE_H */
