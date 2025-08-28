/*
 * =======================================================================================
 *
 *      Filename:  access_x86_clientmem.h
 *
 *      Description:  Header file for the interface to x86 client memory controllers for
 *                    the access module.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
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
#ifndef ACCESS_X86_CLIENTMEM_H
#define ACCESS_X86_CLIENTMEM_H

#include <types.h>

int access_x86_clientmem_init(uint32_t socket);
void access_x86_clientmem_finalize(uint32_t socket);
int access_x86_clientmem_read(PciDeviceIndex dev, uint32_t socket, uint32_t reg, uint64_t *data);
int access_x86_clientmem_write(PciDeviceIndex dev, uint32_t socket, uint32_t reg, uint64_t data);
int access_x86_clientmem_check(PciDeviceIndex dev, uint32_t socket);

#endif /* ACCESS_X86_CLIENTMEM_H */
