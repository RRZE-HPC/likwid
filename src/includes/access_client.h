/*
 * =======================================================================================
 *
 *      Filename:  access_client.h
 *
 *      Description:  Header file for interface to the access daemon for the access module.
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
#ifndef ACCESS_CLIENT_H
#define ACCESS_CLIENT_H

#include <stdint.h>

int access_client_init(uint32_t cpu_id);
int access_client_read(PciDeviceIndex dev, uint32_t cpu_id, uint32_t reg, uint64_t *data);
int access_client_write(PciDeviceIndex dev, uint32_t cpu_id, uint32_t reg, uint64_t data);
void access_client_finalize(uint32_t cpu_id);
int access_client_check(PciDeviceIndex dev, uint32_t cpu_id);

#endif /* ACCESS_CLIENT_H */
