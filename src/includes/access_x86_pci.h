/*
 * =======================================================================================
 *
 *      Filename:  access_x86_pci.h
 *
 *      Description:  Header file for the interface to x86 PCI functions for the access module.
 *
 *      Version:   5.0.2
 *      Released:  06.10.2020
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
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
#ifndef ACCESS_X86_PCI_H
#define ACCESS_X86_PCI_H

#include <types.h>

int access_x86_pci_init(const int socket);
void access_x86_pci_finalize(const int socket);
int access_x86_pci_read(PciDeviceIndex dev, const int socket, uint32_t reg, uint64_t *data);
int access_x86_pci_write(PciDeviceIndex dev, const int socket, uint32_t reg, uint64_t data);
int access_x86_pci_check(PciDeviceIndex dev, int socket);

#endif /* ACCESS_X86_PCI_H */
