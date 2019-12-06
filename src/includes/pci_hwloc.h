/*
 * =======================================================================================
 *
 *      Filename:  pci_hwloc.h
 *
 *      Description:  Header File hwloc based PCI lookup backend
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
#ifndef PCI_HWLOC_H
#define PCI_HWLOC_H

extern int hwloc_pci_init(uint16_t testDevice, char** socket_bus, int* nrSockets);
extern int sysfs_pci_init(uint16_t testDevice, char** socket_bus, int* nrSockets);

#endif
