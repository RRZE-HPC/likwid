/*
 * =======================================================================================
 *
 *      Filename:  pci_types.h
 *
 *      Description:  Types file for pci module.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2013 Jan Treibig 
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


#ifndef PCI_TYPES_H
#define PCI_TYPES_H

#include <stdint.h>


typedef enum {
    PCI_R3QPI_DEVICE_LINK_0 = 0,
    PCI_R3QPI_DEVICE_LINK_1,
    PCI_R2PCIE_DEVICE,
    PCI_IMC_DEVICE_CH_0,
    PCI_IMC_DEVICE_CH_1,
    PCI_IMC_DEVICE_CH_2,
    PCI_IMC_DEVICE_CH_3,
    PCI_HA_DEVICE,
    PCI_QPI_DEVICE_PORT_0,
    PCI_QPI_DEVICE_PORT_1,
    PCI_QPI_MASK_DEVICE_PORT_0,
    PCI_QPI_MASK_DEVICE_PORT_1,
    PCI_QPI_MISC_DEVICE_PORT_0,
    PCI_QPI_MISC_DEVICE_PORT_1,
    MAX_NUM_DEVICES
} PciDeviceIndex;

static short pci_DevicePresent[MAX_NUM_DEVICES] = { 0 };

#endif /*PCI_TYPES_H*/
