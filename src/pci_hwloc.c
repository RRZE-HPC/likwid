/*
 * =======================================================================================
 *
 *      Filename:  pci_hwloc.c
 *
 *      Description:  Interface to hwloc for PCI device lookup
 *
 *      Version:   4.1
 *      Released:  8.8.2016
 *
 *      Author:   Thomas Roehl (tr), thomas.roehl@googlemail.com
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

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>


#include <hwloc.h>
#include <types.h>
#include <bstrlib.h>
#include <affinity.h>
#include <topology.h>
#include <topology_hwloc.h>
#include <error.h>

int 
hwloc_pci_init(uint16_t testDevice, char** socket_bus, int* nrSockets)
{
    int cntr = 0;
    uint16_t testVendor = 0x8086;
    hwloc_obj_t obj;
    int flags;
    int i;

    if (!hwloc_topology)
    {
        likwid_hwloc_topology_init(&hwloc_topology);
        likwid_hwloc_topology_set_flags(hwloc_topology, HWLOC_TOPOLOGY_FLAG_WHOLE_IO );
        likwid_hwloc_topology_load(hwloc_topology);
    }

    for(i = 0; i < likwid_hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_PCI_DEVICE); i++)
    {
        obj = likwid_hwloc_get_obj_by_type(hwloc_topology, HWLOC_OBJ_PCI_DEVICE, i);
        if (obj->attr->pcidev.vendor_id != testVendor)
        {
            continue;
        }
        if ((obj->attr->pcidev.vendor_id == testVendor) && (obj->attr->pcidev.device_id == testDevice))
        {
            socket_bus[cntr] = (char*)malloc(4);
            sprintf(socket_bus[cntr++], "%02x/", obj->attr->pcidev.bus);
        }
    }
    *nrSockets = cntr;

    if (cntr == 0)
    {
        return -ENODEV;
    }

    return 0;
}
