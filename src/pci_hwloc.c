#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>


#include <hwloc.h>
#include <types.h>
#include <accessClient.h>
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
    char nodeset[1024];
    int nodesetlen = 0;
    
    if (!hwloc_topology)
    {
        hwloc_topology_init(&hwloc_topology);
        hwloc_topology_set_flags(hwloc_topology, HWLOC_TOPOLOGY_FLAG_WHOLE_IO );
        hwloc_topology_load(hwloc_topology);
    }

    for(i=0;i<hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_PCI_DEVICE);i++)
    {
        obj = hwloc_get_obj_by_type(hwloc_topology, HWLOC_OBJ_PCI_DEVICE, i);
        if (obj->attr->pcidev.vendor_id != testVendor)
        {
            continue;
        }
        if (obj->attr->pcidev.vendor_id == testVendor && obj->attr->pcidev.device_id == testDevice)
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
