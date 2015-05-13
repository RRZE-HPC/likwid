/*
 * =======================================================================================
 *
 *      Filename:  pci_proc.c
 *
 *      Description:  Interface to procfs/sysfs for PCI device lookup
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 Thomas Roehl
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


#include <types.h>
#include <accessClient.h>
#include <bstrlib.h>
#include <affinity.h>
#include <topology.h>
#include <error.h>


int 
proc_pci_init(uint16_t testDevice, char** socket_bus, int* nrSockets)
{
    FILE *fptr;
    char buf[1024];
    int cntr = 0;
    uint16_t testVendor = 0x8086;
    uint32_t sbus, sdevfn, svend, sdev;
    
    

    if ( (fptr = fopen( "/proc/bus/pci/devices", "r")) == NULL )
    {
        fprintf(stderr, "Unable to open /proc/bus/pci/devices. \
                Thus, no support for PCI based Uncore counters.\n");
        return -ENODEV;
    }

    while( fgets(buf, sizeof(buf)-1, fptr) )
    {
        if ( sscanf(buf, "%2x%2x %4x%4x", &sbus, &sdevfn, &svend, &sdev) == 4 &&
             svend == testVendor && sdev == testDevice )
#ifndef REVERSE_HASWELL_PCI_SOCKETS
        {
            socket_bus[cntr] = (char*)malloc(4);
            sprintf(socket_bus[cntr++], "%02x/", sbus);
        }
#else
        {
            if (cpuid_info.model != HASWELL_EP)
            {
                socket_bus[cntr] = (char*)malloc(4);
                sprintf(socket_bus[cntr++], "%02x/", sbus);
            }
            else
            {
                socket_bus[cpuid_topology.numSockets-cntr-1] = (char*)malloc(4);
                sprintf(socket_bus[cpuid_topology.numSockets-cntr-1], "%02x/", sbus);
                cntr++
            }
        }
#endif
    }
    fclose(fptr);
    
    *nrSockets = cntr;
    
    if ( cntr == 0 )
    {
        //fprintf(stderr, "Uncore not supported on this system\n");
        return -ENODEV;
    }
    
    return 0;
}
