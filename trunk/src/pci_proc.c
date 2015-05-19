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
#include <fcntl.h>


#include <types.h>
#include <accessClient.h>
#include <bstrlib.h>
#include <affinity.h>
#include <topology.h>
#include <error.h>

int getBusFromSocket(const uint32_t socket)
{
    int cur_bus = 0;
    uint32_t cur_socket = 0;
    char pci_filepath[1024];
    int fp;
    int ret = 0;
    while(cur_socket <= socket)
    {
        sprintf(pci_filepath, "/proc/bus/pci/%02x/05.0", cur_bus);
        fp = open(pci_filepath, O_RDONLY);
        if (fp < 0)
        {
            return -1;
        }
        uint32_t cpubusno = 0;
        ret = pread(fp, &cpubusno, sizeof(uint32_t), 0x108);
        if (ret != sizeof(uint32_t))
        {
            close(fp);
            return -1;
        }
        cur_bus = (cpubusno >> 8) & 0x0ff;
        close(fp);
        if(socket == cur_socket)
            return cur_bus;
        ++cur_socket;
        ++cur_bus;
        if(cur_bus > 0x0ff)
           return -1;
    }

    return -1;
}

int
proc_pci_init(uint16_t testDevice, char** socket_bus, int* nrSockets)
{
    FILE *fptr;
    char buf[1024];
    int cntr = 0;
    uint16_t testVendor = 0x8086;
    uint32_t sbus, sdevfn, svend, sdev;
    int busID;
    

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
        {
            socket_bus[cntr] = (char*)malloc(4);
            busID = getBusFromSocket(cntr);
            if (busID == sbus)
            {
                sprintf(socket_bus[cntr], "%02x/", sbus);
            }
            else
            {
                sprintf(socket_bus[cntr], "%02x/", busID);
            }
            cntr++;
        }
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
